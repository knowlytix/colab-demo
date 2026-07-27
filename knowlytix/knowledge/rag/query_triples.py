# SPDX-License-Identifier: Apache-2.0
"""Query-triple extraction: natural-language question -> GMS query triples.

The heart of triple-mediated retrieval (``GEODE_RAG_DESIGN.md`` §3): a question
is turned into one or more ``(head, relation, tail)`` query triples where exactly
the asked value is the bare variable ``?`` and any intermediate unknowns are
named variables (``?x``, ``?y``). The triples are then bound to the graph's
vocabulary and answered through the GMS — never via an opaque vector index.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from knowlytix.knowledge.llm_backend import LLMBackend

__all__ = ["QueryTriple", "QueryTripleExtractor", "GeometricQueryParser",
           "is_var", "ASKED", "schema_from_store"]

ASKED = "?"  # the slot whose value answers the question

_NUMERIC_RE = re.compile(r"^-?[\d,]*\.?\d+%?$")


def schema_from_store(store, *, max_entities: int = 80,
                      exclude_relations=("in_section",)) -> dict:
    """Extract the graph's vocabulary to ground query-triple extraction.

    Showing the LLM the *actual* relation names (and a sample of entity names)
    is the biggest reliability lever: a small model otherwise guesses relation
    names that never bind. Numeric value-entities are filtered out — they are
    answers, not query terms.
    """
    if store.adapter is None:
        return {"relations": [], "entities": []}
    relations = [r for r in store.adapter.relation_to_idx
                 if r not in exclude_relations]
    entities = [e for e in store.adapter.entity_to_idx
                if not _NUMERIC_RE.match(e.strip())]
    return {"relations": sorted(relations),
            "entities": sorted(entities)[:max_entities]}


def is_var(slot: str) -> bool:
    """A slot is a variable (unknown) iff it starts with ``?``."""
    return isinstance(slot, str) and slot.startswith("?")


@dataclass
class QueryTriple:
    """A pattern ``(head, relation, tail)`` with ``?``-prefixed unknown slots."""

    head: str
    relation: str
    tail: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.head, self.relation, self.tail)


_SYSTEM = (
    "You translate a question into knowledge-graph query triples. Output ONLY a "
    "JSON array of objects with keys head, relation, tail.\n"
    'Mark the single value the question asks for as the bare string "?". NEVER '
    "put a guessed or placeholder value (like a number, or words such as "
    '"value"/"amount"/"unknown") in that slot — it must be exactly "?".\n'
    'Use "?x", "?y" for intermediate unknowns that link triples in a multi-hop '
    "question. Relations are lowercase snake_case (e.g. has_revenue, headcount, "
    "led_by, managed_by).\n"
    "Examples:\n"
    'Q: What is Consumer revenue? -> [{"head":"consumer","relation":"revenue","tail":"?"}]\n'
    'Q: How many staff in Research? -> [{"head":"research","relation":"headcount","tail":"?"}]\n'
    'Q: Who manages the lead of Sales? -> '
    '[{"head":"sales","relation":"led_by","tail":"?x"},'
    '{"head":"?x","relation":"managed_by","tail":"?"}]'
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class QueryTripleExtractor:
    """LLM-backed NL→query-triple extractor (LLM is injectable for tests).

    When ``vocab`` (from :func:`schema_from_store`) is supplied, the graph's
    actual relation/entity names are injected into the prompt so the model emits
    binding-compatible triples — the primary extraction-reliability lever.
    """

    def __init__(self, llm: LLMBackend, *, vocab: dict | None = None,
                 max_tokens: int = 512):
        self.llm = llm
        self.vocab = vocab
        self.max_tokens = max_tokens

    def extract(self, question: str, *, hint: str | None = None
                ) -> list[QueryTriple]:
        system = _SYSTEM + self._vocab_block()
        user = question if hint is None else f"{question}\n\nNote: {hint}"
        raw = self.llm.call(system=system, user=user, max_tokens=self.max_tokens)
        return self._ensure_asked(self.parse(raw))

    def _vocab_block(self) -> str:
        if not self.vocab:
            return ""
        rels = ", ".join(self.vocab.get("relations", [])[:60])
        ents = ", ".join(self.vocab.get("entities", [])[:80])
        block = ""
        if rels:
            block += ("\n\nPrefer these exact relation names WHEN ONE MATCHES the "
                      f"attribute the question asks about: {rels}.\n"
                      "If the question asks about an attribute that is NOT in this "
                      "list, use the question's own words (snake_case) as the "
                      "relation -- do NOT substitute a different listed attribute "
                      "(an interest rate is not a fee). A binding/relevance check "
                      "downstream will reject an attribute the entity does not have.")
        if ents:
            block += f"\nKnown entities include: {ents}."
        return block

    @staticmethod
    def _ensure_asked(triples: list[QueryTriple]) -> list[QueryTriple]:
        """Guarantee an asked slot. Small models sometimes omit the bare ``?``
        and fill the queried slot with a placeholder; if no slot in the whole
        response is a variable, mark the last triple's tail as ``?`` (wh-questions
        ask for the object). Query-only — not applied to claim extraction.
        """
        if triples and not any(is_var(s) for t in triples
                               for s in t.as_tuple()):
            last = triples[-1]
            triples[-1] = QueryTriple(last.head, last.relation, ASKED)
        return triples

    @staticmethod
    def parse(raw: str) -> list[QueryTriple]:
        """Parse the LLM response into query triples.

        Tolerant of fenced code blocks and of a ``head | relation | tail``
        line format as a fallback when the model ignores the JSON instruction.
        """
        text = raw.strip()
        m = _FENCE_RE.search(text)
        if m:
            text = m.group(1).strip()
        triples: list[QueryTriple] = []
        try:
            data = json.loads(text)
            for obj in data:
                h, r, t = obj.get("head"), obj.get("relation"), obj.get("tail")
                if h and r and t:
                    triples.append(QueryTriple(str(h), str(r), str(t)))
        except (json.JSONDecodeError, AttributeError, TypeError):
            for line in text.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3 and all(parts):
                    triples.append(QueryTriple(*parts))
        return triples


class GeometricQueryParser:
    """Parse a question into graph-vocab query triples with the tuned encoder +
    graph structure -- no LLM at parse time.

    A small instruct model is unreliable at decomposition: it emits the *attribute*
    as the head (``harm_threshold_usd``) or a *generic* relation (``has_value``),
    which then fail to bind. This parser instead (1) matches the question to the
    relation it asks about, by cosine over the graph's real relation phrases in the
    document-tuned space, and (2) picks the head among the entities that actually
    *assert* that relation, scored by the question's similarity to each entity's
    name and aliases. The result is already graph vocabulary, so the binder passes
    it through. ``extract`` mirrors :class:`QueryTripleExtractor` so it is a drop-in.

    Multi-hop questions are handled by the same machinery: ``extract`` searches
    asserted graph paths up to ``max_hops`` from each head entity and returns the
    highest-scoring one as a ``?x``-chained list of triples (a single-hop path is
    the previous behaviour exactly). See ``extract`` for the scoring rationale.
    """

    def __init__(self, store, encoder=None, *,
                 exclude_relations: tuple[str, ...] = ("in_section", "has_alias",
                                                       "has_product"),
                 top_rel: int = 3, alias_relation: str = "has_alias",
                 max_hops: int = 2):
        self.store = store
        self._encoder = encoder
        self.top_rel = top_rel
        self.max_hops = max(1, max_hops)
        self._exclude = set(exclude_relations)
        # Graph structure: which entities assert each relation, the adjacency
        # (head -> [(relation, tail)]) used for multi-hop path search, and aliases.
        self._heads_with: dict[str, list[str]] = {}
        self._aliases: dict[str, list[str]] = {}
        self._adj: dict[str, list[tuple[str, str]]] = {}
        triples = list(store.doc_graph.triples) if store.doc_graph else []
        for h, r, t in triples:
            if r == alias_relation:
                self._aliases.setdefault(h, []).append(str(t))
            elif r not in self._exclude:
                self._heads_with.setdefault(r, [])
                if h not in self._heads_with[r]:
                    self._heads_with[r].append(h)
                self._adj.setdefault(h, []).append((r, str(t)))
        self._relations = sorted(self._heads_with)
        self._rel_idx = {r: i for i, r in enumerate(self._relations)}
        self._heads = sorted(self._adj)
        self._rel_emb = None   # lazy
        self._name_emb = None  # lazy: (owners, embeddings) for head-name scoring

    @staticmethod
    def _phrase(rel: str) -> str:
        return rel[4:].replace("_", " ") if rel.startswith("has_") else rel.replace("_", " ")

    def _encode(self, texts: list[str]):
        import torch
        if self._encoder is None:
            from knowlytix.core.graph.encoders import encode_texts
            self._encoder = encode_texts
        emb = torch.as_tensor(self._encoder(texts), dtype=torch.float32)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def _relation_embeddings(self):
        if self._rel_emb is None:
            self._rel_emb = self._encode([self._phrase(r) for r in self._relations])
        return self._rel_emb

    def _head_name_embeddings(self):
        """Embeddings for every head entity's name + aliases (one row each),
        cached so a question is scored against all anchor names in one matmul."""
        if self._name_emb is None:
            owners, names = [], []
            for e in self._heads:
                for nm in [e.replace("_", " ")] + self._aliases.get(e, []):
                    owners.append(e)
                    names.append(nm)
            self._name_emb = (owners, self._encode(names) if names else None)
        return self._name_emb

    def extract(self, question: str, *, hint: str | None = None) -> list[QueryTriple]:
        if not self._relations:
            return []
        qv = self._encode([question])[0]
        rel_sims = self._relation_embeddings() @ qv

        # Best similarity of the question to each head entity's name/aliases --
        # the anchor signal that lets a clearly named entity pull the choice
        # toward a chain that starts there.
        owners, name_emb = self._head_name_embeddings()
        head_s: dict[str, float] = {}
        if name_emb is not None:
            for o, s in zip(owners, (name_emb @ qv).tolist()):
                if s > head_s.get(o, float("-inf")):
                    head_s[o] = s

        # Search asserted graph paths (1 .. max_hops) from every head entity and
        # keep the highest-scoring one. A path scores as the anchor's name match
        # plus the sum of its relations' similarities to the question. For a
        # multi-hop question ("which region runs the division that contains X?")
        # the shared bridge relation cancels between competing extensions, so the
        # *terminal* asked relation decides -- and a real two-hop chain outscores
        # stopping at the intermediate. A single-hop path reduces to the previous
        # (head, relation) pick exactly.
        #
        # A path may only be *routed through* (extended past) a relation the
        # question actually mentions -- one of its top-``top_rel`` relations, like
        # the "division" in the example. The terminal asked relation is
        # unrestricted, because the encoder may rank it low (a weak "who heads" ->
        # ``has_head`` match still anchors the answer slot). This keeps a large
        # store from bridging through arbitrary low-ranked relations.
        k = min(self.top_rel, len(self._relations))
        bridge_ok = {self._relations[i]
                     for i in rel_sims.argsort(descending=True)[:k].tolist()}
        best = None  # (score, anchor, [(relation, tail), ...])

        def walk(anchor: str, node: str, path: list,
                 visited: set, rel_score: float) -> None:
            nonlocal best
            for rel, tail in self._adj.get(node, []):
                if tail in visited:  # no cycles
                    continue
                score = rel_score + float(rel_sims[self._rel_idx[rel]])
                new_path = path + [(rel, tail)]
                # ``rel`` is the terminal of ``new_path`` here, so it is recorded
                # unrestricted (the answer slot).
                total = head_s.get(anchor, 0.0) + score
                if best is None or total > best[0]:
                    best = (total, anchor, new_path)
                # Continuing past ``rel`` turns it into a bridge -- only allowed
                # for a mentioned (top-k) relation, within the hop budget.
                if len(new_path) < self.max_hops and rel in bridge_ok:
                    walk(anchor, tail, new_path, visited | {tail}, score)

        for start in self._heads:
            walk(start, start, [], {start}, 0.0)

        if best is None:
            return []
        _, anchor, path = best
        triples, prev = [], anchor
        for i, (rel, _tail) in enumerate(path):
            tail_slot = ASKED if i == len(path) - 1 else f"?x{i}"
            triples.append(QueryTriple(prev, rel, tail_slot))
            prev = tail_slot
        return triples
