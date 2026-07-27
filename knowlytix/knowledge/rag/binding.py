# SPDX-License-Identifier: Apache-2.0
"""Bind query triples to the GMS graph's actual vocabulary.

A query triple is useless unless its entities/relations resolve to the names the
store actually holds ("staff count" -> ``has_headcount``). Binding quality bounds
prose-question recall (``GEODE_RAG_DESIGN.md`` §6).

P1: string/fuzzy binding (reuses ``store.fuzzy_match_entity`` for entities and a
fuzzy relation matcher here). P2 will add embedding-based binding. Variable slots
(``?``, ``?x``) pass through unbound by design.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from knowlytix.knowledge.rag.query_triples import QueryTriple, is_var

__all__ = ["BoundTriple", "TripleBinder"]

# encoder: list[str] -> array-like (N, d). Default wraps the repo's MiniLM
# encoder (downloads on first real use); inject a deterministic encoder in tests.
Encoder = Callable[[list[str]], "object"]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


@dataclass
class BoundTriple:
    """A query triple with concrete slots resolved to graph vocabulary.

    A ``None`` slot is one that was a known (non-variable) term we *failed* to
    bind — that makes :attr:`bound` ``False`` and feeds bind-check abstention.
    Variable slots keep their ``?``-name and do not count against binding.
    """

    original: QueryTriple
    head: str | None
    relation: str | None
    tail: str | None

    @property
    def bound(self) -> bool:
        """True iff every non-variable slot resolved to a real graph term."""
        slots = ((self.original.head, self.head),
                 (self.original.relation, self.relation),
                 (self.original.tail, self.tail))
        return all(is_var(orig) or resolved is not None
                   for orig, resolved in slots)


class TripleBinder:
    """Resolve query-triple slots against a :class:`GMSExpertStore`."""

    def __init__(self, store, *, mode: str = "fuzzy",
                 encoder: Encoder | None = None,
                 bind_threshold: float = 0.5, bind_margin: float = 0.05):
        self.store = store
        if mode not in ("fuzzy", "embedding"):
            raise NotImplementedError(f"binding mode {mode!r} not supported.")
        self.mode = mode
        self._encoder = encoder
        self.bind_threshold = bind_threshold
        self.bind_margin = bind_margin
        self._ent_space: tuple[list[str], object] | None = None
        self._rel_space: tuple[list[str], object] | None = None

    def bind(self, qt: QueryTriple) -> BoundTriple:
        return BoundTriple(
            original=qt,
            head=qt.head if is_var(qt.head) else self._bind_entity(qt.head),
            relation=(qt.relation if is_var(qt.relation)
                      else self._bind_relation(qt.relation)),
            tail=qt.tail if is_var(qt.tail) else self._bind_entity(qt.tail),
        )

    # -- entity / relation slot binding -------------------------------------

    def _bind_entity(self, name: str) -> str | None:
        # Embedding mode: the (document-tuned) encoder is PRIMARY -- a high-
        # confidence semantic match maps a surface head ("overdraft fee policy")
        # onto its canonical graph entity ("overdraft") even when the strings do
        # not overlap. Fuzzy is the fallback when the encoder abstains (low score
        # / ambiguous tie). In fuzzy mode the string matcher is the only path.
        if self.mode == "embedding" and self._encoder is not None:
            names, emb = self._entity_space()
            hit = self._embed_match(name, names, emb)
            return hit if hit is not None else self.store.fuzzy_match_entity(name)
        return self.store.fuzzy_match_entity(name)

    def _bind_relation(self, name: str) -> str | None:
        # A literal relation name (exact / has_<slug>) is unambiguous -- take it.
        # Otherwise the tuned encoder is primary, with fuzzy as the fallback.
        rels = getattr(self.store.adapter, "relation_to_idx", {}) or {}
        if name in rels or _slug(name) in rels or f"has_{_slug(name)}" in rels:
            return self._fuzzy_relation(name)
        if self.mode == "embedding" and self._encoder is not None:
            names, emb = self._relation_space()
            hit = self._embed_match(name, names, emb)
            return hit if hit is not None else self._fuzzy_relation(name)
        return self._fuzzy_relation(name)

    def _fuzzy_relation(self, name: str) -> str | None:
        """Fuzzy-match a relation name against the adapter's relation vocab.

        Precedence: exact, ``has_<slug>`` (the table-column convention),
        case-insensitive-unique, substring-unique. Ambiguous -> ``None``
        (bank-grade: refuse rather than mis-resolve).
        """
        if self.store.adapter is None:
            return None
        rels = self.store.adapter.relation_to_idx
        if name in rels:
            return name
        slug = _slug(name)
        if slug in rels:
            return slug
        prefixed = f"has_{slug}"
        if prefixed in rels:
            return prefixed
        ci = [r for r in rels if r.lower() == name.lower()]
        if len(ci) == 1:
            return ci[0]
        sub = [r for r in rels if slug and (slug in r or r in slug)]
        if len(sub) == 1:
            return sub[0]
        return None

    # -- embedding fallback -------------------------------------------------

    def _encode(self, texts: list[str]):
        import torch

        if self._encoder is None:
            from knowlytix.core.graph.encoders import encode_texts
            self._encoder = encode_texts
        emb = torch.as_tensor(self._encoder(texts), dtype=torch.float32)
        # Defensive L2-normalize so cosine == dot even if the encoder did not.
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def _entity_space(self):
        if self._ent_space is None:
            names = list(self.store.adapter.entity_to_idx.keys())
            self._ent_space = (names, self._encode(names))
        return self._ent_space

    def _relation_space(self):
        if self._rel_space is None:
            names = list(self.store.adapter.relation_to_idx.keys())
            self._rel_space = (names, self._encode(names))
        return self._rel_space

    def _embed_match(self, query: str, names: list[str], emb) -> str | None:
        """Nearest candidate by cosine; refuse on low score or a near tie."""
        if not names:
            return None
        q = self._encode([query])[0]
        sims = (emb @ q)
        order = sims.argsort(descending=True)
        best = int(order[0])
        if float(sims[best]) < self.bind_threshold:
            return None
        if len(names) > 1:
            second = float(sims[int(order[1])])
            if float(sims[best]) - second < self.bind_margin:
                return None  # ambiguous — refuse rather than mis-resolve
        return names[best]
