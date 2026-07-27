# SPDX-License-Identifier: Apache-2.0
"""GMS self-verification of LLM output — a hallucination detector.

Decompose the synthesized answer into claim triples and check each against the
GMS (``GEODE_RAG_DESIGN.md`` §7). Two calibration-free signals:

* **Contradiction** — the answer asserts ``(h, r, t)`` but the graph asserts
  ``(h, r, t')`` with ``t' != t``. Exact and threshold-free; catches an LLM that
  changed a value.
* **Admissibility** — for cap-trained stores, an unasserted claim is implausible
  when its geodesic distance exceeds the learned per-relation cap radius
  ``rho_r`` (a threshold the model learned, not a hand-set constant).

Claims that touch nothing in the graph are ``unverifiable`` (advisory, not a
failure). This turns the GMS into a faithfulness check on generated text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from knowlytix.knowledge.geode.provenance import value_canon
from knowlytix.knowledge.rag.binding import TripleBinder
from knowlytix.knowledge.rag.query_triples import QueryTriple, QueryTripleExtractor

__all__ = ["ClaimVerdict", "VerifyReport", "AnswerVerifier"]

_NUM_RE = re.compile(r"-?[\d,]*\.?\d+")

_SYSTEM = (
    "Extract the factual claims in the text as knowledge-graph triples. Output "
    "ONLY a JSON array of objects with keys head, relation, tail, using concrete "
    "values (no variables). Relations are lowercase snake_case. Include only "
    "claims the text actually asserts."
)


def _values_match(a: str, b: str) -> bool:
    """Numeric-aware equality: compare as floats when both parse, else compare
    with :func:`value_canon` (separators unified, so a prose answer value matches
    the graph's stored slug)."""
    ma, mb = _NUM_RE.search(a.replace(",", "")), _NUM_RE.search(b.replace(",", ""))
    if ma and mb:
        try:
            return abs(float(ma.group()) - float(mb.group())) < 1e-6
        except ValueError:
            pass
    return value_canon(a) == value_canon(b)


def _surface_match(term: str, text_lower: str) -> bool:
    """Word-boundary match of an entity name/alias, snake/spaced tolerant."""
    t = str(term).strip().lower()
    if len(t) <= 1:
        return False
    forms = {t, t.replace("_", " ")}
    return any(re.search(r"\b" + re.escape(f) + r"\b", text_lower) for f in forms)


def _tail_in_text(tail: str, text: str, text_lower: str) -> bool:
    """A numeric tail matches on value (35.0 == '$35'); else word-boundary."""
    s = str(tail).strip()
    try:
        val = float(s)
    except ValueError:
        return _surface_match(s, text_lower)
    for m in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            if abs(float(m.replace(",", "")) - val) < 1e-9:
                return True
        except ValueError:
            continue
    return False


@dataclass
class ClaimVerdict:
    triple: QueryTriple
    status: str          # supported | contradicted | implausible | unverifiable
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status in ("contradicted", "implausible")


@dataclass
class VerifyReport:
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(v.failed for v in self.verdicts)

    @property
    def failures(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.failed]

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "claims": [{"triple": v.triple.as_tuple(), "status": v.status,
                        "detail": v.detail} for v in self.verdicts],
            "failed": [v.triple.as_tuple() for v in self.failures],
        }


class AnswerVerifier:
    """Check an answer's claim triples against the GMS."""

    def __init__(self, store, llm, *, binder: TripleBinder | None = None,
                 max_tokens: int = 512, mode: str = "llm"):
        self.store = store
        self.llm = llm
        self.binder = binder or TripleBinder(store)
        self.max_tokens = max_tokens
        # "llm": parse the answer into claim triples with an LLM (legacy default).
        # "geometric": extract them by walking the store graph (no LLM) so claims
        # bind to canonical vocabulary instead of LLM-invented relations -- the
        # same parse-and-bind idea the query side uses. _check is unchanged.
        self.mode = mode
        self._adj = None  # lazy adjacency for geometric mode

    def verify(self, answer: str) -> VerifyReport:
        if self.mode == "geometric":
            claims = self._geometric_claims(answer)
        else:
            raw = self.llm.call(system=_SYSTEM, user=answer,
                                max_tokens=self.max_tokens)
            claims = QueryTripleExtractor.parse(raw)
        return VerifyReport([self._check(c) for c in claims])

    def _check(self, claim: QueryTriple) -> ClaimVerdict:
        bt = self.binder.bind(claim)
        if bt.head is None or bt.relation is None:
            return ClaimVerdict(claim, "unverifiable",
                                "head/relation not in graph")
        asserted = self.store.query_triples(head=bt.head, relation=bt.relation)
        if asserted:
            if any(_values_match(claim.tail, t) for _, _, t in asserted):
                return ClaimVerdict(claim, "supported")
            return ClaimVerdict(
                claim, "contradicted",
                f"graph asserts {sorted({t for _, _, t in asserted})}")
        return self._check_admissibility(claim, bt)

    def _check_admissibility(self, claim: QueryTriple, bt) -> ClaimVerdict:
        radius = self.store.cap_radius(bt.relation)
        tail = bt.tail if bt.tail is not None else self.store.fuzzy_match_entity(
            claim.tail)
        if radius is None or tail is None:
            return ClaimVerdict(claim, "unverifiable",
                                "no asserted fact and not cap-verifiable")
        score = self.store.score_triple(bt.head, bt.relation, tail)
        if score is None:
            return ClaimVerdict(claim, "unverifiable", "not scorable")
        if score > radius:
            return ClaimVerdict(
                claim, "implausible",
                f"geodesic {score:.3f} > cap radius {radius:.3f}")
        return ClaimVerdict(claim, "supported", "within cap radius")

    # -- geometric (LLM-free) claim extraction --------------------------------
    def _adjacency(self):
        if self._adj is None:
            adj: dict = {}
            aliases: dict = {}
            for h, r, t in self.store.triples:
                if r == "has_alias":
                    aliases.setdefault(h, []).append(str(t))
                elif r != "in_section":
                    adj.setdefault(h, []).append((r, str(t)))
            self._adj = (adj, aliases)
        return self._adj

    def _geometric_claims(self, answer: str) -> "list[QueryTriple]":
        """Extract the answer's claims by walking the store graph -- no LLM. Emit a
        canonical ``(head, relation, tail)`` for every real edge whose head (or an
        alias) and tail the answer states, so a correct answer yields store-bound
        triples that verify, instead of the LLM-invented relations that false-fail.
        Multi-hop chains surface as their constituent edges (each a real triple)."""
        low = answer.lower()
        adj, aliases = self._adjacency()
        claims: list = []
        seen: set = set()
        for head, edges in adj.items():
            names = [str(head).replace("_", " ")] + list(aliases.get(head, []))
            if not any(_surface_match(n, low) for n in names):
                continue
            for rel, tail in edges:
                key = (head, rel, tail)
                if key in seen or not _tail_in_text(tail, answer, low):
                    continue
                seen.add(key)
                claims.append(QueryTriple(head=head, relation=rel, tail=tail))
        return claims
