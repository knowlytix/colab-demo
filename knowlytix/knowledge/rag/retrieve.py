# SPDX-License-Identifier: Apache-2.0
"""GMS retrieval over bound query triples, with provenance.

Answers triple-mediated queries through the GMS only (``GEODE_RAG_DESIGN.md``
§3): a ``?``-tail resolves via ``link_predict``, a ``?``-head via pattern match,
an all-known triple via ``score_triple``. Every matched triple carries its
source span from the :class:`ProvenanceLedger`, so the synthesizer reads real
document text — the provenance-span trick that lets coarse triples cover prose.

Multi-hop chains are resolved greedily over a variable environment: a variable
bound by one triple (e.g. ``?x``) feeds the next. The bare ``?`` slot is the
asked value; its candidates become the answer set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from knowlytix.knowledge.rag.binding import BoundTriple
from knowlytix.knowledge.rag.query_triples import ASKED, is_var

__all__ = ["RetrievedFact", "RetrieveResult", "Retriever"]


def _confidence(score: float) -> float:
    """Map a geodesic distance (>=0, lower=better) to a [0,1] confidence."""
    return float(math.exp(-max(0.0, score)))


@dataclass
class RetrievedFact:
    head: str
    relation: str
    tail: str
    score: float                 # geodesic distance (0 == exact/asserted)
    confidence: float
    source: str                  # "link_predict" | "triple" | "score" | "enm"
    location: str | None = None  # file:line:char
    raw: str | None = None       # source span text


@dataclass
class RetrieveResult:
    facts: list[RetrievedFact] = field(default_factory=list)
    answers: list[tuple[str, float]] = field(default_factory=list)  # (value, conf)

    @property
    def matched(self) -> bool:
        return bool(self.facts)


class Retriever:
    """Resolve bound query triples against the store; attach provenance."""

    def __init__(self, store, ledger=None, *, top_k: int = 10):
        self.store = store
        self.ledger = ledger
        self.top_k = top_k

    def retrieve(self, bound: list[BoundTriple]) -> RetrieveResult:
        env: dict[str, str] = {}     # variable name -> resolved entity
        result = RetrieveResult()
        for bt in bound:
            if not bt.bound:
                continue
            self._resolve_one(bt, env, result)
        return result

    def _resolve_one(self, bt: BoundTriple, env: dict[str, str],
                     result: RetrieveResult) -> None:
        head = env.get(bt.head, bt.head) if is_var(bt.head) else bt.head
        tail = env.get(bt.tail, bt.tail) if is_var(bt.tail) else bt.tail
        rel = bt.relation
        if rel is None:
            return

        head_unknown = is_var(head)
        tail_unknown = is_var(tail)

        if not head_unknown and tail_unknown:
            self._tail_query(head, rel, bt.tail, env, result)
        elif head_unknown and not tail_unknown:
            self._head_query(bt.head, rel, tail, env, result)
        elif not head_unknown and not tail_unknown:
            self._score_query(head, rel, tail, result)
        # both-unknown: unsupported in P1 (needs full pattern scan) — skipped.

    def _tail_query(self, head: str, rel: str, tail_var: str,
                    env: dict[str, str], result: RetrieveResult) -> None:
        # Prefer asserted edges: an in-graph (head, rel, tail) is exact and
        # provenance-consistent (GMS-authoritative). link_predict is only the
        # fallback for genuinely missing edges, scored lower.
        asserted = self.store.query_triples(head=head, relation=rel)
        if asserted:
            for h, r, t in asserted:
                result.facts.append(self._fact(h, r, t, 0.0, "triple"))
            env[tail_var] = asserted[0][2]
            if tail_var == ASKED:
                result.answers = [(t, 1.0) for _, _, t in asserted]
            return
        cands = self.store.link_predict(head, rel, top_k=self.top_k)
        if not cands:
            return
        best, best_score = cands[0]
        env[tail_var] = best
        result.facts.append(self._fact(head, rel, best, best_score,
                                       "link_predict"))
        if tail_var == ASKED:
            result.answers = [(v, _confidence(s)) for v, s in cands]

    def _head_query(self, head_var: str, rel: str, tail: str,
                    env: dict[str, str], result: RetrieveResult) -> None:
        matches = self.store.query_triples(relation=rel, tail=tail)
        if not matches:
            return
        for h, r, t in matches:
            result.facts.append(self._fact(h, r, t, 0.0, "triple"))
        env[head_var] = matches[0][0]
        if head_var == ASKED:
            result.answers = [(h, 1.0) for h, _, _ in matches]

    def _score_query(self, head: str, rel: str, tail: str,
                     result: RetrieveResult) -> None:
        score = self.store.score_triple(head, rel, tail)
        if score is None:
            return
        result.facts.append(self._fact(head, rel, tail, score, "score"))

    def _fact(self, head: str, rel: str, tail: str, score: float,
              source: str) -> RetrievedFact:
        location = raw = None
        if self.ledger is not None:
            prov = self.ledger.resolve(head, rel, tail)
            if prov.line_no > 0:
                location, raw = prov.location(), prov.raw
        return RetrievedFact(head=head, relation=rel, tail=tail, score=score,
                             confidence=_confidence(score), source=source,
                             location=location, raw=raw)
