# SPDX-License-Identifier: Apache-2.0
"""Extraction — the isolated parse⇄bind result of triple-mediated RAG.

The first half of :meth:`RagPipeline.query` (parse the question into query triples,
then bind them to the graph's vocabulary) is a reusable step in its own right: it
turns a natural-language message into the policy-grounded ``(head, relation, ?)``
triples the store can actually answer, and abstains (nothing binds) rather than
guess. Callers that only need *what a message grounds to* — claim-decomposition
extraction, an escalation signal keyed on the bound heads — want that artifact
without paying for retrieval or generation. Callers that need an answer can compute
the extraction once and hand it back into :meth:`RagPipeline.query` so the message
is parsed exactly once.

:class:`Extraction` is that artifact. It is fully serializable (``to_dict`` /
``from_dict``) so it travels through tool-call arguments and audit logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowlytix.knowledge.rag.binding import BoundTriple
from knowlytix.knowledge.rag.query_triples import QueryTriple

__all__ = ["Extraction"]


@dataclass
class Extraction:
    """The parse⇄bind result: query triples and their binding to graph vocab."""

    query_triples: list[QueryTriple] = field(default_factory=list)
    bound_triples: list[BoundTriple] = field(default_factory=list)

    @property
    def is_bound(self) -> bool:
        """True iff at least one triple resolved to the graph — the same
        bind-check :meth:`RagPipeline.query` uses to decide grounded-vs-abstain."""
        return bool(self.query_triples) and any(b.bound for b in self.bound_triples)

    @property
    def bound_facts(self) -> list[tuple[str, str, str]]:
        """The bound ``(head, relation, tail)`` triples — the grounded facts a
        message yields, with the asked/variable slots dropped. This is what an
        escalation signal keys on (each head is a policy entity)."""
        return [(b.head, b.relation, b.tail)
                for b in self.bound_triples if b.bound]

    def to_dict(self) -> dict:
        """Serialize to plain JSON-able types for tool-call args / audit logs."""
        return {
            "query_triples": [list(t.as_tuple()) for t in self.query_triples],
            "bound_triples": [
                {"original": list(b.original.as_tuple()),
                 "head": b.head, "relation": b.relation, "tail": b.tail}
                for b in self.bound_triples
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Extraction":
        """Rebuild from :meth:`to_dict` output so a consumer can hand a prior
        extraction back into the pipeline without re-parsing."""
        qts = [QueryTriple(*t) for t in d.get("query_triples", [])]
        bts = [
            BoundTriple(original=QueryTriple(*b["original"]),
                        head=b.get("head"), relation=b.get("relation"),
                        tail=b.get("tail"))
            for b in d.get("bound_triples", [])
        ]
        return cls(query_triples=qts, bound_triples=bts)
