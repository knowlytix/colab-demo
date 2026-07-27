"""KAL adapter implementations.

v0.2.0 ships ``MockKnowledgeAdapter`` + ``WikidataKnowledgeAdapter``.
v0.7.0 adds ``JsonlKnowledgeAdapter`` (read-only, stdlib-only deps).
v0.9.0 adds ``VectorNodeKnowledgeAdapter`` (read-only, numpy-only
deps; numpy is already a runtime dep transitively via ``pgvector``).
Concrete adapter classes are importable from this subpackage but the
heavier ones are deliberately not re-exported at the package root:
``httpx`` for Wikidata, ``sqlalchemy`` + ``asyncpg`` for Postgres
should not be paid by consumers that only use the protocol surface.
The JSONL, Mock, and VectorNode adapters carry no *extra* deps and
are re-exported here for ergonomics.

Postgres adapter deferred to v0.3.0 pending the KGStore Protocol
decoupling — see ``docs/plans/plan_knowlytix_kal_v0_2.md``.
"""

from knowlytix.kal.adapters.jsonl import JsonlKnowledgeAdapter
from knowlytix.kal.adapters.mock import MockKnowledgeAdapter
from knowlytix.kal.adapters.vector_node import VectorNodeKnowledgeAdapter

__all__ = [
    "JsonlKnowledgeAdapter",
    "MockKnowledgeAdapter",
    "VectorNodeKnowledgeAdapter",
]
