# SPDX-License-Identifier: Apache-2.0
"""Load triples from heterogeneous sources via KAL, as (head, relation, tail).

Phase A uses the read-only JSONL adapter over the finance fixture (source #4,
"triple data" — no extraction). Later phases add MCP / postgres / Neo4j sources
behind the same ``-> list[Triple]`` shape so the GMS side is source-agnostic.

Public ``knowlytix.kal`` imports (mirror of ``knowly.kal``).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from .resources import resolve_fixture_path

from knowlytix.kal.adapters.jsonl import JsonlKnowledgeAdapter
from knowlytix.kal.types import KALQuery, KALTriple, KALTriplePattern

Triple = tuple[str, str, str]

# Match-all: a single empty pattern (all fields None). Bump the limit if a
# fixture ever exceeds it.
_MATCH_ALL = KALQuery(patterns=[KALTriplePattern()], limit=10_000)


def to_triple(t: KALTriple) -> Triple:
    """Map a KALTriple to ``(head, relation, tail)`` using node/literal names.

    Public so callers that load raw ``KALTriple``s (e.g. for the provenance viz
    that needs ``.adapter`` and ``.subject.node_types``) can still hand them
    through to the GMS pipeline, which only wants the 3-tuple form.
    """
    head = t.subject.name
    rel = t.predicate
    if t.object is not None:
        tail = t.object.name
    elif t.object_literal is not None:
        tail = t.object_literal.value
    else:  # pragma: no cover - schema guarantees one of the two
        raise ValueError(f"Triple {t.id!r} has neither object nor object_literal")
    return (head, rel, tail)


async def load_kal_triples(adapter_name: str, path: str | Path) -> list[KALTriple]:
    """Read all triples from a JSONL KAL source as raw ``KALTriple`` objects.

    Returns the rich form (carries ``.adapter`` + ``.subject.node_types`` for
    provenance viz). Use :func:`to_triple` to project to the 3-tuple form the
    GMS pipeline expects, or :func:`load_triples` if you only need 3-tuples.

    .. note::
       ``KALTriple.adapter`` is normally populated by ``FederationRouter`` when
       it merges results from multiple registered adapters, **not** by a bare
       adapter on its own ``query_triples`` return — those come back with
       ``adapter=None``. For our single-source loading pattern that's
       inconvenient (the provenance viz needs an adapter name), so this
       helper stamps ``adapter_name`` onto every returned triple itself.
    """
    resolved_path = resolve_fixture_path(path)
    adapter = JsonlKnowledgeAdapter(adapter_name, path=resolved_path)
    result = await adapter.query_triples(_MATCH_ALL)
    triples = list(result.triples)
    for t in triples:
        t.adapter = adapter_name
    return triples


async def load_triples(adapter_name: str, path: str | Path) -> list[Triple]:
    """Read all triples from a JSONL KAL source as ``(head, relation, tail)``."""
    return [to_triple(t) for t in await load_kal_triples(adapter_name, path)]


def load_triples_sync(adapter_name: str, path: str | Path) -> list[Triple]:
    """Sync wrapper for notebooks / scripts (KAL adapters are async)."""
    return asyncio.run(load_triples(adapter_name, path))


async def load_triples_neo4j(
    *,
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
    node_key: str = "name",
) -> list[Triple]:
    """Project a Neo4j property graph to ``(head, relation, tail)`` via the KAL
    Neo4j adapter — the faithful Lane-2 path: each ``(a)-[r]->(b)`` edge maps
    1:1 to a triple, deterministically (no LLM, unlike Lane-1 extraction)."""
    from knowlytix.kal.adapters.neo4j import Neo4jKnowledgeAdapter

    adapter = Neo4jKnowledgeAdapter(
        name="neo4j", uri=uri, user=user, password=password,
        database=database, node_key=node_key,
    )
    try:
        result = await adapter.query_triples(_MATCH_ALL)
        return [to_triple(t) for t in result.triples]
    finally:
        await adapter.aclose()


def load_triples_neo4j_sync(
    *,
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
    node_key: str = "name",
) -> list[Triple]:
    """Sync wrapper for ``load_triples_neo4j``."""
    return asyncio.run(load_triples_neo4j(
        uri=uri, user=user, password=password, database=database, node_key=node_key,
    ))
