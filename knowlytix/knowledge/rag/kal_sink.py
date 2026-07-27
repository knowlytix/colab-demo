# SPDX-License-Identifier: Apache-2.0
"""Persist GEODE-corrected knowledge into KAL (Knowledge Adapter Layer).

KAL (``knowlytix.kal``) is the org's backend-agnostic, Postgres/pgvector-backed
knowledge-graph store. This module is the bridge: it converts a built RAG store's
triples — with their source-span provenance and (optionally) GEODE verification —
into :class:`KALTriple` records and writes them through a KAL
:class:`KnowledgeAdapter`, so the self-corrected graph lives in an external
database instead of only on local disk.

Scope: the *graph* (triples + provenance + verification) goes to KAL; numeric
ENM values ride along as triple ``object_literal``s. The trained GMS model
checkpoint is a separate artifact and is NOT KAL's concern.

This module imports ``knowlytix.kal`` at top level, so it is an opt-in extra (it
is intentionally not re-exported from ``knowlytix.knowledge.rag.__init__``).
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime

from knowlytix.kal import (
    KALLiteral,
    KALNode,
    KALTriple,
    TripleProvenance,
    VerificationMetadata,
)
from knowlytix.knowledge.geode.provenance import ProvenanceLedger, canon

__all__ = [
    "store_to_kal_triples",
    "persist_store_to_kal",
    "persist_store_to_kal_sync",
    "kal_postgres_adapter",
]

_NUM_RE = re.compile(r"^-?[\d,]*\.?\d+%?$")


def _node(name: str) -> KALNode:
    c = canon(name)
    return KALNode(id=c, name=str(name), normalized_name=c)


def _is_numeric(tail: str) -> bool:
    return bool(_NUM_RE.match(str(tail).strip()))


def _to_kal(h: str, r: str, t: str, *, source: str | None, extractor: str,
            ledger: ProvenanceLedger | None, extracted_at: datetime | None,
            confidence: float | None) -> KALTriple:
    source_text = None
    if ledger is not None:
        p = ledger.resolve(h, r, t)
        if p.line_no > 0:
            source_text = p.raw
    provenance = TripleProvenance(source=source, extractor=extractor,
                                  extracted_at=extracted_at,
                                  source_text=source_text)
    # Numeric tails (exact ENM values) become literals; entities become nodes.
    if _is_numeric(t):
        obj, obj_literal = None, KALLiteral(value=str(t), datatype="number")
    else:
        obj, obj_literal = _node(t), None
    verification = None
    if confidence is not None:
        verification = VerificationMetadata(
            confidence=confidence, verification_status="verified",
            verifier=extractor)
    return KALTriple(subject=_node(h), predicate=r, object=obj,
                     object_literal=obj_literal, provenance=provenance,
                     verification=verification)


def store_to_kal_triples(store, *, source: str | None = None,
                         extractor: str = "geode",
                         ledger: ProvenanceLedger | None = None,
                         exclude_relations: tuple[str, ...] = ("in_section",),
                         extracted_at: datetime | None = None,
                         confidence: float | None = None) -> list[KALTriple]:
    """Convert a built RAG store's triples into KAL triples.

    Provenance ``source_text`` is recovered from the store's markdown via
    :class:`ProvenanceLedger` (built from ``store.markdown`` when ``ledger`` is
    not supplied). Pass ``confidence`` to stamp GEODE verification metadata.
    """
    src = source if source is not None else getattr(store, "store_path", None)
    led = ledger
    if led is None and getattr(store, "markdown", ""):
        led = ProvenanceLedger.from_text(store.markdown)
    return [
        _to_kal(h, r, t, source=src, extractor=extractor, ledger=led,
                extracted_at=extracted_at, confidence=confidence)
        for (h, r, t) in store.triples if r not in exclude_relations
    ]


async def persist_store_to_kal(adapter, store, *, tenant_id: str | None = None,
                               **kwargs) -> int:
    """Write the store's triples through a KAL ``KnowledgeAdapter`` (async).

    Returns the number of triples inserted. ``kwargs`` pass through to
    :func:`store_to_kal_triples`.
    """
    triples = store_to_kal_triples(store, **kwargs)
    return await adapter.insert_triples(triples, tenant_id)


def persist_store_to_kal_sync(adapter, store, *, tenant_id: str | None = None,
                              **kwargs) -> int:
    """Synchronous wrapper around :func:`persist_store_to_kal`."""
    return asyncio.run(
        persist_store_to_kal(adapter, store, tenant_id=tenant_id, **kwargs))


def kal_postgres_adapter(*, host: str, database: str, username: str,
                         password: str, port: int = 5432,
                         schema_name: str = "public", name: str = "geode"):
    """Build a Postgres-backed KAL ``KnowledgeAdapter``.

    Requires KAL's migrations to have been applied to the target database.
    Credentials are encrypted at rest by KAL; they are only passed here to the
    builder. Returns the adapter (its operations are async).
    """
    from knowlytix.kal.factory import AdapterFactory

    return AdapterFactory.build(
        "postgres", name,
        {"host": host, "database": database, "port": port,
         "schema_name": schema_name},
        {"username": username, "password": password},
    )
