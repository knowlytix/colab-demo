"""Data access layer for the federation metadata store.

Four async helpers wrapping ``knowlytix.external_id_registry``. All
write helpers use ``INSERT ... ON CONFLICT DO NOTHING`` so repeated
registration of the same composite key is idempotent — the first
``(tenant, source_adapter, external_id, resource_type) → internal_id``
binding persists; later calls with a different ``internal_id`` are
silently dropped. Adapters need this because a recovery path may
re-register entities it has already seen.

Caller controls commit: every helper takes a live ``AsyncSession`` and
does NOT call ``commit()``. Atomicity with caller-side work (e.g., an
entity insert sharing the same session) is therefore guaranteed.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from knowlytix.kal.external_id_registry.models import (
    UNIQUE_CONSTRAINT_NAME,
    ExternalIdRegistry,
    ResourceType,
)


async def register(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    source_adapter: str,
    external_id: str,
    resource_type: ResourceType,
    internal_id: uuid.UUID,
) -> None:
    """Insert one (tenant, source, ext_id, type) → internal_id mapping.

    Idempotent under the composite unique constraint. The first writer
    wins; subsequent registrations with a different ``internal_id``
    for the same composite key are silently ignored. Adapters that
    need rebind semantics should explicitly delete the row first
    (out of scope for v1).
    """
    stmt = (
        insert(ExternalIdRegistry)
        .values(
            tenant_id=tenant_id,
            source_adapter=source_adapter,
            external_id=external_id,
            resource_type=resource_type,
            internal_id=internal_id,
        )
        .on_conflict_do_nothing(constraint=UNIQUE_CONSTRAINT_NAME)
    )
    await session.execute(stmt)


async def bulk_register(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    source_adapter: str,
    mappings: list[tuple[str, uuid.UUID]],
    resource_type: ResourceType,
) -> int:
    """Insert N (external_id, internal_id) pairs for a single adapter.

    Returns the count of newly inserted rows (zero if every pair
    already existed, between 0 and N for partial overlap). Empty
    input returns 0 without issuing SQL. Same idempotency contract
    as :func:`register`.
    """
    if not mappings:
        return 0
    rows = [
        {
            "tenant_id": tenant_id,
            "source_adapter": source_adapter,
            "external_id": external_id,
            "resource_type": resource_type,
            "internal_id": internal_id,
        }
        for external_id, internal_id in mappings
    ]
    # ``rowcount`` on ``INSERT ... ON CONFLICT DO NOTHING`` counts only
    # rows actually inserted; suppressed conflicts are excluded.
    stmt = (
        insert(ExternalIdRegistry)
        .values(rows)
        .on_conflict_do_nothing(constraint=UNIQUE_CONSTRAINT_NAME)
    )
    result = await session.execute(stmt)
    # ``rowcount`` lives on ``CursorResult``; SQLAlchemy types
    # ``session.execute`` as the base ``Result``, which doesn't expose
    # it. Same `type: ignore` pattern as kg/repository.py uses.
    rowcount: int = result.rowcount  # type: ignore[attr-defined]
    return rowcount


async def lookup_internal_id(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    source_adapter: str,
    external_id: str,
    resource_type: ResourceType,
) -> uuid.UUID | None:
    """Resolve ``(adapter, external_id) → internal_id``; ``None`` when unknown.

    Hits the composite-unique constraint's backing index.
    """
    stmt = select(ExternalIdRegistry.internal_id).where(
        ExternalIdRegistry.tenant_id == tenant_id,
        ExternalIdRegistry.source_adapter == source_adapter,
        ExternalIdRegistry.external_id == external_id,
        ExternalIdRegistry.resource_type == resource_type,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def lookup_external_ids(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    internal_id: uuid.UUID,
    resource_type: ResourceType,
) -> dict[str, str]:
    """Resolve ``internal_id → {source_adapter: external_id}``.

    Hits ``ix_external_id_registry_reverse`` — column order matches the
    WHERE filter exactly, and the index INCLUDEs the projected columns
    so the query is an index-only scan.
    """
    stmt = select(
        ExternalIdRegistry.source_adapter, ExternalIdRegistry.external_id
    ).where(
        ExternalIdRegistry.tenant_id == tenant_id,
        ExternalIdRegistry.internal_id == internal_id,
        ExternalIdRegistry.resource_type == resource_type,
    )
    result = await session.execute(stmt)
    return {row.source_adapter: row.external_id for row in result}
