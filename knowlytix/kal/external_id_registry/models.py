"""ORM model for the federation metadata store.

KAL spec §6.4 Layer 4: maps a backend-specific ``external_id`` for one
adapter to a canonical ``internal_id`` so the same logical entity can
be addressed through heterogeneous backends without identity collisions.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from sqlalchemy import (
    DateTime,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from knowlytix.kal._db import KAL_DB_SCHEMA, UUIDPrimaryKeyMixin
from knowlytix.kal._db import KalBase as Base

# Single source of truth for the constraint + index names. The
# repository references the constraint name in
# ``on_conflict_do_nothing(constraint=...)`` and the migration
# materializes both names — keeping them as module-level constants
# prevents the three definitions from drifting.
UNIQUE_CONSTRAINT_NAME = "uq_external_id_registry_composite"
REVERSE_INDEX_NAME = "ix_external_id_registry_reverse"

# Column-length budgets. ``EXTERNAL_ID_MAX_LEN`` is sized for SPARQL
# URIs and Neo4j composite handles; ``SOURCE_ADAPTER_MAX_LEN`` is a
# slug-style adapter name (matches ``AdapterCapabilities`` naming
# convention).
SOURCE_ADAPTER_MAX_LEN = 64
EXTERNAL_ID_MAX_LEN = 512
RESOURCE_TYPE_MAX_LEN = 16

# ``VARCHAR`` (not a Postgres enum) so new resource_type values can be
# added without a schema migration; mypy narrows callers via the
# ``ResourceType`` Literal below.
type ResourceType = Literal["entity", "triple"]


class ExternalIdRegistry(UUIDPrimaryKeyMixin, Base):
    """Cross-adapter identity mapping.

    Each row records: for tenant T, adapter A asserts that its
    ``external_id`` X refers to the internal ``internal_id`` U for
    resource of type ``resource_type``. The composite-unique constraint
    on ``(tenant_id, source_adapter, external_id, resource_type)``
    guarantees the (T, A, X, type) → U binding is stable: re-registering
    the same key with a different ``internal_id`` is a no-op (the
    repository uses ``ON CONFLICT DO NOTHING``), so the first writer
    wins.

    ``tenant_id`` is a bare UUID column — the standalone package
    doesn't know what a "tenant" means in the consuming application,
    so no FK constraint is enforced. Consuming apps that want
    referential integrity can add their own FK in an app-side
    migration after this table is created.

    No ``updated_at`` — registry rows are append-only under the
    first-writer-wins contract. The single ``created_at`` is enough to
    explain provenance; an updated_at column would imply rebind semantics
    the design explicitly rules out.
    """

    __tablename__ = "external_id_registry"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "source_adapter",
            "external_id",
            "resource_type",
            name=UNIQUE_CONSTRAINT_NAME,
        ),
        # Mirrors the migration's reverse-lookup index: filter by
        # (tenant_id, internal_id, resource_type); ``source_adapter`` /
        # ``external_id`` are INCLUDE columns so reverse lookups are
        # index-only scans.
        Index(
            REVERSE_INDEX_NAME,
            "tenant_id",
            "internal_id",
            "resource_type",
            postgresql_include=["source_adapter", "external_id"],
        ),
        {"schema": KAL_DB_SCHEMA},
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        nullable=False,
    )
    source_adapter: Mapped[str] = mapped_column(
        String(SOURCE_ADAPTER_MAX_LEN), nullable=False
    )
    external_id: Mapped[str] = mapped_column(
        String(EXTERNAL_ID_MAX_LEN), nullable=False
    )
    resource_type: Mapped[str] = mapped_column(
        String(RESOURCE_TYPE_MAX_LEN), nullable=False
    )
    internal_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
