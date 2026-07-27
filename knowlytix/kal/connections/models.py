"""ORM model for ``knowly.kal_connections``.

The persistent home for user-registered adapter connections:
per-tenant rows naming an ``adapter_type``, holding non-secret
``config`` as JSONB, and storing encrypted ``credentials`` as opaque
BYTEA. Liveness (``last_error`` / ``validated_at``) is tracked
separately from admin intent (``status``) so a hydration failure does
not look the same as an intentional disable.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from sqlalchemy import (
    DateTime,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from knowlytix.kal._db import KAL_DB_SCHEMA, UUIDPrimaryKeyMixin
from knowlytix.kal._db import KalBase as Base

# Single source of truth for the unique-constraint name; the migration
# references the same string so renames cannot drift the two.
UNIQUE_CONSTRAINT_NAME = "uq_kal_connections_tenant_name"

# Slug-style user-facing name length. Generous enough for descriptive
# names ("my-customer-postgres-readonly") without blowing past index
# pages — Postgres B-tree keys cap around 2700 bytes total per row, and
# this composite includes the tenant UUID.
NAME_MAX_LEN = 128

# ``adapter_type`` length matches ``AdapterCapabilities`` naming
# convention used in ``external_id_registry.source_adapter`` (64) — same
# slug-like values, kept consistent so a future foreign key between the
# two surfaces is straightforward.
ADAPTER_TYPE_MAX_LEN = 64

# ``status`` is an admin-intent flag. The two-value closed set is
# enforced at the application boundary (Pydantic Literal); using
# ``String(STATUS_MAX_LEN)`` rather than a Postgres ENUM keeps adding
# values (e.g. 'archived') a code-only change.
STATUS_MAX_LEN = 16
type KALConnectionStatus = Literal["active", "disabled"]
DEFAULT_STATUS: KALConnectionStatus = "active"
DISABLED_STATUS: KALConnectionStatus = "disabled"


class KALConnection(UUIDPrimaryKeyMixin, Base):
    """One row per user-registered KG adapter.

    ``tenant_id`` is a bare UUID column — the standalone package
    doesn't know what a "tenant" means in the consuming application,
    so no FK constraint is enforced. Consuming apps that want
    referential integrity can add their own FK in an app-side
    migration after this table is created.

    Field-by-field:

    - ``name`` — the user-facing handle, unique per tenant. The registry
      key in-process is namespaced (``{tenant_id_short}:{name}``) so two
      tenants can both pick e.g. ``wikidata-public`` without colliding.
    - ``adapter_type`` — slug naming the builder in ``AdapterFactory``.
    - ``config`` — non-secret JSONB blob. Safe to surface via API.
    - ``credentials`` — Fernet ciphertext bytes. **Never** appears in any
      API response model.
    - ``capabilities`` — cached ``AdapterCapabilities.model_dump()`` from
      the last successful handshake. Refreshed on ``/test``.
    - ``status`` — admin intent only (``"active"`` / ``"disabled"``).
    - ``last_error`` — last handshake failure message, already truncated
      by the service before write. ``None`` means no failure observed
      since the last success (or no attempts yet).
    - ``validated_at`` — timestamp of the last successful handshake.

    No ``__repr__`` override — SQLAlchemy 2.0 declarative's default
    ``<KALConnection 0x...>`` is column-less and therefore credential-
    safe. Do not add a column-printing repr.
    """

    __tablename__ = "kal_connections"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name=UNIQUE_CONSTRAINT_NAME),
        {"schema": KAL_DB_SCHEMA},
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        nullable=False,
        # No ``index=True`` — the composite UNIQUE on (tenant_id, name)
        # already covers tenant-scoped equality scans via its leading
        # column; a dedicated tenant_id index would double write
        # amplification on every INSERT/UPDATE for no read benefit.
    )
    name: Mapped[str] = mapped_column(String(NAME_MAX_LEN), nullable=False)
    adapter_type: Mapped[str] = mapped_column(
        String(ADAPTER_TYPE_MAX_LEN), nullable=False
    )
    config: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    credentials: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    capabilities: Mapped[dict[str, object] | None] = mapped_column(
        JSONB, nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(STATUS_MAX_LEN), nullable=False, server_default=DEFAULT_STATUS
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
