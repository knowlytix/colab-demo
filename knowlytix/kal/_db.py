"""SQLAlchemy base + mixin for KAL ORM models.

The connection registry and external-ID registry models live in this
package. They need a ``DeclarativeBase`` (so SQLAlchemy can map them)
and a UUID-primary-key mixin (matching the convention every KAL row
uses). v0.1.0 leaned on knowly's ``knowly.shared.database.Base`` —
that's a knowly-specific symbol; the standalone package owns its own.

Consuming applications that ship their own ``DeclarativeBase`` and
want every model under one ``metadata`` can either:

- Set ``KAL_SHARE_METADATA=1`` and call
  ``KalBase.metadata = MyAppBase.metadata`` before any model imports
  (advanced; relies on import-order discipline).
- Run KAL's Alembic migrations under a separate ``version_table``
  + ``version_table_schema`` so the two metadata don't fight.

Default is the second: KAL's tables live in the ``kal`` Postgres
schema (configurable via ``KAL_DB_SCHEMA`` env var), and migrations
track their own version table. No FK references back to consumer-
owned tables; tenant_id is a bare UUID column and the consuming app
is responsible for referential integrity.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Schema name for every table this package owns. Configurable via
# the ``KAL_DB_SCHEMA`` env var so a host application can isolate KAL
# tables under any schema it manages (Alembic migrations honour the
# same env var). Default ``kal`` rather than knowly's historical
# ``knowly`` — the standalone package shouldn't squat on a host-
# application name.
KAL_DB_SCHEMA: str = os.environ.get("KAL_DB_SCHEMA", "kal")


class KalBase(DeclarativeBase):
    """Declarative base for KAL-owned ORM tables.

    Distinct from the consuming application's ``DeclarativeBase``: KAL
    tables track their own ``metadata`` so Alembic autogenerate over
    the consumer's models doesn't see them (and vice versa). If a
    consumer wants unified metadata, they can reassign
    ``KalBase.metadata`` before any model module imports — but the
    default keeps them separate.
    """


class UUIDPrimaryKeyMixin:
    """Mixin that adds a UUID primary key column.

    Matches knowly's pre-v0.2.0 ``knowly.shared.models.UUIDPrimaryKeyMixin``
    exactly so a knowly cutover doesn't have to migrate IDs.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )


class TimestampMixin:
    """Mixin that adds ``created_at`` + ``updated_at`` timestamp columns.

    Prep for v0.3.0 — added because the upcoming ``KalDefaultKGStore``
    ORM models (``KalKgEntity``, ``KalKgRelation``, ``KalKgTriple``,
    ``KalPolicyRule``, ``KalVerificationRun``, ...) inherit from this
    mixin. Mirrors knowly's ``knowly.shared.models.TimestampMixin``
    exactly so cross-tree federation reads see identical column
    semantics.
    """

    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now(),
    )
