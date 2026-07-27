"""Add KAL connections registry table.

The table stores one row per user-registered adapter: per-tenant
``name`` + ``adapter_type`` slug + JSONB ``config`` + Fernet-
ciphertext ``credentials`` BYTEA + cached ``capabilities`` JSONB +
admin-intent ``status`` + last-known liveness (``last_error``,
``validated_at``).

Ported from knowly's ``alembic/versions/046_kal_connections.py``.
Schema renamed from ``knowly`` → configurable ``KAL_DB_SCHEMA``
(default ``kal``); tenant FK to ``knowly.tenants(id)`` dropped
(consuming app owns tenant referential integrity).

ROLLBACK:
    WARNING: Drops all stored connection configs. Re-registration
    requires re-supplying every credential.

VERIFY:
    SELECT id, tenant_id, name, adapter_type, status, validated_at
    FROM <KAL_DB_SCHEMA>.kal_connections LIMIT 5;

Revision ID: kal_002
Revises: kal_001
Create Date: 2026-05-16 00:00:00.000000
"""

import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# Column-length budgets shared with the SQLAlchemy model — single
# source of truth lives in ``knowlytix.kal.connections.models``; the
# migration imports the constants so a rename in one place fails the
# migration's autogenerate-diff rather than silently drifting.
from knowlytix.kal.connections.models import (
    ADAPTER_TYPE_MAX_LEN,
    NAME_MAX_LEN,
    STATUS_MAX_LEN,
)

revision = "kal_002"
down_revision = "kal_001"
branch_labels = None
depends_on = None

SCHEMA = os.environ.get("KAL_DB_SCHEMA", "kal")
TABLE = "kal_connections"
UNIQUE_NAME = "uq_kal_connections_tenant_name"


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(NAME_MAX_LEN), nullable=False),
        sa.Column("adapter_type", sa.String(ADAPTER_TYPE_MAX_LEN), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column("credentials", sa.LargeBinary(), nullable=False),
        sa.Column("capabilities", postgresql.JSONB(), nullable=True),
        sa.Column(
            "status",
            sa.String(STATUS_MAX_LEN),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "validated_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        # No tenant FK — the consuming application owns tenant
        # referential integrity. Apps that want a constraint can add
        # it in an app-side follow-up migration.
        sa.UniqueConstraint("tenant_id", "name", name=UNIQUE_NAME),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table(TABLE, schema=SCHEMA)
