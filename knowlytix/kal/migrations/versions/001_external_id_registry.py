"""Add KAL external_id_registry — federation metadata store.

Creates the KAL Postgres schema (default ``kal``, configurable via
``KAL_DB_SCHEMA``) and the ``external_id_registry`` table specified
in KAL spec §6.4 Layer 4. Maps
``(tenant_id, source_adapter, external_id, resource_type)`` to a
canonical ``internal_id`` so the same logical entity reachable through
different backend-specific identifiers (Postgres UUIDs, Neo4j node IDs,
SPARQL URIs, etc.) collapses to a single internal handle.

This is the first migration in the standalone ``knowlytix-kal`` v0.2.0
package, ported from knowly's ``alembic/versions/045_external_id_registry.py``.
Schema renamed from ``knowlytix`` → configurable ``KAL_DB_SCHEMA``;
tenant FK to ``knowly.tenants(id)`` dropped (consuming app owns
tenant referential integrity).

ROLLBACK:
    Drops the table and its indexes. Does NOT drop the schema —
    future framework-level tables may live there.

VERIFY:
    SELECT tenant_id, source_adapter, external_id, resource_type, internal_id
    FROM <KAL_DB_SCHEMA>.external_id_registry LIMIT 1;

Revision ID: kal_001
Revises:
Create Date: 2026-05-16 00:00:00.000000
"""

import os

import sqlalchemy as sa
from alembic import op

revision = "kal_001"
down_revision = None
branch_labels = ("knowlytix_kal",)
depends_on = None

SCHEMA = os.environ.get("KAL_DB_SCHEMA", "kal")
TABLE = "external_id_registry"
UNIQUE_NAME = "uq_external_id_registry_composite"
REVERSE_INDEX = "ix_external_id_registry_reverse"


def upgrade() -> None:
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    op.create_table(
        TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("source_adapter", sa.String(64), nullable=False),
        sa.Column("external_id", sa.String(512), nullable=False),
        sa.Column("resource_type", sa.String(16), nullable=False),
        sa.Column("internal_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        # No tenant FK — the consuming application owns tenant
        # referential integrity. Apps that want a constraint can add
        # it in an app-side follow-up migration.
        sa.UniqueConstraint(
            "tenant_id",
            "source_adapter",
            "external_id",
            "resource_type",
            name=UNIQUE_NAME,
        ),
        schema=SCHEMA,
    )
    # Reverse-lookup index for ``lookup_external_ids``: filter by
    # ``(tenant_id, internal_id, resource_type)``; ``INCLUDE`` covers
    # the projection so the planner can answer the query with an
    # index-only scan and never touch the heap.
    op.create_index(
        REVERSE_INDEX,
        TABLE,
        ["tenant_id", "internal_id", "resource_type"],
        postgresql_include=["source_adapter", "external_id"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index(REVERSE_INDEX, table_name=TABLE, schema=SCHEMA)
    op.drop_table(TABLE, schema=SCHEMA)
    # NOTE: schema is left in place — future framework-level tables
    # may live there. Dropping it would force a `CREATE SCHEMA`
    # on the next migration that uses it.
