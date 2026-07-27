"""Add ``source`` + ``extractor`` provenance columns to ``kal.kg_triples`` (v0.10.0).

kal-side half of the MCP-ingestion provenance work
(``docs/plans/plan_knowlytix_kal_v0_10.md`` Q1 decision B). Adds two
nullable columns to ``kal.kg_triples``:

- ``source`` (TEXT, indexed) — the revoke-by-source key for triples
  ingested from an external MCP server (e.g. ``mcp://<connection>``).
- ``extractor`` (TEXT) — which extractor/tool produced the triple
  (e.g. ``mcp:resource:<uri>``).

Mirrored to the public KAL package as ``kal_009`` per the revision map in
``scripts/MIRROR_MANIFEST.toml``. Adding nullable columns is
metadata-only in Postgres (instant, no rewrite). Existing rows read back
NULL for both; no backfill required.

ROLLBACK:
    WARNING: drops both columns + their data + the source index. Triples
    ingested under v0.10.0+ lose their MCP origin tag; the rows
    themselves stay (FK chains unchanged).

    def downgrade():
        op.drop_index(
            "idx_kal_kg_triples_source", table_name="kg_triples", schema="kal"
        )
        op.drop_column("kg_triples", "extractor", schema="kal")
        op.drop_column("kg_triples", "source", schema="kal")

VERIFY:
    SELECT column_name FROM information_schema.columns
    WHERE table_schema = 'kal' AND table_name = 'kg_triples'
      AND column_name IN ('source', 'extractor');

Revision ID: kal_009
Revises: kal_008
Create Date: 2026-05-20 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "kal_009"
down_revision = "kal_008"
branch_labels = None
depends_on = None

SCHEMA = "kal"


def upgrade() -> None:
    op.add_column(
        "kg_triples",
        sa.Column(
            "source",
            sa.Text(),
            nullable=True,
            comment="Provenance origin (revoke key), e.g. mcp://<connection>.",
        ),
        schema=SCHEMA,
    )
    op.add_column(
        "kg_triples",
        sa.Column(
            "extractor",
            sa.Text(),
            nullable=True,
            comment="Extractor/tool that produced the triple, e.g. mcp:resource:<uri>.",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_kg_triples_source", "kg_triples", ["source"], schema=SCHEMA
    )


def downgrade() -> None:
    op.drop_index(
        "idx_kal_kg_triples_source", table_name="kg_triples", schema=SCHEMA
    )
    op.drop_column("kg_triples", "extractor", schema=SCHEMA)
    op.drop_column("kg_triples", "source", schema=SCHEMA)
