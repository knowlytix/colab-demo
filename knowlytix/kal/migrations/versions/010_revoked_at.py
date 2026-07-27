"""Add ``revoked_at`` quarantine column to ``kal.kg_triples`` (v0.11.0).

kal-side half of revoke-by-source (quarantine). Adds a nullable
``revoked_at`` timestamp: NULL = live, non-NULL = revoked at that time.
Read paths exclude non-NULL rows unless ``include_revoked`` is set;
reversible via restore.

Mirrored to the public KAL package as ``kal_010`` per the revision map in
``scripts/MIRROR_MANIFEST.toml``. Nullable ``ADD COLUMN`` is metadata-only
in Postgres (instant, no rewrite); existing rows read back NULL. No backfill.

ROLLBACK:
    WARNING: drops the ``revoked_at`` column + its data; quarantine state
    is lost (rows become live).

    def downgrade():
        op.drop_column("kg_triples", "revoked_at", schema="kal")

VERIFY:
    SELECT column_name FROM information_schema.columns
    WHERE table_schema = 'kal' AND table_name = 'kg_triples'
      AND column_name = 'revoked_at';

Revision ID: kal_010
Revises: kal_009
Create Date: 2026-05-20 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "kal_010"
down_revision = "kal_009"
branch_labels = None
depends_on = None

SCHEMA = "kal"


def upgrade() -> None:
    op.add_column(
        "kg_triples",
        sa.Column(
            "revoked_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Quarantine timestamp; NULL = live, non-NULL = revoked.",
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("kg_triples", "revoked_at", schema=SCHEMA)
