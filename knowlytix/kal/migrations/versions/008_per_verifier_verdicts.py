"""Create ``kal.triple_verdicts`` for per-(triple, verifier) verdict rows.

Phase A step 1 of the v0.5.0 plan
(``docs/plans/plan_knowlytix_kal_v0_5.md``). Mirror of 053 for the
``kal`` schema — same rationale, same semantics, applied to the
standalone-package ORM (``KalKgTriple``).

This migration is mirrored as ``kal_008_per_verifier_verdicts.py`` per
the revision map in ``scripts/MIRROR_MANIFEST.toml``.

ROLLBACK:
    See migration 053's docstring. Same caveats — only roll back
    together with the v0.5.0 ``update_verification`` rewrite.

    def downgrade():
        op.drop_table("triple_verdicts", schema="kal")

VERIFY:
    SELECT count(*) FROM kal.triple_verdicts;
    -- After the back-fill, should be >= count of distinct verifiers
    --   across kal.kg_triples WHERE verifier IS NOT NULL.

Revision ID: kal_008
Revises: kal_007
Create Date: 2026-05-18 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "kal_008"
down_revision = "kal_007"
branch_labels = None
depends_on = None

SCHEMA = "kal"


def upgrade() -> None:
    op.create_table(
        "triple_verdicts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("triple_id", sa.Uuid(), nullable=False),
        sa.Column("verifier", sa.String(255), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("verification_status", sa.String(20), nullable=True),
        sa.Column(
            "verified_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("scores", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["triple_id"],
            [f"{SCHEMA}.kg_triples.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "triple_id", "verifier", name="uq_kal_triple_verdict"
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_triple_verdicts_triple",
        "triple_verdicts",
        ["triple_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_triple_verdicts_verifier",
        "triple_verdicts",
        ["tenant_id", "verifier"],
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_triple_verdicts_conf",
        "triple_verdicts",
        ["tenant_id", "confidence"],
        schema=SCHEMA,
    )

    # Back-fill from existing kal.kg_triples — idempotent.
    op.execute(
        f"""
        INSERT INTO {SCHEMA}.triple_verdicts
            (id, tenant_id, triple_id, verifier, confidence,
             verification_status, verified_at, scores)
        SELECT
            gen_random_uuid(),
            tenant_id,
            id,
            verifier,
            confidence,
            verification_status,
            verified_at,
            scores
        FROM {SCHEMA}.kg_triples
        WHERE verifier IS NOT NULL
        ON CONFLICT (triple_id, verifier) DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_index(
        "idx_kal_triple_verdicts_conf",
        "triple_verdicts",
        schema=SCHEMA,
    )
    op.drop_index(
        "idx_kal_triple_verdicts_verifier",
        "triple_verdicts",
        schema=SCHEMA,
    )
    op.drop_index(
        "idx_kal_triple_verdicts_triple",
        "triple_verdicts",
        schema=SCHEMA,
    )
    op.drop_table("triple_verdicts", schema=SCHEMA)
