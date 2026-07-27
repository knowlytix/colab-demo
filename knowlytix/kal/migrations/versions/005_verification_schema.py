"""Create verification-chain tables in ``kal`` schema.

Phase B Unit 4 of the v0.3.0 plan
(``docs/plans/plan_knowlytix_kal_v0_3.md``). Ports knowly's
``VerificationRun`` (``knowly.verification_runs``), ``PolicyRule``
(``knowly.policy_rules``), and ``VerificationEvidence``
(``kg.verification_evidence``) into the ``kal`` schema as
``kal.verification_runs``, ``kal.policy_rules``, and
``kal.kg_evidence``.

The package collapses knowly's verification surface into a single
``kal.*`` neighborhood. ``KalDefaultKGStore.insert_triples`` does NOT
populate these tables — they're for a downstream verification
pipeline that the package consumer wires up. The schema ships so
``KalKgEvidence`` FKs can resolve and a future
``KalVerificationStore`` Protocol (out of scope for v0.3.0) has its
home ready.

Column shapes (including the 768-d ``aggregate_logical_embedding``
NLI dim + the ``rfc2119_severity`` check constraint) byte-mirror
knowly's so federation between knowly-backed and package-backed
deployments round-trips evidence without column drift. The mirror
script (Phase C) ships this migration into the standalone package
as ``kal_005_verification_schema.py``.

ROLLBACK:
    WARNING: drops all verification-chain data in the kal schema —
    runs, rules, evidence rows. The ``kal.kg_*`` + GMS tables from
    migrations 047 / 048 are untouched.

    def downgrade():
        op.drop_table("kg_evidence", schema="kal")
        op.drop_table("policy_rules", schema="kal")
        op.drop_table("verification_runs", schema="kal")

VERIFY:
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'kal'
      AND table_name IN (
          'verification_runs', 'policy_rules', 'kg_evidence'
      );

Revision ID: kal_005
Revises: kal_004
Create Date: 2026-05-16 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from knowlytix.kal.storage_default.models import SEMANTIC_DIM
from knowlytix.kal.storage_default.verification_models import NLI_LOGICAL_DIM

revision = "kal_005"
down_revision = "kal_004"
branch_labels = None
depends_on = None

SCHEMA = "kal"


def upgrade() -> None:
    # --- kal.verification_runs ---
    # FKs: tenant_id (bare UUID — consuming app owns FK integrity). ``policy_id`` is a free UUID
    # column (no FK) — the package doesn't ship a ``policies`` table,
    # consumers manage policy grouping themselves.
    op.create_table(
        "verification_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("claim", postgresql.JSONB(), nullable=False),
        sa.Column("scores", postgresql.JSONB(), nullable=False),
        sa.Column("verified", sa.Boolean(), nullable=False),
        sa.Column("policy_id", sa.Uuid(), nullable=True),
        sa.Column("violations", postgresql.JSONB(), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column(
            "status",
            sa.String(20),
            server_default=sa.text("'completed'"),
            nullable=False,
        ),
        sa.Column(
            "verification_mode",
            sa.String(20),
            server_default=sa.text("'kg'"),
            nullable=False,
        ),
        sa.Column(
            "aggregate_semantic_embedding",
            postgresql.ARRAY(sa.Float()),
            nullable=True,
            comment=f"vector({SEMANTIC_DIM}) — sentence-encoder claim embedding",
        ),
        sa.Column(
            "aggregate_logical_embedding",
            postgresql.ARRAY(sa.Float()),
            nullable=True,
            comment=f"vector({NLI_LOGICAL_DIM}) — NLI logical-embedding projection",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.verification_runs "
        f"ALTER COLUMN aggregate_semantic_embedding TYPE vector({SEMANTIC_DIM}) "
        f"USING aggregate_semantic_embedding::vector({SEMANTIC_DIM})"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.verification_runs "
        f"ALTER COLUMN aggregate_logical_embedding TYPE vector({NLI_LOGICAL_DIM}) "
        f"USING aggregate_logical_embedding::vector({NLI_LOGICAL_DIM})"
    )
    op.create_index(
        "idx_kal_verification_runs_tenant",
        "verification_runs",
        ["tenant_id"],
        schema=SCHEMA,
    )

    # --- kal.policy_rules ---
    # No FK to a ``policies`` parent — knowly groups rules under
    # policies via a separate join table that the package doesn't
    # ship. ``rfc2119_severity`` is constrained to the canonical
    # MUST / MUST_NOT / SHOULD / MAY set.
    op.create_table(
        "policy_rules",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("relation", sa.String(100), nullable=True),
        sa.Column(
            "energy_penalty",
            sa.Float(),
            server_default=sa.text("1.0"),
            nullable=False,
        ),
        sa.Column(
            "severity",
            sa.String(20),
            server_default=sa.text("'medium'"),
            nullable=False,
        ),
        sa.Column("rfc2119_severity", sa.String(20), nullable=True),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "rfc2119_severity IS NULL OR rfc2119_severity IN "
            "('MUST', 'MUST_NOT', 'SHOULD', 'MAY')",
            name="ck_kal_policy_rules_rfc2119",
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_policy_rules_tenant",
        "policy_rules",
        ["tenant_id"],
        schema=SCHEMA,
    )

    # --- kal.kg_evidence ---
    # The evidence row sits at the intersection of a verification
    # run, a KG triple, and a policy rule. ``triple_id`` and
    # ``policy_rule_id`` are SET NULL on delete so evidence is
    # preserved for audit even after the source rows go away;
    # ``run_id`` is CASCADE — a run without its evidence is broken.
    op.create_table(
        "kg_evidence",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("triple_id", sa.Uuid(), nullable=True),
        sa.Column("policy_rule_id", sa.Uuid(), nullable=True),
        sa.Column("rfc2119_severity", sa.String(20), nullable=True),
        sa.Column("geometric_distance", sa.Float(), nullable=True),
        sa.Column("energy_score", sa.Float(), nullable=True),
        sa.Column("energy_band", sa.String(20), nullable=True),
        sa.Column("decision", sa.String(10), nullable=False),
        sa.Column("source_document_id", sa.Uuid(), nullable=True),
        sa.Column("source_section", sa.String(255), nullable=True),
        sa.Column("aggregate_score", sa.Float(), nullable=True),
        sa.Column("input_hash", sa.String(64), nullable=True),
        sa.Column("output_hash", sa.String(64), nullable=True),
        sa.Column("signature", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            [f"{SCHEMA}.verification_runs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["triple_id"],
            [f"{SCHEMA}.kg_triples.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["policy_rule_id"],
            [f"{SCHEMA}.policy_rules.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_kg_evidence_tenant_run",
        "kg_evidence",
        ["tenant_id", "run_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_kg_evidence_tenant_decision",
        "kg_evidence",
        ["tenant_id", "decision"],
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_kg_evidence_tenant_severity",
        "kg_evidence",
        ["tenant_id", "rfc2119_severity"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    # Drop in reverse FK-dependency order: evidence first (depends on
    # runs + rules), then rules + runs.
    op.drop_index(
        "idx_kal_kg_evidence_tenant_severity",
        table_name="kg_evidence",
        schema=SCHEMA,
    )
    op.drop_index(
        "idx_kal_kg_evidence_tenant_decision",
        table_name="kg_evidence",
        schema=SCHEMA,
    )
    op.drop_index(
        "idx_kal_kg_evidence_tenant_run",
        table_name="kg_evidence",
        schema=SCHEMA,
    )
    op.drop_table("kg_evidence", schema=SCHEMA)

    op.drop_index(
        "idx_kal_policy_rules_tenant",
        table_name="policy_rules",
        schema=SCHEMA,
    )
    op.drop_table("policy_rules", schema=SCHEMA)

    op.drop_index(
        "idx_kal_verification_runs_tenant",
        table_name="verification_runs",
        schema=SCHEMA,
    )
    op.drop_table("verification_runs", schema=SCHEMA)
