"""Create GMS support tables in ``kal`` schema.

Phase B Unit 3c of the v0.3.0 plan
(``docs/plans/plan_knowlytix_kal_v0_3.md``). Adds the four GMS-support
tables peer to the ``kal.kg_*`` triple-store tables migration 047
created: ``kal.enm_entries``, ``kal.phase_ranges``,
``kal.relation_synonyms``, ``kal.stiefel_projections``.

Only ``kal.enm_entries`` is written by ``KalDefaultKGStore.insert_triples``
(for numeric-literal triples — same logic as knowly's
``_maybe_write_enm`` in ``persist_claims_to_kg``); the others ship in
the migration so the package's schema is complete out of the box for
users adopting the structured-ingest / autotuning patterns.

Column shapes mirror knowly's ``ENMEntry`` / ``PhaseRange`` /
``RelationSynonym`` / ``StiefelProjection`` byte-equivalently to keep
federation drift-free. The mirror script (Phase C) ships this
migration into the standalone package as
``kal_004_gms_schema.py``.

ROLLBACK:
    WARNING: drops all GMS support data in the kal schema. The
    ``kal.kg_*`` tables from migration 047 are untouched.

    def downgrade():
        op.drop_table("stiefel_projections", schema="kal")
        op.drop_table("relation_synonyms", schema="kal")
        op.drop_table("phase_ranges", schema="kal")
        op.drop_table("enm_entries", schema="kal")

VERIFY:
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'kal'
      AND table_name IN (
          'enm_entries', 'phase_ranges',
          'relation_synonyms', 'stiefel_projections'
      );

Revision ID: kal_004
Revises: kal_003
Create Date: 2026-05-16 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# Single source of truth for the GMS dim — column shapes match the ORM
# (``KalEnmEntry.semantic_embedding``) so a future dim change in one
# place fails the migration rather than silently drifting.
from knowlytix.kal.storage_default.models import GMS_PROJECTED_DIM

revision = "kal_004"
down_revision = "kal_003"
branch_labels = None
depends_on = None

SCHEMA = "kal"


def upgrade() -> None:
    # --- kal.enm_entries ---
    op.create_table(
        "enm_entries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(512), nullable=False),
        sa.Column(
            "value", sa.Float(precision=53), nullable=False,
        ),
        sa.Column("value_hash", sa.String(64), nullable=False),
        sa.Column(
            "semantic_embedding",
            postgresql.ARRAY(sa.Float()),
            nullable=True,
            comment=f"vector({GMS_PROJECTED_DIM}) — GMS semantic embedding for the numeric value",
        ),
        sa.Column(
            "confidence",
            sa.Float(),
            server_default=sa.text("1.0"),
            nullable=True,
        ),
        sa.Column(
            "last_accessed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.enm_entries "
        f"ALTER COLUMN semantic_embedding TYPE vector({GMS_PROJECTED_DIM}) "
        f"USING semantic_embedding::vector({GMS_PROJECTED_DIM})"
    )
    op.create_index(
        "idx_kal_enm_entries_tenant_key",
        "enm_entries",
        ["tenant_id", "key"],
        schema=SCHEMA,
    )

    # --- kal.phase_ranges ---
    op.create_table(
        "phase_ranges",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("relation_name", sa.String(255), nullable=False),
        sa.Column(
            "v_min", sa.Float(precision=53), nullable=False,
        ),
        sa.Column(
            "v_max", sa.Float(precision=53), nullable=False,
        ),
        sa.Column(
            "sample_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_phase_ranges_tenant_relation",
        "phase_ranges",
        ["tenant_id", "relation_name"],
        schema=SCHEMA,
    )

    # --- kal.relation_synonyms ---
    op.create_table(
        "relation_synonyms",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("canonical_name", sa.String(255), nullable=False),
        sa.Column("synonym", sa.String(255), nullable=False),
        sa.Column(
            "source",
            sa.String(50),
            server_default=sa.text("'auto'"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_relation_synonyms_tenant_synonym",
        "relation_synonyms",
        ["tenant_id", "synonym"],
        schema=SCHEMA,
    )

    # --- kal.stiefel_projections ---
    op.create_table(
        "stiefel_projections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("matrix_name", sa.String(8), nullable=False),
        sa.Column("matrix_data", sa.LargeBinary(), nullable=False),
        sa.Column("m", sa.Integer(), nullable=False),
        sa.Column("d", sa.Integer(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_stiefel_projections_tenant_name",
        "stiefel_projections",
        ["tenant_id", "matrix_name"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    # Drop in reverse order of creation. The four tables are
    # independent of each other (no FKs between them) so order is
    # cosmetic; matching upgrade-order-reversed for symmetry with
    # migration 047's pattern.
    op.drop_index(
        "idx_kal_stiefel_projections_tenant_name",
        table_name="stiefel_projections",
        schema=SCHEMA,
    )
    op.drop_table("stiefel_projections", schema=SCHEMA)

    op.drop_index(
        "idx_kal_relation_synonyms_tenant_synonym",
        table_name="relation_synonyms",
        schema=SCHEMA,
    )
    op.drop_table("relation_synonyms", schema=SCHEMA)

    op.drop_index(
        "idx_kal_phase_ranges_tenant_relation",
        table_name="phase_ranges",
        schema=SCHEMA,
    )
    op.drop_table("phase_ranges", schema=SCHEMA)

    op.drop_index(
        "idx_kal_enm_entries_tenant_key",
        table_name="enm_entries",
        schema=SCHEMA,
    )
    op.drop_table("enm_entries", schema=SCHEMA)
