"""Create kal schema + kal.kg_entities / kg_relations / kg_triples.

Phase B Unit 1 of the v0.3.0 plan
(``docs/plans/plan_knowlytix_kal_v0_3.md``). Creates the schema that
``KalDefaultKGStore`` writes to — the greenfield-reference impl of the
``KGStore`` Protocol. Knowly's existing ``kg.*`` tables are NOT
touched; ``KnowlyKGStore`` continues to operate against them
unchanged. The mirror script (Phase C) ships this migration into the
standalone package as ``kal_003_kg_default_schema.py``.

Shape: full column port of knowly's ``kg.entities`` / ``kg.relations``
/ ``kg.triples`` minus the knowly-internal governance / actor /
versioning / extraction-pipeline columns (see
``src/knowly/kal/storage_default/models.py`` for the rationale).
GMS embedding columns (``cayley_embedding`` / ``v_embedding`` /
``u_embedding`` on entities; ``skew_parameters`` / ``rotor_rank`` /
``fitted`` / ``fitted_at`` on relations) ARE preserved so the
default store can offer full ``persist_claims_to_kg``-equivalent
functionality when the GMS embedder lands (Phase B Unit 3).

ROLLBACK:
    WARNING: drops all data in ``kal.kg_entities`` /
    ``kal.kg_relations`` / ``kal.kg_triples`` and the ``kal`` schema
    itself if it has no other tables left. ``KnowlyKGStore``-backed
    deployments are unaffected.

    def downgrade():
        op.drop_index("idx_kal_kg_triples_object", ...)
        ... (reverse order: triples → relations → entities)
        op.execute("DROP SCHEMA IF EXISTS kal CASCADE")

VERIFY:
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'kal'
      AND table_name IN ('kg_entities', 'kg_relations', 'kg_triples');

Revision ID: kal_003
Revises: kal_002
Create Date: 2026-05-16 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# Single source of truth for dim constants — column shapes match the
# ORM exactly, so a future dim change in one place fails the migration
# rather than silently drifting (same idiom as
# ``046_kal_connections.py``).
from knowlytix.kal.storage_default.models import (
    CAYLEY_DIM,
    GMS_PROJECTED_DIM,
    SEMANTIC_DIM,
)

revision = "kal_003"
down_revision = "kal_002"
branch_labels = None
depends_on = None

SCHEMA = "kal"


def upgrade() -> None:
    # Required Postgres extensions. Both are no-ops in the knowly
    # database (migration 001 already enables them), but on the
    # mirrored KAL side this is the first migration that uses
    # ``vector(N)`` columns + ``gin_trgm_ops`` indexes — without these
    # ``CREATE EXTENSION`` calls a standalone KAL install would fail
    # on the first index emission, forcing the user to enable the
    # extensions out-of-band.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # The ``kal`` schema is new to knowly's database with this
    # migration. ``IF NOT EXISTS`` is defensive — re-running the
    # migration in a partial-state environment shouldn't error.
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

    # --- kal.kg_entities ---
    op.create_table(
        "kg_entities",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("type", sa.String(100), nullable=True),
        sa.Column("normalized_name", sa.String(500), nullable=True),
        sa.Column(
            "embedding",
            postgresql.ARRAY(sa.Float()),
            nullable=True,
            comment=f"vector({SEMANTIC_DIM}) — semantic sentence-encoder embedding",
        ),
        sa.Column(
            "cayley_embedding",
            postgresql.ARRAY(sa.Float()),
            nullable=True,
            comment=f"vector({CAYLEY_DIM}) — Cayley-LAG hypersphere embedding",
        ),
        sa.Column(
            "v_embedding",
            postgresql.ARRAY(sa.Float()),
            nullable=True,
            comment=f"vector({GMS_PROJECTED_DIM}) — GMS dual embedding (V)",
        ),
        sa.Column(
            "u_embedding",
            postgresql.ARRAY(sa.Float()),
            nullable=True,
            comment=f"vector({GMS_PROJECTED_DIM}) — GMS dual embedding (U)",
        ),
        sa.Column("specificity", sa.Float(), nullable=True),
        sa.Column(
            "confidence",
            sa.Float(),
            server_default=sa.text("1.0"),
            nullable=True,
        ),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
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
            "confidence >= 0.0 AND confidence <= 1.0",
            name="kal_kg_entities_confidence_check",
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )

    # Swap ARRAY columns for real pgvector columns. The ARRAY decl
    # gets us through SQLAlchemy DDL emission; the ALTER promotes
    # to ``vector(N)`` so pgvector indexes can use them. Same pattern
    # as ``009_knowledge_graph_tables.py``.
    op.execute(
        f"ALTER TABLE {SCHEMA}.kg_entities "
        f"ALTER COLUMN embedding TYPE vector({SEMANTIC_DIM}) "
        f"USING embedding::vector({SEMANTIC_DIM})"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.kg_entities "
        f"ALTER COLUMN cayley_embedding TYPE vector({CAYLEY_DIM}) "
        f"USING cayley_embedding::vector({CAYLEY_DIM})"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.kg_entities "
        f"ALTER COLUMN v_embedding TYPE vector({GMS_PROJECTED_DIM}) "
        f"USING v_embedding::vector({GMS_PROJECTED_DIM})"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.kg_entities "
        f"ALTER COLUMN u_embedding TYPE vector({GMS_PROJECTED_DIM}) "
        f"USING u_embedding::vector({GMS_PROJECTED_DIM})"
    )

    op.create_index(
        "idx_kal_kg_entities_tenant",
        "kg_entities",
        ["tenant_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_kg_entities_type",
        "kg_entities",
        ["type"],
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_kg_entities_normalized",
        "kg_entities",
        ["normalized_name"],
        schema=SCHEMA,
    )
    # FTS GIN on ``name`` — powers ``KGStore.text_search_node_ids``
    # the same way knowly's 044 migration powers ``kg.entities``.
    op.execute(
        f"CREATE INDEX ix_kal_kg_entities_name_fts "
        f"ON {SCHEMA}.kg_entities "
        f"USING gin(to_tsvector('english', name))"
    )
    # Trigram index for fuzzy name search (mirrors 009 on kg.entities).
    op.execute(
        f"CREATE INDEX idx_kal_kg_entities_name_trgm "
        f"ON {SCHEMA}.kg_entities USING gin(name gin_trgm_ops)"
    )
    # IVFFlat cosine indexes for vector similarity search. ``lists=100``
    # is the same starting value 009 uses; tune up via REINDEX once data
    # lands. Each embedding column gets its own index — sentinels never
    # search across embedding spaces.
    op.execute(
        f"CREATE INDEX idx_kal_kg_entities_embedding "
        f"ON {SCHEMA}.kg_entities USING ivfflat "
        f"(embedding vector_cosine_ops) WITH (lists = 100)"
    )
    op.execute(
        f"CREATE INDEX idx_kal_kg_entities_cayley "
        f"ON {SCHEMA}.kg_entities USING ivfflat "
        f"(cayley_embedding vector_cosine_ops) WITH (lists = 100)"
    )

    # --- kal.kg_relations ---
    op.create_table(
        "kg_relations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("skew_parameters", sa.LargeBinary(), nullable=True),
        sa.Column("rotor_rank", sa.Integer(), nullable=True),
        sa.Column(
            "fitted",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "fitted_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
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

    op.create_index(
        "idx_kal_kg_relations_tenant",
        "kg_relations",
        ["tenant_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_kg_relations_name",
        "kg_relations",
        ["name"],
        schema=SCHEMA,
    )

    # --- kal.kg_triples ---
    op.create_table(
        "kg_triples",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("subject_id", sa.Uuid(), nullable=False),
        sa.Column("predicate_id", sa.Uuid(), nullable=False),
        sa.Column("object_id", sa.Uuid(), nullable=False),
        sa.Column(
            "confidence",
            sa.Float(),
            server_default=sa.text("1.0"),
            nullable=True,
        ),
        sa.Column("object_type", sa.String(16), nullable=True),
        sa.Column("object_numeric", sa.Float(precision=53), nullable=True),
        sa.Column("source_text", sa.Text(), nullable=True),
        sa.Column("verification_status", sa.String(32), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verifier", sa.String(128), nullable=True),
        sa.Column("scores", postgresql.JSONB(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["subject_id"],
            [f"{SCHEMA}.kg_entities.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["predicate_id"],
            [f"{SCHEMA}.kg_relations.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["object_id"],
            [f"{SCHEMA}.kg_entities.id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="kal_kg_triples_confidence_check",
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )

    op.create_index(
        "idx_kal_kg_triples_tenant",
        "kg_triples",
        ["tenant_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_kg_triples_subject",
        "kg_triples",
        ["subject_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_kg_triples_predicate",
        "kg_triples",
        ["predicate_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "idx_kal_kg_triples_object",
        "kg_triples",
        ["object_id"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    # Drop in reverse dependency order: triples → relations → entities.
    # Each drop_index call is explicit (mirrors 009) so a partial
    # downgrade-state can be debugged from the alembic log.
    op.drop_index(
        "idx_kal_kg_triples_object", table_name="kg_triples", schema=SCHEMA,
    )
    op.drop_index(
        "idx_kal_kg_triples_predicate", table_name="kg_triples", schema=SCHEMA,
    )
    op.drop_index(
        "idx_kal_kg_triples_subject", table_name="kg_triples", schema=SCHEMA,
    )
    op.drop_index(
        "idx_kal_kg_triples_tenant", table_name="kg_triples", schema=SCHEMA,
    )
    op.drop_table("kg_triples", schema=SCHEMA)

    op.drop_index(
        "idx_kal_kg_relations_name",
        table_name="kg_relations",
        schema=SCHEMA,
    )
    op.drop_index(
        "idx_kal_kg_relations_tenant",
        table_name="kg_relations",
        schema=SCHEMA,
    )
    op.drop_table("kg_relations", schema=SCHEMA)

    op.execute(f"DROP INDEX IF EXISTS {SCHEMA}.idx_kal_kg_entities_cayley")
    op.execute(f"DROP INDEX IF EXISTS {SCHEMA}.idx_kal_kg_entities_embedding")
    op.execute(f"DROP INDEX IF EXISTS {SCHEMA}.idx_kal_kg_entities_name_trgm")
    op.execute(f"DROP INDEX IF EXISTS {SCHEMA}.ix_kal_kg_entities_name_fts")
    op.drop_index(
        "idx_kal_kg_entities_normalized",
        table_name="kg_entities",
        schema=SCHEMA,
    )
    op.drop_index(
        "idx_kal_kg_entities_type", table_name="kg_entities", schema=SCHEMA,
    )
    op.drop_index(
        "idx_kal_kg_entities_tenant",
        table_name="kg_entities",
        schema=SCHEMA,
    )
    op.drop_table("kg_entities", schema=SCHEMA)

    # Drop the schema if it's empty — leaves other ``kal.*`` tables
    # untouched in case a future migration adds more tables to the
    # schema (Unit 3 / Unit 4 will).
    op.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
