"""FK ON DELETE RESTRICT on kal.kg_triples → kal.kg_entities / kal.kg_relations.

Phase A step 1 of the v0.4.0 plan
(``docs/plans/plan_knowlytix_kal_v0_4.md``). Mirror of migration 051
for the ``kal`` schema — same rationale, same semantics, applied to
the standalone-package ORM (``KalKgEntity`` / ``KalKgRelation`` /
``KalKgTriple``).

Column names match knowly's ``kg.triples`` exactly (``subject_id``,
``object_id``, ``predicate_id``); only the schema + parent table
names differ. The FK constraint names follow the same
``{table}_{column}_fkey`` convention but with the ``kg_triples``
table prefix on the kal side.

This migration is mirrored as ``kal_007_fk_restrict_kal_kg_triples.py``
per the revision map in ``scripts/MIRROR_MANIFEST.toml``.

ROLLBACK:
    See migration 051's docstring. Same caveats apply — only roll
    back together with the autotuning refactor (Phase B).

    def downgrade():
        _alter_fk("kg_triples", "subject_id", "kg_entities", "CASCADE")
        _alter_fk("kg_triples", "object_id", "kg_entities", "CASCADE")
        _alter_fk("kg_triples", "predicate_id", "kg_relations", "CASCADE")

VERIFY:
    SELECT conname, confdeltype FROM pg_constraint
    WHERE conrelid = 'kal.kg_triples'::regclass AND contype = 'f';
    -- confdeltype 'r' = RESTRICT (was 'c' = CASCADE before)

Revision ID: kal_007
Revises: kal_006
Create Date: 2026-05-17 00:00:00.000000
"""

from alembic import op

revision = "kal_007"
down_revision = "kal_006"
branch_labels = None
depends_on = None

SCHEMA = "kal"


def _alter_fk(
    table: str, column: str, ref_table: str, ondelete: str
) -> None:
    """Drop + re-create a single FK with a new ON DELETE action.

    FK names follow Alembic's auto-naming convention from migration
    047_kal_kg_default_schema (``{table}_{column}_fkey``). If a name
    doesn't match at apply time, the migration errors loudly — recover
    by patching the name here against the live DB's ``pg_constraint``
    catalog.
    """
    fk_name = f"{table}_{column}_fkey"
    op.drop_constraint(fk_name, table, schema=SCHEMA, type_="foreignkey")
    op.create_foreign_key(
        fk_name,
        source_table=table,
        referent_table=ref_table,
        local_cols=[column],
        remote_cols=["id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        ondelete=ondelete,
    )


def upgrade() -> None:
    _alter_fk("kg_triples", "subject_id", "kg_entities", "RESTRICT")
    _alter_fk("kg_triples", "object_id", "kg_entities", "RESTRICT")
    _alter_fk("kg_triples", "predicate_id", "kg_relations", "RESTRICT")


def downgrade() -> None:
    _alter_fk("kg_triples", "subject_id", "kg_entities", "CASCADE")
    _alter_fk("kg_triples", "object_id", "kg_entities", "CASCADE")
    _alter_fk("kg_triples", "predicate_id", "kg_relations", "CASCADE")
