"""Add non-numeric literal columns to ``kal.kg_triples`` (§6.3).

Phase D Tier 2 #12 of the v0.3.0 plan
(``docs/plans/plan_knowlytix_kal_v0_3.md``). Adds two additive columns
to ``kal.kg_triples`` so non-numeric literals (strings, booleans,
datetimes, anyURI, etc.) round-trip through the adapter faithfully:

- ``object_literal_value`` (TEXT, nullable) — canonical string form of
  the literal. Populated for ALL new literal-object triples (including
  numerics, alongside ``object_numeric`` for float precision).
- ``object_datatype`` (VARCHAR(64), nullable) — the xsd:* datatype URI
  the inserting client supplied (``xsd:string``, ``xsd:decimal``,
  ``xsd:boolean``, ``xsd:dateTime``, ...). Lets reads reconstruct
  ``KALLiteral.datatype`` losslessly rather than guessing from value
  shape.

Pre-§6.3, non-numeric literals fell through to the entity branch
(``store.py::insert_triples``): a ``KALLiteral(value="hello",
datatype="xsd:string")`` got stored as if it were an entity named
"hello", losing literal-ness on the round-trip. The new columns let
the adapter persist the literal explicitly. Existing rows
(``object_type="numeric"`` with ``object_literal_value IS NULL``) are
read via the legacy branch in ``_row_to_kal_triple`` — backwards-
compatible with no data migration required.

This migration is mirrored as ``kal_006_literal_coercion.py`` per the
revision map in ``scripts/MIRROR_MANIFEST.toml``.

ROLLBACK:
    WARNING: drops the two columns + their data. Any new literal-
    object triples persisted under v0.3.0+ will lose their string
    value + datatype; the rows themselves stay (FK chains unchanged).

    def downgrade():
        op.drop_column("kg_triples", "object_datatype", schema="kal")
        op.drop_column("kg_triples", "object_literal_value", schema="kal")

VERIFY:
    SELECT column_name FROM information_schema.columns
    WHERE table_schema = 'kal' AND table_name = 'kg_triples'
      AND column_name IN ('object_literal_value', 'object_datatype');

Revision ID: kal_006
Revises: kal_005
Create Date: 2026-05-16 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "kal_006"
down_revision = "kal_005"
branch_labels = None
depends_on = None

SCHEMA = "kal"


def upgrade() -> None:
    op.add_column(
        "kg_triples",
        sa.Column(
            "object_literal_value",
            sa.Text(),
            nullable=True,
            comment=(
                "Canonical string form of a literal object. For numerics "
                "still also stored in ``object_numeric`` for precision."
            ),
        ),
        schema=SCHEMA,
    )
    op.add_column(
        "kg_triples",
        sa.Column(
            "object_datatype",
            sa.String(64),
            nullable=True,
            comment="xsd:* datatype URI for the literal object.",
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("kg_triples", "object_datatype", schema=SCHEMA)
    op.drop_column("kg_triples", "object_literal_value", schema=SCHEMA)
