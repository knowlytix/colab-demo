"""KalDefaultKGStore — reference ``KGStore`` impl over the ``kal.*`` schema.

Greenfield-ready default that ``PostgresKnowledgeAdapter`` wires up
when no explicit ``kg_store`` is injected. Pairs with the
``KalKgEntity`` / ``KalKgRelation`` / ``KalKgTriple`` ORM in
``models.py`` and the schema created by migration 047.

Per the Phase B design (see ``docs/plans/plan_knowlytix_kal_v0_3.md``):

- **Single source of truth for ORM → KAL materialization.** Module-level
  ``_entity_to_kal_node`` and ``_row_to_kal_triple`` helpers are the
  only place the KAL Protocol surface meets ``KalKg*`` columns.
- **No transaction ownership.** Methods take ``session: AsyncSession``
  per call; the adapter composes ``async with session.begin():`` from
  the outside. Multiple store calls can therefore share a transaction
  (insert + foreign-ID registration in the same batch).
- **No timeout wrapping.** The adapter's ``_with_timeout`` wraps store
  calls from the outside.
- **Pre-parsed UUID tenants.** ``tenant_id: uuid.UUID`` at the boundary.
- **Minimal-shape vector hits.** ``search_similar_nodes`` returns
  ``KALNode(id, name)`` only — callers ``get_node`` to hydrate the rest
  (see ``KGStore.search_similar_nodes`` docstring for the contract).

``insert_triples`` does dedup + entity/relation resolution + insert,
with the configured ``entity_embedder`` filling ``cayley_embedding``
on every new entity (default: ``HashEntityEmbedder(dim=64)``, no
heavy deps) and the optional ``dual_embedder`` filling
``v_embedding`` / ``u_embedding`` (no default — the package stays
torch-free; users inject a GMS dual-embedder when they need v/u
fills). ``embedding`` (384-d semantic) stays NULL — that column is
owned by the upstream extraction pipeline, not the store.
Existing-entity reuse backfills missing embeddings (cayley OR
dual) when NULL but never overwrites — same semantics as knowly's
``_backfill_embeddings``.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from typing import NamedTuple

import numpy as np
from sqlalchemy import Select, delete, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import NamedFromClause

from knowlytix.kal.storage_default._normalize import normalize_entity_name
from knowlytix.kal.storage_default.embedders import (
    DualEntityEmbedder,
    EntityEmbedder,
    HashEntityEmbedder,
)
from knowlytix.kal.storage_default.models import (
    CAYLEY_DIM,
    GMS_PROJECTED_DIM,
    SCHEMA,
    SEMANTIC_DIM,
    KalEnmEntry,
    KalKgEntity,
    KalKgRelation,
    KalKgTriple,
)
from knowlytix.kal.storage_default.verification_models import KalTripleVerdict
from knowlytix.kal.types import (
    AdjacencyDirection,
    KALLiteral,
    KALNode,
    KALTriple,
    PropertyValue,
    TripleProvenance,
    VerificationMetadata,
)

logger = logging.getLogger(__name__)

# Object-type discriminator stored in ``kal.kg_triples.object_type``.
# Matches knowly's convention (see ``kg_kal_store._OBJECT_TYPE_NUMERIC``)
# so federation between knowly-backed and package-backed deployments
# round-trips literal-object triples identically.
_OBJECT_TYPE_NUMERIC = "numeric"
# Generic literal discriminator for non-numeric literals
# (string/boolean/datetime/anyURI/...). Stamped on the row for operator
# clarity in DB dumps; the load path discriminates literal-vs-entity by
# ``object_literal_value IS NOT NULL`` (so reads handle pre-migration
# rows where the new columns are NULL).
_OBJECT_TYPE_LITERAL = "literal"

# Datatype tag stamped on numeric ``KALLiteral`` materializations when
# the caller didn't supply one. Same value the converters module uses
# on the knowly side; pinned here so the package doesn't import
# ``knowly.kal.converters``.
_NUMERIC_DATATYPE = "xsd:decimal"

# XSD datatype URIs the writer accepts as "this is genuinely numeric"
# (vs. a string that happens to be parseable as float). The ENM write
# path uses this to refuse a ``KALLiteral(value="42",
# datatype="xsd:string")`` — without the gate, every string literal
# parseable by ``float()`` would silently pollute the ENM store.
# Source: xsd-numeric-types per W3C XML Schema Part 2 §3.3.
_NUMERIC_DATATYPES: frozenset[str] = frozenset({
    "xsd:decimal", "xsd:double", "xsd:float", "xsd:integer",
    "xsd:int", "xsd:long", "xsd:short", "xsd:byte",
    "xsd:nonNegativeInteger", "xsd:positiveInteger",
    "xsd:negativeInteger", "xsd:nonPositiveInteger",
    "xsd:unsignedLong", "xsd:unsignedInt",
    "xsd:unsignedShort", "xsd:unsignedByte",
})


class _TripleWithJoins(NamedTuple):
    """A ``KalKgTriple`` + its joined subject/object/relation rows.

    KAL-facing — the JOIN-only intermediate used by
    ``_row_to_kal_triple``. Holds (id, name, type, normalized_name) per
    entity rather than full ORM rows so callers don't accidentally
    mutate detached instances.

    The ``verification_*`` fields carry whichever source the SELECT
    chose to surface — either the ``KalKgTriple`` cache columns
    (flag-off path, default) or a joined row from ``kal.triple_verdicts``
    (flag-on path under ``include_all_verifications=True``). The
    materializer in ``_row_to_kal_triple`` doesn't care which: it
    builds a ``VerificationMetadata`` whenever any field is non-None.
    """

    triple_id: uuid.UUID
    confidence: float | None
    object_type: str | None
    object_numeric: float | None
    object_literal_value: str | None
    object_datatype: str | None
    source_text: str | None
    source: str | None
    extractor: str | None
    subject_id: uuid.UUID
    subject_name: str
    subject_type: str | None
    subject_normalized_name: str | None
    predicate: str
    object_id: uuid.UUID
    object_name: str
    object_type_entity: str | None
    object_normalized_name: str | None
    verification_status: str | None
    verified_at: datetime | None
    verifier: str | None
    scores: dict[str, float] | None


def _triples_with_joins_stmt(
    tenant_id: uuid.UUID,
    *,
    include_all_verifications: bool = False,
) -> tuple[Select[tuple[object, ...]], NamedFromClause, NamedFromClause]:
    """Build the base SELECT + JOIN for ``_TripleWithJoins`` rows.

    Returns ``(stmt, subj_alias, obj_alias)`` so callers can add their
    own WHERE / ORDER BY / LIMIT clauses without re-declaring the
    ~30-line JOIN. Aliases are needed because subject + object both
    point at ``kal.kg_entities``.

    Mirrors knowly's ``kg.repository._triples_with_joins_stmt`` shape —
    the two impls have to materialize the same KAL types from
    structurally-identical row shapes.

    When ``include_all_verifications=True`` (v0.5.0 §5.8 Option C),
    LEFT JOIN ``kal.triple_verdicts`` and SELECT the verdict-row's
    verification columns instead of the triple-row cache. Multi-verifier
    triples expand into N rows; unverified triples stay 1 row with all
    verification columns NULL (LEFT-JOIN padding). The
    ``min_confidence`` filter still pushes down on the cache column
    (parent-level filter; documented in the v0.5.0 plan).
    """
    subj = KalKgEntity.__table__.alias("subj")
    obj = KalKgEntity.__table__.alias("obj")
    if include_all_verifications:
        verdict_table = KalTripleVerdict.__table__
        verification_columns = (
            verdict_table.c.confidence.label("triple_confidence"),
            verdict_table.c.verification_status.label("triple_verification_status"),
            verdict_table.c.verified_at.label("triple_verified_at"),
            verdict_table.c.verifier.label("triple_verifier"),
            verdict_table.c.scores.label("triple_scores"),
        )
    else:
        verification_columns = (
            KalKgTriple.confidence.label("triple_confidence"),
            KalKgTriple.verification_status.label("triple_verification_status"),
            KalKgTriple.verified_at.label("triple_verified_at"),
            KalKgTriple.verifier.label("triple_verifier"),
            KalKgTriple.scores.label("triple_scores"),
        )
    stmt = (
        select(
            KalKgTriple.id.label("triple_id"),
            *verification_columns,
            KalKgTriple.object_type.label("triple_object_type"),
            KalKgTriple.object_numeric.label("triple_object_numeric"),
            KalKgTriple.object_literal_value.label("triple_object_literal_value"),
            KalKgTriple.object_datatype.label("triple_object_datatype"),
            KalKgTriple.source_text.label("triple_source_text"),
            KalKgTriple.source.label("triple_source"),
            KalKgTriple.extractor.label("triple_extractor"),
            subj.c.id.label("subj_id"),
            subj.c.name.label("subj_name"),
            subj.c.type.label("subj_type"),
            subj.c.normalized_name.label("subj_normalized_name"),
            KalKgRelation.name.label("predicate"),
            obj.c.id.label("obj_id"),
            obj.c.name.label("obj_name"),
            obj.c.type.label("obj_type"),
            obj.c.normalized_name.label("obj_normalized_name"),
        )
        .select_from(KalKgTriple.__table__)
        .join(subj, KalKgTriple.subject_id == subj.c.id)
        .join(KalKgRelation, KalKgTriple.predicate_id == KalKgRelation.id)
        .join(obj, KalKgTriple.object_id == obj.c.id)
        .where(
            KalKgTriple.tenant_id == tenant_id,
            subj.c.tenant_id == tenant_id,
            obj.c.tenant_id == tenant_id,
            KalKgRelation.tenant_id == tenant_id,
        )
        .order_by(KalKgTriple.created_at.desc(), KalKgTriple.id)
    )
    if include_all_verifications:
        verdict_table = KalTripleVerdict.__table__
        stmt = stmt.outerjoin(
            verdict_table, KalKgTriple.id == verdict_table.c.triple_id
        )
    return stmt, subj, obj


def _row_to_triple_with_joins(row: object) -> _TripleWithJoins:
    """Materialize one ``_TripleWithJoins`` from a SQLAlchemy row."""
    return _TripleWithJoins(
        triple_id=row.triple_id,  # type: ignore[attr-defined]
        confidence=row.triple_confidence,  # type: ignore[attr-defined]
        object_type=row.triple_object_type,  # type: ignore[attr-defined]
        object_numeric=row.triple_object_numeric,  # type: ignore[attr-defined]
        object_literal_value=row.triple_object_literal_value,  # type: ignore[attr-defined]
        object_datatype=row.triple_object_datatype,  # type: ignore[attr-defined]
        source_text=row.triple_source_text,  # type: ignore[attr-defined]
        source=row.triple_source,  # type: ignore[attr-defined]
        extractor=row.triple_extractor,  # type: ignore[attr-defined]
        subject_id=row.subj_id,  # type: ignore[attr-defined]
        subject_name=row.subj_name,  # type: ignore[attr-defined]
        subject_type=row.subj_type,  # type: ignore[attr-defined]
        subject_normalized_name=row.subj_normalized_name,  # type: ignore[attr-defined]
        predicate=row.predicate,  # type: ignore[attr-defined]
        object_id=row.obj_id,  # type: ignore[attr-defined]
        object_name=row.obj_name,  # type: ignore[attr-defined]
        object_type_entity=row.obj_type,  # type: ignore[attr-defined]
        object_normalized_name=row.obj_normalized_name,  # type: ignore[attr-defined]
        verification_status=row.triple_verification_status,  # type: ignore[attr-defined]
        verified_at=row.triple_verified_at,  # type: ignore[attr-defined]
        verifier=row.triple_verifier,  # type: ignore[attr-defined]
        scores=row.triple_scores,  # type: ignore[attr-defined]
    )


def _compute_skew_parameters(dim: int) -> bytes:
    """Generate a random skew-symmetric matrix W for a new relation.

    Stores the full ``(dim, dim)`` skew-symmetric matrix as bytes,
    compatible with the GMS CayleyRotor ``_W`` format. Random init
    scaled by ``0.01`` keeps the initial rotor close to identity so
    untrained relations don't perturb embeddings before fitting runs.

    Algorithmic port of ``service._compute_skew_parameters`` — same
    shape, dtype, and scale, so a CayleyRotor reading the bytes back
    gets the structure it expects regardless of which store wrote
    them. The matrix VALUES are NOT byte-equivalent across processes
    (``np.random.randn`` uses numpy's global RNG state); federation
    cross-reads work because the rotor reads whichever bytes are
    stored, not because both stores would produce the same matrix.
    """
    raw = np.random.randn(dim, dim).astype(np.float32) * 0.01
    skew = (raw - raw.T) / 2
    return skew.tobytes()  # type: ignore[no-any-return]


def _entity_to_kal_node(entity: KalKgEntity) -> KALNode:
    """Materialize a ``KALNode`` from a ``KalKgEntity`` ORM row.

    Property keys (``confidence`` / ``specificity`` / ``extra_metadata``)
    match the knowly side (``kg_kal_store._entity_to_kal_node``) so
    federation round-trips entities without property drift. Empty values
    are skipped — ``KALNode.properties == {}`` for an entity with no
    extras.
    """
    properties: dict[str, PropertyValue] = {}
    if entity.confidence is not None:
        properties["confidence"] = entity.confidence
    if entity.specificity is not None:
        properties["specificity"] = entity.specificity
    if entity.extra_metadata:
        properties["extra_metadata"] = entity.extra_metadata

    return KALNode(
        id=str(entity.id),
        name=entity.name,
        node_types=[entity.type] if entity.type else [],
        normalized_name=entity.normalized_name,
        properties=properties,
    )


def _row_to_kal_triple(row: _TripleWithJoins) -> KALTriple:
    """Materialize a ``KALTriple`` from a joined row.

    Three branches for the object side:

    1. **Generic literal** (§6.3, Phase D Tier 2 #12) —
       ``object_literal_value IS NOT NULL`` means the inserting client
       supplied a literal; reconstruct ``KALLiteral`` from the stored
       value + datatype. Handles strings / booleans / datetimes /
       anyURI / any xsd:* — the writer in ``insert_triples`` always
       populates both columns for new literal triples.
    2. **Legacy numeric** (pre-§6.3) — rows with
       ``object_type="numeric"`` + ``object_numeric IS NOT NULL`` but
       ``object_literal_value IS NULL`` are pre-migration rows.
       Reconstruct via ``object_numeric`` stringification + the
       hardcoded ``xsd:decimal`` datatype.
    3. **Entity** — anything else materializes as a node-object.

    Store does NOT stamp ``KALTriple.adapter`` — that's the
    federation router's job (KAL spec sec 3.1/3.2).
    """
    subject = KALNode(
        id=str(row.subject_id),
        name=row.subject_name,
        node_types=[row.subject_type] if row.subject_type else [],
        normalized_name=row.subject_normalized_name,
    )

    obj: KALNode | None = None
    object_literal: KALLiteral | None = None
    if row.object_literal_value is not None:
        object_literal = KALLiteral(
            value=row.object_literal_value,
            datatype=row.object_datatype,
        )
    elif row.object_type == _OBJECT_TYPE_NUMERIC and row.object_numeric is not None:
        # Legacy numeric branch — rows that pre-date migration 050.
        object_literal = KALLiteral(
            value=str(row.object_numeric),
            datatype=_NUMERIC_DATATYPE,
        )
    else:
        obj = KALNode(
            id=str(row.object_id),
            name=row.object_name,
            node_types=[row.object_type_entity] if row.object_type_entity else [],
            normalized_name=row.object_normalized_name,
        )

    has_verification = (
        row.confidence is not None
        or row.verification_status is not None
        or row.verified_at is not None
        or row.verifier is not None
        or row.scores is not None
    )
    verification = (
        VerificationMetadata(
            confidence=row.confidence,
            verification_status=row.verification_status,
            verified_at=row.verified_at,
            verifier=row.verifier,
            scores=row.scores,
        )
        if has_verification
        else None
    )
    has_provenance = (
        row.source_text is not None
        or row.source is not None
        or row.extractor is not None
    )
    provenance = (
        TripleProvenance(
            source=row.source,
            extractor=row.extractor,
            source_text=row.source_text,
        )
        if has_provenance
        else None
    )

    return KALTriple(
        id=str(row.triple_id),
        subject=subject,
        predicate=row.predicate,
        object=obj,
        object_literal=object_literal,
        provenance=provenance,
        verification=verification,
        adapter=None,
    )


class KalDefaultKGStore:
    """``KGStore`` impl backed by the ``kal.*`` schema.

    Constructor:

    - ``semantic_dim`` (default 384) — dim of the
      ``kal.kg_entities.embedding`` column; matches knowly's
      sentence-encoder output. The adapter reads ``self.vector_dim``
      at ``search_similar_nodes`` time to validate incoming query
      vectors.
    - ``entity_embedder`` — strategy for filling
      ``cayley_embedding`` on newly-created entities + backfilling
      where NULL on existing-row reuse. Defaults to
      ``HashEntityEmbedder(dim=CAYLEY_DIM)`` (deterministic, no
      heavy deps); callers with richer needs inject a
      semantic-embedding impl. The embedder's ``dim`` must match
      the schema's ``cayley_embedding`` column dim (64-d in the
      stock migration); a mismatch surfaces at insert time as a
      pgvector error.
    - ``dual_embedder`` — optional GMS dual-embedding strategy for
      filling ``v_embedding`` / ``u_embedding``. No default —
      production dual-embedders need ``torch`` + GMS infrastructure
      the package deliberately doesn't drag in. When ``None``, the
      v/u columns stay NULL on new and existing entities (matches
      knowly's behaviour without ``[train]`` extras installed). The
      ``dim`` must match the schema's ``v_embedding`` /
      ``u_embedding`` column dim (128-d in the stock migration).
    """

    def __init__(
        self,
        *,
        semantic_dim: int = SEMANTIC_DIM,
        entity_embedder: EntityEmbedder | None = None,
        dual_embedder: DualEntityEmbedder | None = None,
    ) -> None:
        self._semantic_dim = semantic_dim
        # ``HashEntityEmbedder(dim=CAYLEY_DIM)`` matches the schema
        # default; users who target a different dim must also update
        # the migration. Constructor injection lets a deployment swap
        # in a sentence-transformer-backed embedder without touching
        # the store.
        self._entity_embedder: EntityEmbedder = (
            entity_embedder
            if entity_embedder is not None
            else HashEntityEmbedder(dim=CAYLEY_DIM)
        )
        # No default for the dual embedder — the package stays
        # torch-free. ``None`` means "don't fill v/u". If a deployment
        # opts in, the embedder is responsible for its own lazy model
        # loading; the store catches ``ImportError`` at call time and
        # treats it as "dual unavailable" so a missing extra doesn't
        # break the whole insert path.
        self._dual_embedder: DualEntityEmbedder | None = dual_embedder

    @property
    def vector_dim(self) -> int:
        return self._semantic_dim

    # --- Read operations ---

    async def query_triples(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        *,
        subject_names: list[str] | None,
        predicate_names: list[str] | None,
        object_names: list[str] | None,
        limit: int,
        offset: int,
        min_confidence: float | None,
        verified_only: bool,
        node_id_in_filter: list[uuid.UUID] | None,
        include_all_verifications: bool = False,
        include_revoked: bool = False,
    ) -> list[KALTriple]:
        # v0.5.0 §5.8 Option C: when True, LEFT JOIN against
        # ``kal.triple_verdicts`` so multi-verifier triples expand into
        # N rows. ``min_confidence`` still pushes down on the cache
        # column (most-recent verdict's confidence) — flag-on doesn't
        # change which triples appear, just how many rows each one
        # contributes.
        stmt, subj, obj = _triples_with_joins_stmt(
            tenant_id, include_all_verifications=include_all_verifications
        )
        if not include_revoked:
            stmt = stmt.where(KalKgTriple.revoked_at.is_(None))
        if subject_names is not None:
            stmt = stmt.where(subj.c.name.in_(subject_names))
        if predicate_names is not None:
            stmt = stmt.where(KalKgRelation.name.in_(predicate_names))
        if object_names is not None:
            stmt = stmt.where(obj.c.name.in_(object_names))
        if min_confidence is not None:
            stmt = stmt.where(KalKgTriple.confidence >= min_confidence)
        if verified_only:
            stmt = stmt.where(KalKgTriple.verification_status == "verified")
        if node_id_in_filter is not None:
            if not node_id_in_filter:
                # Empty list — no entity matched the text-search pre-pass;
                # return zero rows rather than emitting ``IN ()`` which
                # some drivers handle oddly.
                return []
            stmt = stmt.where(
                or_(
                    KalKgTriple.subject_id.in_(node_id_in_filter),
                    KalKgTriple.object_id.in_(node_id_in_filter),
                )
            )
        stmt = stmt.limit(limit).offset(offset)
        result = await session.execute(stmt)
        return [
            _row_to_kal_triple(_row_to_triple_with_joins(row))
            for row in result.all()
        ]

    async def text_search_node_ids(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        query: str,
        limit: int,
    ) -> list[uuid.UUID]:
        # ``plainto_tsquery`` (not ``to_tsquery``) so callers don't have
        # to escape operators in user input. The GIN index
        # ``ix_kal_kg_entities_name_fts`` (migration 047) backs the
        # ``@@`` predicate.
        stmt = text(
            f"SELECT id FROM {SCHEMA}.kg_entities "
            "WHERE tenant_id = :tenant_id "
            "AND to_tsvector('english', name) @@ "
            "plainto_tsquery('english', :query) "
            "LIMIT :limit"
        )
        result = await session.execute(
            stmt,
            {"tenant_id": str(tenant_id), "query": query, "limit": limit},
        )
        return [row.id for row in result.all()]

    async def query_adjacent_triples(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        node_id: uuid.UUID,
        *,
        predicate: str | None,
        direction: AdjacencyDirection,
        limit: int,
        include_all_verifications: bool = False,
        include_revoked: bool = False,
    ) -> list[KALTriple]:
        # Same flag semantics as ``query_triples``: both are federation
        # fan-out targets returning ``KALTriple``, so asymmetry would
        # be a footgun (v0.5.0 plan Tier 1 #5a).
        stmt, _subj, _obj = _triples_with_joins_stmt(
            tenant_id, include_all_verifications=include_all_verifications
        )
        if not include_revoked:
            stmt = stmt.where(KalKgTriple.revoked_at.is_(None))
        if direction == "outgoing":
            stmt = stmt.where(KalKgTriple.subject_id == node_id)
        elif direction == "incoming":
            stmt = stmt.where(KalKgTriple.object_id == node_id)
        elif direction == "both":
            stmt = stmt.where(
                or_(
                    KalKgTriple.subject_id == node_id,
                    KalKgTriple.object_id == node_id,
                )
            )
        else:
            # ``AdjacencyDirection`` is a Literal; this branch only fires
            # if the adapter passes through an unchecked string. Mirror
            # knowly's behaviour: surface a ValueError the debug router
            # turns into 400.
            raise ValueError(
                "direction must be 'outgoing', 'incoming', or 'both', "
                f"got {direction!r}"
            )
        if predicate is not None:
            stmt = stmt.where(KalKgRelation.name == predicate)
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return [
            _row_to_kal_triple(_row_to_triple_with_joins(row))
            for row in result.all()
        ]

    async def get_node(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        node_id: uuid.UUID,
    ) -> KALNode | None:
        stmt = select(KalKgEntity).where(
            KalKgEntity.id == node_id,
            KalKgEntity.tenant_id == tenant_id,
        )
        result = await session.execute(stmt)
        entity = result.scalar_one_or_none()
        if entity is None:
            return None
        return _entity_to_kal_node(entity)

    async def search_similar_nodes(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        query_vector: list[float],
        limit: int,
    ) -> list[tuple[KALNode, float]]:
        # Same minimal-node + plain-``float`` contract knowly's impl
        # honours (see ``KGStore.search_similar_nodes`` docstring).
        # ``float(distance)`` coerces pgvector's numpy-float subtype to
        # plain Python so JSON serialization doesn't depend on numpy
        # hooks.
        stmt = text(
            f"SELECT id, name, embedding <=> :query AS distance "
            f"FROM {SCHEMA}.kg_entities "
            "WHERE tenant_id = :tenant_id "
            "ORDER BY distance "
            "LIMIT :limit"
        )
        result = await session.execute(
            stmt,
            {
                "query": str(query_vector),
                "tenant_id": str(tenant_id),
                "limit": limit,
            },
        )
        return [
            (KALNode(id=str(row.id), name=row.name), float(row.distance))
            for row in result.all()
        ]

    # --- Write operations ---

    async def insert_triples(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        triples: list[KALTriple],
    ) -> int:
        """Persist a batch of ``KALTriple``s; return count of newly-created rows.

        Two-layer dedup. Entities + relations resolve by
        ``normalize_entity_name(name)`` / ``name`` via a within-batch
        cache (so two triples referencing the same entity share the
        same row even when both fall through the create branch) plus a
        within-DB lookup (so a re-insert of an existing name doesn't
        spawn a duplicate). Triple-level dedup checks for an existing
        ``(subject_id, predicate_id, object_id)`` row in the tenant
        before inserting; reused triples don't bump the return count.
        Because the entity-resolution step happens first, two input
        triples whose subjects differ only by case (``"Alpha"`` vs
        ``"alpha"``) collapse to one row at the triple-dedup stage.

        Embedding fills on entity creation: ``cayley_embedding``
        comes from the configured ``entity_embedder``;
        ``v_embedding`` / ``u_embedding`` come from the optional
        ``dual_embedder`` when configured (None → those columns
        stay NULL); ``embedding`` (semantic) stays NULL — that one
        is owned by the upstream pipeline, not the store. Reused
        entities backfill any NULL embedding columns; never
        overwritten.
        """
        if not triples:
            return 0

        entity_cache: dict[str, KalKgEntity] = {}
        relation_cache: dict[str, KalKgRelation] = {}
        triples_created = 0

        for triple in triples:
            subj_entity = await self._resolve_entity(
                session, tenant_id, triple.subject, entity_cache,
            )
            relation = await self._resolve_relation(
                session, tenant_id, triple.predicate, relation_cache,
            )
            object_literal_value: str | None = None
            object_datatype: str | None = None
            if triple.object is not None:
                obj_entity = await self._resolve_entity(
                    session, tenant_id, triple.object, entity_cache,
                )
                object_type = "entity"
                object_numeric: float | None = None
            else:
                # Literal-object branch — placeholder entity row matches
                # knowly's convention so the FK stays non-NULL and
                # adjacency queries don't silently drop literal objects.
                assert triple.object_literal is not None  # validated by KALTriple
                literal = triple.object_literal
                placeholder_name = literal.value
                obj_entity = await self._resolve_entity(
                    session,
                    tenant_id,
                    KALNode(id=placeholder_name, name=placeholder_name),
                    entity_cache,
                )
                # §6.3 generic literal coercion: store ANY literal
                # faithfully via the new columns (value + datatype). For
                # numerics, ALSO populate ``object_numeric`` so float
                # precision survives the round-trip and ``_maybe_write_enm``
                # downstream still has the typed value to key on.
                #
                # The "is this numeric?" gate honors the caller's
                # ``datatype``: an xsd:* numeric tag (or ``None``,
                # meaning the caller didn't specify) lets us parse as
                # float; an explicit non-numeric datatype like
                # ``xsd:string`` skips the float attempt so a
                # numeric-looking string doesn't accidentally get
                # ENM-indexed downstream.
                object_literal_value = literal.value
                object_datatype = literal.datatype
                numeric_eligible = (
                    literal.datatype is None
                    or literal.datatype in _NUMERIC_DATATYPES
                )
                if numeric_eligible:
                    try:
                        object_numeric = float(literal.value)
                        object_type = _OBJECT_TYPE_NUMERIC
                    except (TypeError, ValueError):
                        object_numeric = None
                        object_type = _OBJECT_TYPE_LITERAL
                else:
                    object_numeric = None
                    object_type = _OBJECT_TYPE_LITERAL

            # Triple-level dedup: skip if an identical
            # (tenant, subject, predicate, object) row already exists.
            existing_triple = await session.execute(
                select(KalKgTriple.id).where(
                    KalKgTriple.tenant_id == tenant_id,
                    KalKgTriple.subject_id == subj_entity.id,
                    KalKgTriple.predicate_id == relation.id,
                    KalKgTriple.object_id == obj_entity.id,
                )
            )
            if existing_triple.scalar_one_or_none() is not None:
                continue

            confidence = (
                triple.verification.confidence
                if triple.verification is not None
                else None
            )
            source_text = (
                triple.provenance.source_text
                if triple.provenance is not None
                else None
            )
            source = (
                triple.provenance.source
                if triple.provenance is not None
                else None
            )
            extractor = (
                triple.provenance.extractor
                if triple.provenance is not None
                else None
            )

            new_triple = KalKgTriple(
                tenant_id=tenant_id,
                subject_id=subj_entity.id,
                predicate_id=relation.id,
                object_id=obj_entity.id,
                confidence=confidence,
                object_type=object_type,
                object_numeric=object_numeric,
                object_literal_value=object_literal_value,
                object_datatype=object_datatype,
                source_text=source_text,
                source=source,
                extractor=extractor,
                verification_status=(
                    triple.verification.verification_status
                    if triple.verification is not None
                    else None
                ),
                verified_at=(
                    triple.verification.verified_at
                    if triple.verification is not None
                    else None
                ),
                verifier=(
                    triple.verification.verifier
                    if triple.verification is not None
                    else None
                ),
                scores=(
                    triple.verification.scores
                    if triple.verification is not None
                    else None
                ),
            )
            session.add(new_triple)
            await session.flush()
            triples_created += 1

            # Mirror knowly's ``_maybe_write_enm`` (``service.py:400``):
            # numeric-literal triples get an ENM row keyed
            # ``"{normalized(subject)}|{normalized(predicate)}"``.
            # ``object_numeric`` is non-None only on the numeric branch
            # above; this is the same gate knowly uses.
            if (
                object_type == _OBJECT_TYPE_NUMERIC
                and object_numeric is not None
            ):
                enm_key = (
                    f"{normalize_entity_name(triple.subject.name)}"
                    f"|{normalize_entity_name(triple.predicate)}"
                )
                await self._upsert_enm_entry(
                    session, tenant_id, enm_key, object_numeric,
                )

        return triples_created

    async def _upsert_enm_entry(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        key: str,
        value: float,
    ) -> None:
        """Create or update an ENM entry by ``(tenant_id, key)``.

        Byte-equivalent port of
        ``knowly.modules.kg.repository.upsert_enm_entry``:
        ``value_hash`` is SHA-256 over the canonical ``str(value)``,
        existing rows have their ``value`` + ``value_hash`` updated
        in place, missing rows get inserted. No explicit timestamps —
        ``created_at`` is server-default, ``last_accessed_at`` stays
        NULL until an autotuning path bumps it.
        """
        value_hash = hashlib.sha256(str(value).encode()).hexdigest()
        existing = await session.execute(
            select(KalEnmEntry).where(
                KalEnmEntry.tenant_id == tenant_id,
                KalEnmEntry.key == key,
            )
        )
        entry = existing.scalar_one_or_none()
        if entry is not None:
            entry.value = value
            entry.value_hash = value_hash
        else:
            session.add(
                KalEnmEntry(
                    tenant_id=tenant_id,
                    key=key,
                    value=value,
                    value_hash=value_hash,
                )
            )
        await session.flush()

    async def _resolve_entity(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        node: KALNode,
        cache: dict[str, KalKgEntity],
    ) -> KalKgEntity:
        """Cache-aware lookup-or-create for an entity.

        Within-batch cache keyed by normalized_name avoids redundant
        SELECTs and ensures two triples referencing the same entity
        share the same row even if both fall through the "create"
        branch (the second hit reuses the freshly-created row).

        Embedding policy: new entities get ``cayley_embedding`` filled
        from the configured ``entity_embedder``, plus ``v_embedding`` /
        ``u_embedding`` from the optional ``dual_embedder`` when
        configured. Existing-row reuse backfills any NULL embedding
        column (cayley OR dual) — deployments that upgrade their
        embedder converge on coverage over time — but NEVER overwrites
        a non-NULL value; that would silently invalidate stored
        vectors. Mirrors knowly's ``_backfill_embeddings`` semantics
        including the graceful ``ImportError`` fallback when the
        dual-embedder's lazy model load fails.
        """
        normalized = normalize_entity_name(node.name)
        if normalized in cache:
            return cache[normalized]

        existing = await session.execute(
            select(KalKgEntity).where(
                KalKgEntity.normalized_name == normalized,
                KalKgEntity.tenant_id == tenant_id,
            )
        )
        entity = existing.scalar_one_or_none()
        if entity is not None:
            mutated = False
            if entity.cayley_embedding is None:
                entity.cayley_embedding = self._embed(normalized)
                mutated = True
            if entity.v_embedding is None:
                dual = self._embed_dual(normalized)
                if dual is not None:
                    entity.v_embedding, entity.u_embedding = dual
                    mutated = True
            if mutated:
                await session.flush()
            cache[normalized] = entity
            return entity

        dual = self._embed_dual(normalized)
        entity = KalKgEntity(
            tenant_id=tenant_id,
            name=node.name,
            type=node.primary_type,
            normalized_name=normalized,
            cayley_embedding=self._embed(normalized),
            v_embedding=dual[0] if dual is not None else None,
            u_embedding=dual[1] if dual is not None else None,
        )
        session.add(entity)
        await session.flush()
        cache[normalized] = entity
        return entity

    def _embed(self, normalized_name: str) -> list[float]:
        """Run the configured embedder; return a plain ``list[float]``.

        ``EntityEmbedder.embed`` returns a numpy array; convert via
        ``.tolist()`` so the pgvector column adapter gets the type it
        expects and JSON serializers don't pull in numpy hooks.
        """
        return self._entity_embedder.embed(normalized_name).tolist()  # type: ignore[no-any-return]

    def _embed_dual(
        self, normalized_name: str,
    ) -> tuple[list[float], list[float]] | None:
        """Run the configured dual-embedder; return plain Python lists.

        Returns ``None`` when no dual-embedder is configured OR when
        the configured embedder raises ``ImportError`` on its first
        use (lazy model loading that fails). On ``ImportError``, the
        embedder is disabled for this store instance — subsequent
        calls early-return without the failed-load cost or the
        repeated DEBUG-log spam. Mirrors knowly's
        ``_embed_entity_dual`` (``service.py``: clears
        ``_DUAL_EMBEDDER`` after the first ImportError); the
        per-store-instance scope is finer-grained than knowly's
        module-global but functionally equivalent for the common
        one-store-per-process deployment.
        """
        if self._dual_embedder is None:
            return None
        try:
            v, u = self._dual_embedder.embed_dual(normalized_name)
        except ImportError:
            logger.debug(
                "dual-embedder unavailable (lazy ImportError) "
                "— disabling for this store; v/u left NULL for %r",
                normalized_name,
            )
            self._dual_embedder = None
            return None
        return v.tolist(), u.tolist()  # type: ignore[no-any-return]

    async def _resolve_relation(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        name: str,
        cache: dict[str, KalKgRelation],
    ) -> KalKgRelation:
        """Cache-aware lookup-or-create for a relation, keyed by name."""
        if name in cache:
            return cache[name]

        existing = await session.execute(
            select(KalKgRelation).where(
                KalKgRelation.name == name,
                KalKgRelation.tenant_id == tenant_id,
            )
        )
        relation = existing.scalar_one_or_none()
        if relation is not None:
            cache[name] = relation
            return relation

        # New relation: seed ``skew_parameters`` with a random skew-
        # symmetric matrix so the GMS CayleyRotor has a valid initial
        # state. Matches knowly's ``_find_or_create_relation``
        # (``service.py:265``).
        relation = KalKgRelation(
            tenant_id=tenant_id,
            name=name,
            skew_parameters=_compute_skew_parameters(GMS_PROJECTED_DIM),
        )
        session.add(relation)
        await session.flush()
        cache[name] = relation
        return relation

    async def delete_triples(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        triple_ids: list[uuid.UUID],
    ) -> int:
        if not triple_ids:
            return 0
        stmt = delete(KalKgTriple).where(
            KalKgTriple.id.in_(triple_ids),
            KalKgTriple.tenant_id == tenant_id,
        )
        result = await session.execute(stmt)
        rowcount: int = result.rowcount  # type: ignore[attr-defined]
        return rowcount

    async def quarantine_triples_by_source(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        source: str,
    ) -> int:
        stmt = (
            update(KalKgTriple)
            .where(
                KalKgTriple.tenant_id == tenant_id,
                KalKgTriple.source == source,
                KalKgTriple.revoked_at.is_(None),
            )
            .values(revoked_at=func.now())
        )
        result = await session.execute(stmt)
        rowcount: int = result.rowcount  # type: ignore[attr-defined]
        return rowcount

    async def restore_triples_by_source(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        source: str,
    ) -> int:
        stmt = (
            update(KalKgTriple)
            .where(
                KalKgTriple.tenant_id == tenant_id,
                KalKgTriple.source == source,
                KalKgTriple.revoked_at.is_not(None),
            )
            .values(revoked_at=None)
        )
        result = await session.execute(stmt)
        rowcount: int = result.rowcount  # type: ignore[attr-defined]
        return rowcount

    async def delete_node(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        node_id: uuid.UUID,
    ) -> bool:
        """Delete an entity row by ID; return existence flag.

        FK on ``kg_triples.{subject,object}_id`` is RESTRICT from
        migration ``kal_007`` — referencing triples must be removed
        first or the DELETE raises ``IntegrityError``.
        """
        stmt = delete(KalKgEntity).where(
            KalKgEntity.id == node_id,
            KalKgEntity.tenant_id == tenant_id,
        )
        result = await session.execute(stmt)
        rowcount: int = result.rowcount  # type: ignore[attr-defined]
        return rowcount > 0

    async def delete_predicate(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        predicate: str,
    ) -> bool:
        """Delete a relation row by name; return existence flag.

        Matches by exact ``name`` — ``KalKgRelation`` carries no
        ``normalized_name`` column (lookups in ``_resolve_relation``
        also use exact-match), so callers must pass the predicate
        string as it was originally registered.

        FK on ``kg_triples.predicate_id`` is RESTRICT from migration
        ``kal_007`` — referencing triples must be removed first.
        """
        stmt = delete(KalKgRelation).where(
            KalKgRelation.name == predicate,
            KalKgRelation.tenant_id == tenant_id,
        )
        result = await session.execute(stmt)
        rowcount: int = result.rowcount  # type: ignore[attr-defined]
        return rowcount > 0

    async def update_verification(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        triple_id: uuid.UUID,
        verification: VerificationMetadata,
    ) -> bool:
        """Persist a verifier's verdict on a triple; return parent-existence flag.

        v0.5.0 §5.8 — vertical-axis write half:

        - **Verifier present** → UPSERT into ``kal.triple_verdicts`` keyed
          on ``(triple_id, verifier)``. Writing a second verifier's verdict
          on the same triple inserts a new row instead of erasing the
          first; same-verifier rewrite UPDATEs the existing row in place
          (``created_at`` stays, ``updated_at`` advances).
        - **Verifier absent** → no verdict row written; only the triple-row
          cache columns are touched. ``PostgresKnowledgeAdapter`` rejects
          this case at the boundary in v0.5.0, but the store stays naive
          so direct callers (and the legacy ``confidence``-only path used
          in conformance tests) keep working.

        In both paths, the triple-row cache columns
        (``verifier`` / ``confidence`` / ``verification_status`` /
        ``verified_at`` / ``scores``) are updated to reflect the
        most-recently-written verdict across all verifiers — preserving
        the indexed ``min_confidence`` pushdown. Partial-update semantics
        hold for both the verdict row and the cache: only non-None fields
        from ``verification`` are written; existing values on other
        columns are left intact.

        Return value: ``True`` iff the parent triple exists in this
        tenant. FK-miss (or cross-tenant ID) returns ``False`` without
        writing anything, so the verdict-row INSERT can't leak across
        tenants.
        """
        existence = await session.execute(
            select(KalKgTriple.id).where(
                KalKgTriple.id == triple_id,
                KalKgTriple.tenant_id == tenant_id,
            )
        )
        if existence.scalar_one_or_none() is None:
            return False

        cache_values: dict[str, object] = {}
        if verification.confidence is not None:
            cache_values["confidence"] = verification.confidence
        if verification.verification_status is not None:
            cache_values["verification_status"] = verification.verification_status
        if verification.verified_at is not None:
            cache_values["verified_at"] = verification.verified_at
        if verification.verifier is not None:
            cache_values["verifier"] = verification.verifier
        if verification.scores is not None:
            cache_values["scores"] = verification.scores

        if verification.verifier is not None:
            insert_stmt = pg_insert(KalTripleVerdict).values(
                tenant_id=tenant_id,
                triple_id=triple_id,
                verifier=verification.verifier,
                confidence=verification.confidence,
                verification_status=verification.verification_status,
                verified_at=verification.verified_at,
                scores=verification.scores,
            )
            # Same partial-update semantics on the verdict row: only the
            # fields supplied on THIS call get overwritten. ``updated_at``
            # always advances so a downstream "stalest verdict" query has
            # a real signal.
            update_set: dict[str, object] = {"updated_at": func.now()}
            if verification.confidence is not None:
                update_set["confidence"] = insert_stmt.excluded.confidence
            if verification.verification_status is not None:
                update_set["verification_status"] = (
                    insert_stmt.excluded.verification_status
                )
            if verification.verified_at is not None:
                update_set["verified_at"] = insert_stmt.excluded.verified_at
            if verification.scores is not None:
                update_set["scores"] = insert_stmt.excluded.scores
            await session.execute(
                insert_stmt.on_conflict_do_update(
                    constraint="uq_kal_triple_verdict",
                    set_=update_set,
                )
            )

        if cache_values:
            await session.execute(
                update(KalKgTriple)
                .where(
                    KalKgTriple.id == triple_id,
                    KalKgTriple.tenant_id == tenant_id,
                )
                .values(**cache_values)
            )

        return True

    # --- Helpers used during write paths ---

    async def resolve_node_ids_by_normalized_name(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        normalized_names: list[str],
    ) -> dict[str, uuid.UUID]:
        if not normalized_names:
            return {}
        result = await session.execute(
            select(
                KalKgEntity.normalized_name,
                KalKgEntity.id,
            ).where(
                KalKgEntity.normalized_name.in_(normalized_names),
                KalKgEntity.tenant_id == tenant_id,
            )
        )
        return {
            normalized: entity_id
            for normalized, entity_id in result.all()
            if normalized is not None
        }
