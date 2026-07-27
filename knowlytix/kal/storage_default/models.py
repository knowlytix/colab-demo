"""ORM models for the KAL default storage backend (``kal`` schema).

Mirrors knowly's ``kg.{entities, relations, triples}`` columns that are
relevant to the KAL Protocol — same shapes, same constraints, different
schema. Knowly-internal columns dropped (per Phase B scope decision in
``docs/plans/plan_knowlytix_kal_v0_3.md``):

- ``policy_id`` / ``source_run_id`` — knowly governance + verification-
  run attribution; package users don't have those tables.
- ``created_by`` / ``updated_by`` / ``actor_type`` — knowly auth/actor
  model.
- ``version`` / ``merged_into`` — knowly version history.
- ``confidence_logit`` — redundant with ``confidence`` (logit transform).
- ``is_structured`` / ``is_enriched`` / ``derivation_rule`` /
  ``extraction_backend`` / ``object_numeric_hash`` (on triples) —
  knowly extraction-pipeline internals.

GMS embedding columns are PRESERVED (``cayley_embedding`` /
``v_embedding`` / ``u_embedding`` on entities; ``skew_parameters`` /
``rotor_rank`` / ``fitted`` / ``fitted_at`` on relations) so the
default store can offer full ``persist_claims_to_kg``-equivalent
functionality when the GMS embedder is wired in (Phase B Unit 3).

Embedding dims live as module-level constants so the migration and ORM
share a single source of truth.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from knowlytix.kal._db import KalBase as Base
from knowlytix.kal._db import TimestampMixin, UUIDPrimaryKeyMixin

# Schema name for all KAL default-store tables. Mirror script swaps
# nothing here — same schema in both knowly and the standalone package.
SCHEMA = "kal"

# Package form: ``tenant_id`` is a bare UUID column with NO FK.
# Knowly's form scopes tenant referential integrity via FK to
# ``knowly.tenants(id)``; the standalone package doesn't know what a
# "tenant" means in the consuming application. Consumers that want
# referential integrity add their own FK in an app-side migration.
# Matches the pattern v0.2.0 applied to ``connections.models`` and
# ``external_id_registry.models``.

# Embedding dimensions. The semantic embedding (384) matches knowly's
# sentence-transformer encoder dim; cayley (64) is the Cayley-LAG
# hypersphere embedding; v/u (128 each) are the GMS dual embeddings
# (Stiefel-projected from cayley). Mirror knowly's
# ``modules.kg.models`` constants intentionally so the migration
# diffs cleanly across both repos.
SEMANTIC_DIM = 384
CAYLEY_DIM = 64
GMS_PROJECTED_DIM = 128


class KalKgEntity(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Knowledge graph entity in the ``kal`` schema.

    KAL Protocol surface: ``id`` + ``name`` + ``type`` +
    ``normalized_name`` populate ``KALNode``; ``embedding`` powers
    ``search_similar_nodes``; ``properties`` (per-row JSONB) carries
    adapter-specific extras (``confidence``, ``specificity``,
    ``extra_metadata`` per ``_entity_to_kal_node``).
    """

    __tablename__ = "kg_entities"
    __table_args__ = (
        Index("idx_kal_kg_entities_tenant", "tenant_id"),
        Index("idx_kal_kg_entities_type", "type"),
        Index("idx_kal_kg_entities_normalized", "normalized_name"),
        {"schema": SCHEMA},
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
    )
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    normalized_name: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )
    embedding: Mapped[list[float] | None] = mapped_column(  # type: ignore[type-arg]
        Vector(SEMANTIC_DIM),
        nullable=True,
    )
    cayley_embedding: Mapped[list[float] | None] = mapped_column(  # type: ignore[type-arg]
        Vector(CAYLEY_DIM),
        nullable=True,
    )
    v_embedding: Mapped[list[float] | None] = mapped_column(  # type: ignore[type-arg]
        Vector(GMS_PROJECTED_DIM),
        nullable=True,
    )
    u_embedding: Mapped[list[float] | None] = mapped_column(  # type: ignore[type-arg]
        Vector(GMS_PROJECTED_DIM),
        nullable=True,
    )
    specificity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    confidence: Mapped[float | None] = mapped_column(
        Float,
        default=1.0,
        nullable=True,
    )
    # JSONB carrier for adapter-specific properties — kept as
    # ``"metadata"`` column name to match knowly's schema convention and
    # the existing ``_entity_to_kal_node`` helper's ``extra_metadata``
    # field key. Python attribute is ``extra_metadata`` to avoid
    # shadowing SQLAlchemy's ``metadata`` attribute on declarative bases.
    extra_metadata: Mapped[dict[str, str] | None] = mapped_column(  # type: ignore[type-arg]
        "metadata",
        JSONB,
        nullable=True,
        default=dict,
    )


class KalKgRelation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Knowledge graph relation type in the ``kal`` schema.

    Carries Cayley-LAG rotation parameters (``skew_parameters`` /
    ``rotor_rank``) used by GMS dual-embedding inference. ``fitted`` +
    ``fitted_at`` track whether the Cayley fit has run for this
    relation — the fitter writes them lazily on first access.
    """

    __tablename__ = "kg_relations"
    __table_args__ = (
        Index("idx_kal_kg_relations_tenant", "tenant_id"),
        Index("idx_kal_kg_relations_name", "name"),
        {"schema": SCHEMA},
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    skew_parameters: Mapped[bytes | None] = mapped_column(
        LargeBinary,
        nullable=True,
    )
    rotor_rank: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    fitted: Mapped[bool] = mapped_column(
        Boolean,
        server_default="false",
        nullable=False,
    )
    fitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    extra_metadata: Mapped[dict[str, str] | None] = mapped_column(  # type: ignore[type-arg]
        "metadata",
        JSONB,
        nullable=True,
        default=dict,
    )


class KalKgTriple(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Subject-predicate-object triple in the ``kal`` schema.

    Verification metadata columns (``verification_status`` /
    ``verified_at`` / ``verifier`` / ``scores``) match knowly's KAL
    Story 5 typed columns so ``include_unverified`` filtering and
    per-engine score lookups use real indexes instead of JSONB scans
    (see ``_row_to_kal_triple`` + ``KGStore.update_verification``).

    ``object_type`` + ``object_numeric``: literal-object encoding per
    the Test Conventions literal-encoding note. ``object_type =
    "numeric"`` puts the value in ``object_numeric`` (Float64) and
    leaves ``object_id`` pointing at a placeholder entity so the FK
    stays non-NULL.
    """

    __tablename__ = "kg_triples"
    __table_args__ = (
        Index("idx_kal_kg_triples_tenant", "tenant_id"),
        Index("idx_kal_kg_triples_subject", "subject_id"),
        Index("idx_kal_kg_triples_predicate", "predicate_id"),
        Index("idx_kal_kg_triples_object", "object_id"),
        # v0.10.0 — revoke-by-source key for MCP-ingested triples.
        Index("idx_kal_kg_triples_source", "source"),
        {"schema": SCHEMA},
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
    )
    subject_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SCHEMA}.kg_entities.id", ondelete="CASCADE"),
    )
    predicate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SCHEMA}.kg_relations.id", ondelete="CASCADE"),
    )
    object_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SCHEMA}.kg_entities.id", ondelete="CASCADE"),
    )
    confidence: Mapped[float | None] = mapped_column(
        Float,
        default=1.0,
        nullable=True,
    )
    object_type: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
    )
    object_numeric: Mapped[float | None] = mapped_column(
        Float(precision=53),
        nullable=True,
    )
    # §6.3 non-numeric literal coercion (Phase D Tier 2 #12, migration
    # 050 / kal_006). Pre-§6.3, non-numeric literals fell through to the
    # entity branch in ``store.py::insert_triples`` and lost their
    # literal shape on the round-trip. These columns let the adapter
    # persist any KALLiteral faithfully — value + xsd:* datatype URI.
    # ``object_numeric`` is still populated alongside for numeric
    # literals so float precision survives the round-trip; rows older
    # than this migration leave both new columns NULL and read via the
    # legacy ``object_numeric``-only branch in ``_row_to_kal_triple``.
    object_literal_value: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    object_datatype: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    source_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # v0.10.0 §Q1(B) — provenance origin for ingested triples.
    # ``source`` is the indexed revoke key (e.g. ``mcp://<connection>``);
    # ``extractor`` records which extractor/tool produced the triple
    # (e.g. ``mcp:resource:<uri>``). Both nullable.
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    extractor: Mapped[str | None] = mapped_column(Text, nullable=True)
    # v0.11.0 — quarantine timestamp. NULL = live; non-NULL = revoked at
    # that time (revoke-by-source). Read paths exclude non-NULL rows
    # unless include_revoked is set; reversible via restore.
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # KAL Story 5 typed verification columns — indexed lookup beats
    # JSONB scans on ``include_unverified`` / per-engine filters.
    #
    # v0.5.0 §5.8 update: these columns are now the **denormalised
    # cache** of the most-recently-written row in
    # ``kal.triple_verdicts`` (across all verifiers).
    # ``KalTripleVerdict`` is the authoritative per-(triple, verifier)
    # record; ``KalDefaultKGStore.update_verification`` UPSERTs there
    # and keeps these cache columns in sync inside the same
    # transaction. ``KALQuery.min_confidence`` uses the cache
    # (indexed pushdown); callers wanting "any-verifier above
    # threshold" semantics must set
    # ``KALQuery.include_all_verifications=True`` and filter in app
    # code.
    verification_status: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    verifier: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    scores: Mapped[dict[str, float] | None] = mapped_column(
        JSONB,
        nullable=True,
    )


# ---------------------------------------------------------------------------
# GMS support tables — peer to the KG triple-store tables above. Used by
# ``persist_claims_to_kg``-equivalent flows in ``KalDefaultKGStore`` and by
# the separate ``ingest_structured_data`` path. ``KalEnmEntry`` is written
# during ``insert_triples`` for numeric-literal triples; the others
# (``KalPhaseRange`` / ``KalRelationSynonym`` / ``KalStiefelProjection``)
# are written by structured-ingest / autotuning paths and ship now so
# the package's schema is complete out of the box. Column shapes mirror
# knowly's ``ENMEntry`` / ``PhaseRange`` / ``RelationSynonym`` /
# ``StiefelProjection`` byte-equivalently so federation doesn't drift.
# ---------------------------------------------------------------------------


class KalEnmEntry(UUIDPrimaryKeyMixin, Base):
    """Exact Numerical Memory entry — lossless numeric storage.

    Written by ``KalDefaultKGStore.insert_triples`` for every numeric-
    literal triple: key is ``"{normalized(subject)}|{normalized(predicate)}"``;
    value is the raw float64 numeric (precision=53 matches IEEE-754
    double). ``value_hash`` is a SHA-256 over the canonical string
    representation — gives byte-equal identity across recomputes for
    federation/dedup.
    """

    __tablename__ = "enm_entries"
    __table_args__ = ({"schema": SCHEMA},)

    tenant_id: Mapped[uuid.UUID] = mapped_column(
    )
    key: Mapped[str] = mapped_column(String(512), nullable=False)
    value: Mapped[float] = mapped_column(Float(precision=53), nullable=False)
    value_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    semantic_embedding: Mapped[list[float] | None] = mapped_column(  # type: ignore[type-arg]
        Vector(GMS_PROJECTED_DIM),
        nullable=True,
    )
    confidence: Mapped[float | None] = mapped_column(
        Float,
        server_default="1.0",
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
    )


class KalPhaseRange(UUIDPrimaryKeyMixin, Base):
    """Phase encoder range for a relation's numeric values.

    Not written by ``insert_triples`` directly; populated by the
    separate ``ingest_structured_data`` path on the knowly side. The
    table ships in the migration so the package's schema is complete
    for users that adopt that pattern themselves (e.g. via their own
    autotuning loop).
    """

    __tablename__ = "phase_ranges"
    __table_args__ = ({"schema": SCHEMA},)

    tenant_id: Mapped[uuid.UUID] = mapped_column(
    )
    relation_name: Mapped[str] = mapped_column(String(255), nullable=False)
    v_min: Mapped[float] = mapped_column(Float(precision=53), nullable=False)
    v_max: Mapped[float] = mapped_column(Float(precision=53), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, server_default="0")


class KalRelationSynonym(UUIDPrimaryKeyMixin, Base):
    """Maps alternative relation names to canonical KG relation names.

    Same not-written-by-``insert_triples`` story as
    ``KalPhaseRange``: ships in the schema for completeness; populated
    by structured-ingest / autotuning paths.
    """

    __tablename__ = "relation_synonyms"
    __table_args__ = ({"schema": SCHEMA},)

    tenant_id: Mapped[uuid.UUID] = mapped_column(
    )
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    synonym: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(
        String(50), server_default="auto", nullable=False,
    )


class KalStiefelProjection(UUIDPrimaryKeyMixin, Base):
    """Stiefel projection matrix P_v or P_u for GMS DualEmbedding.

    Holds the projection matrix bytes used by the GMS dual-embedder
    to map sentence-encoder outputs into the v/u embedding spaces.
    Not written by ``insert_triples`` — the matrix is fit offline
    (typically by an autotuning loop) and persisted via a separate
    update path.
    """

    __tablename__ = "stiefel_projections"
    __table_args__ = ({"schema": SCHEMA},)

    tenant_id: Mapped[uuid.UUID] = mapped_column(
    )
    matrix_name: Mapped[str] = mapped_column(String(8), nullable=False)
    matrix_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    m: Mapped[int] = mapped_column(Integer, nullable=False)
    d: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
    )
