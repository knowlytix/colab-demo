"""ORM models for the KAL verification chain in the ``kal`` schema.

Ports knowly's ``VerificationRun`` (``modules/verification/models.py``),
``PolicyRule`` (``modules/policy/models.py``), and
``VerificationEvidence`` (``modules/verification/models.py``)
byte-equivalently into ``kal.*`` so the package ships a complete
verification-chain schema. Knowly-side these live across two schemas
(``knowly.*`` for runs + rules, ``kg.*`` for evidence) and across two
modules; the package collapses them into one ``kal.*`` neighborhood
because their lifecycles are intertwined and a downstream consumer
wiring up its own verification pipeline shouldn't have to chase the
split.

**Not populated by ``KalDefaultKGStore.insert_triples``.** These
tables are written by the verification pipeline that consumes the KG
— a package consumer wires their own verification flow to call
into them. The schema ships so the FKs from ``KalKgEvidence`` can
resolve and so a future ``KalVerificationStore`` Protocol (out of
scope for v0.3.0) has its home ready.

Column shapes mirror knowly's exactly so federation cross-reads
work: a run started by knowly + an evidence row written by the
package-side flow round-trip without column drift. Includes the
NLI-specific shapes (``aggregate_logical_embedding`` 768-d, the
``rfc2119_severity`` check constraint on ``PolicyRule``) — those
are concrete commitments the v0.3.0 schema makes; a more
verifier-agnostic shape would require a redesign that we're
deferring to a later release.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from knowlytix.kal._db import KalBase as Base
from knowlytix.kal._db import TimestampMixin, UUIDPrimaryKeyMixin
from knowlytix.kal.storage_default.models import (
    SCHEMA,
    SEMANTIC_DIM,
)

# Dim for the NLI logical-embedding column on
# ``KalVerificationRun.aggregate_logical_embedding`` and the
# corresponding column on ``KalKgEvidence``. Hardcoded today to
# match knowly's 768-d shape; mirrored as a constant so the migration
# + ORM share a single source of truth (same idiom as
# ``SEMANTIC_DIM`` / ``CAYLEY_DIM`` / ``GMS_PROJECTED_DIM`` in
# ``models.py``).
NLI_LOGICAL_DIM = 768


class KalVerificationRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A single verification run record.

    Tracks claim + scores + the verification decision. ``policy_id``
    is a free UUID column (no FK) so a run can reference a policy
    that lives in a different schema (the package doesn't ship a
    ``policies`` table — consumers manage their policy-grouping
    layer separately). The two aggregate embeddings are the
    NLI-shaped projections used by paper §6 verification geometry.
    """

    __tablename__ = "verification_runs"
    __table_args__ = ({"schema": SCHEMA},)

    tenant_id: Mapped[uuid.UUID] = mapped_column(
    )
    claim: Mapped[dict[str, str]] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=False,
    )
    scores: Mapped[dict[str, float]] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=False,
    )
    verified: Mapped[bool] = mapped_column(nullable=False)
    # Free UUID (no FK) — package doesn't ship ``policies`` table.
    policy_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    violations: Mapped[dict[str, str] | None] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=True,
    )
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        server_default=text("'completed'"),
    )
    verification_mode: Mapped[str] = mapped_column(
        String(20),
        server_default=text("'kg'"),
    )
    aggregate_semantic_embedding: Mapped[list[float] | None] = mapped_column(  # type: ignore[type-arg]
        Vector(SEMANTIC_DIM),
        nullable=True,
    )
    aggregate_logical_embedding: Mapped[list[float] | None] = mapped_column(  # type: ignore[type-arg]
        Vector(NLI_LOGICAL_DIM),
        nullable=True,
    )


class KalPolicyRule(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A single compliance rule.

    Standalone — no FK to a ``policies`` parent (knowly groups rules
    under policies via a separate ``policy_rule_assignments`` join
    table that the package doesn't ship for v0.3.0). Consumers that
    want policy grouping run their own assignment layer.

    ``rfc2119_severity`` is constrained to the canonical MUST /
    MUST_NOT / SHOULD / MAY set — same check constraint knowly's
    ``kg.policy_rules`` carries.
    """

    __tablename__ = "policy_rules"
    __table_args__ = (
        CheckConstraint(
            "rfc2119_severity IS NULL OR rfc2119_severity IN "
            "('MUST', 'MUST_NOT', 'SHOULD', 'MAY')",
            name="ck_kal_policy_rules_rfc2119",
        ),
        {"schema": SCHEMA},
    )

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    relation: Mapped[str | None] = mapped_column(String(100), nullable=True)
    energy_penalty: Mapped[float] = mapped_column(
        Float,
        server_default=text("1.0"),
    )
    severity: Mapped[str] = mapped_column(
        String(20),
        server_default=text("'medium'"),
    )
    rfc2119_severity: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        server_default=text("true"),
    )


class KalKgEvidence(UUIDPrimaryKeyMixin, Base):
    """Evidence chain row — one (claim, rule) check per row.

    Paper 5 Section 3.2: ``decision`` is ``PASS`` / ``FLAG`` /
    ``REJECT``. ``rfc2119_severity`` mirrors the matching rule's
    severity at decision time so an evidence row stays interpretable
    even if the rule's severity changes later.

    Three FK columns trace the row back to its source:
    - ``run_id`` — the verification run that produced this evidence
      (NOT NULL; an evidence row without a run is meaningless).
    - ``triple_id`` — the KG triple under check (nullable + SET NULL
      on delete: triples can be deleted while evidence is kept for
      audit, so the FK doesn't cascade away the evidence).
    - ``policy_rule_id`` — the rule whose check produced this
      decision (nullable + SET NULL on delete: rules can be revised
      while evidence is preserved).

    ``input_hash`` / ``output_hash`` / ``signature`` are the
    tamper-evidence trio populated by the HMAC step in knowly's
    verification pipeline (paper §3.5). The package ships the
    columns so a downstream consumer wiring HMAC into their own
    flow has the storage; the columns stay NULL when no signer is
    configured.
    """

    __tablename__ = "kg_evidence"
    __table_args__ = (
        Index("idx_kal_kg_evidence_tenant_run", "tenant_id", "run_id"),
        Index("idx_kal_kg_evidence_tenant_decision", "tenant_id", "decision"),
        Index(
            "idx_kal_kg_evidence_tenant_severity",
            "tenant_id",
            "rfc2119_severity",
        ),
        {"schema": SCHEMA},
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        nullable=False,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SCHEMA}.verification_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    triple_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(f"{SCHEMA}.kg_triples.id", ondelete="SET NULL"),
        nullable=True,
    )
    policy_rule_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(f"{SCHEMA}.policy_rules.id", ondelete="SET NULL"),
        nullable=True,
    )
    rfc2119_severity: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )
    geometric_distance: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    energy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy_band: Mapped[str | None] = mapped_column(String(20), nullable=True)
    decision: Mapped[str] = mapped_column(String(10), nullable=False)
    source_document_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    source_section: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    aggregate_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    output_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    signature: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class KalTripleVerdict(UUIDPrimaryKeyMixin, Base):
    """One verifier's verdict on one triple — §5.8 vertical-axis write half.

    Per the v0.5.0 plan
    (``docs/plans/plan_knowlytix_kal_v0_5.md``), each row is a single
    verifier's stance on a single triple. ``UNIQUE (triple_id,
    verifier)`` is the UPSERT target for
    ``KalDefaultKGStore.update_verification``; writing a second
    verifier's verdict on the same triple inserts a NEW row instead of
    overwriting the existing one. v0.3.0 / v0.4.0 keyed the read-side
    dedup on ``(content_hash, verifier)`` already — v0.5.0 fills in the
    write-side so the multi-verifier rows can actually exist.

    **Distinct from ``KalVerificationRun``.** ``KalVerificationRun`` is
    a *run-record* (one row per GMS verification execution; carries
    aggregate embeddings + scores). ``KalTripleVerdict`` is a *verdict*
    (one verifier's stance on one triple; no run scoping). Different
    lifecycles, different keys — they live in the same module because
    they're both part of the verification chain but they don't share a
    table.

    **Cascade on triple delete.** ``triple_id`` FK uses ``ON DELETE
    CASCADE`` because verdicts are dependent on their triple — if the
    triple is gone, its verdicts have no referent. Differs from the
    v0.4.0 RESTRICT direction (which protected entity-delete from
    silently nuking triples; that's the inverse FK direction).

    **Cache invariant.** ``KalKgTriple`` carries cache columns
    (``verifier`` / ``confidence`` / ``verification_status`` /
    ``verified_at`` / ``scores``) that hold the **most-recently-written**
    verdict across all verifiers. The cache is updated by
    ``update_verification`` in the same transaction as the UPSERT here.
    ``min_confidence`` query filters use the cache (indexed pushdown
    preserved); callers wanting "any-verifier above threshold" semantics
    must set ``KALQuery.include_all_verifications=True`` and filter in
    app code.
    """

    __tablename__ = "triple_verdicts"
    __table_args__ = (
        UniqueConstraint(
            "triple_id", "verifier", name="uq_kal_triple_verdict"
        ),
        Index("idx_kal_triple_verdicts_triple", "triple_id"),
        Index(
            "idx_kal_triple_verdicts_verifier", "tenant_id", "verifier"
        ),
        Index(
            "idx_kal_triple_verdicts_conf", "tenant_id", "confidence"
        ),
        {"schema": SCHEMA},
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        nullable=False,
    )
    triple_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SCHEMA}.kg_triples.id", ondelete="CASCADE"),
        nullable=False,
    )
    verifier: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    verification_status: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    scores: Mapped[dict[str, float] | None] = mapped_column(  # type: ignore[type-arg]
        JSONB, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
