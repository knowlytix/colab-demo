"""KAL core types — backend-agnostic representation of nodes, triples, queries.

Per KAL spec v1.4.1 sections 3.1-3.3. Mypy-strict; no ``Any`` allowed.

The ``# type: ignore[explicit-any]`` on each ``class Foo(BaseModel):`` line
suppresses a Pydantic v2 + mypy strict-any interaction; the project uses the
same suppression in extraction/schemas.py etc.
"""

from __future__ import annotations

import builtins
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Per KAL spec §4.3 — the canonical set of values for
# ``AdapterError.error_type``. Pydantic validates incoming values against
# this Literal at construction time, so adapter authors and router code
# can't drift apart. ``"capability"`` fires when an adapter raises
# ``CapabilityNotSupportedError`` (defensive — the router's pre-dispatch
# filter should normally prevent this). ``"circuit_open"`` is reserved
# for a future circuit-breaker story; not emitted today.
type AdapterErrorType = Literal[
    "timeout",
    "auth",
    "connection",
    "transport",
    "capability",
    "circuit_open",
    "internal",
]

# "Anything" value type for free-form property dicts. KAL spec uses Python's
# builtin ``object`` here, but inside ``KALTriple`` the field name ``object``
# shadows the builtin, so we alias via ``builtins.object`` and use the alias
# everywhere for consistency.
type PropertyValue = builtins.object

# Direction for one-hop adjacency walks (``KGStore.query_adjacent_triples``).
# Lives here — not in ``knowly.kal.storage`` — so that re-exporting it at the
# package top level (``knowlytix.kal``) doesn't drag SQLAlchemy in via the
# Protocol module. ``"outgoing"`` = node is subject; ``"incoming"`` = node
# is object; ``"both"`` = either.
type AdjacencyDirection = Literal["outgoing", "incoming", "both"]


class KALNode(BaseModel):  # type: ignore[explicit-any]
    """Universal entity/concept.

    ``id`` is ``str`` (not UUID) because external backends use URIs, integers,
    or other ID schemes. ``adapter`` is stamped by the federation router, not
    by individual adapters (per Story 1 contract — direct adapter calls leave
    it ``None``).
    """

    id: str
    name: str
    node_types: list[str] = []
    normalized_name: str | None = None
    properties: dict[str, PropertyValue] = {}
    adapter: str | None = None

    @property
    def primary_type(self) -> str | None:
        """First node_type, or None when empty. Convenience for single-label backends."""
        return self.node_types[0] if self.node_types else None


class KALLiteral(BaseModel):  # type: ignore[explicit-any]
    """Scalar value for triple objects (RDF typed/language literals, numeric values).

    Backend-agnostic — maps to RDF literals, Neo4j property values, or any
    non-node scalar.
    """

    value: str
    datatype: str | None = None
    language: str | None = None


class TripleProvenance(BaseModel):  # type: ignore[explicit-any]
    """Where a triple came from. Maps to W3C PROV-O concepts."""

    source: str | None = None
    extracted_at: datetime | None = None
    extractor: str | None = None
    source_text: str | None = None


class VerificationMetadata(BaseModel):  # type: ignore[explicit-any]
    """Verification scores/status that travel with a triple across federation.

    Knowly's core differentiator — every triple can carry quantified
    verification metadata as first-class data.

    Per v0.5.0 §5.8: ``verifier`` is the vertical-axis identity. When
    provided, it must be a non-empty string (``min_length=1`` rejects
    ``""``). ``None`` stays allowed at the model layer (default); the
    adapter boundary rejects it separately when
    ``update_verification`` is called, because "anonymous verification"
    has no meaningful place on the §3.5 dual-axis federation model.
    Note: ``min_length=1`` counts characters, so a whitespace-only
    string like ``"   "`` passes the validator — callers shouldn't
    pass such strings, but the model doesn't enforce trim-then-check.
    """

    confidence: float | None = None
    verification_status: str | None = None
    verified_at: datetime | None = None
    verifier: str | None = Field(default=None, min_length=1)
    scores: dict[str, float] | None = None


class KALTriple(BaseModel):  # type: ignore[explicit-any]
    """Universal triple with provenance + verification metadata.

    Per v1.4.1 §3.2: exactly one of ``object`` (node) or ``object_literal``
    (scalar) must be set. ``id`` is **adapter-local** — use ``content_hash``
    for stable cross-adapter identity. ``adapter`` is stamped by the
    federation router, not by the adapter itself.
    """

    id: str | None = None
    subject: KALNode
    predicate: str
    object: KALNode | None = None
    object_literal: KALLiteral | None = None
    provenance: TripleProvenance | None = None
    verification: VerificationMetadata | None = None
    properties: dict[str, PropertyValue] = {}
    adapter: str | None = None

    @model_validator(mode="after")
    def validate_object(self) -> KALTriple:
        has_node = self.object is not None
        has_literal = self.object_literal is not None
        if not has_node and not has_literal:
            raise ValueError("KALTriple must have either object or object_literal")
        if has_node and has_literal:
            raise ValueError("KALTriple cannot have both object and object_literal")
        return self

    @property
    def content_hash(self) -> tuple[str, ...]:
        """Stable cross-adapter identity tuple. Used by federation dedup.

        Per v1.4.1 §3.2 + E1 fix: tuple starts with the subject key (subject
        is always a node — literal subjects unsupported). The 3rd element
        is the object-kind discriminator: ``"node"`` or ``"lit"``.
        """
        s_key = self.subject.normalized_name or self.subject.name
        if self.object is not None:
            o_key = self.object.normalized_name or self.object.name
            return (s_key, self.predicate, "node", o_key)
        # Literal branch — guaranteed by validate_object
        lit = self.object_literal
        assert lit is not None
        return (
            s_key,
            self.predicate,
            "lit",
            lit.value,
            lit.datatype or "",
            lit.language or "",
        )


class KALTriplePattern(BaseModel):  # type: ignore[explicit-any]
    """A single match pattern for triple queries.

    ``object`` matches node objects by name/ID only — literal-object triples
    are not reachable via the ``object`` pattern field (per v1.4.1 E8 limit).
    They can be filtered via ``subject`` and ``predicate`` only.
    """

    subject: str | None = None
    predicate: str | None = None
    object: str | None = None


class KALQuery(BaseModel):  # type: ignore[explicit-any]
    """Query specification for adapter reads.

    ``timeout_ms`` is per-query federation budget (Story 9). ``ge=0``
    rejects negative values at validation; ``timeout_ms=0`` is
    explicitly allowed and treated the same as ``None`` (use the
    router's default) — per v1.4.1 E2 parens-fix note in
    ``FederationRouter._resolve_timeout_seconds``.
    """

    patterns: list[KALTriplePattern] = []
    text_search: str | None = None
    vector: list[float] | None = None
    limit: int = 100
    offset: int = 0
    min_confidence: float | None = None
    include_unverified: bool = True
    timeout_ms: int | None = Field(default=None, ge=0)
    # v0.5.0 §5.8 — when True, the read path materialises one row per
    # ``(triple, verifier)`` instead of one row per triple. Default off
    # for direct adapter callers (preserves the pre-v0.5.0 single-row
    # shape). The federation router flips it on for
    # ``ConflictStrategy.ALL`` so the vertical-axis dedup keys on
    # ``(content_hash, verifier)`` actually surface multi-verifier rows;
    # the router leaves it off for ``HIGHEST_CONFIDENCE`` / ``MOST_RECENT``
    # which already pick one row per triple by their own ordering.
    include_all_verifications: bool = False
    # v0.11.0 — when False (default), quarantined triples (``revoked_at``
    # set) are excluded from results everywhere; set True for audit views
    # that want revoked rows. No-op for adapters with no revoke concept
    # (file/SPARQL backends).
    include_revoked: bool = False


class AdapterError(BaseModel):  # type: ignore[explicit-any]
    """Per-adapter failure encountered during a federated query.

    NOTE: This is a Pydantic **data model**, NOT an exception. It rides
    inside ``KALQueryResult.errors`` to describe what went wrong on each
    adapter that contributed to a federated result. Single-adapter calls
    (no federation router) raise from the ``knowly.kal.errors`` hierarchy
    (``TenantRequiredError``, ``AdapterAuthError``, etc.) instead of
    populating this model.

    Populated by ``FederationRouter`` (Story 9).
    """

    adapter: str
    error_type: AdapterErrorType
    message: str


class KALQueryResult(BaseModel):  # type: ignore[explicit-any]
    """Result of a KAL query.

    ``errors`` is non-empty when any adapter in a federated query failed.
    Callers MUST inspect before treating triples as authoritative.

    ``query_time_ms`` is wall-clock end-to-end: for federated calls it
    includes fan-out, slowest adapter, and merge cost; for direct
    adapter calls it covers the single backend round-trip plus result
    materialization. Populated by the merge / adapter return path; a
    ``0`` value means **either** the call finished in under 1 ms
    (``int(seconds * 1000)`` floors) **or** the producer chose not to
    instrument (test fixtures, mock adapters). Tracing consumers should
    treat sub-ms buckets as noise rather than as a "not instrumented"
    sentinel.
    """

    triples: list[KALTriple]
    query_time_ms: int = 0
    errors: list[AdapterError] = []
