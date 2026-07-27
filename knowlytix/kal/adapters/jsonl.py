"""``JsonlKnowledgeAdapter`` — read-only adapter backed by a ``.jsonl`` file.

Per v0.7.0 plan (``docs/plans/plan_knowlytix_kal_v0_7.md``). Targets paper
§6.6 "stores without provenance/verification fields as read-only adapters"
— flips that row from 🚧 to ✅.

Shape: one JSON-encoded ``KALTriple`` per line. Loaded once at construction
into an in-memory list (eager parse — format errors surface at startup with
the offending line number, not on first query). Reads are O(N) scans over
the cached list; the adapter is the lightweight option, not the indexed
one. Use Postgres for >100k-row workloads.

Capability profile:

- ``can_read=True``
- ``can_write=False`` / ``can_delete=False``
- ``supports_vector_search=False``  (no embeddings on disk)
- ``supports_text_search=False``    (no inverted index)
- ``supports_provenance=True``      (each triple carries its own)
- ``supports_verification_metadata=True``
- ``supports_atomic_writes=False``  (read-only)

Direct-call error contract (per ``AdapterError`` docstring in
``knowly.kal.types``): single-adapter callers see exceptions from
``knowly.kal.errors``; the federation router catches and classifies them
into ``KALQueryResult.errors`` for its consumers. Concretely:

- ``text_search`` / ``vector`` on ``query_triples`` →
  ``CapabilityNotSupportedError``
- ``search_similar_nodes`` → ``CapabilityNotSupportedError``
- ``insert_triples`` / ``update_verification`` / ``delete_triples`` →
  ``CapabilityNotSupportedError`` ("read-only")
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from knowlytix.kal.errors import CapabilityNotSupportedError
from knowlytix.kal.protocol import AdapterCapabilities
from knowlytix.kal.types import (
    AdjacencyDirection,
    KALNode,
    KALQuery,
    KALQueryResult,
    KALTriple,
    KALTriplePattern,
    VerificationMetadata,
)

_CAPABILITIES = AdapterCapabilities(
    can_read=True,
    can_write=False,
    can_delete=False,
    supports_vector_search=False,
    supports_text_search=False,
    supports_provenance=True,
    supports_verification_metadata=True,
    supports_atomic_writes=False,
)


class JsonlKnowledgeAdapter:
    """Read-only ``KnowledgeAdapter`` over a JSONL file of ``KALTriple`` rows."""

    def __init__(self, name: str, *, path: Path | str) -> None:
        self._name = name
        self._path = Path(path)
        self._triples: list[KALTriple] = _load_jsonl(self._path)

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> AdapterCapabilities:
        return _CAPABILITIES

    # --- Read operations ---

    async def query_triples(
        self,
        query: KALQuery,
        tenant_id: str | None = None,
    ) -> KALQueryResult:
        if query.text_search is not None:
            raise CapabilityNotSupportedError(
                "JSONL adapter does not support text search"
            )
        if query.vector is not None:
            raise CapabilityNotSupportedError(
                "JSONL adapter does not support vector search"
            )
        start = time.monotonic()
        filters = _PatternFilters.from_patterns(query.patterns)
        limit_cap = query.offset + query.limit
        matches: list[KALTriple] = []
        for triple in self._triples:
            if not _passes_filters(triple, query, filters):
                continue
            matches.append(triple)
            if len(matches) >= limit_cap:
                break
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return KALQueryResult(
            triples=matches[query.offset : limit_cap],
            query_time_ms=elapsed_ms,
        )

    async def get_node(
        self,
        node_id: str,
        tenant_id: str | None = None,
    ) -> KALNode | None:
        for triple in self._triples:
            if triple.subject.id == node_id:
                return triple.subject
            if triple.object is not None and triple.object.id == node_id:
                return triple.object
        return None

    async def query_adjacent_triples(
        self,
        node_id: str,
        predicate: str | None = None,
        direction: str = "outgoing",
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[KALTriple]:
        # KAL Protocol types ``direction`` as ``str``; narrow to the
        # ``AdjacencyDirection`` Literal for the internal walk (same
        # pattern as ``PostgresKnowledgeAdapter.query_adjacent_triples``).
        narrowed = cast(AdjacencyDirection, direction)
        matches: list[KALTriple] = []
        for triple in self._triples:
            if not _is_adjacent(triple, node_id, narrowed):
                continue
            if predicate is not None and triple.predicate != predicate:
                continue
            matches.append(triple)
            if len(matches) >= limit:
                break
        return matches

    # --- Write operations (rejected — read-only) ---

    async def insert_triples(
        self,
        triples: list[KALTriple],
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError("JSONL adapter is read-only")

    async def update_verification(
        self,
        triple_id: str,
        verification: VerificationMetadata,
        tenant_id: str | None = None,
    ) -> bool:
        raise CapabilityNotSupportedError("JSONL adapter is read-only")

    async def delete_triples(
        self,
        triple_ids: list[str],
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError("JSONL adapter is read-only")

    async def quarantine_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError("JSONL adapter is read-only")

    async def restore_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError("JSONL adapter is read-only")

    # --- Vector search (rejected — no embeddings) ---

    async def search_similar_nodes(
        self,
        query_vector: list[float],
        tenant_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[KALNode, float]]:
        raise CapabilityNotSupportedError(
            "JSONL adapter does not support vector search"
        )


def _load_jsonl(path: Path) -> list[KALTriple]:
    """Parse a ``.jsonl`` file into ``KALTriple`` rows.

    Eager so format errors surface at startup with the offending line
    number rather than on first query. Blank lines are tolerated (JSONL
    convention); each non-blank line must be a valid ``KALTriple`` JSON
    object. ``ValidationError`` covers both malformed JSON and schema
    violations — Pydantic wraps the underlying ``json.JSONDecodeError``.
    """
    triples: list[KALTriple] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                triples.append(KALTriple.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(
                    f"{path}: malformed KALTriple at line {lineno}: {exc}"
                ) from exc
    return triples


@dataclass(frozen=True)
class _PatternFilters:
    """Pre-computed ``KALTriplePattern`` projections for the hot path.

    Built once per ``query_triples`` call so the per-triple loop doesn't
    rebuild the set comprehensions on every iteration (would otherwise
    be ``O(triples * patterns)`` set rebuilds). Set-membership-within-field
    + AND-across-fields semantics mirror the Postgres adapter's
    ``_collect_filter`` (see its docstring): multiple patterns
    constraining the same field compile to ``IN (...)``; different
    fields AND together. A single pattern collapses to pure AND.
    """

    subjects: frozenset[str]
    predicates: frozenset[str]
    objects: frozenset[str]

    @classmethod
    def from_patterns(
        cls, patterns: list[KALTriplePattern]
    ) -> _PatternFilters:
        return cls(
            subjects=frozenset(
                p.subject for p in patterns if p.subject is not None
            ),
            predicates=frozenset(
                p.predicate for p in patterns if p.predicate is not None
            ),
            objects=frozenset(
                p.object for p in patterns if p.object is not None
            ),
        )

    @property
    def is_empty(self) -> bool:
        return not (self.subjects or self.predicates or self.objects)


def _passes_filters(
    triple: KALTriple, query: KALQuery, filters: _PatternFilters
) -> bool:
    """Apply non-pattern filters and pattern match."""
    if not query.include_unverified and triple.verification is None:
        return False
    if query.min_confidence is not None:
        if triple.verification is None:
            return False
        confidence = triple.verification.confidence
        if confidence is None or confidence < query.min_confidence:
            return False
    if filters.is_empty:
        return True
    return _matches_patterns(triple, filters)


def _matches_patterns(triple: KALTriple, filters: _PatternFilters) -> bool:
    if filters.subjects and not _node_matches_any(
        triple.subject, filters.subjects
    ):
        return False
    if filters.predicates and triple.predicate not in filters.predicates:
        return False
    if filters.objects:
        # Literal-object triples are not reachable via the ``object``
        # pattern field (per v1.4.1 E8 limit on ``KALTriplePattern``).
        if triple.object is None:
            return False
        if not _node_matches_any(triple.object, filters.objects):
            return False
    return True


def _node_matches_any(node: KALNode, values: frozenset[str]) -> bool:
    """``KALTriplePattern`` doc says nodes are matched "by name/ID only";
    accepting either keeps fixture authoring forgiving.
    """
    return node.id in values or node.name in values


def _is_adjacent(
    triple: KALTriple, node_id: str, direction: AdjacencyDirection
) -> bool:
    subj_match = triple.subject.id == node_id
    obj_match = triple.object is not None and triple.object.id == node_id
    if direction == "outgoing":
        return subj_match
    if direction == "incoming":
        return obj_match
    return subj_match or obj_match
