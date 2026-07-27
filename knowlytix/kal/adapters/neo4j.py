"""Read-only Neo4j (Bolt) source adapter.

KAL §7.3 Pattern 1 — a native adapter that projects a Neo4j property
graph into ``KALTriple`` rows at query time. Modeled on
``WikidataAdapter`` (the other read-only external source), with two
differences that make it *simpler*, not harder:

- **No injection surface.** Cypher binds parameters, so every caller
  value rides through as ``$subject`` / ``$predicate`` / ``$object`` /
  ``$id`` / ``$limit``, and the node-key property is reached via
  dynamic access ``s[$node_key]``. Nothing is string-interpolated, so
  the SPARQL adapter's regex-validate-then-interpolate dance is
  unnecessary. The one identifier that *names* (rather than values) a
  Cypher fragment is ``node_key``; it is validated at construction.
- **Generic projection.** A property graph has arbitrary labels and
  relationship types; the adapter matches the generic shape
  ``(s)-[r]->(o)`` and projects ``(s[node_key], type(r), o[node_key])``.
  ``node_key`` (default ``"name"``) is the node property used as the
  KAL node id/name — the join key that makes the projection legible.

Read-only: every write/update/delete/quarantine/vector-search method
raises ``CapabilityNotSupportedError`` so a router fallback can never
send a mutation here. ``tenant_id`` is accepted (Protocol contract) but
ignored — a Neo4j connection points at one database; multi-tenancy is
the connection service's concern, not this adapter's.

The ``neo4j`` driver is imported lazily (only when the adapter builds
its own driver) so the module imports fine without the ``[neo4j]``
extra; tests inject a fake driver and never import it.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Final

from knowlytix.kal.errors import (
    AdapterAuthError,
    AdapterConnectionError,
    CapabilityNotSupportedError,
)
from knowlytix.kal.protocol import AdapterCapabilities
from knowlytix.kal.types import (
    KALNode,
    KALQuery,
    KALQueryResult,
    KALTriple,
    KALTriplePattern,
    VerificationMetadata,
)

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


# Default Bolt endpoint used by the builder + tests when the row's
# config omits ``uri``. The standard single-instance Bolt port.
DEFAULT_NEO4J_URI: Final[str] = "bolt://localhost:7687"

# Node property used as the KAL node id/name when the config omits it.
# A property graph has no universal identity property; ``name`` is the
# most common human-legible choice and matches the demo fixtures.
DEFAULT_NODE_KEY: Final[str] = "name"

# Maximum rows a single query returns. The federation router applies
# its own LIMIT via ``KALQuery.limit``; capping here too means a
# malformed unbounded query can't blow up the response. Matches the
# Wikidata adapter's hard cap and ``AdapterCapabilities.max_batch_size``.
_CYPHER_HARD_LIMIT: Final[int] = 1000

# ``node_key`` names a Cypher property; it is reached via dynamic
# access ``s[$node_key]`` (bound), so it never reaches the query as
# raw text — but a non-identifier value is still a config mistake we
# want to surface at construction rather than as an empty result.
_NODE_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_]\w*$")

# Projection shared by every read path: a row is ``{s, p, o}`` where s
# and o are the node-key property values and p is the relationship type.
_RETURN_TRIPLE: Final[str] = (
    "RETURN s[$node_key] AS s, type(r) AS p, o[$node_key] AS o"
)


class Neo4jKnowledgeAdapter:
    """KAL adapter projecting a Neo4j property graph into triples.

    ``driver`` is injectable so tests can hand in a fake async driver
    instead of opening a Bolt connection. Production callers (the
    builder) leave it ``None`` so the adapter constructs its own driver
    and the builder's ``aclose`` is the canonical cleanup path.
    """

    def __init__(
        self,
        *,
        name: str,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        node_key: str = DEFAULT_NODE_KEY,
        driver: AsyncDriver | None = None,
    ) -> None:
        if not _NODE_KEY_PATTERN.match(node_key):
            raise ValueError(
                f"node_key must match {_NODE_KEY_PATTERN.pattern!r}; "
                f"got {node_key!r}"
            )
        self._name = name
        self._database = database
        self._node_key = node_key
        self._closed = False
        self._driver = driver if driver is not None else _build_driver(uri, user, password)

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> AdapterCapabilities:
        # Read-only projection: no writes, deletes, or vector search.
        # ``supports_text_search`` stays False because the read paths
        # don't honour ``KALQuery.text_search`` — flip it on only when
        # a full-text clause is wired in, so the router never dispatches
        # a text query to a backend that silently ignores the term.
        return AdapterCapabilities(
            can_read=True,
            can_write=False,
            can_delete=False,
            supports_vector_search=False,
            supports_text_search=False,
            supports_provenance=False,
            supports_verification_metadata=False,
            supports_atomic_writes=False,
            max_batch_size=_CYPHER_HARD_LIMIT,
        )

    async def aclose(self) -> None:
        """Close the Bolt driver. Idempotent — a second call is a no-op."""
        if self._closed:
            return
        self._closed = True
        await self._driver.close()

    async def __aenter__(self) -> Neo4jKnowledgeAdapter:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        await self.aclose()

    async def query_triples(
        self,
        query: KALQuery,
        tenant_id: str | None = None,
    ) -> KALQueryResult:
        """Run a Cypher MATCH for triples matching ``query.patterns``.

        ``tenant_id`` is accepted (Protocol contract) but ignored — the
        driver points at one database. ``query.limit`` is honoured up
        to the configured hard cap; ``query.timeout_ms`` is the router's
        concern, not the adapter's.
        """
        cypher, params = self._build_query(query)
        rows = await self._run_cypher(cypher, params)
        return KALQueryResult(triples=_rows_to_triples(rows))

    async def get_node(
        self,
        node_id: str,
        tenant_id: str | None = None,
    ) -> KALNode | None:
        """Resolve a node by its key property. Returns ``None`` when no
        node has ``node_key == node_id``."""
        cypher = (
            "MATCH (n) WHERE n[$node_key] = $id "
            "RETURN n[$node_key] AS name, labels(n) AS labels LIMIT 1"
        )
        rows = await self._run_cypher(
            cypher, {"node_key": self._node_key, "id": node_id}
        )
        if not rows:
            return None
        name = _as_str(rows[0].get("name"))
        if not name:
            return None
        return KALNode(id=node_id, name=name, node_types=_as_labels(rows[0].get("labels")))

    async def query_adjacent_triples(
        self,
        node_id: str,
        predicate: str | None = None,
        direction: str = "outgoing",
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[KALTriple]:
        """Walk one hop from ``node_id``. ``direction`` is
        ``"outgoing"`` (node is subject), ``"incoming"`` (node is
        object), or ``"both"``. ``predicate``, when given, filters on
        the relationship type. All inputs are bound parameters."""
        cypher, params = self._build_adjacent(node_id, predicate, direction, limit)
        rows = await self._run_cypher(cypher, params)
        return _rows_to_triples(rows)

    # --- write methods raise; capability flags filter them out at the
    # router level too, this is the second line of defense.

    async def insert_triples(
        self,
        triples: list[KALTriple],
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError(
            "Neo4j adapter is read-only; insert_triples is not supported"
        )

    async def update_verification(
        self,
        triple_id: str,
        verification: VerificationMetadata,
        tenant_id: str | None = None,
    ) -> bool:
        raise CapabilityNotSupportedError(
            "Neo4j adapter is read-only; update_verification is not supported"
        )

    async def delete_triples(
        self,
        triple_ids: list[str],
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError(
            "Neo4j adapter is read-only; delete_triples is not supported"
        )

    async def quarantine_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError(
            "Neo4j adapter is read-only; quarantine is not supported"
        )

    async def restore_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError(
            "Neo4j adapter is read-only; restore is not supported"
        )

    async def search_similar_nodes(
        self,
        query_vector: list[float],
        tenant_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[KALNode, float]]:
        raise CapabilityNotSupportedError(
            "Neo4j adapter does not support vector search"
        )

    # --- internals

    def _build_query(self, query: KALQuery) -> tuple[str, dict[str, object]]:
        params: dict[str, object] = {
            "node_key": self._node_key,
            "limit": _capped_limit(query.limit),
        }
        clauses: list[str] = []
        if query.patterns:
            if len(query.patterns) > 1:
                # Matches the Wikidata adapter's v1 stance: honour the
                # first pattern and warn loudly. Multi-pattern joins
                # need per-pattern variable scoping — a separate design
                # problem. Warning beats silently truncating.
                logger.warning(
                    "neo4j adapter received %d patterns; only the first "
                    "is honoured. Multi-pattern joins are a v2 feature.",
                    len(query.patterns),
                )
            clauses = self._pattern_clauses(query.patterns[0], params)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        cypher = f"MATCH (s)-[r]->(o){where} {_RETURN_TRIPLE} LIMIT $limit"
        return cypher, params

    @staticmethod
    def _pattern_clauses(
        pattern: KALTriplePattern, params: dict[str, object]
    ) -> list[str]:
        # Each concrete slot becomes a bound-parameter equality. The
        # node-key property is reached via dynamic access so even the
        # property name is bound, not interpolated.
        clauses: list[str] = []
        if pattern.subject is not None:
            clauses.append("s[$node_key] = $subject")
            params["subject"] = pattern.subject
        if pattern.predicate is not None:
            clauses.append("type(r) = $predicate")
            params["predicate"] = pattern.predicate
        if pattern.object is not None:
            clauses.append("o[$node_key] = $object")
            params["object"] = pattern.object
        return clauses

    def _build_adjacent(
        self,
        node_id: str,
        predicate: str | None,
        direction: str,
        limit: int,
    ) -> tuple[str, dict[str, object]]:
        params: dict[str, object] = {
            "node_key": self._node_key,
            "id": node_id,
            "limit": _capped_limit(limit),
        }
        if direction == "incoming":
            anchor = "o[$node_key] = $id"
        elif direction == "both":
            anchor = "(s[$node_key] = $id OR o[$node_key] = $id)"
        else:  # "outgoing" (default)
            anchor = "s[$node_key] = $id"
        clauses = [anchor]
        if predicate is not None:
            clauses.append("type(r) = $predicate")
            params["predicate"] = predicate
        where = " AND ".join(clauses)
        cypher = f"MATCH (s)-[r]->(o) WHERE {where} {_RETURN_TRIPLE} LIMIT $limit"
        return cypher, params

    async def _run_cypher(
        self, cypher: str, params: dict[str, object]
    ) -> list[dict[str, object]]:
        # Errors classify into the KAL taxonomy so the federation router
        # can render them as ``AdapterError`` rows; non-neo4j exceptions
        # (real bugs) propagate unchanged rather than masquerading as
        # connection failures.
        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run(cypher, params)
                rows = await result.data()
            return [row for row in rows if isinstance(row, dict)]
        except Exception as exc:
            mapped = _map_neo4j_error(exc)
            if mapped is None:
                raise
            raise mapped from exc


def _build_driver(uri: str, user: str, password: str) -> AsyncDriver:
    # Lazy import keeps the module importable without the ``[neo4j]``
    # extra. ``driver()`` is lazy — no Bolt connection opens until the
    # first session runs — so construction is cheap and offline-safe.
    from neo4j import AsyncGraphDatabase

    return AsyncGraphDatabase.driver(uri, auth=(user, password))


def _map_neo4j_error(exc: BaseException) -> Exception | None:
    # Lazy import: the error-mapping path only runs after a driver call,
    # by which point neo4j is importable. Returns the mapped KAL error,
    # or ``None`` when ``exc`` is not a neo4j error (so the caller
    # re-raises it unchanged).
    try:
        from neo4j import exceptions as neo4j_exc
    except ImportError:  # pragma: no cover — driver present once a call ran
        return AdapterConnectionError(f"neo4j driver error: {exc}")
    if isinstance(exc, neo4j_exc.AuthError):
        return AdapterAuthError(f"neo4j rejected credentials: {exc}")
    if isinstance(exc, (neo4j_exc.ServiceUnavailable, neo4j_exc.SessionExpired)):
        return AdapterConnectionError(f"neo4j unreachable: {exc}")
    if isinstance(exc, neo4j_exc.Neo4jError):
        return AdapterConnectionError(f"neo4j query failed: {exc}")
    return None


def _capped_limit(requested: int | None) -> int:
    # ``None`` → hard cap; non-positive → 1 (``LIMIT 0`` is valid Cypher
    # but rarely intended and almost always a caller bug). Mirrors the
    # Wikidata adapter's clamp.
    if requested is None:
        return _CYPHER_HARD_LIMIT
    return max(1, min(requested, _CYPHER_HARD_LIMIT))


def _rows_to_triples(rows: list[dict[str, object]]) -> list[KALTriple]:
    """Project ``{s, p, o}`` Cypher rows into ``KALTriple`` rows.

    Rows whose subject or object key is null/empty are **skipped**: a
    Neo4j node that lacks the configured ``node_key`` property projects
    to an empty endpoint, and an empty-named entity is not a usable
    triple — it would collapse every keyless node into one phantom entity
    in the consumer graph (GMS keys entities by name). Dropping such rows
    is the faithful projection.

    ``KALTriple.id`` stays ``None`` — a projection has no stable
    statement-level id, and the router's dedup keys on ``content_hash``
    anyway, so an absent id is the honest answer (symmetric with the
    Wikidata adapter).
    """
    triples: list[KALTriple] = []
    for row in rows:
        subject = _as_str(row.get("s"))
        obj = _as_str(row.get("o"))
        if not subject or not obj:
            continue
        triples.append(
            KALTriple(
                subject=KALNode(id=subject, name=subject),
                predicate=_as_str(row.get("p")),
                object=KALNode(id=obj, name=obj),
            )
        )
    return triples


def _as_str(value: object) -> str:
    return "" if value is None else str(value)


def _as_labels(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(label) for label in value]
