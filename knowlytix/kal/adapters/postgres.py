"""PostgresKnowledgeAdapter — KAL surface over a pluggable ``KGStore``.

Per KAL spec v1.4.1 section 7 + paper §7 ("the adapter delegates rather
than reimplements"). The adapter owns the KAL surface contracts —
``capabilities``, tenant enforcement (``_ensure_tenant``), per-call
timeouts (``_with_timeout``), session composition over the injected
``async_session_factory`` (KAL spec sec 7.3), input validation,
foreign external-ID registration. It delegates every backend
operation to a ``KGStore`` (default: ``KalDefaultKGStore``, which
targets this package's ``kal.*`` schema; can be swapped for a host-
provided impl such as knowly's ``KnowlyKGStore``).

The store handles the storage primitives — pattern queries, adjacency
walks, vector search, dedup + entity resolution + embedding on insert,
ORM-to-KAL materialization. The adapter is the KAL boundary; the
store is the schema boundary.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Awaitable
from typing import Literal, TypeVar, cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from knowlytix.kal.errors import TenantRequiredError
from knowlytix.kal.external_id_registry import bulk_register
from knowlytix.kal.protocol import AdapterCapabilities
from knowlytix.kal.storage import AdjacencyDirection, KGStore
from knowlytix.kal.storage_default import KalDefaultKGStore
from knowlytix.kal.storage_default._normalize import normalize_entity_name
from knowlytix.kal.types import (
    KALNode,
    KALQuery,
    KALQueryResult,
    KALTriple,
    KALTriplePattern,
    VerificationMetadata,
)

# Fallback default for ``KALQuery.timeout_ms`` resolution when neither
# the per-call value nor the per-adapter override is set. Knowly's
# ``Settings.kal_adapter_default_timeout_ms`` plays this role on the
# knowly side; the standalone package reads from ``KAL_ADAPTER_DEFAULT_TIMEOUT_MS``
# (or falls back to 30s) so it doesn't depend on knowly's Settings
# layer. Hosts that want a different default override per adapter via
# the ``default_query_timeout_ms`` constructor arg.
_DEFAULT_TIMEOUT_MS_ENV: str = "KAL_ADAPTER_DEFAULT_TIMEOUT_MS"
_DEFAULT_TIMEOUT_MS_FALLBACK: int = 30_000


def _resolve_default_timeout_ms() -> int:
    """Read the package-level default timeout from the env (or fall back).

    Resolved on every adapter call rather than cached so an operator
    can change the env at runtime without bouncing the process.
    """
    try:
        return int(os.environ.get(_DEFAULT_TIMEOUT_MS_ENV, _DEFAULT_TIMEOUT_MS_FALLBACK))
    except ValueError:
        return _DEFAULT_TIMEOUT_MS_FALLBACK

# Fields on KALTriplePattern that ``_collect_filter`` knows how to compile
# into SQL ``IN (...)`` clauses. Narrowed via Literal so a typo at the
# call site is caught by mypy instead of silently producing an empty
# filter at runtime.
type _PatternField = Literal["subject", "predicate", "object"]

# Canonical name of the app-internal Postgres adapter that lives next
# to the verification pipeline. Wired in by ``main.py``'s lifespan,
# claimed by the debug router's protected-name list, and seeded into
# ``TenantIndex.shared_adapters`` so every tenant's federation
# fan-out sees it. One literal, one source of truth.
POSTGRES_MAIN_ADAPTER_NAME: str = "postgres-main"

# Entity-side limit for the text-search pre-fetch. The adapter resolves
# ``KALQuery.text_search`` to a set of matching entity IDs FIRST, then
# materializes triples that touch those entities. Capping the entity
# fetch independently of the caller's ``query.limit`` avoids silently
# dropping matches when an entity sits on many triples.
_TEXT_SEARCH_ENTITY_CAP = 5000

# Used in ``_with_timeout`` so each adapter method can wrap its inner
# coroutine without losing the return type.
_T = TypeVar("_T")


def _ensure_tenant(tenant_id: str | None) -> uuid.UUID:
    """Convert tenant_id to UUID; raise TenantRequiredError if missing.

    Per KAL spec section 7.3.
    """
    if not tenant_id:
        raise TenantRequiredError(
            "PostgresKnowledgeAdapter requires a non-empty tenant_id"
        )
    return uuid.UUID(tenant_id)


class PostgresKnowledgeAdapter:
    """KAL adapter that delegates storage to a pluggable ``KGStore``.

    Constructor:
    - ``async_session_factory`` (required, KAL spec sec 7.3) — session
      lifecycle owner; the adapter opens a session per call and
      composes transactions around store-method invocations.
    - ``name`` — registry key on ``adapter.name``.
    - ``default_query_timeout_ms`` — fallback for the per-call
      ``KALQuery.timeout_ms`` slot; ``None`` defers to
      ``Settings.kal_adapter_default_timeout_ms``.
    - ``kg_store`` — storage backend Protocol impl. Default
      ``KalDefaultKGStore()`` targets the package's ``kal.*``
      schema. Inject a custom ``KGStore`` (e.g. knowly's
      ``KnowlyKGStore``) to bind the adapter to a different
      KG layer.
    """

    def __init__(
        self,
        async_session_factory: async_sessionmaker[AsyncSession],
        name: str = POSTGRES_MAIN_ADAPTER_NAME,
        *,
        default_query_timeout_ms: int | None = None,
        kg_store: KGStore | None = None,
    ) -> None:
        if default_query_timeout_ms is not None and default_query_timeout_ms < 0:
            # A negative timeout would propagate to ``asyncio.wait_for``
            # as negative seconds and raise at first call — fail fast at
            # construction so misconfiguration surfaces during startup,
            # not in the hot path.
            raise ValueError(
                f"default_query_timeout_ms must be >= 0; "
                f"got {default_query_timeout_ms}"
            )
        self._session_factory = async_session_factory
        self._name = name
        # ``None`` defers to the Settings-backed default
        # (``KAL_ADAPTER_DEFAULT_TIMEOUT_MS``); an explicit positive
        # int overrides per adapter instance for deployments that
        # need tighter or looser local budgets.
        self._default_query_timeout_ms = default_query_timeout_ms
        # Paper §7: "the adapter delegates rather than reimplements". The
        # store owns the kg-schema knowledge (dedup, embedding, ENM,
        # materialization); the adapter owns the KAL surface (capability
        # profile, tenant enforcement, timeouts, session composition).
        # Default to ``KalDefaultKGStore`` (this package's own impl
        # targeting the ``kal.*`` schema); callers can inject a
        # different impl to bind the adapter to a different KG layer.
        self._kg_store: KGStore = kg_store or KalDefaultKGStore()

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Advertised capabilities for this adapter.

        Reflects the full KAL surface the adapter exposes. The
        underlying capability (e.g. vector search) is contributed by
        the injected ``KGStore`` — a future store that can't vector-
        search would either fail capability-advertise at construction
        time or surface ``CapabilityNotSupportedError`` when called.
        Today's default (``KnowlyKGStore``) supports everything below.
        """
        return AdapterCapabilities(
            can_read=True,
            can_write=True,
            can_delete=True,
            supports_vector_search=True,
            # GIN index ``ix_kg_entities_name_fts`` powers
            # ``KALQuery.text_search`` via the store's
            # ``text_search_node_ids`` method.
            supports_text_search=True,
            supports_provenance=True,
            supports_verification_metadata=True,
            supports_atomic_writes=True,
        )

    def _resolve_query_timeout_seconds(
        self, query_timeout_ms: int | None
    ) -> float:
        """Resolve the effective per-call timeout (seconds).

        Priority — highest to lowest:
        1. ``KALQuery.timeout_ms`` (if set and positive)
        2. ``default_query_timeout_ms`` constructor knob
        3. ``Settings.kal_adapter_default_timeout_ms`` (env-backed default)

        ``timeout_ms=0`` is treated identically to ``None`` (per v1.4.1
        E2 parens-fix note in the federation router): a literal zero
        would otherwise fire ``wait_for`` instantly, which is almost
        certainly a caller bug rather than an intent to disable the
        timeout.
        """
        if query_timeout_ms:
            return query_timeout_ms / 1000
        if self._default_query_timeout_ms:
            return self._default_query_timeout_ms / 1000
        return _resolve_default_timeout_ms() / 1000

    async def _with_timeout(
        self, query_timeout_ms: int | None, coro: Awaitable[_T]
    ) -> _T:
        """Run ``coro`` under ``asyncio.wait_for`` at the resolved timeout.

        Centralizes the ``wait_for`` wrap so every adapter method
        applies the §5.4 contract identically. Cancellation is
        cooperative — the inner SQL statement keeps running on the
        Postgres side until the server notices the client connection
        is gone or the query completes naturally; if hard server-side
        cancellation is needed, pass ``statement_timeout`` to the
        ``async_session_factory`` at construction.

        Defence-in-depth note: ``FederationRouter`` also wraps every
        adapter call in ``asyncio.wait_for`` with its own resolved
        budget. The inner (adapter) timer typically fires fractionally
        later than the outer (router) timer; both raise ``TimeoutError``
        and the federation router classifies them as
        ``error_type="timeout"`` regardless of which fired. The adapter
        wrap is therefore the **fallback** that satisfies §5.4 for
        direct callers (debug API, future SDK use) — federated callers
        already get coverage from the router.
        """
        return await asyncio.wait_for(
            coro, timeout=self._resolve_query_timeout_seconds(query_timeout_ms)
        )

    def _check_batch_size(self, count: int, operation: str) -> None:
        """Raise ``ValueError`` if ``count`` exceeds the advertised
        ``capabilities.max_batch_size``. Both ``insert_triples`` and
        ``delete_triples`` need this guard before opening a session —
        a single transaction wraps the whole batch, so an unbounded
        ``count`` would hold one transaction open for the entire load.
        """
        max_batch = self.capabilities.max_batch_size
        if count > max_batch:
            raise ValueError(
                f"{operation} batch of {count} exceeds adapter "
                f"max_batch_size={max_batch}; split into smaller batches"
            )

    # --- Read operations ---

    async def query_triples(
        self,
        query: KALQuery,
        tenant_id: str | None = None,
    ) -> KALQueryResult:
        # Wall-clock from method entry through result materialization.
        # Direct-call callers (debug API, future SDK) read this off
        # ``KALQueryResult.query_time_ms`` for trace attribution.
        # Federated calls also get an outer wall-clock from
        # ``FederationRouter._merge_query_outcomes`` — the router
        # overrides this inner value with its end-to-end measurement.
        start = time.monotonic()
        triples = await self._with_timeout(
            query.timeout_ms, self._query_triples_inner(query, tenant_id)
        )
        return KALQueryResult(
            triples=triples,
            query_time_ms=int((time.monotonic() - start) * 1000),
        )

    async def _query_triples_inner(
        self,
        query: KALQuery,
        tenant_id: str | None,
    ) -> list[KALTriple]:
        tenant_uuid = _ensure_tenant(tenant_id)

        subject_names = _collect_filter(query.patterns, "subject")
        predicate_names = _collect_filter(query.patterns, "predicate")
        object_names = _collect_filter(query.patterns, "object")

        async with self._session_factory() as session:
            # If a free-text query is supplied, resolve it to a set of
            # matching entity IDs FIRST, then materialize the triples
            # that touch those entities. The two queries share a session
            # so we don't pay an extra connection round-trip.
            #
            # The entity pre-fetch uses a FIXED generous cap independent
            # of the caller's triple ``limit`` — an entity can sit on many
            # triples, so capping at ``limit`` could silently drop matches
            # (e.g. caller asks for 500 triples; FTS matches 150 entities;
            # only 100 entities are pulled → far fewer than 500 returned).
            # 5000 is high enough to cover realistic dev workloads and the
            # GIN-indexed FTS scan stays cheap.
            node_id_in_filter: list[uuid.UUID] | None = None
            if query.text_search:
                node_id_in_filter = await self._kg_store.text_search_node_ids(
                    session,
                    tenant_uuid,
                    query.text_search,
                    limit=_TEXT_SEARCH_ENTITY_CAP,
                )
            return await self._kg_store.query_triples(
                session,
                tenant_uuid,
                subject_names=subject_names,
                predicate_names=predicate_names,
                object_names=object_names,
                limit=query.limit,
                offset=query.offset,
                min_confidence=query.min_confidence,
                verified_only=not query.include_unverified,
                node_id_in_filter=node_id_in_filter,
                include_revoked=query.include_revoked,
            )

    async def get_node(
        self,
        node_id: str,
        tenant_id: str | None = None,
    ) -> KALNode | None:
        """Fetch a single node by ID.

        Returns ``None`` (not raises) when the entity isn't found. A bad-shape
        ``node_id`` (not a UUID) raises ``ValueError`` — the debug router
        catches that and turns it into a 400.

        Timeout: applies the adapter's default (constructor / Settings)
        since the Protocol does not pass a ``KALQuery``.
        """
        return await self._with_timeout(
            None, self._get_node_inner(node_id, tenant_id)
        )

    async def _get_node_inner(
        self, node_id: str, tenant_id: str | None
    ) -> KALNode | None:
        tenant_uuid = _ensure_tenant(tenant_id)
        entity_uuid = uuid.UUID(node_id)
        async with self._session_factory() as session:
            return await self._kg_store.get_node(
                session, tenant_uuid, entity_uuid
            )

    async def query_adjacent_triples(
        self,
        node_id: str,
        predicate: str | None = None,
        direction: str = "outgoing",
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[KALTriple]:
        """Query triples adjacent to ``node_id`` via direct ID match.

        ``direction`` is one of ``"outgoing"`` / ``"incoming"`` / ``"both"``;
        anything else raises ``ValueError`` from the underlying store —
        the debug router turns that into a 400.

        Timeout: adapter default (the Protocol does not pass a query).
        """
        return await self._with_timeout(
            None,
            self._query_adjacent_triples_inner(
                node_id,
                predicate=predicate,
                direction=direction,
                tenant_id=tenant_id,
                limit=limit,
            ),
        )

    async def _query_adjacent_triples_inner(
        self,
        node_id: str,
        *,
        predicate: str | None,
        direction: str,
        tenant_id: str | None,
        limit: int,
    ) -> list[KALTriple]:
        tenant_uuid = _ensure_tenant(tenant_id)
        node_uuid = uuid.UUID(node_id)
        # KAL Protocol types ``direction`` as ``str``; KGStore narrows
        # to ``AdjacencyDirection`` (``Literal["outgoing", "incoming",
        # "both"]``). The underlying repo helper raises ``ValueError``
        # on anything else at runtime — the debug router turns that
        # into a 400. Cast here to satisfy mypy at the typed boundary.
        async with self._session_factory() as session:
            return await self._kg_store.query_adjacent_triples(
                session,
                tenant_uuid,
                node_uuid,
                predicate=predicate,
                direction=cast(AdjacencyDirection, direction),
                limit=limit,
            )

    # --- Write operations ---

    async def insert_triples(
        self,
        triples: list[KALTriple],
        tenant_id: str | None = None,
    ) -> int:
        """Persist KALTriples via the configured ``KGStore``.

        The store handles dedup (within-tenant) + entity resolution +
        embedding + persistence; this adapter wraps the call in a
        ``session.begin()`` transaction and follows it with foreign
        external-ID registration (federation plumbing that stays on
        the adapter — the store doesn't know about cross-adapter ID
        spaces). Returns the count of newly created rows per KAL spec
        §4.2 (not the IDs; callers re-query if they need them).

        Atomicity: the whole batch runs inside one ``session.begin()``
        transaction. Any exception (FK violation, constraint failure,
        etc.) rolls the entire batch back — partial inserts never
        persist. This is what ``supports_atomic_writes=True`` advertises.

        ``KALTriple.id`` on input is ignored: the store autogenerates
        primary keys, and accepting caller-provided IDs would risk
        collisions with existing rows.
        """
        tenant_uuid = _ensure_tenant(tenant_id)
        if not triples:
            return 0

        self._check_batch_size(len(triples), "insert")

        async def _persist() -> int:
            async with self._session_factory() as session, session.begin():
                created = await self._kg_store.insert_triples(
                    session, tenant_uuid, triples
                )
                await self._register_foreign_external_ids(
                    session, triples, tenant_uuid
                )
            return created

        # §5.4: writes don't take ``KALQuery``, so the per-call slot is
        # unset — the timeout flows from the ctor knob, then the
        # settings env default (resolved by ``_with_timeout``). Wrapping
        # the whole session-begin block matches the read-side contract:
        # a slow connection acquire OR slow SQL both surface as
        # ``asyncio.TimeoutError``.
        return await self._with_timeout(None, _persist())

    async def _register_foreign_external_ids(
        self,
        session: AsyncSession,
        triples: list[KALTriple],
        tenant_uuid: uuid.UUID,
    ) -> None:
        """Map foreign-adapter external_ids to the resolved internal UUIDs.

        A triple's ``subject.id`` / ``object.id`` are "foreign" external
        IDs when ``triple.adapter`` is set and differs from this
        adapter's name — that's the federated-write case where another
        adapter assembled the triple in its own ID space. We register
        ``(source_adapter, external_id) → internal_id`` so a follow-up
        call from the same source can resolve back to the canonical
        internal entity without re-running normalized-name matching.

        Registry rows land in the same session as the store's
        ``insert_triples`` call, so the insert + registration either
        commit together or roll back together. With no foreign triples
        in the batch, the loop below iterates zero times and emits zero
        SQL.
        """
        foreign_entities = self._group_foreign_entities_by_adapter(triples)
        for source_adapter, ext_id_to_name in foreign_entities.items():
            mappings = await self._resolve_foreign_mappings(
                session, ext_id_to_name, tenant_uuid
            )
            if mappings:
                await bulk_register(
                    session,
                    tenant_id=tenant_uuid,
                    source_adapter=source_adapter,
                    mappings=mappings,
                    resource_type="entity",
                )

    def _group_foreign_entities_by_adapter(
        self, triples: list[KALTriple]
    ) -> dict[str, dict[str, str]]:
        """Bucket foreign-triple subjects + objects by their source adapter.

        Returns ``{source_adapter: {external_id: entity_name}}``.
        Per-adapter grouping lets ``_register_foreign_external_ids``
        issue one ``bulk_register`` call per source rather than one
        per triple.

        Last-write-wins on ``external_id`` collisions within one
        adapter's bucket: the federation contract assumes the same
        ``(adapter, external_id)`` always names the same logical
        entity, so a name disagreement across triples in one batch is
        a caller bug — we don't merge or warn here.
        """
        foreign_entities: dict[str, dict[str, str]] = {}
        for triple in triples:
            source_adapter = triple.adapter
            if source_adapter is None or source_adapter == self._name:
                continue
            bucket = foreign_entities.setdefault(source_adapter, {})
            bucket[triple.subject.id] = triple.subject.name
            if triple.object is not None:
                bucket[triple.object.id] = triple.object.name
        return foreign_entities

    async def _resolve_foreign_mappings(
        self,
        session: AsyncSession,
        ext_id_to_name: dict[str, str],
        tenant_uuid: uuid.UUID,
    ) -> list[tuple[str, uuid.UUID]]:
        """Resolve each foreign external_id's normalized entity name to
        the internal UUID written through ``insert_triples``.

        Normalization is adapter-side (paper §7: the adapter owns the
        KAL surface contracts including name canonicalization); the
        normalized-name → internal-id lookup is the store's job and
        flows through ``KGStore.resolve_node_ids_by_normalized_name``.
        Single bulk lookup per adapter rather than one SELECT per
        external_id — matters for batches in the hundreds. Returns
        ``(external_id, internal_id)`` pairs ready for
        ``bulk_register``; entries whose normalized name didn't
        resolve (e.g. validation rejected the name) are silently
        dropped.

        Precondition on the caller: ``KALNode.name`` is the canonical
        human-readable name, not the foreign adapter's opaque ID
        (URI / UUID / int). Passing ``name == id`` pollutes
        ``normalized_name`` and breaks dedup-by-normalized-name
        across adapters.
        """
        if not ext_id_to_name:
            return []
        # Normalize once per entity_name; collapse synonyms that map to
        # the same canonical form by keeping a list of external_ids
        # per normalized_name (two foreign IDs may name the same entity).
        normalized_to_ext_ids: dict[str, list[str]] = {}
        for external_id, entity_name in ext_id_to_name.items():
            key = normalize_entity_name(entity_name)
            normalized_to_ext_ids.setdefault(key, []).append(external_id)

        name_to_id = await self._kg_store.resolve_node_ids_by_normalized_name(
            session, tenant_uuid, list(normalized_to_ext_ids.keys())
        )
        mappings: list[tuple[str, uuid.UUID]] = []
        for normalized, internal_id in name_to_id.items():
            for external_id in normalized_to_ext_ids[normalized]:
                mappings.append((external_id, internal_id))
        return mappings

    async def update_verification(
        self,
        triple_id: str,
        verification: VerificationMetadata,
        tenant_id: str | None = None,
    ) -> bool:
        """Write verification metadata onto a triple.

        Per plan task 4: only the non-None fields on ``verification`` are
        written; passing ``VerificationMetadata(confidence=0.85)`` leaves
        the other typed columns untouched. Returns ``False`` (no rows
        changed) when the triple_id doesn't exist for the tenant — never
        raises for "missing", per the KAL Protocol's ``-> bool`` contract.

        Edge case: when **every** field on ``verification`` is ``None``
        (so there's nothing to write), the adapter still issues a cheap
        SELECT to confirm the row exists and returns that boolean — i.e.
        an all-None update against a present row returns ``True`` even
        though no columns are updated. This keeps the return value
        truthful about whether the addressed row exists.

        v0.5.0 §5.8: ``verification.verifier`` MUST be set. "Anonymous"
        verification has no meaningful place on the §3.5 dual-axis
        federation model — without a verifier identity the vertical
        axis collapses to a flat overwrite and §3.5 stops being honest.
        ``None`` raises ``ValueError`` at the adapter boundary; empty
        string is rejected one layer earlier by the Pydantic
        ``min_length=1`` on ``VerificationMetadata.verifier``. The
        store layer stays naive so direct callers (and the legacy
        confidence-only conformance path) keep working.
        """
        tenant_uuid = _ensure_tenant(tenant_id)
        triple_uuid = uuid.UUID(triple_id)
        if verification.verifier is None:
            raise ValueError(
                "VerificationMetadata.verifier is required "
                "(§5.8 vertical-axis identity); pass a non-empty "
                "verifier name."
            )

        async def _update() -> bool:
            async with self._session_factory() as session, session.begin():
                return await self._kg_store.update_verification(
                    session, tenant_uuid, triple_uuid, verification
                )

        # §5.4: see ``insert_triples`` for the timeout-flow rationale.
        return await self._with_timeout(None, _update())

    async def delete_triples(
        self,
        triple_ids: list[str],
        tenant_id: str | None = None,
    ) -> int:
        """Delete triples by ID. Returns the actual rowcount.

        Unknown IDs are silently ignored — passing ``[real_id, fake_id]``
        deletes ``real_id`` and returns ``1`` (not 2). The Protocol's
        ``-> int`` contract is "count actually deleted", not "count
        requested". Tenant filtering is enforced by the store.

        Bad-shape ``triple_ids`` (non-UUID strings) raises ``ValueError``
        → 400 at the debug router. Empty list is a no-op returning 0.
        """
        tenant_uuid = _ensure_tenant(tenant_id)
        if not triple_ids:
            return 0

        self._check_batch_size(len(triple_ids), "delete")

        # Parse all IDs up front so a single bad ID fails fast before
        # we open a session / start a transaction. Surface the original
        # bad token in the error message — the bare UUID parse error
        # ("badly formed hexadecimal UUID string") doesn't say which
        # input failed.
        triple_uuids: list[uuid.UUID] = []
        for raw in triple_ids:
            try:
                triple_uuids.append(uuid.UUID(raw))
            except ValueError as exc:
                raise ValueError(
                    f"delete_triples: bad triple_id {raw!r} — {exc}"
                ) from exc

        async def _delete() -> int:
            async with self._session_factory() as session, session.begin():
                return await self._kg_store.delete_triples(
                    session, tenant_uuid, triple_uuids
                )

        # §5.4: see ``insert_triples`` for the timeout-flow rationale.
        return await self._with_timeout(None, _delete())

    async def quarantine_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        """Soft-revoke every live triple with ``provenance.source == source``.
        Returns the count newly quarantined. Tenant-scoped by the store."""
        tenant_uuid = _ensure_tenant(tenant_id)

        async def _quarantine() -> int:
            async with self._session_factory() as session, session.begin():
                return await self._kg_store.quarantine_triples_by_source(
                    session, tenant_uuid, source
                )

        return await self._with_timeout(None, _quarantine())

    async def restore_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        """Reverse a quarantine for ``source``. Returns the count restored."""
        tenant_uuid = _ensure_tenant(tenant_id)

        async def _restore() -> int:
            async with self._session_factory() as session, session.begin():
                return await self._kg_store.restore_triples_by_source(
                    session, tenant_uuid, source
                )

        return await self._with_timeout(None, _restore())

    # --- Vector search ---

    async def search_similar_nodes(
        self,
        query_vector: list[float],
        tenant_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[KALNode, float]]:
        """Rank nodes by vector distance per the configured store.

        Returns ``(KALNode, distance)`` pairs in ascending distance order
        (lower = more similar). The store decides its own distance
        metric (knowly's default: pgvector cosine over
        ``kg.entities.embedding``); the adapter just forwards the
        query vector and limit.

        The returned ``KALNode``s carry ``id`` + ``name`` only — vector
        search returns a candidate list; callers ``get_node`` to
        hydrate ``node_types`` / ``normalized_name`` / ``properties``
        when needed.

        Validates the query vector's dimension against
        ``self._kg_store.vector_dim`` before opening a session, so
        callers get a clear ``ValueError`` instead of a backend error
        deep inside the impl. Tenant filtering is enforced by the
        store.

        Timeout: adapter default (the Protocol does not pass a query).
        """
        # Dim check up front so a bad vector raises immediately without
        # paying for ``wait_for`` task scheduling. The expected dim
        # comes from the store (each impl advertises its embedding
        # space); the adapter validates at this KAL boundary so
        # callers get a clear ``ValueError`` rather than a pgvector
        # backend error deep inside the impl.
        expected_dim = self._kg_store.vector_dim
        if len(query_vector) != expected_dim:
            raise ValueError(
                f"query_vector must be {expected_dim}-d (matches "
                f"the store's embedding space); got {len(query_vector)}-d"
            )
        return await self._with_timeout(
            None,
            self._search_similar_nodes_inner(
                query_vector, tenant_id=tenant_id, limit=limit
            ),
        )

    async def _search_similar_nodes_inner(
        self,
        query_vector: list[float],
        *,
        tenant_id: str | None,
        limit: int,
    ) -> list[tuple[KALNode, float]]:
        tenant_uuid = _ensure_tenant(tenant_id)
        async with self._session_factory() as session:
            return await self._kg_store.search_similar_nodes(
                session, tenant_uuid, query_vector, limit=limit
            )


def _collect_filter(
    patterns: list[KALTriplePattern], field: _PatternField
) -> list[str] | None:
    """Translate a list of patterns into a flat ``list[str]`` filter.

    Returns ``None`` when no pattern constrains the field (match anything).

    Semantics note: KAL spec sec 3.3 calls patterns an "AND conjunction".
    For a single pattern this is unambiguous. For multiple patterns that
    constrain the **same** field (e.g. ``[P(subject="A"), P(subject="B")]``)
    a strict AND reading would yield zero matches (a triple has one subject;
    cannot equal "A" AND "B"). The realistic interpretation — and what
    callers want in practice — is set membership: "subject in {A, B}".
    We compile to ``WHERE subj.name IN (...)``, i.e. OR-within-same-field
    while still ANDing across different fields. Document this divergence
    here; if a future consumer needs strict AND semantics for multi-pattern,
    revisit (and probably also revisit the spec).
    """
    values = [getattr(p, field) for p in patterns if getattr(p, field) is not None]
    return values if values else None
