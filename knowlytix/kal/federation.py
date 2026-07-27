"""Federation router for KAL adapters.

Per KAL spec v1.4.1 §5. Story 8 ships the single-adapter passthrough +
multi-adapter fan-out with content-hash dedup + capability filtering +
write routing. Story 9 hardens timeout + partial-failure handling: each
adapter runs under ``asyncio.wait_for``; exceptions are classified into
``KALQueryResult.errors`` instead of cancelling sibling fan-outs.

``FederationRouter`` is **not** a module-level singleton — the spec keeps
it per-app-instance so tests can spin up a fresh router with its own
registry. ``main.py``'s lifespan stashes one on ``app.state``.

Federation behaviors implemented here (per spec §5 "Federation Behaviors"
table):

- **Fan-out**: ``query_triples`` calls every applicable adapter
  concurrently via ``asyncio.gather(..., return_exceptions=True)`` so a
  single adapter failure no longer cancels its siblings (Story 9).
- **Adapter stamping**: every returned ``KALTriple`` has its
  ``adapter`` field set to the originating adapter's ``name``.
- **Dedup**: by ``KALTriple.content_hash`` (handles literal + node
  objects per spec v1.4.1 E1 fix).
- **Capability filtering**: vector search only routes to
  ``supports_vector_search=True`` adapters; writes only to
  ``can_write=True``.
- **Write routing**: ``insert_triples`` routes each triple to the
  adapter named in ``KALTriple.adapter``; falls back to the first
  registered writable adapter when the field is ``None``.
- **Timeout** (Story 9): each adapter call runs under
  ``asyncio.wait_for(call, timeout=(query.timeout_ms or default)/1000)``
  — note the v1.4.1 E2 parens fix; without them an explicit
  ``timeout_ms=0`` would short-circuit to the default.
- **Error classification** (Story 9, extended Story 15; v0.4.0 Phase C
  added the transport row): per-adapter exceptions become
  ``AdapterError`` rows in ``KALQueryResult.errors`` so siblings still
  return their triples. ``TimeoutError`` → ``"timeout"``;
  ``AdapterAuthError`` → ``"auth"``; ``AdapterConnectionError`` →
  ``"connection"`` (KAL's domain class); stdlib ``ConnectionError`` /
  ``OSError`` → ``"transport"`` (raw transport leakage from an
  adapter, distinct from the KAL-domain ``"connection"`` tag above);
  ``CapabilityNotSupportedError`` → ``"capability"``; everything else
  → ``"internal"`` with the exception type in the message (full
  traceback only goes to logs).

Deferred to a later story:
- circuit-breaker semantics (failure-rate-based adapter eviction; the
  ``"circuit_open"`` ``AdapterError.error_type`` value is reserved for
  this story but not emitted today)
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum

from knowlytix.kal.errors import (
    ADAPTER_ERROR_MESSAGE_MAX_CHARS,
    AdapterAuthError,
    AdapterConnectionError,
    AdapterNotFoundError,
    AdapterWriteError,
    CapabilityNotSupportedError,
    truncate_error_message,
)
from knowlytix.kal.protocol import KnowledgeAdapter
from knowlytix.kal.registry import AdapterRegistry
from knowlytix.kal.tenant_index import TenantIndex
from knowlytix.kal.types import (
    AdapterError,
    AdapterErrorType,
    KALNode,
    KALQuery,
    KALQueryResult,
    KALTriple,
    VerificationMetadata,
)

logger = logging.getLogger(__name__)


class ConflictStrategy(StrEnum):
    """Dedup tie-break for triples with the same ``(content_hash, verifier)``
    bucket across adapters, per KAL spec §5.6 / §5.8.

    Dedup is keyed on ``(content_hash, verifier)`` so triples from
    different verifiers (vertical axis, §3.5) never collide regardless
    of strategy. The strategies below tie-break only WITHIN a single
    ``(content_hash, verifier)`` bucket.

    ``ALL`` keeps the first-registered adapter's copy within each
    bucket (v1 behaviour, applies per-bucket under the vertical-axis
    split). ``HIGHEST_CONFIDENCE`` compares ``verification.confidence``
    among same-verifier candidates; ``MOST_RECENT`` compares
    ``provenance.extracted_at`` the same way. ``None`` loses to any
    set value; with both unset, falls back to first-registered.
    """

    ALL = "all"
    HIGHEST_CONFIDENCE = "highest_confidence"
    MOST_RECENT = "most_recent"


class FederationRouter:
    """Fan queries out to registered adapters and merge results.

    ``default_timeout_ms`` is applied per-adapter when the caller's
    ``KALQuery.timeout_ms`` is ``None`` or unset.
    """

    def __init__(
        self,
        registry: AdapterRegistry,
        *,
        conflict_strategy: ConflictStrategy = ConflictStrategy.ALL,
        default_timeout_ms: int = 5000,
        tenant_index: TenantIndex | None = None,
    ) -> None:
        # ``tenant_index`` defaults to ``None`` so existing call sites
        # (and tests built against the pre-Story-8 signature) keep
        # working unchanged: with no index, ``_resolve_targets``
        # returns the full ``all_adapters()`` list as before. Wiring
        # an index in flips on multi-tenant scoping at the routing
        # boundary — see ``_resolve_targets`` for the filter point.
        self._registry = registry
        self._conflict_strategy = conflict_strategy
        self._default_timeout_ms = default_timeout_ms
        self._tenant_index = tenant_index

    @property
    def default_timeout_ms(self) -> int:
        return self._default_timeout_ms

    @property
    def conflict_strategy(self) -> ConflictStrategy:
        return self._conflict_strategy

    def with_conflict_strategy(
        self, strategy: ConflictStrategy
    ) -> FederationRouter:
        """Return a new router sharing this one's registry +
        tenant_index, but with a different ``conflict_strategy``.

        Useful when a caller needs to compare strategies on the same
        query without mutating the lifespan-bound router. The federate
        debug endpoint uses this to apply a per-request strategy
        override; demos use it to render side-by-side strategy
        outputs over one shared registry. The original router is
        unchanged.
        """
        return FederationRouter(
            self._registry,
            conflict_strategy=strategy,
            default_timeout_ms=self._default_timeout_ms,
            tenant_index=self._tenant_index,
        )

    # --- Read operations ---

    async def query_triples(
        self,
        query: KALQuery,
        tenant_id: str | None = None,
        adapters: list[str] | None = None,
    ) -> KALQueryResult:
        """Fan ``query_triples`` out to registered adapters and merge.

        ``adapters`` (optional) names a subset to query; ``None`` means
        all registered. Each returned ``KALTriple`` is stamped with its
        originating adapter's name and deduplicated across adapters via
        ``KALTriple.content_hash`` — two adapters returning the
        semantically-identical triple collapse to one row (the spec's
        ``ConflictStrategy.ALL`` default is "return all matches", but
        identical content_hash is by definition not a conflict).

        **Dedup tie-break**: when two adapters return triples with the
        same ``content_hash``, the **first-registered adapter's copy
        wins** (``dict.setdefault`` keeps the first occurrence and
        ``_resolve_targets`` preserves registry registration order).
        This makes the merged result deterministic across runs — callers
        relying on a specific adapter's metadata for a duplicated triple
        can rely on registration order.

        **Per-adapter timeout** (Story 9): each adapter call runs under
        ``asyncio.wait_for((query.timeout_ms or default_timeout_ms)/1000)``.
        Timeouts and per-adapter exceptions become ``AdapterError`` rows
        in ``KALQueryResult.errors``; sibling adapters' results are
        unaffected (``return_exceptions=True`` on the gather).
        """
        targets = self._resolve_targets(adapters, tenant_id=tenant_id)
        if not targets:
            return KALQueryResult(triples=[])

        # v0.5.0 §5.8: auto-flip ``include_all_verifications=True`` for
        # ``ConflictStrategy.ALL`` so the vertical-axis dedup keyed on
        # ``(content_hash, verifier)`` actually surfaces multi-verifier
        # rows. ``HIGHEST_CONFIDENCE`` / ``MOST_RECENT`` already pick
        # one row per triple by their own ordering, so they leave the
        # flag off and consume the cache-row shape. Direct adapter
        # callers (not via this router) keep the v0.4.0 default of
        # one-row-per-triple regardless.
        dispatch_query = query
        if self._conflict_strategy == ConflictStrategy.ALL and not (
            query.include_all_verifications
        ):
            dispatch_query = query.model_copy(
                update={"include_all_verifications": True}
            )

        timeout_s = self._resolve_timeout_seconds(query.timeout_ms)
        # User-facing latency = fan-out + slowest adapter (subject to
        # the per-adapter timeout) + merge cost. ``time.monotonic`` is
        # the right clock here — it ignores wall-clock adjustments
        # (NTP, DST) that ``time.time`` is vulnerable to.
        start = time.monotonic()
        outcomes = await asyncio.gather(
            *(
                self._run_with_timeout(
                    a.query_triples(dispatch_query, tenant_id=tenant_id),
                    timeout_s,
                )
                for a in targets
            ),
            return_exceptions=True,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return self._merge_query_outcomes(targets, outcomes, elapsed_ms)

    async def search_similar_nodes(
        self,
        query_vector: list[float],
        tenant_id: str | None = None,
        adapters: list[str] | None = None,
        limit: int = 10,
    ) -> list[tuple[KALNode, float]]:
        """Vector-similarity search across adapters with
        ``supports_vector_search=True``.

        Each adapter contributes up to ``limit`` candidates; the final
        list is sorted by distance ascending and capped at ``limit``.
        Capability mismatch is filtered BEFORE dispatch per spec §5.1
        — calling an adapter without the capability would be a router
        bug, not a runtime condition. With no registered adapter
        supporting vector search, returns ``[]`` (graceful degradation
        per plan task 6).

        §5.6 partial-failure contract: per-adapter failures and timeouts
        are isolated — a slow adapter times out under the router's
        ``default_timeout_ms`` budget; a failing adapter is
        logged-and-skipped — siblings continue contributing. The method
        returns a plain list (no ``errors`` slot), so failures do not
        appear in the return value; consult the logs for adapter-level
        diagnostics. The asymmetry with ``query_triples`` is the absence
        of a structured ``errors`` slot — search returns are
        distance-ranked candidates, not a result envelope.
        """
        targets = [
            a for a in self._resolve_targets(adapters, tenant_id=tenant_id)
            if a.capabilities.supports_vector_search
        ]
        if not targets:
            return []

        timeout_s = self._resolve_timeout_seconds(None)
        outcomes = await asyncio.gather(
            *(
                asyncio.wait_for(
                    a.search_similar_nodes(
                        query_vector, tenant_id=tenant_id, limit=limit
                    ),
                    timeout=timeout_s,
                )
                for a in targets
            ),
            return_exceptions=True,
        )

        success_targets, success_results = self._partition_search_outcomes(
            targets, outcomes
        )
        return self._merge_search_results(success_targets, success_results, limit)

    @staticmethod
    def _partition_search_outcomes(
        targets: list[KnowledgeAdapter],
        outcomes: list[list[tuple[KALNode, float]] | BaseException],
    ) -> tuple[list[KnowledgeAdapter], list[list[tuple[KALNode, float]]]]:
        """Split ``outcomes`` into (success_targets, success_results).

        Two failure-class branches, both mirroring ``_classify_error`` /
        ``_merge_query_outcomes`` on the ``query_triples`` side:

        - ``Exception`` subclasses — adapter-side errors (timeout, auth,
          connection, unclassified internal): log with full traceback
          via ``logger.exception`` and drop the adapter from the merge.
          The full stack still survives in logs so a latent adapter
          bug (e.g. distance scoring exception) doesn't silently
          vanish behind an ``INFO``-filtered logger.
        - ``BaseException``-not-``Exception`` (``KeyboardInterrupt``,
          ``SystemExit``, ``asyncio.CancelledError``): re-raise so the
          calling task / process can respond. ``gather(return_exceptions=True)``
          *does* aggregate these per Python's task-exception contract,
          so without the re-raise we'd silently swallow shutdown
          signals — a correctness bug, not just an observability gap.

        The BaseException check runs as a pre-pass so the abort
        decision is made BEFORE any per-adapter allocation work — and
        so the contract is uniformly "any BaseException = drop the
        whole call." Partial successes are intentionally discarded
        on shutdown: the caller is no longer interested in distance-
        ranked candidates if its task is being cancelled.
        """
        # Pre-pass: any shutdown-signal exception aborts the whole call.
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and not isinstance(
                outcome, Exception
            ):
                raise outcome

        # Main pass: partition adapter-side ``Exception`` (log+skip)
        # vs successful list-of-pairs results. The inner BaseException
        # re-raise below is unreachable given the pre-pass — kept as a
        # defensive narrow for mypy and as a safety net if the pre-pass
        # invariant ever changes (better to raise loudly than feed a
        # raw exception into ``_merge_search_results``).
        success_targets: list[KnowledgeAdapter] = []
        success_results: list[list[tuple[KALNode, float]]] = []
        for adapter, outcome in zip(targets, outcomes, strict=True):
            if isinstance(outcome, Exception):
                logger.exception(
                    "adapter %r failed during search_similar_nodes",
                    adapter.name,
                    exc_info=outcome,
                )
                continue
            if isinstance(outcome, BaseException):
                raise outcome
            success_targets.append(adapter)
            success_results.append(outcome)
        return success_targets, success_results

    async def query_adjacent_triples(
        self,
        node_id: str,
        *,
        predicate: str | None,
        direction: str,
        limit: int,
        tenant_id: str | None = None,
        adapters: list[str] | None = None,
    ) -> KALQueryResult:
        """Fan-out adjacency walk across registered adapters.

        Returns ``KALQueryResult`` (envelope with ``errors``), NOT bare
        ``list[KALTriple]``: the router is the partial-failure-
        surfacing layer, and the v0.6.0 MCP `query_adjacent_triples`
        tool needs ``errors`` visibility for the LLM caller. Adapter-
        side bare-list results are wrapped into a single-result
        ``KALQueryResult`` shim before feeding ``_merge_query_outcomes``
        so the existing dedup / partial-failure / query-time-stamping
        machinery is reused without forking.

        Dedup keys on ``(content_hash, verifier)`` (same as
        ``query_triples``) so the §5.8 vertical axis carries through
        adjacency walks identically. ``ConflictStrategy.ALL`` is the
        default tie-break.

        v0.6.0 Phase A: added so the §3.6 MCP tool surface can wrap
        the router uniformly. ``KnowledgeAdapter.query_adjacent_triples``
        keeps its existing ``-> list[KALTriple]`` Protocol shape
        unchanged — only the router's facade is wider.
        """
        targets = self._resolve_targets(adapters, tenant_id=tenant_id)
        if not targets:
            return KALQueryResult(triples=[])

        timeout_s = self._resolve_timeout_seconds(None)
        start = time.monotonic()
        # ``_run_with_timeout`` is typed as ``-> KALQueryResult`` (only
        # matches ``query_triples``'s adapter shape). Adjacency adapters
        # return ``list[KALTriple]``, so call ``asyncio.wait_for``
        # directly — same pattern as ``search_similar_nodes`` and
        # ``get_node``.
        raw_outcomes = await asyncio.gather(
            *(
                asyncio.wait_for(
                    a.query_adjacent_triples(
                        node_id,
                        predicate=predicate,
                        direction=direction,
                        tenant_id=tenant_id,
                        limit=limit,
                    ),
                    timeout=timeout_s,
                )
                for a in targets
            ),
            return_exceptions=True,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Wrap each adapter's ``list[KALTriple]`` into a single-result
        # ``KALQueryResult`` so we can reuse ``_merge_query_outcomes``
        # (which expects ``KALQueryResult | BaseException`` per slot).
        # Exceptions pass through unchanged for the merge helper's
        # partition step to classify.
        outcomes: list[KALQueryResult | BaseException] = [
            outcome
            if isinstance(outcome, BaseException)
            else KALQueryResult(triples=outcome)
            for outcome in raw_outcomes
        ]
        return self._merge_query_outcomes(targets, outcomes, elapsed_ms)

    async def get_node(
        self,
        node_id: str,
        tenant_id: str | None = None,
        adapters: list[str] | None = None,
    ) -> KALNode | None:
        """Fan-out single-node lookup across registered adapters.

        Returns the **first non-None match** in ``_resolve_targets``
        order (registration order, tenant-filtered). Adapter failures
        are logged-and-skipped per §5.6 — same shape as
        ``search_similar_nodes``: a slow / broken adapter does not
        block a sibling's successful match. No ``KALQueryResult``
        envelope: the return is bare ``KALNode | None`` matching the
        adapter Protocol's own shape; the v0.6.0 MCP `get_node` tool
        surfaces the source adapter via ``KALNode.adapter`` (already
        present on the returned node from the adapter's own
        materializer pipeline).

        v0.6.0 Phase A: added so the §3.6 MCP tool surface can wrap
        the router uniformly. Direct callers that previously hit
        ``adapter.get_node`` keep working — this method is additive.

        ``adapters`` (optional): restrict fan-out to a named subset;
        ``None`` = all registered.

        First-non-None tie-break vs. merge: a logical entity could
        exist across multiple adapters with diverging metadata.
        First-wins matches the ``ConflictStrategy.ALL`` triple-side
        first-wins; callers needing cross-adapter merge use
        ``KALNode.merged_into`` to chase the canonical ID rather than
        relying on a router-side merge.
        """
        targets = self._resolve_targets(adapters, tenant_id=tenant_id)
        if not targets:
            return None

        timeout_s = self._resolve_timeout_seconds(None)
        outcomes = await asyncio.gather(
            *(
                asyncio.wait_for(
                    a.get_node(node_id, tenant_id=tenant_id),
                    timeout=timeout_s,
                )
                for a in targets
            ),
            return_exceptions=True,
        )

        # BaseException pre-pass — re-raise shutdown signals; same
        # safety net as ``_partition_search_outcomes``.
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and not isinstance(
                outcome, Exception
            ):
                raise outcome

        # Walk in registration order; return the first non-None
        # match. Adapter-side exceptions are logged and skipped.
        # Stamp the source adapter onto ``KALNode.adapter`` (mirrors
        # ``_merge_query_triples``'s triple-side stamping) — the v0.6.0
        # MCP tool surfaces this field so the LLM can see which
        # adapter's view it received.
        for adapter, outcome in zip(targets, outcomes, strict=True):
            if isinstance(outcome, Exception):
                logger.exception(
                    "adapter %r failed during get_node",
                    adapter.name,
                    exc_info=outcome,
                )
                continue
            if isinstance(outcome, BaseException):
                # Defensive — pre-pass should have caught this.
                raise outcome
            if outcome is not None:
                return outcome.model_copy(update={"adapter": adapter.name})
        return None

    # --- Write operations ---

    async def insert_triples(
        self,
        triples: list[KALTriple],
        tenant_id: str | None = None,
    ) -> int:
        """Route each triple to the adapter named on it; fall back to the
        first writable adapter when ``triple.adapter`` is ``None``.

        Returns the *total* count of newly-created rows summed across
        all addressed adapters. Per spec §5.3, writes do NOT silently
        degrade: per-adapter failures are aggregated into an
        ``AdapterWriteError`` carrying both the per-adapter error rows
        and the count of triples successfully persisted across
        surviving adapters. Pre-dispatch validation (unknown target,
        read-only target) raises ``AdapterNotFoundError`` directly
        BEFORE any write is attempted — those are caller errors and
        must not leave partial state.
        """
        if not triples:
            return 0

        writable = self._registry.get_writable()
        if not writable:
            raise AdapterNotFoundError(
                "no writable adapter registered — cannot route insert"
            )
        groups = self._group_triples_by_target(triples, default_target=writable[0])

        # Validate every target adapter before issuing any write — a typo
        # in ``triple.adapter`` raised mid-loop would leave earlier batches
        # persisted while a later one fails, violating spec §5.3.
        resolved: list[tuple[KnowledgeAdapter, list[KALTriple]]] = []
        for target_name, batch in groups.items():
            target = self._registry.get(target_name)
            if not target.capabilities.can_write:
                raise AdapterNotFoundError(
                    f"adapter {target_name!r} is not writable"
                )
            resolved.append((target, batch))

        total = 0
        errors: list[AdapterError] = []
        for target, batch in resolved:
            try:
                total += await target.insert_triples(batch, tenant_id=tenant_id)
            except Exception as exc:
                errors.append(self._classify_error(target.name, exc))

        if errors:
            raise AdapterWriteError(
                adapter_errors=errors, successful_count=total
            )
        return total

    async def delete_triples(
        self,
        triple_ids: list[str],
        tenant_id: str | None = None,
    ) -> int:
        """Delete triples by ID against the first writable adapter.

        Unlike ``insert_triples`` (which can route per-triple by
        ``KALTriple.adapter``), ``delete_triples`` receives just the
        IDs — no adapter tag to dispatch on. The simplest correct
        contract: route the whole batch to the first registered
        writable adapter. Multi-adapter deployments with deletes
        spanning backends are out of scope for v1; the §6.4
        external-ID registry would need a per-id adapter lookup
        before this method could disambiguate.

        Returns total rowcount across the addressed adapter. Empty
        list short-circuits with 0 (no writable required). Per spec
        §5.3, adapter exceptions are wrapped in ``AdapterWriteError``
        with a classified per-adapter error so the caller sees the
        same typed shape as ``insert_triples``.
        """
        if not triple_ids:
            return 0

        writable = self._registry.get_writable()
        if not writable:
            raise AdapterNotFoundError(
                "no writable adapter registered — cannot route delete"
            )
        target = writable[0]
        try:
            return await target.delete_triples(
                triple_ids, tenant_id=tenant_id
            )
        except Exception as exc:
            raise AdapterWriteError(
                adapter_errors=[self._classify_error(target.name, exc)],
                successful_count=0,
            ) from exc

    async def quarantine_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        """Soft-revoke (quarantine) every live triple with ``source`` against
        the first writable adapter. Returns the count newly quarantined.

        Same single-writable-target routing as ``delete_triples`` — there's
        no per-triple adapter tag to dispatch on. Adapter exceptions wrap in
        ``AdapterWriteError`` (per spec §5.3)."""
        target = self._single_writable("quarantine")
        try:
            return await target.quarantine_triples_by_source(
                source, tenant_id=tenant_id
            )
        except Exception as exc:
            raise AdapterWriteError(
                adapter_errors=[self._classify_error(target.name, exc)],
                successful_count=0,
            ) from exc

    async def restore_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        """Reverse a quarantine for ``source`` against the first writable
        adapter. Returns the count restored."""
        target = self._single_writable("restore")
        try:
            return await target.restore_triples_by_source(
                source, tenant_id=tenant_id
            )
        except Exception as exc:
            raise AdapterWriteError(
                adapter_errors=[self._classify_error(target.name, exc)],
                successful_count=0,
            ) from exc

    def _single_writable(self, op: str) -> KnowledgeAdapter:
        """First writable adapter, or ``AdapterNotFoundError`` — the shared
        routing target for the by-source revoke/restore ops (no per-triple
        adapter tag to dispatch on, same as ``delete_triples``)."""
        writable = self._registry.get_writable()
        if not writable:
            raise AdapterNotFoundError(
                f"no writable adapter registered — cannot route {op}"
            )
        return writable[0]

    async def update_verification(
        self,
        triple_id: str,
        verification: VerificationMetadata,
        tenant_id: str | None = None,
    ) -> bool:
        """Update verification metadata on a single triple.

        Routes to the first writable adapter — same single-target
        rationale as ``delete_triples``: the caller provides a triple
        ID with no adapter tag to dispatch on. Returns the adapter's
        success boolean directly. ``can_write=False`` adapters are
        excluded by ``get_writable``; ``no writable registered``
        raises ``AdapterNotFoundError`` to surface the misconfiguration
        rather than silently no-op. Adapter exceptions are wrapped in
        ``AdapterWriteError`` per spec §5.3, matching the typed-error
        contract of ``insert_triples`` and ``delete_triples``.
        """
        writable = self._registry.get_writable()
        if not writable:
            raise AdapterNotFoundError(
                "no writable adapter registered — cannot route "
                "update_verification"
            )
        target = writable[0]
        try:
            return await target.update_verification(
                triple_id, verification, tenant_id=tenant_id
            )
        except Exception as exc:
            raise AdapterWriteError(
                adapter_errors=[self._classify_error(target.name, exc)],
                successful_count=0,
            ) from exc

    # --- helpers ---

    def _resolve_targets(
        self,
        adapters: list[str] | None,
        *,
        tenant_id: str | None = None,
    ) -> list[KnowledgeAdapter]:
        """Translate the optional ``adapters`` filter into a list of
        registered ``KnowledgeAdapter`` instances. ``None`` means
        "all registered" — preserves registration order.

        Tenant scoping (Story 8): when a ``TenantIndex`` is wired in
        AND the caller passed a non-``None`` ``tenant_id``, the result
        is further restricted to adapters owned by that tenant plus
        the shared set. ``tenant_id=None`` deliberately bypasses the
        filter — that signals a system-level call (pipeline bridge,
        background job) that legitimately runs across the whole
        registry. An empty string is treated as a tenant id (the
        filter still applies and yields only shared adapters), since
        ``""`` is not a meaningful tenant identifier but the code
        path is identical to a regular lookup. Without a
        ``TenantIndex``, behaviour matches the pre-Story-8 contract;
        existing test suites and the ``postgres-main``-only
        deployment path keep working unchanged.
        """
        candidates = (
            self._registry.all_adapters()
            if adapters is None
            else [self._registry.get(name) for name in adapters]
        )
        if self._tenant_index is None or tenant_id is None:
            return candidates
        allowed = self._tenant_index.adapters_for(tenant_id)
        return [a for a in candidates if a.name in allowed]

    def _resolve_timeout_seconds(self, query_timeout_ms: int | None) -> float:
        """Convert the caller's ms-resolution timeout into seconds.

        The ``(timeout_ms or default)`` parens (v1.4.1 E2 fix) matter:
        a literal ``timeout_ms=0`` would otherwise fall through to the
        default. We treat ``0`` and ``None`` identically as "use the
        configured default" because a literal zero would make
        ``wait_for`` fire instantly — almost certainly a bug, not a
        user intent.
        """
        ms = query_timeout_ms or self._default_timeout_ms
        return ms / 1000

    @staticmethod
    async def _run_with_timeout(
        coro: object, timeout_s: float
    ) -> KALQueryResult:
        """Wrap one adapter call in ``asyncio.wait_for``.

        ``coro`` is typed loosely because ``asyncio.gather`` requires
        awaitables; the static checker can't see across the gather
        boundary anyway. ``wait_for`` cancels the inner task on
        timeout — adapters are expected to cooperate with cancellation
        (raise ``CancelledError`` rather than swallow it).
        """
        return await asyncio.wait_for(coro, timeout=timeout_s)  # type: ignore[arg-type, no-any-return]

    # Cap on AdapterError.message length — adapter exceptions can embed
    # connection URLs with credentials; the full text still goes to logs.
    # Re-exported from ``knowly.kal.errors`` so the value is shared with
    # any other surface that has to truncate (the connection service's
    # handshake-failure classifier in particular).
    _ERROR_MESSAGE_MAX_CHARS: int = ADAPTER_ERROR_MESSAGE_MAX_CHARS

    _ERROR_CLASSIFICATION: tuple[
        tuple[
            type[BaseException] | tuple[type[BaseException], ...],
            AdapterErrorType,
            Callable[[str, BaseException], str],
        ],
        ...,
    ] = (
        (
            (asyncio.TimeoutError, TimeoutError),
            "timeout",
            lambda name, _exc: f"adapter {name!r} exceeded timeout",
        ),
        (AdapterAuthError, "auth", lambda _name, exc: str(exc)),
        (AdapterConnectionError, "connection", lambda _name, exc: str(exc)),
        # Stdlib transport errors leaked from an adapter — distinct from
        # KAL's domain ``AdapterConnectionError`` above (which signals a
        # KAL-aware connection failure with auth context). Per spec §5.3
        # row 3. ``TimeoutError`` is a subclass of ``OSError`` in Python
        # 3.10+, but the timeout row above catches it first so the
        # ``"timeout"`` tag wins. ``AdapterConnectionError`` does NOT
        # inherit from stdlib ``ConnectionError``, so the order above is
        # also safe — KAL's domain class keeps its ``"connection"`` tag.
        (
            (ConnectionError, OSError),
            "transport",
            lambda _name, exc: f"{type(exc).__name__}: {exc}",
        ),
        # Defensive (per spec §5.2): the pre-dispatch capability filter
        # should make this unreachable, so emission here signals either
        # a router bug or an adapter whose declared capabilities drifted.
        (
            CapabilityNotSupportedError,
            "capability",
            lambda _name, exc: str(exc),
        ),
    )

    @classmethod
    def _classify_error(cls, adapter_name: str, exc: BaseException) -> AdapterError:
        """Convert an adapter exception into an ``AdapterError`` row.

        Maps known exception types via ``_ERROR_CLASSIFICATION``; anything
        else becomes ``"internal"`` with the exception's type prefixed on
        the message. Full traceback goes to logs (so debugging keeps
        the stack) but the response only carries type+message (avoids
        leaking adapter internals to API callers).
        """
        for exc_types, error_type, build_message in cls._ERROR_CLASSIFICATION:
            if isinstance(exc, exc_types):
                return AdapterError(
                    adapter=adapter_name,
                    error_type=error_type,
                    message=cls._truncate_message(build_message(adapter_name, exc)),
                )
        logger.exception(
            "adapter %r raised %s during federated query",
            adapter_name,
            type(exc).__name__,
            exc_info=exc,
        )
        return AdapterError(
            adapter=adapter_name,
            error_type="internal",
            message=cls._truncate_message(f"{type(exc).__name__}: {exc}"),
        )

    @classmethod
    def _truncate_message(cls, message: str) -> str:
        # Thin wrapper preserving the prior call-site shape; the
        # actual cap lives in ``knowly.kal.errors`` so the
        # connection-service classifier shares the same length.
        return truncate_error_message(message)

    def _merge_query_outcomes(
        self,
        targets: list[KnowledgeAdapter],
        outcomes: list[KALQueryResult | BaseException],
        elapsed_ms: int,
    ) -> KALQueryResult:
        """Partition outcomes into (results, errors) and stamp the
        wall-clock elapsed onto ``KALQueryResult.query_time_ms``.

        Successful adapters feed the dedup; failing adapters become
        ``AdapterError`` rows. With every adapter erroring, ``triples``
        is empty and ``errors`` carries one row per adapter — the
        caller can tell "no data" from "everything broke".
        """
        success_targets: list[KnowledgeAdapter] = []
        success_results: list[KALQueryResult] = []
        errors: list[AdapterError] = []
        for adapter, outcome in zip(targets, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                errors.append(self._classify_error(adapter.name, outcome))
            else:
                success_targets.append(adapter)
                success_results.append(outcome)
        return KALQueryResult(
            triples=self._merge_query_triples(success_targets, success_results),
            query_time_ms=elapsed_ms,
            errors=errors,
        )

    # Sort-key sentinels for collision tie-break: a triple with no
    # comparable metadata sorts last, so any set value beats an unset one.
    _UNSET_CONFIDENCE: float = float("-inf")
    _UNSET_TIMESTAMP: datetime = datetime.min

    @classmethod
    def _confidence_key(cls, triple: KALTriple) -> float:
        if triple.verification is None or triple.verification.confidence is None:
            return cls._UNSET_CONFIDENCE
        return triple.verification.confidence

    @classmethod
    def _extracted_at_key(cls, triple: KALTriple) -> datetime:
        # KAL spec §3.2 doesn't pin timezone-awareness, so we normalise
        # tz-aware → naive (Python refuses to compare mixed forms).
        if triple.provenance is None or triple.provenance.extracted_at is None:
            return cls._UNSET_TIMESTAMP
        extracted = triple.provenance.extracted_at
        if extracted.tzinfo is not None:
            return extracted.replace(tzinfo=None)
        return extracted

    @classmethod
    def _prefer(
        cls,
        strategy: ConflictStrategy,
        candidate: KALTriple,
        incumbent: KALTriple,
    ) -> bool:
        """Return True iff ``candidate`` should replace ``incumbent`` on a
        ``content_hash`` collision. Strict ``>`` so ties keep first-seen.
        """
        match strategy:
            case ConflictStrategy.HIGHEST_CONFIDENCE:
                return cls._confidence_key(candidate) > cls._confidence_key(
                    incumbent
                )
            case ConflictStrategy.MOST_RECENT:
                return cls._extracted_at_key(candidate) > cls._extracted_at_key(
                    incumbent
                )
            case ConflictStrategy.ALL:
                return False

    def _merge_query_triples(
        self,
        targets: list[KnowledgeAdapter],
        results: list[KALQueryResult],
    ) -> list[KALTriple]:
        """Dedup by ``(content_hash, verifier)`` + stamp adapter-tagged triples.

        ``content_hash`` is content-only (adapter excluded); the dedup key
        extends to ``(content_hash, verifier)`` so the §3.5 / §5.8
        vertical-axis (verdict-plane) contract is satisfied — two adapters
        returning the same logical triple under different
        ``VerificationMetadata.verifier`` identities surface as separate
        rows rather than collapsing. A triple with no verification
        metadata is keyed under ``verifier=None`` and dedups against
        other unverified rows for the same ``content_hash``.

        ``_prefer`` still applies inside a single ``(content_hash,
        verifier)`` bucket so ``HIGHEST_CONFIDENCE`` and ``MOST_RECENT``
        keep their tie-break semantics for same-verifier-different-confidence
        collisions; cross-verifier rows never enter the comparison.

        We dedup the raw triple and ``model_copy``-stamp only the
        survivors — avoiding N allocations when N adapters return the
        same ``(content_hash, verifier)`` pair. Returns the
        deduped+stamped triples list; the wrapping ``KALQueryResult``
        is constructed once by the caller so timing and error fields
        all land in the same allocation.
        """
        seen: dict[tuple[tuple[str, ...], str | None], tuple[KALTriple, str]] = {}
        strategy = self._conflict_strategy
        for adapter, result in zip(targets, results, strict=True):
            for triple in result.triples:
                verifier = (
                    triple.verification.verifier
                    if triple.verification is not None
                    else None
                )
                key = (triple.content_hash, verifier)
                existing = seen.get(key)
                if existing is None or self._prefer(strategy, triple, existing[0]):
                    seen[key] = (triple, adapter.name)
        return [
            triple.model_copy(update={"adapter": name})
            for triple, name in seen.values()
        ]

    @staticmethod
    def _merge_search_results(
        targets: list[KnowledgeAdapter],
        results: list[list[tuple[KALNode, float]]],
        limit: int,
    ) -> list[tuple[KALNode, float]]:
        """Merge + select top ``limit`` similar-node candidates, sorted
        by distance ascending.

        ``heapq.nsmallest`` is O(n log k) vs sort's O(n log n) — matters
        when many adapters each return ``limit`` candidates. The
        ``model_copy`` adapter-stamp happens AFTER selection so we don't
        allocate copies for the (N - limit) candidates that get
        discarded (quadratic-allocation guard).
        """
        # Carry the originating adapter name alongside each pair so we
        # can stamp only the survivors. A 3-tuple keeps the heapq key
        # simple — distance is in a stable position.
        candidates: list[tuple[float, KALNode, str]] = []
        for adapter, pairs in zip(targets, results, strict=True):
            for node, distance in pairs:
                candidates.append((distance, node, adapter.name))
        winners = heapq.nsmallest(limit, candidates, key=lambda c: c[0])
        return [
            (node.model_copy(update={"adapter": name}), distance)
            for distance, node, name in winners
        ]

    @staticmethod
    def _group_triples_by_target(
        triples: list[KALTriple],
        *,
        default_target: KnowledgeAdapter,
    ) -> dict[str, list[KALTriple]]:
        """Bucket triples by destination adapter name so we issue one
        batched call per adapter, preserving each adapter's atomicity
        guarantee. ``triple.adapter=None`` falls back to ``default_target``.
        """
        groups: dict[str, list[KALTriple]] = defaultdict(list)
        for triple in triples:
            target_name = (
                triple.adapter if triple.adapter is not None else default_target.name
            )
            groups[target_name].append(triple)
        return groups
