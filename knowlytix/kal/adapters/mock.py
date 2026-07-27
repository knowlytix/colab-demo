"""MockKnowledgeAdapter — configurable in-memory adapter for tests + dev.

Per Story 9 task 1 of plan_kal_implementation.md. Lives in production
source (not ``tests/``) because the dev-only debug router imports it
to expose ``POST /v1/_kal/_mock/register`` for manual testing of
federation timeout + partial-failure behavior.

Constructor knobs (all optional):
- ``fixed_triples``: what reads return; defaults to ``[]``
- ``sleep_ms``: per-method latency injection — every method awaits
  ``asyncio.sleep(sleep_ms/1000)`` BEFORE running, so a slow mock paired
  with a timed-out federation query genuinely exercises the timeout
  path. Use 100s for "always times out" scenarios.
- ``raise_error``: one of ``"auth" | "connection" | "internal" | None``.
  When set, every method raises after the sleep, simulating an adapter
  that's reachable but failing — the federation router classifies the
  exception into ``KALQueryResult.errors[*].error_type``.
- ``capabilities``: defaults to a full-feature cap set (read + write +
  vector + text + verification metadata) so a mock can substitute for
  any single-feature adapter in tests. Pass a narrower
  ``AdapterCapabilities`` to test capability-filtering paths.

Contract notes (matching the Protocol):
- ``query_triples`` returns triples with ``adapter=None`` regardless of
  what's set on ``fixed_triples`` — adapters DO NOT stamp themselves,
  the router does (per Story 1 + Story 8 contract). The mock
  enforces this by re-issuing ``model_copy(update={"adapter": None})``
  before returning, so a caller who pre-stamped fixtures doesn't
  accidentally test a non-router code path.
- ``insert_triples`` returns ``len(triples)`` (pretends it dedupped
  nothing); the in-memory ``fixed_triples`` is left untouched so
  subsequent reads stay reproducible.
- ``search_similar_nodes`` returns ``(node, _PLACEHOLDER_SIMILARITY_DISTANCE)``
  for the first ``limit`` distinct subject nodes of ``fixed_triples`` —
  the distance is a placeholder; tests that care about ranking should
  construct their own scoring mock.
"""

from __future__ import annotations

import asyncio
from typing import Literal

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
    VerificationMetadata,
)

# Canonical raise_error vocabulary — single source of truth for both
# the static ``_RaiseChoice`` Literal (used by typed callers) and the
# runtime ``VALID_RAISE_ERRORS`` frozenset (used by API-boundary
# validators in ``debug_api.py``). When adding a new failure mode, edit
# this list AND the ``_gate`` branch below; mypy will help catch the
# Literal mismatch but won't catch a missing handler.
VALID_RAISE_ERRORS: frozenset[str] = frozenset(
    {"auth", "connection", "capability", "internal"}
)

type _RaiseChoice = Literal["auth", "connection", "capability", "internal"]

# Default cap profile — "full read + write" matches what Story 9 tests
# expect from a mock that should be reachable by every router path.
_DEFAULT_CAPABILITIES = AdapterCapabilities(
    can_read=True,
    can_write=True,
    can_delete=True,
    supports_vector_search=True,
    supports_text_search=True,
    supports_provenance=True,
    supports_verification_metadata=True,
)

# Placeholder distance returned by ``search_similar_nodes`` — the mock
# doesn't have real embeddings, so it can't rank candidates. Tests that
# need a scoring mock can subclass and override; for federation tests
# we only need a stable non-zero value to verify routing.
_PLACEHOLDER_SIMILARITY_DISTANCE: float = 0.5


class MockKnowledgeAdapter:
    """In-memory ``KnowledgeAdapter`` for federation + capability tests."""

    def __init__(
        self,
        name: str,
        *,
        fixed_triples: list[KALTriple] | None = None,
        sleep_ms: int = 0,
        raise_error: _RaiseChoice | None = None,
        capabilities: AdapterCapabilities | None = None,
    ) -> None:
        self._name = name
        self._fixed_triples = fixed_triples or []
        self._sleep_ms = sleep_ms
        self._raise_error = raise_error
        self._capabilities = capabilities or _DEFAULT_CAPABILITIES

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    async def _gate(self) -> None:
        """Sleep then optionally raise — shared prologue for every method.

        Order matters: a mock with ``sleep_ms > timeout`` must time out
        BEFORE any ``raise_error`` would fire, so the federation router
        sees ``error_type="timeout"`` rather than the configured error.
        Tests of error classification therefore set ``sleep_ms=0``.
        """
        if self._sleep_ms > 0:
            await asyncio.sleep(self._sleep_ms / 1000)
        if self._raise_error == "auth":
            raise AdapterAuthError(f"mock auth failure ({self._name})")
        if self._raise_error == "connection":
            raise AdapterConnectionError(f"mock connection failure ({self._name})")
        if self._raise_error == "capability":
            raise CapabilityNotSupportedError(
                f"mock capability failure ({self._name})"
            )
        if self._raise_error == "internal":
            raise RuntimeError(f"mock internal failure ({self._name})")

    @staticmethod
    def _strip_triple(triple: KALTriple) -> KALTriple:
        """Reset ``adapter=None`` on the triple — Protocol contract.

        Adapters return un-stamped triples; the federation router stamps
        the field. Without this strip, a fixture mistakenly pre-stamped
        would silently mask router bugs.
        """
        return triple.model_copy(update={"adapter": None})

    @staticmethod
    def _strip_node(node: KALNode) -> KALNode:
        """Reset ``adapter=None`` on the node — Protocol contract twin
        of ``_strip_triple`` for ``get_node`` / ``search_similar_nodes``.
        """
        return node.model_copy(update={"adapter": None})

    # --- Read operations ---

    async def query_triples(
        self, query: KALQuery, tenant_id: str | None = None
    ) -> KALQueryResult:
        await self._gate()
        return KALQueryResult(
            triples=[self._strip_triple(t) for t in self._fixed_triples]
        )

    async def get_node(
        self, node_id: str, tenant_id: str | None = None
    ) -> KALNode | None:
        await self._gate()
        for triple in self._fixed_triples:
            if triple.subject.id == node_id:
                return self._strip_node(triple.subject)
            if triple.object is not None and triple.object.id == node_id:
                return self._strip_node(triple.object)
        return None

    async def query_adjacent_triples(
        self,
        node_id: str,
        predicate: str | None = None,
        direction: str = "outgoing",
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[KALTriple]:
        await self._gate()
        matches: list[KALTriple] = []
        for triple in self._fixed_triples:
            subj_match = triple.subject.id == node_id
            obj_match = (
                triple.object is not None and triple.object.id == node_id
            )
            if direction == "outgoing" and not subj_match:
                continue
            if direction == "incoming" and not obj_match:
                continue
            if direction == "both" and not (subj_match or obj_match):
                continue
            if predicate is not None and triple.predicate != predicate:
                continue
            matches.append(self._strip_triple(triple))
            if len(matches) >= limit:
                break
        return matches

    # --- Write operations ---

    async def insert_triples(
        self, triples: list[KALTriple], tenant_id: str | None = None
    ) -> int:
        await self._gate()
        return len(triples)

    async def update_verification(
        self,
        triple_id: str,
        verification: VerificationMetadata,
        tenant_id: str | None = None,
    ) -> bool:
        await self._gate()
        return True

    async def delete_triples(
        self, triple_ids: list[str], tenant_id: str | None = None
    ) -> int:
        await self._gate()
        return len(triple_ids)

    async def quarantine_triples_by_source(
        self, source: str, tenant_id: str | None = None
    ) -> int:
        await self._gate()
        return 0

    async def restore_triples_by_source(
        self, source: str, tenant_id: str | None = None
    ) -> int:
        await self._gate()
        return 0

    # --- Vector search ---

    async def search_similar_nodes(
        self,
        query_vector: list[float],
        tenant_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[KALNode, float]]:
        await self._gate()
        pairs: list[tuple[KALNode, float]] = []
        seen_ids: set[str] = set()
        for triple in self._fixed_triples:
            node = triple.subject
            if node.id in seen_ids:
                continue
            seen_ids.add(node.id)
            pairs.append(
                (self._strip_node(node), _PLACEHOLDER_SIMILARITY_DISTANCE)
            )
            if len(pairs) >= limit:
                break
        return pairs
