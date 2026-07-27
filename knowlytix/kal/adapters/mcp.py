"""Live query-time MCP adapter — KAL §7.3 Pattern 2 (query-time synthesis).

A read-only ``KnowledgeAdapter`` that synthesizes ``KALTriple``s **per query**
from an external MCP server: a ``text_search`` query is fed to an
operator-allowlisted retrieval tool, the tool's text output is run through
an injected ``ClaimExtractor``, and the resulting triples are stamped with
``mcp://<source>`` provenance, filtered to the query's patterns, and
returned — never persisted (the inverse of the Pattern-3 ingest connector).

Generic over the ``ClaimExtractor`` seam (no concrete extractor here) and
extractor-agnostic (it never names an extraction backend) — so it mirrors
to the ``knowlytix-kal`` package cleanly; the consuming app injects its own
extractor + a ``client_factory`` (a builder closure over
``McpClientSession.connect``). Connection is **per-query** (connect → call →
close); a persistent/pooled session is a follow-up if stdio per-query spawn
cost bites.

The ``mcp`` SDK is imported lazily inside ``query_triples`` so the module is
importable without the ``[mcp]`` extra.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from knowlytix.kal.errors import (
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
    TripleProvenance,
    VerificationMetadata,
)

if TYPE_CHECKING:
    from knowlytix.kal.sources.mcp.client import McpClientSession
    from knowlytix.kal.sources.mcp.protocol import ClaimExtractor

logger = logging.getLogger(__name__)

# A factory yielding a fresh connected client per query (connect-per-query).
# In production a builder closes over ``McpClientSession.connect(cfg, creds)``;
# tests inject a fake.
ClientFactory = Callable[[], "AbstractAsyncContextManager[McpClientSession]"]


class McpAdapterConfig(BaseModel):  # type: ignore[explicit-any]
    """Retrieval + provenance config for the live MCP adapter.

    Connection details live in the injected ``client_factory``; this carries
    only the query-time knobs. Extractor-agnostic — no extraction tool here
    (that's a knowly-side builder concern, kept out of the mirror).
    """

    source_name: str = Field(
        min_length=1,
        description="Provenance key — synthesized triples get mcp://<source_name>.",
    )
    retrieval_tool: str = Field(
        min_length=1,
        description="The operator-allowlisted MCP tool called per query.",
    )
    retrieval_query_arg: str = Field(
        min_length=1,
        description="The tool-arg key the query text is injected into.",
    )
    retrieval_extra_args: dict[str, object] = Field(default_factory=dict)
    max_results: int = Field(default=100, ge=1)


class McpKnowledgeAdapter:
    """Pattern-2 query-time synthesis adapter (read-only)."""

    def __init__(
        self,
        *,
        name: str,
        client_factory: ClientFactory,
        extractor: ClaimExtractor,
        config: McpAdapterConfig,
    ) -> None:
        self._name = name
        self._client_factory = client_factory
        self._extractor = extractor
        self._config = config

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> AdapterCapabilities:
        # Read-only, text-search-driven. Stamps provenance (mcp://<name>);
        # synthesized triples are unverified (no verification metadata).
        return AdapterCapabilities(
            can_read=True,
            can_write=False,
            can_delete=False,
            supports_vector_search=False,
            supports_text_search=True,
            supports_provenance=True,
            supports_verification_metadata=False,
            supports_atomic_writes=False,
        )

    # --- Read ---

    async def query_triples(
        self, query: KALQuery, tenant_id: str | None = None
    ) -> KALQueryResult:
        """Synthesize triples for a ``text_search`` query.

        Short-circuits to an empty result (no connect) when this adapter
        can't satisfy the query — it serves only ``text_search`` queries and
        produces only *unverified* triples, so a pattern-only query, a
        verified-only query (``include_unverified=False``), or a
        ``min_confidence`` filter all return empty. This matters because the
        federation router sends every ``query_triples`` to every adapter; the
        short-circuit avoids a per-query connect when we have nothing to add.
        Otherwise: call the retrieval tool → normalize → extract → stamp
        ``mcp://<source>`` → post-filter to ``query.patterns`` → cap to
        ``limit``. The router's ``timeout_ms`` bounds the whole call.
        """
        query_text = query.text_search
        if (
            not query_text
            or not query_text.strip()
            or not query.include_unverified
            or query.min_confidence is not None
        ):
            return KALQueryResult(triples=[])

        # Lazy import: the mcp SDK is an optional extra.
        from knowlytix.kal.sources.mcp.normalize import normalize_tool_result

        # Static extra args first, then the query injection — so the query
        # ALWAYS wins even if extra_args is misconfigured with the same key.
        args: dict[str, object] = {
            **self._config.retrieval_extra_args,
            self._config.retrieval_query_arg: query_text,
        }
        async with self._client_factory() as client:
            tool_result = await client.call_tool(
                self._config.retrieval_tool, args
            )
        if tool_result.isError:
            # A failing retrieval tool returns its error text as content —
            # do NOT extract triples from an error message. Graceful empty +
            # a server-side trace (the connector's "recorded, not extracted"
            # stance, query-time edition).
            logger.warning(
                "MCP retrieval tool %r returned isError for source %r",
                self._config.retrieval_tool,
                self._config.source_name,
            )
            return KALQueryResult(triples=[])
        text = normalize_tool_result(tool_result)
        triples = await self._extractor.extract(
            text, source_uri=f"mcp://{self._config.source_name}"
        )
        stamped = self._stamp(triples)
        matched = [t for t in stamped if _matches_patterns(t, query.patterns)]
        # Cap by the caller's limit AND the adapter's max_results safety cap.
        cap = min(query.limit, self._config.max_results)
        return KALQueryResult(triples=matched[:cap])

    async def get_node(
        self, node_id: str, tenant_id: str | None = None
    ) -> KALNode | None:
        # Identity can't be synthesized from a text-retrieval source.
        return None

    async def query_adjacent_triples(
        self,
        node_id: str,
        predicate: str | None = None,
        direction: str = "outgoing",
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[KALTriple]:
        # Adjacency can't be synthesized meaningfully; query_triples is the
        # Pattern-2 surface.
        return []

    async def verify_source(self) -> None:
        """Liveness probe for the connection-registry handshake.

        Connects and asserts the source actually exposes the configured
        ``retrieval_tool``. The default ``query_triples`` handshake is useless
        for this adapter — an empty/pattern-only probe short-circuits without
        ever connecting — so the builder routes its handshake here instead.
        Raises ``AdapterConnectionError`` on a transport failure (from the
        client) or when the configured tool is absent.
        """
        async with self._client_factory() as client:
            tools = await client.list_tools()
        if self._config.retrieval_tool not in tools:
            raise AdapterConnectionError(
                f"MCP source {self._config.source_name!r} does not expose the "
                f"configured retrieval tool {self._config.retrieval_tool!r}; "
                f"available tools: {sorted(tools)}"
            )

    # --- Write surface: unsupported (read-only, like Wikidata) ---

    async def insert_triples(
        self, triples: list[KALTriple], tenant_id: str | None = None
    ) -> int:
        raise CapabilityNotSupportedError(
            "McpKnowledgeAdapter is read-only (query-time synthesis); "
            "use the MCP ingest connector to persist."
        )

    async def update_verification(
        self,
        triple_id: str,
        verification: VerificationMetadata,
        tenant_id: str | None = None,
    ) -> bool:
        raise CapabilityNotSupportedError(
            "McpKnowledgeAdapter is read-only; synthesized triples are ephemeral."
        )

    async def delete_triples(
        self, triple_ids: list[str], tenant_id: str | None = None
    ) -> int:
        raise CapabilityNotSupportedError(
            "McpKnowledgeAdapter is read-only; nothing is stored to delete."
        )

    async def quarantine_triples_by_source(
        self, source: str, tenant_id: str | None = None
    ) -> int:
        raise CapabilityNotSupportedError(
            "McpKnowledgeAdapter has no stored triples to quarantine."
        )

    async def restore_triples_by_source(
        self, source: str, tenant_id: str | None = None
    ) -> int:
        raise CapabilityNotSupportedError(
            "McpKnowledgeAdapter has no stored triples to restore."
        )

    async def search_similar_nodes(
        self,
        query_vector: list[float],
        tenant_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[KALNode, float]]:
        raise CapabilityNotSupportedError(
            "McpKnowledgeAdapter does not support vector search."
        )

    # --- Lifecycle (connect-per-query → aclose is a no-op) ---

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> McpKnowledgeAdapter:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        await self.aclose()

    def _stamp(self, triples: list[KALTriple]) -> list[KALTriple]:
        # Mirrors the ingest connector's provenance stamp so live (Pattern 2)
        # and ingest (Pattern 3) produce identical ``mcp://<source>`` keys —
        # the per-source revoke covers either path. Synthesized triples stay
        # unverified.
        now = datetime.now(tz=UTC)
        out: list[KALTriple] = []
        for triple in triples:
            base = triple.provenance or TripleProvenance()
            prov = base.model_copy(
                update={
                    "source": f"mcp://{self._config.source_name}",
                    "extractor": self._config.retrieval_tool,
                    "extracted_at": now,
                }
            )
            out.append(triple.model_copy(update={"provenance": prov}))
        return out


def _matches_patterns(
    triple: KALTriple, patterns: list[KALTriplePattern]
) -> bool:
    """Post-extraction pattern filter — mirrors the JSONL/Postgres semantics
    (set-membership within a field, AND across fields; empty = match all).
    A triple matches if it satisfies the union of the patterns' fields."""
    subjects = {p.subject for p in patterns if p.subject is not None}
    predicates = {p.predicate for p in patterns if p.predicate is not None}
    objects = {p.object for p in patterns if p.object is not None}
    if not (subjects or predicates or objects):
        return True
    if subjects and not _node_in(triple.subject, subjects):
        return False
    if predicates and triple.predicate not in predicates:
        return False
    return not (
        objects
        and (triple.object is None or not _node_in(triple.object, objects))
    )


def _node_in(node: KALNode, values: set[str]) -> bool:
    return node.id in values or node.name in values
