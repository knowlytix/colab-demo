"""MCP tool definitions — the 4 KAL read methods as typed tool calls.

Per the v0.6.0 plan (``docs/plans/plan_knowlytix_kal_v0_6.md``) — paper
§3.6 "LLMs as direct consumers of KAL".

The 4 tools registered here close over a passed-in ``FederationRouter``
and ``tenant_id`` so the LLM caller cannot override tenant via tool
args. Tenant resolution lives **only** at the auth/transport layer
(stdio: server-bound; HTTP: bearer-token map). Tool argument schemas
come from Pydantic's ``model_json_schema()`` directly via FastMCP's
introspection — no hand-maintained JSON Schema; field docstrings on
``KALQuery`` / ``KALNode`` / ``KALTriple`` become user-facing tool
documentation.

Read-only surface by design. Write tools (``insert_triples``,
``update_verification``, ``delete_triples``) are deliberately
excluded — the §3.6 framing positions MCP as the "LLMs as direct
consumers" path; LLM-driven writes belong to a deliberate later
release with explicit audit + verifier-identity story.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from knowlytix.kal.types import KALNode, KALQuery, KALQueryResult

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from knowlytix.kal.federation import FederationRouter
    from knowlytix.kal.mcp.auth import TenantProvider

logger = logging.getLogger(__name__)


def register_tools(  # type: ignore[explicit-any]
    app: FastMCP,
    router: FederationRouter,
    *,
    tenant_provider: TenantProvider,
    prefix: str = "",
) -> None:
    """Register the 4 KAL read tools on the given FastMCP app.

    Tools close over ``router`` and ``tenant_provider``; the LLM
    cannot override tenant via tool args. ``tenant_provider`` is
    called on each tool invocation to resolve the active tenant —
    stdio mode binds it to a constant at server-construction; HTTP
    mode resolves per-request from the bearer-token map. The same
    closure body serves both transports.

    ``prefix`` (default empty) is prepended to each tool name — set
    it to e.g. ``"kal_"`` when the same agent has multiple KAL-shaped
    MCP servers attached and bare names would collide.
    """

    @app.tool(
        name=f"{prefix}query_triples",
        description=(
            "Query KAL triples by pattern, text-search, or vector. "
            "Returns a KALQueryResult envelope: triples (deduped + merged "
            "across federated adapters) + per-adapter errors. Under "
            "ConflictStrategy.ALL the router auto-flips "
            "include_all_verifications=True so multi-verifier rows "
            "surface as separate entries (§5.8 vertical axis)."
        ),
    )
    async def query_triples(query: KALQuery) -> KALQueryResult:
        return await router.query_triples(query, tenant_id=tenant_provider())

    @app.tool(
        name=f"{prefix}get_node",
        description=(
            "Fetch a single KAL node by ID across federated adapters. "
            "Returns the first non-None match in registration order, "
            "with KALNode.adapter stamped to the source adapter's name "
            "so provenance is visible. Returns null if no adapter has "
            "the node."
        ),
    )
    async def get_node(node_id: str) -> KALNode | None:
        return await router.get_node(node_id, tenant_id=tenant_provider())

    @app.tool(
        name=f"{prefix}query_adjacent_triples",
        description=(
            "One-hop adjacency walk from a node across federated "
            "adapters. direction='outgoing' (node is subject), "
            "'incoming' (node is object), or 'both'. predicate "
            "optionally filters by relation name. Returns "
            "KALQueryResult envelope with the same dedup + "
            "partial-failure semantics as query_triples."
        ),
    )
    async def query_adjacent_triples(
        node_id: str,
        predicate: str | None = None,
        direction: str = "outgoing",
        limit: int = 100,
    ) -> KALQueryResult:
        return await router.query_adjacent_triples(
            node_id,
            predicate=predicate,
            direction=direction,
            limit=limit,
            tenant_id=tenant_provider(),
        )

    @app.tool(
        name=f"{prefix}search_similar_nodes",
        description=(
            "Vector-similarity search across federated adapters with "
            "supports_vector_search=True. Returns nodes ranked by "
            "ascending distance. Limit caps the merged result set "
            "size; each adapter contributes up to limit candidates "
            "before merging."
        ),
    )
    async def search_similar_nodes(
        query_vector: list[float],
        limit: int = 10,
    ) -> list[dict[str, object]]:
        # Router returns ``list[tuple[KALNode, float]]``; tuples don't
        # JSON-serialise as named pairs, so flatten to dict form for
        # tool consumers. ``model_dump()`` produces the same KALNode
        # JSON shape FastMCP would have produced via its own pydantic
        # serialiser.
        results = await router.search_similar_nodes(
            query_vector, tenant_id=tenant_provider(), limit=limit,
        )
        return [
            {"node": node.model_dump(), "distance": distance}
            for node, distance in results
        ]

    # Silence "registered but never used" linter — the decorator
    # mutates ``app`` as a side effect.
    _ = (query_triples, get_node, query_adjacent_triples, search_similar_nodes)
