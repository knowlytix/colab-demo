"""``KalMcpServer`` — wraps ``FederationRouter`` as an MCP server.

Per the v0.6.0 plan (``docs/plans/plan_knowlytix_kal_v0_6.md``). Single
endpoint exposing the federated KAL view as 4 typed MCP tools
(``query_triples`` / ``get_node`` / ``query_adjacent_triples`` /
``search_similar_nodes``); read-only by design (§3.6).

Stdio mode (the dominant local-agent shape — Claude Desktop, Cursor,
local CLI agents): construct with a server-bound ``tenant_id`` and
call ``run_stdio()``. The trust boundary is the local process, so no
auth — the caller IS the user.

HTTP mode (Streamable HTTP for remote agents): construct with
``allowed_hosts`` set (DNS-rebinding protection), pass a
``TokenTenantResolver`` to ``run_http_async``. Per-request tenant
resolution happens in the auth middleware; the tool closures read
the resolved tenant via a ``ContextVar``.

Construction sketches::

    # Stdio
    from knowlytix.kal.federation import FederationRouter
    from knowlytix.kal.mcp import KalMcpServer
    server = KalMcpServer(router, tenant_id="tenant-a")
    server.run_stdio()

    # HTTP
    from knowlytix.kal.mcp import KalMcpServer, TokenTenantResolver
    server = KalMcpServer(
        router, allowed_hosts=["mcp.example.com"],
    )
    resolver = TokenTenantResolver({"<token>": "tenant-a"})
    asyncio.run(server.run_http_async(resolver))
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette

from knowlytix.kal.federation import FederationRouter
from knowlytix.kal.mcp.auth import _SmartTenantResolver
from knowlytix.kal.mcp.tools import register_tools
from knowlytix.kal.mcp.transports import (
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_PORT,
    build_http_app,
)

if TYPE_CHECKING:
    from knowlytix.kal.mcp.auth import TokenTenantResolver

logger = logging.getLogger(__name__)


class KalMcpServer:
    """Wrap a ``FederationRouter`` as an MCP server exposing the 4 KAL
    read methods as tools.

    Parameters:
    - ``router`` — the federation router to wrap. The router's
      registered adapters determine which backends contribute.
    - ``tenant_id`` — server-bound tenant scope. ``None`` for
      single-tenant deployments or for tests where the registered
      adapters ignore tenant scoping (``MockKnowledgeAdapter``).
    - ``name`` — MCP server name advertised to the client. Defaults
      to ``"knowlytix-kal"``. Distinct from a multi-server agent's
      tool prefix; agents that want both still need ``prefix`` to
      disambiguate same-named tools.
    - ``prefix`` — string prepended to each tool name. Empty by
      default so the tools register as bare ``query_triples`` /
      ``get_node`` / etc. Set to e.g. ``"kal_"`` when the same agent
      has multiple KAL-shaped MCP servers attached.
    - ``instructions`` — optional server-level instructions surfaced
      to the client at handshake. Default helps the LLM understand
      what the tools collectively do.
    """

    _DEFAULT_INSTRUCTIONS = (
        "Federated knowledge-graph access via the Knowledge Adapter "
        "Layer (KAL). Use query_triples for pattern / text / vector "
        "queries, get_node for single-entity lookups, "
        "query_adjacent_triples for one-hop neighbourhood walks, and "
        "search_similar_nodes for vector-similarity ranking. All "
        "tools are read-only; writes are not exposed."
    )

    def __init__(
        self,
        router: FederationRouter,
        *,
        tenant_id: str | None = None,
        name: str = "knowlytix-kal",
        prefix: str = "",
        instructions: str | None = None,
        allowed_hosts: list[str] | None = None,
    ) -> None:
        self._router = router
        self._tenant_id = tenant_id
        # ``streamable_http_path="/mcp"`` matches the DEFAULT_HTTP_PATH
        # constant in ``transports.py`` — keep them in sync so the
        # Phase-C HTTP entry point doesn't need to override the
        # FastMCP default.
        #
        # ``allowed_hosts`` configures FastMCP's DNS-rebinding
        # protection. Default ``None`` → MCP SDK's built-in
        # protection rejects any Host header (safe-by-default for
        # stdio mode; HTTP mode MUST pass an explicit list). Pass
        # e.g. ``["mcp.example.com"]`` for a production HTTP
        # deployment behind a reverse proxy.
        transport_security = (
            TransportSecuritySettings(allowed_hosts=allowed_hosts)
            if allowed_hosts is not None
            else None
        )
        self._app = FastMCP(
            name=name,
            instructions=instructions or self._DEFAULT_INSTRUCTIONS,
            streamable_http_path="/mcp",
            transport_security=transport_security,
        )
        # SmartTenantResolver prefers HTTP request context (set by the
        # bearer-token middleware) and falls back to the bound tenant.
        # The same resolver instance serves stdio + HTTP — tool
        # closures don't need to know which transport is active.
        self._tenant_resolver = _SmartTenantResolver(tenant_id)
        register_tools(
            self._app,
            router,
            tenant_provider=self._tenant_resolver,
            prefix=prefix,
        )

    @property
    def app(self) -> FastMCP:  # type: ignore[explicit-any]
        """Expose the underlying FastMCP app for advanced wiring +
        in-process testing (``await app.call_tool(name, args)``).

        ``FastMCP`` signatures embed ``Any`` in their generic
        parameters (see ``LifespanResultT``); ``disallow_any_explicit``
        flags the leak. Suppressed at the boundary same way debug_api
        suppresses Pydantic BaseModel explicit-any leaks.
        """
        return self._app

    @property
    def tenant_id(self) -> str | None:
        return self._tenant_id

    async def run_stdio_async(self) -> None:
        """Run the server over stdio. Async entry point — call from
        an existing event loop.

        Blocks until the client disconnects. Cancellation propagates
        to the in-flight tool calls; cooperative-cancellation adapters
        (the §5.6 contract) clean up correctly.
        """
        logger.info(
            "starting KalMcpServer (stdio) tenant_id=%r",
            self._tenant_id,
        )
        await self._app.run_stdio_async()

    def run_stdio(self) -> None:
        """Synchronous stdio entry point — convenience for CLI scripts.

        Internally calls ``FastMCP.run(transport="stdio")`` which
        manages its own event loop. **Do not call from inside an
        existing event loop** — that would attempt to start a nested
        loop and raise ``RuntimeError: This event loop is already
        running``. Embedded-async callers MUST use ``run_stdio_async``.
        """
        logger.info(
            "starting KalMcpServer (stdio, sync) tenant_id=%r",
            self._tenant_id,
        )
        self._app.run(transport="stdio")

    def build_http_app(
        self, token_resolver: TokenTenantResolver,
    ) -> Starlette:
        """Build the auth-wrapped ASGI app without serving it.

        Returns the Starlette app with bearer-token middleware
        applied. Used by:

        - ``run_http_async`` to hand off to uvicorn.
        - Tests, to mount on an in-process ``httpx`` ASGI client.

        Each call returns a fresh wrapped app (a new
        ``streamable_http_app()`` instance + a new middleware stack);
        don't cache the return across server lifecycles. The
        middleware is added once on the returned app instance, so
        calling this twice on the same server is supported.
        """
        return build_http_app(
            self._app.streamable_http_app(), token_resolver,
        )

    async def run_http_async(
        self,
        token_resolver: TokenTenantResolver,
        *,
        host: str = DEFAULT_HTTP_HOST,
        port: int = DEFAULT_HTTP_PORT,
    ) -> None:
        """Run the server over Streamable HTTP. Async entry point.

        Wraps ``FastMCP.streamable_http_app()`` with the bearer-token
        middleware via ``build_http_app``, then serves via uvicorn.
        Blocks until uvicorn shuts down (SIGTERM, ``server.shutdown()``
        from another task, etc.).

        ``host`` / ``port`` default to ``127.0.0.1:8765`` (loopback);
        deployments fronted by a gateway opt in to ``0.0.0.0``
        explicitly. ``token_resolver`` is required (no anonymous-by-
        default fallback — silent auth bypass would be a multi-tenant
        leak primitive).
        """
        # Imported here so the dependency only resolves when HTTP is
        # actually used — stdio-only deployments avoid the import.
        import uvicorn

        app = self.build_http_app(token_resolver)
        logger.info(
            "starting KalMcpServer (Streamable HTTP) host=%s port=%d "
            "tenant_id=%r",
            host, port, self._tenant_id,
        )
        config = uvicorn.Config(
            app, host=host, port=port, log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
