"""MCP client-session wrapper for ingestion pulls (KAL as MCP client).

Wraps the ``mcp`` SDK ``ClientSession`` with the read surface the
ingestion connector needs — ``list_resources`` / ``read_resource`` /
``list_tools`` / ``call_tool`` — over **stdio** (spawn a local server) or
**Streamable-HTTP** (remote) transports, behind a connect context
manager that owns the transport + session lifecycle.

Transport + protocol failures surface as KAL ``AdapterConnectionError``
so the connector classifies them the same way the federation router does.

Requires the ``mcp`` SDK (the ``[mcp]`` extra); imported lazily from the
package ``__init__`` so ``import knowlytix.kal.sources.mcp`` works without it.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl

from knowlytix.kal.errors import AdapterConnectionError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mcp.types import CallToolResult, ReadResourceResult

    from knowlytix.kal.sources.mcp.protocol import (
        McpSourceConfig,
        McpSourceCredentials,
    )

logger = logging.getLogger(__name__)

# Errors translated into KAL's AdapterConnectionError at the wrapper
# boundary. McpError covers protocol-level failures from the server;
# ConnectionError / OSError cover raw transport failures.
_TRANSPORT_ERRORS = (McpError, ConnectionError, OSError)

DEFAULT_MAX_ITEMS = 1000


class McpClientSession:
    """Thin read-surface wrapper over an ``mcp`` ``ClientSession``.

    Construct directly around an existing session (tests wrap the
    in-memory transport this way), or use :meth:`connect` to own a real
    stdio / HTTP transport for the duration of an ingest.
    """

    def __init__(self, session: ClientSession) -> None:
        self._session = session

    @classmethod
    @asynccontextmanager
    async def connect(
        cls,
        config: McpSourceConfig,
        credentials: McpSourceCredentials,
    ) -> AsyncIterator[McpClientSession]:
        """Open a transport + initialized session for ``config``.

        Used by the runtime (knowly endpoint / worker). The transport and
        session are torn down on exit. Connect-time failures surface as
        ``AdapterConnectionError``.
        """
        async with AsyncExitStack() as stack:
            try:
                if config.transport == "stdio":
                    assert config.command is not None  # config validator guarantee
                    params = StdioServerParameters(
                        command=config.command, args=config.command_args
                    )
                    read, write = await stack.enter_async_context(
                        stdio_client(params)
                    )
                else:
                    assert config.url is not None  # config validator guarantee
                    headers = (
                        {"Authorization": f"Bearer {credentials.bearer_token}"}
                        if credentials.bearer_token
                        else None
                    )
                    read, write, _ = await stack.enter_async_context(
                        streamablehttp_client(config.url, headers=headers)
                    )
                session = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
            except _TRANSPORT_ERRORS as exc:
                raise AdapterConnectionError(
                    f"MCP connect failed for source {config.name!r}: {exc}"
                ) from exc
            logger.debug(
                "connected MCP source %s via %s", config.name, config.transport
            )
            yield cls(session)

    async def list_resources(
        self, max_items: int = DEFAULT_MAX_ITEMS
    ) -> list[str]:
        """Return resource URIs, walking pagination up to ``max_items``."""
        uris: list[str] = []
        cursor: str | None = None
        try:
            while True:
                result = await self._session.list_resources(cursor)
                # Guard against a malformed server that returns an empty
                # page with a non-None cursor — that would spin forever.
                if not result.resources:
                    break
                uris.extend(str(r.uri) for r in result.resources)
                cursor = result.nextCursor
                if cursor is None or len(uris) >= max_items:
                    break
        except _TRANSPORT_ERRORS as exc:
            raise AdapterConnectionError(
                f"list_resources failed: {exc}"
            ) from exc
        return uris[:max_items]

    async def read_resource(self, uri: str) -> ReadResourceResult:
        """Read one resource by URI."""
        try:
            return await self._session.read_resource(AnyUrl(uri))
        except _TRANSPORT_ERRORS as exc:
            raise AdapterConnectionError(
                f"read_resource {uri!r} failed: {exc}"
            ) from exc

    async def list_tools(self, max_items: int = DEFAULT_MAX_ITEMS) -> list[str]:
        """Return tool names, walking pagination up to ``max_items``."""
        names: list[str] = []
        cursor: str | None = None
        try:
            while True:
                result = await self._session.list_tools(cursor)
                # Guard against an empty page with a non-None cursor.
                if not result.tools:
                    break
                names.extend(tool.name for tool in result.tools)
                cursor = result.nextCursor
                if cursor is None or len(names) >= max_items:
                    break
        except _TRANSPORT_ERRORS as exc:
            raise AdapterConnectionError(f"list_tools failed: {exc}") from exc
        return names[:max_items]

    async def call_tool(
        self, name: str, args: dict[str, object]
    ) -> CallToolResult:
        """Call one tool with an argument template."""
        try:
            return await self._session.call_tool(name, args)
        except _TRANSPORT_ERRORS as exc:
            raise AdapterConnectionError(
                f"call_tool {name!r} failed: {exc}"
            ) from exc
