"""KAL Model Context Protocol surface — paper §3.6 "LLMs as direct consumers".

v0.6.0 anchor (see ``docs/plans/plan_knowlytix_kal_v0_6.md``). Exposes a
``KnowledgeAdapter`` / ``FederationRouter`` as MCP tools so an LLM
agent (Claude Desktop, Cursor, hosted Claude, other MCP-aware clients)
can invoke ``query_triples`` / ``get_node`` / ``query_adjacent_triples`` /
``search_similar_nodes`` as typed tool calls during inference.

Read-only surface by design — writes (``insert_triples``,
``update_verification``, ``delete_triples``) stay reachable only
through the adapter/router directly. Letting an LLM call writes from a
tool widens the audit / hallucination-injection surface beyond what
the verification chain currently exists to manage; a deliberate
write-tool story is a separate plan.

Optional dependency: requires the ``mcp`` SDK. Install via
``pip install knowlytix-kal[mcp]``; bare ``knowlytix-kal`` does not
import this submodule's heavy paths at top level. Submodule imports
are intentionally lazy so a consumer that doesn't install the extras
can still ``import knowlytix.kal`` without an ``ImportError``.

Submodule layout:

- ``server.py`` — ``KalMcpServer`` (the actual server object;
  constructed once, ``run_stdio_async()`` / ``run_stdio()`` /
  ``run_http_async()`` entry points).
- ``tools.py`` — the 4 read-tool definitions; JSON Schema derived from
  ``KALQuery`` / ``KALNode`` / ``KALTriple`` via Pydantic's
  ``model_json_schema()``.
- ``auth.py`` — ``TokenTenantResolver`` (HTTP bearer-token → tenant_id
  map) + the ``ContextVar``-backed smart resolver that bridges stdio
  (server-bound tenant) and HTTP (per-request tenant) under one
  ``TenantProvider`` callable.
- ``transports.py`` — bearer-token Starlette middleware + the
  ``build_http_app`` helper that wraps FastMCP's
  ``streamable_http_app()``. Stdio doesn't need a wrapper — it
  delegates straight to ``FastMCP.run_stdio_async()``. Default HTTP
  bind: ``127.0.0.1:8765``, path ``/mcp``.
"""

from __future__ import annotations

# Lazy ``__getattr__`` keeps the ``mcp`` SDK import off the
# ``import knowlytix.kal`` path so consumers without the ``[mcp]``
# extras don't hit an ImportError at the top-level package import.
__all__ = [
    "KalMcpServer",
    "TokenTenantResolver",
]


def __getattr__(name: str) -> object:
    """Lazy export — defer the ``mcp`` SDK import until first use."""
    if name == "KalMcpServer":
        from knowlytix.kal.mcp.server import KalMcpServer
        return KalMcpServer
    if name == "TokenTenantResolver":
        from knowlytix.kal.mcp.auth import TokenTenantResolver
        return TokenTenantResolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
