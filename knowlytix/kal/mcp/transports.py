"""Transport wiring — stdio + Streamable HTTP entry points.

Per the v0.6.0 plan (``docs/plans/plan_knowlytix_kal_v0_6.md``).

- **Stdio** lives on ``KalMcpServer.run_stdio()`` /
  ``run_stdio_async()`` in ``server.py`` directly — FastMCP's
  built-in stdio transport needs no wrapper.
- **Streamable HTTP**: ``KalMcpServer.run_http_async()`` wraps
  FastMCP's ``streamable_http_app()`` with a Starlette bearer-token
  middleware, then serves via uvicorn. The middleware resolves the
  bearer token to a ``tenant_id`` via ``TokenTenantResolver``,
  attaches it to a ``ContextVar`` for the duration of the request,
  and the tool closures read it back via ``_SmartTenantResolver``.
  Missing or invalid tokens return 401 without dispatching the
  request to the MCP app.

The auth model is deliberately minimum-viable:

- Bearer token in the ``Authorization`` header is the only
  credential format.
- Tokens are looked up in a static map; no OAuth, no JWT signature
  verification.
- Production deployments are expected to front the HTTP transport
  with their own gateway (Cloudflare Access, API gateway, mTLS) for
  defense-in-depth and revocation. The bearer-token check here is
  the inner-most ring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from knowlytix.kal.mcp.auth import _HTTP_TENANT_VAR

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from knowlytix.kal.mcp.auth import TokenTenantResolver

logger = logging.getLogger(__name__)

DEFAULT_HTTP_HOST: Final[str] = "127.0.0.1"
DEFAULT_HTTP_PORT: Final[int] = 8765
DEFAULT_HTTP_PATH: Final[str] = "/mcp"


class _BearerTokenAuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that gates every request behind a bearer
    token + maps the token to a ``tenant_id`` via a
    ``TokenTenantResolver``.

    Resolution flow:

    1. Read ``Authorization: Bearer <token>`` header.
    2. Resolve ``<token>`` to a ``tenant_id`` via the configured
       ``TokenTenantResolver``.
    3. Attach the tenant to ``_HTTP_TENANT_VAR`` for the request's
       lifetime (reset via ``ContextVar.reset`` in a ``finally``).
    4. Missing / malformed header → 401. Unknown token → 401. Both
       cases short-circuit before the MCP app sees the request.

    Defensive logging: the Authorization header value is never
    logged. The ``BaseHTTPMiddleware`` base doesn't log the request
    body by default; tests pin that no token bleeds through any
    log record this middleware emits.
    """

    def __init__(
        self,
        app: Starlette,
        resolver: TokenTenantResolver,
    ) -> None:
        super().__init__(app)
        self._resolver = resolver

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            # Log without surfacing the raw header (which is "").
            logger.warning(
                "rejecting MCP request: missing or malformed "
                "Authorization header (path=%s)",
                request.url.path,
            )
            return JSONResponse(
                {"error": "missing or malformed Authorization header"},
                status_code=401,
            )
        token = auth_header[len("Bearer "):]
        tenant_id = self._resolver.resolve(token)
        if tenant_id is None:
            # Token characters MUST NOT reach logs. Record a fixed
            # placeholder so observability still shows "auth failed
            # here" without exposing the credential.
            logger.warning(
                "rejecting MCP request: token not in resolver map "
                "(path=%s)",
                request.url.path,
            )
            return JSONResponse(
                {"error": "invalid token"}, status_code=401,
            )
        token_handle = _HTTP_TENANT_VAR.set(tenant_id)
        try:
            return await call_next(request)
        finally:
            _HTTP_TENANT_VAR.reset(token_handle)


def build_http_app(
    fastmcp_http_app: Starlette,
    resolver: TokenTenantResolver,
) -> Starlette:
    """Wrap a FastMCP streamable_http app with bearer-token auth.

    ``add_middleware`` registers the auth middleware before the
    Starlette app finalises its routing — must be called BEFORE the
    first request. We do this synchronously at ``run_http_async``
    construction time, before yielding control to uvicorn.

    Returns the same Starlette instance for fluency; uvicorn (or an
    ASGI test client) serves it as-is.
    """
    # Starlette's ``add_middleware`` factory protocol expects a
    # narrow Callable signature that ``BaseHTTPMiddleware`` subclasses
    # don't structurally match (the resolver kwarg gets bound by
    # Starlette via partial application). Runtime accepts the class +
    # kwargs cleanly; the static-type mismatch is established
    # ecosystem noise.
    fastmcp_http_app.add_middleware(
        _BearerTokenAuthMiddleware,  # type: ignore[arg-type]
        resolver=resolver,
    )
    return fastmcp_http_app
