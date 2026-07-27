"""HTTP auth + tenant resolution for the MCP server.

Per the v0.6.0 plan (``docs/plans/plan_knowlytix_kal_v0_6.md``)
§"Auth + tenant resolution". The HTTP transport (Phase C) needs a
way to map a bearer token to a tenant scope so that a single MCP
endpoint can serve multiple tenants without exposing tenant_id as a
tool argument (which would be a prompt-injection vector).

Three pieces:

- ``TokenTenantResolver`` — the token → tenant_id map itself.
  Configured at server-startup; raises on duplicate-token entries
  (which would cross-route between tenants).
- ``_HTTP_TENANT_VAR`` — a ``ContextVar`` the HTTP middleware sets
  per-request from the resolved token. Tool handlers read this via
  the smart resolver below.
- ``_SmartTenantResolver`` — the ``Callable[[], str | None]``
  injected into ``register_tools``. Returns the per-request HTTP
  tenant when one is set; falls back to the server-bound tenant
  (stdio mode) otherwise. Single resolver works for both transports
  without re-registering tools per request.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Final

# Per-request tenant binding set by the HTTP auth middleware. Stays
# unset in stdio mode — ``_SmartTenantResolver`` falls back to the
# server-bound tenant when this is None.
_HTTP_TENANT_VAR: Final[ContextVar[str | None]] = ContextVar(
    "kal_mcp_http_tenant", default=None,
)


# Callable signature for tenant resolution: tool handlers call this on
# every invocation to get the active tenant scope. No args — context
# (HTTP request, server-bound state) is captured by the closure.
TenantProvider = Callable[[], str | None]


class _SmartTenantResolver:
    """Tenant provider that prefers HTTP request context, falls back
    to the server-bound tenant.

    Stdio mode: ``_HTTP_TENANT_VAR`` is never set, so every call
    returns the bound value. HTTP mode: middleware sets the
    ContextVar per-request, so the call returns the request's
    tenant. Same resolver instance works for both — tool handlers
    don't need to know which transport they're under.
    """

    def __init__(self, bound: str | None) -> None:
        self._bound = bound

    @property
    def bound(self) -> str | None:
        return self._bound

    def __call__(self) -> str | None:
        http_tenant = _HTTP_TENANT_VAR.get()
        return http_tenant if http_tenant is not None else self._bound


class TokenTenantResolver:
    """Bearer-token → tenant_id map for the HTTP transport.

    Multi-tenant HTTP deployments construct one resolver at startup
    and pass it to ``KalMcpServer.run_http_async``. Single-tenant
    stdio deployments don't need this — the server-bound tenant_id
    handles them.

    Construct from a ``dict[str, str]`` directly, or from a list of
    ``(token, tenant_id)`` pairs via ``from_pairs`` when the source
    config could in principle carry duplicate entries (e.g. a
    list-shaped config file). ``from_pairs`` raises on duplicate
    tokens — silent dedup would cross-route between tenants.

    Multiple tokens MAY map to the same tenant (that's how token
    rotation works: issue a new token for the same tenant, retire
    the old one). Tokens are 1:1 with tenant identities; the inverse
    is many-to-1.
    """

    def __init__(self, token_to_tenant: dict[str, str]) -> None:
        if not token_to_tenant:
            raise ValueError(
                "TokenTenantResolver requires at least one entry; "
                "an empty map would reject every request"
            )
        # Defensive copy — caller mutating their dict shouldn't shift
        # routing.
        self._map = dict(token_to_tenant)

    @classmethod
    def from_pairs(
        cls, entries: list[tuple[str, str]]
    ) -> TokenTenantResolver:
        """Construct from a (token, tenant_id) sequence.

        Raises ``ValueError`` if the same token appears more than
        once — a list-shaped config could carry duplicates that a
        dict literal cannot, and silent dedup of an "oldest wins" or
        "newest wins" flavour would cross-route between tenants.
        """
        seen: set[str] = set()
        for token, _ in entries:
            if token in seen:
                raise ValueError(
                    f"duplicate token entry in TokenTenantResolver: "
                    f"{token!r}"
                )
            seen.add(token)
        return cls(dict(entries))

    def resolve(self, token: str) -> str | None:
        """Look up the tenant_id for a bearer token; ``None`` if
        unknown. ``None`` is what the middleware turns into 401."""
        return self._map.get(token)
