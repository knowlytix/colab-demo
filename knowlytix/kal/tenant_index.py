"""Adapter → tenant ownership map for federated tenant scoping.

The in-process ``AdapterRegistry`` (``knowly.kal.registry``) is
tenant-blind: it stores adapters by name and ``all_adapters()`` returns
every registered instance. The connection-registry layer
(``knowly.modules.kal_connections``) however knows which tenant owns
each user-registered adapter — that information is the ``tenant_id``
column on the persisted ``kal_connections`` row.

``TenantIndex`` is the thin in-memory map that carries that ownership
information from the connection service into the
``FederationRouter`` without coupling the registry to tenant semantics.
The router calls ``adapters_for(tenant_id)`` while resolving fan-out
targets; if a ``TenantIndex`` is wired in, targets are filtered to the
caller's tenant plus the always-shared adapters (``postgres-main`` and
any future cross-tenant rows). With no index wired in, behaviour is
identical to today — every registered adapter is in scope.

Spec § naming: the federation router contract talks about "the
caller's tenant" without dictating where ownership data lives. This
module is Knowly's chosen home; future implementations could swap it
for a Redis-backed map without changing the router's API.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable


# Adapter / tenant identifiers are passed as strings on the federation
# router boundary (the API uses ``uuid.UUID`` internally but the router
# carries a generic ``str | None``). Normalising every UUID through
# ``str()`` here gives us one canonical form regardless of whether the
# caller passed a hyphenated string, a hex string, or a raw UUID, so
# ``bind(..., tenant_id=tenant_uuid)`` followed by
# ``adapters_for(str(tenant_uuid))`` always round-trips.
def _normalize_tenant_id(tenant_id: uuid.UUID | str) -> str:
    if isinstance(tenant_id, uuid.UUID):
        return str(tenant_id)
    return tenant_id


class TenantIndex:
    """Per-process adapter ownership table.

    Threading: the running app is single-threaded async per Fly machine
    and the index is mutated only by the connection service's register
    / unregister calls, which serialise on the request loop. No locks
    needed today; multi-worker deployment will need a Redis pub/sub
    invalidation channel (out of scope here).
    """

    def __init__(
        self,
        *,
        shared_adapters: Iterable[str] = (),
    ) -> None:
        """``shared_adapters`` names adapters visible to every tenant.

        The default empty iterable means "no shared adapters". Knowly
        wires in ``("postgres-main",)`` at lifespan startup since the
        canonical app-internal adapter is owned by the platform, not a
        tenant — every tenant's pipeline routes through it.
        """
        self._owner: dict[str, str] = {}
        self._shared: set[str] = set(shared_adapters)

    def bind(
        self, adapter_name: str, tenant_id: uuid.UUID | str
    ) -> None:
        """Record that ``adapter_name`` belongs to ``tenant_id``.

        Accepts ``uuid.UUID`` or pre-stringified UUID; both are
        canonicalised via ``_normalize_tenant_id`` so a later
        ``adapters_for(tenant_uuid)`` always round-trips. Replace
        semantics deliberate: re-registering an existing name (e.g.
        PATCH rebuilt the adapter under the same namespaced name)
        just refreshes the ownership row. No-op if the name is
        already in ``shared_adapters`` — admins can't reassign a
        shared adapter to a single tenant via a normal connection
        row, only via configuration.
        """
        if adapter_name in self._shared:
            return
        self._owner[adapter_name] = _normalize_tenant_id(tenant_id)

    def unbind(self, adapter_name: str) -> None:
        """Remove the ownership entry. Idempotent — unknown names
        silently no-op so disable/delete paths don't have to guard.
        """
        self._owner.pop(adapter_name, None)

    def adapters_for(self, tenant_id: uuid.UUID | str) -> frozenset[str]:
        """Names visible to ``tenant_id`` — the tenant's own adapters
        plus the always-shared set.

        Returns a ``frozenset`` so the router can use it as a fast
        membership filter without worrying about a caller mutating
        internal state. ``uuid.UUID`` or pre-stringified UUID both
        work — ``_normalize_tenant_id`` canonicalises so the lookup
        round-trips with whatever form ``bind`` was called with.
        Cost: O(n) over registered user-adapter rows, which is
        bounded by the per-tenant cap times tenant count — negligible
        at realistic sizes.
        """
        normalized = _normalize_tenant_id(tenant_id)
        owned = {
            name
            for name, owner in self._owner.items()
            if owner == normalized
        }
        return frozenset(owned | self._shared)

    def shared_adapters(self) -> frozenset[str]:
        """Snapshot of names visible to every tenant.

        Exposed for diagnostics + the demo's "which adapters are
        shared across tenants" panel. Returns a ``frozenset`` so a
        caller cannot mutate internal state.
        """
        return frozenset(self._shared)
