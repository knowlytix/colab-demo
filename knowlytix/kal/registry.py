"""KAL adapter registry.

Per KAL spec v1.4.1 §6. Module-level singleton + ``get_registry()`` getter
+ ``register_adapter()`` helper — matches the global-singleton-with-setter
pattern used elsewhere in the codebase (``kg/service.py``).

The registry keys adapters by ``adapter.name`` so a single
``register(adapter)`` call is enough — no separate name argument. The
federation router reads from the same registry (Story 8) and routes
based on capabilities advertised by each adapter.

Test isolation: tests that exercise registry behavior construct a fresh
``AdapterRegistry()`` rather than touching the module-level singleton —
see ``tests/unit/kal/test_registry.py`` for the pattern. The singleton
is only mutated during the FastAPI app's lifespan (dev environment), so
unit tests don't have to snapshot-and-restore.
"""

from __future__ import annotations

from knowlytix.kal.errors import AdapterNotFoundError
from knowlytix.kal.protocol import KnowledgeAdapter


class AdapterRegistry:
    """In-process registry of ``KnowledgeAdapter`` instances.

    Not thread-safe by itself — the running app is single-threaded async,
    and the registry is mutated only during lifespan startup / shutdown.
    Test isolation is the caller's responsibility (snapshot fixture).
    """

    def __init__(self) -> None:
        # Preserve insertion order so write-fallback ("first writable
        # adapter") is deterministic, per KAL spec §5 write routing.
        self._adapters: dict[str, KnowledgeAdapter] = {}

    def register(self, adapter: KnowledgeAdapter) -> None:
        """Register an adapter by its ``adapter.name``.

        Re-registering with the same name replaces the prior entry —
        useful for tests + dev hot-reload; harmless in production where
        adapters are constructed once at lifespan startup.
        """
        self._adapters[adapter.name] = adapter

    def unregister(self, name: str) -> None:
        """Remove an adapter. Silently no-ops on unknown names so
        idempotent shutdown / teardown paths don't have to guard.
        """
        self._adapters.pop(name, None)

    def get(self, name: str) -> KnowledgeAdapter:
        """Fetch an adapter by name. Raises ``AdapterNotFoundError`` when
        the name isn't registered — per KAL spec §11.3 errors module.
        """
        if name not in self._adapters:
            raise AdapterNotFoundError(
                f"no adapter registered with name {name!r}"
            )
        return self._adapters[name]

    def list_adapters(self) -> list[str]:
        """Return the names of registered adapters in registration order."""
        return list(self._adapters.keys())

    def all_adapters(self) -> list[KnowledgeAdapter]:
        """Return registered adapter instances in registration order.

        Helper for the federation router which fans queries out to every
        registered adapter (capability-filtered).
        """
        return list(self._adapters.values())

    def get_writable(self) -> list[KnowledgeAdapter]:
        """Adapters with ``can_write=True`` in registration order."""
        return [a for a in self._adapters.values() if a.capabilities.can_write]

    def get_with_capability(self, capability: str) -> list[KnowledgeAdapter]:
        """Adapters whose ``capabilities`` dataclass has ``capability`` truthy.

        ``capability`` is a field name on ``AdapterCapabilities``
        (e.g., ``"supports_vector_search"``, ``"supports_text_search"``).
        Unknown attribute names raise ``AttributeError`` — that's a
        programmer error, not a runtime condition worth defending.
        """
        return [
            a for a in self._adapters.values()
            if getattr(a.capabilities, capability)
        ]


# Module-level singleton. Mutated by ``main.py``'s lifespan on startup,
# read by the federation router on every query.
_registry = AdapterRegistry()


def get_registry() -> AdapterRegistry:
    """Access the process-wide adapter registry."""
    return _registry


def register_adapter(adapter: KnowledgeAdapter) -> None:
    """Shorthand for ``get_registry().register(adapter)``."""
    _registry.register(adapter)
