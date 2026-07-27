"""Adapter factory and builder Protocol for the connection registry.

The per-type dispatch surface that turns a ``kal_connections`` row's
``adapter_type`` + ``config`` + ``credentials`` into a live
``KnowledgeAdapter``. The same surface backs the JSON-Schema export
served by the ``GET /v1/kal/adapter-types`` endpoint.

Built-in builders auto-register on import of this module — see the
bottom of the file. Adding a new adapter type means writing a builder
class, registering it in its own module, and importing that module
here so the registration runs.
"""

from __future__ import annotations

from typing import ClassVar, Protocol

from pydantic import BaseModel

from knowlytix.kal.protocol import KnowledgeAdapter
from knowlytix.kal.types import KALQuery


class UnknownAdapterTypeError(KeyError):
    """Raised when a caller references an ``adapter_type`` slug that no
    builder has registered. Subclasses ``KeyError`` so callers that
    already handle dict-style lookup errors keep working, while
    ``isinstance(..., UnknownAdapterTypeError)`` gives a more specific
    discrimination point.
    """


class AdapterBuilder(Protocol):
    """Per-type builder. One implementation per ``adapter_type``.

    Builders own any private resources they construct: a Postgres
    builder owns its ``async_engine``, an HTTP-backed builder owns its
    ``httpx.AsyncClient``, etc. ``aclose`` releases those resources;
    the protocol does not mandate the discipline (per KAL spec §4.2
    adapters are opaque to the registry), so the builder is the place
    it lives.
    """

    config_schema: type[BaseModel]
    credentials_schema: type[BaseModel]

    async def build(
        self,
        name: str,
        config: dict[str, object],
        creds: dict[str, object],
    ) -> KnowledgeAdapter:
        # ``config`` / ``creds`` are pre-validated dicts: the service
        # layer runs them through ``config_schema`` /
        # ``credentials_schema`` before reaching the builder, so build
        # implementations can trust the shape.
        ...

    async def aclose(self, adapter: KnowledgeAdapter) -> None:
        # Idempotent — service may call aclose during disable, again on
        # delete, and again on lifespan shutdown. Implementations either
        # use natively-idempotent disposal primitives
        # (``AsyncEngine.dispose`` qualifies under SQLAlchemy 2.x) or
        # track their own "already closed" flag.
        ...

    async def handshake(
        self, adapter: KnowledgeAdapter, tenant_id: str
    ) -> None:
        # Raise ``AdapterAuthError`` / ``AdapterConnectionError`` / any
        # backend-specific subclass on failure. Default implementation
        # lives on ``BaseBuilder``.
        ...


class BaseBuilder:
    """Mixin that supplies the default ``handshake`` plus stub fields.

    Real builders should subclass this, set ``config_schema`` /
    ``credentials_schema``, and override ``build`` + ``aclose``. The
    default handshake fires the cheapest possible probe that exercises
    the auth + read path — ``query_triples`` with an empty pattern
    list and ``limit=1``. Builders for backends with cheaper liveness
    checks (e.g. SPARQL ``ASK { ?s ?p ?o }``) may override.
    """

    # Subclasses fill these in; the placeholders keep the Protocol
    # contract type-checkable.
    config_schema: type[BaseModel] = BaseModel
    credentials_schema: type[BaseModel] = BaseModel

    async def handshake(
        self, adapter: KnowledgeAdapter, tenant_id: str
    ) -> None:
        await adapter.query_triples(
            KALQuery(patterns=[], limit=1), tenant_id=tenant_id
        )


class AdapterFactory:
    """Builder registry keyed by ``adapter_type`` slug.

    Stateless w.r.t. live adapters — all live state sits on builders
    (which own engines / HTTP clients) and on the in-process
    ``AdapterRegistry``. This class only routes calls.
    """

    _builders: ClassVar[dict[str, AdapterBuilder]] = {}

    @classmethod
    def register_builder(
        cls, adapter_type: str, builder: AdapterBuilder
    ) -> None:
        # Replace semantics deliberate: tests rebuild factories
        # per-case and a built-in builder can be swapped in a fixture
        # without state leaking between cases.
        cls._builders[adapter_type] = builder

    @classmethod
    def clear_builders(cls) -> None:
        # Test-only seam: production code never clears the registry
        # (the built-in builders register at import time and stay for
        # the process lifetime). Exposed as a method rather than direct
        # ``_builders`` access so tests don't couple to the storage
        # attribute.
        cls._builders.clear()

    @classmethod
    def list_types(cls) -> list[str]:
        return list(cls._builders.keys())

    @classmethod
    def get_builder(cls, adapter_type: str) -> AdapterBuilder:
        # Single point of dispatch for both external callers (the
        # connection-service needs the per-type Pydantic schemas) and
        # the dispatch methods on this class.
        try:
            return cls._builders[adapter_type]
        except KeyError as exc:
            raise UnknownAdapterTypeError(
                f"no builder registered for adapter_type={adapter_type!r}"
            ) from exc

    @classmethod
    async def build(
        cls,
        adapter_type: str,
        name: str,
        config: dict[str, object],
        creds: dict[str, object],
    ) -> KnowledgeAdapter:
        return await cls.get_builder(adapter_type).build(name, config, creds)

    @classmethod
    async def aclose(
        cls, adapter_type: str, adapter: KnowledgeAdapter
    ) -> None:
        await cls.get_builder(adapter_type).aclose(adapter)

    @classmethod
    async def handshake(
        cls,
        adapter_type: str,
        adapter: KnowledgeAdapter,
        tenant_id: str,
    ) -> None:
        await cls.get_builder(adapter_type).handshake(adapter, tenant_id)

    @classmethod
    def config_schema_json(cls, adapter_type: str) -> dict[str, object]:
        return cls.get_builder(adapter_type).config_schema.model_json_schema()

    @classmethod
    def credentials_schema_json(cls, adapter_type: str) -> dict[str, object]:
        return cls.get_builder(adapter_type).credentials_schema.model_json_schema()


# Built-in builders register on module import. Each new adapter type
# adds its own import + register call below; the corresponding module
# is the one place the builder class itself lives.
from knowlytix.kal.adapters.neo4j_builder import Neo4jBuilder  # noqa: E402
from knowlytix.kal.adapters.postgres_builder import PostgresBuilder  # noqa: E402
from knowlytix.kal.adapters.wikidata_builder import WikidataBuilder  # noqa: E402

AdapterFactory.register_builder("postgres", PostgresBuilder())
AdapterFactory.register_builder("wikidata", WikidataBuilder())
AdapterFactory.register_builder("neo4j", Neo4jBuilder())
