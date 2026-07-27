"""Postgres ``AdapterBuilder`` for the connection registry.

Builds a ``PostgresKnowledgeAdapter`` from a ``kal_connections``
row's ``config`` + ``credentials`` and owns the lifecycle of the
per-connection ``async_engine`` so disable/delete operations dispose
the pool cleanly.

The platform's always-on ``postgres-main`` adapter (the one that
shares the app's session pool) is constructed separately in
``main.py`` and does **not** come through this builder — it has no
``kal_connections`` row and no private engine. Only customer-
configured connections route here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from sqlalchemy import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from knowlytix.kal.adapters.postgres import PostgresKnowledgeAdapter
from knowlytix.kal.factory import BaseBuilder
from knowlytix.kal.protocol import KnowledgeAdapter


class PostgresConfigSchema(BaseModel):  # type: ignore[explicit-any]
    """Non-secret Postgres connection parameters. Surfaced via the
    ``GET /v1/kal/adapter-types`` JSON-Schema endpoint so a UI can
    render the form without a code change.
    """

    host: str = Field(description="Hostname or IP of the Postgres server.")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(description="Database name to connect to.")
    # ``schema_name`` rather than ``schema`` because Pydantic v2 reserves
    # the latter (model schema introspection).
    schema_name: str = Field(
        default="public",
        description=(
            "Search-path schema. The adapter scopes all queries to this "
            "schema via the application's existing ``kg`` schema "
            "convention."
        ),
    )


class PostgresCredentialsSchema(BaseModel):  # type: ignore[explicit-any]
    """Postgres authentication payload. **Never** appears in any API
    response model — the row's ``credentials`` BYTEA stores the
    ``KALSecretBox``-encrypted form of these fields, decryption only
    happens at handshake / hydrate time, and the plaintext dict flows
    directly into ``build`` without crossing any logging boundary.
    """

    username: str
    password: str


class _RowOwnedPostgresAdapter(PostgresKnowledgeAdapter):
    """Postgres adapter whose lifetime owns the underlying engine.

    The base ``PostgresKnowledgeAdapter`` is engine-agnostic — it only
    sees a session factory. For ``kal_connections``-backed adapters we
    need a hook that the builder's ``aclose`` can call to dispose the
    engine. Holding a private reference is the simplest binding: the
    builder doesn't have to keep a side-table of ``id(adapter) →
    engine`` and the subclass relationship preserves ``isinstance``
    checks for any test that needs to confirm a row-owned adapter
    versus the always-on ``postgres-main`` instance.

    Underscore-prefixed to signal "do not construct this directly" —
    the builder is the only caller.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        name: str,
        engine: AsyncEngine,
    ) -> None:
        super().__init__(session_factory, name=name)
        self._owned_engine = engine

    async def dispose_engine(self) -> None:
        """Idempotent under SQLAlchemy 2.x: a second ``dispose()`` is a
        no-op rather than an error, so the builder's ``aclose`` can be
        called multiple times without bookkeeping (e.g. service
        disable → service delete → lifespan shutdown).
        """
        await self._owned_engine.dispose()


class PostgresBuilder(BaseBuilder):
    """Builder for ``adapter_type='postgres'``.

    Constructs a per-row ``async_engine`` from the decrypted DSN, wraps
    it in an ``async_sessionmaker``, and feeds both into a
    ``_RowOwnedPostgresAdapter`` so the engine is disposable through
    the adapter handle.

    The default ``handshake`` (inherited from ``BaseBuilder``) suffices
    here: ``query_triples`` exercises both the auth path (Postgres
    rejects the connection on bad creds) and the read path (a wrong
    schema produces a ``ProgrammingError``).
    """

    # Explicit ``type[BaseModel]`` annotation rather than relying on
    # type inference: without this the subclass attribute narrows to
    # ``type[PostgresConfigSchema]`` and breaks the ``AdapterBuilder``
    # Protocol's wider ``type[BaseModel]`` contract.
    config_schema: type[BaseModel] = PostgresConfigSchema
    credentials_schema: type[BaseModel] = PostgresCredentialsSchema

    async def build(
        self,
        name: str,
        config: dict[str, object],
        creds: dict[str, object],
    ) -> _RowOwnedPostgresAdapter:
        cfg = PostgresConfigSchema.model_validate(config)
        cr = PostgresCredentialsSchema.model_validate(creds)
        # ``URL.create`` is the only safe way to build the DSN: a
        # password containing ``@``, ``:``, ``/``, ``%`` or ``#`` would
        # corrupt an f-string-interpolated URL (an attacker-controlled
        # password could even redirect the connection to a different
        # host). SQLAlchemy URL-encodes each component for us.
        dsn = URL.create(
            "postgresql+asyncpg",
            username=cr.username,
            password=cr.password,
            host=cfg.host,
            port=cfg.port,
            database=cfg.database,
        )
        # Per-row engines stay cold most of the time, so the default
        # pool_size=5 / max_overflow=10 (= 15 backend slots per row)
        # would multiply across tenants and exhaust Postgres
        # ``max_connections``. Narrow to one keep-alive connection
        # plus a small overflow; pre-ping catches recycled stale
        # sockets without paying for an always-on keepalive thread.
        engine = create_async_engine(
            dsn,
            pool_size=1,
            max_overflow=2,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        session_factory = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )
        return _RowOwnedPostgresAdapter(
            session_factory, name=name, engine=engine
        )

    async def aclose(self, adapter: KnowledgeAdapter) -> None:
        # The factory routes ``aclose`` to the same builder that built
        # the adapter, so adapter is always the subclass we returned
        # from ``build``. ``assert`` narrows the type for mypy without
        # the defensive-no-op branch the previous version had.
        assert isinstance(adapter, _RowOwnedPostgresAdapter)
        await adapter.dispose_engine()
