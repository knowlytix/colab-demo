"""Builder for ``adapter_type='neo4j'``.

Builds a ``Neo4jKnowledgeAdapter`` from the row's config + credentials.
The split mirrors ``PostgresBuilder`` / ``WikidataBuilder``: connection
shape (``uri`` / ``database`` / ``node_key``) is non-secret config; the
auth pair (``user`` / ``password``) is credentials, so the password
stays in the credentials JSON Schema served by
``GET /v1/kal/adapter-types`` rather than leaking into the config schema.

``aclose`` closes the driver the adapter owns. The default
``BaseBuilder.handshake`` (an empty ``query_triples`` with ``limit=1``)
fires the cheapest read that still exercises the auth + Bolt path, so
no override is needed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from knowlytix.kal.adapters.neo4j import (
    DEFAULT_NEO4J_URI,
    DEFAULT_NODE_KEY,
    Neo4jKnowledgeAdapter,
)
from knowlytix.kal.factory import BaseBuilder
from knowlytix.kal.protocol import KnowledgeAdapter


class Neo4jConfigSchema(BaseModel):  # type: ignore[explicit-any]
    """Non-secret Neo4j connection parameters."""

    uri: str = Field(
        default=DEFAULT_NEO4J_URI,
        description=(
            "Bolt URI of the Neo4j server (e.g. bolt://host:7687 or "
            "neo4j+s://host for an Aura instance)."
        ),
    )
    database: str | None = Field(
        default=None,
        description=(
            "Target database name. Leave empty for the server's default "
            "database (DBMS-wide 'neo4j' on a single-database instance)."
        ),
    )
    node_key: str = Field(
        default=DEFAULT_NODE_KEY,
        description=(
            "Node property used as the KAL node id/name when projecting "
            "the property graph to triples. Defaults to 'name'."
        ),
    )


class Neo4jCredentialsSchema(BaseModel):  # type: ignore[explicit-any]
    """Neo4j authentication payload."""

    user: str = Field(
        default="neo4j",
        description="Neo4j username. Defaults to the built-in 'neo4j' user.",
    )
    password: str = Field(
        default="",
        description="Neo4j password for the configured user.",
    )


class Neo4jBuilder(BaseBuilder):
    """Builder for ``adapter_type='neo4j'``."""

    config_schema: type[BaseModel] = Neo4jConfigSchema
    credentials_schema: type[BaseModel] = Neo4jCredentialsSchema

    async def build(
        self,
        name: str,
        config: dict[str, object],
        creds: dict[str, object],
    ) -> Neo4jKnowledgeAdapter:
        cfg = Neo4jConfigSchema.model_validate(config)
        cr = Neo4jCredentialsSchema.model_validate(creds)
        return Neo4jKnowledgeAdapter(
            name=name,
            uri=cfg.uri,
            user=cr.user,
            password=cr.password,
            database=cfg.database,
            node_key=cfg.node_key,
        )

    async def aclose(self, adapter: KnowledgeAdapter) -> None:
        # The factory routes ``aclose`` only to the builder that built
        # the adapter; ``assert`` narrows the type for mypy.
        assert isinstance(adapter, Neo4jKnowledgeAdapter)
        await adapter.aclose()
