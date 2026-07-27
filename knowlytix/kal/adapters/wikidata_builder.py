"""Builder for ``adapter_type='wikidata'``.

Builds a ``WikidataAdapter`` from the row's config + (optional)
credentials. Wikidata's public endpoint requires no auth, so the
``credentials_schema`` is intentionally minimal — a ``bearer_token``
field that callers leave empty for the public endpoint.

Compared to ``PostgresBuilder`` this is a structurally simpler
builder: there's no per-row engine to dispose, just the httpx
``AsyncClient`` the adapter owns. ``aclose`` calls
``adapter.aclose()`` which delegates to the client.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from knowlytix.kal.adapters.wikidata import (
    DEFAULT_WIKIDATA_ENDPOINT,
    WikidataAdapter,
)
from knowlytix.kal.factory import BaseBuilder
from knowlytix.kal.protocol import KnowledgeAdapter
from knowlytix.kal.types import KALQuery

# Knowly-identifying User-Agent string. Wikidata policy requires a
# descriptive UA that includes a contact path; sending a generic
# ``python-httpx`` gets rate-limited quickly. The string itself is
# pinned to the project + an issue tracker URL so operators can be
# reached if a query pattern misbehaves.
_DEFAULT_USER_AGENT: str = (
    "Knowly/0.1 (+https://github.com/knowlytix/knowly) "
    "knowly-kal-wikidata-adapter"
)


class WikidataConfigSchema(BaseModel):  # type: ignore[explicit-any]
    """Non-secret Wikidata connection parameters."""

    endpoint_url: str = Field(
        default=DEFAULT_WIKIDATA_ENDPOINT,
        description=(
            "SPARQL endpoint URL. Defaults to the public Wikidata Query "
            "Service. Override for self-hosted Wikibase mirrors."
        ),
    )
    user_agent: str = Field(
        default=_DEFAULT_USER_AGENT,
        description=(
            "User-Agent header sent on every request. Wikidata "
            "rate-limits anonymous traffic — operators running a "
            "deployment should override with a project-identifying "
            "string + contact URL."
        ),
    )


class WikidataCredentialsSchema(BaseModel):  # type: ignore[explicit-any]
    """Optional Wikidata authentication payload.

    The public endpoint is unauthenticated. Self-hosted Wikibase
    mirrors or enterprise endpoints may require a bearer token;
    leave empty otherwise. The field is structurally separate from
    ``WikidataConfigSchema`` so the credentials JSON Schema is
    distinguishable in the ``GET /v1/kal/adapter-types`` response.
    """

    bearer_token: str = Field(
        default="",
        description=(
            "Optional Bearer token for a private SPARQL endpoint. "
            "Leave empty for the public Wikidata Query Service."
        ),
    )


class WikidataBuilder(BaseBuilder):
    """Builder for ``adapter_type='wikidata'``."""

    # Mirrors the ``PostgresBuilder`` pattern: explicit
    # ``type[BaseModel]`` annotation widens the subclass narrowing.
    config_schema: type[BaseModel] = WikidataConfigSchema
    credentials_schema: type[BaseModel] = WikidataCredentialsSchema

    async def build(
        self,
        name: str,
        config: dict[str, object],
        creds: dict[str, object],
    ) -> WikidataAdapter:
        cfg = WikidataConfigSchema.model_validate(config)
        cr = WikidataCredentialsSchema.model_validate(creds)
        return WikidataAdapter(
            name=name,
            endpoint_url=cfg.endpoint_url,
            user_agent=cfg.user_agent,
            bearer_token=cr.bearer_token or None,
        )

    async def aclose(self, adapter: KnowledgeAdapter) -> None:
        # The factory routes ``aclose`` only to the builder that
        # built the adapter; ``assert`` narrows the type for mypy.
        # WikidataAdapter.aclose disposes the owned httpx client.
        assert isinstance(adapter, WikidataAdapter)
        await adapter.aclose()

    async def handshake(
        self, adapter: KnowledgeAdapter, tenant_id: str
    ) -> None:
        # Override the default ``query_triples(KALQuery(patterns=[]))``
        # handshake with a Wikidata-specific minimum-cost probe.
        # ``ASK { ?s ?p ?o }`` is cheaper than the default ``SELECT
        # ... LIMIT 1`` — the endpoint returns a bare boolean and
        # the parser can short-circuit. The adapter's own SELECT
        # path also works; we go through it (rather than constructing
        # a custom ASK) so all of the adapter's error-classification
        # is shared with the runtime path.
        assert isinstance(adapter, WikidataAdapter)
        await adapter.query_triples(
            KALQuery(patterns=[], limit=1), tenant_id=tenant_id
        )
