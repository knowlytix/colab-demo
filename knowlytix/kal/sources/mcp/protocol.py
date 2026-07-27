"""Typed boundary for the MCP ingestion source — config + result models
+ the ``ClaimExtractor`` seam.

Import-safe without the ``mcp`` SDK: pure Pydantic + ``typing``. The
connector (``connector.py``) and client (``client.py``) import the SDK;
this module deliberately does not, so ``McpSourceConfig`` validation and
the result envelope are usable from the FastAPI layer without the
``[mcp]`` extra installed.

The ``# type: ignore[explicit-any]`` on each ``BaseModel`` subclass
matches the project-wide Pydantic-v2 / mypy-strict suppression used in
``knowly.kal.types`` and ``knowly.modules.extraction.schemas``.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from knowlytix.kal.types import KALTriple


@runtime_checkable
class ClaimExtractor(Protocol):
    """Turns a chunk of source text into typed KAL triples.

    The connector is generic over this seam so the public package ships
    no concrete (LLM-dependent, possibly proprietary) extractor. Knowly
    injects a backend wrapping ``extract_claims`` + ``claims_to_kal_triples``;
    package consumers inject their own.

    Returning ``list[KALTriple]`` (not the knowly-internal
    ``ExtractedClaim``) keeps this boundary free of
    ``knowly.modules.extraction`` types. ``source_uri`` lets an extractor
    record where the text came from; the connector also stamps provenance
    independently.
    """

    async def extract(
        self, text: str, *, source_uri: str
    ) -> list[KALTriple]: ...


@runtime_checkable
class TripleInserter(Protocol):
    """Persistence sink for ingested triples.

    The connector depends on this narrow surface rather than the concrete
    ``FederationRouter`` (which satisfies it structurally) — dependency
    inversion so the connector is testable with a fake sink and free of a
    federation import.
    """

    async def insert_triples(
        self, triples: list[KALTriple], tenant_id: str | None = None
    ) -> int: ...


class McpToolSpec(BaseModel):  # type: ignore[explicit-any]
    """An operator-allowlisted MCP tool to call, with its argument template.

    Phase 1 never lets an LLM choose tool calls — the operator names the
    tool and supplies the args. An empty ``McpSourceConfig.tools`` list
    means no tools are called at all.
    """

    name: str
    args: dict[str, object] = {}


class McpSourceCredentials(BaseModel):  # type: ignore[explicit-any]
    """Secret half of an MCP source connection.

    Only the Streamable-HTTP transport needs a credential; stdio spawns a
    local process and leaves this empty.
    """

    bearer_token: str = ""


class McpSourceConfig(BaseModel):  # type: ignore[explicit-any]
    """Non-secret configuration for one external MCP source."""

    name: str
    transport: Literal["stdio", "http"]
    # stdio transport: spawn a local MCP server process.
    command: str | None = None
    command_args: list[str] = []
    # http transport: connect to a remote Streamable-HTTP MCP server.
    url: str | None = None
    # Pull selection.
    resource_include: list[str] = ["*"]
    resource_exclude: list[str] = []
    tools: list[McpToolSpec] = []
    predicate_vocabulary: dict[str, str] = {}
    # Hard cap on paginated resource pulls so a runaway source can't
    # ingest unbounded data in a single run. ``ge=1`` — a zero/negative
    # cap is a configuration mistake, not a "pull nothing" request.
    max_resources: int = Field(default=1000, ge=1)

    @model_validator(mode="after")
    def _check_transport(self) -> McpSourceConfig:
        # Fail fast at config time, not at connect time.
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio transport requires `command`")
        if self.transport == "http" and not self.url:
            raise ValueError("http transport requires `url`")
        return self


class IngestItemError(BaseModel):  # type: ignore[explicit-any]
    """A single per-item failure during an ingest run.

    Mirrors ``KALQueryResult.errors`` semantics — one bad resource/tool
    is recorded here and the run continues, rather than aborting the
    whole batch.
    """

    source_uri: str
    error_type: str
    message: str


class IngestResult(BaseModel):  # type: ignore[explicit-any]
    """Outcome of a one-shot ingest — counts + a degraded-but-not-failed
    error envelope.
    """

    triples_inserted: int = 0
    triples_extracted: int = 0
    resources_pulled: int = 0
    tools_called: int = 0
    items_skipped: int = 0
    errors: list[IngestItemError] = []


class SourceManifest(BaseModel):  # type: ignore[explicit-any]
    """What an external MCP server exposes — resource URIs + tool names."""

    resources: list[str] = []
    tools: list[str] = []
