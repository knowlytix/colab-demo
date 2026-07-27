"""MCP ingestion connector — pulls an external MCP server into KAL.

Pipeline: discover → pull (resources + allowlisted tools) → normalize
(content blocks → text) → extract (injected ``ClaimExtractor``) → stamp
provenance (source / extractor / extracted_at) → persist (via a
``TripleInserter`` sink, e.g. ``FederationRouter``).

This is **not** a ``KnowledgeAdapter`` — it produces triples into the
data plane rather than answering queries over it. It's generic over the
``ClaimExtractor`` seam so the public package ships no concrete extractor;
the connector itself never imports the ``mcp`` SDK (the injected client
does), keeping it light.

Per-item failures (a bad resource / tool) are recorded in
``IngestResult.errors`` and the run continues, mirroring
``KALQueryResult.errors``. A persist-step failure is *not* swallowed —
``FederationRouter.insert_triples`` raises ``AdapterWriteError`` per KAL
spec §5.3 (writes don't silently degrade), which propagates to the caller.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

from knowlytix.kal.errors import AdapterConnectionError
from knowlytix.kal.sources.mcp.normalize import (
    normalize_resource_contents,
    normalize_tool_result,
)
from knowlytix.kal.sources.mcp.protocol import (
    IngestItemError,
    IngestResult,
    SourceManifest,
)
from knowlytix.kal.types import KALTriple, TripleProvenance

if TYPE_CHECKING:
    from knowlytix.kal.sources.mcp.client import McpClientSession
    from knowlytix.kal.sources.mcp.protocol import (
        ClaimExtractor,
        McpSourceConfig,
        TripleInserter,
    )

logger = logging.getLogger(__name__)

_CONNECTION_ERROR = "connection"


def _filter_globs(
    uris: list[str], include: list[str], exclude: list[str]
) -> list[str]:
    """Keep URIs matching any ``include`` glob and no ``exclude`` glob.

    Uses ``fnmatchcase`` (not ``fnmatch``) so URI matching is
    case-sensitive and identical across platforms — ``fnmatch`` lowercases
    via ``os.path.normcase`` on macOS/Windows, which would make globs
    behave differently than on Linux.
    """
    kept: list[str] = []
    for uri in uris:
        if not any(fnmatchcase(uri, pat) for pat in include):
            continue
        if any(fnmatchcase(uri, pat) for pat in exclude):
            continue
        kept.append(uri)
    return kept


def _stamp(
    triples: list[KALTriple], *, source: str, extractor: str
) -> list[KALTriple]:
    """Stamp MCP provenance onto extracted triples (generic + mirrored).

    ``source`` becomes ``mcp://<source>`` (the revoke-by-source key);
    ``extractor`` identifies the resource / tool that produced the text.
    Any extractor-set provenance (e.g. ``source_text``) is preserved.
    """
    now = datetime.now(tz=UTC)
    stamped: list[KALTriple] = []
    for triple in triples:
        base = triple.provenance or TripleProvenance()
        prov = base.model_copy(
            update={
                "source": f"mcp://{source}",
                "extractor": extractor,
                "extracted_at": now,
            }
        )
        stamped.append(triple.model_copy(update={"provenance": prov}))
    return stamped


class McpIngestConnector:
    """Pulls an external MCP server and ingests it into KAL."""

    def __init__(
        self,
        *,
        client: McpClientSession,
        extractor: ClaimExtractor,
        inserter: TripleInserter,
        config: McpSourceConfig,
    ) -> None:
        self._client = client
        self._extractor = extractor
        self._inserter = inserter
        self._config = config

    async def discover(self) -> SourceManifest:
        """List what the external server exposes (resources + tools)."""
        return SourceManifest(
            resources=await self._client.list_resources(
                self._config.max_resources
            ),
            tools=await self._client.list_tools(),
        )

    async def ingest(self, tenant_id: str) -> IngestResult:
        """One-shot pull → extract → stamp → persist."""
        result = IngestResult()
        triples: list[KALTriple] = []
        await self._pull_resources(result, triples)
        await self._pull_tools(result, triples)
        if triples:
            result.triples_inserted = await self._inserter.insert_triples(
                triples, tenant_id=tenant_id
            )
        return result

    async def _pull_resources(
        self, result: IngestResult, triples: list[KALTriple]
    ) -> None:
        try:
            uris = await self._client.list_resources(self._config.max_resources)
        except AdapterConnectionError as exc:
            result.errors.append(
                IngestItemError(
                    source_uri="<resources/list>",
                    error_type=_CONNECTION_ERROR,
                    message=str(exc),
                )
            )
            return
        selected = _filter_globs(
            uris,
            self._config.resource_include,
            self._config.resource_exclude,
        )
        for uri in selected:
            await self._ingest_resource(uri, result, triples)

    async def _ingest_resource(
        self, uri: str, result: IngestResult, triples: list[KALTriple]
    ) -> None:
        try:
            contents = await self._client.read_resource(uri)
        except AdapterConnectionError as exc:
            result.errors.append(
                IngestItemError(
                    source_uri=uri,
                    error_type=_CONNECTION_ERROR,
                    message=str(exc),
                )
            )
            return
        text = normalize_resource_contents(contents)
        if not text:
            result.items_skipped += 1
            return
        if await self._extract_stamp_collect(
            text,
            source_uri=uri,
            extractor_label=f"mcp:resource:{uri}",
            result=result,
            triples=triples,
        ):
            result.resources_pulled += 1

    async def _extract_stamp_collect(
        self,
        text: str,
        *,
        source_uri: str,
        extractor_label: str,
        result: IngestResult,
        triples: list[KALTriple],
    ) -> bool:
        """Extract → stamp → collect for one item. On extractor failure,
        record a per-item error and return ``False`` so one bad item
        doesn't abort the batch (the ``IngestResult.errors`` envelope).
        """
        try:
            extracted = await self._extractor.extract(text, source_uri=source_uri)
        except Exception as exc:
            # Extraction runs an LLM for the knowly extractor — anything
            # can surface. Batch-resilient by design.
            result.errors.append(
                IngestItemError(
                    source_uri=source_uri,
                    error_type="extraction",
                    message=str(exc),
                )
            )
            return False
        triples.extend(
            _stamp(
                extracted, source=self._config.name, extractor=extractor_label
            )
        )
        result.triples_extracted += len(extracted)
        return True

    async def _pull_tools(
        self, result: IngestResult, triples: list[KALTriple]
    ) -> None:
        for spec in self._config.tools:
            tool_uri = f"tool:{spec.name}"
            try:
                tool_result = await self._client.call_tool(spec.name, spec.args)
            except AdapterConnectionError as exc:
                result.errors.append(
                    IngestItemError(
                        source_uri=tool_uri,
                        error_type=_CONNECTION_ERROR,
                        message=str(exc),
                    )
                )
                continue
            text = normalize_tool_result(tool_result)
            if tool_result.isError:
                # A failing tool returns its error text in the content —
                # don't extract triples from an error message.
                result.errors.append(
                    IngestItemError(
                        source_uri=tool_uri,
                        error_type="tool_error",
                        message=(text or "tool returned isError")[:200],
                    )
                )
                continue
            if not text:
                result.items_skipped += 1
                continue
            if await self._extract_stamp_collect(
                text,
                source_uri=tool_uri,
                extractor_label=f"mcp:tool:{spec.name}",
                result=result,
                triples=triples,
            ):
                result.tools_called += 1
