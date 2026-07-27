"""KAL-as-MCP-client ingestion source (v0.10.0).

KAL connects to an *external* MCP server, pulls its resources +
allowlisted tool results, runs them through an injected ``ClaimExtractor``
to produce typed ``KALTriple``s, stamps MCP provenance, and persists via
``FederationRouter.insert_triples``. This is the inbound mirror of the
v0.6.0 ``knowly.kal.mcp`` server (which exposes KAL *as* a server).

The connector is generic over the ``ClaimExtractor`` seam so the public
package ships no concrete (LLM-dependent, possibly proprietary) extractor.

``protocol.py`` is import-safe without the ``mcp`` SDK (pure Pydantic +
typing). ``client.py`` / ``connector.py`` import the SDK and are deferred
behind ``__getattr__`` so ``import knowlytix.kal.sources.mcp`` works without
the ``[mcp]`` extra installed.
"""

from __future__ import annotations

# Always-safe re-exports (no mcp SDK dependency).
from knowlytix.kal.sources.mcp.protocol import (
    ClaimExtractor,
    IngestItemError,
    IngestResult,
    McpSourceConfig,
    McpSourceCredentials,
    McpToolSpec,
    SourceManifest,
    TripleInserter,
)

__all__ = [
    "ClaimExtractor",
    "IngestItemError",
    "IngestResult",
    "McpClientSession",
    "McpIngestConnector",
    "McpSourceConfig",
    "McpSourceCredentials",
    "McpToolSpec",
    "SourceManifest",
    "TripleInserter",
]


def __getattr__(name: str) -> object:
    """Lazy export — defer the ``mcp`` SDK import until first use."""
    if name == "McpClientSession":
        from knowlytix.kal.sources.mcp.client import McpClientSession

        return McpClientSession
    if name == "McpIngestConnector":
        from knowlytix.kal.sources.mcp.connector import McpIngestConnector

        return McpIngestConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
