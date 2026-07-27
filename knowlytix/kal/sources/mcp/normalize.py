"""Normalize MCP content into plain text for the extraction pipeline.

Two MCP shapes need handling:

- **resource reads** (``resources/read``) yield ``TextResourceContents``
  (``.text``) / ``BlobResourceContents`` (``.blob``);
- **tool results** (``tools/call``) yield ``TextContent`` /
  ``ImageContent`` / ``EmbeddedResource`` (+ others).

Only text is usable by a text extractor; image / blob blocks are skipped
(debug-logged), not failed. An item that yields no text comes back as the
empty string, which the connector counts as skipped.
"""

from __future__ import annotations

import logging

from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ReadResourceResult,
    TextContent,
    TextResourceContents,
)

logger = logging.getLogger(__name__)

_JOIN = "\n\n"


def normalize_resource_contents(result: ReadResourceResult) -> str:
    """Join the text of a ``resources/read`` result; skip binary blobs."""
    parts: list[str] = []
    skipped = 0
    for content in result.contents:
        if isinstance(content, TextResourceContents):
            parts.append(content.text)
        else:
            skipped += 1
    if skipped:
        logger.debug("skipped %d non-text resource content block(s)", skipped)
    return _JOIN.join(parts)


def normalize_tool_result(result: CallToolResult) -> str:
    """Join the text of a ``tools/call`` result; skip image / audio / blob."""
    parts: list[str] = []
    skipped = 0
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, EmbeddedResource) and isinstance(
            block.resource, TextResourceContents
        ):
            parts.append(block.resource.text)
        else:
            skipped += 1
    if skipped:
        logger.debug("skipped %d non-text tool content block(s)", skipped)
    return _JOIN.join(parts)
