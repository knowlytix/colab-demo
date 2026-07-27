"""KAL data-source connectors — KAL acting as a *client* of external systems.

Distinct from ``knowly.kal.adapters`` (which wrap backends that already
hold triples) and from ``knowly.kal.mcp`` (which exposes KAL *as* an MCP
server). A source pulls unstructured data from an external system and
runs it through extraction to produce ``KALTriple``s — it sits in front
of the data plane as a producer, not as a query surface.
"""

from __future__ import annotations
