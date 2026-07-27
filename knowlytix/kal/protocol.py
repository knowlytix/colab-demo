"""KnowledgeAdapter Protocol + AdapterCapabilities.

Per KAL spec v1.4.1 sections 4.1-4.2. Mypy-strict; no ``Any``.

Naming convention:
- ``get_*``    — fetch a single item by ID
- ``query_*``  — fetch a collection by filter/pattern
- ``search_*`` — similarity/semantic search, ranked
- ``insert_*`` / ``update_*`` / ``delete_*`` — standard CRUD verbs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from knowlytix.kal.types import (
    KALNode,
    KALQuery,
    KALQueryResult,
    KALTriple,
    VerificationMetadata,
)


@dataclass(frozen=True)
class AdapterCapabilities:
    """Adapter feature declaration. Frozen so caps are advertised, not mutated.

    Federation router inspects capabilities to filter routing; consumers
    should check before assuming an operation is supported.
    """

    can_read: bool = True
    can_write: bool = False
    can_delete: bool = False
    supports_vector_search: bool = False
    supports_text_search: bool = False
    supports_provenance: bool = False
    supports_verification_metadata: bool = False
    supports_atomic_writes: bool = False
    max_batch_size: int = 1000


@runtime_checkable
class KnowledgeAdapter(Protocol):
    """Protocol for knowledge graph adapters.

    Adapters bridge the verification layer and any KG backend. ``@runtime_checkable``
    only validates attribute presence — mypy is the real signature validator.
    """

    @property
    def name(self) -> str:
        """Unique adapter name (e.g. ``"postgres-main"``)."""
        ...

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Declare what this adapter supports."""
        ...

    # --- Read operations ---

    async def query_triples(
        self,
        query: KALQuery,
        tenant_id: str | None = None,
    ) -> KALQueryResult:
        """Query triples matching the given patterns."""
        ...

    async def get_node(
        self,
        node_id: str,
        tenant_id: str | None = None,
    ) -> KALNode | None:
        """Fetch a single node by ID. Returns None if not found."""
        ...

    async def query_adjacent_triples(
        self,
        node_id: str,
        predicate: str | None = None,
        direction: str = "outgoing",
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[KALTriple]:
        """Query triples connected to a node."""
        ...

    # --- Write operations (optional, guarded by capabilities) ---

    async def insert_triples(
        self,
        triples: list[KALTriple],
        tenant_id: str | None = None,
    ) -> int:
        """Insert triples, returning count of newly created (after dedup)."""
        ...

    async def update_verification(
        self,
        triple_id: str,
        verification: VerificationMetadata,
        tenant_id: str | None = None,
    ) -> bool:
        """Update verification metadata on a triple. Returns success."""
        ...

    async def delete_triples(
        self,
        triple_ids: list[str],
        tenant_id: str | None = None,
    ) -> int:
        """Delete triples by ID. Returns count actually deleted."""
        ...

    async def quarantine_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        """Soft-revoke (quarantine) every live triple whose
        ``provenance.source == source``. Returns the count newly
        quarantined. Write-capable backends implement this; read-only
        adapters raise ``CapabilityNotSupportedError``."""
        ...

    async def restore_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        """Reverse a quarantine for ``source``. Returns the count restored."""
        ...

    # --- Vector search (optional, guarded by capabilities) ---

    async def search_similar_nodes(
        self,
        query_vector: list[float],
        tenant_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[KALNode, float]]:
        """Find nodes similar to ``query_vector``. Returns (node, distance) pairs."""
        ...
