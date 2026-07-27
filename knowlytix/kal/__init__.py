"""Knowledge Adapter Layer (KAL) public surface.

Re-exports the canonical KAL contract — types, the ``KnowledgeAdapter``
and ``KGStore`` Protocols, ``AdapterCapabilities``, the federation
primitives (``FederationRouter`` and ``ConflictStrategy``), and the
adapter-error hierarchy. The ``AdapterRegistry`` setup-time helper
stays in ``knowlytix.kal.registry``; concrete adapter implementations
stay in ``knowlytix.kal.adapters.*`` and the default storage impl
stays in ``knowlytix.kal.storage_default`` (their heavy imports —
SQLAlchemy, asyncpg, pgvector — should not be paid by a third-party
adapter or store author whose code only needs the contract surface).
"""

from knowlytix.kal.errors import (
    AdapterAuthError,
    AdapterConnectionError,
    AdapterNotFoundError,
    AdapterWriteError,
    CapabilityNotSupportedError,
    KALError,
    TenantRequiredError,
)
from knowlytix.kal.federation import ConflictStrategy, FederationRouter
from knowlytix.kal.protocol import AdapterCapabilities, KnowledgeAdapter
from knowlytix.kal.storage import KGStore
from knowlytix.kal.tenant_index import TenantIndex
from knowlytix.kal.types import (
    AdapterError,
    AdapterErrorType,
    AdjacencyDirection,
    KALLiteral,
    KALNode,
    KALQuery,
    KALQueryResult,
    KALTriple,
    KALTriplePattern,
    TripleProvenance,
    VerificationMetadata,
)

__all__ = [
    "AdapterAuthError",
    "AdapterCapabilities",
    "AdapterConnectionError",
    "AdapterError",
    "AdapterErrorType",
    "AdapterNotFoundError",
    "AdapterWriteError",
    "AdjacencyDirection",
    "CapabilityNotSupportedError",
    "ConflictStrategy",
    "FederationRouter",
    "KALError",
    "KALLiteral",
    "KALNode",
    "KALQuery",
    "KALQueryResult",
    "KALTriple",
    "KALTriplePattern",
    "KGStore",
    "KnowledgeAdapter",
    "TenantIndex",
    "TenantRequiredError",
    "TripleProvenance",
    "VerificationMetadata",
]
