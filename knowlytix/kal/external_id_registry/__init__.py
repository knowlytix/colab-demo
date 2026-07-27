"""Federation metadata store — cross-adapter identity resolution.

KAL spec §6.4 Layer 4. Maps ``(tenant_id, source_adapter, external_id,
resource_type) → internal_id`` so the same logical entity reachable
through different backend-specific identifiers (Postgres UUIDs, Neo4j
node IDs, SPARQL URIs, etc.) collapses to a single internal handle.

Postgres is the PoC choice for the reference deployment; a customer
running a non-Postgres primary KG could put the registry in
MySQL/MariaDB or any RDBMS without changing the adapter contract.

Public surface:
- ``ExternalIdRegistry`` — SQLAlchemy ORM model (table
  ``knowlytix.external_id_registry``).
- ``register`` — insert one mapping; idempotent via
  ``ON CONFLICT DO NOTHING``.
- ``bulk_register`` — batched variant; returns count of newly inserted rows.
- ``lookup_internal_id`` — forward lookup
  (``(source, ext_id) → internal_id``); returns ``None`` when unknown.
- ``lookup_external_ids`` — reverse lookup
  (``internal_id → {source: ext_id}``); returns an empty dict when no
  mappings exist for the internal_id.
- ``ResourceType`` — ``Literal["entity", "triple"]`` narrowing the
  resource_type field at the function boundary.
"""

from knowlytix.kal.external_id_registry.models import (
    ExternalIdRegistry,
    ResourceType,
)
from knowlytix.kal.external_id_registry.repository import (
    bulk_register,
    lookup_external_ids,
    lookup_internal_id,
    register,
)

__all__ = [
    "ExternalIdRegistry",
    "ResourceType",
    "bulk_register",
    "lookup_external_ids",
    "lookup_internal_id",
    "register",
]
