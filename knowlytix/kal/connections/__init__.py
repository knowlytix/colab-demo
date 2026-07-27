"""Persistent registry of user-registered KG adapters (KAL connections).

Hosts the persistence layer (table, SQLAlchemy model, schema /
constraint constants the migration and the service share) plus the
``ConnectionService`` that orchestrates the
validate → handshake → encrypt → persist → register create flow.

This module is the **library surface** of the connection registry.
v0.2.0 does not ship a FastAPI router — consumers wire their own
HTTP layer over ``ConnectionService``, calling its methods with
their own auth context. The reference router (with FastAPI ``Depends``
abstraction for auth) lands in v0.3.0 once the dep-injection shape
is settled.
"""

from knowlytix.kal.connections.models import (
    ADAPTER_TYPE_MAX_LEN,
    DEFAULT_STATUS,
    DISABLED_STATUS,
    NAME_MAX_LEN,
    STATUS_MAX_LEN,
    UNIQUE_CONSTRAINT_NAME,
    KALConnection,
    KALConnectionStatus,
)
from knowlytix.kal.connections.repository import KALConnectionNotFoundError
from knowlytix.kal.connections.schemas import (
    AdapterTypeInfo,
    ConnectionCreateRequest,
    ConnectionResponse,
    ConnectionUpdateRequest,
)
from knowlytix.kal.connections.service import (
    ConnectionService,
    DuplicateNameError,
    HandshakeFailedError,
    TenantConnectionCapError,
)

__all__ = [
    "ADAPTER_TYPE_MAX_LEN",
    "DEFAULT_STATUS",
    "DISABLED_STATUS",
    "NAME_MAX_LEN",
    "STATUS_MAX_LEN",
    "UNIQUE_CONSTRAINT_NAME",
    "AdapterTypeInfo",
    "ConnectionCreateRequest",
    "ConnectionResponse",
    "ConnectionService",
    "ConnectionUpdateRequest",
    "DuplicateNameError",
    "HandshakeFailedError",
    "KALConnection",
    "KALConnectionNotFoundError",
    "KALConnectionStatus",
    "TenantConnectionCapError",
]
