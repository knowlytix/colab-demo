"""Pydantic request/response schemas for the connection-registry API.

Two strict invariants on the response surface:

- ``ConnectionResponse`` has no ``credentials`` field at all. The
  encrypted bytes never reach a response model — a leak via the API
  would be a type error, not a code-review oversight.
- ``adapter_type`` and ``status`` reuse the model's length / literal
  constraints so the API surface and the persistence layer share one
  source of truth.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from knowlytix.kal.connections.models import (
    ADAPTER_TYPE_MAX_LEN,
    NAME_MAX_LEN,
    KALConnection,
    KALConnectionStatus,
)


class ConnectionCreateRequest(BaseModel):  # type: ignore[explicit-any]
    """Body of ``POST /v1/kal/connections``.

    ``config`` and ``credentials`` are dicts — the service runs them
    through the per-type Pydantic schemas the factory exposes, so any
    type-shape errors surface as 422 rather than as a builder error
    deep inside the create flow.
    """

    name: str = Field(min_length=1, max_length=NAME_MAX_LEN)
    adapter_type: str = Field(min_length=1, max_length=ADAPTER_TYPE_MAX_LEN)
    config: dict[str, object]
    credentials: dict[str, object]


class ConnectionUpdateRequest(BaseModel):  # type: ignore[explicit-any]
    """Body of ``PATCH /v1/kal/connections/{id}``.

    Both fields are optional but at least one must be present —
    enforced at the endpoint level so an empty body returns 400
    rather than Pydantic's auto-422. Credentials are all-or-nothing:
    a partial merge would require decrypting first and we don't need
    that flexibility.
    """

    config: dict[str, object] | None = None
    credentials: dict[str, object] | None = None


class ConnectionResponse(BaseModel):  # type: ignore[explicit-any]
    """Public projection of a ``KALConnection`` row.

    Deliberately omits ``credentials``. ``capabilities`` and
    ``last_error`` can both be ``None`` — capabilities only populate
    after a successful handshake, ``last_error`` only after a failure.
    """

    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    adapter_type: str
    config: dict[str, object]
    capabilities: dict[str, object] | None
    status: KALConnectionStatus
    last_error: str | None
    validated_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: KALConnection) -> ConnectionResponse:
        # The cast on ``status`` reflects the runtime guarantee that
        # the column only holds members of the Literal alias — the ORM
        # column is ``String(STATUS_MAX_LEN)`` to keep adding statuses
        # a code-only change, so Python's type system doesn't see the
        # narrowing.
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            name=row.name,
            adapter_type=row.adapter_type,
            config=row.config,
            capabilities=row.capabilities,
            status=row.status,  # type: ignore[arg-type]
            last_error=row.last_error,
            validated_at=row.validated_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class AdapterTypeInfo(BaseModel):  # type: ignore[explicit-any]
    """One entry in ``GET /v1/kal/adapter-types``.

    Returns the per-type JSON Schemas so a form-driven UI can render
    config + credentials inputs without a code change. Schemas come
    from the per-type Pydantic models the builders register.
    """

    adapter_type: str
    config_schema: dict[str, object]
    credentials_schema: dict[str, object]
