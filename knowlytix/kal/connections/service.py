"""``ConnectionService`` — orchestrates KAL connection CRUD.

The repository handles persistence and the factory handles per-type
construction; the service is the place that:

- validates ``config`` + ``credentials`` against the per-type Pydantic
  schemas the factory exposes,
- enforces the configurable per-tenant connection cap,
- builds the adapter, runs the handshake under classified error
  handling, and disposes the engine on handshake failure so no
  leaked-pool state ever survives a failed create,
- encrypts credentials with ``KALSecretBox`` *after* handshake passes
  (so the row never holds a ciphertext for a never-validated config),
- persists the row, registers the adapter in the in-process
  ``AdapterRegistry``, and stamps ``capabilities`` + ``validated_at``
  on the row,
- on delete, unregisters the adapter from the in-process registry,
  closes the engine, and removes the row in a single atomic step.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from knowlytix.kal.connections import repository
from knowlytix.kal.connections.models import (
    DEFAULT_STATUS,
    DISABLED_STATUS,
    KALConnection,
)
from knowlytix.kal.errors import (
    AdapterExceptionClass,
    AdapterNotFoundError,
    classify_adapter_exception,
    truncate_error_message,
)
from knowlytix.kal.factory import AdapterFactory
from knowlytix.kal.protocol import KnowledgeAdapter
from knowlytix.kal.registry import AdapterRegistry
from knowlytix.kal.secrets import KALSecretBox
from knowlytix.kal.tenant_index import TenantIndex

logger = logging.getLogger(__name__)


class HandshakeFailedError(Exception):
    """Service-level error wrapping a builder-raised handshake failure.

    ``error_type`` mirrors the ``AdapterError`` taxonomy used by the
    federation router — ``"auth"`` for ``AdapterAuthError``,
    ``"connection"`` for ``AdapterConnectionError``, ``"internal"`` for
    anything else. The API layer maps this to HTTP 401 / 502 / 500
    respectively. The underlying exception is preserved on
    ``__cause__`` for callers that want the original detail. The
    message is already length-capped by ``classify_adapter_exception``
    — the full traceback only goes to ``logger.exception``.
    """

    def __init__(
        self, error_type: AdapterExceptionClass, message: str
    ) -> None:
        super().__init__(message)
        self.error_type: AdapterExceptionClass = error_type


class DuplicateNameError(Exception):
    """A connection with the same ``name`` already exists for the tenant.

    Surfaced explicitly (rather than letting the DB ``IntegrityError``
    propagate) so the API can return 409 Conflict with a stable error
    message. The service does a pre-insert lookup *and* translates the
    DB-level unique-constraint violation, since two concurrent creates
    can both pass the pre-check before either commits.
    """


class _HydrationError(Exception):
    """Internal-only error raised inside ``hydrate_all`` to short-circuit
    a single row's hydration with a pre-capped message. Never propagates
    past ``_hydrate_row`` — the message lands on the row's
    ``last_error`` and the next row continues.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TenantConnectionCapError(Exception):
    """The tenant has hit its configured connection cap.

    Cap is configurable per-service-instance via
    ``max_connections_per_tenant`` (default
    ``DEFAULT_MAX_CONNECTIONS_PER_TENANT``). Protects against
    unbounded per-tenant pool growth while leaving headroom for a
    realistic multi-backend deployment. The check is best-effort
    under heavy concurrent creates — two callers racing at the cap
    boundary can both pass and INSERT, leaving the tenant cap+1 over.
    """


# Default cap on connections per tenant. Configurable per
# ``ConnectionService`` instance.
DEFAULT_MAX_CONNECTIONS_PER_TENANT: int = 10

# Default per-row handshake budget at lifespan hydration. Picked to
# match the ``kal_hydration_handshake_timeout_seconds`` setting default
# so a service constructed without explicit settings still follows the
# documented behaviour.
DEFAULT_HYDRATION_HANDSHAKE_TIMEOUT_SECONDS: float = 2.0

# Width of the tenant-id prefix used inside the namespaced adapter
# name (``{prefix}:{name}``). 16 hex chars give 2^64 distinct prefixes
# — birthday-paradox collisions only become non-negligible past a few
# billion tenants, comfortably above any realistic SaaS scale. A
# narrower prefix risks two tenants colliding on the registry key and
# routing a query to the wrong tenant's adapter.
TENANT_NAMESPACE_HEX_LEN: int = 16


def namespaced_adapter_name(tenant_id: uuid.UUID, name: str) -> str:
    """Compose the in-process registry key for a tenant-owned adapter.

    The user-facing ``name`` is unique only inside a tenant; the
    registry key must be unique across the whole running process,
    so we prefix with a slice of the tenant UUID wide enough to
    make cross-tenant collisions negligible. Public so the
    lifespan-shutdown path can reconstruct the registry key from a
    persisted ``(tenant_id, name)`` pair without duplicating the
    namespacing convention.
    """
    return f"{tenant_id.hex[:TENANT_NAMESPACE_HEX_LEN]}:{name}"


# Internal alias preserved so call sites inside this module stay
# concise; external callers import the public name.
_namespace = namespaced_adapter_name


class ConnectionService:
    """Orchestrates the connection-registry create-flow.

    State stays on the injected ``AdapterRegistry`` and the
    repository; the service itself is stateless beyond its
    construction-time dependencies.
    """

    def __init__(
        self,
        *,
        registry: AdapterRegistry,
        secret_box: KALSecretBox,
        max_connections_per_tenant: int = DEFAULT_MAX_CONNECTIONS_PER_TENANT,
        hydration_handshake_timeout_seconds: float = (
            DEFAULT_HYDRATION_HANDSHAKE_TIMEOUT_SECONDS
        ),
        tenant_index: TenantIndex | None = None,
    ) -> None:
        # ``tenant_index`` defaults to ``None`` so existing callers
        # (tests, the demo, anything wired up pre-Story-8) keep working
        # without touching this signature. When set, every register
        # / unregister mutation is mirrored into the index so
        # ``FederationRouter`` can constrain fan-out to the caller's
        # tenant. The shared adapters (e.g. ``postgres-main``) live
        # on the index itself, not on the service — lifespan owns
        # that initial seeding.
        self._registry = registry
        self._secret_box = secret_box
        self._max_connections = max_connections_per_tenant
        self._hydration_handshake_timeout = hydration_handshake_timeout_seconds
        self._tenant_index = tenant_index

    def _register(
        self, adapter: KnowledgeAdapter, tenant_id: uuid.UUID
    ) -> None:
        # Centralises the (registry + tenant-index) pair so every
        # register call site stays one line and the ownership
        # invariant ("a row's tenant_id matches its registry binding")
        # has exactly one enforcement point. ``bind`` canonicalises
        # the UUID itself via ``_normalize_tenant_id`` — no
        # explicit ``str()`` needed here.
        self._registry.register(adapter)
        if self._tenant_index is not None:
            self._tenant_index.bind(adapter.name, tenant_id)

    def _unregister(self, adapter_name: str) -> None:
        # Symmetric to ``_register``: any time the registry loses an
        # adapter, the tenant index loses its ownership row too so a
        # later create with the same name doesn't inherit stale
        # ownership.
        self._registry.unregister(adapter_name)
        if self._tenant_index is not None:
            self._tenant_index.unbind(adapter_name)

    async def create(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        name: str,
        adapter_type: str,
        config: dict[str, object],
        credentials: dict[str, object],
    ) -> KALConnection:
        """Validate → build → handshake → encrypt → persist → commit → register.

        Create is a leaf operation: the service commits the row before
        touching the in-process registry so a rolled-back transaction
        can't leave a phantom adapter registered against a row that
        never landed. Any exception between build and successful
        register (including ``CancelledError`` at an ``await``
        boundary or a ``commit`` failure under the persist step)
        routes through ``_dispose_quietly`` — no engine survives a
        failed create. Callers that need to compose multiple
        operations in a single transaction must reach for lower-level
        helpers; the registry mutation isn't transactional.
        """
        await self._reject_duplicate_or_cap(
            session, tenant_id=tenant_id, name=name
        )
        validated_config, validated_creds = self._validate_schemas(
            adapter_type, config, credentials
        )
        adapter = await self._build_and_handshake(
            adapter_type,
            namespaced=_namespace(tenant_id, name),
            config=validated_config,
            creds=validated_creds,
            tenant_id=tenant_id,
        )
        row = await self._persist_and_commit(
            session,
            tenant_id=tenant_id,
            name=name,
            adapter_type=adapter_type,
            config=validated_config,
            creds=validated_creds,
            adapter=adapter,
        )
        self._register(adapter, tenant_id)
        return row

    async def _persist_and_commit(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        name: str,
        adapter_type: str,
        config: dict[str, object],
        creds: dict[str, object],
        adapter: KnowledgeAdapter,
    ) -> KALConnection:
        # Owns the commit-or-dispose bookkeeping so ``create`` stays a
        # flat orchestrator. Any failure between ``_persist`` and
        # ``commit`` disposes the half-built adapter via ``finally``.
        committed = False
        try:
            row = await self._persist(
                session,
                tenant_id=tenant_id,
                name=name,
                adapter_type=adapter_type,
                config=config,
                creds=creds,
                adapter=adapter,
            )
            await session.commit()
            committed = True
            return row
        finally:
            if not committed:
                await self._dispose_quietly(adapter_type, adapter)

    @staticmethod
    def _validate_schemas(
        adapter_type: str,
        config: dict[str, object],
        credentials: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        builder = AdapterFactory.get_builder(adapter_type)
        cfg = builder.config_schema.model_validate(config).model_dump()
        crd = builder.credentials_schema.model_validate(credentials).model_dump()
        return cfg, crd

    async def _build_and_handshake(
        self,
        adapter_type: str,
        *,
        namespaced: str,
        config: dict[str, object],
        creds: dict[str, object],
        tenant_id: uuid.UUID,
    ) -> KnowledgeAdapter:
        adapter = await AdapterFactory.build(
            adapter_type, namespaced, config, creds
        )
        # ``try``/``finally`` so ``CancelledError`` between build and
        # handshake-complete still disposes the engine (the
        # ``except Exception`` branch handles the modeled
        # adapter-failure case). ``handshake_ok`` flips to True only
        # after the handshake await returns successfully —
        # cancellation interrupting that await leaves the flag False
        # and the ``finally`` disposes.
        handshake_ok = False
        try:
            await AdapterFactory.handshake(
                adapter_type, adapter, tenant_id=str(tenant_id)
            )
            handshake_ok = True
        except Exception as exc:
            error_type, message = classify_adapter_exception(exc)
            raise HandshakeFailedError(error_type, message) from exc
        finally:
            if not handshake_ok:
                await self._dispose_quietly(adapter_type, adapter)
        return adapter

    async def _persist(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        name: str,
        adapter_type: str,
        config: dict[str, object],
        creds: dict[str, object],
        adapter: KnowledgeAdapter,
    ) -> KALConnection:
        # Encryption only happens after handshake passes — a row's
        # presence in the table implies the credentials reached the
        # backend successfully at least once. Encrypting the
        # Pydantic-validated dict (not the raw input) keeps the
        # ciphertext aligned with the validated shape. The
        # ``IntegrityError`` translation covers the race where two
        # concurrent creates both pass the pre-check on
        # ``get_by_name`` and the unique constraint catches the
        # losing INSERT.
        try:
            ciphertext = self._secret_box.encrypt(creds)
            return await repository.create(
                session,
                tenant_id=tenant_id,
                name=name,
                adapter_type=adapter_type,
                config=config,
                credentials=ciphertext,
                capabilities=dataclasses.asdict(adapter.capabilities),
                status=DEFAULT_STATUS,
                validated_at=_now_utc(),
            )
        except IntegrityError as exc:
            raise DuplicateNameError(
                _duplicate_name_message(name, tenant_id)
            ) from exc

    async def list_for_tenant(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
    ) -> list[KALConnection]:
        """Tenant-scoped list of connection rows. No registry side
        effects — pure read of the persistence layer.
        """
        return await repository.list_for_tenant(
            session, tenant_id=tenant_id
        )

    async def get(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        row_id: uuid.UUID,
    ) -> KALConnection:
        """Fetch one connection by id, enforcing the tenant scope.
        Raises ``KALConnectionNotFoundError`` for both "no such row"
        and "row belongs to another tenant" so callers can't use the
        API as a cross-tenant presence oracle.
        """
        return await repository.get_by_id(
            session, tenant_id=tenant_id, row_id=row_id
        )

    async def delete(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        row_id: uuid.UUID,
    ) -> None:
        """Delete row → commit → unregister → aclose.

        Symmetric with ``create``'s commit-before-register order. If
        the DB delete or commit raises, the registry stays intact and
        federation keeps routing to the live adapter — the operator
        can retry. Only after commit succeeds does the in-process
        registry mutation run; an aclose failure during disposal is
        log-and-swallowed by ``_dispose_quietly`` so a phantom-engine
        leak is bounded by process life rather than crashing the
        request.

        The unregister-after-commit window is dict-pop-narrow, so an
        in-flight federation request that snapshotted ``registry``
        before unregister can still issue queries against the
        soon-to-be-disposed engine — those will fail when the
        underlying pool is closed. Cleaner draining (an explicit
        grace period) is a follow-up; for now, the bounded window
        is preferable to leaving the registry inconsistent with the
        DB across an entire commit attempt.
        """
        # ``get_by_id`` enforces the tenant scope and gives us the
        # row's ``adapter_type`` + ``name`` for namespacing and
        # disposal. Reusing the already-fetched row in the DELETE
        # below saves a round-trip vs ``repository.delete``'s
        # internal re-fetch.
        row = await repository.get_by_id(
            session, tenant_id=tenant_id, row_id=row_id
        )
        adapter_type = row.adapter_type
        namespaced = _namespace(row.tenant_id, row.name)
        await session.delete(row)
        # ``commit`` flushes internally — no separate ``flush``
        # needed before it.
        await session.commit()
        adapter = self._pop_registered(namespaced)
        if adapter is not None:
            await self._dispose_quietly(adapter_type, adapter)

    async def update(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        row_id: uuid.UUID,
        config: dict[str, object] | None,
        credentials: dict[str, object] | None,
    ) -> KALConnection:
        """PATCH a connection's config and/or credentials.

        Rebuilds + handshakes the adapter with the effective config +
        creds (request body fields override stored values; missing
        body fields fall back to the row), commits the row update,
        then swaps the registry entry. The swap (pop → register →
        aclose old) runs **after** commit so a commit failure can't
        leave the registry pointing at a row that never landed.
        On a ``"disabled"`` row the freshly-built adapter is disposed
        without ever entering the registry — PATCH verifies the new
        config but doesn't flip status; admin re-enables explicitly.
        """
        row, new_adapter, validated_config, validated_creds = (
            await self._fetch_and_rebuild(
                session,
                tenant_id=tenant_id,
                row_id=row_id,
                config_override=config,
                creds_override=credentials,
            )
        )
        # Encrypt under the same ``try``/``finally`` guard as commit
        # so a Fernet failure between handshake and persist disposes
        # the freshly-built adapter rather than leaking the engine.
        return await self._commit_and_swap(
            session,
            tenant_id=tenant_id,
            row_id=row_id,
            row=row,
            new_adapter=new_adapter,
            new_config=validated_config if config is not None else None,
            plaintext_creds_to_encrypt=(
                validated_creds if credentials is not None else None
            ),
            new_status=None,
            should_register=row.status == DEFAULT_STATUS,
        )

    async def test(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        row_id: uuid.UUID,
    ) -> KALConnection:
        """Build a transient adapter from the row's stored config +
        decrypted creds, handshake it, refresh ``capabilities`` +
        ``validated_at``, then dispose. Registry membership is never
        touched — ``/test`` is a credentials liveness probe.

        Works on active and disabled rows alike: a disabled row's
        creds can be validated before ``/enable``. The transient
        adapter is disposed via ``finally`` so its engine never
        leaks regardless of the update path.
        """
        row, test_adapter, _, _ = await self._fetch_and_rebuild(
            session, tenant_id=tenant_id, row_id=row_id
        )
        try:
            updated_row = await repository.update(
                session,
                tenant_id=tenant_id,
                row_id=row_id,
                capabilities=dataclasses.asdict(test_adapter.capabilities),
                validated_at=_now_utc(),
                last_error=None,
            )
            await session.commit()
            return updated_row
        finally:
            await self._dispose_quietly(row.adapter_type, test_adapter)

    async def enable(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        row_id: uuid.UUID,
    ) -> KALConnection:
        """Build → handshake → register, flip status to ``"active"``.

        Always handshakes — per the plan, an old ``validated_at``
        isn't enough confidence that credentials still work, and
        the handshake cost is bounded. Idempotent on an already-
        active row: handshake reruns, registry entry refreshes,
        old adapter (if any) is disposed.

        The registry swap runs after commit. An in-flight federation
        request that snapshotted the registry before the swap can
        still hold a reference to the old adapter and issue queries
        against the soon-to-be-disposed engine — the window is
        dict-pop-narrow but non-zero, same as ``delete``.
        """
        row, new_adapter, _, _ = await self._fetch_and_rebuild(
            session, tenant_id=tenant_id, row_id=row_id
        )
        return await self._commit_and_swap(
            session,
            tenant_id=tenant_id,
            row_id=row_id,
            row=row,
            new_adapter=new_adapter,
            new_config=None,
            plaintext_creds_to_encrypt=None,
            new_status=DEFAULT_STATUS,
            should_register=True,
        )

    async def disable(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        row_id: uuid.UUID,
    ) -> KALConnection:
        """Flip status to ``"disabled"``, pop the registered adapter
        out of the registry, dispose its engine.

        Symmetric with ``delete`` — commit before mutating the
        registry so a commit failure leaves the adapter live and
        federation keeps routing. Idempotent on an already-disabled
        row: returns the row unchanged without re-issuing the
        no-op UPDATE so an accidental double-click doesn't churn
        the DB.

        The short-circuit reads ``row.status`` without locking, so
        a concurrent ``enable`` + ``disable`` on the same row can
        interleave inconsistently — fine under the single-worker
        deployment assumption, deferred to a later story when
        Knowly runs N FastAPI workers.
        """
        row = await repository.get_by_id(
            session, tenant_id=tenant_id, row_id=row_id
        )
        if row.status == DISABLED_STATUS:
            return row
        adapter_type = row.adapter_type
        namespaced = _namespace(row.tenant_id, row.name)
        updated_row = await repository.update(
            session,
            tenant_id=tenant_id,
            row_id=row_id,
            status=DISABLED_STATUS,
        )
        await session.commit()
        adapter = self._pop_registered(namespaced)
        if adapter is not None:
            await self._dispose_quietly(adapter_type, adapter)
        return updated_row

    async def _fetch_and_rebuild(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        row_id: uuid.UUID,
        config_override: dict[str, object] | None = None,
        creds_override: dict[str, object] | None = None,
    ) -> tuple[
        KALConnection,
        KnowledgeAdapter,
        dict[str, object],
        dict[str, object],
    ]:
        # Shared prelude for ``update``/``test``/``enable``: fetch
        # the row under tenant scope, decide the effective config +
        # creds (overrides win, otherwise the stored values are
        # decrypted on demand), validate against the per-type
        # schemas, then build + handshake. Returns the validated
        # values too so ``update`` can reuse them for re-encryption
        # without re-validating.
        row = await repository.get_by_id(
            session, tenant_id=tenant_id, row_id=row_id
        )
        effective_config = (
            config_override if config_override is not None else row.config
        )
        effective_creds = (
            creds_override
            if creds_override is not None
            else self._secret_box.decrypt(row.credentials)
        )
        validated_config, validated_creds = self._validate_schemas(
            row.adapter_type, effective_config, effective_creds
        )
        new_adapter = await self._build_and_handshake(
            row.adapter_type,
            namespaced=_namespace(row.tenant_id, row.name),
            config=validated_config,
            creds=validated_creds,
            tenant_id=row.tenant_id,
        )
        return row, new_adapter, validated_config, validated_creds

    async def _commit_and_swap(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        row_id: uuid.UUID,
        row: KALConnection,
        new_adapter: KnowledgeAdapter,
        new_config: dict[str, object] | None,
        plaintext_creds_to_encrypt: dict[str, object] | None,
        new_status: str | None,
        should_register: bool,
    ) -> KALConnection:
        # Shared commit-then-swap dance for ``update`` + ``enable``.
        # Encryption happens inside the commit-protected ``try`` so a
        # Fernet failure between handshake and persist routes
        # through the same ``_dispose_quietly`` path as a DB
        # commit failure — no engine leaks. The three nullable
        # kwargs let the caller add a status flip (``enable``)
        # and/or a config + credentials swap (``update``) without
        # sneaking unknown kwargs past the repository's typed
        # signature. Only swap into the registry after a clean commit.
        adapter_type = row.adapter_type
        committed = False
        try:
            new_credentials = (
                self._secret_box.encrypt(plaintext_creds_to_encrypt)
                if plaintext_creds_to_encrypt is not None
                else None
            )
            updated_row = await repository.update(
                session,
                tenant_id=tenant_id,
                row_id=row_id,
                config=new_config,
                credentials=new_credentials,
                status=new_status,
                capabilities=dataclasses.asdict(new_adapter.capabilities),
                validated_at=_now_utc(),
                last_error=None,
            )
            await session.commit()
            committed = True
        finally:
            if not committed:
                await self._dispose_quietly(adapter_type, new_adapter)
        await self._swap_registered(
            adapter_type,
            namespaced=_namespace(row.tenant_id, row.name),
            new_adapter=new_adapter,
            should_register=should_register,
            tenant_id=row.tenant_id,
        )
        return updated_row

    async def _swap_registered(
        self,
        adapter_type: str,
        *,
        namespaced: str,
        new_adapter: KnowledgeAdapter,
        should_register: bool,
        tenant_id: uuid.UUID,
    ) -> None:
        # Disabled row: PATCH only verified the new config works, so
        # the freshly-built adapter is disposed without ever entering
        # the registry — admin re-enables explicitly. Active row:
        # read the prior reference (without popping) so the
        # subsequent ``_register`` reuses the existing dict slot —
        # Python's ``d[k] = v`` preserves position when the key
        # already exists, so the swapped adapter stays at its
        # original registration index. Popping first and re-inserting
        # would move the adapter to the END of the dict and break
        # the router's "first-registered wins" tie-break + write
        # fallback (federation.py:151-156, 255).
        if not should_register:
            await self._dispose_quietly(adapter_type, new_adapter)
            return
        old = self._peek_registered(namespaced)
        self._register(new_adapter, tenant_id)
        if old is not None:
            await self._dispose_quietly(adapter_type, old)

    def _peek_registered(self, namespaced: str) -> KnowledgeAdapter | None:
        # Read-only lookup — returns the live adapter without mutating
        # the registry so the swap path (``_swap_registered``) can
        # re-assign by name and keep the existing dict slot.
        try:
            return self._registry.get(namespaced)
        except AdapterNotFoundError:
            return None

    def _pop_registered(self, namespaced: str) -> KnowledgeAdapter | None:
        # Composes ``_peek_registered`` (lookup) with ``_unregister``
        # (mutation) so the "return-None-on-missing" branch lives in
        # one place.
        adapter = self._peek_registered(namespaced)
        if adapter is None:
            return None
        self._unregister(namespaced)
        return adapter

    async def _reject_duplicate_or_cap(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        name: str,
    ) -> None:
        existing_named = await repository.get_by_name(
            session, tenant_id=tenant_id, name=name
        )
        if existing_named is not None:
            raise DuplicateNameError(
                _duplicate_name_message(name, tenant_id)
            )
        # Count rather than ``list_for_tenant`` so the cap check stays
        # O(1) round-trip instead of materialising every row as a
        # SQLAlchemy instance on each create.
        existing_count = await repository.count_for_tenant(
            session, tenant_id=tenant_id
        )
        if existing_count >= self._max_connections:
            raise TenantConnectionCapError(
                f"tenant {tenant_id} has reached its cap of "
                f"{self._max_connections} KAL connections"
            )

    async def hydrate_all(self, session: AsyncSession) -> None:
        """Re-register every ``status='active'`` row in the in-process
        registry. Called once at lifespan startup.

        Each row's handshake runs under
        ``hydration_handshake_timeout_seconds`` so a single hung backend
        cannot stall the whole app boot. Failures (timeout, auth,
        connection, anything else) leave the row visible-but-unregistered
        with a capped ``last_error`` for the admin to inspect; intent
        stays ``'active'`` so the difference between "admin disabled it"
        and "boot couldn't reach it" survives crash recovery — see the
        plan's "Intent vs liveness" section.

        Rows are committed individually as their ``last_error`` /
        ``validated_at`` is stamped, so a single failing row doesn't
        roll back the success-stamps for earlier rows in the same boot.
        """
        rows = await repository.list_all(session, status=DEFAULT_STATUS)
        # Snapshot just the (tenant_id, row_id) pairs upfront and
        # re-fetch each row inside the loop body. A failure-stamp
        # rollback in the middle of the loop expires every attached
        # object in the session; re-fetching guarantees the next
        # iteration sees fresh, non-expired attributes instead of
        # tripping ``MissingGreenlet`` on an implicit async refresh.
        targets = [(row.tenant_id, row.id) for row in rows]
        for tenant_id, row_id in targets:
            try:
                fresh = await repository.get_by_id(
                    session, tenant_id=tenant_id, row_id=row_id
                )
                await self._hydrate_row(session, fresh)
            except Exception:
                # A row's hydration must never abort the boot loop —
                # the docstring promises per-row independence, and the
                # outer lifespan handler is the wrong place to recover
                # from a single bad row (it would skip every subsequent
                # row in this batch). The exception is already logged
                # by the path that raised it; this catch is the
                # backstop that keeps the loop turning.
                logger.exception(
                    "unhandled error hydrating row %s; continuing", row_id
                )
                # If the row blew up inside ``_record_hydration_success``
                # (commit failed → re-raised after dispose), the session
                # is now in a failed-transaction state and the next
                # iteration's ``get_by_id`` would raise
                # ``PendingRollbackError``. Roll back here so the loop
                # can continue against a clean session — the
                # ``_record_hydration_failure`` path already handles
                # its own rollback, so an unconditional rollback here
                # is safe and idempotent.
                await session.rollback()

    async def _hydrate_row(
        self, session: AsyncSession, row: KALConnection
    ) -> None:
        # One row's worth of build → handshake → register. Modelled
        # failures (``_HydrationError``) get stamped on the row's
        # ``last_error``; unmodeled failures from the
        # ``_record_hydration_success`` commit dispose the adapter and
        # bubble up to ``hydrate_all``'s outer guard, which logs +
        # continues with the next row.
        try:
            adapter = await self._build_and_handshake_for_hydration(row)
        except _HydrationError as failure:
            await self._record_hydration_failure(session, row, failure.message)
            return
        try:
            await self._record_hydration_success(session, row, adapter)
        except Exception:
            await self._dispose_quietly(row.adapter_type, adapter)
            raise
        self._register(adapter, row.tenant_id)

    async def _build_and_handshake_for_hydration(
        self, row: KALConnection
    ) -> KnowledgeAdapter:
        # Hydration is structurally similar to ``_build_and_handshake``
        # but (a) decrypts ciphertext from the row, (b) wraps handshake
        # in a budget so a stuck backend can't block boot, and (c)
        # raises ``_HydrationError`` so the boot loop can swallow any
        # failure mode (auth, decrypt, missing builder, timeout). The
        # body is a flat orchestrator; each step is its own helper so
        # rollback-on-failure ownership stays local to each step.
        credentials = self._decrypt_for_hydration(row)
        adapter = await self._build_for_hydration(row, credentials)
        try:
            await self._handshake_under_budget(row, adapter)
        except BaseException:
            await self._dispose_quietly(row.adapter_type, adapter)
            raise
        return adapter

    def _decrypt_for_hydration(self, row: KALConnection) -> dict[str, object]:
        try:
            return self._secret_box.decrypt(row.credentials)
        except Exception as exc:
            raise _HydrationError(
                _format_internal_error("credential decrypt failed", exc)
            ) from exc

    async def _build_for_hydration(
        self, row: KALConnection, credentials: dict[str, object]
    ) -> KnowledgeAdapter:
        try:
            return await AdapterFactory.build(
                row.adapter_type,
                _namespace(row.tenant_id, row.name),
                row.config,
                credentials,
            )
        except Exception as exc:
            raise _HydrationError(
                _format_internal_error("adapter build failed", exc)
            ) from exc

    async def _handshake_under_budget(
        self, row: KALConnection, adapter: KnowledgeAdapter
    ) -> None:
        try:
            await asyncio.wait_for(
                AdapterFactory.handshake(
                    row.adapter_type, adapter, tenant_id=str(row.tenant_id)
                ),
                timeout=self._hydration_handshake_timeout,
            )
        except TimeoutError as exc:
            raise _HydrationError(
                truncate_error_message(
                    "handshake exceeded hydration timeout "
                    f"({self._hydration_handshake_timeout}s)"
                )
            ) from exc
        except Exception as exc:
            _, message = classify_adapter_exception(exc)
            raise _HydrationError(message) from exc

    async def _record_hydration_success(
        self,
        session: AsyncSession,
        row: KALConnection,
        adapter: KnowledgeAdapter,
    ) -> None:
        # Mutate the already-attached row directly rather than calling
        # ``repository.update`` (which re-fetches via ``get_by_id``,
        # paying one redundant SELECT per row at boot). The ``delete``
        # path uses the same shortcut — see its comment on
        # save-a-round-trip.
        row.capabilities = dataclasses.asdict(adapter.capabilities)
        row.validated_at = _now_utc()
        row.last_error = None
        await session.commit()

    async def _record_hydration_failure(
        self, session: AsyncSession, row: KALConnection, message: str
    ) -> None:
        # Goes through ``repository.update`` (an extra SELECT per
        # failure-stamp) rather than mutating the attached ``row``
        # directly. The cost is bounded by the number of failing
        # rows at boot, which is tiny in practice; the win is that a
        # subsequent ``session.rollback()`` (in the inner ``except``)
        # cannot expire the next loop iteration's row attributes —
        # ``repository.update`` re-fetches by id, so the next call
        # works against a freshly-loaded row even if the session
        # state got expired. Best-effort: a DB failure here doesn't
        # promote to a boot crash, the row just keeps its prior
        # ``last_error``.
        try:
            await repository.update(
                session,
                tenant_id=row.tenant_id,
                row_id=row.id,
                last_error=message,
            )
            await session.commit()
        except Exception:
            logger.exception(
                "failed to record hydration failure for row %s", row.id
            )
            await session.rollback()

    @staticmethod
    async def _dispose_quietly(
        adapter_type: str, adapter: KnowledgeAdapter
    ) -> None:
        # Cleanup must itself never raise — a second exception while
        # handling the first would hide the real cause. Asymmetric
        # with the handshake path (which narrows to ``Exception`` so
        # cancellation propagates): here ``BaseException`` is
        # deliberate so a cancellation mid-aclose still completes
        # the dispose attempt rather than leaving the engine open
        # in the failing-create path on its way out.
        try:
            await AdapterFactory.aclose(adapter_type, adapter)
        except BaseException:
            logger.exception(
                "aclose failed while rolling back failed handshake "
                "for adapter_type=%r",
                adapter_type,
            )


def _now_utc() -> datetime:
    # Centralised so tests can monkeypatch the timestamp source if they
    # ever need deterministic ``validated_at`` assertions.
    return datetime.now(UTC)


def _duplicate_name_message(name: str, tenant_id: uuid.UUID) -> str:
    # Locked-in wording for ``DuplicateNameError``. Two call sites
    # raise this — the pre-insert ``get_by_name`` check and the
    # ``IntegrityError`` translation on the racing-concurrent-create
    # path — and both feed the API's 409 detail. Centralising the
    # format keeps the two paths from drifting on a future copy edit.
    return f"connection {name!r} already exists for tenant {tenant_id}"


def _format_internal_error(prefix: str, exc: BaseException) -> str:
    # Adds ``prefix`` in front of the same ``{type(exc).__name__}:
    # {exc}`` body that ``classify_adapter_exception`` produces for
    # its "internal" branch — the prefix marks WHICH hydration step
    # failed (decrypt vs build) while the suffix matches the
    # federation-error shape on the rest of the system.
    # ``truncate_error_message`` caps the result so an asyncpg /
    # SQLAlchemy envelope can't blow past the column budget.
    return truncate_error_message(f"{prefix}: {type(exc).__name__}: {exc}")
