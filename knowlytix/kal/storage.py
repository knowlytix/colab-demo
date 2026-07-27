"""KGStore — storage Protocol the KAL Postgres adapter delegates to.

Per paper §7 (Reference Adapter Design): "the adapter delegates rather
than reimplements". This module is the rendering of that sentence in
code. ``PostgresKnowledgeAdapter`` holds a ``KGStore`` in its
constructor and dispatches every backend operation through it; each
``KGStore`` implementation owns its own KG schema (entities + relations
+ triples + embeddings) and provides the storage primitives the
adapter needs to fulfill the ``KnowledgeAdapter`` Protocol.

Two implementations ship in this codebase (one in the standalone
package, one knowly-only):

- ``KalDefaultKGStore`` — reference impl with the package's own ORM
  in the ``kal`` schema (``KalKgEntity`` / ``KalKgRelation`` /
  ``KalKgTriple`` / ``KalKgEvidence``). Greenfield default. Ships in
  ``knowlytix-kal`` from v0.3.0; lives at
  ``knowly/src/knowly/kal/storage_default/`` for development so the
  mirror picks it up cleanly.

- ``KnowlyKGStore`` — wraps knowly's existing ``knowly.modules.kg.*``
  layer (``KGEntity`` / ``KGRelation`` / ``KGTriple`` ORM, the
  ``kg.repository`` helpers, ``kg.service.persist_claims_to_kg``).
  Stays in knowly only — never mirrored — because it binds to
  knowly's schema specifically.

Design decisions captured during the Phase A step-1 design pass
(see ``docs/plans/plan_knowlytix_kal_v0_3.md``):

- **Returns KAL-typed objects.** ``query_triples`` returns
  ``list[KALTriple]`` directly; ``get_node`` returns ``KALNode``;
  ``search_similar_nodes`` returns ``list[tuple[KALNode, float]]``.
  No intermediate "row dataclass" — each impl handles its own
  ORM-to-KAL materialization. The adapter becomes a thin pass-through.
- **Stateless w.r.t. transactions.** Every method takes ``session:
  AsyncSession`` as the first arg. The store does not own a session
  factory; the adapter composes sessions around store calls.
- **Pre-parsed tenant ID.** ``tenant_id`` is ``uuid.UUID`` (not
  ``str``). The adapter parses ``str → UUID`` at its boundary
  (``_ensure_tenant``) and passes the parsed value through.
- **No timeout wrapping in the store.** Timeouts live on the adapter
  (``_with_timeout(...)``), which wraps store calls from the outside.
  Store methods stay timeout-naive — keeps each impl free of timeout
  concerns and lets the adapter compose them uniformly.
- **No external-ID registration.** ``_register_foreign_external_ids``
  is federation plumbing, not KG storage. The adapter (which holds
  the session) writes to the external-ID registry directly via
  ``knowlytix.kal.external_id_registry.bulk_register``.
- **No name normalization.** ``normalize_entity_name`` is an
  adapter-side concern. Until a backend needs different normalization
  rules, this lives at the adapter / kal-level, not on KGStore.
  (Future: if a backend wants different rules, promote to a
  ``KGStore.normalize_name`` staticmethod.)
- **``vector_dim`` is a property.** The adapter reads it once and
  validates incoming query vectors against it before calling
  ``search_similar_nodes`` — fail-fast rather than letting the
  backend raise a dimension mismatch deep in pgvector.

Design pass status: **Draft v1.** Open questions resolved during
this pass are documented in ``docs/plans/plan_knowlytix_kal_v0_3.md``
§"Open questions to resolve during Phase A". This Protocol is
expected to evolve as the refactor (Phase A step 4) exposes
shape problems.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from knowlytix.kal.types import (
    AdjacencyDirection,
    KALNode,
    KALTriple,
    VerificationMetadata,
)

# ``AsyncSession`` only appears in method signatures and ``from __future__
# import annotations`` evaluates those as strings. Keeping the import behind
# ``TYPE_CHECKING`` means ``import knowlytix.kal.storage`` (and any module
# that re-exports ``KGStore``) doesn't pull SQLAlchemy in at parse time.
if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@runtime_checkable
class KGStore(Protocol):
    """Storage backend the KAL Postgres adapter delegates to.

    Implementations own a specific KG schema and provide the storage
    primitives the adapter calls to fulfill the ``KnowledgeAdapter``
    Protocol. Per paper §7, the adapter's job is the KAL surface
    (capability declaration, tenant enforcement at the boundary,
    GMS metadata typing); the store's job is the storage primitives
    (dedup, entity resolution, embedding, persistence). The boundary
    here is intentionally thin: KGStore methods stay close to "what
    the kg backend natively does" so a store can be a near-direct
    pass-through to its underlying ORM.

    All methods take ``session: AsyncSession`` as the first arg. The
    store does not own session lifecycle — the adapter composes
    sessions around store calls so it can wrap multiple store calls
    in a single transaction (e.g. ``insert_triples`` + foreign-ID
    registration share a session per write batch).

    All methods take ``tenant_id: uuid.UUID``. The adapter parses
    ``str → uuid.UUID`` at its boundary (where it also enforces
    ``tenant_id`` is non-empty per the Postgres adapter's
    ``TenantRequiredError`` contract).
    """

    @property
    def vector_dim(self) -> int:
        """Embedding dimensionality this store advertises.

        The adapter reads this once at ``search_similar_nodes`` time
        and validates incoming query vectors against it. Fail-fast
        rather than letting the backend raise deep in pgvector.

        Constructor arg in concrete impls (with a sensible default
        like 384 matching the Cayley sentence-encoder dim); not a
        class attribute so different deployments of the same store
        class can target different embedding spaces.
        """
        ...

    # --- Read operations ---

    async def query_triples(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        *,
        subject_names: list[str] | None,
        predicate_names: list[str] | None,
        object_names: list[str] | None,
        limit: int,
        offset: int,
        min_confidence: float | None,
        verified_only: bool,
        node_id_in_filter: list[uuid.UUID] | None,
        include_all_verifications: bool = False,
        include_revoked: bool = False,
    ) -> list[KALTriple]:
        """Pattern-filtered triple query.

        The adapter has already extracted name filters from
        ``KALQuery.patterns`` and split text-search into the optional
        ``node_id_in_filter`` (see ``text_search_node_ids`` below for
        the FTS pre-pass shape).

        Returns ``KALTriple`` rows directly — the store handles its
        own ORM-to-KAL materialization. Each returned triple's
        ``adapter`` field is left ``None`` so the federation router
        can stamp it.

        Implementations are expected to:
        - Apply ``tenant_id`` scoping on every row read.
        - Honour ``min_confidence`` (drop rows below the threshold).
        - Honour ``verified_only`` (drop ``unverified`` rows when
          ``True``).
        - Honour ``node_id_in_filter`` when non-``None`` (restrict to
          triples touching the named nodes; subject OR object).
        - Apply ``limit`` + ``offset`` for pagination.
        - Honour ``include_all_verifications`` (v0.5.0 §5.8): when
          ``True``, LEFT JOIN against the ``triple_verdicts`` table so
          the result has one row per ``(triple, verifier)`` (or one
          row with ``verification=None`` for unverified triples); when
          ``False`` (default), return one row per triple with the
          cache columns materialised onto ``KALTriple.verification``.
        - Convert literal-object triples to ``KALTriple.object_literal``
          (not ``KALTriple.object``).
        - Honour ``include_revoked`` (v0.11.0): when ``False`` (default),
          exclude quarantined rows (``revoked_at IS NOT NULL``); when
          ``True``, include them (audit views).
        """
        ...

    async def text_search_node_ids(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        query: str,
        limit: int,
    ) -> list[uuid.UUID]:
        """Free-text-search for matching node IDs.

        Used by the adapter's ``query_triples`` flow as a pre-pass:
        when ``KALQuery.text_search`` is set, the adapter first
        resolves it to a set of matching entity IDs via this method,
        then passes them as ``node_id_in_filter`` to ``query_triples``.

        Backends without FTS support should raise
        ``CapabilityNotSupportedError`` here; the adapter declares
        its capability profile so callers know whether to pass
        ``text_search`` at all.

        Returns the matching node IDs up to ``limit``. The adapter
        passes a generous limit (~5000) because one entity can sit
        on many triples and under-fetching here can silently drop
        downstream triple results.
        """
        ...

    async def query_adjacent_triples(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        node_id: uuid.UUID,
        *,
        predicate: str | None,
        direction: AdjacencyDirection,
        limit: int,
        include_all_verifications: bool = False,
        include_revoked: bool = False,
    ) -> list[KALTriple]:
        """One-hop adjacency walk from ``node_id``.

        Direction:
        - ``"outgoing"`` — triples where ``node_id`` is the subject
        - ``"incoming"`` — triples where ``node_id`` is the object
        - ``"both"`` — either

        ``predicate`` optionally filters by relation name; ``None``
        returns all predicates touching the node.

        Same KAL-materialization contract as ``query_triples``:
        returns ``KALTriple`` rows with ``adapter=None``. Honours
        ``include_all_verifications`` with the same flag-gated
        multi-row shape (v0.5.0 §5.8) so the storage layer is
        symmetric across the two read entry points. Note: the
        ``KnowledgeAdapter`` Protocol's ``query_adjacent_triples``
        does not currently surface this kwarg — the flag is reachable
        only from direct ``KGStore`` callers + the conformance suite
        until the adapter Protocol grows a query-object parameter.
        """
        ...

    async def get_node(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        node_id: uuid.UUID,
    ) -> KALNode | None:
        """Fetch a single node by ID.

        Returns ``None`` (not raises) when the node isn't found in
        the tenant's scope. The adapter's ``get_node`` does ``str →
        UUID`` parsing at its boundary; a malformed ``node_id`` never
        reaches this method.
        """
        ...

    async def search_similar_nodes(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        query_vector: list[float],
        limit: int,
    ) -> list[tuple[KALNode, float]]:
        """Vector-search the node embeddings.

        Returns ``(KALNode, distance)`` tuples sorted by ascending
        distance (closest first). Distance metric is the impl's
        choice — knowly's default is cosine distance via pgvector;
        document in the impl's class docstring.

        **Minimal-node materialization expected.** Returned ``KALNode``s
        carry ``id`` + ``name`` only; ``node_types`` / ``normalized_name``
        / ``properties`` are left at their defaults. Vector search
        returns a candidate list — callers ``get_node`` to hydrate the
        full shape when needed. Impls that eagerly hydrate add joins
        the adapter has deliberately avoided.

        Distance values MUST be plain Python ``float`` (cast explicitly
        on backends that surface numpy float subtypes) so JSON
        serialization downstream doesn't depend on numpy hooks.

        The adapter validates ``len(query_vector) == self.vector_dim``
        BEFORE calling this method; impls can assume the dimension
        is correct.
        """
        ...

    # --- Write operations ---

    async def insert_triples(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        triples: list[KALTriple],
    ) -> int:
        """Persist a batch of triples; return count of newly-created rows.

        The store does its own dedup / entity-resolution / embedding /
        triple-insert pipeline. Implementations are EXPECTED to:

        - Dedup by ``(subject_normalized, predicate_normalized,
          object_normalized)`` within the tenant scope
        - Resolve or create the subject/object entities + the predicate
          relation as needed
        - Generate embeddings for newly-created entities (if the impl's
          embedding strategy applies)
        - Honour each ``KALTriple``'s provenance + verification metadata
        - Be idempotent: re-inserting an existing triple returns 0
          new rows for that triple

        Return value is the count of triples that were actually
        persisted (not the input length). The adapter uses this for
        the ``insert_triples`` return value the KAL Protocol promises.

        The store does NOT handle external-ID registration — the
        adapter writes to the external-ID registry separately around
        this call (same session, same transaction batch).
        """
        ...

    async def delete_triples(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        triple_ids: list[uuid.UUID],
    ) -> int:
        """Delete triples by ID; return rowcount.

        Tenant-scoped — only triples belonging to ``tenant_id`` are
        deleted, even if the caller passes an ID belonging to another
        tenant (silent no-op for cross-tenant IDs is intentional;
        prevents tenant-A from probing tenant-B's ID space).
        """
        ...

    async def quarantine_triples_by_source(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        source: str,
    ) -> int:
        """Soft-revoke (quarantine) every live triple with ``provenance.source
        == source`` in the tenant; return the count newly quarantined.

        Sets ``revoked_at = now()`` WHERE ``source = :source`` AND tenant AND
        ``revoked_at IS NULL`` (idempotent — already-revoked rows are not
        re-stamped, so re-quarantining returns 0). Tenant-scoped. Reversible
        via :meth:`restore_triples_by_source`.
        """
        ...

    async def restore_triples_by_source(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        source: str,
    ) -> int:
        """Reverse a quarantine: clear ``revoked_at`` for the tenant's triples
        with ``source``; return the count restored.

        Sets ``revoked_at = NULL`` WHERE ``source = :source`` AND tenant AND
        ``revoked_at IS NOT NULL``. Tenant-scoped.
        """
        ...

    async def update_verification(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        triple_id: uuid.UUID,
        verification: VerificationMetadata,
    ) -> bool:
        """Update verification metadata on a triple; return existence flag.

        Returns ``True`` iff the triple existed (and was updated);
        ``False`` if no such triple exists in the tenant's scope.

        The store unpacks ``VerificationMetadata`` to whatever columns
        its schema uses (knowly's columns: ``confidence``,
        ``verification_status``, ``verified_at``, ``verifier``,
        ``scores``). The Protocol passes the typed KAL struct so the
        adapter doesn't have to know the column layout.
        """
        ...

    async def delete_node(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        node_id: uuid.UUID,
    ) -> bool:
        """Delete an entity row by ID; return existence flag.

        Storage-level primitive for the autotuning mutation path
        (paper §8.2). NOT exposed on the ``KnowledgeAdapter`` surface
        — entity-row CRUD is store-specific (not every backend has an
        entity table) so it lives below the federated KAL boundary.

        Returns ``True`` iff the entity existed in the tenant's scope
        (and was deleted); ``False`` if no such entity exists.

        **Tenant-scoped.** Deleting an ID belonging to another tenant
        is a silent no-op (returns ``False``) — same security
        invariant as ``delete_triples``.

        **FK-RESTRICT semantics.** From v0.4.0 the FKs on
        ``triples.subject_id`` and ``triples.object_id`` are ``ON
        DELETE RESTRICT``; deleting an entity that any triple still
        references raises ``sqlalchemy.exc.IntegrityError``. Callers
        (notably ``mutations.merge_entities``) MUST delete or rewrite
        the referencing triples first, then call ``delete_node``. This
        surfaces caller bugs at the failed-DELETE site rather than
        silently nuking data via the prior CASCADE.
        """
        ...

    async def delete_predicate(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        predicate: str,
    ) -> bool:
        """Delete a relation row by name; return existence flag.

        Storage-level primitive for ``mutations.merge_relations``
        (paper §8.2). NOT exposed on the ``KnowledgeAdapter`` surface
        — relation-row lifecycle is store-specific.

        Addresses the relation by its predicate name (the KAL surface
        identifies predicates by string, not UUID — see
        ``KALTriple.predicate``). Both reference implementations match
        on exact ``name`` (neither ``KGRelation`` nor ``KalKgRelation``
        carries a ``normalized_name`` column today); callers must pass
        the predicate string as it was originally registered.

        Returns ``True`` iff a matching relation existed (and was
        deleted); ``False`` if no such predicate exists in the
        tenant's scope.

        **Tenant-scoped.** Same security invariant as ``delete_node``.

        **FK-RESTRICT semantics.** Same as ``delete_node`` — from
        v0.4.0 the FK on ``triples.predicate_id`` is ``ON DELETE
        RESTRICT``; deleting a relation that any triple still
        references raises ``IntegrityError``.
        """
        ...

    # --- Helpers used during write paths ---

    async def resolve_node_ids_by_normalized_name(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        normalized_names: list[str],
    ) -> dict[str, uuid.UUID]:
        """Resolve normalized entity names to internal node IDs.

        Used by the adapter's foreign-ID registration step
        (``_register_foreign_external_ids``): after ``insert_triples``
        persists rows, the adapter looks up the freshly-resolved
        internal IDs for each foreign entity's normalized name so it
        can write ``(external_id → internal_id)`` mappings to the
        external-ID registry.

        Returns a dict of ``{normalized_name: node_id}``. Names not
        found in the tenant's scope are simply absent from the dict
        — the adapter treats those as "not in the KG yet" and skips
        registry registration for them.
        """
        ...
