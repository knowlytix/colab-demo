"""Read-only Wikidata SPARQL adapter.

A proof-of-abstraction second adapter type for Story 9 of the
connection-registry plan. The contract surface:

- ``can_read=True``; every write/update/delete method raises
  ``CapabilityNotSupportedError`` — Wikidata is a public read-only
  endpoint and we never want a router fallback to send writes here.
- ``query_triples`` translates the small ``KALTriplePattern`` shape
  (subject / predicate / object name strings) into a SPARQL ``SELECT``
  via ``rdfs:label`` matches. Wikidata's URIs are opaque (``Q42``,
  ``P31``), so the adapter resolves human-readable labels to entity
  / property URIs at query time. Pattern-less queries fall back to a
  cheap ``ASK { ?s ?p ?o }`` probe which is also what the builder's
  override handshake fires.
- ``query_adjacent_triples`` walks a single node's neighbours via
  ``DESCRIBE``-style ``WHERE { wd:Qxxx ?p ?o }`` (outgoing) or the
  symmetric incoming form.
- ``get_node`` round-trips a Wikidata QID into a ``KALNode``.
- The HTTP client is created in the constructor and disposed via the
  builder's ``aclose``; the adapter has no engine, so cleanup is just
  the httpx ``AsyncClient.aclose()`` call.

Wikidata is multi-tenant in spirit but tenant-blind in fact: the
``tenant_id`` argument is accepted (Protocol contract) but ignored —
every tenant sees the same public knowledge graph. The connection
service still namespaces the adapter name so two tenants can both
configure their own ``wikidata-public`` connection without
colliding in the registry.
"""

from __future__ import annotations

import logging
import re
from http import HTTPStatus
from typing import Final

import httpx

from knowlytix.kal.errors import (
    AdapterAuthError,
    AdapterConnectionError,
    CapabilityNotSupportedError,
)
from knowlytix.kal.protocol import AdapterCapabilities
from knowlytix.kal.types import (
    KALNode,
    KALQuery,
    KALQueryResult,
    KALTriple,
    KALTriplePattern,
    VerificationMetadata,
)

logger = logging.getLogger(__name__)


# Default endpoint used by the demo + tests when the row's config
# omits ``endpoint_url``. Wikidata's published query service. Pinned
# locally rather than imported from the builder so the adapter can
# be constructed directly in unit tests.
DEFAULT_WIKIDATA_ENDPOINT: Final[str] = "https://query.wikidata.org/sparql"

# Per-request HTTP timeout in seconds. Federation timeouts wrap each
# adapter call, but a bare httpx default would still take 5 seconds
# to surface a network failure when the federation budget is 5s —
# the adapter timeout has to be strictly less than the federation
# budget so cancellation lands cleanly.
_HTTP_TIMEOUT_SECONDS: Final[float] = 4.0

# Maximum result rows we ask Wikidata for. The router applies its
# own LIMIT semantics on top via KALQuery.limit, but capping the
# SPARQL LIMIT here too means a malformed unbounded query can't
# blow up the response size. 1000 matches AdapterCapabilities'
# max_batch_size convention.
_SPARQL_HARD_LIMIT: Final[int] = 1000

# Wikidata accepts JSON results via the standard SPARQL Accept
# header. ``Accept: application/sparql-results+json`` is the
# canonical MIME type for SPARQL 1.1 Protocol; some endpoints also
# accept ``application/json`` but the canonical form is preferred
# for forward compatibility.
_SPARQL_JSON_ACCEPT: Final[str] = "application/sparql-results+json"


# Wikidata entity / property URIs are ``http://www.wikidata.org/
# entity/Q\\d+`` and ``http://www.wikidata.org/prop/direct/P\\d+``
# respectively. The compiled patterns let us cheaply turn a binding
# value back into a ``Q42`` / ``P31`` short form for the KAL nodes.
_QID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^https?://www\.wikidata\.org/entity/(Q\d+)$"
)
_PID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^https?://www\.wikidata\.org/prop/direct/(P\d+)$"
)

# Tight regexes used to whitelist values that get interpolated into
# SPARQL strings. Anything not matching is either escaped as a
# literal or rejected — the read-only Wikidata endpoint can't run
# destructive SQL, but a hostile predicate could still craft an
# expensive query that exhausts the WDQS budget or coerces the
# result shape. Defense-in-depth on every interpolation site.
_QID_BARE: Final[re.Pattern[str]] = re.compile(r"^Q\d+$")
_PID_BARE: Final[re.Pattern[str]] = re.compile(r"^P\d+$")
_SPARQL_VAR: Final[re.Pattern[str]] = re.compile(r"^\?[A-Za-z_]\w*$")


class _SPARQLValidationError(ValueError):
    """Raised when a Wikidata adapter caller passes an identifier that
    would break out of its SPARQL slot. Surfaces as ``ValueError`` so
    the federation router's catch-all classifies it as ``"internal"``;
    the connection-service handshake reports it as a config-shape
    failure.
    """


def _shorten_uri(uri: str) -> str:
    """Strip the Wikidata URI prefix to leave the bare QID/PID.

    Non-matching URIs return unchanged so labels (literal binding
    values) pass through untouched.
    """
    qid_match = _QID_PATTERN.match(uri)
    if qid_match is not None:
        return qid_match.group(1)
    pid_match = _PID_PATTERN.match(uri)
    if pid_match is not None:
        return pid_match.group(1)
    return uri


class WikidataAdapter:
    """KAL adapter wrapping the public Wikidata SPARQL endpoint.

    ``user_agent`` is required by Wikidata's policy — anonymous
    queries get rate-limited or refused. The builder feeds in a
    Knowly-identifying string at construction time; we pass it
    through every request.
    """

    def __init__(
        self,
        *,
        name: str,
        endpoint_url: str,
        user_agent: str,
        bearer_token: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        # ``http_client`` is injectable so tests can hand in an httpx
        # ``AsyncClient`` with a ``MockTransport`` instead of going
        # over the network. Production callers (the builder) leave it
        # ``None`` so the adapter constructs its own client and the
        # builder's ``aclose`` is the canonical cleanup path.
        self._name = name
        self._endpoint_url = endpoint_url
        self._headers: dict[str, str] = {
            "Accept": _SPARQL_JSON_ACCEPT,
            "User-Agent": user_agent,
        }
        if bearer_token:
            # Wikidata's public endpoint doesn't require auth, but a
            # mirror or enterprise endpoint might. Keep the token
            # out of logs by writing it only into the Authorization
            # header dict — ``httpx`` does not log request headers
            # at any default level.
            self._headers["Authorization"] = f"Bearer {bearer_token}"
        self._client = http_client or httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT_SECONDS, headers=self._headers
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> AdapterCapabilities:
        # Read-only; no writes, no deletes, no vector search.
        # ``supports_text_search`` stays False because the v1
        # ``_build_sparql`` doesn't honour ``KALQuery.text_search``
        # — the router would otherwise dispatch text queries to a
        # backend that silently ignores the search term. Flip
        # to True when (and only when) the SPARQL builder wires
        # in an ``rdfs:label CONTAINS ...`` clause.
        return AdapterCapabilities(
            can_read=True,
            can_write=False,
            can_delete=False,
            supports_vector_search=False,
            supports_text_search=False,
            supports_provenance=False,
            supports_verification_metadata=False,
            supports_atomic_writes=False,
            max_batch_size=_SPARQL_HARD_LIMIT,
        )

    async def aclose(self) -> None:
        """Idempotent under httpx — a second ``aclose`` is a no-op."""
        await self._client.aclose()

    async def __aenter__(self) -> WikidataAdapter:
        # ``async with`` support is purely for test ergonomics — the
        # production lifecycle is owned by ``WikidataBuilder.aclose``
        # via the ``AdapterFactory``. The protocol doesn't require
        # context-manager methods.
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        await self.aclose()

    async def query_triples(
        self,
        query: KALQuery,
        tenant_id: str | None = None,
    ) -> KALQueryResult:
        """Run a SPARQL SELECT for triples matching ``query.patterns``.

        Wikidata's tenant model is "everyone sees the same graph", so
        ``tenant_id`` is accepted (Protocol contract) but does not
        change the query. ``query.limit`` is honoured up to the
        configured hard cap; ``query.timeout_ms`` is the router's
        concern, not the adapter's (the adapter has its own httpx
        timeout to bound a single roundtrip).
        """
        sparql = self._build_sparql(query)
        rows = await self._execute_sparql(sparql)
        return KALQueryResult(triples=[_binding_to_triple(row) for row in rows])

    async def get_node(
        self,
        node_id: str,
        tenant_id: str | None = None,
    ) -> KALNode | None:
        """Resolve a QID to a labelled node. Returns ``None`` when
        the entity has no rdfs:label in any of the requested
        languages, which we treat as "node not found" for KAL
        purposes — a Wikidata QID with no label is effectively
        unidentifiable. ``node_id`` is validated against ``Q\\d+``
        before interpolation to close the SPARQL-injection vector
        (a caller passing ``"Q42 . ?x ?y ?z"`` would otherwise
        escape the WHERE braces).
        """
        validated_qid = _require_qid(node_id)
        sparql = (
            'SELECT ?label WHERE { '
            f'wd:{validated_qid} rdfs:label ?label . '
            'FILTER(LANG(?label) = "en") '
            '} LIMIT 1'
        )
        rows = await self._execute_sparql(sparql)
        if not rows:
            return None
        label_value = _extract_value(rows[0], "label")
        if not label_value:
            return None
        return KALNode(id=node_id, name=label_value)

    async def query_adjacent_triples(
        self,
        node_id: str,
        predicate: str | None = None,
        direction: str = "outgoing",
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[KALTriple]:
        """Walk one hop from ``node_id``. ``predicate`` is a PID
        (e.g. ``P31``) when given. ``direction`` is ``"outgoing"``,
        ``"incoming"``, or ``"both"`` — Wikidata triples are
        directional, so the adapter rewrites the WHERE clause
        accordingly. Both identifier inputs are regex-validated so
        an attacker can't inject SPARQL via the f-string sites in
        ``_build_adjacent_sparql``.
        """
        validated_qid = _require_qid(node_id)
        validated_pid = _require_pid(predicate) if predicate else None
        sparql = self._build_adjacent_sparql(
            validated_qid, validated_pid, direction, limit
        )
        rows = await self._execute_sparql(sparql)
        return [_binding_to_triple(row) for row in rows]

    # --- write methods raise; capability flags filter them out
    # at the router level too, this is the second line of defense.

    async def insert_triples(
        self,
        triples: list[KALTriple],
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError(
            "Wikidata adapter is read-only; insert_triples is not supported"
        )

    async def update_verification(
        self,
        triple_id: str,
        verification: VerificationMetadata,
        tenant_id: str | None = None,
    ) -> bool:
        raise CapabilityNotSupportedError(
            "Wikidata adapter is read-only; update_verification is not supported"
        )

    async def delete_triples(
        self,
        triple_ids: list[str],
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError(
            "Wikidata adapter is read-only; delete_triples is not supported"
        )

    async def quarantine_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError(
            "Wikidata adapter is read-only; quarantine is not supported"
        )

    async def restore_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError(
            "Wikidata adapter is read-only; restore is not supported"
        )

    async def search_similar_nodes(
        self,
        query_vector: list[float],
        tenant_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[KALNode, float]]:
        raise CapabilityNotSupportedError(
            "Wikidata adapter does not support vector search"
        )

    # --- internals

    def _build_sparql(self, query: KALQuery) -> str:
        # Empty patterns → the cheapest probe Wikidata accepts; this
        # is also what the builder's handshake fires, so building it
        # here keeps the "what does an empty query look like" answer
        # local to the adapter. ``query.limit`` is honoured even on
        # the probe path (the handshake passes ``limit=1``).
        cap = _capped_limit(query.limit)
        if not query.patterns:
            return f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT {cap}"
        # v1: take the first pattern only and warn callers loudly
        # when they pass more. Real multi-pattern SPARQL synthesis
        # would need per-pattern variable scoping (each pattern
        # contributes its own ``?s``/``?p``/``?o`` and the join
        # semantics depend on which vars overlap) — that's its
        # own design problem outside Story 9's "prove the
        # abstraction" scope. Warning rather than silently
        # truncating gives operators a signal that the federation
        # result is partial.
        if len(query.patterns) > 1:
            logger.warning(
                "wikidata adapter received %d patterns; only the "
                "first is honoured. Multi-pattern joins are a v2 "
                "feature.",
                len(query.patterns),
            )
        triples_clause = self._pattern_to_triples(query.patterns[0])
        return (
            f"SELECT ?s ?p ?o WHERE {{ {triples_clause} }} "
            f"LIMIT {cap}"
        )

    @staticmethod
    def _pattern_to_triples(pattern: KALTriplePattern) -> str:
        # Keep ``?s``/``?p``/``?o`` in the BGP and BIND the
        # concrete-valued slots so the SELECT projection sees all
        # three variables populated. Inlining a literal IRI directly
        # into the WHERE (e.g. ``wd:Q42 ?p ?o``) is valid SPARQL,
        # but the SELECT ``?s`` would then come back unbound on real
        # Wikidata → ``_binding_to_triple`` would yield
        # ``KALNode(id="")``. Symmetric with the BIND-anchored
        # fix in ``_build_adjacent_sparql``. ``_sparql_term``
        # whitelists the value before interpolation.
        binds: list[str] = []
        for slot, value in (
            ("s", pattern.subject),
            ("p", pattern.predicate),
            ("o", pattern.object),
        ):
            if value is None:
                continue
            binds.append(f"BIND({_sparql_term(value)} AS ?{slot})")
        bind_clause = " ".join(binds)
        bgp = "?s ?p ?o ."
        return f"{bind_clause} {bgp}" if bind_clause else bgp

    @staticmethod
    def _build_adjacent_sparql(
        node_id: str,
        predicate: str | None,
        direction: str,
        limit: int,
    ) -> str:
        # Every WHERE branch ``BIND``s every concrete-valued slot
        # into its variable so the response carries fully-populated
        # ``?s``/``?p``/``?o`` bindings regardless of direction. The
        # previous form left whichever slot was a literal IRI
        # unbound — real Wikidata returns those rows without that
        # binding and ``_binding_to_triple`` mapped them to
        # ``KALNode(id="")``. Now both the anchor QID *and* a
        # supplied predicate filter ride through BIND so the
        # SELECT projection always populates every column.
        cap = _capped_limit(limit)
        anchor_bind = f"BIND(wd:{node_id} AS ?{{anchor}})"
        pred_bind = (
            f"BIND(wdt:{predicate} AS ?p) " if predicate else ""
        )
        bgp = "?s ?p ?o ."
        if direction == "outgoing":
            where = f"{anchor_bind.format(anchor='s')} {pred_bind}{bgp}"
        elif direction == "incoming":
            where = f"{anchor_bind.format(anchor='o')} {pred_bind}{bgp}"
        else:
            where = (
                f"{{ {anchor_bind.format(anchor='s')} {pred_bind}{bgp} }} "
                "UNION "
                f"{{ {anchor_bind.format(anchor='o')} {pred_bind}{bgp} }}"
            )
        return f"SELECT ?s ?p ?o WHERE {{ {where} }} LIMIT {cap}"

    async def _execute_sparql(self, query: str) -> list[dict[str, object]]:
        # POST is the safer choice for arbitrary-length queries —
        # Wikidata's GET path has URL-length limits that show up
        # only in production with realistic queries. ``data=`` puts
        # the SPARQL in the body as ``application/x-www-form-
        # urlencoded`` which Wikidata accepts. Errors classify into
        # the KAL taxonomy so the federation router can render
        # them as ``AdapterError`` rows.
        response = await self._post_sparql(query)
        _raise_for_http_status(response.status_code)
        return _parse_results_envelope(response)

    async def _post_sparql(self, query: str) -> httpx.Response:
        # ``httpx.RequestError`` is the parent of every transport-
        # level failure (Connect/Read/Write/RemoteProtocol/
        # PoolTimeout/Proxy/Network/Timeout). Catching the whole
        # tree keeps the federation router classifying every
        # transport issue as ``connection`` rather than letting a
        # ``WriteError`` slip through as ``internal``. ``TimeoutException``
        # gets a more specific message because it's the most common
        # case operators debug.
        try:
            return await self._client.post(
                self._endpoint_url, data={"query": query}
            )
        except httpx.TimeoutException as exc:
            raise AdapterConnectionError(
                f"wikidata request timed out: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise AdapterConnectionError(
                f"wikidata endpoint unreachable: {exc}"
            ) from exc


# HTTP status sentinels — ``http.HTTPStatus`` gives them stable
# names, and centralising the auth-vs-connection split keeps the
# adapter's error mapping in one place.
_AUTH_FAILED_STATUSES: Final[frozenset[int]] = frozenset(
    {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}
)
_CLIENT_ERROR_FLOOR: Final[int] = HTTPStatus.BAD_REQUEST


def _raise_for_http_status(status_code: int) -> None:
    if status_code in _AUTH_FAILED_STATUSES:
        raise AdapterAuthError(
            f"wikidata rejected credentials (HTTP {status_code})"
        )
    if status_code >= _CLIENT_ERROR_FLOOR:
        raise AdapterConnectionError(
            f"wikidata returned HTTP {status_code}"
        )


def _parse_results_envelope(
    response: httpx.Response,
) -> list[dict[str, object]]:
    try:
        body = response.json()
    except ValueError as exc:
        raise AdapterConnectionError(
            f"wikidata returned non-JSON body: {exc}"
        ) from exc
    # Standard SPARQL JSON Results format:
    # ``{"results": {"bindings": [{...}, ...]}}``.
    results = body.get("results") if isinstance(body, dict) else None
    bindings = results.get("bindings") if isinstance(results, dict) else None
    if not isinstance(bindings, list):
        return []
    return [row for row in bindings if isinstance(row, dict)]


def _capped_limit(requested: int | None) -> int:
    # ``None`` → use the hard cap. A real ``0`` is collapsed to 1
    # because ``LIMIT 0`` is a valid SPARQL clause but rarely what a
    # caller meant — the federation router will pass the user's
    # original limit through, so a zero almost always indicates a
    # bug at the caller. Negative values are similarly clamped to
    # 1 so the SPARQL endpoint never sees a syntactically invalid
    # ``LIMIT -5``.
    if requested is None:
        return _SPARQL_HARD_LIMIT
    return max(1, min(requested, _SPARQL_HARD_LIMIT))


def _require_qid(node_id: str) -> str:
    if not _QID_BARE.match(node_id):
        raise _SPARQLValidationError(
            f"wikidata node_id must match {_QID_BARE.pattern!r}; "
            f"got {node_id!r}"
        )
    return node_id


def _require_pid(predicate: str) -> str:
    if not _PID_BARE.match(predicate):
        raise _SPARQLValidationError(
            f"wikidata predicate must match {_PID_BARE.pattern!r}; "
            f"got {predicate!r}"
        )
    return predicate


def _sparql_term(term: str) -> str:
    # SPARQL variable / IRI / literal disambiguation. The regex
    # checks are strict so a hostile field can't inject SPARQL via
    # any of the "pass-through" branches:
    # - ``?x``  → SPARQL variable, must match ``?[A-Za-z_]\\w*``
    # - bare ``Q\\d+`` / ``P\\d+`` → expand to the canonical IRI form
    # - everything else (including pre-shaped ``wd:`` / ``wdt:``
    #   IRIs) → escape as a quoted literal. We deliberately do NOT
    #   trust caller-supplied IRI prefixes, since the only safe
    #   prefix path is "regex-validated bare ID then prepend the
    #   prefix here".
    if _SPARQL_VAR.match(term):
        return term
    if _QID_BARE.match(term):
        return f"wd:{term}"
    if _PID_BARE.match(term):
        return f"wdt:{term}"
    # Escape the literal — backslash and quote are the only
    # characters SPARQL requires escaped inside a quoted literal.
    escaped = term.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _binding_to_triple(row: dict[str, object]) -> KALTriple:
    """Turn a SPARQL JSON-results row into a ``KALTriple``.

    Bindings come in as ``{"s": {"type": "uri", "value": "..."},
    "p": {...}, "o": {...}}``. We pull the ``value`` field and
    shorten the URI for readability. ``KALTriple.id`` stays ``None``
    because Wikidata's default SPARQL service doesn't expose
    statement-level identifiers — the router's dedup uses
    ``content_hash`` anyway, so an absent id is the honest answer.
    """
    subject = _shorten_uri(_extract_value(row, "s"))
    predicate = _shorten_uri(_extract_value(row, "p"))
    obj = _shorten_uri(_extract_value(row, "o"))
    return KALTriple(
        subject=KALNode(id=subject, name=subject),
        predicate=predicate,
        object=KALNode(id=obj, name=obj),
    )


def _extract_value(row: dict[str, object], key: str) -> str:
    binding = row.get(key)
    if not isinstance(binding, dict):
        return ""
    value = binding.get("value", "")
    return str(value)
