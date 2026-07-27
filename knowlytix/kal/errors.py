"""KAL exception hierarchy.

Per KAL spec v1.4.1 §11.3. Mirrors the ``AdapterError`` taxonomy that
the federation router emits and the connection service consumes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from knowlytix.kal.types import AdapterError


# Length cap on adapter exception messages before they cross a
# persistence or response boundary. This is a coarse defence — short
# secrets (e.g. an 8-char password) fit comfortably below the cap and
# pass through unmodified. Its job is to bound how much of an
# asyncpg / SQLAlchemy ``[SQL: ...]`` envelope leaks, not to redact;
# adapters that produce credential-rich errors should sanitise at the
# source. The full traceback still goes to ``logger.exception`` so
# debugging keeps the stack.
ADAPTER_ERROR_MESSAGE_MAX_CHARS: int = 200

# Suffix appended to over-length messages so callers can tell a
# truncated string from one that happened to fit. Coupled to the
# length-budget arithmetic in ``truncate_error_message`` — adjusting
# one without the other would either over- or under-fill the cap.
_TRUNCATION_SUFFIX: str = "..."


# Shared with ``AdapterErrorType`` in ``types.py``; declared here so
# the federation router and the connection-service can both classify
# without importing types.
type AdapterExceptionClass = Literal[
    "timeout",
    "auth",
    "connection",
    "transport",
    "capability",
    "circuit_open",
    "internal",
]


def truncate_error_message(message: str) -> str:
    """Cap an exception-derived message at the length budget."""
    if len(message) <= ADAPTER_ERROR_MESSAGE_MAX_CHARS:
        return message
    keep = ADAPTER_ERROR_MESSAGE_MAX_CHARS - len(_TRUNCATION_SUFFIX)
    return message[:keep] + _TRUNCATION_SUFFIX


def classify_adapter_exception(
    exc: BaseException,
) -> tuple[AdapterExceptionClass, str]:
    """Map an adapter-raised exception to ``(error_type, message)``.

    Shared between ``FederationRouter._classify_error`` (which feeds
    the ``AdapterError`` rows on ``KALQueryResult.errors``) and the
    connection-service's create-flow handshake handler. Message is
    already truncated to ``ADAPTER_ERROR_MESSAGE_MAX_CHARS``.

    Taxonomy:

    - ``AdapterAuthError`` → ``"auth"``
    - ``AdapterConnectionError`` → ``"connection"`` (KAL-domain class)
    - stdlib ``ConnectionError`` / ``OSError`` → ``"transport"``
      (raw transport leakage from an adapter that didn't wrap the
      error in ``AdapterConnectionError``; v0.4.0 Phase C spec §5.3
      row 3). ``TimeoutError`` is a subclass of ``OSError`` in
      Python 3.10+ and would also classify as ``"transport"`` here
      if it reached this helper — typically it doesn't, because
      federation timeouts come from ``asyncio.wait_for`` and are
      classified by the router itself.
    - everything else → ``"internal"`` with the exception type
      prefixed on the message

    Timeout / capability classification stays in the federation router
    because the timeout exception type is ``asyncio.TimeoutError``
    raised by ``wait_for`` — produced inside the router itself rather
    than by the adapter — and the capability case is the router's own
    defensive check. Callers that need the full taxonomy compose those
    cases above this helper.
    """
    if isinstance(exc, AdapterAuthError):
        return "auth", truncate_error_message(str(exc))
    if isinstance(exc, AdapterConnectionError):
        return "connection", truncate_error_message(str(exc))
    if isinstance(exc, (ConnectionError, OSError)):
        return "transport", truncate_error_message(
            f"{type(exc).__name__}: {exc}"
        )
    return "internal", truncate_error_message(f"{type(exc).__name__}: {exc}")


class KALError(Exception):
    """Base class for all KAL errors."""


class TenantRequiredError(KALError):
    """Raised when an adapter that requires tenant_id is called without one.

    Postgres adapter raises this when ``tenant_id`` is ``None`` or empty
    (per KAL spec §7.3). External adapters that ignore tenancy do not raise.
    """


class AdapterNotFoundError(KALError):
    """Raised by ``AdapterRegistry.get`` when no adapter with the given name
    is registered.
    """


class CapabilityNotSupportedError(KALError):
    """Raised when a Protocol method is called on an adapter that does not
    advertise the required capability. The adapter's own methods may raise
    this; the federation router prefers capability-filtering at the routing
    layer.
    """


class AdapterAuthError(KALError):
    """Raised by an adapter when authentication / authorization fails.

    The federation router catches this per-adapter and classifies it as
    ``error_type="auth"`` inside ``KALQueryResult.errors``; sibling
    adapters in the same federated query still return their results.
    """


class AdapterConnectionError(KALError):
    """Raised by an adapter when the underlying backend is unreachable.

    Classified as ``error_type="connection"`` inside
    ``KALQueryResult.errors``. Sibling adapters keep returning results.
    """


class AdapterWriteError(KALError):
    """Raised by ``FederationRouter.insert_triples`` when one or more
    adapters in a federated write fail.

    Per KAL spec §5.3 — writes are asymmetric to reads: a silent
    degraded write would leave the graph inconsistent, so the router
    raises rather than returning a structured-error result. The
    exception aggregates per-adapter failures and the count of triples
    that DID persist before/around the failure, so the caller can
    reconcile state without losing visibility into which adapters
    succeeded.

    Pre-dispatch validation errors (unknown target adapter, read-only
    target) are NOT wrapped here — they raise ``AdapterNotFoundError``
    directly because they're caller errors, not adapter failures, and
    short-circuit before any write is attempted.
    """

    def __init__(
        self,
        adapter_errors: list[AdapterError],
        successful_count: int,
    ) -> None:
        self.adapter_errors = adapter_errors
        self.successful_count = successful_count
        super().__init__(
            f"federation insert failed on {len(adapter_errors)} adapter(s); "
            f"{successful_count} triple(s) persisted across surviving adapters"
        )
