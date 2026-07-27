"""Fernet-backed encrypt/decrypt for KAL adapter credentials.

The only path through which credentials cross the encryption boundary
on the way into the connection registry's ``credentials`` BYTEA
column.

Key rotation (Story 10): construct ``KALSecretBox`` with one
**primary** key plus zero or more **historical** keys to enable
``MultiFernet``-backed rotation. ``encrypt`` always uses the primary
key; ``decrypt`` tries the primary first and falls back to each
historical key in order. Rotating a key means:

1. Generate a new key, prepend it to ``KAL_CREDENTIAL_KEY``
   (the comma-separated list — primary first, prior keys after).
2. Boot the app. Reads of existing rows decrypt with the historical
   key; writes encrypt with the new primary.
3. Run ``scripts/rotate_kal_credentials.py`` (which delegates to
   ``knowlytix.kal.connections.rotation.rotate_kal_connections``)
   to re-encrypt every row's ``credentials`` BYTEA under the new
   primary key.
4. Once every row has been re-encrypted, drop the historical key
   from the env list at the next deploy.

Importing this module never reads env vars, so dev/test environments
that don't configure ``KAL_CREDENTIAL_KEY`` boot without surprises —
the constructor fails fast at first secret op instead.

Environment-variable naming note (v0.2.0): the env var is
``KAL_CREDENTIAL_KEY`` (was ``KNOWLY_KAL_CREDENTIAL_KEY`` in
knowly's in-tree v0.1.x). Consumers cutting over should rename the
env var; the in-tree form will continue working in knowly until the
cutover is complete.
"""

from __future__ import annotations

import json
import os
from typing import cast

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

# Separator the env-var ``KAL_CREDENTIAL_KEY`` uses to join multiple
# keys during rotation. Comma is a conventional env-list separator
# and is not part of the Fernet key alphabet (url-safe base64), so
# it can't collide with a key byte.
KEY_SEPARATOR: str = ","

# Env-var name. Single source of truth — both ``from_env`` and any
# operator-facing error message reference this constant.
CREDENTIAL_KEY_ENV_VAR: str = "KAL_CREDENTIAL_KEY"


class KALSecretBox:
    """Fernet (or ``MultiFernet``) encrypt/decrypt for adapter credentials.

    The plaintext domain is restricted to ``dict[str, object]`` rather
    than ``bytes`` so callers don't have to think about JSON
    serialisation themselves: the box handles ``json.dumps`` on encrypt
    and ``json.loads`` on decrypt, with UTF-8 as the wire encoding for
    string values (covers passwords with non-ASCII characters).

    Two construction shapes:

    - ``KALSecretBox(primary_key)`` — single-key mode, backed directly
      by ``cryptography.fernet.Fernet``. Identical behaviour to the
      pre-Story-10 form.
    - ``KALSecretBox(primary_key, historical_keys=[old_key, ...])`` —
      rotation mode, backed by ``cryptography.fernet.MultiFernet``.
      ``encrypt`` writes with the primary key; ``decrypt`` falls back
      through each historical key until one accepts the ciphertext or
      every key has been tried.
    """

    def __init__(
        self,
        primary_key: str | bytes,
        historical_keys: list[str | bytes] | None = None,
    ) -> None:
        """Construct from one or more 32-url-safe-base64-byte Fernet keys.

        Raises ``ValueError`` if the primary key is empty, any key is
        malformed, any historical key is empty (common rotation
        mistake from a trailing comma in the env list), or the primary
        key also appears in the historical list (operator likely
        pasted the same key twice — would silently report a successful
        rotation that did nothing).
        """
        if not primary_key:
            raise ValueError(
                "KALSecretBox requires a non-empty Fernet key; set "
                f"{CREDENTIAL_KEY_ENV_VAR} to a 32 url-safe-base64-byte "
                "string (output of Fernet.generate_key())."
            )
        keys = [primary_key, *(historical_keys or [])]
        _validate_key_list(keys)
        try:
            fernets = [Fernet(k) for k in keys]
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "KALSecretBox key is not a valid Fernet key"
            ) from exc
        # ``MultiFernet`` accepts a list and uses the first for
        # encrypt while attempting each in order on decrypt — a
        # single-key list behaves identically to the bare ``Fernet``,
        # so we use ``MultiFernet`` unconditionally. This collapses
        # the two encrypt/decrypt code paths into one and keeps the
        # rotation behaviour exercised by every boot, not just by
        # rotation-mode deployments. We also keep a primary-only
        # Fernet so ``is_primary_encrypted`` can probe a row's
        # ciphertext without consulting the historical keys — that's
        # how the rotation script short-circuits already-rotated rows.
        self._fernet: MultiFernet = MultiFernet(fernets)
        self._primary_fernet: Fernet = fernets[0]
        self._key_count: int = len(fernets)

    @property
    def key_count(self) -> int:
        """Number of keys in the box (primary + historical).

        Exposed for diagnostics + the rotation script's progress
        reporting: an operator running ``rotate_kal_credentials.py``
        wants to see "2 keys loaded — primary + 1 historical" so
        they know the rotation path is active.
        """
        return self._key_count

    @property
    def has_historical_keys(self) -> bool:
        """True when the box carries at least one historical key —
        i.e. rotation is configured. The script's ``--commit``
        guard uses this directly so the "did the operator
        actually set up rotation?" check has a self-documenting
        name rather than a ``key_count >= 2`` comparison.
        """
        return self._key_count > 1

    @classmethod
    def from_env(cls, env_var: str = CREDENTIAL_KEY_ENV_VAR) -> KALSecretBox:
        """Build a box from an environment variable.

        The value may be a single key OR a ``KEY_SEPARATOR``-joined
        list with the primary key first and historical keys after,
        for rotation. Missing / empty env var raises ``ValueError``
        via the constructor's "empty primary" branch.
        """
        raw = os.environ.get(env_var, "")
        primary, historical = _split_key_list(raw)
        return cls(primary, historical_keys=list(historical))

    def encrypt(self, plain: dict[str, object]) -> bytes:
        return self._fernet.encrypt(json.dumps(plain).encode("utf-8"))

    def decrypt(self, ciphertext: bytes) -> dict[str, object]:
        # Raise ``InvalidToken`` rather than returning a non-dict and
        # letting the type confusion surface deep in the caller: a
        # tampered-but-still-decryptable payload that decoded to a
        # JSON list or string would otherwise slip past the ``cast``
        # and reach Pydantic validation as a TypeError. ``MultiFernet``
        # raises ``InvalidToken`` itself when every key fails, so the
        # caller's exception contract matches the single-key form.
        plain = json.loads(self._fernet.decrypt(ciphertext).decode("utf-8"))
        if not isinstance(plain, dict):
            raise InvalidToken(
                "KALSecretBox payload decoded to non-dict; expected JSON object"
            )
        return cast(dict[str, object], plain)

    def rotate(self, ciphertext: bytes) -> bytes:
        """Re-encrypt ``ciphertext`` under the primary key.

        ``MultiFernet.rotate`` decrypts using whichever historical
        key matches, then re-encrypts with the primary. The
        rotation script calls this in a loop over every row's
        ``credentials`` BYTEA; once every row's ciphertext has
        been rotated, the historical keys can be safely dropped
        from the env list.
        """
        return self._fernet.rotate(ciphertext)

    def is_primary_encrypted(self, ciphertext: bytes) -> bool:
        """Return True iff ``ciphertext`` decrypts under the primary
        key alone (i.e. doesn't need a historical-key fallback).

        Fernet ciphertexts embed an IV + timestamp, so calling
        ``rotate`` on an already-primary row still produces fresh
        bytes — a naive byte-equality check after ``rotate`` would
        never report "no change". This probe is the one safe way
        to detect "already on the primary"; the rotation script
        uses it to skip rows that don't need a DB write.
        """
        try:
            self._primary_fernet.decrypt(ciphertext)
        except InvalidToken:
            return False
        return True


def _validate_key_list(keys: list[str | bytes]) -> None:
    # Two failure modes the constructor would otherwise surface
    # awkwardly: (1) an empty entry from a trailing-comma env value;
    # (2) the primary key duplicated in the historical list, which
    # silently produces a "rotated 0 rows" run that an operator might
    # mistake for "rotation finished". Both raise ``ValueError`` so
    # ``KALSecretBox`` callers see the same exception class as for
    # malformed-key cases.
    for index, key in enumerate(keys):
        if not key:
            raise ValueError(
                f"KALSecretBox key at position {index} is empty — "
                f"check for a trailing comma in {CREDENTIAL_KEY_ENV_VAR}"
            )
    normalised = [k if isinstance(k, bytes) else k.encode("ascii") for k in keys]
    if len(set(normalised)) != len(normalised):
        raise ValueError(
            "KALSecretBox key list contains duplicates — every "
            "historical key must differ from the primary key, "
            "otherwise rotation silently no-ops."
        )


def _split_key_list(raw: str) -> tuple[str, list[str]]:
    """Split the env-var value into (primary, historical).

    Empty input → ``("", [])`` so the constructor's "empty primary"
    branch fires with its descriptive error. Whitespace around each
    comma is stripped because operators editing env files commonly
    add a trailing space.
    """
    parts = [piece.strip() for piece in raw.split(KEY_SEPARATOR)]
    if not parts:
        return "", []
    primary = parts[0]
    historical = [p for p in parts[1:] if p]
    return primary, historical
