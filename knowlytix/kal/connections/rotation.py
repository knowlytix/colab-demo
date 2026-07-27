"""Bulk credential-rotation helper for ``kal_connections``.

Story 10 of the connection-registry plan. The script
``scripts/rotate_kal_credentials.py`` is the operator-facing CLI; the
load-bearing logic lives here so the rotation contract is testable
without driving the CLI shape, and so ``knowly.kal.secrets`` stays a
leaf module (a primitive in the ``kal`` package that does not know
about the persistence layer).

The flow is documented in ``knowly.kal.secrets.KALSecretBox`` and on
the CLI script; in short: prepend the new key to
``KNOWLY_KAL_CREDENTIAL_KEY``, boot the app, run this helper with
``commit=True`` to re-encrypt every row under the new primary, then
drop the historical key at the next deploy.
"""

from __future__ import annotations

from cryptography.fernet import InvalidToken
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from knowlytix.kal.connections import repository
from knowlytix.kal.connections.models import KALConnection
from knowlytix.kal.secrets import KALSecretBox


async def rotate_kal_connections(
    session: AsyncSession,
    secret_box: KALSecretBox,
    *,
    commit: bool,
) -> tuple[int, int]:
    """Walk every ``kal_connections`` row and rotate its
    ``credentials`` BYTEA under the box's primary key.

    Scope: ``repository.list_all`` is **unscoped by status**, so
    both ``'active'`` and ``'disabled'`` rows are rotated.
    Disabled rows still hold ciphertext under whatever key was
    primary when they were last written; once the operator drops
    the historical key from the env list, a disabled row that
    wasn't rotated becomes undecryptable on the next ``/enable``
    or ``/test``. Rotating everything keeps the on-disk state
    consistent with the env-key set.

    Returns ``(rows_seen, rows_rotated)``. ``rows_rotated`` counts
    rows whose ciphertext actually changed; rows already under the
    primary key are skipped (``MultiFernet.rotate`` always
    produces fresh bytes because Fernet ciphertexts embed a
    timestamp, so a naive byte-equality short-circuit would always
    report "rotated"). When ``commit=False`` the function still
    counts what *would* rotate but skips the write — that's how
    the script's dry-run path reports without mutating the table.
    """
    rows = await repository.list_all(session)
    rotated = 0
    for row in rows:
        if secret_box.is_primary_encrypted(row.credentials):
            continue
        try:
            new_ciphertext = secret_box.rotate(row.credentials)
        except InvalidToken as exc:
            # A row encrypted by a key not in the current env list
            # would otherwise abort the loop with a bare
            # ``InvalidToken`` — operators get a confusing
            # traceback and no idea which row is poisoned.
            # Re-raise with the row id so the failure is
            # actionable.
            raise InvalidToken(
                f"kal_connections row {row.id} could not be decrypted by "
                "any loaded key — add the missing historical key to "
                "KNOWLY_KAL_CREDENTIAL_KEY and re-run."
            ) from exc
        rotated += 1
        if not commit:
            continue
        await session.execute(
            update(KALConnection)
            .where(KALConnection.id == row.id)
            .values(credentials=new_ciphertext)
        )
    if commit:
        await session.commit()
    return len(rows), rotated
