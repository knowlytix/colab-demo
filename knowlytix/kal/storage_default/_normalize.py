"""Entity name canonicalization for the KAL default store.

Ported from ``knowly.modules.verification.entity_matcher`` so the
``storage_default/`` package is self-contained — the mirror script
ships this directory verbatim into the standalone package without
needing to drag in ``knowly.modules.verification``. Both implementations
must stay byte-equivalent so a write through ``KnowlyKGStore`` followed
by a read through ``KalDefaultKGStore`` (or vice versa, e.g. during
federation) resolves the same entities.

Underscore-prefixed because this is an internal helper of the default
store, not a Protocol surface. ``KGStore.normalize_name`` is explicitly
not on the Protocol (see Phase A design-decisions doc).
"""

from __future__ import annotations

import re

_MULTI_SPACE = re.compile(r"\s+")
_NUMERIC_RE = re.compile(r"^-?[\d]+(?:\.[\d]+)?(?:[eE][+-]?\d+)?$")


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name to canonical form.

    Transformations:
    - Lowercase
    - Underscores and hyphens → spaces (hyphens only between letters,
      so e.g. ``"foo-bar"`` → ``"foo bar"`` but ``"-1"`` stays ``"-1"``)
    - Collapse multiple spaces
    - Strip leading/trailing whitespace
    - Numeric values normalized to canonical form (``"0.0"`` → ``"0"``,
      ``"1.000"`` → ``"1"``)
    - Preserve slashes, brackets, scientific notation
    """
    name = name.lower().strip()

    # Numeric strings go through float round-trip → canonical text.
    if _NUMERIC_RE.match(name):
        try:
            num = float(name)
            if num == int(num) and abs(num) < 1e15:
                return str(int(num))
            return str(num)
        except (ValueError, OverflowError):
            pass

    name = name.replace("_", " ")
    name = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", name)
    name = _MULTI_SPACE.sub(" ", name)
    return name.strip()
