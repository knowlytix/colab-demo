# SPDX-License-Identifier: Apache-2.0
"""Access fixture files bundled with :mod:`knowlytix_demo`."""
from __future__ import annotations

from pathlib import Path
import shutil

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def available_fixtures() -> tuple[str, ...]:
    """Return the names of all bundled fixture files."""
    return tuple(sorted(path.name for path in _FIXTURE_DIR.iterdir() if path.is_file()))


def fixture_path(name: str | Path) -> Path:
    """Return the installed filesystem path for one bundled fixture."""
    candidate = _FIXTURE_DIR / Path(name).name
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Unknown KnowlytiX demo fixture {Path(name).name!r}; "
            f"available: {', '.join(available_fixtures())}"
        )
    return candidate


def resolve_fixture_path(path: str | Path) -> Path:
    """Resolve a requested path, falling back to a same-named bundled fixture.

    Existing paths always win. This lets the unchanged public notebook retain
    values such as ``fixtures/gov_roles.jsonl`` even though the wheel is
    installed under ``site-packages`` rather than beside the notebook.
    """
    requested = Path(path).expanduser()
    if requested.is_file():
        return requested

    bundled = _FIXTURE_DIR / requested.name
    if bundled.is_file():
        return bundled

    return requested


def materialize_fixtures(destination: str | Path = "/content/fixtures") -> Path:
    """Copy all bundled fixtures into a normal directory and return it."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for source in _FIXTURE_DIR.iterdir():
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    return destination.resolve()
