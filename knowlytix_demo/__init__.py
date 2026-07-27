# SPDX-License-Identifier: Apache-2.0
"""Utilities and fixtures for the public KnowlytiX KAL + GMS demo."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("knowlytix-demo")
except PackageNotFoundError:  # source-tree execution
    __version__ = "0.0.0.dev0"

from .resources import (
    available_fixtures,
    fixture_path,
    materialize_fixtures,
    resolve_fixture_path,
)

__all__ = [
    "__version__",
    "available_fixtures",
    "fixture_path",
    "materialize_fixtures",
    "resolve_fixture_path",
]
