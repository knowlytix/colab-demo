# SPDX-License-Identifier: Apache-2.0
"""Fail-early environment validation for KnowlytiX notebook and script users.

The helper validates the installed KnowlytiX packages and license, then
optionally verifies the API key required by ``GMS_LLM_MODEL``. In Google
Colab it also reads matching values from the notebook's Secrets store.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Iterable

DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"

HOME_KEY_FILES = {
    "ANTHROPIC_API_KEY": "~/.anthropic_key",
    "OPENAI_API_KEY": "~/.openai_key",
    "GEMINI_API_KEY": "~/.gemini_key",
    "GOOGLE_API_KEY": "~/.google_key",
}

# Colab Secrets names to copy into os.environ when available. Existing
# environment variables always win.
COLAB_SECRET_NAMES = (
    "GMS_LLM_MODEL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)

REQUIRED_PACKAGES = (
    "knowlytix.kal",
    "knowlytix.core",       # Import also triggers the license check.
    "knowlytix.knowledge",
    "knowlytix.benchmark",
    "knowlytix.harness",
)

# Each provider maps to alternative environment variables. At least one
# variable in the tuple must be present.
PROVIDER_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

# Providers that do not require an API key in the local environment.
KEYLESS_PROVIDERS = {"ollama"}


class PreflightError(RuntimeError):
    """Raised when the demo environment is missing packages, license or keys."""


def _load_colab_secrets(secret_names: Iterable[str] = COLAB_SECRET_NAMES) -> None:
    """Copy available Colab Secrets into ``os.environ`` without printing them."""
    try:
        from google.colab import userdata
    except ImportError:
        return

    for name in secret_names:
        if os.environ.get(name):
            continue
        try:
            value = userdata.get(name)
        except Exception:
            # Missing, unshared or inaccessible secret. The later validation
            # produces a clear message when the value is actually required.
            continue
        if value:
            os.environ[name] = str(value).strip()


def _load_home_key_files() -> None:
    """Load provider keys from conventional files when env vars are unset."""
    for env_var, path_str in HOME_KEY_FILES.items():
        if os.environ.get(env_var):
            continue
        path = Path(path_str).expanduser()
        if not path.is_file():
            continue
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            os.environ[env_var] = value


def _load_environment() -> None:
    """Load .env, home key files and Colab Secrets, without overwriting env."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass  # python-dotenv is optional.
    else:
        load_dotenv(override=False)

    _load_home_key_files()
    _load_colab_secrets()


def _check_packages(packages: Iterable[str] = REQUIRED_PACKAGES) -> list[str]:
    problems: list[str] = []

    for package in packages:
        try:
            importlib.import_module(package)
        except ModuleNotFoundError as exc:
            if exc.name == package or package.startswith(f"{exc.name}."):
                problems.append(
                    f"  - package {package!r} is not installed"
                )
            else:
                problems.append(
                    f"  - {package!r} is installed but dependency "
                    f"{exc.name!r} is missing"
                )
        except ImportError as exc:
            problems.append(f"  - {package!r} failed to import: {exc}")
        except Exception as exc:
            # knowlytix.core can fail here when the license is not activated.
            problems.append(
                f"  - {package!r} failed to initialize: {exc} "
                "(check the KnowlytiX license activation)"
            )

    return problems


def _check_provider_key(model: str | None = None) -> list[str]:
    configured_model = model or os.environ.get("GMS_LLM_MODEL", DEFAULT_MODEL)
    provider = (
        configured_model.split("/", 1)[0].lower()
        if "/" in configured_model
        else ""
    )

    if provider in KEYLESS_PROVIDERS:
        return []

    env_vars = PROVIDER_ENV.get(provider)
    if not env_vars:
        return [
            f"  - GMS_LLM_MODEL={configured_model!r} has unknown provider "
            f"prefix {provider!r}; set the model and its provider key explicitly"
        ]

    if not any(os.environ.get(name) for name in env_vars):
        alternatives = " or ".join(env_vars)
        return [
            f"  - GMS_LLM_MODEL={configured_model!r} needs {alternatives}; "
            "add it under Colab Secrets, a .env file or os.environ"
        ]

    return []


def preflight(
    *,
    require_llm: bool = True,
    model: str | None = None,
    packages: Iterable[str] = REQUIRED_PACKAGES,
) -> None:
    """Validate packages, license and optionally the configured LLM key.

    Parameters
    ----------
    require_llm:
        Set to ``False`` for notebook sections that make no LLM calls.
    model:
        Optional model override used only for the provider-key check. When
        omitted, ``GMS_LLM_MODEL`` is read, falling back to ``DEFAULT_MODEL``.
    packages:
        Import paths to validate. The default checks all KnowlytiX demo wheels.
    """
    _load_environment()

    problems = _check_packages(packages)
    if require_llm:
        problems.extend(_check_provider_key(model))

    if problems:
        raise PreflightError(
            "KnowlytiX preflight failed:\n"
            + "\n".join(problems)
            + "\n\nFix the items above, then rerun this cell."
        )

    configured_model = model or os.environ.get("GMS_LLM_MODEL", DEFAULT_MODEL)
    if require_llm:
        print(
            "Preflight OK — packages, license and provider key are present "
            f"for {configured_model}."
        )
    else:
        print("Preflight OK — packages and license are present (LLM key skipped).")
