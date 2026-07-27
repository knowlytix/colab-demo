# SPDX-License-Identifier: Apache-2.0
"""Bundled benchmark instances — financial report markdown files."""

import os

INSTANCES_DIR = os.path.dirname(__file__)


def get_instance_path(name: str) -> str:
    """Return absolute path to a bundled instance markdown file.

    Args:
        name: Instance name (e.g. "model_validation", "fair_lending").
              The .md extension is added automatically.

    Returns:
        Absolute path to the markdown file.

    Raises:
        FileNotFoundError: If the instance does not exist.
    """
    path = os.path.join(INSTANCES_DIR, f"{name}.md")
    if not os.path.exists(path):
        available = [f.replace(".md", "") for f in os.listdir(INSTANCES_DIR)
                     if f.endswith(".md")]
        raise FileNotFoundError(
            f"Instance '{name}' not found. Available: {available}"
        )
    return path


def list_instances() -> list[str]:
    """Return names of all bundled instances."""
    return sorted(
        f.replace(".md", "") for f in os.listdir(INSTANCES_DIR)
        if f.endswith(".md")
    )
