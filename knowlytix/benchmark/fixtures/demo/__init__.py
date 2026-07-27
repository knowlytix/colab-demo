# SPDX-License-Identifier: Apache-2.0
"""Bundled read-only demo fixtures shipped with knowlytix-bench.

Loaded by demo-mode (``KNOWLYTIX_DEMO_MODE=1``, or keyless) notebooks and by
:class:`knowlytix.core.llm.client.DemoLLMClient`.  See the public
:func:`load` helper for the top-level entry point.

Ships a curated **credit-risk** demo corpus (GMS-273) — a small
``GeometricKnowledgeGraph``, pre-scored benchmark questions, canned LLM
responses, one MockHarness trace, and a pre-ingested document.  Regenerate
with ``scripts/build_demo_corpus.py`` (see this directory's ``README.md``).
The ``default`` fallbacks below keep :func:`load` robust if a file is
absent.
"""

from __future__ import annotations

import importlib.resources
import json
from typing import Any


def load() -> dict[str, Any]:
    """Return all bundled demo fixtures as a dict.

    Keys match the fixture categories documented in
    ``distribution/user_journey_plan.md`` §Open decisions #5:

    - ``"knowledge_graph"``: path to the small demo KG blob (string).
    - ``"questions"``: pre-scored benchmark questions (list).
    - ``"agent_trace"``: one recorded MockHarness trace (dict).
    - ``"document"``: pre-ingested demo document metadata (dict).
    - ``"llm_responses"``: canned prompt→response map (dict).

    Each value is the parsed JSON content of the corresponding shipped
    fixture file (or a path-reference string for the non-JSON KG blob).
    Missing files resolve to empty defaults so callers don't need to
    guard on presence.
    """
    pkg = importlib.resources.files("knowlytix.benchmark.fixtures.demo")
    out: dict[str, Any] = {}
    for name, default in (
        ("questions", []),
        ("agent_trace", {}),
        ("document", {}),
        ("llm_responses", {}),
    ):
        try:
            out[name] = json.loads(
                pkg.joinpath(f"{name}.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            out[name] = default
    # The KG blob is binary (pickled tensors); return a resource reference
    # instead of reading into memory.  Callers pass this to the loader of
    # their choice.
    out["knowledge_graph"] = str(pkg.joinpath("knowledge_graph.pkl"))
    return out


__all__ = ["load"]
