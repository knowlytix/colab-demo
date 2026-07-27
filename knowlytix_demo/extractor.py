# SPDX-License-Identifier: Apache-2.0
"""Lane 1 — extract triples from unstructured/markdown text (LLM extraction).

Wraps the public ``knowlytix.benchmark.ingest_markdown`` parser:
  * ``mode="regex"``  — deterministic, no LLM (good for offline/CI/demo).
  * ``mode="hybrid"`` — regex backbone + LLM augmentation (richer on prose;
    needs a provider key via GMS_LLM_MODEL).

Returns ``(head, relation, tail)`` triples so the GMS side stays
source-agnostic — the same shape ``kal_loader`` produces.
"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from .resources import resolve_fixture_path

from knowlytix.benchmark import ingest_markdown

Triple = tuple[str, str, str]

DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"

# Provenance relations the parser emits (e.g. ``in_section``) are not facts to
# verify and create high-degree hub nodes that distort a small graph's
# geometry. Drop them by default; callers can override.
DEFAULT_DROP_RELATIONS = frozenset({"in_section"})


def _llm_callable(llm_model: str | None) -> Callable[[str], str]:
    """Build a ``Callable[[str], str]`` over a LiteLLM backend (hybrid mode)."""
    from knowlytix.knowledge.llm_backend import LiteLLMBackend

    backend = LiteLLMBackend(llm_model or os.environ.get("GMS_LLM_MODEL", DEFAULT_MODEL))
    return lambda prompt: backend.call(system="", user=prompt)


def extract_triples(
    text: str,
    *,
    document_entity: str | None = None,
    mode: str = "regex",
    llm_model: str | None = None,
    drop_relations: frozenset[str] = DEFAULT_DROP_RELATIONS,
) -> tuple[list[Triple], dict[str, object]]:
    """Extract triples from a markdown ``text`` blob.

    Returns ``(triples, stats)``; ``stats`` is the DocumentGraph's own stats
    (triples / ENM counts) so callers can see what the parser produced.
    Provenance relations in ``drop_relations`` are filtered from the returned
    triples (they remain in ``stats``).
    """
    callable_ = _llm_callable(llm_model) if mode in ("hybrid", "llm_only") else None

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(text)
        tmp_path = f.name
    try:
        graph = ingest_markdown(
            tmp_path, document_entity=document_entity, mode=mode, llm_callable=callable_,
        )
    finally:
        os.unlink(tmp_path)

    triples = [(h, r, t) for (h, r, t) in graph.triples if r not in drop_relations]
    return triples, graph.stats()


def extract_triples_from_file(
    path: str | Path, **kwargs
) -> tuple[list[Triple], dict[str, object]]:
    """Convenience: read a .md file and extract its triples."""
    resolved_path = resolve_fixture_path(path)
    return extract_triples(resolved_path.read_text(encoding="utf-8"), **kwargs)
