# SPDX-License-Identifier: Apache-2.0
"""Natural-language helpers over the GMS store (provider-agnostic).

Used by Phase D2 (NL front-end: `parse_question` + `compose_answer`) and Phase E
(DOE: `rephrase_question`).

    English question --(LLM tool-call)--> (source, [relations])
        --multi_hop--> answer  --(LLM)--> one-sentence grounded answer

Uses the package's own provider-agnostic LLM layer (`knowlytix.core.llm`):
  * the **structured parse** goes through `litellm.completion(..., tools=...)`
    directly (with the store vocab as enums + forced tool_choice) because
    `LLMClient.complete()` returns text only and can't surface tool_calls;
  * the **answer composition** uses `get_llm().complete(...)` (free text).

Switch providers/models via `GMS_LLM_MODEL` (default Claude — most reliable for
the forced-tool-call parse). Needs the provider key (preflight checks it).
"""
from __future__ import annotations

import json
from pathlib import Path

from .resources import resolve_fixture_path

import litellm
from .gms_query import Hop

from knowlytix.core.llm import get_llm

_PARSE_SYSTEM = (
    "You translate a user's question into a multi-hop query over a knowledge "
    "graph: a source entity plus an ordered relation chain whose left-to-right "
    "composition answers the question (relation_1 acts on the source, "
    "relation_2 on its result, ...). Call make_query using ONLY tokens from the "
    "provided enums."
)

_ANSWER_SYSTEM = (
    "Answer the user's question in ONE concise sentence, grounded ONLY in the "
    "facts provided. Do not add outside information; translate underscored "
    "identifiers into natural words."
)

# Hedged-inference mode (§6 of the e2e demo): the geometric chain the user
# asked for didn't validate against the source triples (``path_valid=False``),
# so we walk what *does* exist from the source and produce a deliberately
# labelled best-guess. The model MUST start with "INFERENCE — " so a reader
# can never confuse this with a grounded answer.
_INFERENCE_SYSTEM = (
    "The asked relation chain has NO grounded path. Using ONLY the available "
    "outgoing edges below, write a one-sentence best-guess answer to the "
    "user's question. Start your reply with the literal token 'INFERENCE — ' "
    "and explicitly note which relation you had to substitute. Do not add "
    "outside information. Translate underscored identifiers into natural words."
)
_INFERENCE_PREFIX = "INFERENCE — "

# No-graph baseline (§6 of the e2e demo): the LLM answers with no context, no
# vocab constraint, no facts. This is the hallucination control — kept
# deliberately minimal so the contrast with the grounded answer is the demo's
# headline, not a prompt-engineering artefact.
_NO_GRAPH_SYSTEM = (
    "You are a risk and governance analyst. Answer the user's question in "
    "ONE concise sentence."
)

# Doc-grounded baseline (§6 of the e2e demo): same system prompt as the no-graph
# baseline — the only A/B difference is whether the policy documents the
# extractor saw are also placed in the LLM's context window. This is the
# *fair* LLM comparison: both the extractor (offline) and the LLM have access
# to the same source material; the question is whether geometric grounding via
# GMS catches what an in-context LLM misses (omissions, drift across phrasings,
# confident fabrication when the docs don't cover the question).
_WITH_DOCS_SYSTEM = _NO_GRAPH_SYSTEM


def _make_query_tool(
    entity_vocab: list[str], relation_vocab: list[str]
) -> dict[str, object]:
    """OpenAI/LiteLLM tool schema constraining the parse to the store vocab."""
    return {
        "type": "function",
        "function": {
            "name": "make_query",
            "description": (
                "Translate the question into (source_entity, [relation, ...]). "
                "Use ONLY tokens from the enums."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "enum": entity_vocab},
                    "relations": {
                        "type": "array",
                        "items": {"type": "string", "enum": relation_vocab},
                        "minItems": 1, "maxItems": 4,
                    },
                },
                "required": ["source", "relations"],
            },
        },
    }


def parse_question(
    question: str, entity_vocab: list[str], relation_vocab: list[str]
) -> tuple[str, list[str]]:
    """Map an NL question to ``(source, relations)`` via a forced LLM tool-call."""
    client = get_llm()  # resolves GMS_LLM_MODEL + settings
    resp = litellm.completion(
        model=client.model,
        messages=[{"role": "system", "content": _PARSE_SYSTEM},
                  {"role": "user", "content": question}],
        tools=[_make_query_tool(entity_vocab, relation_vocab)],
        tool_choice={"type": "function", "function": {"name": "make_query"}},
        temperature=0,
        timeout=client.settings.timeout_seconds,
    )
    tool_calls = resp.choices[0].message.tool_calls
    if not tool_calls:
        raise RuntimeError("model did not return a make_query tool call")
    args = json.loads(tool_calls[0].function.arguments)
    return args["source"], list(args["relations"])


_REPHRASE_SYSTEM = (
    "Rewrite the user's question to match the given persona/style/length while "
    "preserving its EXACT meaning (the answer must be unchanged). Output ONLY "
    "the rewritten question on one line — no preamble, no quotes."
)


def rephrase_question(question: str, factors: dict[str, str]) -> str:
    """Rephrase a question per a DOE factor assignment (meaning preserved)."""
    spec = ", ".join(f"{k}={v}" for k, v in factors.items())
    text = get_llm().complete(
        [{"role": "system", "content": _REPHRASE_SYSTEM},
         {"role": "user", "content": f"Question: {question}\nFactors: {spec}\n"
                                     f"Rewrite it."}],
        max_tokens=120,
    )
    return text.strip().strip('"').strip()


_NO_GROUNDED_PATH_MSG = (
    "The graph has no grounded path that answers this question."
)


def compose_answer(
    question: str,
    answer: str | None,
    hops: list[Hop],
    *,
    path_valid: bool = True,
) -> str:
    """Compose a one-sentence NL answer grounded in the retrieved path.

    ``path_valid`` (default True for backward compat) is the calibrated gate
    described in ``gms_query.is_path_valid``: when False, the geometric walk
    produced a phantom tail (no supporting edge in the training triples) and
    we refuse rather than dressing up a hallucination as grounded.
    """
    if answer is None or not path_valid:
        return _NO_GROUNDED_PATH_MSG
    path = " ; ".join(f"{h.head} {h.relation} {h.tail}" for h in hops)
    user = (f"Question: {question}\n"
            f"Facts (graph path): {path}\n"
            f"Retrieved answer: {answer}\n"
            f"Write the one-sentence answer.")
    return get_llm().complete(
        [{"role": "system", "content": _ANSWER_SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=120,
    )


def available_edges(
    triples: list[tuple[str, str, str]], source: str
) -> list[tuple[str, str]]:
    """Return outgoing ``(relation, tail)`` edges from ``source`` in order.

    Pure; feeds :func:`compose_inference` so the hedged answer is grounded in
    what the graph *does* know about the source, even when the asked chain
    has no support.
    """
    return [(rel, tail) for head, rel, tail in triples if head == source]


def compose_inference(
    question: str,
    source: str,
    edges: list[tuple[str, str]],
) -> str:
    """Hedged one-sentence answer over the source's available outgoing edges.

    Used when ``path_valid`` is False — the asked chain has no support, so
    we produce a clearly-labelled best-guess instead of refusing entirely.
    The "INFERENCE — " prefix is enforced regardless of what the LLM returns;
    a downstream reader can never confuse this with a grounded answer.
    """
    if not edges:
        return (f"{_INFERENCE_PREFIX}no outgoing edges from {source!r} in the "
                f"graph — cannot infer.")
    facts = "\n".join(f"  {source} {rel} {tail}" for rel, tail in edges)
    user = (f"Question: {question}\n"
            f"Source entity: {source}\n"
            f"Available outgoing edges:\n{facts}\n"
            f"Write the one-sentence hedged answer.")
    text = get_llm().complete(
        [{"role": "system", "content": _INFERENCE_SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=140,
    ).strip()
    if text.upper().startswith("INFERENCE"):
        return text
    return _INFERENCE_PREFIX + text


def answer_without_graph(question: str) -> str:
    """LLM-only baseline — no graph context, no vocab constraint.

    Used in §6 as the hallucination control: same model, same question, no
    grounding. Whatever it returns is what a vanilla LLM would have said.
    """
    return get_llm().complete(
        [{"role": "system", "content": _NO_GRAPH_SYSTEM},
         {"role": "user", "content": question}],
        max_tokens=140,
    ).strip()


def _read_doc(path: str | Path) -> str:
    """Read a markdown doc and tag it with a header for in-context grounding."""
    p = resolve_fixture_path(path)
    return f"# Document: {p.name}\n\n{p.read_text(encoding="utf-8")}"


def answer_with_docs(question: str, doc_paths: list[str | Path]) -> str:
    """LLM baseline with the source markdown documents in context.

    The fair A/B against ``answer_without_graph``: same model, same system
    prompt, same question — the only difference is the policy documents are
    placed in the user message. Mirrors what a vanilla LLM-RAG setup would do
    if the same documents the extractor saw were retrieved.

    The system prompt is held identical to the no-graph baseline so the
    contrast in §6 is purely "with docs vs. without docs", not a
    prompt-engineering artefact. The model is free to answer from the docs,
    refuse, or guess — the demo's headline is which one it does.
    """
    docs = "\n\n---\n\n".join(_read_doc(p) for p in doc_paths)
    user = (f"Reference policy documents:\n\n{docs}\n\n"
            f"---\n\nQuestion: {question}")
    return get_llm().complete(
        [{"role": "system", "content": _WITH_DOCS_SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=140,
    ).strip()
