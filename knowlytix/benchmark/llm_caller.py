# SPDX-License-Identifier: Apache-2.0
"""LLM caller for benchmark evaluation — routed through :mod:`gms.llm`.

``create_client()`` returns an :class:`gms.llm.LLMClient` configured via
:class:`gms.llm.LLMSettings` for :data:`ModelPurpose.SCORER`.
``call_llm()`` accepts either that client or a legacy Anthropic SDK
client (duck-typed) for backwards compatibility, and routes through
LiteLLM under the hood.

Callers who still want direct Anthropic SDK access can call the private
``_legacy_anthropic_call`` path; new code should migrate to
``create_client()`` returning an LLMClient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from knowlytix.core.llm import LLMClient


SYSTEM_PROMPT = (
    "You are an expert analyst. You have been given a report in markdown format. "
    "Answer questions precisely using ONLY the data in the report. "
    "Do NOT approximate or round values — copy numbers exactly as they appear. "
    "Format your answer using the specified tags."
)


def call_llm(
    client: Any,
    markdown_context: str,
    question_prompt: str,
    model: str | None = None,
) -> str:
    """Dispatch a scorer-tier LLM call and return the assistant text.

    ``client`` is either an :class:`gms.llm.LLMClient` (new path, default
    return of :func:`create_client`) or a raw ``anthropic.Anthropic``
    instance (legacy duck type, preserved only so existing tests that
    inject a bare SDK client keep working).

    ``model`` optionally overrides the client's bound model for this
    single call; when ``None`` the client's default (driven by
    ``GMS_LLM_MODEL_SCORER`` or ``GMS_LLM_MODEL``) is used.
    """

    # New path: gms.llm.LLMClient carries `.complete(messages, **kw)`.
    if hasattr(client, "complete"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"<report>\n{markdown_context}\n</report>\n\n"
                    f"{question_prompt}"
                ),
            },
        ]
        kwargs: dict[str, Any] = {"max_tokens": 1024}
        if model:
            kwargs["model"] = model
        return client.complete(messages, **kwargs)

    # Legacy fallback: raw anthropic.Anthropic() instance.
    return _legacy_anthropic_call(client, markdown_context, question_prompt, model)


def _legacy_anthropic_call(
    client: Any,
    markdown_context: str,
    question_prompt: str,
    model: str | None,
) -> str:
    """Backward-compat path for callers still holding an ``anthropic.Anthropic``.

    Retained behind a ``hasattr(client, 'complete')`` check so new
    LLMClient-based code takes the fast path and legacy tests injecting
    a bare SDK client keep working. Callers that land here must supply
    an explicit model name (no hardcoded default).
    """

    if not model:
        raise ValueError(
            "call_llm received a legacy anthropic.Anthropic client without "
            "an explicit model name. Migrate to gms.llm.LLMClient via "
            "knowlytix.benchmark.llm_caller.create_client(), which reads "
            "GMS_LLM_MODEL_SCORER (falling back to GMS_LLM_MODEL)."
        )
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"<report>\n{markdown_context}\n</report>\n\n"
                    f"{question_prompt}"
                ),
            }
        ],
    )
    return response.content[0].text


def create_client() -> "LLMClient":
    """Return a scorer-tier :class:`gms.llm.LLMClient`.

    The model is resolved from :class:`gms.llm.LLMSettings` for
    :data:`ModelPurpose.SCORER` — overridable via
    ``GMS_LLM_MODEL_SCORER`` or falling back to ``GMS_LLM_MODEL``.
    Provider API keys (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
    ``OLLAMA_BASE_URL`` …) are read from the environment by LiteLLM.
    """

    from knowlytix.core.llm import ModelPurpose, get_llm

    return get_llm(ModelPurpose.SCORER)
