# SPDX-License-Identifier: Apache-2.0
"""GEODE agent LLM — single, locked entry point (local Qwen 3B only).

The GEODE actor's *only* runtime LLM is **Qwen 3B, run locally**. No
hosted/frontier model is invoked inside the agent loop. Every actor LLM call
obtains its callable from here, which rejects any non-Qwen-3B model — this is
how the small-model-viability premise (geometry + the loop compensate for a 3B
model) is kept honest rather than a large model quietly doing the work.
"""

from __future__ import annotations

from typing import Callable

QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"
_ALLOWED = {QWEN_3B, "Qwen/Qwen2.5-3B-Instruct-AWQ"}


def _assert_qwen_3b(model_name: str) -> None:
    if model_name not in _ALLOWED:
        raise ValueError(
            f"GEODE agent LLM must be Qwen 3B (one of {sorted(_ALLOWED)}); "
            f"got {model_name!r}. The agent runtime may not use a frontier model."
        )


def qwen_agent_callable(
    model_name: str = QWEN_3B,
    device: str = "cuda",
    *,
    system: str = "",
    max_tokens: int = 256,
) -> Callable[[str], str]:
    """Return the ``Callable[[str], str]`` the GEODE actor uses for every LLM
    step, backed exclusively by a local Qwen 3B (via the shipped on-device
    backend). Import of the heavy backend is deferred to call time.
    """
    _assert_qwen_3b(model_name)
    from knowlytix.knowledge.llm_backend import LocalTransformersBackend

    backend = LocalTransformersBackend(model_name, device=device)
    return lambda prompt: backend.call(system=system, user=prompt,
                                       max_tokens=max_tokens)
