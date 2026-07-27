# SPDX-License-Identifier: Apache-2.0
"""Unified LLM interface for the ``knowlytix.knowledge`` package.

Two backend flavors are exposed:

- ``LiteLLMBackend`` — default for hosted providers (Anthropic, OpenAI,
  Google, Ollama, …). Routes through :mod:`gms.llm`, which wraps LiteLLM,
  so model selection, timeouts, retries, and temperature come from
  :class:`gms.llm.LLMSettings` (``GMS_LLM_*`` env vars).
- ``LocalTransformersBackend`` — unchanged, for on-device HuggingFace
  models on GPU.

For backwards compatibility, ``AnthropicBackend`` is retained as a thin
subclass of ``LiteLLMBackend`` that auto-prepends the ``anthropic/``
provider prefix when the caller passes a bare Claude model name (e.g.
``"claude-sonnet-4-20250514"``). New code should prefer
``LiteLLMBackend`` directly.
"""

from abc import ABC, abstractmethod

from knowlytix.knowledge.config import ConvertConfig


class LLMBackend(ABC):
    """Abstract LLM backend."""

    @abstractmethod
    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class LiteLLMBackend(LLMBackend):
    """Provider-agnostic backend routed through :mod:`gms.llm`.

    Accepts model names in the LiteLLM format ``<provider>/<id>`` (e.g.
    ``anthropic/claude-sonnet-4-20250514``, ``openai/gpt-4o``,
    ``ollama/llama3``). Internally constructs an :class:`gms.llm.LLMClient`
    with a custom model override and delegates ``.call()`` to its
    ``.complete()`` method.
    """

    def __init__(self, model: str):
        if not model or not model.strip():
            raise ValueError(
                "LiteLLMBackend requires a non-empty model name. "
                "Pass --model on the CLI, assign ConvertConfig.llm_model, "
                "or configure GMS_LLM_MODEL via gms.llm.LLMSettings."
            )
        from knowlytix.core.llm import LLMClient, LLMSettings

        self._model = model
        self._client = LLMClient(model=model, settings=LLMSettings())

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self._client.complete(messages, max_tokens=max_tokens)

    @property
    def model_name(self) -> str:
        return self._model


class AnthropicBackend(LiteLLMBackend):
    """Backwards-compatible Anthropic-flavored backend.

    Accepts either a bare Claude model name (e.g.
    ``"claude-sonnet-4-20250514"``) or a LiteLLM-format name (e.g.
    ``"anthropic/claude-sonnet-4-20250514"``). Bare names are auto-
    prefixed with ``anthropic/`` before routing through LiteLLM so
    existing callers that encoded Anthropic-specific defaults keep
    working.

    Exposes the original caller-supplied model via ``bare_model_name``
    for code paths that still need to call the Anthropic SDK directly
    (e.g. provider-specific features like web_search_20250305 in
    ``knowlytix/knowledge/web_agent.py``).

    New code should instantiate :class:`LiteLLMBackend` directly with a
    provider-prefixed model name.
    """

    def __init__(self, model: str):
        if not model or not model.strip():
            raise ValueError(
                "AnthropicBackend requires a non-empty model name. "
                "Set ConvertConfig.llm_model, pass --model on the CLI, "
                "or configure GMS_LLM_MODEL via gms.llm.LLMSettings."
            )
        # Auto-prefix bare model names so callers passing
        # "claude-sonnet-4-20250514" still work. "foo/" or "/bar" are
        # malformed; treat as bare and re-prefix so LiteLLM receives a
        # valid "anthropic/<id>".
        trimmed = model.strip()
        head, sep, tail = trimmed.partition("/")
        if head and tail:
            resolved = trimmed
            bare = tail
        else:
            bare = trimmed.strip("/")
            resolved = f"anthropic/{bare}"
        self._bare_model = bare
        super().__init__(resolved)

    @property
    def bare_model_name(self) -> str:
        """Original caller-supplied model without any ``provider/`` prefix.

        Use this when invoking provider-specific SDK features that
        expect a bare model identifier.
        """

        return self._bare_model


class LocalTransformersBackend(LLMBackend):
    """Local HuggingFace model on GPU."""

    def __init__(self, model_name: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_name = model_name
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device,
        )

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = f"{system}\n\n{user}" if system else user

        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        outputs = self._model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    @property
    def model_name(self) -> str:
        return self._model_name


def create_backend(config: ConvertConfig) -> LLMBackend:
    """Factory: create LLM backend from config.

    Returns :class:`LocalTransformersBackend` when the user requested the
    local backend and supplied a ``local_model_name``; otherwise
    constructs an :class:`AnthropicBackend` (which delegates to
    :class:`LiteLLMBackend` via LiteLLM).

    Raises :class:`ValueError` if no model name has been supplied —
    surfaces the misconfiguration before the first (failing) API call.
    """

    if config.llm_backend == "local" and config.local_model_name:
        return LocalTransformersBackend(config.local_model_name)
    if not config.llm_model:
        raise ValueError(
            "knowlytix.knowledge: ConvertConfig.llm_model is empty. Either pass "
            "--model on the CLI, assign `config.convert.llm_model` "
            "directly, or configure GMS_LLM_MODEL via "
            "gms.llm.LLMSettings."
        )
    return AnthropicBackend(config.llm_model)
