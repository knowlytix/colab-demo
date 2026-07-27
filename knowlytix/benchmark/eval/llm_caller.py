# SPDX-License-Identifier: Apache-2.0
"""Domain-agnostic LLM caller for benchmark evaluation."""


SYSTEM_PROMPT = (
    "You are an expert analyst. You have been given a report in markdown format. "
    "Answer questions precisely using ONLY the data in the report. "
    "Do NOT approximate or round values — copy numbers exactly as they appear. "
    "Format your answer using the specified tags."
)


def call_llm(client, markdown_context, question_prompt,
             model="claude-sonnet-4-20250514"):
    """Call Claude API with the markdown as context and a question."""
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


def create_client():
    """Create an Anthropic client."""
    # Lazy import so module-level ``from knowlytix.benchmark.eval.llm_caller import call_llm``
    # works in environments that don't install the optional ``[llm]``
    # extra (e.g. CI's ``[dev]``-only install).
    import anthropic

    return anthropic.Anthropic()
