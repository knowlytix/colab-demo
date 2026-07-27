# SPDX-License-Identifier: Apache-2.0
"""Answer assembly: synthesize a grounded answer from retrieved facts + spans.

The synthesis LLM (user's choice) sees the GMS-retrieved facts and their source
spans and is instructed to answer ONLY from them — so the answer is grounded in
verified structure, not the model's parametric memory.
"""

from __future__ import annotations

from knowlytix.knowledge.llm_backend import LLMBackend
from knowlytix.knowledge.rag.retrieve import RetrievedFact

__all__ = ["Assembler"]

_SYSTEM = (
    "You are given GMS-verified FACTS, each written as 'FACT: head | relation | "
    "tail', that have ALREADY been selected as the relevant grounded knowledge for "
    "the question. Your only job is to VERBALIZE these facts -- not to judge "
    "whether they fully answer the question, and not to add anything not present. "
    "State EVERY provided FACT in full as a short declarative sentence that keeps "
    "the subject (head), the attribute (relation) and the value (tail) together, "
    "so all three appear in your answer. For 'FACT: overdraft | has_fee_amount | "
    "35.0' say 'The overdraft fee is $35.0.'; for 'FACT: manager | has_max_reversal "
    "| 500.0' say 'A manager can reverse up to 500.0.'; for 'FACT: disputes | "
    "has_provisional_credit | issued' say 'Disputes have provisional credit "
    "issued.'. When several facts are given (e.g. a multi-hop chain), state ALL of "
    "them. Keep every value exactly as given; never reply with a bare value, a "
    "relation name without its value, or a yes/no alone. "
    "NEVER say 'cannot answer', 'the facts do not provide ...', or any refusal "
    "while at least one FACT line is present -- simply state the facts. Abstain "
    "ONLY when there are literally no FACT lines. "
    "Do NOT restate the question and do NOT narrate or show reasoning steps, even "
    "if the question says to think step by step."
)


class Assembler:
    def __init__(self, llm: LLMBackend, *, max_tokens: int = 1024):
        self.llm = llm
        self.max_tokens = max_tokens

    def assemble(self, question: str, facts: list[RetrievedFact]) -> str:
        return self.llm.call(system=_SYSTEM,
                            user=self._context(question, facts),
                            max_tokens=self.max_tokens)

    def assemble_passages(self, question: str, passages: list[str]) -> str:
        """Synthesize from raw text passages (dense fallback — unverified)."""
        body = "\n\n".join(f"PASSAGE:\n{p}" for p in passages) or "(none)"
        user = (f"<passages>\n{body}\n</passages>\n\nQuestion: {question}")
        return self.llm.call(system=_SYSTEM, user=user, max_tokens=self.max_tokens)

    @staticmethod
    def _context(question: str, facts: list[RetrievedFact]) -> str:
        lines: list[str] = []
        for f in facts:
            lines.append(f"FACT: {f.head} | {f.relation} | {f.tail}")
            if f.raw:
                loc = f" ({f.location})" if f.location else ""
                lines.append(f"SOURCE{loc}: {f.raw}")
        evidence = "\n".join(lines) if lines else "(no facts retrieved)"
        return (f"<evidence>\n{evidence}\n</evidence>\n\n"
                f"Question: {question}")
