# SPDX-License-Identifier: Apache-2.0
"""Base classes for auto-generated benchmark questions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
import random


@dataclass
class GeneratedQuestion:
    """A single auto-generated benchmark question."""
    qid: str                                    # e.g. "exact_recall_007"
    category: str                               # question category
    natural_language: str                        # human-readable question
    gms_answer_fn: Callable                     # function(store) -> answer
    ground_truth: Any                           # expected answer
    llm_prompt: str                             # prompt for the LLM
    answer_type: str                            # "float", "int", "bool", "set_str", "str"
    metadata: dict = field(default_factory=dict)  # source keys/triples


# LLM answer format instructions per type
ANSWER_FORMATS = {
    "float": "Report the EXACT value with ALL decimal places. Format: ANSWER: <number>",
    "int": "Format: ANSWER: <integer>",
    "bool": "Format: ANSWER: true or false",
    "set_str": "List ALL items, comma-separated. Format: ANSWER: item1, item2, ...",
    "str": "Format: ANSWER: <exact string>",
}


class QuestionGenerator(ABC):
    """Base class for question generators that mine a GMSStore."""

    category: str = ""

    @abstractmethod
    def generate(self, store) -> list[GeneratedQuestion]:
        """Mine the store and produce candidate questions."""
        ...

    def sample(self, candidates: list[GeneratedQuestion],
               max_questions: int, seed: int = 42) -> list[GeneratedQuestion]:
        """Deterministic sampling from candidate pool."""
        if len(candidates) <= max_questions:
            return candidates
        rng = random.Random(seed)
        return rng.sample(candidates, max_questions)

    def _validate(self, store, question: GeneratedQuestion) -> bool:
        """Verify the GMS answer function returns the expected ground truth."""
        try:
            answer = question.gms_answer_fn(store)
            if answer is None:
                return False
            if isinstance(question.ground_truth, float):
                return abs(answer - question.ground_truth) < 1e-9
            return answer == question.ground_truth
        except Exception:
            return False
