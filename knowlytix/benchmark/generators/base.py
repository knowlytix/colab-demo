# SPDX-License-Identifier: Apache-2.0
"""Base classes for auto-generated benchmark questions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
import random

from knowlytix.benchmark.graph import DocumentGraph


@dataclass
class GeneratedQuestion:
    """A single auto-generated benchmark question."""
    qid: str
    category: str
    natural_language: str
    graph_answer_fn: Callable[[DocumentGraph], Any]
    ground_truth: Any
    llm_prompt: str
    answer_type: str  # "float", "int", "bool", "set_str", "str"
    metadata: dict = field(default_factory=dict)


ANSWER_FORMATS = {
    "float": "Report the EXACT value with ALL decimal places. Format: ANSWER: <number>",
    "int": "Format: ANSWER: <integer>",
    "bool": "Format: ANSWER: true or false",
    "set_str": "List ALL items, comma-separated. Format: ANSWER: item1, item2, ...",
    "str": "Format: ANSWER: <exact string>",
}


class QuestionGenerator(ABC):
    category: str = ""

    @abstractmethod
    def generate(self, graph: DocumentGraph) -> list[GeneratedQuestion]:
        ...

    def sample(self, candidates: list[GeneratedQuestion],
               max_questions: int, seed: int = 42) -> list[GeneratedQuestion]:
        if len(candidates) <= max_questions:
            return candidates
        rng = random.Random(seed)
        return rng.sample(candidates, max_questions)

    def _validate(self, graph: DocumentGraph, q: GeneratedQuestion) -> bool:
        try:
            answer = q.graph_answer_fn(graph)
            if answer is None:
                return False
            if isinstance(q.ground_truth, float):
                return abs(answer - q.ground_truth) < 1e-9
            return answer == q.ground_truth
        except Exception:
            return False
