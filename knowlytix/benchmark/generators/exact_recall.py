# SPDX-License-Identifier: Apache-2.0
"""Exact Recall — mines ENM for scalar value questions."""

from knowlytix.benchmark.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS


class ExactRecallGenerator(QuestionGenerator):
    category = "exact_recall"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        for key, entry in list(graph.enm.items()):
            cat, eid = key.type, key.id

            def make_fn(c, e):
                def fn(g):
                    return g.lookup(c, e)
                return fn

            q = GeneratedQuestion(
                qid=f"exact_recall_{idx:03d}",
                category=self.category,
                natural_language=f"What is the exact value of '{eid}' (category: {cat})?",
                graph_answer_fn=make_fn(cat, eid),
                ground_truth=entry.value,
                llm_prompt=(
                    f"Find the exact numeric value for '{eid}' in the "
                    f"'{cat}' section of the report. "
                    f"Copy the number exactly as shown with ALL decimal places. "
                    f"{ANSWER_FORMATS['float']}"
                ),
                answer_type="float",
                metadata={"enm_type": cat, "enm_id": eid},
            )

            if self._validate(graph, q):
                questions.append(q)
                idx += 1

        return questions
