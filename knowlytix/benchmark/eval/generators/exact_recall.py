# SPDX-License-Identifier: Apache-2.0
"""Exact Recall generator — mines ENM keys for scalar value questions."""

from knowlytix.benchmark.eval.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS


class ExactRecallGenerator(QuestionGenerator):
    category = "exact_recall"

    def generate(self, store) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        for key, entry in list(store.enm._store.items()):
            if entry.value.size != 1:
                continue

            value = float(entry.value.item())
            cat = key.type
            eid = key.id

            def make_fn(c, e):
                def fn(s):
                    val = s.query_exact(c, e)
                    return float(val.item()) if val is not None else None
                return fn

            q = GeneratedQuestion(
                qid=f"exact_recall_{idx:03d}",
                category=self.category,
                natural_language=(
                    f"What is the exact value of '{eid}' "
                    f"(category: {cat})?"
                ),
                gms_answer_fn=make_fn(cat, eid),
                ground_truth=value,
                llm_prompt=(
                    f"Find the exact numeric value for '{eid}' in the "
                    f"'{cat}' category/section of the report. "
                    f"Copy the number exactly as shown with ALL decimal places. "
                    f"{ANSWER_FORMATS['float']}"
                ),
                answer_type="float",
                metadata={"enm_type": cat, "enm_id": eid},
            )

            if self._validate(store, q):
                questions.append(q)
                idx += 1

        return questions
