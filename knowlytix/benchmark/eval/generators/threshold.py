# SPDX-License-Identifier: Apache-2.0
"""Threshold generator — mines numeric ENM values against phase encoder bounds."""

from knowlytix.benchmark.eval.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS


# Default thresholds per phase encoder
DEFAULT_THRESHOLDS = {
    "AUC": [(0.5, "ge", "better than random"), (0.5, "lt", "worse than random")],
    "AIR": [(0.8, "ge", "the regulatory 4/5ths rule")],
    "importance": [(0.1, "ge", "significance (>0.1)")],
    "JS": [(0.3, "ge", "high divergence (>0.3)")],
}

# Map ENM categories to phase encoders
ENM_TO_ENCODER = {
    "rf_cluster": "AUC",
    "accuracy": "AUC",
    "slice_accuracy": "AUC",
    "weak_slice": "AUC",
    "fairness": "AIR",
    "fanova_fi": "importance",
    "residual_fi": "importance",
    "js_divergence": "JS",
}


class ThresholdGenerator(QuestionGenerator):
    category = "threshold"

    def generate(self, store) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        for key, entry in list(store.enm._store.items()):
            if entry.value.size != 1:
                continue

            value = float(entry.value.item())
            cat = key.type
            eid = key.id
            encoder_name = ENM_TO_ENCODER.get(cat)
            if encoder_name is None:
                continue

            thresholds = DEFAULT_THRESHOLDS.get(encoder_name, [])
            for thresh, op, description in thresholds:
                satisfied, margin = store.query_phase(encoder_name, value, thresh, op)
                if satisfied is None:
                    continue

                op_word = "at least" if op == "ge" else "less than"

                def make_fn(c, e, enc, th, o):
                    def fn(s):
                        val = s.query_exact(c, e)
                        if val is None:
                            return None
                        v = float(val.item())
                        ok, _ = s.query_phase(enc, v, th, o)
                        return ok
                    return fn

                q = GeneratedQuestion(
                    qid=f"threshold_{idx:03d}",
                    category=self.category,
                    natural_language=(
                        f"Does '{eid}' ({cat}) meet {description}? "
                        f"Is its value {op_word} {thresh}?"
                    ),
                    gms_answer_fn=make_fn(cat, eid, encoder_name, thresh, op),
                    ground_truth=satisfied,
                    llm_prompt=(
                        f"Find the value of '{eid}' in the '{cat}' section. "
                        f"Is that value {op_word} {thresh}? "
                        f"{ANSWER_FORMATS['bool']}"
                    ),
                    answer_type="bool",
                    metadata={
                        "enm_type": cat, "enm_id": eid,
                        "encoder": encoder_name, "threshold": thresh, "op": op,
                    },
                )

                if self._validate(store, q):
                    questions.append(q)
                    idx += 1

        return questions
