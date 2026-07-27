# SPDX-License-Identifier: Apache-2.0
"""Threshold — mines numeric values against phase encoder bounds."""

from knowlytix.benchmark.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS

# Default thresholds per encoder type
DEFAULT_THRESHOLDS = {
    "AUC": [(0.5, "ge", "better than random (AUC >= 0.5)"),
            (0.5, "lt", "worse than random (AUC < 0.5)")],
    "accuracy": [(0.5, "ge", "better than random")],
    "AIR": [(0.8, "ge", "the regulatory 4/5ths rule (AIR >= 0.80)")],
    "importance": [(0.1, "ge", "significance (>= 0.1)")],
    "divergence": [(0.3, "ge", "high divergence (>= 0.3)")],
    "loss": [(0.5, "lt", "acceptable loss (< 0.5)")],
    "rate": [(0.05, "lt", "below 5% threshold")],
    "capital_ratio": [(4.5, "ge", "minimum CET1 (>= 4.5%)")],
    "pvalue": [(0.05, "lt", "statistically significant (p < 0.05)")],
}

# Map ENM categories to encoders by keyword matching
CATEGORY_KEYWORDS = {
    "auc": "AUC",
    "accuracy": "accuracy",
    "air": "AIR",
    "fairness": "AIR",
    "importance": "importance",
    "divergence": "divergence",
    "js": "divergence",
    "loss": "loss",
    "brier": "loss",
    "rate": "rate",
    "capital": "capital_ratio",
    "cet1": "capital_ratio",
    "pvalue": "pvalue",
}


def _match_encoder(category: str, entity_id: str) -> str | None:
    """Try to match an ENM entry to a phase encoder."""
    text = f"{category} {entity_id}".lower()
    for keyword, encoder in CATEGORY_KEYWORDS.items():
        if keyword in text:
            return encoder
    return None


class ThresholdGenerator(QuestionGenerator):
    category = "threshold"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        for key, entry in list(graph.enm.items()):
            cat, eid = key.type, key.id
            value = entry.value
            encoder_name = _match_encoder(cat, eid)

            if encoder_name is None or encoder_name not in graph.phase_encoders:
                continue

            thresholds = DEFAULT_THRESHOLDS.get(encoder_name, [])
            for thresh, op, description in thresholds:
                satisfied, margin = graph.check_threshold(encoder_name, value, thresh, op)
                if satisfied is None:
                    continue

                op_word = "at least" if op == "ge" else ("greater than" if op == "gt"
                          else ("at most" if op == "le" else "less than"))

                def make_fn(c, e, enc, th, o):
                    def fn(g):
                        v = g.lookup(c, e)
                        if v is None:
                            return None
                        ok, _ = g.check_threshold(enc, v, th, o)
                        return ok
                    return fn

                q = GeneratedQuestion(
                    qid=f"threshold_{idx:03d}",
                    category=self.category,
                    natural_language=(
                        f"Does '{eid}' ({cat}) meet {description}? "
                        f"Is its value {op_word} {thresh}?"
                    ),
                    graph_answer_fn=make_fn(cat, eid, encoder_name, thresh, op),
                    ground_truth=satisfied,
                    llm_prompt=(
                        f"Find the value of '{eid}' in the '{cat}' section. "
                        f"Is that value {op_word} {thresh}? "
                        f"{ANSWER_FORMATS['bool']}"
                    ),
                    answer_type="bool",
                    metadata={"encoder": encoder_name, "threshold": thresh, "op": op},
                )

                if self._validate(graph, q):
                    questions.append(q)
                    idx += 1

        return questions
