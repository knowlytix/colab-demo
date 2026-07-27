# SPDX-License-Identifier: Apache-2.0
"""Contradiction — detects pass/fail inconsistencies in the graph."""

from knowlytix.benchmark.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS


def _extract_contradicting_features(contradictions):
    """Extract unique features from contradiction tuples."""
    features = set()
    for entity, p, f in contradictions:
        for test in [p, f]:
            parts = test.split("_", 1)
            if len(parts) >= 2:
                feat = parts[1].split(":")[0].split("/")[0]
                features.add(feat)
    return features


class ContradictionGenerator(QuestionGenerator):
    category = "contradiction"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        contradictions = graph.find_contradictions()
        if not contradictions:
            return questions

        features_with_contradictions = _extract_contradicting_features(contradictions)

        # Q: How many features have contradictions?
        def make_count_fn():
            def fn(g):
                contras = g.find_contradictions()
                return len(_extract_contradicting_features(contras))
            return fn

        questions.append(GeneratedQuestion(
            qid=f"contradiction_{idx:03d}",
            category=self.category,
            natural_language=(
                "How many features have contradictory performance — "
                "passing in some segments but failing in others?"
            ),
            graph_answer_fn=make_count_fn(),
            ground_truth=len(features_with_contradictions),
            llm_prompt=(
                "Examine all per-feature performance tables. For each feature, "
                "check if it has BOTH passing AND failing segments. "
                "A feature with ALL segments failing is NOT a contradiction. "
                "How many features show this pass/fail contradiction? "
                f"{ANSWER_FORMATS['int']}"
            ),
            answer_type="int",
            metadata={"features": sorted(features_with_contradictions)},
        ))
        idx += 1

        # Q: Which features have contradictions?
        def make_set_fn():
            def fn(g):
                contras = g.find_contradictions()
                return _extract_contradicting_features(contras)
            return fn

        questions.append(GeneratedQuestion(
            qid=f"contradiction_{idx:03d}",
            category=self.category,
            natural_language="Which features have contradictory performance across segments?",
            graph_answer_fn=make_set_fn(),
            ground_truth=features_with_contradictions,
            llm_prompt=(
                "List every feature that has BOTH passing and failing segments. "
                "Exclude features where ALL segments fail. "
                f"{ANSWER_FORMATS['set_str']}"
            ),
            answer_type="set_str",
            metadata={"features": sorted(features_with_contradictions)},
        ))
        idx += 1

        # Per-feature boolean questions
        all_features = set()
        for h, r, t in graph.triples:
            if r in ("passes", "fails"):
                parts = t.split("_", 1)
                if len(parts) >= 2:
                    all_features.add(parts[1].split(":")[0].split("/")[0])

        for feat in sorted(all_features):
            has_contra = feat in features_with_contradictions

            def make_per_fn(feature):
                def fn(g):
                    contras = g.find_contradictions()
                    feats = _extract_contradicting_features(contras)
                    return feature in feats
                return fn

            questions.append(GeneratedQuestion(
                qid=f"contradiction_{idx:03d}",
                category=self.category,
                natural_language=f"Does '{feat}' have contradictory performance across segments?",
                graph_answer_fn=make_per_fn(feat),
                ground_truth=has_contra,
                llm_prompt=(
                    f"Look at the performance table for '{feat}'. Does it have BOTH "
                    f"passing AND failing segments? If all segments fail, answer false. "
                    f"{ANSWER_FORMATS['bool']}"
                ),
                answer_type="bool",
                metadata={"feature": feat},
            ))
            idx += 1

        return questions
