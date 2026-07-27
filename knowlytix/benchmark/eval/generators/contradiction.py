# SPDX-License-Identifier: Apache-2.0
"""Contradiction generator — detects pass/fail inconsistencies in the graph."""

from knowlytix.benchmark.eval.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS


class ContradictionGenerator(QuestionGenerator):
    category = "contradiction"

    def generate(self, store) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        contradictions = store.find_contradictions()
        if not contradictions:
            return questions

        # Group contradictions by feature
        features_with_contradictions = set()
        feature_details = {}
        for entity, pass_test, fail_test in contradictions:
            for test in [pass_test, fail_test]:
                parts = test.split("_", 1)
                if len(parts) >= 2:
                    feat = parts[1].split(":")[0]
                    features_with_contradictions.add(feat)
                    feature_details.setdefault(feat, []).append(
                        (entity, pass_test, fail_test)
                    )

        # Q: How many features have contradictions?
        count = len(features_with_contradictions)

        def make_count_fn():
            def fn(s):
                contras = s.find_contradictions()
                feats = set()
                for _, p, f in contras:
                    for test in [p, f]:
                        parts = test.split("_", 1)
                        if len(parts) >= 2:
                            feats.add(parts[1].split(":")[0])
                return len(feats)
            return fn

        questions.append(GeneratedQuestion(
            qid=f"contradiction_{idx:03d}",
            category=self.category,
            natural_language=(
                "How many features have contradictory performance — "
                "passing in some segments but failing in others?"
            ),
            gms_answer_fn=make_count_fn(),
            ground_truth=count,
            llm_prompt=(
                "Examine all per-feature slicing/performance tables in the report. "
                "For each feature, check if it has BOTH passing segments AND failing "
                "segments. A feature with ALL segments failing is NOT a contradiction. "
                "How many features show this pass/fail contradiction? "
                f"{ANSWER_FORMATS['int']}"
            ),
            answer_type="int",
            metadata={"features": list(features_with_contradictions)},
        ))
        idx += 1

        # Q: Which features have contradictions?
        def make_set_fn():
            def fn(s):
                contras = s.find_contradictions()
                feats = set()
                for _, p, f in contras:
                    for test in [p, f]:
                        parts = test.split("_", 1)
                        if len(parts) >= 2:
                            feats.add(parts[1].split(":")[0])
                return feats
            return fn

        questions.append(GeneratedQuestion(
            qid=f"contradiction_{idx:03d}",
            category=self.category,
            natural_language=(
                "Which features have contradictory performance across segments?"
            ),
            gms_answer_fn=make_set_fn(),
            ground_truth=features_with_contradictions,
            llm_prompt=(
                "Examine all per-feature slicing/performance tables. List every "
                "feature that has BOTH passing and failing segments. Exclude "
                "features where ALL segments fail (no contradiction). "
                f"{ANSWER_FORMATS['set_str']}"
            ),
            answer_type="set_str",
            metadata={"features": list(features_with_contradictions)},
        ))
        idx += 1

        # Per-feature: does feature X have contradictions?
        all_features = set()
        for h, r, t in store.triples:
            if r in ("passes", "fails"):
                parts = t.split("_", 1)
                if len(parts) >= 2:
                    all_features.add(parts[1].split(":")[0])

        for feat in sorted(all_features):
            has_contradiction = feat in features_with_contradictions

            def make_per_feature_fn(feature):
                def fn(s):
                    contras = s.find_contradictions()
                    feats = set()
                    for _, p, f in contras:
                        for test in [p, f]:
                            parts = test.split("_", 1)
                            if len(parts) >= 2:
                                feats.add(parts[1].split(":")[0])
                    return feature in feats
                return fn

            questions.append(GeneratedQuestion(
                qid=f"contradiction_{idx:03d}",
                category=self.category,
                natural_language=(
                    f"Does the feature '{feat}' have contradictory performance "
                    f"(passing in some segments, failing in others)?"
                ),
                gms_answer_fn=make_per_feature_fn(feat),
                ground_truth=has_contradiction,
                llm_prompt=(
                    f"Look at the slicing table for '{feat}'. Does it have BOTH "
                    f"passing segments AND failing segments? If all segments fail, "
                    f"answer false. "
                    f"{ANSWER_FORMATS['bool']}"
                ),
                answer_type="bool",
                metadata={"feature": feat},
            ))
            idx += 1

        return questions
