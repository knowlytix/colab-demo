# SPDX-License-Identifier: Apache-2.0
"""Counting generator — aggregation over triple sets and ENM entries."""

from knowlytix.benchmark.eval.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS


class CountingGenerator(QuestionGenerator):
    category = "counting"

    def generate(self, store) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        # Count triples by relation
        relation_counts = {}
        for h, r, t in store.triples:
            relation_counts.setdefault(r, 0)
            relation_counts[r] += 1

        for rel, count in relation_counts.items():
            if count < 2:
                continue

            def make_count_fn(relation):
                def fn(s):
                    return len(s.query_triples(relation=relation))
                return fn

            q = GeneratedQuestion(
                qid=f"counting_{idx:03d}",
                category=self.category,
                natural_language=(
                    f"How many '{rel}' relationships exist in the report?"
                ),
                gms_answer_fn=make_count_fn(rel),
                ground_truth=count,
                llm_prompt=(
                    f"Count ALL instances in the report where an entity has "
                    f"the '{rel}' relationship. Include every occurrence across "
                    f"all tables and sections. "
                    f"{ANSWER_FORMATS['int']}"
                ),
                answer_type="int",
                metadata={"relation": rel, "expected_count": count},
            )

            if self._validate(store, q):
                questions.append(q)
                idx += 1

        # Count by (relation, tail) — filtered counts
        relation_tail_counts = {}
        for h, r, t in store.triples:
            key = (r, t)
            relation_tail_counts.setdefault(key, set()).add(h)

        for (rel, tail), heads in relation_tail_counts.items():
            if len(heads) < 2:
                continue

            def make_filtered_fn(relation, tail_val):
                def fn(s):
                    return len(s.query_triples(relation=relation, tail=tail_val))
                return fn

            q = GeneratedQuestion(
                qid=f"counting_{idx:03d}",
                category=self.category,
                natural_language=(
                    f"How many entities have '{rel}' = '{tail}'?"
                ),
                gms_answer_fn=make_filtered_fn(rel, tail),
                ground_truth=len(heads),
                llm_prompt=(
                    f"In the report, count how many entities are classified as "
                    f"'{tail}' for the '{rel}' category. "
                    f"{ANSWER_FORMATS['int']}"
                ),
                answer_type="int",
                metadata={"relation": rel, "tail": tail},
            )

            if self._validate(store, q):
                questions.append(q)
                idx += 1

        # Count unique head entities per relation
        relation_unique_heads = {}
        for h, r, t in store.triples:
            relation_unique_heads.setdefault(r, set()).add(h)

        for rel, heads in relation_unique_heads.items():
            if len(heads) < 3:
                continue

            def make_unique_fn(relation):
                def fn(s):
                    return len(set(h for h, _, _ in s.query_triples(relation=relation)))
                return fn

            q = GeneratedQuestion(
                qid=f"counting_{idx:03d}",
                category=self.category,
                natural_language=(
                    f"How many unique entities participate in '{rel}' relationships?"
                ),
                gms_answer_fn=make_unique_fn(rel),
                ground_truth=len(heads),
                llm_prompt=(
                    f"Count the number of UNIQUE entities that have a '{rel}' "
                    f"relationship in the report. "
                    f"{ANSWER_FORMATS['int']}"
                ),
                answer_type="int",
                metadata={"relation": rel, "unique_heads": len(heads)},
            )

            if self._validate(store, q):
                questions.append(q)
                idx += 1

        # ENM category counts
        enm_types = {}
        for key in store.enm._store:
            enm_types.setdefault(key.type, 0)
            enm_types[key.type] += 1

        for enm_type, count in enm_types.items():
            if count < 3:
                continue

            def make_enm_count_fn(t):
                def fn(s):
                    return sum(1 for k in s.enm._store if k.type == t)
                return fn

            q = GeneratedQuestion(
                qid=f"counting_{idx:03d}",
                category=self.category,
                natural_language=(
                    f"How many distinct metrics are stored under the "
                    f"'{enm_type}' category?"
                ),
                gms_answer_fn=make_enm_count_fn(enm_type),
                ground_truth=count,
                llm_prompt=(
                    f"Count how many distinct numeric values/metrics appear "
                    f"under the '{enm_type}' category in the report. "
                    f"{ANSWER_FORMATS['int']}"
                ),
                answer_type="int",
                metadata={"enm_type": enm_type, "count": count},
            )

            if self._validate(store, q):
                questions.append(q)
                idx += 1

        return questions
