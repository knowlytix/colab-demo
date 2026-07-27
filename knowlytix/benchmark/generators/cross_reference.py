# SPDX-License-Identifier: Apache-2.0
"""Cross-reference — finds entity overlap across relation types."""

from itertools import combinations
from knowlytix.benchmark.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS


class CrossReferenceGenerator(QuestionGenerator):
    category = "cross_reference"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        # Build relation -> head entities index
        relation_heads = {}
        for h, r, t in graph.triples:
            relation_heads.setdefault(r, set()).add(h)

        # Skip very common relations (noise)
        skip = {"in_section", "has_value", "has_effect"}
        relations = [r for r in relation_heads if r not in skip and len(relation_heads[r]) >= 2]

        for r1, r2 in combinations(relations, 2):
            heads1 = relation_heads[r1]
            heads2 = relation_heads[r2]
            overlap = heads1 & heads2

            if len(overlap) < 2 or len(overlap) > 50:
                continue

            def make_fn(rel1, rel2):
                def fn(g):
                    h1 = {h for h, _, _ in g.query_triples(relation=rel1)}
                    h2 = {h for h, _, _ in g.query_triples(relation=rel2)}
                    return h1 & h2
                return fn

            q = GeneratedQuestion(
                qid=f"cross_reference_{idx:03d}",
                category=self.category,
                natural_language=(
                    f"Which entities appear in both the '{r1}' and "
                    f"'{r2}' relationships?"
                ),
                graph_answer_fn=make_fn(r1, r2),
                ground_truth=overlap,
                llm_prompt=(
                    f"Cross-reference the report: find all entities associated "
                    f"with both '{r1}' AND '{r2}'. "
                    f"List every entity that appears in both contexts. "
                    f"{ANSWER_FORMATS['set_str']}"
                ),
                answer_type="set_str",
                metadata={"relation_1": r1, "relation_2": r2},
            )

            if self._validate(graph, q):
                questions.append(q)
                idx += 1

        return questions
