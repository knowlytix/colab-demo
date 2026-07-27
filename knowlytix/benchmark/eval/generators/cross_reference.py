# SPDX-License-Identifier: Apache-2.0
"""Cross-reference generator — finds entity overlap across relation types."""

from itertools import combinations
from knowlytix.benchmark.eval.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS


class CrossReferenceGenerator(QuestionGenerator):
    category = "cross_reference"

    def generate(self, store) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        # Build relation index: relation -> set of head entities
        relation_heads = {}
        for h, r, t in store.triples:
            relation_heads.setdefault(r, set()).add(h)

        # Find relation pairs with overlapping head entity sets
        relations = list(relation_heads.keys())
        for r1, r2 in combinations(relations, 2):
            heads1 = relation_heads[r1]
            heads2 = relation_heads[r2]
            overlap = heads1 & heads2

            if len(overlap) < 2:
                continue

            def make_fn(rel1, rel2):
                def fn(s):
                    h1 = {h for h, _, _ in s.query_triples(relation=rel1)}
                    h2 = {h for h, _, _ in s.query_triples(relation=rel2)}
                    return h1 & h2
                return fn

            q = GeneratedQuestion(
                qid=f"cross_reference_{idx:03d}",
                category=self.category,
                natural_language=(
                    f"Which entities appear in both the '{r1}' and "
                    f"'{r2}' relationships?"
                ),
                gms_answer_fn=make_fn(r1, r2),
                ground_truth=overlap,
                llm_prompt=(
                    f"Cross-reference the report to find entities that are "
                    f"associated with both '{r1}' AND '{r2}'. "
                    f"List every entity that appears in both contexts. "
                    f"{ANSWER_FORMATS['set_str']}"
                ),
                answer_type="set_str",
                metadata={"relation_1": r1, "relation_2": r2},
            )

            if self._validate(store, q):
                questions.append(q)
                idx += 1

        # Resolution comparison: find original vs resolved ENM type pairs
        enm_types = set()
        for key in store.enm._store:
            enm_types.add(key.type)

        for t in list(enm_types):
            resolved_t = f"resolved_{t}"
            if resolved_t not in enm_types:
                continue

            # Find shared entity IDs between original and resolved
            orig_ids = {}
            resolved_ids = {}
            for key, entry in list(store.enm._store.items()):
                if entry.value.size != 1:
                    continue
                if key.type == t:
                    orig_ids[key.id] = float(entry.value.item())
                elif key.type == resolved_t:
                    resolved_ids[key.id] = float(entry.value.item())

            shared = set(orig_ids.keys()) & set(resolved_ids.keys())
            for eid in shared:
                orig_val = orig_ids[eid]
                res_val = resolved_ids[eid]
                improved = res_val < orig_val  # For most metrics, lower is "improved"

                def make_comparison_fn(orig_type, res_type, entity_id):
                    def fn(s):
                        o = s.query_exact(orig_type, entity_id)
                        r = s.query_exact(res_type, entity_id)
                        if o is not None and r is not None:
                            ov = float(o.item())
                            rv = float(r.item())
                            return rv < ov
                        return None
                    return fn

                q = GeneratedQuestion(
                    qid=f"cross_reference_{idx:03d}",
                    category=self.category,
                    natural_language=(
                        f"Did the resolved model reduce the '{eid}' "
                        f"metric compared to the original?"
                    ),
                    gms_answer_fn=make_comparison_fn(t, resolved_t, eid),
                    ground_truth=improved,
                    llm_prompt=(
                        f"Compare the '{eid}' metric between the original and "
                        f"resolved model sections. Did the resolved value decrease "
                        f"compared to the original? "
                        f"{ANSWER_FORMATS['bool']}"
                    ),
                    answer_type="bool",
                    metadata={
                        "original_type": t, "resolved_type": resolved_t,
                        "entity_id": eid,
                    },
                )

                if self._validate(store, q):
                    questions.append(q)
                    idx += 1

        return questions
