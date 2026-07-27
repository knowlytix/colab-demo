# SPDX-License-Identifier: Apache-2.0
"""Multi-hop generator — builds chain queries from ENM + triple adjacency."""

from knowlytix.benchmark.eval.generators.base import QuestionGenerator, GeneratedQuestion


class MultiHopGenerator(QuestionGenerator):
    category = "multi_hop"

    def generate(self, store) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        # Pattern 1: argmin/argmax over an ENM category, then look up a related triple
        # Group ENM entries by type
        enm_by_type = {}
        for key, entry in list(store.enm._store.items()):
            if entry.value.size != 1:
                continue
            enm_by_type.setdefault(key.type, []).append(
                (key.id, float(entry.value.item()))
            )

        for enm_type, entries in enm_by_type.items():
            if len(entries) < 3:
                continue

            # Find min and max
            sorted_entries = sorted(entries, key=lambda x: x[1])
            for which, eid, val in [
                ("lowest", sorted_entries[0][0], sorted_entries[0][1]),
                ("highest", sorted_entries[-1][0], sorted_entries[-1][1]),
            ]:
                # Check if this entity participates in any triples as head
                entity_triples = store.query_triples(head=eid)
                if not entity_triples:
                    # Try without the path prefix (e.g. "cluster_0/AUC" -> "cluster_0")
                    base_entity = eid.split("/")[0]
                    entity_triples = store.query_triples(head=base_entity)

                if entity_triples:
                    relations = set(r for _, r, _ in entity_triples)
                    for rel in relations:
                        tails = [t for _, _, t in entity_triples if _ == rel
                                 for _, r2, t in entity_triples if r2 == rel]
                        # Re-query properly
                        matching = [(h, r, t) for h, r, t in entity_triples if r == rel]
                        tail_set = set(t for _, _, t in matching)

                        def make_fn(e_type, w, relation):
                            def fn(s):
                                # Find argmin/argmax
                                best = None
                                best_val = None
                                for k, entry in s.enm._store.items():
                                    if k.type != e_type or entry.value.size != 1:
                                        continue
                                    v = float(entry.value.item())
                                    if best_val is None:
                                        best = k.id
                                        best_val = v
                                    elif w == "lowest" and v < best_val:
                                        best = k.id
                                        best_val = v
                                    elif w == "highest" and v > best_val:
                                        best = k.id
                                        best_val = v
                                # Follow relation from that entity
                                base = best.split("/")[0] if best else None
                                if base:
                                    triples = s.query_triples(head=base, relation=relation)
                                    return (base, best_val, set(t for _, _, t in triples))
                                return None
                            return fn

                        ground_truth_base = eid.split("/")[0]

                        q = GeneratedQuestion(
                            qid=f"multi_hop_{idx:03d}",
                            category=self.category,
                            natural_language=(
                                f"Find the entity with the {which} value in "
                                f"'{enm_type}', then list its '{rel}' relationships."
                            ),
                            gms_answer_fn=make_fn(enm_type, which, rel),
                            ground_truth=(ground_truth_base, val, tail_set),
                            llm_prompt=(
                                f"Step 1: In the '{enm_type}' data, find the entry with "
                                f"the {which} value. What is it and its exact value?\n"
                                f"Step 2: For that entity, look up its '{rel}' relationships.\n"
                                f"Report: ENTITY: <name>, VALUE: <number>, "
                                f"RELATED: <comma-separated list>"
                            ),
                            answer_type="str",  # Complex, needs custom parsing
                            metadata={
                                "enm_type": enm_type, "which": which,
                                "relation": rel,
                            },
                        )
                        questions.append(q)
                        idx += 1

        # Pattern 2: Two-step ENM lookups (argmin in one type -> lookup in another)
        for type1, entries1 in enm_by_type.items():
            if len(entries1) < 3:
                continue
            for type2, entries2 in enm_by_type.items():
                if type1 == type2 or len(entries2) < 2:
                    continue

                # Check if any entity from type1's argmin/argmax appears in type2
                sorted1 = sorted(entries1, key=lambda x: x[1])
                for which, eid1, val1 in [
                    ("lowest", sorted1[0][0], sorted1[0][1]),
                    ("highest", sorted1[-1][0], sorted1[-1][1]),
                ]:
                    base1 = eid1.split("/")[0]
                    # Check if base1 appears as a key prefix in type2
                    matching2 = [(e, v) for e, v in entries2
                                 if e.startswith(base1) or base1 in e]
                    if matching2:
                        def make_chain_fn(t1, t2, w):
                            def fn(s):
                                best = None
                                best_val = None
                                for k, entry in s.enm._store.items():
                                    if k.type != t1 or entry.value.size != 1:
                                        continue
                                    v = float(entry.value.item())
                                    if best_val is None:
                                        best = k.id
                                        best_val = v
                                    elif w == "lowest" and v < best_val:
                                        best = k.id
                                        best_val = v
                                    elif w == "highest" and v > best_val:
                                        best = k.id
                                        best_val = v
                                base = best.split("/")[0] if best else None
                                # Lookup in type2
                                results = {}
                                for k, entry in s.enm._store.items():
                                    if k.type == t2 and entry.value.size == 1:
                                        if base and (k.id.startswith(base) or base in k.id):
                                            results[k.id] = float(entry.value.item())
                                return (base, best_val, results) if results else None
                            return fn

                        q = GeneratedQuestion(
                            qid=f"multi_hop_{idx:03d}",
                            category=self.category,
                            natural_language=(
                                f"Find the entity with the {which} '{type1}' value. "
                                f"Then look up that entity's '{type2}' metrics."
                            ),
                            gms_answer_fn=make_chain_fn(type1, type2, which),
                            ground_truth=None,  # Computed at validation
                            llm_prompt=(
                                f"Step 1: In '{type1}', find the entry with the "
                                f"{which} value.\n"
                                f"Step 2: Find that entity's data in '{type2}'.\n"
                                f"Report the entity name, its {type1} value, and "
                                f"the {type2} values. Use exact numbers.\n"
                                f"Format: ENTITY: <name>, {type1.upper()}: <number>"
                            ),
                            answer_type="str",
                            metadata={"type1": type1, "type2": type2, "which": which},
                        )
                        # Compute ground truth
                        gt = q.gms_answer_fn(store)
                        if gt is not None:
                            q.ground_truth = gt
                            questions.append(q)
                            idx += 1

        return questions
