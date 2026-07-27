# SPDX-License-Identifier: Apache-2.0
"""Multi-hop — chained queries combining ENM lookups and triple traversal."""

from knowlytix.benchmark.generators.base import QuestionGenerator, GeneratedQuestion


class MultiHopGenerator(QuestionGenerator):
    category = "multi_hop"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        # Group ENM entries by type
        enm_by_type = {}
        for key, entry in list(graph.enm.items()):
            enm_by_type.setdefault(key.type, []).append((key.id, entry.value))

        # Pattern: argmin/argmax over ENM type, report entity + value
        for enm_type, entries in enm_by_type.items():
            if len(entries) < 3:
                continue

            sorted_entries = sorted(entries, key=lambda x: x[1])

            for which, eid, val in [
                ("lowest", sorted_entries[0][0], sorted_entries[0][1]),
                ("highest", sorted_entries[-1][0], sorted_entries[-1][1]),
            ]:
                def make_fn(e_type, w):
                    def fn(g):
                        best = None
                        best_val = None
                        for k, entry in g.enm.items():
                            if k.type != e_type:
                                continue
                            v = entry.value
                            if best_val is None:
                                best = k.id
                                best_val = v
                            elif w == "lowest" and v < best_val:
                                best = k.id
                                best_val = v
                            elif w == "highest" and v > best_val:
                                best = k.id
                                best_val = v
                        return (best, best_val)
                    return fn

                q = GeneratedQuestion(
                    qid=f"multi_hop_{idx:03d}",
                    category=self.category,
                    natural_language=(
                        f"Which entry has the {which} value in '{enm_type}', "
                        f"and what is that value?"
                    ),
                    graph_answer_fn=make_fn(enm_type, which),
                    ground_truth=(eid, val),
                    llm_prompt=(
                        f"In the '{enm_type}' data, find the entry with the "
                        f"{which} numeric value. Report its name and exact value.\n"
                        f"Format:\n"
                        f"ENTITY: <name>\n"
                        f"VALUE: <exact number>"
                    ),
                    answer_type="str",
                    metadata={"enm_type": enm_type, "which": which},
                )

                if self._validate(graph, q):
                    questions.append(q)
                    idx += 1

        # Pattern: argmin/max in type1, then lookup in type2
        for type1, entries1 in enm_by_type.items():
            if len(entries1) < 3:
                continue
            sorted1 = sorted(entries1, key=lambda x: x[1])

            for type2, entries2 in enm_by_type.items():
                if type1 == type2 or len(entries2) < 2:
                    continue

                for which, eid1, val1 in [
                    ("lowest", sorted1[0][0], sorted1[0][1]),
                    ("highest", sorted1[-1][0], sorted1[-1][1]),
                ]:
                    base1 = eid1.split("/")[0]
                    matching2 = [(e, v) for e, v in entries2
                                 if e == base1 or e.startswith(base1 + "/")]

                    if not matching2:
                        continue

                    def make_chain_fn(t1, t2, w):
                        def fn(g):
                            best = None
                            best_val = None
                            for k, entry in g.enm.items():
                                if k.type != t1:
                                    continue
                                v = entry.value
                                if best_val is None or \
                                   (w == "lowest" and v < best_val) or \
                                   (w == "highest" and v > best_val):
                                    best = k.id
                                    best_val = v
                            base = best.split("/")[0] if best else None
                            if not base:
                                return None
                            results = {}
                            for k, entry in g.enm.items():
                                if k.type == t2 and (k.id == base or k.id.startswith(base + "/")):
                                    results[k.id] = entry.value
                            return (base, best_val, results) if results else None
                        return fn

                    gt = make_chain_fn(type1, type2, which)(graph)
                    if gt is not None:
                        q = GeneratedQuestion(
                            qid=f"multi_hop_{idx:03d}",
                            category=self.category,
                            natural_language=(
                                f"Find the entity with the {which} '{type1}' value. "
                                f"Then look up that entity's '{type2}' data."
                            ),
                            graph_answer_fn=make_chain_fn(type1, type2, which),
                            ground_truth=gt,
                            llm_prompt=(
                                f"Step 1: In '{type1}', find the entry with the {which} value.\n"
                                f"Step 2: Find that same entity's data in '{type2}'.\n"
                                f"Report the entity name and both values exactly.\n"
                                f"Format:\n"
                                f"ENTITY: <name>\n"
                                f"{type1.upper()}_VALUE: <number>\n"
                                f"{type2.upper()}_VALUE: <number>"
                            ),
                            answer_type="str",
                            metadata={"type1": type1, "type2": type2, "which": which},
                        )
                        questions.append(q)
                        idx += 1

        return questions
