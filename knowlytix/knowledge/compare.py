# SPDX-License-Identifier: Apache-2.0
"""Three-way comparison: GMS-only vs LLM-only vs GMS+LLM.

Auto-generates questions from the document graph topology using
knowlytix.benchmark generators, then evaluates all three modes.
"""

import json
import time
from dataclasses import dataclass, field

from knowlytix.benchmark.generators import default_generators
from knowlytix.benchmark.scorers import score_answer, PARSERS

from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.query import QueryEngine, QueryResult
from knowlytix.knowledge.store import GMSExpertStore


@dataclass
class ComparisonRow:
    qid: str = ""
    category: str = ""
    question: str = ""
    ground_truth: str = ""
    # GMS-only
    gms_answer: str = ""
    gms_correct: bool = False
    # LLM-only
    llm_answer: str = ""
    llm_correct: bool = False
    llm_plausibility: float = 0.0
    llm_contradictions: int = 0
    llm_holonomy_violations: int = 0
    # GMS+LLM
    gms_llm_answer: str = ""
    gms_llm_correct: bool = False
    gms_llm_plausibility: float = 0.0
    gms_llm_contradictions: int = 0
    gms_llm_learned: bool = False


@dataclass
class ComparisonReport:
    document: str = ""
    total_questions: int = 0
    scores: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)
    rows: list[ComparisonRow] = field(default_factory=list)
    facts_learned: int = 0
    timestamp: str = ""
    model: str = ""


class ComparisonRunner:
    """Run auto-generated questions through all three modes."""

    def __init__(self, store: GMSExpertStore, engine: QueryEngine,
                 config: DocGMSConfig, max_per_category: int = 10,
                 seed: int = 42):
        self.store = store
        self.engine = engine
        self.config = config
        self.max_per_category = max_per_category
        self.seed = seed

    def run(self, modes: list[str] | None = None) -> ComparisonReport:
        """Run comparison across specified modes."""
        if modes is None:
            modes = ["gms_only", "llm_only", "gms_llm"]

        # Disable auto-learning during benchmark to prevent store mutation
        orig_auto_learn = self.config.learn.auto_learn
        self.config.learn.auto_learn = False

        report = ComparisonReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model=self.engine.llm.model_name if self.engine.llm else "",
        )
        report.scores = {m: 0 for m in modes}

        # Generate questions
        generators = default_generators()
        all_qs = []
        print("\nGenerating questions from graph topology...")
        for gen in generators:
            candidates = gen.generate(self.store.doc_graph)
            sampled = gen.sample(candidates, self.max_per_category, self.seed)
            print(f"  {gen.category}: {len(candidates)} candidates -> "
                  f"{len(sampled)} selected")
            all_qs.extend(sampled)

        report.total_questions = len(all_qs)
        print(f"  Total: {len(all_qs)} questions")
        print(f"\nEvaluating across modes: {modes}")

        for i, q in enumerate(all_qs):
            row = ComparisonRow(
                qid=q.qid,
                category=q.category,
                question=q.natural_language,
                ground_truth=str(q.ground_truth),
            )

            tag = f"[{i+1}/{len(all_qs)}] {q.qid}"

            # GMS-only: use graph_answer_fn directly
            if "gms_only" in modes:
                try:
                    gms_ans = q.graph_answer_fn(self.store.doc_graph)
                    row.gms_answer = str(gms_ans)
                    sr = score_answer(gms_ans, q.ground_truth)
                    row.gms_correct = sr.correct
                    if sr.correct:
                        report.scores["gms_only"] += 1
                except Exception:
                    pass

            # LLM-only
            if "llm_only" in modes:
                print(f"  {tag} llm_only...", end=" ", flush=True)
                try:
                    qr = self._eval_llm_mode(q, "llm_only")
                    row.llm_answer = qr.answer
                    parsed = self._parse_answer(qr.answer, q.answer_type)
                    sr = score_answer(parsed, q.ground_truth)
                    row.llm_correct = sr.correct
                    if sr.correct:
                        report.scores["llm_only"] += 1
                    if qr.plausibility:
                        row.llm_plausibility = qr.plausibility.mean_score
                    if qr.verification:
                        row.llm_contradictions = sum(
                            1 for c in qr.verification.contradictions
                            if c.is_contradiction
                        )
                        row.llm_holonomy_violations = sum(
                            1 for h in qr.verification.inconsistencies
                            if not h.is_consistent
                        )
                    print("PASS" if sr.correct else "FAIL")
                except Exception as e:
                    print(f"ERROR: {e}")

            # GMS+LLM: try GMS first, fall back to LLM only if GMS can't answer
            if "gms_llm" in modes:
                print(f"  {tag} gms_llm...", end=" ", flush=True)
                gms_answered = False
                try:
                    gms_ans = q.graph_answer_fn(self.store.doc_graph)
                    if gms_ans is not None and str(gms_ans) != "":
                        row.gms_llm_answer = str(gms_ans)
                        sr = score_answer(gms_ans, q.ground_truth)
                        row.gms_llm_correct = sr.correct
                        gms_answered = True
                        if sr.correct:
                            report.scores["gms_llm"] += 1
                        print(f"GMS-{'PASS' if sr.correct else 'FAIL'}")
                except Exception:
                    pass

                if not gms_answered:
                    # GMS couldn't answer — use LLM with GMS context
                    try:
                        qr = self._eval_llm_mode(q, "gms_llm")
                        row.gms_llm_answer = qr.answer
                        parsed = self._parse_answer(qr.answer, q.answer_type)
                        sr = score_answer(parsed, q.ground_truth)
                        row.gms_llm_correct = sr.correct
                        if sr.correct:
                            report.scores["gms_llm"] += 1
                        if qr.plausibility:
                            row.gms_llm_plausibility = qr.plausibility.mean_score
                        if qr.verification:
                            row.gms_llm_contradictions = sum(
                                1 for c in qr.verification.contradictions
                                if c.is_contradiction
                            )
                        row.gms_llm_learned = qr.learned
                        if qr.learned:
                            report.facts_learned += 1
                        print(f"LLM-{'PASS' if sr.correct else 'FAIL'}")
                    except Exception as e:
                        print(f"ERROR: {e}")

            # Category tracking
            cat = q.category
            if cat not in report.by_category:
                report.by_category[cat] = {m: {"correct": 0, "total": 0}
                                            for m in modes}
            for m in modes:
                report.by_category[cat][m]["total"] += 1
            if row.gms_correct and "gms_only" in modes:
                report.by_category[cat]["gms_only"]["correct"] += 1
            if row.llm_correct and "llm_only" in modes:
                report.by_category[cat]["llm_only"]["correct"] += 1
            if row.gms_llm_correct and "gms_llm" in modes:
                report.by_category[cat]["gms_llm"]["correct"] += 1

            report.rows.append(row)

        # Restore auto-learn setting
        self.config.learn.auto_learn = orig_auto_learn

        return report

    def _eval_llm_mode(self, q, mode: str) -> QueryResult:
        """Evaluate a generated question in LLM mode."""
        # Build prompt with format instructions
        prompt = q.llm_prompt
        return self.engine.query(prompt, mode=mode)

    def _parse_answer(self, raw: str, answer_type: str):
        """Parse LLM answer using knowlytix.benchmark parsers."""
        parser = PARSERS.get(answer_type, PARSERS["str"])
        return parser(raw)

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------

    @staticmethod
    def print_report(report: ComparisonReport):
        modes = list(report.scores.keys())
        n = report.total_questions

        print(f"\n{'='*70}")
        print("DOCGMS COMPARISON REPORT")
        print(f"{'='*70}")
        print(f"  Model:     {report.model}")
        print(f"  Questions: {n}")
        print(f"  Facts learned: {report.facts_learned}")

        # Overall scores
        print("\n  Overall Accuracy:")
        for m in modes:
            s = report.scores[m]
            pct = s / n * 100 if n else 0
            print(f"    {m:<15} {s:>3}/{n}  ({pct:.0f}%)")

        # Per-category
        header = f"  {'Category':<20}"
        for m in modes:
            header += f" {m:>12}"
        print(f"\n{header}")
        print(f"  {'─'*(20 + 13 * len(modes))}")

        for cat in sorted(report.by_category):
            line = f"  {cat:<20}"
            for m in modes:
                c = report.by_category[cat].get(m, {"correct": 0, "total": 0})
                line += f" {c['correct']:>3}/{c['total']:<3}     "
            print(line)

        # LLM failures
        failures = [r for r in report.rows
                     if not r.gms_llm_correct and "gms_llm" in modes]
        if failures:
            print(f"\n  GMS+LLM Failures ({len(failures)}):")
            for f in failures[:10]:
                print(f"    {f.qid}: {f.question[:60]}")
                print(f"      Expected: {f.ground_truth[:50]}")
                print(f"      Got:      {f.gms_llm_answer[:50]}")

    @staticmethod
    def save_report(report: ComparisonReport, path: str):
        data = {
            "document": report.document,
            "total": report.total_questions,
            "scores": report.scores,
            "model": report.model,
            "timestamp": report.timestamp,
            "facts_learned": report.facts_learned,
            "by_category": report.by_category,
            "questions": [
                {
                    "qid": r.qid, "category": r.category,
                    "question": r.question, "ground_truth": r.ground_truth,
                    "gms_answer": r.gms_answer, "gms_correct": r.gms_correct,
                    "llm_answer": r.llm_answer, "llm_correct": r.llm_correct,
                    "llm_plausibility": r.llm_plausibility,
                    "llm_contradictions": r.llm_contradictions,
                    "gms_llm_answer": r.gms_llm_answer,
                    "gms_llm_correct": r.gms_llm_correct,
                    "gms_llm_plausibility": r.gms_llm_plausibility,
                    "gms_llm_contradictions": r.gms_llm_contradictions,
                    "gms_llm_learned": r.gms_llm_learned,
                }
                for r in report.rows
            ],
        }
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                     exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Report saved: {path}")
