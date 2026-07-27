# SPDX-License-Identifier: Apache-2.0
"""
Benchmark Orchestrator — Auto-generated GMS vs LLM evaluation.

Usage:
    python -m knowlytix.benchmark.eval.benchmark demo/weakness_report.md
"""

import json
import time
from dataclasses import dataclass, field

from knowlytix.benchmark.eval.generators import default_generators, GeneratedQuestion
from knowlytix.benchmark.eval.scorers import score_answer, PARSERS
from knowlytix.benchmark.eval.llm_caller import call_llm, create_client


@dataclass
class QuestionResult:
    qid: str
    category: str
    question: str
    ground_truth: str
    gms_answer: str
    gms_correct: bool
    gms_detail: str
    llm_answer: str = None
    llm_correct: bool = False
    llm_detail: str = ""
    llm_raw: str = ""


@dataclass
class BenchmarkResult:
    total_questions: int = 0
    gms_score: int = 0
    llm_score: int = 0
    by_category: dict = field(default_factory=dict)
    questions: list = field(default_factory=list)
    model: str = ""
    timestamp: str = ""


class Benchmark:
    def __init__(self, store, markdown_path,
                 generators=None, max_per_category=10, seed=42):
        self.store = store
        self.markdown_path = markdown_path
        with open(markdown_path) as f:
            self.markdown = f.read()
        self.generators = generators or default_generators()
        self.max_per_category = max_per_category
        self.seed = seed

    def generate_questions(self) -> list[GeneratedQuestion]:
        """Run all generators, sample, return question bank."""
        all_qs = []
        for gen in self.generators:
            candidates = gen.generate(self.store)
            sampled = gen.sample(candidates, self.max_per_category, self.seed)
            print(f"  {gen.category}: {len(candidates)} candidates -> {len(sampled)} selected")
            all_qs.extend(sampled)
        return all_qs

    def run(self, llm_client=None, model="claude-sonnet-4-20250514") -> BenchmarkResult:
        """Generate questions, run GMS + LLM, score, return results."""
        print("Generating questions from graph structure...")
        questions = self.generate_questions()
        print(f"Total questions: {len(questions)}\n")

        result = BenchmarkResult(
            total_questions=len(questions),
            model=model,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        for i, q in enumerate(questions):
            # GMS answer
            gms_answer = q.gms_answer_fn(self.store)
            gms_score = score_answer(gms_answer, q.ground_truth)
            if gms_score.correct:
                result.gms_score += 1

            qr = QuestionResult(
                qid=q.qid,
                category=q.category,
                question=q.natural_language,
                ground_truth=str(q.ground_truth),
                gms_answer=str(gms_answer),
                gms_correct=gms_score.correct,
                gms_detail=gms_score.detail,
            )

            # LLM answer
            if llm_client:
                try:
                    print(f"  [{i+1}/{len(questions)}] {q.qid}...", end=" ", flush=True)
                    raw = call_llm(llm_client, self.markdown, q.llm_prompt, model)
                    parser = PARSERS.get(q.answer_type, PARSERS["str"])
                    llm_answer = parser(raw)
                    llm_score = score_answer(llm_answer, q.ground_truth)
                    if llm_score.correct:
                        result.llm_score += 1
                    qr.llm_answer = str(llm_answer)
                    qr.llm_correct = llm_score.correct
                    qr.llm_detail = llm_score.detail
                    qr.llm_raw = raw[:500]
                    status = "PASS" if llm_score.correct else "FAIL"
                    print(f"{status} (got: {str(llm_answer)[:60]})")
                except Exception as e:
                    qr.llm_answer = f"ERROR: {e}"
                    qr.llm_detail = str(e)
                    print(f"ERROR: {e}")

            # Track by category
            cat = q.category
            if cat not in result.by_category:
                result.by_category[cat] = {"gms": 0, "llm": 0, "total": 0}
            result.by_category[cat]["total"] += 1
            if gms_score.correct:
                result.by_category[cat]["gms"] += 1
            if qr.llm_correct:
                result.by_category[cat]["llm"] += 1

            result.questions.append(qr)

        return result

    def print_results(self, result: BenchmarkResult):
        """Print a summary report."""
        n = result.total_questions
        print(f"\n{'=' * 70}")
        print(f"BENCHMARK RESULTS — {n} Auto-Generated Questions")
        print(f"{'=' * 70}")
        print(f"  Model:  {result.model}")
        print(f"  Report: {self.markdown_path}")
        print(f"  Time:   {result.timestamp}")
        print(f"\n  GMS:            {result.gms_score}/{n} ({result.gms_score/n*100:.0f}%)")
        print(f"  LLM+Markdown:   {result.llm_score}/{n} ({result.llm_score/n*100:.0f}%)")

        print(f"\n  {'Category':<20} {'GMS':>10} {'LLM':>10}")
        print(f"  {'─' * 45}")
        for cat, counts in sorted(result.by_category.items()):
            t = counts['total']
            print(f"  {cat:<20} {counts['gms']:>3}/{t:<3}      {counts['llm']:>3}/{t:<3}")

        # Show failures
        failures = [q for q in result.questions if not q.llm_correct]
        if failures:
            print(f"\n  LLM Failures ({len(failures)}):")
            for f in failures[:20]:
                print(f"    {f.qid}: {f.question[:70]}")
                print(f"      Expected: {f.ground_truth[:60]}")
                print(f"      Got:      {f.llm_answer[:60] if f.llm_answer else 'None'}")
                print(f"      Detail:   {f.llm_detail[:60]}")

    def save_results(self, result: BenchmarkResult, path: str):
        """Save results to JSON."""
        data = {
            "total": result.total_questions,
            "gms_score": result.gms_score,
            "llm_score": result.llm_score,
            "model": result.model,
            "timestamp": result.timestamp,
            "by_category": result.by_category,
            "questions": [
                {
                    "qid": q.qid,
                    "category": q.category,
                    "question": q.question,
                    "ground_truth": q.ground_truth,
                    "gms_answer": q.gms_answer,
                    "gms_correct": q.gms_correct,
                    "llm_answer": q.llm_answer,
                    "llm_correct": q.llm_correct,
                    "llm_detail": q.llm_detail,
                }
                for q in result.questions
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Results saved to: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GMS vs LLM Benchmark")
    parser.add_argument("markdown", help="Path to markdown report")
    parser.add_argument("--max-per-category", type=int, default=10)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Import ingest (assumes demo/ingest_to_gms.py exists for this report)
    from demo.ingest_to_gms import ingest

    print("Loading GMS store...\n")
    store = ingest()
    print()

    bench = Benchmark(
        store, args.markdown,
        max_per_category=args.max_per_category,
        seed=args.seed,
    )

    client = None if args.no_llm else create_client()
    result = bench.run(llm_client=client, model=args.model)
    bench.print_results(result)

    if args.output:
        bench.save_results(result, args.output)


if __name__ == "__main__":
    main()
