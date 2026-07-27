# SPDX-License-Identifier: Apache-2.0
"""
FinStructBench — Benchmark Runner.

Usage:
    source ~/jupyterlab/ga_verify/venv/bin/activate
    python -m knowlytix.benchmark.benchmark instances/model_validation.md --max-per-category 10
"""

import json
import time
import os
from dataclasses import dataclass, field

from knowlytix.benchmark.ingest import ingest_markdown
from knowlytix.benchmark.generators import default_generators
from knowlytix.benchmark.scorers import score_answer, PARSERS
from knowlytix.benchmark.llm_caller import call_llm, create_client


@dataclass
class QuestionResult:
    qid: str
    category: str
    question: str
    ground_truth: str
    graph_answer: str
    graph_correct: bool
    graph_detail: str
    llm_answer: str = None
    llm_correct: bool = False
    llm_detail: str = ""
    llm_raw: str = ""


@dataclass
class BenchmarkResult:
    instance_name: str = ""
    total_questions: int = 0
    graph_score: int = 0
    llm_score: int = 0
    by_category: dict = field(default_factory=dict)
    questions: list = field(default_factory=list)
    model: str = ""
    timestamp: str = ""
    graph_stats: dict = field(default_factory=dict)


class Benchmark:
    def __init__(self, markdown_path, generators=None,
                 max_per_category=10, seed=42):
        self.markdown_path = markdown_path
        self.instance_name = os.path.splitext(os.path.basename(markdown_path))[0]

        print(f"Ingesting: {markdown_path}")
        self.graph = ingest_markdown(markdown_path)
        stats = self.graph.stats()
        print(f"  ENM entries: {stats['enm_entries']}")
        print(f"  KG triples: {stats['triples']}")
        print(f"  Phase encoders: {stats['phase_encoders']}")

        with open(markdown_path) as f:
            self.markdown = f.read()

        self.generators = generators or default_generators()
        self.max_per_category = max_per_category
        self.seed = seed

    def generate_questions(self):
        all_qs = []
        print("\nGenerating questions from graph topology...")
        for gen in self.generators:
            candidates = gen.generate(self.graph)
            sampled = gen.sample(candidates, self.max_per_category, self.seed)
            print(f"  {gen.category}: {len(candidates)} candidates -> {len(sampled)} selected")
            all_qs.extend(sampled)
        print(f"  Total: {len(all_qs)}")
        return all_qs

    def run(self, llm_client=None, model: str | None = None):
        """Run the benchmark with an optional LLM evaluator.

        ``model`` defaults to ``None``; when using the LLMClient path
        the model is resolved from ``GMS_LLM_MODEL_SCORER`` (falling
        back to ``GMS_LLM_MODEL``) by ``create_client``. Pass an
        explicit string to override per-call.
        """

        questions = self.generate_questions()

        # For reporting purposes, derive a display model name from the
        # client when none was supplied explicitly.
        reported_model = model
        if reported_model is None and llm_client is not None:
            reported_model = getattr(llm_client, "model", "") or ""

        result = BenchmarkResult(
            instance_name=self.instance_name,
            total_questions=len(questions),
            model=reported_model or "",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            graph_stats=self.graph.stats(),
        )

        print(f"\nEvaluating {len(questions)} questions...")
        for i, q in enumerate(questions):
            graph_answer = q.graph_answer_fn(self.graph)
            graph_score = score_answer(graph_answer, q.ground_truth)
            if graph_score.correct:
                result.graph_score += 1

            qr = QuestionResult(
                qid=q.qid,
                category=q.category,
                question=q.natural_language,
                ground_truth=str(q.ground_truth),
                graph_answer=str(graph_answer),
                graph_correct=graph_score.correct,
                graph_detail=graph_score.detail,
            )

            if llm_client:
                try:
                    tag = f"[{i+1}/{len(questions)}] {q.qid}"
                    print(f"  {tag}...", end=" ", flush=True)
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
                    print(f"{status}")
                except Exception as e:
                    qr.llm_answer = f"ERROR: {e}"
                    print(f"ERROR: {e}")

            cat = q.category
            if cat not in result.by_category:
                result.by_category[cat] = {"graph": 0, "llm": 0, "total": 0}
            result.by_category[cat]["total"] += 1
            if graph_score.correct:
                result.by_category[cat]["graph"] += 1
            if qr.llm_correct:
                result.by_category[cat]["llm"] += 1

            result.questions.append(qr)

        return result

    @staticmethod
    def print_results(result):
        n = result.total_questions
        print(f"\n{'=' * 70}")
        print(f"FINSTRUCTBENCH — {result.instance_name}")
        print(f"{'=' * 70}")
        print(f"  Model:     {result.model}")
        print(f"  Questions: {n}")
        print(f"  Graph:     {result.graph_stats.get('enm_entries', 0)} ENM, "
              f"{result.graph_stats.get('triples', 0)} triples")
        print(f"\n  {'Graph Baseline:':<20} {result.graph_score}/{n} "
              f"({result.graph_score/n*100:.0f}%)")
        print(f"  {'LLM + Markdown:':<20} {result.llm_score}/{n} "
              f"({result.llm_score/n*100:.0f}%)")

        print(f"\n  {'Category':<20} {'Graph':>8} {'LLM':>8}")
        print(f"  {'─' * 40}")
        for cat in sorted(result.by_category):
            c = result.by_category[cat]
            print(f"  {cat:<20} {c['graph']:>3}/{c['total']:<3}   {c['llm']:>3}/{c['total']:<3}")

        failures = [q for q in result.questions if not q.llm_correct]
        if failures:
            print(f"\n  LLM Failures ({len(failures)}):")
            for f in failures[:15]:
                print(f"    {f.qid}: {f.question[:65]}")
                print(f"      Expected: {f.ground_truth[:55]}")
                print(f"      Got:      {(f.llm_answer or 'None')[:55]}")

    @staticmethod
    def save_results(result, path):
        data = {
            "instance": result.instance_name,
            "total": result.total_questions,
            "graph_score": result.graph_score,
            "llm_score": result.llm_score,
            "model": result.model,
            "timestamp": result.timestamp,
            "graph_stats": result.graph_stats,
            "by_category": result.by_category,
            "questions": [
                {
                    "qid": q.qid, "category": q.category,
                    "question": q.question, "ground_truth": q.ground_truth,
                    "graph_answer": q.graph_answer, "graph_correct": q.graph_correct,
                    "llm_answer": q.llm_answer, "llm_correct": q.llm_correct,
                    "llm_detail": q.llm_detail,
                }
                for q in result.questions
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FinStructBench")
    parser.add_argument("markdown", help="Path to financial report markdown")
    parser.add_argument("--max-per-category", type=int, default=10)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional per-call model override (LiteLLM format). If "
            "omitted, the model is resolved from GMS_LLM_MODEL_SCORER "
            "or GMS_LLM_MODEL via gms.llm.LLMSettings."
        ),
    )
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bench = Benchmark(
        args.markdown,
        max_per_category=args.max_per_category,
        seed=args.seed,
    )

    client = None if args.no_llm else create_client()
    result = bench.run(llm_client=client, model=args.model)
    bench.print_results(result)

    output = args.output or f"finstructbench_results/{bench.instance_name}.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    bench.save_results(result, output)


if __name__ == "__main__":
    main()
