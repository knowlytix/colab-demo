# SPDX-License-Identifier: Apache-2.0
"""FinStructBench — command-line interface."""

import argparse
import os
import sys

from knowlytix.benchmark.benchmark import Benchmark


def main():
    parser = argparse.ArgumentParser(
        prog="finstructbench",
        description="FinStructBench: Benchmark for structured retrieval from financial documents",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_p = sub.add_parser("run", help="Run benchmark on a markdown document")
    run_p.add_argument("markdown", help="Path to financial report markdown")
    run_p.add_argument("--max-per-category", type=int, default=10)
    run_p.add_argument(
        "--model",
        default=None,
        help=(
            "Optional per-call model override (LiteLLM format, e.g. "
            "'anthropic/claude-sonnet-4-20250514'). If omitted, the model "
            "is resolved from GMS_LLM_MODEL_SCORER or GMS_LLM_MODEL via "
            "gms.llm.LLMSettings."
        ),
    )
    run_p.add_argument("--no-llm", action="store_true",
                       help="Graph-only evaluation (no LLM API calls)")
    run_p.add_argument("--output", default=None, help="Output JSON path")
    run_p.add_argument("--seed", type=int, default=42)

    # --- list-instances ---
    sub.add_parser("list-instances",
                            help="List bundled benchmark instances")

    # --- info ---
    info_p = sub.add_parser("info",
                            help="Show graph statistics for a document")
    info_p.add_argument("markdown", help="Path to financial report markdown")

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "list-instances":
        _cmd_list_instances()
    elif args.command == "info":
        _cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_run(args):
    bench = Benchmark(
        args.markdown,
        max_per_category=args.max_per_category,
        seed=args.seed,
    )

    client = None
    if not args.no_llm:
        from knowlytix.benchmark.llm_caller import create_client
        client = create_client()

    result = bench.run(llm_client=client, model=args.model)
    bench.print_results(result)

    output = args.output or f"finstructbench_results/{bench.instance_name}.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    bench.save_results(result, output)


def _cmd_list_instances():
    instances_dir = os.path.join(os.path.dirname(__file__), "instances")
    print("Bundled benchmark instances:")
    print()
    for f in sorted(os.listdir(instances_dir)):
        if f.endswith(".md"):
            name = f.replace(".md", "")
            path = os.path.join(instances_dir, f)
            lines = sum(1 for _ in open(path))
            print(f"  {name:<25} {lines:>5} lines   {path}")
    print()
    print("Run with: finstructbench run <path-to-instance.md>")


def _cmd_info(args):
    from knowlytix.benchmark.ingest import ingest_markdown

    print(f"Ingesting: {args.markdown}")
    graph = ingest_markdown(args.markdown)
    stats = graph.stats()

    print("\nDocumentGraph Statistics:")
    print(f"  ENM entries:    {stats['enm_entries']}")
    print(f"  KG triples:     {stats['triples']}")
    print(f"  Phase encoders: {stats['phase_encoders']}")
    print("\n  ENM by type:")
    for t, c in sorted(stats["enm_types"].items()):
        print(f"    {t:<35} {c:>5}")
    print("\n  Triples by relation:")
    for r, c in sorted(stats["relations"].items(), key=lambda x: -x[1]):
        print(f"    {r:<35} {c:>5}")
