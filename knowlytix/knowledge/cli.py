# SPDX-License-Identifier: Apache-2.0
"""DocGMS command-line interface."""

import argparse
import os
import sys

import torch

from knowlytix.knowledge.config import DocGMSConfig


def main():
    parser = argparse.ArgumentParser(
        prog="docgms",
        description="DocGMS: Geometric Expert System with LLM-Augmented Learning",
    )
    parser.add_argument("--store-path", default="docgms_store",
                        help="Path to persistent store directory")
    parser.add_argument("--device", default=None,
                        help="Device: cuda or cpu")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="LLM model name")
    parser.add_argument("--llm-backend", choices=["anthropic", "local"],
                        default="anthropic")

    sub = parser.add_subparsers(dest="command")

    # --- ingest ---
    ingest_p = sub.add_parser("ingest",
                               help="Ingest a document into the expert store")
    ingest_p.add_argument("document", help="Path to PDF, TeX, or markdown")
    ingest_p.add_argument("--epochs", type=int, default=200)
    ingest_p.add_argument("--no-llm", action="store_true",
                          help="Skip LLM for conversion (text-only extraction)")

    # --- query ---
    query_p = sub.add_parser("query", help="Query the expert system")
    query_p.add_argument("question", nargs="?", default=None,
                         help="Question to ask")
    query_p.add_argument("--mode", choices=["gms_only", "llm_only", "gms_llm"],
                         default="gms_llm")
    query_p.add_argument("--interactive", action="store_true",
                         help="Interactive Q&A session")
    query_p.add_argument("--no-learn", action="store_true",
                         help="Disable auto-learning from answers")

    # --- compare ---
    compare_p = sub.add_parser("compare",
                                help="Run three-way comparison")
    compare_p.add_argument("document", nargs="?", default=None,
                           help="Document to ingest first (optional)")
    compare_p.add_argument("--max-per-category", type=int, default=10)
    compare_p.add_argument("--seed", type=int, default=42)
    compare_p.add_argument("--output", default=None)
    compare_p.add_argument("--modes", nargs="+",
                           default=["gms_only", "llm_only", "gms_llm"])

    # --- research ---
    research_p = sub.add_parser("research",
                                 help="Research a question via web agent")
    research_p.add_argument("question", help="Question to research")
    research_p.add_argument("--max-sources", type=int, default=5,
                            help="Max web sources to fetch")

    # --- status ---
    sub.add_parser("status", help="Show store statistics")

    # --- setup ---
    setup_p = sub.add_parser("setup",
                              help="Set up GMS expert system for Claude Code")
    setup_p.add_argument("--stores-dir", default="gms_stores",
                         help="Directory for expert system stores")
    setup_p.add_argument("--project-dir", default=".",
                         help="Project directory to write .mcp.json into")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "setup":
        _cmd_setup(args)
        return

    # Build config
    config = DocGMSConfig(store_path=args.store_path)
    config.convert.llm_model = args.model
    config.convert.llm_backend = args.llm_backend

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.command == "ingest":
        _cmd_ingest(args, config, device)
    elif args.command == "query":
        _cmd_query(args, config, device)
    elif args.command == "compare":
        _cmd_compare(args, config, device)
    elif args.command == "research":
        _cmd_research(args, config, device)
    elif args.command == "status":
        _cmd_status(config, device)


def _cmd_ingest(args, config: DocGMSConfig, device: torch.device):
    from knowlytix.knowledge.ingest import ingest_document
    from knowlytix.knowledge.llm_backend import create_backend
    from knowlytix.knowledge.store import GMSExpertStore

    config.train.epochs = args.epochs
    store = GMSExpertStore(config, device)
    store.load()  # load existing if present

    llm = None if args.no_llm else create_backend(config.convert)
    result = ingest_document(store, args.document, llm, config, device)

    print(f"\nDone. Store at: {config.store_path}")
    store._print_stats()


def _cmd_query(args, config: DocGMSConfig, device: torch.device):
    from knowlytix.knowledge.llm_backend import create_backend
    from knowlytix.knowledge.query import QueryEngine
    from knowlytix.knowledge.store import GMSExpertStore

    store = GMSExpertStore(config, device)
    if not store.load():
        print("No store found. Run 'docgms ingest <document>' first.")
        sys.exit(1)

    if args.no_learn:
        config.learn.auto_learn = False

    llm = create_backend(config.convert)
    engine = QueryEngine(store, llm, config)

    if args.interactive:
        _interactive_loop(engine, args.mode)
    elif args.question:
        result = engine.query(args.question, mode=args.mode)
        _print_query_result(result)
    else:
        print("Provide a question or use --interactive.")
        sys.exit(1)


def _cmd_compare(args, config: DocGMSConfig, device: torch.device):
    from knowlytix.knowledge.compare import ComparisonRunner
    from knowlytix.knowledge.llm_backend import create_backend
    from knowlytix.knowledge.query import QueryEngine
    from knowlytix.knowledge.store import GMSExpertStore

    store = GMSExpertStore(config, device)

    # Optionally ingest a document first
    if args.document and not store.exists():
        from knowlytix.knowledge.ingest import ingest_document
        llm = create_backend(config.convert)
        ingest_document(store, args.document, llm, config, device)
    elif not store.load():
        print("No store found. Provide a document or run 'docgms ingest' first.")
        sys.exit(1)

    llm = create_backend(config.convert)
    engine = QueryEngine(store, llm, config)
    runner = ComparisonRunner(
        store, engine, config,
        max_per_category=args.max_per_category,
        seed=args.seed,
    )
    report = runner.run(modes=args.modes)
    runner.print_report(report)

    output = args.output or f"{config.store_path}/comparison.json"
    runner.save_report(report, output)


def _cmd_research(args, config: DocGMSConfig, device: torch.device):
    from knowlytix.knowledge.llm_backend import create_backend
    from knowlytix.knowledge.store import GMSExpertStore
    from knowlytix.knowledge.web_agent import WebAgent

    store = GMSExpertStore(config, device)
    if not store.load():
        print("No store found. Run 'docgms ingest <document>' first.")
        sys.exit(1)

    llm = create_backend(config.convert)
    agent = WebAgent(store, llm, config, device,
                     max_sources=args.max_sources)
    result = agent.research(args.question)

    print(f"\n{'='*60}")
    print("WEB RESEARCH RESULT")
    print(f"{'='*60}")
    print(f"  Query:    {result.query}")
    print(f"  Sources:  {len(result.sources)}")
    print(f"  Triples:  {result.total_triples_found} found, "
          f"{result.triples_merged} merged, "
          f"{result.triples_rejected} rejected")
    print(f"  Source:   {result.answer_source}")
    print(f"  Time:     {result.elapsed_seconds:.1f}s")

    if result.sources:
        print("\n  Sources consulted:")
        for src in result.sources:
            status = "CONSISTENT" if src.is_consistent else "CONTRADICTORY"
            print(f"    [{status}] {src.title[:60]}")
            print(f"      {src.url}")
            print(f"      Triples: {len(src.triples)} "
                  f"(novel: {src.n_novel}, contradictions: {src.n_contradictions})")

    if result.answer:
        if result.answer_source == "gms":
            label = "[GMS Knowledge Store]"
        elif result.answer_source == "synthesis":
            label = "[External Knowledge — Web Sources, Verified by GMS]"
        else:
            label = f"[{result.answer_source}]"
        print(f"\n  {label}")
        print(f"  Answer:\n  {result.answer[:1000]}")
    else:
        print("\n  No answer found.")

    # Training runs in background — wait for it before final save
    print(f"\n  Training status: {agent.learner.training_status()}")
    if agent.learner.is_training:
        print("  Waiting for background training to complete...")
        agent.learner.wait_for_training(timeout=300)
        print(f"  Training: {agent.learner.last_train_result}")
    else:
        # No background training needed, save immediately
        store.save()
        print("  Store saved.")


def _cmd_status(config: DocGMSConfig, device: torch.device):
    from knowlytix.knowledge.store import GMSExpertStore

    store = GMSExpertStore(config, device)
    if not store.load():
        print(f"No store found at: {config.store_path}")
        return

    s = store.stats()
    print(f"\nDocGMS Expert Store: {config.store_path}")
    print(f"  Entities:  {s['entities']}")
    print(f"  Relations: {s['relations']}")
    print(f"  Triples:   {s['triples']}")
    print(f"  ENM:       {s['enm_entries']}")
    print(f"  Documents: {s['documents']}")


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _interactive_loop(engine, mode: str):
    print(f"\nDocGMS Interactive Q&A (mode: {mode})")
    print("Type 'quit' to exit, 'mode <m>' to switch mode.\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower().startswith("mode "):
            mode = question.split()[1]
            print(f"  Switched to mode: {mode}")
            continue

        result = engine.query(question, mode=mode)
        _print_query_result(result)
        print()


def _print_query_result(result):
    # Knowledge source label
    source = result.source
    if source == "GMS":
        if result.source_detail == "no_match":
            label = "[GMS Knowledge Store — No Match Found]"
        elif result.source_detail == "enriched_by_web":
            label = "[GMS Knowledge Store, Enriched by Web Sources]"
        else:
            label = "[GMS Knowledge Store]"
    elif source == "External":
        label = "[External Knowledge — Web Sources, Verified by GMS]"
    elif source == "llm":
        label = "[LLM Response, Verified by GMS]"
    else:
        label = f"[{source}]"

    print(f"\n  {label}")
    print(f"A: {result.answer[:500]}")
    print(f"  Source: {result.source}  Confidence: {result.confidence:.2f}")
    if result.confidence_method:
        print(f"  Confidence method: {result.confidence_method}")

    if result.extracted_triples:
        n_novel = sum(1 for s in result.extracted_triples if s.is_novel)
        n_known = len(result.extracted_triples) - n_novel
        print(f"  Triples: {len(result.extracted_triples)} extracted "
              f"({n_known} known, {n_novel} novel)")

    if result.plausibility:
        p = result.plausibility
        print(f"  Plausibility: mean={p.mean_score:.3f} "
              f"worst={p.worst_score:.3f} "
              f"{'PASS' if p.plausible else 'FAIL'}")

    if result.verification:
        v = result.verification
        n_c = sum(1 for c in v.contradictions if c.is_contradiction)
        n_h = sum(1 for h in v.inconsistencies if not h.is_consistent)
        print(f"  Verification: {'CONSISTENT' if v.overall_consistent else 'ISSUES'} "
              f"(contradictions={n_c}, holonomy_violations={n_h})")
        if v.detail and v.detail != "All checks passed":
            for line in v.detail.split("\n")[:3]:
                print(f"    {line}")

    if result.web_sources:
        n_src = len(result.web_sources)
        n_consistent = sum(1 for s in result.web_sources if s.is_consistent)
        print(f"  Web sources: {n_src} ({n_consistent} consistent)")
        for src in result.web_sources[:3]:
            print(f"    {'OK' if src.is_consistent else 'XX'} {src.title[:50]} "
                  f"({len(src.triples)} triples)")
    if result.triples_merged > 0:
        print(f"  Triples merged into GMS: {result.triples_merged}")

    if result.learned:
        lr = result.learn_result
        print(f"  Learned: {lr.reason}")


def _cmd_setup(args):
    """Set up GMS expert system builder for Claude Code."""
    import json as _json

    project_dir = os.path.abspath(args.project_dir)
    stores_dir = os.path.abspath(args.stores_dir)

    # Find Python path
    python_path = sys.executable
    # Repo root: knowlytix/knowledge/cli.py → knowlytix/knowledge/ → knowlytix/ → repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    mcp_config = {
        "mcpServers": {
            "gms": {
                "command": python_path,
                "args": [
                    "-m", "knowlytix.knowledge.mcp_server",
                    "--stores-dir", stores_dir,
                ],
                "cwd": repo_root,
                "env": {
                    "PYTHONPATH": repo_root,
                },
            }
        }
    }

    mcp_path = os.path.join(project_dir, ".mcp.json")

    # If .mcp.json already exists, merge rather than overwrite
    if os.path.isfile(mcp_path):
        with open(mcp_path) as f:
            existing = _json.load(f)
        existing.setdefault("mcpServers", {})["gms"] = mcp_config["mcpServers"]["gms"]
        mcp_config = existing

    with open(mcp_path, "w") as f:
        _json.dump(mcp_config, f, indent=2)

    os.makedirs(stores_dir, exist_ok=True)

    print("\nGMS Expert System Builder — Setup Complete")
    print(f"{'='*50}")
    print(f"  MCP config:   {mcp_path}")
    print(f"  Stores dir:   {stores_dir}")
    print(f"  Python:       {python_path}")
    print(f"  Package:      {repo_root}")
    print()
    print("Restart Claude Code to activate the GMS tools.")
    print()
    print("Available tools in Claude Code:")
    print("  gms_create   — Create a new expert system")
    print("  gms_ingest   — Add documents (PDF, TeX, Markdown)")
    print("  gms_query    — Ask questions (3-tier: GMS → LLM → Web)")
    print("  gms_search   — Find geometric connections between concepts")
    print("  gms_lookup   — Look up entity knowledge")
    print("  gms_list     — List all expert systems")
    print("  gms_use      — Switch active expert system")
    print("  gms_status   — Show store statistics")
    print()
    print("Quick start:")
    print("  1. Restart Claude Code in this directory")
    print('  2. Say: "Ingest my-paper.pdf as a new expert system called research"')
    print('  3. Say: "What are the main findings?"')
    print('  4. Say: "How does X relate to Y?"')
