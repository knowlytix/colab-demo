# SPDX-License-Identifier: Apache-2.0
"""
FinStructBench — Benchmark for Structured Retrieval from Financial Documents.

Auto-generates questions from document graph topology and evaluates
LLM accuracy against a provably correct graph traversal baseline.

Quick start::

    from knowlytix.benchmark import Benchmark
    from knowlytix.benchmark.instances import get_instance_path

    bench = Benchmark(get_instance_path("model_validation"))
    result = bench.run()            # graph-only (no LLM)
    bench.print_results(result)

With LLM evaluation::

    from knowlytix.benchmark.llm_caller import create_client

    client = create_client()        # requires ANTHROPIC_API_KEY
    result = bench.run(llm_client=client)
"""

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("knowlytix-benchmark")
except _PkgNotFound:  # source tree (dev install is the `gms` distribution)
    __version__ = "0.0.0.dev0"

from knowlytix.benchmark.benchmark import Benchmark, BenchmarkResult
from knowlytix.benchmark.config import FinStructBenchSettings
from knowlytix.benchmark.generators import default_generators
from knowlytix.benchmark.generators.base import GeneratedQuestion
from knowlytix.benchmark.graph import DocumentGraph, ENMEntry, ENMKey, PhaseEncoder
from knowlytix.benchmark.ingest import ingest_markdown
from knowlytix.benchmark.instances import get_instance_path, list_instances
from knowlytix.benchmark.scorers import ScoreResult, score_answer

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "DocumentGraph",
    "ENMEntry",
    "ENMKey",
    "FinStructBenchSettings",
    "GeneratedQuestion",
    "PhaseEncoder",
    "ScoreResult",
    "default_generators",
    "get_instance_path",
    "ingest_markdown",
    "list_instances",
    "score_answer",
]
