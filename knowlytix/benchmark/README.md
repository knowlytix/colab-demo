# FinStructBench

**Benchmark for Structured Information Retrieval from Financial Documents**

FinStructBench evaluates how well LLMs extract, aggregate, and reason over structured data in financial documents. Questions are auto-generated from a document's knowledge graph, making ground truth **provably correct by construction**.

## Key Results

| Category | Graph Baseline | Claude Sonnet 4 |
|---|---|---|
| Threshold | 100% | 86% |
| Cross-Reference | 100% | 62% |
| Exact Recall | 100% | 44% |
| Counting | 100% | 34% |
| Contradiction | 100% | 20% |
| Multi-Hop | 100% | **0%** |
| **Overall** | **100%** | **45%** |

## Installation

```bash
pip install knowlytix-benchmark
```

For LLM evaluation (requires an Anthropic API key):

```bash
pip install knowlytix-benchmark[llm]
```

## Quick Start

### Command Line

The wheel installs a `benchmark` console script.

```bash
# List bundled financial report instances
benchmark list-instances

# Run graph-only evaluation (no API calls)
benchmark run knowlytix/benchmark/instances/model_validation.md --no-llm

# Run with LLM evaluation
export ANTHROPIC_API_KEY=sk-ant-...
benchmark run knowlytix/benchmark/instances/fair_lending.md

# Customize
benchmark run report.md --max-per-category 20 --model claude-sonnet-4-20250514 --seed 42
```

### Python API

```python
from knowlytix.benchmark import Benchmark, get_instance_path

# Load a bundled instance
bench = Benchmark(get_instance_path("model_validation"))

# Graph-only evaluation
result = bench.run()
bench.print_results(result)

# With LLM
from knowlytix.benchmark.llm_caller import create_client
client = create_client()
result = bench.run(llm_client=client)
bench.print_results(result)
bench.save_results(result, "results.json")
```

### Bring Your Own Document

FinStructBench works with any markdown document containing tables:

```python
from knowlytix.benchmark import Benchmark

bench = Benchmark("path/to/your/report.md")
result = bench.run()
bench.print_results(result)
```

## Bundled Instances

| Instance | Regulatory Context | ENM | Triples | Questions |
|---|---|---|---|---|
| `model_validation` | SR 11-7 | 261 | 5,586 | 60 |
| `fair_lending` | ECOA / HMDA | 866 | 1,962 | 50 |
| `stress_test` | CCAR / DFAST | 207 | 1,355 | 50 |
| `credit_portfolio` | OCC Guidelines | 211 | 2,334 | 50 |
| `basel_capital` | Basel III Pillar 3 | 27 | 785 | 48 |

## Question Categories

1. **Exact Recall** — Look up a single numeric value by key
2. **Threshold** — Check if a value meets a regulatory bound
3. **Cross-Reference** — Find entities appearing in two relation types
4. **Counting** — Count triples matching a pattern
5. **Contradiction** — Detect pass/fail inconsistencies across segments
6. **Multi-Hop** — Chained queries (argmin/argmax + cross-type lookup)

## How It Works

1. **Ingest**: Markdown document → `DocumentGraph` (ENM entries + KG triples + phase encoders)
2. **Generate**: Mine graph topology → questions with provably correct answers
3. **Evaluate**: Graph baseline (100% by construction) vs. LLM under test

## Extending

### Add a question generator

```python
from knowlytix.benchmark.generators.base import QuestionGenerator, GeneratedQuestion

class MyGenerator(QuestionGenerator):
    category = "my_category"

    def generate(self, graph):
        questions = []
        # Mine graph topology for questions...
        return questions
```

### Add a benchmark instance

Write a markdown document with tables and run:

```bash
benchmark info my_report.md        # Check ingestion statistics
benchmark run my_report.md --no-llm  # Verify graph baseline
```

## Citation

```bibtex
@article{sudjianto2026finstructbench,
  title={FinStructBench: A Benchmark for Structured Information Retrieval
         from Financial Documents Using Graph-Verifiable Questions},
  author={Sudjianto, Agus},
  year={2026}
}
```

## License

Apache 2.0
