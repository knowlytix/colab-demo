# Demo-mode fixtures

Bundled read-only artifacts loaded when `KNOWLYTIX_DEMO_MODE=1` (or with no
license key — keyless demo). Kept small on purpose — target <5 MB total
across all files (currently ~0.7 MB). A curated **credit-risk** demo dataset
(GMS-273) drives the keyless-demo quickstart notebook at
`knowlytix/harness/demo/demo_quickstart.ipynb`.

| File | Shape | Purpose |
|---|---|---|
| `questions.json` | list of scored benchmark questions | bench scoring demo |
| `agent_trace.json` | dict, one MockHarness run | replay / regression demo |
| `document.json` | dict, pre-ingested doc metadata | query / retrieval demo |
| `llm_responses.json` | dict, SHA256(prompt)→response | `DemoLLMClient` canned responses |
| `knowledge_graph.pkl` | pickle, small `GeometricKnowledgeGraph` (names + `state_dict` + facts) | read-only KG query demo |

Consumers call `knowlytix.benchmark.fixtures.demo.load()` for everything.
**Regenerate** all five artifacts with
`KNOWLYTIX_DEMO_MODE=1 PYTHONPATH=. python scripts/build_demo_corpus.py`
(deterministic — a fixed seed keeps `knowledge_graph.pkl` byte-stable).
