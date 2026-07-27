# knowlytix.knowledge.rag — GEODE triple-mediated graph-RAG

A retrieval-augmented generation pipeline built on the Geometric Memory System
(GMS). Unlike vector-RAG, retrieval is **triple-mediated**: questions (prose or
structured) are translated into query triples and answered *through the GMS*,
with provenance-linked source spans handed to the LLM. Bank-grade by default —
GMS-grounded answers, optional answer self-verification, honest abstention, and
no opaque vector index. See `../../../GEODE_RAG_DESIGN.md` for the full design.

## Quick start

```python
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.llm_backend import LocalTransformersBackend
from knowlytix.knowledge.rag import RagPipeline, RagConfig

# Query-time LLM is your choice. Here: local Qwen 3B (fully on-device).
llm = LocalTransformersBackend("Qwen/Qwen2.5-3B-Instruct", device="cuda")

cfg = RagConfig(
    llm=llm,                     # synthesis (required)
    binding="embedding",         # paraphrases bind to graph vocab (else "fuzzy")
    verify_llm_output=True,      # check the answer's claims against the GMS
    on_verify_fail="abstain",
)

# Build a self-corrected store from a document, then query it.
pipe = RagPipeline.build("report.md", DocGMSConfig(ingest_mode="regex"), cfg)
# or wrap an existing GMSExpertStore: RagPipeline.from_store(store, cfg)

ans = pipe.query("What was Consumer division revenue?")
print(ans.decision, ans.confidence)   # accept | abstain ; calibrated
print(ans.answer)
for s in ans.sources:                 # provenance: file:line:char + text
    print(s.head, s.relation, s.tail, s.location)
print(ans.verification)               # per-claim GMS verdicts
print(pipe.coverage().blind_spots)    # doc sections with no triples (blind spots)
```

A runnable end-to-end demo (offline stub or local Qwen 3B):

```bash
python demo/geode_rag_demo.py --stub   # instant, no model
python demo/geode_rag_demo.py          # local Qwen 3B + embedding binding
```

## How a query flows

```
question
  -> extract query triples (LLM)         e.g. (consumer, revenue, ?)
  -> bind to graph vocab                  exact/fuzzy, then embedding fallback
  -> bind-check                           nothing bound -> ABSTAIN (don't guess)
  -> GMS retrieve                         asserted edge / link_predict / multi-hop
  -> attach provenance                    source span per fact
  -> synthesize (LLM)                     grounded in facts + spans only
  -> self-verify (optional)               claim triples checked vs GMS
  -> decision                             accept | abstain
```

## Components

| Module | Role |
|---|---|
| `config.py` | `RagConfig` — per-role LLMs, binding mode, thresholds, verify/audit/dense flags |
| `query_triples.py` | `QueryTripleExtractor` — NL → query triples (`?` = asked slot) |
| `binding.py` | `TripleBinder` — resolve slots to graph entities/relations (fuzzy / embedding) |
| `retrieve.py` | `Retriever` — GMS retrieval (asserted edges preferred) + provenance |
| `assemble.py` | `Assembler` — grounded synthesis from facts + spans |
| `verify.py` | `AnswerVerifier` — GMS self-verification (hallucination detector) |
| `coverage.py` | `coverage_report` — measurable triple-mediation blind spots |
| `pipeline.py` | `RagPipeline` / `RagAnswer` — orchestration, decisions, audit trail |

## Design stance

- **GMS is the only authoritative retriever.** Dense vector retrieval is
  distrusted — opt-in, off by default, and flagged unverified when used.
- **Abstain over guess.** If query triples don't bind or no fact matches, the
  pipeline abstains; the coverage monitor makes the blind spots measurable.
- **Pluggable LLMs (per role).** The query-time LLM is the user's choice; GEODE's
  *ingestion* actor stays Qwen-3B-locked separately.
- **Exact numerics.** Asserted facts (ENM/triples) are preferred over predicted
  ones, so financial values are exact and provenance-consistent.
