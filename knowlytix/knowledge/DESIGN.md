# DocGMS: Geometric Expert System with LLM-Augmented Learning

## 1. Vision

DocGMS is a **persistent geometric expert system** that combines:
- **GMS** (Geometric Memory Systems) as the structured knowledge back-end: exact recall, contradiction detection, consistency checking, relational reasoning
- **LLM** (Claude or local models) as the reasoning/language front-end: open-ended reasoning, natural language understanding, zero-shot generalization

The system ingests documents, builds a geometric knowledge store on the hypersphere, answers queries with geometric confidence, and **grows smarter over time** by learning verified knowledge from LLM interactions.

**Key principle:** The LLM proposes. GMS verifies. Verified knowledge grows GMS. GMS gets smarter.

---

## 2. Architecture

```
                    ┌──────────────────────────────┐
                    │         LLM (Claude)          │
                    │  - Open-ended reasoning       │
                    │  - NLU / ambiguous questions   │
                    │  - Zero-shot generalization    │
                    └──────────┬───────────────────┘
                               │ answer text
                               ▼
                    ┌──────────────────────────────┐
                    │    Triple Extraction (LLM)    │
                    │  answer → (h, r, t) triples   │
                    │         + numeric values       │
                    └──────────┬───────────────────┘
                               │ extracted triples
                               ▼
              ┌────────────────────────────────────────┐
              │         GMS Verification Layer          │
              │                                        │
              │  1. score_triple(h, r, t)              │
              │     → geodesic distance on S^{m-1}     │
              │     → lower = more plausible            │
              │                                        │
              │  2. tension_energy(a, b)               │
              │     → Clifford algebra: E = 2sin(θ/2)  │
              │     → 0=agree, √2=irrelevant, 2=contra │
              │                                        │
              │  3. path_holonomy(path, direct)        │
              │     → ‖R_path R_direct^{-1} - I‖_F    │
              │     → 0=consistent, high=inconsistent   │
              └──────┬─────────────────┬───────────────┘
                     │                 │
              contradicts?        consistent?
                     │                 │
                     ▼                 ▼
              ┌──────────┐    ┌────────────────────┐
              │  REJECT   │    │   LEARN (grow GMS)  │
              │  + report │    │  1. Expand embeddings│
              │  why      │    │  2. Sentence-xformer │
              └──────────┘    │     init             │
                              │  3. Freeze existing   │
                              │  4. Riemannian GD on  │
                              │     new embeddings    │
                              │  5. ENM write for     │
                              │     numeric facts     │
                              │  6. Post-verify       │
                              └────────────────────┘
```

### 2.1 The Query Loop

Every query follows this flow:

1. **GMS first**: Attempt structured answer via ENM lookup, triple scoring, phase check, or relational transport. If confident answer found, return immediately.

2. **LLM augmented**: If GMS cannot answer (open-ended, ambiguous, or novel domain), query the LLM with GMS context prepended for grounding:
   - Relevant triples from the store
   - Exact numeric values from ENM
   - Phase-encoded threshold results
   - Known contradictions and tensions

3. **Extract**: Decompose LLM answer into (head, relation, tail) triples + numeric values.

4. **Verify**: Score each extracted triple against GMS:
   - `score_triple()`: geodesic distance — is this fact plausible given the manifold?
   - `tension_energy()`: does this contradict existing entities?
   - `path_holonomy()`: is this consistent with existing relational paths?

5. **Learn or Reject**:
   - If verified (no contradictions, consistent paths) and contains novel facts:
     - Expand GMS embedding tables for new entities
     - Initialize via sentence-transformers
     - Freeze existing embeddings
     - Riemannian gradient descent on new embeddings only
     - Write numeric values to ENM
     - The store grows — future queries benefit
   - If contradictions detected:
     - Reject the write
     - Report which facts contradicted and why (tension energy values, holonomy defects)

6. **Return**: Answer with full provenance — source, confidence, verification report, whether new facts were learned.

### 2.2 GMS vs LLM: Complementary Strengths

| Capability | GMS Handles | LLM Handles |
|---|---|---|
| Exact numeric recall | ENM register (100%, lossless, SHA-256) | - |
| Contradiction detection | Tension energy E(a,b) in u-space | - |
| Path consistency | Holonomy defect via rotor composition | - |
| Threshold/inequality | Phase encoding on S^1 | - |
| Multi-hop reasoning | Cayley rotor transport: R_L...R_1 @ v_h | - |
| Link prediction | score_all_tails(): geodesic ranking | - |
| Open-ended reasoning | - | Natural language generation |
| Ambiguous questions | - | NLU + intent classification |
| Novel domains (zero-shot) | - | Generalization from pretraining |
| Explanation generation | - | Articulate reasoning in text |
| Answer verification | Scores + checks LLM outputs | - |
| Knowledge growth | Riemannian GD on verified facts | Proposes new facts for verification |

### 2.3 Why GMS + LLM Beats LLM Alone

1. **Exact Numerical Recall**: GMS 100% vs LLM ~2%. ENM stores IEEE 754 float64 with SHA-256 integrity.

2. **Contradiction Detection**: Tension energy (Clifford algebra) is a continuous geometric signal, not pattern matching. F1 = 0.71 vs 0.57 in paper experiments.

3. **Every LLM Answer is Audited**: Extracted triples are scored on the manifold. The system quantifies how plausible each claim is and flags contradictions — the LLM never gets the final word unverified.

4. **Persistent Memory**: The store persists to disk and grows over sessions. An LLM's context window is ephemeral.

5. **Integrity Protection**: Contradiction gates prevent bad knowledge from entering the store. RAG has no such mechanism.

6. **Speed**: Structured queries answered in <1ms (manifold ops) vs ~2s (API call).

7. **The Store Gets Smarter**: Every verified interaction that introduces novel, consistent facts expands the manifold. Link prediction improves, contradiction detection refines, multi-hop paths become richer.

---

## 3. File Structure

```
docgms/
  __init__.py          # Package + public API exports
  __main__.py          # Entry: python -m docgms
  cli.py               # CLI: ingest, query, compare, status
  config.py            # Configuration dataclasses
  convert.py           # PDF/TeX/MD -> markdown conversion
  llm_backend.py       # Unified LLM interface (Anthropic + local)
  store.py             # GMSExpertStore: persistent geometric knowledge store
  ingest.py            # Document ingestion into store
  extract.py           # LLM answer -> triple extraction + GMS scoring
  verify.py            # Contradiction (tension) + consistency (holonomy)
  learn.py             # Runtime memory growth via Riemannian GD
  query.py             # Query engine: the full GMS + LLM + verify + learn loop
  compare.py           # GMS+LLM vs LLM-only evaluation
```

---

## 4. Module Specifications

### 4.1 `config.py` — Configuration

```python
@dataclass
class ConvertConfig:
    llm_model: str = "claude-sonnet-4-20250514"
    llm_backend: str = "anthropic"        # "anthropic" or "local"
    local_model_name: str = ""
    max_pages: int = 200

@dataclass
class VerifyConfig:
    tau_contra: float = 1.7               # tension energy contradiction threshold
    tau_ent: float = 0.8                  # entailment threshold
    tau_path: float = 0.5                 # holonomy consistency threshold
    holonomy_alpha: float = 0.1           # decay for effective holonomy
    min_plausibility: float = 1.0         # max geodesic dist for plausible triple

@dataclass
class LearnConfig:
    n_steps: int = 50                     # Riemannian GD steps for new entities
    lr_write: float = 5e-3
    freeze_existing: bool = True
    contradiction_gate: bool = True       # reject writes that introduce contradictions
    auto_learn: bool = True               # auto-write verified novel facts from Q&A

@dataclass
class DocGMSConfig:
    convert: ConvertConfig
    geometry: GeometryConfig              # from knowlytix.core.config (d_v=128, d_u=128, m=64)
    loss: LossConfig                      # from knowlytix.core.config
    train: TrainConfig                    # from knowlytix.core.config
    memory: MemoryConfig                  # from knowlytix.core.config
    verify: VerifyConfig
    learn: LearnConfig
    store_path: str = "docgms_store/"     # persistent store directory
```

Imports from: `knowlytix.core.config.GeometryConfig`, `LossConfig`, `TrainConfig`, `MemoryConfig`

### 4.2 `llm_backend.py` — LLM Interface

```python
class LLMBackend(ABC):
    def call(self, system: str, user: str, max_tokens: int = 2048) -> str: ...

class AnthropicBackend(LLMBackend):
    """Anthropic API. Pattern from knowlytix/benchmark/llm_caller.py."""
    def __init__(self, model: str = "claude-sonnet-4-20250514"): ...

class LocalTransformersBackend(LLMBackend):
    """HuggingFace model on GPU via transformers."""
    def __init__(self, model_name: str, device: str = "cuda"): ...

def create_backend(config: ConvertConfig) -> LLMBackend: ...
```

### 4.3 `convert.py` — Document Conversion

```python
def detect_format(path: str) -> str:
    """Returns 'pdf', 'tex', 'md', 'txt', 'docx'."""

def convert_pdf_to_markdown(path: str, config: ConvertConfig, llm_fn: Callable) -> str:
    """PyMuPDF for text extraction. LLM for complex table reconstruction."""

def convert_tex_to_markdown(path: str, config: ConvertConfig, llm_fn: Callable) -> str:
    """pandoc subprocess for structure. Post-process LaTeX remnants.
    LLM fallback for tabular environments."""

def convert_document(path: str, config: ConvertConfig, llm_fn: Callable) -> str:
    """Top-level dispatcher. .md files read directly."""
```

Dependencies: `pymupdf`, system `pandoc`

### 4.4 `store.py` — GMSExpertStore (the core)

```python
class GMSExpertStore:
    """Persistent geometric expert system store.

    Manages: trained GKG model, ENM register, compression memory,
    transport layer, adapter mappings. Supports ingestion of new
    documents and runtime growth from verified Q&A.

    Disk layout (store_path/):
      model.pt          — GKG weights
      enm.json          — ENM key-value pairs
      adapter.json      — entity/relation vocabularies
      metadata.json     — ingestion history, stats
      documents/        — ingested markdown copies
    """

    def __init__(self, config: DocGMSConfig, device: torch.device): ...

    # --- Core state ---
    model: GeometricKnowledgeGraph
    adapter: GraphToGMS
    enm: ExactNumericalMemory
    doc_graph: DocumentGraph
    transport: RelationalTransport
    compression: CompressionMemory
    router: MemoryRouter

    # --- Persistence ---
    def save(self): ...
    def load(self) -> bool: ...
    def exists(self) -> bool: ...

    # --- GMS Operations (thin wrappers around model/transport/enm) ---
    def score_triple(self, head: str, rel: str, tail: str) -> float:
        """Geodesic distance on S^{m-1}. Lower = more plausible."""

    def tension_energy(self, entity_a: str, entity_b: str) -> float:
        """Clifford tension energy. 0=agree, sqrt(2)=irrelevant, 2=contradict.
        Uses: knowlytix.core.graph.gkg.tension_energy_pairs() -> knowlytix.core.geometry.clifford.tension_energy()"""

    def check_holonomy(self, relation_path: list[str], direct_relation: str) -> float:
        """Holonomy defect. Uses: transport.path_holonomy()"""

    def is_path_consistent(self, relation_path: list[str], direct: str) -> bool:
        """True if holonomy defect <= tau_path. Uses: transport.is_path_consistent()"""

    def lookup_enm(self, category: str, entity_id: str) -> float | None:
        """Exact numeric lookup with SHA-256 integrity."""

    def link_predict(self, head: str, relation: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Top-k tail predictions using model.score_all_tails()."""

    def query_triples(self, head=None, relation=None, tail=None) -> list[tuple]:
        """Pattern match on doc_graph.triples."""

    def find_contradictions(self) -> list[tuple]:
        """doc_graph.find_contradictions()."""

    def stats(self) -> dict:
        """entities, relations, triples, enm_entries, documents ingested."""
```

Imports:
- `knowlytix.core.graph.gkg.GeometricKnowledgeGraph`
- `knowlytix.core.graph.transport.RelationalTransport`
- `knowlytix.core.memory.enm.ExactNumericalMemory`, `ENMKey`
- `knowlytix.core.memory.compression.CompressionMemory`
- `knowlytix.core.memory.router.MemoryRouter`
- `knowlytix.core.train_finstructbench.GraphToGMS`
- `knowlytix.benchmark.graph.DocumentGraph`

### 4.5 `ingest.py` — Document Ingestion

```python
def ingest_document(store: GMSExpertStore, document_path: str,
                    llm_backend: LLMBackend, config: DocGMSConfig,
                    device: torch.device) -> IngestResult:
    """Ingest a document into the expert store.

    First document: full pipeline
      1. convert_document(path) -> markdown
      2. ingest_markdown(path) -> DocumentGraph     [knowlytix.benchmark.ingest]
      3. GraphToGMS(doc_graph) -> adapter            [knowlytix.core.train_finstructbench]
      4. train_gms(adapter, device) -> model         [knowlytix.core.train_finstructbench]
      5. populate_enm(doc_graph, adapter) -> enm     [knowlytix.core.train_finstructbench]
      6. Create transport, compression, router
      7. Save store

    Subsequent documents: incremental growth
      1. convert_document(path) -> markdown
      2. ingest_markdown(path) -> new DocumentGraph
      3. Identify new entities and triples not in store
      4. learn.expand_and_optimize(new_triples, store)  [learn.py]
      5. Add new ENM entries
      6. Merge doc_graph triples into store.doc_graph
      7. Save store

    Returns IngestResult(new_entities, new_triples, new_enm, stats).
    """

def ingest_markdown_text(store, markdown_text, config, device) -> IngestResult:
    """Ingest raw markdown string (for programmatic use)."""
```

### 4.6 `extract.py` — Triple Extraction + GMS Scoring

```python
@dataclass
class ExtractedTriple:
    head: str
    relation: str
    tail: str
    numeric_value: float | None = None   # if triple contains a number

@dataclass
class ScoredTriple:
    triple: ExtractedTriple
    gms_score: float                     # geodesic distance (lower = better)
    head_match: str | None               # nearest entity in store, or None
    tail_match: str | None
    is_novel: bool                       # True if entity not in store

@dataclass
class PlausibilityReport:
    plausible: bool
    mean_score: float
    worst_score: float
    novel_count: int
    scored_triples: list[ScoredTriple]

def extract_triples(answer_text: str, question: str,
                     llm_backend: LLMBackend) -> list[ExtractedTriple]:
    """Prompt LLM to decompose answer into structured triples.

    Prompt template:
      Given this question and answer, extract all factual claims as
      (subject, relationship, object) triples. For numeric facts,
      include the exact value. Format: one triple per line as
      TRIPLE: subject | relationship | object [| numeric_value]
    """

def score_triples(triples: list[ExtractedTriple],
                   store: GMSExpertStore,
                   device: torch.device) -> list[ScoredTriple]:
    """Score each triple against GMS manifold.

    For each triple:
    1. Fuzzy-match head/tail to nearest entity in store.adapter vocab
       (normalized string matching + optional semantic matching via
        encode_texts + cosine similarity against store embeddings)
    2. If matched: store.score_triple(h, r, t) -> geodesic distance
    3. If not matched: mark is_novel=True (candidate for learning)
    4. For numeric values: cross-check against ENM if entity exists
    """

def assess_plausibility(scored: list[ScoredTriple],
                         threshold: float = 1.0) -> PlausibilityReport:
    """Aggregate triple scores into overall plausibility verdict."""
```

### 4.7 `verify.py` — Contradiction + Consistency Checking

```python
@dataclass
class ContradictionResult:
    entity_a: str
    entity_b: str
    tension_energy: float                # 0=agree, sqrt(2)=irrelevant, 2=contradict
    is_contradiction: bool               # E > tau_contra
    is_entailment: bool                  # E < tau_ent

@dataclass
class HolonomyResult:
    path: list[str]                      # relation path [r1, r2, ...]
    direct: str                          # direct relation
    defect: float                        # holonomy defect value
    is_consistent: bool                  # defect <= tau_path

@dataclass
class VerificationReport:
    contradictions: list[ContradictionResult]
    inconsistencies: list[HolonomyResult]
    overall_consistent: bool
    confidence: float                    # 1.0=fully verified, 0.0=many issues

def check_contradictions(entity_pairs: list[tuple[str, str]],
                          store: GMSExpertStore,
                          config: VerifyConfig) -> list[ContradictionResult]:
    """Tension energy check for each entity pair.

    Uses store.tension_energy(a, b) which calls:
      knowlytix.core.graph.gkg.tension_energy_pairs(idx_a, idx_b)
        -> knowlytix.core.geometry.clifford.tension_energy(u_a, u_b)
        -> E = 2 * sin(theta/2)
    """

def check_holonomy(triples: list[ExtractedTriple],
                    store: GMSExpertStore,
                    config: VerifyConfig) -> list[HolonomyResult]:
    """Find relation triangles formed by new triples + existing graph.

    For each new triple (h, r_new, t):
      - Find existing paths from h to t through intermediate entities
      - Compute path_holonomy(existing_path, r_new)
      - Flag if defect > tau_path

    Uses store.check_holonomy() which calls:
      transport.path_holonomy(relation_path, direct_relation)
        -> compose_rotors(path) @ R_direct^{-1} - I -> Frobenius norm
        -> effective_holonomy(raw_defect, path_length, alpha)
    """

def verify_answer(scored_triples: list[ScoredTriple],
                   store: GMSExpertStore,
                   config: VerifyConfig) -> VerificationReport:
    """Combined verification: contradictions + holonomy + confidence.

    1. For all pairs of (new entity, existing entity): check_contradictions
    2. For all new triples forming paths: check_holonomy
    3. Compute confidence: 1.0 - (n_contradictions + n_inconsistencies) / n_triples
    """
```

### 4.8 `learn.py` — Runtime Memory Growth

```python
@dataclass
class LearnResult:
    accepted: bool
    reason: str                          # "consistent" or "contradiction: ..."
    new_entities: list[str]
    scores_before: list[float]           # geodesic scores pre-optimization
    scores_after: list[float]            # geodesic scores post-optimization
    tension_before: list[float]
    tension_after: list[float]
    n_steps_run: int

class RuntimeLearner:
    """Grow GMS by writing verified knowledge via Riemannian gradient descent.

    Follows the protocol from scripts/experiment_llm_knowlytix.core.py Experiment 7:
    1. Freeze trained GKG (rotors, projections, existing embeddings)
    2. Expand embedding tables with new entities
    3. Init new entity embeddings via sentence-transformers (encode_texts)
    4. Riemannian GD on new embeddings only, minimizing:
       - Triple scoring loss: d_geo(R_r @ v_h, v_t) for new triples
       - Tension energy targets: E=0 for consistent pairs, E=2 for contradictory
    5. Post-write verification
    6. If contradiction gate enabled and post-write contradictions found -> rollback
    """

    def __init__(self, store: GMSExpertStore, config: LearnConfig): ...

    def learn_triples(self, new_triples: list[ExtractedTriple],
                       device: torch.device) -> LearnResult:
        """Full learning pipeline with contradiction guard.

        Steps:
        a. Identify new entities (not in store.adapter)
        b. Pre-check: verify_answer() on new triples against store
        c. If contradiction_gate and contradictions -> return rejected
        d. Expand embedding tables
           (pattern: experiment_llm_knowlytix.core.py:1889-1939)
        e. Encode new entities: encode_texts() for v-space and u-space
           (using sentence-transformers/all-MiniLM-L6-v2 and nli-mpnet-base-v2)
        f. Freeze existing params
        g. Create new-entity-only params (v, u_proj, specificity, confidence)
        h. Riemannian GD loop for n_steps
           (pattern: experiment_llm_knowlytix.core.py:1982-2138)
        i. Post-verification: re-check tension + holonomy
        j. If still contradictory -> rollback (restore original weights)
        k. Update adapter mappings (entity_to_idx, idx_to_entity)
        l. Return LearnResult with before/after metrics
        """

    def learn_enm(self, category: str, entity_id: str, value: float) -> bool:
        """Write exact numeric value to ENM. Always accepts (no contradiction
        possible for exact storage)."""

    def _expand_embeddings(self, new_names: list[str], device): ...
    def _riemannian_optimize(self, triple_tensors, tension_targets, device): ...
    def _rollback(self): ...
```

Imports:
- `knowlytix.core.graph.encoders.encode_texts` — sentence-transformer encoding
- `knowlytix.core.geometry.sphere.normalize` — sphere projection
- `knowlytix.core.memory.enm.ENMKey` — ENM key construction
- Pattern from `scripts/experiment_llm_knowlytix.core.py:1848-2238`

### 4.9 `query.py` — The Expert System Query Engine

```python
@dataclass
class QueryResult:
    answer: Any                          # the answer
    source: str                          # "enm"|"triple"|"phase"|"transport"|"llm"|"gms+llm"
    confidence: float                    # GMS-derived confidence
    extracted_triples: list[ScoredTriple]  # triples from LLM answer (if LLM was used)
    verification: VerificationReport | None  # contradiction + holonomy (if LLM used)
    learned: bool                        # True if novel facts were written to store
    learn_result: LearnResult | None     # details of learning (if it happened)
    gms_context: str                     # GMS context provided to LLM (if augmented)

class QueryEngine:
    """Expert system query interface.

    The full query loop:
    1. GMS attempts structured answer
    2. If insufficient: LLM answers (with GMS context)
    3. Extract triples from LLM answer
    4. Score + verify (contradiction + holonomy)
    5. If verified + novel + auto_learn: grow GMS
    6. Return answer with full provenance
    """

    def __init__(self, store: GMSExpertStore, llm_backend: LLMBackend,
                 config: DocGMSConfig): ...

    def query(self, question: str, mode: str = "gms_llm") -> QueryResult:
        """Query the expert system.

        Modes:
          "gms_only"  — structured GMS answer only
          "llm_only"  — LLM answer only (still verified by GMS)
          "gms_llm"   — GMS attempts first, LLM augments, verified, learned
        """

    def query_batch(self, questions: list[str], mode: str = "gms_llm") -> list[QueryResult]:
        """Batch query."""

    # --- Internal routing ---

    def _classify_question(self, question: str) -> str:
        """Heuristic classification:
        - 'exact value', 'what is the', 'how much' -> exact_recall
        - 'above', 'below', 'exceeds', 'threshold' -> threshold
        - 'contradict', 'conflict', 'inconsistent' -> contradiction
        - 'path', 'chain', 'leads to', 'through' -> multi_hop
        - 'both', 'intersection', 'and also' -> cross_reference
        - 'how many', 'count' -> counting
        - default -> open_ended
        """

    def _gms_answer(self, question: str, qtype: str) -> tuple[Any, float] | None:
        """Attempt GMS-only answer based on question type.

        exact_recall: ENM lookup -> store.lookup_enm()
        threshold: Phase check -> doc_graph phase encoders
        contradiction: Tension energy -> store.tension_energy()
        multi_hop: Transport -> transport.path_score()
        cross_reference: Triple intersection -> store.query_triples()
        counting: Triple count -> len(store.query_triples())

        Returns (answer, confidence) or None if GMS can't answer.
        """

    def _gather_gms_context(self, question: str) -> str:
        """Build GMS context for LLM augmentation.

        1. Keyword-match entities in store.adapter vocab
        2. For matched entities: gather related triples
        3. For matched entities: lookup ENM values
        4. For matched entities: check phase thresholds
        5. Format as structured context string
        """

    def _llm_answer(self, question: str, gms_context: str | None) -> str:
        """Call LLM. If gms_context provided, prepend to prompt.

        System prompt:
          You are an expert analyst backed by a geometric knowledge store.
          Use the provided structured context as ground truth.
          For numeric values, prefer the exact values from the knowledge store.
          If the context contradicts your knowledge, flag the discrepancy.

        User prompt:
          <context>{gms_context}</context>
          <question>{question}</question>
        """

    def _attempt_learn(self, scored_triples: list[ScoredTriple],
                        verification: VerificationReport,
                        device: torch.device) -> LearnResult | None:
        """If auto_learn=True and verification.overall_consistent and novel triples exist:
        call RuntimeLearner.learn_triples() to grow GMS."""
```

### 4.10 `compare.py` — GMS+LLM vs LLM-Only Evaluation

```python
@dataclass
class ComparisonRow:
    qid: str
    category: str
    question: str
    ground_truth: Any
    # GMS-only
    gms_answer: Any
    gms_correct: bool
    # LLM-only (verified by GMS)
    llm_answer: Any
    llm_correct: bool
    llm_plausibility: float
    llm_contradictions: int
    llm_holonomy_violations: int
    # GMS+LLM (augmented, verified, learned)
    gms_llm_answer: Any
    gms_llm_correct: bool
    gms_llm_plausibility: float
    gms_llm_contradictions: int
    gms_llm_learned: bool

@dataclass
class ComparisonReport:
    document: str
    total_questions: int
    scores: dict[str, int]               # {mode: n_correct}
    by_category: dict[str, dict]
    rows: list[ComparisonRow]
    plausibility_summary: dict[str, float]  # {mode: mean_plausibility}
    contradiction_summary: dict[str, int]   # {mode: total_contradictions}
    facts_learned: int                      # novel facts added to GMS
    timestamp: str

class ComparisonRunner:
    """Three-way comparison: gms_only vs llm_only vs gms_llm.

    Uses knowlytix.benchmark.generators.default_generators() to auto-generate
    questions from store.doc_graph topology.
    """

    def __init__(self, store: GMSExpertStore, query_engine: QueryEngine,
                 config: DocGMSConfig): ...

    def run(self) -> ComparisonReport:
        """
        1. Generate questions: default_generators().generate(store.doc_graph)
        2. For each question, run query_engine.query(q, mode) for all 3 modes
        3. Score: knowlytix.benchmark.scorers.score_answer(answer, ground_truth)
        4. Aggregate: accuracy + plausibility + contradictions + facts learned
        """

    def print_report(self, report: ComparisonReport): ...
    def save_report(self, report: ComparisonReport, path: str): ...
```

Imports:
- `knowlytix.benchmark.generators.default_generators` — question generation
- `knowlytix.benchmark.scorers.score_answer`, `PARSERS` — scoring

### 4.11 `cli.py` + `__main__.py`

```
Usage:
  docgms ingest <document>           # Ingest PDF/TeX/MD into expert store
  docgms query "question"            # Query the expert system
  docgms query --interactive         # Interactive Q&A session
  docgms compare <document>          # Three-way comparison
  docgms status                      # Show store statistics
  docgms export <path>               # Export store state

Global flags:
  --store-path PATH                  # Store directory (default: docgms_store/)
  --device cuda|cpu
  --mode gms_only|llm_only|gms_llm
  --auto-learn / --no-auto-learn
  --model MODEL                      # LLM model name
  --epochs N                         # Training epochs (first ingest only)
```

---

## 5. Reuse Map

| Component | Source | Usage |
|---|---|---|
| `ingest_markdown()` | `knowlytix.benchmark.ingest` | Markdown → DocumentGraph |
| `GraphToGMS` | `knowlytix.core.train_finstructbench` | DocumentGraph → training tensors |
| `train_gms()` | `knowlytix.core.train_finstructbench` | Training loop with Riemannian SGD |
| `populate_enm()` | `knowlytix.core.train_finstructbench` | DocumentGraph ENM → GMS ENM |
| `default_generators()` | `knowlytix.benchmark.generators` | Auto-generate evaluation questions |
| `score_answer()`, `PARSERS` | `knowlytix.benchmark.scorers` | Score answers + parse LLM output |
| `GeometricKnowledgeGraph` | `knowlytix.core.graph.gkg` | Triple scoring, tension energy |
| `RelationalTransport` | `knowlytix.core.graph.transport` | Multi-hop, holonomy |
| `ExactNumericalMemory` | `knowlytix.core.memory.enm` | Exact numeric storage |
| `CompressionMemory` | `knowlytix.core.memory.compression` | Approximate memory |
| `MemoryRouter` | `knowlytix.core.memory.router` | ENM vs compression dispatch |
| `encode_texts()` | `knowlytix.core.graph.encoders` | Sentence-transformer encoding |
| `RiemannianSGD` | `knowlytix.core.optim.riemannian` | Manifold-aware optimization |
| `GMSLoss` | `knowlytix.core.losses.combined` | Multi-component loss |
| `PhaseEncoder` | `knowlytix.core.geometry.phase` | Numeric threshold queries |
| `tension_energy()` | `knowlytix.core.geometry.clifford` | Clifford algebra contradiction |
| `holonomy_defect()` | `knowlytix.core.geometry.cayley` | Rotor composition consistency |
| Test-time write pattern | `scripts/experiment_llm_knowlytix.core.py:1848-2238` | Adapted in learn.py |
| LLM call pattern | `knowlytix/benchmark/llm_caller.py` | Adapted in llm_backend.py |

---

## 6. Data Flow Summary

```
Document (PDF/TeX/MD)
    │
    ▼
[convert.py] ──────────────────────────► Markdown text
    │
    ▼
[knowlytix.benchmark.ingest] ───────────────► DocumentGraph
    │                                     (ENM + triples + phase)
    ▼
[knowlytix.core.train_finstructbench] ────────────► Trained GKG + ENM + Transport
    │
    ▼
[store.py: GMSExpertStore] ────────────► Persistent on disk
    │
    ▼
[query.py: QueryEngine]
    │
    ├── GMS answers structured queries (ENM, triples, phase, transport)
    │
    ├── LLM answers open-ended/ambiguous/zero-shot
    │       │
    │       ▼
    │   [extract.py] → (h, r, t) triples
    │       │
    │       ▼
    │   [verify.py] → tension energy + holonomy
    │       │
    │       ├── Contradicts → REJECT + report
    │       │
    │       └── Consistent → [learn.py] → GROW GMS
    │                           │
    │                           ├── Expand embeddings
    │                           ├── Riemannian GD
    │                           ├── Post-verify
    │                           └── Save store
    │
    ▼
QueryResult (answer + provenance + verification + learning)
```

---

## 7. Implementation Order

| Phase | Files | Dependency | Estimated Lines |
|---|---|---|---|
| 1 | `config.py`, `__init__.py` | None | ~70 |
| 2 | `llm_backend.py` | config | ~100 |
| 3 | `convert.py` | llm_backend | ~250 |
| 4 | `store.py` | config, gms.*, knowlytix.benchmark.* | ~300 |
| 5 | `ingest.py` | store, convert | ~150 |
| 6 | `extract.py` | store, llm_backend | ~200 |
| 7 | `verify.py` | store | ~180 |
| 8 | `learn.py` | store, verify | ~250 |
| 9 | `query.py` | store, extract, verify, learn, llm_backend | ~300 |
| 10 | `compare.py` | query, knowlytix.benchmark.generators/scorers | ~150 |
| 11 | `cli.py`, `__main__.py` | all | ~120 |
| **Total** | | | **~2070** |

---

## 8. Testing Strategy

| Test | Validates | Expected |
|---|---|---|
| Ingest model_validation.md | Full pipeline | ~5000 triples, ~260 ENM |
| ENM exact recall | store.lookup_enm() | 100% accuracy |
| Score known triple | store.score_triple() | Low geodesic distance |
| Score random triple | store.score_triple() | High geodesic distance |
| Tension: contradictory pair | store.tension_energy() | E > 1.7 |
| Tension: consistent pair | store.tension_energy() | E < 0.8 |
| Holonomy: valid path | store.check_holonomy() | defect < 0.5 |
| Extract triples from LLM | extract_triples() | Parseable triples |
| Score extracted triples | score_triples() | Scores + entity matches |
| Verify consistent answer | verify_answer() | overall_consistent=True |
| Verify contradictory answer | verify_answer() | overall_consistent=False |
| Learn consistent triples | learn_triples() | accepted=True, store grows |
| Learn contradictory triples | learn_triples() | accepted=False, rollback |
| Multi-document ingest | Second ingest | Store expands |
| Store save/load roundtrip | save() + load() | Identical state |
| Compare on auto-questions | ComparisonRunner.run() | 3-column report |
| End-to-end: ingest PDF + query | Full pipeline | Answer with provenance |
