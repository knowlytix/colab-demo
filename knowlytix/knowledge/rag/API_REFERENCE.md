# GEODE graph-RAG — API Reference

Public surface of `knowlytix.knowledge.rag`. For prose and recipes see
[USER_GUIDE.md](USER_GUIDE.md) and [README.md](README.md); for design rationale
see `GEODE_RAG_DESIGN.md` at the repo root. Most names below import from
`knowlytix.knowledge.rag`; the geometric query parser and the geometric relevance
gate (with its calibrator) are *not* re-exported and import from their submodules
(noted per entry). `License-Identifier: Apache-2.0`.

| symbol | role |
| ------ | ---- |
| `RagPipeline`, `RagAnswer`, `RagConfig` | orchestration + configuration |
| `QueryTripleExtractor`, `QueryTriple`, `schema_from_store`, `is_var` | NL → query triples (LLM) |
| `GeometricQueryParser` | NL → query triples (LLM-free, encoder + graph structure) |
| `TripleBinder`, `BoundTriple` | resolve slots to graph vocabulary |
| `RelevanceGate` | question↔attribute relevance gate (LLM) |
| `GeometricRelevanceGate`, `calibrate_relevance_thresholds` | question↔attribute relevance gate (LLM-free, dual-space) + its calibrator |
| `Retriever`, `RetrieveResult`, `RetrievedFact` | GMS retrieval + provenance |
| `Assembler` | grounded synthesis |
| `AnswerVerifier`, `VerifyReport`, `ClaimVerdict` | answer self-verification |
| `coverage_report`, `CoverageReport`, `RegionCoverage` | blind-spot monitor |
| `evaluate`, `EvalReport`, `EvalCase`, `calibrate_bind_threshold` | evaluation |
| `VectorBackend`, `InMemoryVectorBackend`, `DenseSpan`, `build_spans` | dense fallback |
| `NumericUSpaceOrder`, `OrderResult` (`knowlytix.core.geometry.numeric_order`) | u-space numeric ordering behind the `numeric_order` route |

---

## `RagConfig`

```python
@dataclass
class RagConfig:
    llm: LLMBackend                                  # synthesis (required)
    llm_extract: LLMBackend | None = None            # default: llm
    llm_verify: LLMBackend | None = None             # default: llm_extract
    binding: Literal["fuzzy", "embedding"] = "fuzzy"
    encoder: Callable[[list[str]], object] | None = None   # default MiniLM; the v-encoder
    bind_threshold: float = 0.5
    bind_margin: float = 0.05
    top_k: int = 10
    max_hops: int = 5
    max_extract_retries: int = 1
    ground_extraction: bool = True
    query_parse_mode: Literal["llm", "geometric"] = "geometric"
    numeric_order: bool = True
    numeric_order_tie_margin: float = 0.0
    numeric_order_floor: float = 0.25
    verify_llm_output: bool = False
    on_verify_fail: Literal["annotate", "abstain", "regenerate"] = "abstain"
    relevance_gate: bool = True
    relevance_mode: Literal["llm", "geometric"] = "llm"
    relevance_u_encoder: Callable[[list[str]], object] | None = None
    relevance_tau_accept: float = 0.30
    relevance_tau_contra: float = 0.75
    relevance_tau_contra_per_relation: dict = {}
    relevance_prefilter: bool = True
    relevance_prefilter_floor: float = 0.25
    accept_threshold: float = 0.0
    dense_fallback: bool = False
    vector_backend: object | None = None             # VectorBackend
    strict_mode: bool = False
    audit_sink: Callable[[dict], None] | None = None
```

See USER_GUIDE §11 for field effects. Helper methods `extract_llm()` and
`verify_llm()` resolve the per-role LLM defaults.

#### Query-parse and relevance fields

| field | type | default | meaning |
| ----- | ---- | ------- | ------- |
| `query_parse_mode` | `Literal["llm", "geometric"]` | `"geometric"` | How a question becomes query triples. `"geometric"` (default) uses `GeometricQueryParser` (tuned-encoder + graph structure, no parse-time LLM); it prefers `encoder` set to the document-tuned v-encoder, falls back to MiniLM, and is robust where a small model emits attribute-as-head or generic relations. `"llm"` uses `QueryTripleExtractor` (set it when there is no doc graph to parse against). |
| `relevance_gate` | `bool` | `True` | Run the post-binding question↔attribute relevance gate at all. |
| `relevance_mode` | `Literal["llm", "geometric"]` | `"llm"` | Which gate. `"llm"` uses `RelevanceGate` (a 3B judge over the head's relations). `"geometric"` uses `GeometricRelevanceGate` (LLM-free dual-space); it needs a v-encoder (`encoder`) and a contradiction-tuned u-encoder (`relevance_u_encoder`). |
| `relevance_u_encoder` | `Callable[[list[str]], object] \| None` | `None` | The `full`-mode contradiction-tuned u-encoder used by the geometric gate's veto. When `None` the geometric gate runs accept-only (no u-veto). |
| `relevance_tau_accept` | `float` | `0.30` | Geometric gate: min v-cosine to accept the nearest relation. |
| `relevance_tau_contra` | `float` | `0.75` | Geometric gate: max u-tension to the accepted relation before the veto fires (per-relation overrides apply). |
| `relevance_tau_contra_per_relation` | `dict[str, float]` | `{}` | Geometric gate: per-relation overrides of `relevance_tau_contra`. |
| `relevance_prefilter` / `relevance_prefilter_floor` | `bool` / `float` | `True` / `0.25` | LLM-gate (`RelevanceGate`) latency prefilter: abstain immediately when the question's max cosine to every head relation is below the floor. Not used by the geometric gate. |

`encoder` is the document-tuned **v-encoder**, shared by embedding binding, the
geometric query parser, and the geometric relevance gate's accept step.

#### Numeric-order fields

| field | type | default | meaning |
| ----- | ---- | ------- | ------- |
| `numeric_order` | `bool` | `True` | Enable the numeric-order route: a superlative/comparison question (which names no anchor entity, so the triple route cannot bind it) is answered by ranking the matched ENM type through u-space directed entailment over the exact values. Tried before extraction in `query()`; produces `route="numeric_order"`. |
| `numeric_order_tie_margin` | `float` | `0.0` | The route abstains only when the winner's u-space energy gap to the nearest *distinct* value is `<=` this. The default `0.0` abstains only in the degenerate all-equal case (no distinct runner-up); the exact ENM values already separate a near-but-distinct second place, so it is not a tie. |
| `numeric_order_floor` | `float` | `0.25` | Minimum cosine of the question to an ENM-type phrase to route there (and reused as the floor for the follow-on relation match). Needs `encoder`; without one the type name must appear in the question as a substring. |

The numeric-order route needs at least one rankable ENM type (a numeric type with
≥ 2 scalar entries); the cosine matching uses `encoder` (the v-encoder).

---

## `RagPipeline`

Triple-mediated RAG over a `GMSExpertStore`.

```python
RagPipeline(store, rag: RagConfig, *, ledger: ProvenanceLedger | None = None)
```

The ledger defaults to one derived from `store.markdown`. The constructor wires
the extractor (grounded with `schema_from_store` when `ground_extraction`), the
binder, the retriever, the assembler, the optional verifier, and the optional
dense backend.

### Constructors

| classmethod | what |
| ----------- | ---- |
| `RagPipeline.from_store(store, rag) -> RagPipeline` | wrap an existing store |
| `RagPipeline.build(md_path, config, rag, *, device=None, **build_kwargs) -> RagPipeline` | run GEODE self-correction + train a store, then wrap it. `build_kwargs` forward to `knowlytix.knowledge.geode.rag.build_rag_store` (e.g. `llm`, `max_iters`). |

### Methods

#### `query(question: str) -> RagAnswer`
Run the full online flow: extract → bind → bind-check → relevance → retrieve →
synthesize → (optional) self-verify → decide. Abstains (rather than guesses) when
nothing binds, no fact matches, or verification fails under
`on_verify_fail="abstain"`. Emits the audit record to `audit_sink` if configured.

A **numeric-order route** is tried *first*, before extraction (when
`RagConfig.numeric_order` is set and the store has at least one rankable ENM type).
A superlative question (cue such as `highest`/`lowest`/`largest`/`smallest`/
`most`/`least`) names no anchor entity, so it is answered by matching the target
ENM type (encoder cosine, or substring fallback) and ranking it with
`store.enm_extreme`, then chaining a follow-on relation lookup for the compound
answer. Such answers carry `route="numeric_order"`, `verified=True`, an `enm`
source with the exact value, and (when the runner-up is not distinct) abstain. When
the question is not a superlative or no ENM type matches, `query()` falls through to
the triple route. The relevant internals are `_try_numeric_order`, `_which_extreme`,
`_match_enm_type`, and `_followon_relation`.

#### `coverage() -> CoverageReport`
Coverage report for the store — measurable triple-mediation blind spots.

---

## `RagAnswer`

```python
@dataclass
class RagAnswer:
    answer: str
    confidence: float
    decision: str            # "accept" | "abstain" | "escalate"
    route: str               # "triple" | "numeric_order" | "dense_fallback"
    verified: bool
    query_triples: list[QueryTriple] = []
    bound_triples: list[BoundTriple] = []
    sources: list[RetrievedFact] = []
    dense_sources: list[DenseSpan] = []
    verification: dict = {}
    notice: str | None = None
```

#### `audit() -> dict`
Structured audit record: `decision`, `route`, `verified`, `confidence`,
`query_triples`, `bound_triples`, `sources` (triple + source + score + location),
`verification`, `dense_sources` (locations), and `notice`.

---

## Query triples — `query_triples`

### `QueryTriple` (dataclass)
`head`, `relation`, `tail` — a pattern with `?`-prefixed unknown slots.
`as_tuple() -> (head, relation, tail)`.

### `is_var(slot: str) -> bool`
True for a variable slot (`?`, `?x`, …).

### `schema_from_store(store, *, max_entities=80, exclude_relations=("in_section",)) -> dict`
Extract the graph vocabulary (`{"relations": [...], "entities": [...]}`) to ground
extraction. Numeric value-entities are filtered out — they are answers, not query
terms. Showing the LLM real relation names is the biggest reliability lever.

### `QueryTripleExtractor`
```python
QueryTripleExtractor(llm, *, vocab: dict | None = None, max_tokens: int = ...)
extractor.extract(question: str, *, hint: str | None = None) -> list[QueryTriple]
```
Translate a question into query triples; the asked slot is the bare string `"?"`.
`hint` (used by the pipeline's repair retry) appends a note naming failed terms
and legal relations. `QueryTripleExtractor.parse(raw)` parses the LLM's JSON array.

### `GeometricQueryParser`
Imports from `knowlytix.knowledge.rag.query_triples` (not re-exported from the
package).
```python
GeometricQueryParser(store, encoder=None, *,
                     exclude_relations: tuple[str, ...] = ("in_section", "has_alias",
                                                           "has_product"),
                     top_rel: int = 3, alias_relation: str = "has_alias",
                     max_hops: int = 2)
parser.extract(question: str, *, hint: str | None = None) -> list[QueryTriple]
```
LLM-free NL → query-triple parser: matches the question to the graph's real
relation phrases by cosine in the document-tuned `encoder` space (top `top_rel`
relations), then picks the head among the entities that actually *assert* that
relation, scored by the question's similarity to each entity's name and aliases.
Returns the best asserted path as a `?x`-chained `list[QueryTriple]` (single-hop is
one `QueryTriple(head, relation, "?")`; `[]` when the graph has no content
relations); `max_hops` bounds the path search. `encoder` defaults to the repo MiniLM `encode_texts`; pass the
document-tuned v-encoder. Reads graph structure from `store.doc_graph.triples`;
`alias_relation` edges populate per-entity aliases and `exclude_relations` are
dropped from the relation vocabulary. `extract` mirrors `QueryTripleExtractor` (the
`hint` argument is accepted and ignored), so it is a drop-in.

---

## Binding — `binding`

### `BoundTriple` (dataclass)
`original: QueryTriple`, `head/relation/tail: str | None`. A `None` slot is a
known (non-variable) term that *failed* to bind. Property `bound: bool` is True
only when every non-variable slot resolved.

### `TripleBinder`
```python
TripleBinder(store, *, mode="fuzzy", encoder=None,
             bind_threshold=0.5, bind_margin=0.05)
binder.bind(qt: QueryTriple) -> BoundTriple
```
`mode` is `"fuzzy"` (string) or `"embedding"`; any other mode raises
`NotImplementedError`. In `"embedding"` mode the tuned `encoder` is **primary**:

- **entities** — the encoder's high-confidence cosine match is taken first (it
  maps a surface head like `overdraft fee policy` onto the canonical entity
  `overdraft` even without string overlap); `store.fuzzy_match_entity` is the
  fallback only when the encoder abstains (low score or top1−top2 tie).
- **relations** — a *literal* relation name (exact, slugged, or `has_<slug>` in
  the adapter vocab) is taken directly via fuzzy match; otherwise the encoder is
  primary with fuzzy as the fallback.

The encoder match still enforces `bind_threshold` (min cosine) and `bind_margin`
(min top1−top2 gap, else refuse as ambiguous). `"fuzzy"` mode uses the string
matcher only.

---

## Relevance — `relevance`

Post-binding question↔attribute gate: a bound relation can be graph-valid yet not
the attribute the question asked about (e.g. an interest-rate question binding to
`has_fee_amount`). Wired by the pipeline when `RagConfig.relevance_gate` is set;
`relevance_mode` selects the gate. `RelevanceGate` is re-exported from the package;
`GeometricRelevanceGate` and `calibrate_relevance_thresholds` import from
`knowlytix.knowledge.rag.relevance`.

### `RelevanceGate` (LLM)
```python
RelevanceGate(store, llm, *, encoder=None, prefilter=True,
              prefilter_floor=0.25, max_tokens=16,
              exclude_relations=("in_section",))
gate.relevant_relation(question: str, head: str) -> tuple[str | None, str]
gate.head_relations(head: str) -> list[str]
```
A 3B judge picks the one matching attribute from the head's real relations (or
`NONE`). `relevant_relation` returns `(matched_relation, detail)`; `None` abstains.
The cosine `prefilter` (floor `prefilter_floor`) cheaply abstains when the question
is unrelated to every head relation, otherwise the borderline call goes to the LLM.

### `GeometricRelevanceGate` (LLM-free, dual-space)
```python
GeometricRelevanceGate(store, v_encoder=None, u_encoder=None, *,
                       tau_accept: float = 0.30, tau_contra: float = 0.75,
                       tau_contra_per_relation: dict[str, float] | None = None,
                       exclude_relations: tuple[str, ...] = ("in_section",))
gate.relevant_relation(question: str, head: str) -> tuple[str | None, str]
gate.head_relations(head: str) -> list[str]
```
v proposes, u vetoes. **Accept (v-space):** the nearest of the head's relations to
the question by cosine; below `tau_accept` nothing is close enough → abstain.
**Veto (u-space):** symmetric tension `2·sin(θ/2)` from the question to the
accepted relation in the contradiction-tuned u-space; above the relation's cut
(`tau_contra_per_relation.get(rel, tau_contra)`) the asked attribute *contradicts*
the relation → abstain. Returns `(matched_relation, detail)`; `None` abstains. Both
encoders default to the repo MiniLM `encode_texts`; pass the GEODE embed-loop
v-encoder and `full`-mode contradiction-tuned u-encoder. When `u_encoder is None`
the veto is skipped (accept-only).

### `calibrate_relevance_thresholds(relation_phrasings, v_encoder, u_encoder, *, accept_margin=0.05, contra_margin=0.02) -> dict`
Fit the geometric gate's operating points from build-time phrasings — no constants.
`relation_phrasings` is `{relation: [phrasing, ...]}`. Returns
`{"tau_accept", "default_tau_contra", "tau_contra_per_relation", "overlap"}`:
- `tau_accept` — global v-accept floor: the lowest v-cosine any genuine phrasing
  has to its own relation phrase, minus `accept_margin` (floored at `0.05`).
- `tau_contra_per_relation` — per-relation recall-first u-veto cut:
  `max(consistent-tension) + contra_margin`, so a valid attribute is never vetoed.
- `default_tau_contra` — median of the per-relation cuts.
- `overlap` — `{relation: min(contradictory-tension)}` where the contradictory band
  overlaps the cut (the residual leak reported instead of sacrificing recall).

Feed the result into `RagConfig.relevance_tau_accept`, `relevance_tau_contra`
(`default_tau_contra`), and `relevance_tau_contra_per_relation`.

---

## Retrieval — `retrieve`

### `RetrievedFact` (dataclass)
`head`, `relation`, `tail`, `score` (geodesic distance; `0` == exact/asserted),
`confidence`, `source` (`"link_predict" | "triple" | "score" | "enm"`),
`location` (`file:line:char`), `raw` (source span text).

### `RetrieveResult` (dataclass)
`facts: list[RetrievedFact]`, `answers: list[(value, conf)]`. Property
`matched: bool`.

### `Retriever`
```python
Retriever(store, ledger, *, top_k=10)
retriever.retrieve(bound: list[BoundTriple]) -> RetrieveResult
```
GMS retrieval over bound triples (asserted edges preferred over predicted),
attaching provenance from the ledger.

---

## Synthesis — `assemble`

### `Assembler`
```python
Assembler(llm)
assembler.assemble(question, facts: list[RetrievedFact]) -> str       # grounded in facts + spans
assembler.assemble_passages(question, passages: list[str]) -> str     # dense-fallback path
```

---

## Verification — `verify`

### `ClaimVerdict` (dataclass)
`triple: QueryTriple`, `status` (`supported | contradicted | implausible |
unverifiable`), `detail`. Property `failed` is True for `contradicted`/`implausible`.

### `VerifyReport` (dataclass)
`verdicts: list[ClaimVerdict]`. Properties `ok` (no failures), `failures`. Method
`as_dict()` → `{ok, claims, failed}`.

### `AnswerVerifier`
```python
AnswerVerifier(store, llm, *, binder=None, max_tokens=512, mode="llm")
verifier.verify(answer: str) -> VerifyReport
```
Extract the answer's claim triples and check each against the GMS — the
hallucination detector behind `verify_llm_output`. `mode="llm"` (default) parses
the claims with the LLM; `mode="geometric"` extracts them by walking the store
graph (every real edge whose head/alias and tail the answer states), so a correct
answer yields store-bound triples that verify instead of LLM-invented relations
that false-fail. `_check` is identical in both modes.

---

## Coverage — `coverage`

### `RegionCoverage` (dataclass)
`title`, `line_start`, `line_end`, `body_lines`, `triple_count`. Properties
`covered` (`triple_count > 0`) and `blind_spot` (has body text, no triples).

### `CoverageReport` (dataclass)
`regions: list[RegionCoverage]`, `unaligned_triples: int`. Properties
`blind_spots` and `coverage_ratio` (fraction of content regions with ≥1 triple);
`as_dict()`.

### `coverage_report(store, *, exclude_relations=("in_section",)) -> CoverageReport`
Walk the document's header regions and count triples whose provenance lands in
each.

### `GraphCoverage` (dataclass)
`n_entities`, `n_subject_entities`, `n_fact_entities`, `orphan_entities`,
`relation_fact_counts`, `singleton_relations`, `declared_relations`,
`unpopulated_declared`. Property `orphan_ratio`; `as_dict()`.

### `graph_coverage(triples) -> GraphCoverage`
Entity/relation completeness diagnostics: **orphan entities** (a subject reachable
only through structural edges), **singleton relations** (one fact edge — a
candidate for under-extraction) and **declared-but-unpopulated relations** (a
relation declared `is_functional` that no fact edge uses). Pass the **full**
pre-filter graph (the loop's triples or `ingest_markdown(...).triples`); the served
store strips the `in_section`/`is_functional` edges these signals depend on.

---

## Evaluation — `eval`

### `EvalCase` (dataclass)
`question`, `expected_answer: str | None` (substring/value expected), 
`expect_decision: str | None` (`"accept" | "abstain"`).

### `EvalReport` (dataclass)
`results: list[CaseResult]`. Properties `bind_rate`, `accept_rate`,
`answer_accuracy`, `decision_accuracy`; `as_dict()`.

### `evaluate(pipeline, cases: list[EvalCase]) -> EvalReport`
Run each case through the pipeline and score binding, decision, and answer
correctness.

### `calibrate_bind_threshold(binder, positives, negatives, *, grid=...) -> (float, float)`
Sweep `binder.bind_threshold` to best separate `positives` (`(term,
expected_entity)`) from `negatives` (terms that should not bind). Returns
`(best_threshold, accuracy)` and sets it on the binder. Requires an
embedding-mode binder.

### `calibrate_accept_threshold(records, *, max_false_accept=0.05) -> dict`
Fit the accept-gate `tau` on a labeled accept/abstain cohort. Each record is
`(label, abstained, confidence)` — `label` 1 for answerable (ACCEPT) else 0
(ABSTAIN), `abstained` whether a pre-threshold gate already abstained,
`confidence` the pipeline's retrieval confidence (collect with
`accept_threshold=0` so the gate does not pre-filter). Maximizes balanced accuracy
`0.5*(recall + (1 - false_accept))` under a `max_false_accept` ceiling (a
tie-breaker, not a hard constraint), with a midpoint refinement for a clean
bimodal split. Returns a payload ready to persist as `rag_gate_calibration.json`
(`accept_threshold`, `balanced_accuracy`, `recall`, `false_accept`,
`false_accept_ceiling_met`, Wilson `accuracy_ci`, cohort counts).

### `benchmark_query_parse(store, parsers, *, binder=None, n=0) -> dict`
Compare query parsers on the parse→bind step with the binder held constant.
Reverses one templated, gold-anchored question per distinct `(head, relation)`
fact (`in_section` excluded), runs each `parsers[name].extract(question)`, binds
with a shared embedding `TripleBinder`, and scores recovery of the asked
`(head, relation)`. Returns `{n_questions, results: {name: {parse_rate, bind_rate,
recover_rate, latency_ms_mean, latency_ms_p50, examples}}}` — geometric and LLM
parsers measured on the same questions.

---

## Label classification — `label_classifier`

A standalone geometric classifier, not a step in the query pipeline.

### `GeometricLabelClassifier`
```python
GeometricLabelClassifier(exemplars: dict[str, list[str]], encoder=None, threshold=0.0)
clf.classify(text: str, *, abstain_label: str) -> tuple[str, float]
clf.calibrate(examples: list[tuple[str, str]], *, abstain_label: str,
              false_accept_ceiling: float = 0.10) -> float
```
Closed-vocabulary classifier: score a text against each label's exemplars by max
cosine in the `encoder` space (defaults to MiniLM `encode_texts`; pass the
document-tuned v-encoder to match the GMS space) and return the nearest label, or
`abstain_label` when the top score is below `threshold`. Same refuse-over-fabricate
stance as `GeometricQueryParser` — no LLM, no off-taxonomy strings. `calibrate`
fits `threshold` on a labeled cohort to the smallest cut honoring a *false-accept
ceiling* (committing to a concrete label when the truth is the abstain class), so
recall on concrete labels is preserved; persist and reload the fitted threshold
like the other gates.

---

## Dense fallback — `dense`

Opt-in, distrusted. Off unless `RagConfig.dense_fallback=True`.

### `DenseSpan` (dataclass)
`text`, `location` (`source_path:line_start-line_end`), `line_start`, `line_end`.

### `build_spans(markdown: str, *, source_path="") -> list[DenseSpan]`
Chunk markdown into blank-line-separated paragraph spans with line-range
provenance (header-only blocks skipped).

### `VectorBackend` (ABC)
```python
backend.index(spans: list[DenseSpan]) -> None
backend.search(query: str, top_k: int) -> list[tuple[DenseSpan, float]]
```

### `InMemoryVectorBackend(VectorBackend)`
Default in-memory cosine backend (`InMemoryVectorBackend(encoder=None)` uses
MiniLM). Pass a custom `VectorBackend` via `RagConfig.vector_backend` to swap in
an external index.

---

## Numeric ordering — `GMSExpertStore` ENM methods

The numeric-order route is backed by four methods on `GMSExpertStore`
(`knowlytix.knowledge.store`, *not* part of the `rag` package). They aggregate the
exact ENM scalars of a type and rank/compare them through u-space directed
entailment (see *Numeric ordering — core geometry* below). All require the store's
`enm` to be present; the rankers fall back to the model's u-space dimension (else
`32`). `which` is `"highest"` or `"lowest"`.

### `enm_by_type(category: str) -> list[tuple[str, float]]`
All scalar ENM entries `(id, value)` of type `category` (non-scalar ENM entries are
skipped). Returns `[]` when there is no ENM or no entry of that type.

### `enm_rank(category: str, which: str = "highest") -> list[tuple[str, float]]`
The entities of `category` as `(id, value)` ordered from most to least extreme in
the `which` direction, via `NumericUSpaceOrder.rank`. `[]` when the type is empty.

### `enm_extreme(category: str, which: str = "highest") -> OrderResult | None`
The argmax (`"highest"`) / argmin (`"lowest"`) entity of `category` as an
`OrderResult` (`entity`, `value`, `margin`, `relevance`), via
`NumericUSpaceOrder.extreme`. `None` when the type is empty; `margin == 0.0` when
the extremum is tied (no distinct runner-up).

### `enm_compare(category: str, id_a: str, id_b: str) -> OrderResult | None`
Which of `id_a` / `id_b` (both of type `category`) is higher, as an `OrderResult`
whose `entity` is the higher-valued id, via `NumericUSpaceOrder.compare`. `None`
when the type is empty or an id is not present.

---

## Numeric ordering — core geometry

`knowlytix.core.geometry.numeric_order` (`__all__ = ["NumericUSpaceOrder",
"OrderResult", "DEFAULT_LAMBDA_SPEC"]`). The query-time counterpart of
`PhaseEncoder.check_inequality`: thresholds compare one value to a limit, ordering
compares values to each other. Each ENM scalar is encoded onto a great circle of
the u-sphere via `PhaseEncoder` (`theta(v) = pi * (v - v_min) / (v_max - v_min)`);
the *specificity grade* is that phase angle, and ordering is decided by the
directed-entailment energy `E_->(i, j) = E_ij + lambda * [g(j) - g(i)]_+`. The
symmetric tension `E_ij` cancels in the net cost (so ordering is driven purely by
grade), while the residual tension supplies the **margin** in the same units as the
GMS contradiction/entailment gates — so a superlative or comparison is decided by
the same operator and thresholds as a logical-entailment edge.

### `OrderResult`
```python
OrderResult(entity: str, value: float, margin: float, relevance: float)
```
An extremum/comparison decision (`__slots__`, no dataclass). `entity` — the winning
id; `value` — its exact ENM value; `margin` — the directed-entailment energy gap
`lambda * |theta_a - theta_b|` to the runner-up (extreme) or the other operand
(compare), gateable by a calibrated tension threshold (`0.0` == tied); `relevance` —
the symmetric tension `E_ij` to that operand (how far apart the two values sit on
the circle).

### `NumericUSpaceOrder`
```python
NumericUSpaceOrder(values, *, lambda_spec: float = DEFAULT_LAMBDA_SPEC,
                   dim: int = 32, plane: tuple[int, int] = (0, 1))
```
`values` is a `{id: value}` dict or a list of `(id, value)` pairs. `lambda_spec` is
the specificity-penalty weight of the directed entailment (`DEFAULT_LAMBDA_SPEC =
2.0`); `dim` is the ambient embedding dim (the phase circle is 2D — any `dim >= 2`
works; pass the model's u-space dim so the encoded vectors share that space);
`plane` selects the two coordinates carrying the phase circle.

| method | returns | what |
| ------ | ------- | ---- |
| `rank(which="highest")` | `list[tuple[str, float]]` | entities `(id, value)` from most to least extreme; `[]` when empty |
| `extreme(which="highest")` | `OrderResult \| None` | the argmax/argmin; `margin` is the gap to the nearest *distinct* value (`0.0` when tied) |
| `compare(id_a, id_b)` | `OrderResult \| None` | the higher of the two with its margin; `None` if an id is absent |

```python
from knowlytix.core.geometry.numeric_order import NumericUSpaceOrder

# the ENM fee scalars behind a "which fee is highest?" superlative query
order = NumericUSpaceOrder({"overdraft": 35.0, "wire": 25.0, "atm": 3.0})
winner = order.extreme("highest")
print(winner.entity, winner.value)   # overdraft 35.0
print(winner.margin)                 # > 0: directed-entailment gap to "wire";
                                     # gate it against the GMS confidence floor
order.rank("lowest")                 # [("atm", 3.0), ("wire", 25.0), ("overdraft", 35.0)]
```

---

## Admissibility filtering — core geometry

`knowlytix.core.graph.admissibility`. Two orthogonal, **calibrated** gates decide
whether a tail `t` is admissible for `(h, r)` — used at ingestion to keep predicted
edges honest and at query time to filter retrieval candidates. **Semantic (v-cap):**
is `t` inside the head-conditioned spherical cap `rho_{h,r}` (with a per-head
margin)? **Logical (u-tension):** does `t` *contradict* the tails the head already
asserts under `r`? Both thresholds are fit against a label-free negative cloud at a
target false-positive rate — *not* fixed constants. The tension gate **recuses
itself** (returns `2.0`, inert) when the positive and negative clouds do not
separate, deferring to the cap; this is the safety rule that stops a mis-calibrated
tension cut from arbitrarily slicing legitimate one-to-many tails. Heads/relations
are integer-encoded as in the model adapter (`h`, `r`, `t` are vocab ids).

### `calibrate_cap_margins_per_head(model, adapter, *, fpr_target=0.05, n_neg=24, min_neg=5) -> dict[tuple[int, int], float]`
Per-`(head, relation)` SEMANTIC v-cap margin on `Delta = cap_distance - rho_{h,r}`.
`m_{h,r}` is the `fpr_target`-quantile of the head's *own* hard-negative cloud (its
`n_neg` nearest non-positive tails). Per-head, so a narrow head does not inherit a
loose boundary from a wide one. A head with fewer than `min_neg` negatives falls
back to `0.0` (the raw head-conditioned radius).

### `calibrate_tension_threshold(model, adapter, *, fpr_target=0.05, min_neg=5, min_gap=0.15) -> dict[int, float]`
Per-relation LOGICAL u-tension cutoff `tau_r` for the contradiction gate, from two
clouds: cross-head tails (should read high) and each established tail vs its head's
other tails (should read low). `tau_r` is the `fpr_target`-quantile of the negative
cloud **only if** the clouds separate (`hi_pos + min_gap <= lo_neg`); otherwise — `u`
carries no usable contradiction signal for this relation — it returns `2.0` and the
gate goes **inert**, deferring entirely to the cap.

### `admissible(model, adapter, h, r, t, *, cap_margins, tension_tau) -> dict`
Layered admissibility for one triple. `cap_margins` and `tension_tau` are the dicts
from the two calibrators (looked up with safe fallbacks of `0.0` / `2.0`). Returns:

| key | type | meaning |
| --- | ---- | ------- |
| `admissible` | `bool` | `cap_ok and tension_ok` |
| `cap_ok` | `bool` | tail inside the head-conditioned cap within its margin |
| `tension_ok` | `bool` | tail does not contradict the head's established tails |
| `cap_distance` | `float` | cap distance `d` of `t` under `(h, r)` |
| `tension` | `float` | max contradiction tension of `t` vs established tails (`0.0` when none) |

```python
from knowlytix.core.graph.admissibility import (
    calibrate_cap_margins_per_head, calibrate_tension_threshold, admissible)

# calibrate once at ingestion against the trained model + adapter
cap_margins = calibrate_cap_margins_per_head(model, adapter)
tension_tau = calibrate_tension_threshold(model, adapter)

# gate each candidate (h, r, t) at query time
verdict = admissible(model, adapter, h, r, t,
                     cap_margins=cap_margins, tension_tau=tension_tau)
if verdict["admissible"]:
    keep(h, r, t)        # passed both the semantic cap and the logical tension gate
```

### `filter_admissible_facts(store, facts, *, cap_margins=None, tension_tau=None, fpr_target=0.05, use_tension=True) -> tuple[list, list[dict]]`

RAG-level wrapper over the core primitives above, exported from
`knowlytix.knowledge.rag`. Splits retrieved facts into `(kept, dropped)` by
cap (+tension) admissibility. `facts` is any iterable of objects with
`head`/`relation`/`tail` attributes (e.g. `RetrievedFact`). Calibrations are
computed on first use when `None`; pass them back in to amortize across queries.
`use_tension=True` AND-s the contradiction-tension gate with the cap (inert
unless `u` carries contradiction structure, so it is safe to leave on).

Two safety rules: a fact is dropped only when **scorable and inadmissible** — a
fact whose head/relation/tail is outside the model vocab is **kept** (never
dropped on missing signal) — and when `model.cap_enabled` is false the filter is
a **no-op** (returns `(facts, [])`). `dropped` is a list of audit dicts (never
silent):

| key | type | meaning |
| --- | ---- | ------- |
| `fact` | object | the dropped fact (the original `head`/`relation`/`tail` object) |
| `cap_ok` | `bool` | passed the semantic v-cap |
| `tension_ok` | `bool` | passed the logical u-tension gate |
| `cap_distance` | `float` | cap distance under `(head, relation)` |
| `tension` | `float` | max contradiction tension vs established tails |

Caller-invoked by design — it does not edit the retriever, so it composes
without collision.

```python
from knowlytix.knowledge.rag import filter_admissible_facts

kept, dropped = filter_admissible_facts(store, retrieved_facts, fpr_target=0.05)
```

---

## KAL persistence — `kal_sink`

Persist the GEODE-corrected graph into KAL (Postgres/pgvector). **Opt-in extra**:
imports `knowlytix.kal` and is *not* re-exported from the package — import from
`knowlytix.knowledge.rag.kal_sink` directly. Triples (with provenance and
optional verification) go to KAL; numeric tails become `object_literal`s; the GMS
checkpoint is out of scope.

### `store_to_kal_triples(store, *, source=None, extractor="geode", ledger=None, exclude_relations=("in_section",), extracted_at=None, confidence=None) -> list[KALTriple]`
Convert a built store's triples into `KALTriple` records. Source-span text is
recovered from the store's markdown via `ProvenanceLedger` (built automatically
when `ledger` is `None`). Pass `confidence` to stamp `VerificationMetadata`
(`verification_status="verified"`). Numeric tails (`_NUM_RE`) become
`KALLiteral(datatype="number")`; entity tails become `KALNode`s.

### `async persist_store_to_kal(adapter, store, *, tenant_id=None, **kwargs) -> int`
Write the store's triples through a KAL `KnowledgeAdapter`; returns the number
inserted. `kwargs` forward to `store_to_kal_triples`.

### `persist_store_to_kal_sync(adapter, store, *, tenant_id=None, **kwargs) -> int`
Synchronous wrapper around `persist_store_to_kal` (`asyncio.run`).

### `kal_postgres_adapter(*, host, database, username, password, port=5432, schema_name="public", name="geode")`
Build a Postgres-backed KAL `KnowledgeAdapter` (operations are async). KAL's
migrations must already be applied to the target database; credentials are
encrypted at rest by KAL.
