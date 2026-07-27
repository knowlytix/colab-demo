# GEODE — API Reference

Public surface of `knowlytix.knowledge.geode`. For prose and recipes see
[USER_GUIDE.md](USER_GUIDE.md); for design and validation see
`EXTRACTION_AGENT_DESIGN.md` at the repo root.

Unless noted otherwise, the names below are importable from
`knowlytix.knowledge.geode`. The fact-only filter and the embedding loop are
**not** re-exported from the package root — import them from their submodules
(`...geode.rag` and `...geode.embed_loop`); the import path is given in each entry.

| symbol | role | import from |
| ------ | ---- | ----------- |
| `build_rag_store`, `RagBuildResult`, `store_from_triples` | one-call store build with self-correction | `...geode` |
| `fact_only_triples`, `DEFAULT_NOISE_RELATIONS` | strip non-fact relations before training | `...geode.rag` |
| `GeodeLoop`, `LoopResult`, `make_default_trainer` | the orchestrator | `...geode` |
| `CompositionCritic`, `Flag` | geometric critic | `...geode` |
| `AnchorChecker`, `AnchorViolation`, `SumConstraint`, `enm_from_triples`, `numeric_facts_from_triples` | external anchors | `...geode` |
| `ProvenanceLedger`, `Provenance`, `is_consistent` | span ↔ triple provenance | `...geode` |
| `GeodeEmbedLoop`, `EmbedLoopConfig`, `EmbedLoopResult`, `graph_entity_labels`, `geometry_entity_links`, `contradiction_sft` | document-tuned encoder SFT | `...geode.embed_loop` |
| `qwen_agent_callable`, `QWEN_3B` | the (only) actor LLM | `...geode` |

`License-Identifier: Apache-2.0`.

---

## High level — `build_rag_store`

```python
def build_rag_store(
    md_path: str, config, *, device=None,
    geode_trainer: Trainer | None = None,
    llm=None, max_iters: int = 8,
    residual_threshold: float | None = None,
) -> RagBuildResult
```

Run GEODE self-correction over `md_path`, then build a production RAG store from the corrected triples.

| arg | meaning |
| --- | ------- |
| `md_path` | source markdown document |
| `config` | `knowlytix.knowledge.config.DocGMSConfig` for the production store (geometry, training, `store_path`, `loss_mode`) |
| `device` | torch device (defaults to cuda if available) |
| `geode_trainer` | trainer for the *correction* loop; defaults to `make_default_trainer`. Distinct from the production training inside `store_from_triples`. |
| `llm` | optional Qwen-3B actor `Callable[[str], str]`; geometry stays authoritative |
| `max_iters` | loop iteration bound |
| `residual_threshold` | critic flag threshold; `None` calibrates |

Returns `RagBuildResult`.

### `RagBuildResult` (dataclass)

```python
@dataclass
class RagBuildResult:
    store: object              # GMSExpertStore
    converged: bool
    iterations: int
    n_entities: int
    n_triples: int
    n_enm: int
    corrections: list[dict] = []
    anchor_violations: list[dict] = []
```

### `store_from_triples`

```python
def store_from_triples(
    md_path: str, triples: list[Triple], config, device=None,
    *, drop_relations=DEFAULT_NOISE_RELATIONS,
) -> GMSExpertStore
```

Build and train a production store from an explicit triple set (the corrected graph). Numeric ENM and phase encoders come from a fresh deterministic regex ingest of the same file; only the triple set is swapped for the corrected one.

| arg | type | default | meaning |
| --- | ---- | ------- | ------- |
| `md_path` | `str` | — | source markdown document |
| `triples` | `list[Triple]` | — | corrected triple set to train on |
| `config` | `DocGMSConfig` | — | production store config (geometry, training, `store_path`, `loss_mode`, `cap`, `embedding`) |
| `device` | torch device | `None` (cuda if available) | training device |
| `drop_relations` | iterable of `str` | `DEFAULT_NOISE_RELATIONS` | relations stripped via `fact_only_triples` before training; pass `()` to keep everything |

Behavior changes:
- **Fact-only by default.** Triples are first passed through `fact_only_triples(triples, drop_relations=drop_relations)` so query binding cannot land on a fact-less section-header entity. `drop_relations=()` restores the pre-filter behavior.
- **Forwards `config.embedding`.** The trainer now receives `getattr(config, "embedding", None)`, so a custom `EmbeddingConfig` (e.g. a `GeodeEmbedLoop` v-vector warm-start via Mode B) takes effect on the GEODE build path. Previously dropped — `config.embedding` was a silent no-op here.

### `build_calibrated_rag_store`

```python
def build_calibrated_rag_store(
    md_path, config, *, device=None, llm=None, geode_trainer=None,
    max_iters=8, residual_threshold=None, canonicalize=False,
    embed_sft=True, embed_sft_config=None, embed_trainer_epochs=150,
    embed_max_iters=2, pool=None, relation_phraser=None,
    contradiction_epochs=400) -> RagBuildResult
```

The full GEODE build. `build_rag_store` does the minimum (loop → store); this also
tunes both encoders, calibrates the relevance gate and writes the coverage report,
so one call yields a complete, encoder-tuned, calibrated store. Steps, writing
artifacts into `config.store_path`: (1) self-correct (`GeodeLoop`, optional
`canonicalize`); (2) tune the v-encoder (`GeodeEmbedLoop`) → `v_emb.pt` (Mode-B
warm-start, set on `config.embedding.v_vectors_path`) + `tuned_encoder/`; (3) tune
the u-encoder (`contradiction_sft` on `relation_phraser(rel)` phrasings, default the
relation phrase) → `contradiction_encoder/`; (4) `calibrate_relevance_thresholds` →
`relevance_calibration.json`; (5) train the production store (picks up the Mode-B
vectors); (6) coverage → `coverage_report.json`. Returns a `RagBuildResult` whose
`artifacts` dict lists the written paths. `embed_sft=False` reduces it to
`build_rag_store`. The accept-gate / judge calibration is separate (it needs a
labeled cohort, not just the document): see `rag.eval.calibrate_accept_threshold`.

---

## Fact-only filtering — `fact_only_triples`, `DEFAULT_NOISE_RELATIONS`

Import from `knowlytix.knowledge.geode.rag` (not re-exported from the package root).

```python
DEFAULT_NOISE_RELATIONS: frozenset[str] = frozenset(
    {"in_section", "is_functional", "fails", "passes", "is_weak", "is_strong"})

def fact_only_triples(
    triples, *, drop_relations=DEFAULT_NOISE_RELATIONS,
) -> list[Triple]
```

`DEFAULT_NOISE_RELATIONS` — relations the regex ingester emits that are not answerable policy facts and pollute query binding if kept:
- `in_section` — organizational; its tail is a fact-less section-header entity a query head can wrongly bind to.
- `is_functional` — a schema (single-valued) declaration, not a fact.
- `fails` / `passes` / `is_weak` / `is_strong` — an opaque boolean-column encoding.

`fact_only_triples(triples, *, drop_relations=DEFAULT_NOISE_RELATIONS) -> list[Triple]` — return the input triples with every triple whose relation is in `drop_relations` removed, so triple-mediated retrieval binds questions only onto typed policy facts. Applied by default inside `store_from_triples`.

| arg | type | default | meaning |
| --- | ---- | ------- | ------- |
| `triples` | iterable of `(head, relation, tail)` | — | triples to filter |
| `drop_relations` | iterable of `str` | `DEFAULT_NOISE_RELATIONS` | relation names to strip |

Returns a `list[Triple]` of the kept (fact-bearing) triples.

---

## Orchestrator — `GeodeLoop`

```python
class GeodeLoop:
    def __init__(self, trainer: Trainer, *, device=None,
                 llm: Callable[[str], str] | None = None,
                 residual_threshold: float | None = None, max_iters: int = 8,
                 check_anchors: bool = True, dedup: bool = True,
                 dedup_fpr: float = 0.05, canonicalize: bool = False,
                 canon_sim: float = 0.86, canon_margin: float = 0.05,
                 canon_contra_tau: float = 1.7, v_encoder=None, u_encoder=None)

    def run(self, md_path: str) -> LoopResult
```

`Trainer = Callable[[list[Triple]], tuple[model, adapter, enm]]`.

| arg | meaning |
| --- | ------- |
| `trainer` | maps a triple list to `(model, adapter, enm)`; use `make_default_trainer`. The `enm` is an `ExactNumericalMemory` the anchors read for integrity-checked exact values. |
| `device` | torch device for the critic (defaults to the model's). |
| `llm` | optional actor; re-derives a disputed value from base relations. Applied only as confidence — geometry's prediction is authoritative. |
| `residual_threshold` | critic flag threshold; `None` self-calibrates. |
| `max_iters` | safety bound (default 8). |
| `check_anchors` | run the external-anchor pass once the composition critic is satisfied (default `True`). |
| `dedup` / `dedup_fpr` | cap+tension duplicate-tail resolution (`resolve_duplicates`) and its false-positive target. |
| `canonicalize` | opt-in entity/relation merge (`canonicalize_graph`) run before the critic so it sees unified vocabulary. Off by default: it rewrites the vocabulary, so validate the proposed merges per corpus first. |
| `canon_sim` / `canon_margin` / `canon_contra_tau` | merge similarity floor, ambiguity margin and contradiction-veto tension. |
| `v_encoder` / `u_encoder` | optional tuned encoders for the relation-merge proposal and contradiction veto; entity merges use the model geometry. |

`run(md_path)`: regex-ingest → iterate (train → [canonicalize] → critic flags → localize via provenance → predict tail → apply → retrain; then dedup) until no flags or `max_iters`, then optionally anchor-check. Returns `LoopResult`.

### `LoopResult` (dataclass)

```python
@dataclass
class LoopResult:
    triples: list[Triple]
    converged: bool
    iterations: int
    corrections: list[dict] = []           # per-fix audit records
    anchor_violations: list[dict] = []     # value errors with no geometric redundancy
    duplicates_removed: list[dict] = []    # spurious tails dropped by cap+tension
    duplicates_flagged: list[dict] = []    # ambiguous multi-tail groups (kept, for review)
    canonicalizations: list[dict] = []     # applied entity/relation merges
    canonicalize_flagged: list[dict] = []  # surfaces too ambiguous to merge
```

Each `corrections` entry: `triple`, `residual`, `location`, `method`, `geometry`, `actor`, `agree`, `applied`. Each `canonicalizations` entry: `kind` (`entity`/`relation`), `canonical`, `merged`, `max_similarity`.

### `make_default_trainer(device=None, *, epochs=300, lambda_path=1.0) -> Trainer`

Returns a trainer that builds + trains the validated cap+path GMS (`GeometryConfig(d_v=64, d_u=64, m=32, d=32)`, `loss_mode="cap"`, `seed=42`) and an ENM from the current triples. Heavy imports (torch, core training) deferred to call time.

---

## Canonicalization — `canonicalize_graph`

```python
canonicalize_graph(triples, *, model=None, adapter=None,
                   v_encoder=None, u_encoder=None,
                   sim_threshold=0.86, margin=0.05, contra_tau=1.7)
    -> (rewritten_triples, merges, flagged)
```

Merge co-referent **entities and relations** so the critic/dedup/anchors (which
match by exact vocabulary) are not defeated by `overdraft fee` / `od fee` and
`has_fee` / `has_fee_amount` appearing as distinct terms.

- **Entities** (needs `model` + `adapter`): propose on v-space proximity
  (`project_v`); **require corroboration** — ≥1 shared attribute whose value
  agrees — and **veto** on any shared attribute that disagrees or on u-space
  tension ≥ `contra_tau`. Numeric values are never candidates; canonical = the
  most-anchored surface. Corroboration is required because v-proximity alone, on
  a small/under-trained GMS, places a fee product next to a regulation.
- **Relations**: propose on phrase similarity (`v_encoder`, else MiniLM); veto on
  a declared `opposite_of`, a functional conflict (a head asserting both with
  different tails) or u-encoder tension.
- Both: mutual-nearest + `margin`; an ambiguous surface (within `margin` of two
  candidates) is excluded from every merge and flagged. Merges are monotonic.

Pass the **pre-filter** graph (the loop's triples), not the served store — the
store strips the structural/schema edges these signals depend on. `merges` and
`flagged` are audit records (`Merge` is a JSON-able `dict` subclass).

---

## Critic — `CompositionCritic`

```python
class CompositionCritic:
    def __init__(self, model, adapter, device, *, residual_threshold: float | None = None)
    def composition_rules(self) -> list[tuple[int, int, int]]
    def residuals(self) -> list[Flag]
    def calibrated_threshold(self, n_neg: int = 8, seed: int = 0, ...) -> float
    def effective_threshold(self) -> float
    def flags(self) -> list[Flag]
    def predict_tail(self, head_idx: int, rule: tuple[int, int, int]) -> str
```

Flags triples that violate a learned relation composition (holonomy / path-consistency residual). `flags()` returns the residuals above `effective_threshold` (the supplied `residual_threshold`, or a label-free calibrated one). `predict_tail(head_idx, rule)` returns the geometry's corrected tail for a flagged triple.

### `Flag` (dataclass)

Carries the flagged `triple`, its `residual`, `head_idx`, and the composition `rule` that fired.

---

## Duplicate resolution — `resolve_duplicates`

```python
resolve_duplicates(model, adapter, triples, ledger=None, *, fpr_target=0.05)
    -> (kept, removed, flagged)
```

The loop stage between the critic and the anchors. For each multi-tail `(head,
relation)` group it drops spurious tails by calibrated admissibility — cap first
(a tail v-far from the head's calibrated cap is a `cap_outlier`), then tension
majority (keep the largest mutually-agreeing cluster, drop dissenters). A
legitimate one-to-many group (all admissible, one agreeing cluster) is kept; an
all-inadmissible or tied group is **flagged for review, never split on a guess**.
`removed` and `flagged` are audit records (with provenance when `ledger` is given);
a removal triggers a retrain in `GeodeLoop` and is monotonic, so the loop converges.

---

## External anchors — `AnchorChecker`

Catches value errors the geometry has no redundancy for (declared constraints, exact-numeric integrity).

```python
class AnchorChecker:
    @classmethod
    def from_enm(cls, enm: ExactNumericalMemory, ...) -> "AnchorChecker"
    def auto_sum_constraints(self, ...) -> list[SumConstraint]
    def check_sum(self, c: SumConstraint, *, rel_tol: float = 1e-3) -> AnchorViolation | None
    def check_duplicates(self, triples) -> list[AnchorViolation]
    def check_all(self, triples, constraints: list[SumConstraint] | None = None) -> list[AnchorViolation]
```

| symbol | role |
| ------ | ---- |
| `SumConstraint` | a declared "these parts sum to this whole" relationship |
| `AnchorViolation` | a detected breach, surfaced for review (never auto-rewritten) |
| `numeric_facts_from_triples(triples) -> dict[Key, float]` | extract numeric facts |
| `enm_from_triples(triples) -> ExactNumericalMemory` | build the exact-numeric register the anchors check against |

---

## Provenance — `ProvenanceLedger`

```python
class ProvenanceLedger:
    @classmethod
    def from_text(cls, text: str, source_path: str = "") -> "ProvenanceLedger"
    def build(self, triples) -> dict[tuple[str, str, str], Provenance]
    def resolve(self, head: str, relation: str, tail: str) -> Provenance
    def line(self, line_no: int) -> str
    def table_block(self, line_no: int) -> str
```

Maps each triple to its exact source span. `resolve(...)` returns a `Provenance` (with `.location()` and `.method`). `is_consistent(p: Provenance) -> bool` checks a provenance record. `canon(s: str) -> str` is the string canonicalizer used for matching.

---

## Encoder SFT loop — `GeodeEmbedLoop`

Import from `knowlytix.knowledge.geode.embed_loop` (not re-exported from the package root). A geometry-supervised, iterative v-space (semantic) encoder fine-tuning loop over a single document: train a GMS, read its entity geometry to extend the document's stated alias/name supervision, low-rank SFT the encoder, optionally pseudo-label an unlabelled pool, repeat to a label fixed point.

```python
class GeodeEmbedLoop:
    def __init__(self, trainer: Callable | None, config: EmbedLoopConfig,
                 *, llm=None)

    def run(self, md_path: str, *, pool: list[str] | None = None,
            eval_pairs: list[tuple[str, str]] | None = None,
            exclude_surfaces: set[str] | None = None) -> EmbedLoopResult
```

| arg | type | default | meaning |
| --- | ---- | ------- | ------- |
| `trainer` | `Callable[[list[Triple]], (model, adapter, enm)]` \| `None` | — | GMS trainer; used only when `config.use_geometry`. Use `make_default_trainer(...)`. |
| `config` | `EmbedLoopConfig` | — | loop knobs (below) |
| `llm` | callable \| `LLMBackend` \| `None` | `None` | actor for `hybrid` / `llm_only` ingest; a bare `str→str` callable or a backend with `.call` |

`run(md_path, *, pool=None, eval_pairs=None, exclude_surfaces=None) -> EmbedLoopResult`:

| arg | type | default | meaning |
| --- | ---- | ------- | ------- |
| `md_path` | `str` | — | source document |
| `pool` | `list[str]` \| `None` | `None` | unlabelled customer-vocabulary surfaces the self-training step may pseudo-label and fold into supervision |
| `eval_pairs` | `list[tuple[str, str]]` \| `None` | `None` | held-out `(surface, canonical)` probes; per-iteration nearest-prototype accuracy is recorded in `history` |
| `exclude_surfaces` | `set[str]` \| `None` | `None` | surface tails dropped from the ingested graph before supervision, so a held-out eval set is absent from both the SFT labels and the geometry (clean generalization test) |

Raises `ValueError` if fewer than 2 canonical entities (heads with alias/name edges) are found.

### `EmbedLoopConfig` (dataclass)

```python
@dataclass
class EmbedLoopConfig:
    sft: object                  # EmbeddingSFTConfig (required)
    max_iters: int = 4
    use_geometry: bool = True    # extend supervision with GMS-discovered links
    geometry_margin: float = 0.15  # min geodesic gap for a confident link
    pseudo_label: bool = True
    pseudo_label_floor: float = 0.55
    ingest_mode: str = "regex"   # "regex" | "hybrid" | "llm_only"
```

### `EmbedLoopResult` (dataclass)

```python
@dataclass
class EmbedLoopResult:
    ft: object                       # FineTunedEmbedding (document-tuned encoder)
    labels: dict[str, set[str]]      # canonical -> surface texts (final)
    canonicals: list[str]
    iterations: int
    converged: bool
    history: list[dict] = []         # per-iter: n_classes, n_labels,
                                     # geometry_links_added, pseudo_labeled,
                                     # pool_remaining, heldout_acc
```

### Supervision helpers

```python
def graph_entity_labels(
    triples, *, alias_relations=("has_alias",),
    name_relations=("has_policy_name", "has_name"),
) -> dict[str, set[str]]
```
Explicit supervision from the graph's alias structure: `{canonical_entity -> {surface texts}}`, seeded from each head's humanized id, name(s), and alias tails.

```python
def geometry_entity_links(
    triples, model, adapter, canonicals,
    *, relation: str | None = None, margin: float = 0.15,
) -> dict[str, set[str]]
```
Geometry-discovered supervision: assign each non-canonical surface entity to the canonical the trained GMS scores nearest (via `model.score_triple`), but only when the nearest beats the runner-up by `margin`. Returns `{canonical -> {surface entities}}`; a no-op (empty) when the model is untrained or the relation is absent.

```python
def contradiction_sft(
    relation_phrasings: dict[str, list[str]],
    *, base_model="sentence-transformers/nli-mpnet-base-v2",
    rank: int = 32, epochs: int = 500, margin: float = 1.3,
    drift_weight: float = 0.05, weight_decay: float = 1e-4, seed: int = 0,
) -> FineTunedEmbedding
```
Trains a `full`-mode u-space (logical) contradiction encoder over an NLI base from per-relation phrasings (same-relation consistent vs cross-relation contradictory), for the relevance gate's contradiction veto. `full` mode is mandatory — a rotation preserves angles and cannot change tension.

---

## Actor LLM — `qwen_agent_callable`

```python
def qwen_agent_callable(
    model_name: str = QWEN_3B, device: str = "cuda",
    *, system: str = "", max_tokens: int = 256,
) -> Callable[[str], str]
```

Returns the `str → str` callable the GEODE actor uses, backed **exclusively** by a local Qwen 3B (`QWEN_3B`); the model id is asserted, and the heavy backend import is deferred to call time. Pass the result as `GeodeLoop(..., llm=...)` or `build_rag_store(..., llm=...)`.
