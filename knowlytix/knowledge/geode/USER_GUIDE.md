# GEODE — User Guide

GEODE — a self-supervised geometric extraction agent. Turn a markdown document
into an accurate GMS knowledge store **without human labels**, and have the
geometry catch and fix its own extraction errors.

For function-level detail see [API_REFERENCE.md](API_REFERENCE.md). For the full
design and validation, see `EXTRACTION_AGENT_DESIGN.md` at the repo root.

---

## 1. The idea

GEODE runs an **actor / critic / gate** loop:

- **Actor** — a cheap proposer (regex backbone + a small local Qwen-3B + dual-embedding entity resolution) emits candidate triples and numeric facts from the document.
- **Critic** — the GMS geometry itself, *computed in closed form, not learned*: plausibility, cap admissibility, tension, holonomy / path-consistency residuals. This is what makes the supervision free.
- **Gate** — a calibrated threshold accepts / rejects / sends-to-review at a controlled error rate. The review band becomes the next iteration's work queue.

Confirmed extractions feed the next round; the loop repeats until the geometry reaches a self-consistent fixed point.

**What the geometry can and cannot fix (the honest limit):**
- ✅ Errors that violate *redundancy* — a relation composition that should hold (holonomy), or a declared constraint — are caught and corrected.
- ❌ Isolated values with no cross-check (a number that appears once, nowhere else derivable) cannot be corrected by geometry. GEODE surfaces these as **anchor violations for review** — it never silently rewrites a parsed number.

---

## 2. Quick start

The one-call path builds a production RAG store with GEODE self-correction applied first:

```python
import torch
from knowlytix.knowledge.geode import build_rag_store
from knowlytix.knowledge.config import DocGMSConfig

cfg = DocGMSConfig(store_path="stores/policy_rag")     # geometry, training, store_path
res = build_rag_store("docs/basel_capital.md", cfg)

print(res.converged, res.iterations, res.n_triples, res.n_entities)
for c in res.corrections:       # what the geometry fixed
    print(c["location"], c["triple"], "->", c["applied"])
for v in res.anchor_violations: # value errors flagged for human review
    print(v)
```

`build_rag_store`:
1. runs the GEODE correction loop over the document (a small GMS trained iteratively to detect/fix errors),
2. trains the **production** store once on the final clean triple set,
3. returns a `RagBuildResult` carrying the saved store plus full diagnostics.

The correction-loop trainer and the production trainer are deliberately separate: the loop trains a small GMS repeatedly to find errors; the store trains once on the corrected graph.

For the **full** build — self-correct, tune both encoders, calibrate the relevance gate and write the coverage report in one call — use `build_calibrated_rag_store`:

```python
from knowlytix.knowledge.geode import build_calibrated_rag_store

res = build_calibrated_rag_store("docs/basel_capital.md", cfg, llm=actor,
                                 canonicalize=True)   # embed_sft=True by default
print(res.artifacts)   # tuned_encoder/, contradiction_encoder/, v_emb.pt,
                       # relevance_calibration.json, coverage_report.json
```

It runs `GeodeLoop` (with optional canonicalization), tunes the v-encoder (`GeodeEmbedLoop`) and the u-encoder (`contradiction_sft`), wires the tuned v-vectors via Mode B, calibrates the relevance gate and writes the coverage report. `embed_sft=False` reduces it to `build_rag_store`. The accept-gate / judge calibration stays separate because it needs a labeled cohort (`rag.eval.calibrate_accept_threshold` + `GMSJudge`).

### Adding the LLM actor (optional)

By default the loop runs geometry-only. Supply a Qwen-3B actor to *raise confidence* on repairs — the geometry stays authoritative, the actor only confirms:

```python
from knowlytix.knowledge.geode import qwen_agent_callable, build_rag_store

llm = qwen_agent_callable(device="cuda")    # locked to local Qwen 3B
res = build_rag_store("docs/basel_capital.md", cfg, llm=llm)
```

When a composition residual fires, the actor is asked to re-derive the disputed value from the *base* relations (with the suspect relation withheld). The repair is applied with the geometry's prediction; agreement from the actor is recorded (`agree=True`) as added confidence, not as a tiebreaker.

---

## 3. Running the loop directly

For diagnostics without building a production store, drive `GeodeLoop` yourself:

```python
import torch
from knowlytix.knowledge.geode import GeodeLoop, make_default_trainer

trainer = make_default_trainer(torch.device("cuda"))   # validated cap+path GMS
loop = GeodeLoop(trainer, max_iters=8)                 # residual_threshold=None → auto-calibrated
result = loop.run("docs/basel_capital.md")

print(result.converged, result.iterations)
print(result.triples)            # corrected triple set
print(result.corrections)        # per-fix audit records
print(result.anchor_violations)  # value errors with no redundancy
```

Key knobs on `GeodeLoop`:

| arg | effect |
| --- | ------ |
| `trainer` | `list[Triple] → (model, adapter, enm)`; use `make_default_trainer(...)`. |
| `llm` | optional `Callable[[str], str]` actor (Qwen 3B). |
| `residual_threshold` | critic flag threshold; `None` lets the critic self-calibrate. |
| `max_iters` | safety bound on iterations (default 8). |
| `check_anchors` | run the external-anchor pass after the composition critic is satisfied (default `True`). |
| `dedup` / `dedup_fpr` | cap+tension duplicate-tail resolution and its false-positive target. |
| `canonicalize` | opt-in entity/relation merge before the critic (default `False`); `canon_sim` / `canon_margin` / `canon_contra_tau` tune it. |

`make_default_trainer(device, *, epochs=300, lambda_path=1.0)` returns a trainer that builds the validated cap+path GMS (`d_v=d_u=64, m=d=32`, `loss_mode="cap"`). Heavy imports are deferred to call time so importing GEODE stays light.

### Canonicalizing co-referent vocabulary (opt-in)

Ingest normalizes surface forms only lexically, so `overdraft fee` / `od fee` and `has_fee` / `has_fee_amount` can survive as distinct entities and relations. That fragmentation defeats the critic (which mines triangles by exact relation identity), dedup and the anchors. Setting `canonicalize=True` runs `canonicalize_graph` before the critic each iteration: the geometry proposes merges by v-space proximity, the logic vetoes them (a shared attribute that disagrees, a declared `opposite_of`, or u-space contradiction), and an entity merge also requires positive corroboration — at least one shared attribute that agrees. Ambiguous surfaces abstain and are flagged, never merged on a guess.

```python
loop = GeodeLoop(trainer, canonicalize=True)
res = loop.run("docs/policy.md")
print(res.canonicalizations)        # applied merges (kind, canonical, merged, similarity)
print(res.canonicalize_flagged)     # surfaces too ambiguous to merge
```

It rewrites graph vocabulary, so it is off by default — validate the proposed merges per corpus before shipping a canonicalized store. It is a no-op on a clean single-document store, and pays off on prose-heavy or cross-document corpora. Run it against a well-trained model: the loop's small diagnostic GMS under-separates, which is why corroboration is required.

You can also call it standalone on a trained model: `canonicalize_graph(triples, model=model, adapter=adapter, v_encoder=tuned_v) -> (rewritten, merges, flagged)`. Pass the pre-filter triples, not the served store (the store strips the structural edges the diagnostics depend on).

---

## 4. Convergence and diagnostics

Each iteration: train → run the `CompositionCritic` → if it flags a triple, localize it via the `ProvenanceLedger`, predict the correct tail from the composition rule, apply, and re-train. When the critic is satisfied (no flags), the external `AnchorChecker` runs once to catch value errors the geometry has no redundancy for. The loop stops at the fixed point or `max_iters`.

`LoopResult.corrections` records, per fix: `triple`, `residual`, source `location`, ingest `method`, `geometry` prediction, `actor` value, whether they `agree`, and what was `applied`. `anchor_violations` lists declared/ENM-constraint breaches surfaced for review.

---

## 5. Fact-only filtering — keep binding off noise relations

The regex ingester emits more than answerable policy facts. It also emits
*organizational*, *schema*, and *boolean-column* relations that are useful for
parsing but are **not facts a question should bind to**:

- `in_section` — organizational. Its tail is a section-*header* entity
  (`## Overdraft Fee Policy` → `overdraft_fee_policy`) that owns no fact edges. A
  question head can wrongly bind to that fact-less header instead of the real
  entity, poisoning retrieval.
- `is_functional` — a schema (single-valued) declaration, not a fact.
- `fails` / `passes` / `is_weak` / `is_strong` — an opaque boolean-column encoding.

So `store_from_triples` (and therefore `build_rag_store`) **drops these by
default** before training, keeping only fact-bearing triples. This is the
`DEFAULT_NOISE_RELATIONS` set, applied through `fact_only_triples`. No action is
needed for the common case — the quick-start build above already gets it.

Filter a triple set yourself (e.g. to inspect what would be dropped):

```python
from knowlytix.knowledge.geode.rag import fact_only_triples, DEFAULT_NOISE_RELATIONS

kept = fact_only_triples(my_triples)                       # default noise set
dropped = [t for t in my_triples if t not in kept]
print(DEFAULT_NOISE_RELATIONS)                             # what gets stripped
```

Override the set per build. Pass your own relations to strip, or `()` to keep
everything (the pre-filter behavior):

```python
from knowlytix.knowledge.geode.rag import store_from_triples

# keep every triple (no fact-only filtering)
store = store_from_triples("docs/basel_capital.md", triples, cfg, drop_relations=())

# strip a custom set
store = store_from_triples("docs/basel_capital.md", triples, cfg,
                           drop_relations={"in_section", "my_noise_rel"})
```

> Note: `fact_only_triples` and `DEFAULT_NOISE_RELATIONS` live in
> `knowlytix.knowledge.geode.rag` (import from there, not the package root).

### Custom embeddings on the GEODE build path

`store_from_triples` now forwards `config.embedding` into the GMS trainer, so a
store config carrying an `EmbeddingConfig` (e.g. a `GeodeEmbedLoop` v-vector
warm-start via Mode B) takes effect on the GEODE build path. Previously this was
dropped, silently making `config.embedding` a no-op when building through GEODE.

---

## 6. Tuning the encoder to the document — `GeodeEmbedLoop`

The base loop self-corrects the *symbolic graph* but leaves the *text encoder*
frozen: the MiniLM that maps a customer phrase ("chargeback", "NSF charge") onto
a canonical policy entity never learns the document's own vocabulary.
`GeodeEmbedLoop` closes that gap — a geometry-supervised, iterative encoder SFT
loop in the v-space (semantic) channel.

Each round: train a GMS on the document, read its entity geometry to *extend*
the document's stated alias/name supervision (a surface the GMS places inside a
policy's neighbourhood is labelled with that policy even with no explicit alias
edge), low-rank SFT the encoder on that supervision, then optionally
pseudo-label an unlabelled customer-vocabulary pool and fold the confident ones
into the next round. It stops when the label set reaches a fixed point.

```python
from knowlytix.embedding import EmbeddingSFTConfig
from knowlytix.knowledge.geode.embed_loop import EmbedLoopConfig, GeodeEmbedLoop
from knowlytix.knowledge.geode.loop import make_default_trainer

sft = EmbeddingSFTConfig(rank=8, mode="full",
                         encoder="sentence-transformers/all-MiniLM-L6-v2",
                         epochs=200, drift_weight=0.5, seed=42, device="cpu")
cfg = EmbedLoopConfig(sft=sft, max_iters=4, use_geometry=True,
                      geometry_margin=0.12, pseudo_label=True,
                      pseudo_label_floor=0.55, ingest_mode="regex")

trainer = make_default_trainer(device="cpu", epochs=150)   # used when use_geometry=True
loop = GeodeEmbedLoop(trainer, cfg)

pool = ["bounced check charge", "dispute a transaction", "waive the overdraft fee"]
res = loop.run("data/bank_policies.md", pool=pool)

print(res.converged, res.iterations, res.canonicals)
encoder = res.ft                       # document-tuned FineTunedEmbedding (drop-in binder)
preds, scores = encoder.classify(["my balance went negative and you charged me"])
```

The output `res.ft` is a `FineTunedEmbedding` tuned to the document: a drop-in
encoder for the RAG binder, and a name-keyed vector export for feeding GMS
`EmbeddingConfig` Mode B (which `store_from_triples` now forwards — see §5).

**When to reach for it:** when customer phrasings or short domain jargon ("NSF",
"reg E claim") bind poorly to the canonical policy ids under the frozen base
encoder. The document must have ≥ 2 canonical entities with alias/name edges
(the seed supervision); otherwise the loop raises.

**Evaluating generalization.** Pass `eval_pairs=[(surface, canonical)]` to record
held-out binding accuracy per iteration in `res.history`, and `exclude_surfaces`
to drop those probes from *both* the SFT labels and the geometry — a clean
generalization test rather than memorization. The loop and its u-space sibling
`contradiction_sft` (a `full`-mode NLI-based contradiction encoder for the
relevance gate's tension veto) live in `knowlytix.knowledge.geode.embed_loop`.

---

## 7. Provenance

Every triple maps back to its exact source span through the `ProvenanceLedger`, so a correction or a flagged value can be traced to a line / table block in the document. Build a ledger directly with `ProvenanceLedger.from_text(text, source_path)` and `resolve(head, relation, tail)`.

---

## 8. Scope and constraints

- **Actor LLM is locked to local Qwen 3B.** `qwen_agent_callable` asserts the model id; GEODE does not call hosted models. This keeps the actor on-device and reproducible.
- **Numbers are never silently rewritten.** GEODE corrects *relational* triples; parsed exact numbers flow through the deterministic ingest's ENM unchanged, and numeric inconsistencies are surfaced as `anchor_violations`.
- **License:** the GEODE package is `Apache-2.0` (distinct from the GMS-core EULA components it composes).
