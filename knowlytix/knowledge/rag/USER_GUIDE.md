# GEODE graph-RAG — User Guide

Retrieval-augmented generation over a GMS expert store, where retrieval is
**triple-mediated** rather than vector-based: a question (prose or structured) is
translated into query triples, answered *through the GMS* with provenance-linked
source spans, and synthesized by an LLM of your choice. Bank-grade by default —
GMS-grounded answers, honest abstention, optional answer self-verification, and
no opaque dense index unless you opt in.

For a short overview see [README.md](README.md); for function-level detail see
[API_REFERENCE.md](API_REFERENCE.md); for the full design rationale see
`GEODE_RAG_DESIGN.md` at the repo root.

---

## 1. Why triple-mediated

Vector-RAG retrieves passages by embedding similarity and trusts the LLM to read
them faithfully. That fails the bank-grade bar two ways: a near-miss passage can
look relevant and isn't, and the LLM can fabricate beyond what the passage says.

GEODE-RAG inverts the trust model:

- **The GMS is the only authoritative retriever.** A question becomes query
  triples like `(consumer, revenue, ?)`; the `?` is the asked slot. The GMS
  resolves it from asserted edges (preferred, exact), link prediction, or a
  multi-hop chain.
- **Abstain over guess.** If the query triples do not bind to known
  entities/relations, or no fact matches, the pipeline abstains — it does not
  improvise. The coverage monitor makes the resulting blind spots measurable.
- **Provenance always.** Every fact carries a `file:line:char` source span, so
  the synthesized answer is traceable.
- **Dense retrieval is distrusted.** It is off by default; when enabled it is
  quarantined — flagged `verified=False` with a notice, and refused entirely in
  strict mode.

---

## 2. Quick start

```python
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.llm_backend import LocalTransformersBackend
from knowlytix.knowledge.rag import RagPipeline, RagConfig

# The query-time LLM is your choice. Here: local Qwen 3B, fully on-device.
llm = LocalTransformersBackend("Qwen/Qwen2.5-3B-Instruct", device="cuda")

cfg = RagConfig(
    llm=llm,                     # synthesis (required)
    binding="embedding",         # paraphrases bind to graph vocab (else "fuzzy")
    verify_llm_output=True,      # check the answer's claims against the GMS
    on_verify_fail="abstain",
)

# Build a self-corrected store from a document, then query it.
pipe = RagPipeline.build("report.md", DocGMSConfig(ingest_mode="regex"), cfg)

ans = pipe.query("What was Consumer division revenue?")
print(ans.decision, ans.confidence)   # accept | abstain ; calibrated
print(ans.answer)
for s in ans.sources:                 # provenance per fact
    print(s.head, s.relation, s.tail, s.location)
print(ans.verification)               # per-claim GMS verdicts
print(pipe.coverage().blind_spots)    # doc sections with no triples
```

Runnable end-to-end demo:

```bash
python demo/geode_rag_demo.py --stub   # instant, no model
python demo/geode_rag_demo.py          # local Qwen 3B + embedding binding
```

---

## 3. Building or wrapping a store

A `RagPipeline` always sits on top of a `GMSExpertStore`. Two ways to get one:

```python
# (a) Build from a document — runs GEODE self-correction, then trains a store.
pipe = RagPipeline.build("report.md", DocGMSConfig(ingest_mode="regex"), cfg)

# build_kwargs forward to knowlytix.knowledge.geode.rag.build_rag_store:
pipe = RagPipeline.build("report.md", doc_cfg, cfg,
                         llm=qwen_actor, max_iters=8)

# (b) Wrap a store you already built.
pipe = RagPipeline.from_store(existing_store, cfg)
```

`build(...)` is the GEODE bridge: the *ingestion* actor (the Qwen-3B passed as
`llm=` in `build_kwargs`) is separate from the query-time LLM in `RagConfig`. The
provenance ledger is derived from the store's markdown automatically.

For a store that is also encoder-tuned and gate-calibrated in one step, build with
`knowlytix.knowledge.geode.build_calibrated_rag_store` (tunes the v/u encoders,
calibrates the relevance gate, writes the coverage report) and wrap the result with
`RagPipeline.from_store`.

---

## 4. How a query flows

```
question
  -> parse query triples             LLM (default) OR geometric (encoder, no LLM)
  -> bind to graph vocab             tuned encoder primary, fuzzy fallback
  -> bind-check                      nothing bound -> ABSTAIN (don't guess)
  -> relevance gate                  asked attribute really held? else ABSTAIN
  -> GMS retrieve                    asserted edge / link_predict / multi-hop
  -> attach provenance               source span per fact
  -> synthesize (LLM)                grounded in facts + spans only
  -> self-verify (optional)          claim triples checked vs GMS
  -> decide                          accept | abstain
```

The result is a `RagAnswer`. Key fields: `decision`, `confidence`,
`answer`, `route` (`"triple"`, `"numeric_order"`, or `"dense_fallback"`),
`verified`, `sources`, `verification`, and `notice`.

The numeric-order route (below) is tried *first*, before extraction — a
superlative question is answered by ranking an ENM type, not by binding a triple.

### Numeric ordering (superlatives and comparisons over the ENM)

A question like *"which fee is highest?"* names no anchor entity — the asked thing
*is* the answer. The triple route can't bind it: there is no head to put in
`(head, relation, ?)`. So before extraction the pipeline tries a numeric-order
route that ranks the relevant numeric type directly.

It triggers when both hold:

- the question carries a superlative cue (`highest`, `lowest`, `largest`,
  `smallest`, `most`, `least`, `cheapest`, `priciest`, …) — a *single* direction;
  a question with both kinds of cue, or neither, is not a superlative and falls
  through to the triple route;
- the question matches one of the store's rankable ENM types (a numeric type with
  ≥ 2 scalar entries). With an `encoder` set, the match is by cosine of the
  question to the type phrase (gated by `numeric_order_floor`); without one, the
  type name must appear in the question as a substring.

When it fires, the matched ENM type is ranked by `enm_extreme` — u-space directed
entailment over the **exact** ENM values (§ API_REFERENCE → Numeric ordering) —
and the winner is returned as an `accept` with `route="numeric_order"`,
`verified=True`, and the exact value as an `enm` source. The pipeline then chains a
follow-on relation lookup on the winning entity (the relation whose phrase best
matches the question, else all of its asserted relations), so a compound question
such as *"which division has the highest revenue, and who runs it?"* returns the
ranked fact plus the linked detail in one answer.

This route is **GMS-native and verifiable**: the ranking is the same
directed-entailment operator the GMS uses for logical entailment, applied to the
exact ENM scalars, so the winner and its margin are recomputable rather than read
out of a model.

```python
cfg = RagConfig(
    llm=llm,
    binding="embedding",
    encoder=v_encoder,        # lets the route match the type by paraphrase
    numeric_order=True,       # the default
)
pipe = RagPipeline.from_store(store, cfg)

ans = pipe.query("which fee is highest?")
print(ans.route)              # "numeric_order"
print(ans.decision, ans.answer)   # accept ; "The highest 'fee' value is ... at ..."
for s in ans.sources:
    print(s.source, s.head, s.relation, s.tail)   # the exact ENM value + linked facts
```

**Knobs.**

- `numeric_order=False` turns the route off entirely (every question goes to the
  triple route, as before).
- `numeric_order_tie_margin` (default `0.0`) — the route abstains only when there
  is no *distinct* runner-up, i.e. all entries of the type are equal. Because the
  ENM values are exact, a near-but-distinct second place is **not** a tie; the
  exact values already separate them. Raise it only if you want the route to back
  off when the top two values are within a u-space energy gap of each other.
- `numeric_order_floor` (default `0.25`) — minimum cosine of the question to an
  ENM-type phrase to route there (also reused as the floor for the follow-on
  relation match). Needs an `encoder`; without one the type name must appear in
  the question verbatim.

### Extraction reliability (the small-model lever)

Small models guess relation names that do not exist in the graph. Two knobs fix
most of it:

- `ground_extraction=True` (default) injects the store's actual relation and
  entity vocabulary into the extraction prompt.
- `max_extract_retries=1` (default) re-asks the extractor once when binding
  fails, this time naming the failed terms and the legal relations. This repair
  loop is the single biggest reliability gain for 3B-class models.

### Geometric query parsing (LLM-free)

Even with grounding and a repair retry, a small instruct model still mis-decomposes
hard questions: it puts the *attribute* in the head slot (`harm_threshold_usd`) or
emits a *generic* relation (`has_value`), neither of which binds. `query_parse_mode`
selects the parser, and **`"geometric"` is the default** — a parser with no LLM at
parse time:

```python
cfg = RagConfig(
    llm=llm,
    binding="embedding",
    encoder=v_encoder,           # the document-tuned v-encoder (also used by binding)
    query_parse_mode="geometric",   # default; set "llm" to use the LLM extractor
)
```

The `GeometricQueryParser` (1) matches the question to the graph's *real* relation
phrases by cosine in the document-tuned space, and (2) picks the head among the
entities that actually *assert* that relation, scored by the question's similarity
to each entity's name and aliases. Multi-hop questions come out as a `?x`-chained
triple list (the search walks asserted paths up to `max_hops`); a single-hop
question is one `(head, relation, ?)` triple. Its output is already graph
vocabulary, so the binder passes it through. It prefers `encoder` set to the
document-tuned v-encoder and falls back to MiniLM otherwise (weaker on the
document's terms). Set `"llm"` only when there is no doc graph to parse against, or
to drive an injected extractor (for example in a test).

### Post-retrieval admissibility filter

A third orthogonal check, alongside the relevance gate (§6, question ↔ attribute)
and answer self-verification (§7, claim ↔ GMS): **semantic admissibility** of each
retrieved fact. It asks, per fact, whether the tail sits inside the head-conditioned,
calibrated cap (and, AND-ed in, whether it contradicts the head's established tails
under the contradiction-tension gate). `filter_admissible_facts` is the reusable,
self-contained home for that logic — exported from `knowlytix.knowledge.rag`.

```python
from knowlytix.knowledge.rag import filter_admissible_facts

# facts: RetrievedFacts (head/relation/tail). Calibrations are computed on first
# use; pass cap_margins / tension_tau back in to amortize across queries.
kept, dropped = filter_admissible_facts(store, facts, fpr_target=0.05)
for d in dropped:
    print(d["fact"], d["cap_ok"], d["tension_ok"], d["cap_distance"], d["tension"])
```

It is **caller-invoked, not a default pipeline stage** — it deliberately does not
edit the retriever, so it composes without collision (a caller such as
`agentlab.search()` decides when to apply it). Two safety rules keep it honest: a
fact is dropped only when it is **scorable and inadmissible** — an unscorable fact
(head/relation/tail not in the model's vocab) is *kept*, never dropped on missing
signal — and when the model is not cap-trained the filter is a **no-op**. Every drop
is returned as an audit dict (never silent). The underlying calibrators and the
single-triple `admissible(...)` primitive live in `knowlytix.core.graph.admissibility`
(see API_REFERENCE → *Admissibility filtering*).

---

## 5. Binding modes

| mode | how slots resolve | when to use |
| ---- | ----------------- | ----------- |
| `"fuzzy"` (default) | string match against graph vocab | exact-ish phrasing; no encoder needed |
| `"embedding"` | tuned encoder primary, fuzzy fallback | paraphrases ("staff" → `headcount`); the bank-grade default for prose |

In `"embedding"` mode the document-tuned `encoder` is **primary**, not a fallback:

- **Entities** — a high-confidence cosine match is taken first, so a surface head
  like "overdraft fee policy" maps onto the canonical entity `overdraft` even
  with no string overlap. The string matcher (`fuzzy_match_entity`) is used only
  when the encoder abstains (low score or an ambiguous top1−top2 tie).
- **Relations** — a *literal* relation name (exact, slugged, or `has_<slug>` that
  already exists in the graph vocab) is taken directly; only a non-literal name
  goes to the encoder (then fuzzy as the fallback).

Embedding binding keeps two guards on the encoder match:

- `bind_threshold` (default `0.5`) — minimum cosine to accept a match.
- `bind_margin` (default `0.05`) — minimum top1−top2 gap, else the match is
  refused as ambiguous (abstain rather than pick the wrong entity).

Tune the threshold on labeled terms with `calibrate_bind_threshold` (§8).

---

## 6. Relevance gate (question ↔ attribute)

Binding can succeed yet still answer the wrong question: an interest-rate question
binds to a head that holds `has_fee_amount`, and the (correct) fee comes back. The
fact is true and admissible — it just isn't what was asked. Neither the cap/tension
gates nor embedding cosine catches this; it is a *relevance* problem one layer up,
at question → relation. The relevance gate runs after binding (before retrieval)
and abstains when the bound relation is not the attribute the question asked about.

It is **on by default** (`relevance_gate=True`, bank-grade). Two implementations,
chosen by `relevance_mode`:

| mode | how it decides | needs |
| ---- | -------------- | ----- |
| `"llm"` (default) | a 3B judge picks the one matching attribute from the head's real relations, or `NONE` | an LLM (reuses `llm_extract`); a cosine prefilter keeps it cheap |
| `"geometric"` | LLM-free, GEODE-native dual space: v-space accepts the nearest relation, u-space contradiction tension vetoes an absent-but-adjacent attribute | a v-encoder (`encoder`) + a contradiction-tuned u-encoder (`relevance_u_encoder`) |

The LLM gate is accurate on a small policy but a 3B model over-abstains on a richer
store (many typed, snake-case relations) — and an LLM in the loop is what GEODE's
geometry is meant to replace. Prefer `"geometric"` once you have the embed-loop
encoders: v proposes (semantic nearest relation), u vetoes (the asked attribute is
the *opposite* of that relation — a contradiction, not a distance):

```python
cfg = RagConfig(
    llm=llm,
    binding="embedding",
    encoder=v_encoder,                 # accept step (and binding / parsing)
    relevance_mode="geometric",
    relevance_u_encoder=u_encoder,     # full-mode contradiction-tuned veto
    relevance_tau_accept=0.30,         # min v-cosine to accept a relation
    relevance_tau_contra=0.75,         # max u-tension before the veto fires
)
```

With `relevance_u_encoder=None` the geometric gate runs accept-only (no veto). Turn
the gate off entirely with `relevance_gate=False` (not recommended for regulated
use). The LLM gate's `relevance_prefilter` / `relevance_prefilter_floor` are a
latency optimization for `"llm"` mode only — the floor sits well below any observed
relevant question, so it never decides a borderline case.

### Calibrating the geometric thresholds

Don't hand-pick `tau_accept` / `tau_contra` — fit them from the store's own relation
phrasings, the same no-constants discipline as the cap and tension gates:

```python
from knowlytix.knowledge.rag.relevance import calibrate_relevance_thresholds

# {relation: [build-time phrasings of that attribute]}
phrasings = {
    "has_fee_amount":     ["fee", "monthly charge", "amount billed"],
    "has_interest_rate":  ["interest rate", "annual percentage rate", "APR"],
    # ...
}
cal = calibrate_relevance_thresholds(phrasings, v_encoder, u_encoder)

cfg = RagConfig(
    llm=llm, binding="embedding", encoder=v_encoder,
    relevance_mode="geometric", relevance_u_encoder=u_encoder,
    relevance_tau_accept=cal["tau_accept"],
    relevance_tau_contra=cal["default_tau_contra"],
    relevance_tau_contra_per_relation=cal["tau_contra_per_relation"],
)
print(cal["overlap"])   # relations whose contradictory band overlaps the cut
```

The calibrator returns `tau_accept` (global v-accept floor), a per-relation u-veto
cut, a `default_tau_contra` (the median cut), and `overlap` — the recall-first cuts
never veto a valid attribute, so where a contradictory phrasing leaks through it is
*reported* in `overlap` rather than hidden by tightening the cut.

`GeometricRelevanceGate` and `calibrate_relevance_thresholds` import from
`knowlytix.knowledge.rag.relevance` (they are not re-exported from the package).

---

## 7. Answer self-verification

With `verify_llm_output=True`, the synthesized answer is itself parsed into claim
triples and checked against the GMS. Each claim gets a `ClaimVerdict.status`:
`supported`, `contradicted`, `implausible`, or `unverifiable`. `on_verify_fail`
controls what happens when a claim is `contradicted`/`implausible`:

| `on_verify_fail` | behavior |
| ---------------- | -------- |
| `"abstain"` (recommended) | drop the answer; return the abstain message |
| `"regenerate"` | re-synthesize once from the same facts; if it still fails, annotate |
| `"annotate"` | keep the answer but attach a notice and zero the confidence |

This is the hallucination gate: an answer that drifts beyond the retrieved facts
is caught before it reaches the user.

The `AnswerVerifier` takes `mode="llm"` (default, parse the answer with the LLM) or
`mode="geometric"` (extract the answer's claims by walking the store graph — every
real edge whose head/alias and tail the answer states). The geometric mode binds
claims to canonical vocabulary, so a correct answer yields store-bound triples that
verify instead of LLM-invented relations that false-fail.

---

## 8. Decisions, calibration, and evaluation

A `RagAnswer.decision` is `accept` when `confidence >= accept_threshold`, else
`abstain`. In P1 `accept_threshold=0.0` accepts any grounded answer; calibrate it
against a labeled set:

```python
from knowlytix.knowledge.rag import EvalCase, evaluate, calibrate_bind_threshold

cases = [
    EvalCase("What was Consumer revenue?", expected_answer="4.2", expect_decision="accept"),
    EvalCase("What is the CEO's shoe size?", expect_decision="abstain"),
]
report = evaluate(pipe, cases)
print(report.bind_rate, report.accept_rate,
      report.answer_accuracy, report.decision_accuracy)

# Fit the embedding bind threshold to separate should-bind from should-not.
best_th, acc = calibrate_bind_threshold(
    pipe.binder,
    positives=[("staff", "headcount"), ("revenue", "revenue")],
    negatives=["shoe size", "favorite color"],
)
```

`evaluate` returns an `EvalReport` with `bind_rate`, `accept_rate`,
`answer_accuracy`, `decision_accuracy` and `as_dict()` for logging.

**Calibrating the accept gate.** Fit `accept_threshold` (tau) on a labeled
accept/abstain cohort with `calibrate_accept_threshold`. Run the cohort through a
pipeline with `accept_threshold=0` (so the gate does not pre-filter), collect
`(label, abstained, confidence)` records, and fit:

```python
from knowlytix.knowledge.rag import calibrate_accept_threshold

records = [(1, a.decision == "abstain", a.confidence) for a in pos_answers] \
        + [(0, a.decision == "abstain", a.confidence) for a in neg_answers]
cal = calibrate_accept_threshold(records, max_false_accept=0.05)
# cal["accept_threshold"], cal["balanced_accuracy"], cal["false_accept_ceiling_met"], ...
```

It maximizes balanced accuracy under a false-accept ceiling (a tie-breaker, not a
hard constraint), with a midpoint refinement for a clean bimodal split. Persist the
returned payload as `rag_gate_calibration.json`.

**Benchmarking the parser.** `benchmark_query_parse(store, parsers, binder=...)`
reverses gold questions from the store and scores each parser's parse→bind recovery
and latency with the binder held constant — use it to compare the geometric parser
against an LLM parser on the same questions.

---

## 9. Coverage — measuring blind spots

Triple-mediation can only answer what made it into the graph. `pipe.coverage()`
returns a `CoverageReport` that walks the document's header regions and counts
how many triples' provenance lands in each:

```python
cov = pipe.coverage()
print(cov.coverage_ratio)        # fraction of content regions with >=1 triple
for r in cov.blind_spots:        # regions with body text but no triples
    print(r.title, f"lines {r.line_start}-{r.line_end}", r.body_lines)
```

A blind spot is a section the pipeline *cannot* answer from — surfacing it is how
GEODE-RAG stays honest instead of silently guessing.

`graph_coverage(triples)` adds the entity/relation view region coverage cannot see:
**orphan entities** (a subject reachable only through structural edges), **singleton
relations** (one fact edge — a candidate for under-extraction) and
**declared-but-unpopulated relations** (declared `is_functional`, never used as a
fact edge). Pass the full pre-filter graph (the loop's triples), not the served
store — the store strips the `in_section`/`is_functional` edges these signals depend
on.

```python
from knowlytix.knowledge.rag import graph_coverage

g = graph_coverage(loop_result.triples)
print(g.orphan_entities, g.singleton_relations, g.unpopulated_declared)
```

---

## 10. The audit trail

For bank-grade logging, set `audit_sink` — a callable invoked with every answer's
structured `audit()` dict (decision, route, verified flag, confidence, query and
bound triples, sources with locations, verification verdicts, notice):

```python
records = []
cfg = RagConfig(llm=llm, audit_sink=records.append)
# ... query ...
# records now holds one audit dict per answer, ready to persist.
```

---

## 11. Dense fallback (opt-in, distrusted)

By default, a question that does not bind abstains. If you must answer outside
the graph, enable the quarantined dense route:

```python
cfg = RagConfig(llm=llm, dense_fallback=True, strict_mode=False)
```

When binding fails and `dense_fallback=True`:
- the document is chunked into paragraph spans (with line-range provenance) and
  searched by embedding similarity;
- any answer is returned with `route="dense_fallback"`, `verified=False`, and a
  notice that it is **not** GMS-verified.

With `strict_mode=True`, a dense route is downgraded to `abstain` instead of
returned — use this for regulated deployments where an unverified answer is worse
than no answer. Supply your own `vector_backend` (a `VectorBackend`) to replace
the default in-memory one.

---

## 12. Configuration reference (most-used)

| field | default | effect |
| ----- | ------- | ------ |
| `llm` | — (required) | synthesis backend |
| `llm_extract` | `llm` | question → query triples |
| `llm_verify` | `llm_extract` | answer → claim triples |
| `binding` | `"fuzzy"` | slot resolution strategy |
| `bind_threshold` / `bind_margin` | `0.5` / `0.05` | embedding-binding guards |
| `top_k` | `10` | candidate tails per `?`-slot |
| `max_hops` | `5` | multi-hop chain bound |
| `max_extract_retries` | `1` | binding-failure re-ask count |
| `ground_extraction` | `True` | inject graph schema into extraction |
| `query_parse_mode` | `"llm"` | `"llm"` extractor vs `"geometric"` LLM-free parser (needs `encoder`) |
| `numeric_order` | `True` | enable the superlative/comparison ENM ranking route |
| `numeric_order_tie_margin` | `0.0` | abstain only when no distinct runner-up (all entries equal) |
| `numeric_order_floor` | `0.25` | min cosine of the question to an ENM-type phrase to route there |
| `verify_llm_output` | `False` | run answer self-verification |
| `on_verify_fail` | `"abstain"` | action on a failed claim |
| `relevance_gate` | `True` | run the question↔attribute relevance gate |
| `relevance_mode` | `"llm"` | `"llm"` judge vs `"geometric"` dual-space gate |
| `relevance_u_encoder` | `None` | contradiction-tuned u-encoder for the geometric veto |
| `relevance_tau_accept` / `relevance_tau_contra` | `0.30` / `0.75` | geometric accept / veto thresholds (calibrate) |
| `relevance_tau_contra_per_relation` | `{}` | per-relation overrides of the veto cut |
| `relevance_prefilter` / `relevance_prefilter_floor` | `True` / `0.25` | LLM-gate latency prefilter |
| `accept_threshold` | `0.0` | min confidence to accept |
| `dense_fallback` / `strict_mode` | `False` / `False` | opt-in distrusted retrieval |
| `audit_sink` | `None` | per-answer audit hook |

---

## 13. Persisting the graph to KAL (Postgres/pgvector)

By default a built store lives on local disk. To push the GEODE-corrected graph
into **KAL** (the Knowledge Adapter Layer — the org's backend-agnostic,
Postgres/pgvector-backed knowledge store), use the `kal_sink` module. It is an
**opt-in extra**: it imports `knowlytix.kal` and is intentionally *not*
re-exported from the package, so import it explicitly:

```python
from knowlytix.knowledge.rag.kal_sink import (
    kal_postgres_adapter, persist_store_to_kal_sync,
)

adapter = kal_postgres_adapter(
    host="db.internal", database="knowledge", username="geode", password=...,
)
n = persist_store_to_kal_sync(
    adapter, pipe.store,
    tenant_id="acme",
    extractor="geode",
    confidence=0.95,        # stamps GEODE verification metadata on each triple
)
print(f"persisted {n} triples")
```

What goes to KAL: the **graph** — triples with their source-span provenance and
(optionally) verification metadata. Numeric ENM values ride along as triple
literals (`object_literal`, datatype `number`); entity tails become nodes. The
trained GMS model checkpoint is a separate artifact and is **not** KAL's concern.

The adapter's operations are async; `persist_store_to_kal_sync` is a synchronous
wrapper, and `persist_store_to_kal` is the `await`-able form. KAL's migrations
must already be applied to the target database, and credentials are encrypted at
rest by KAL.

> **License:** the RAG package is `Apache-2.0` (like GEODE), distinct from the
> GMS-core EULA components it composes.
