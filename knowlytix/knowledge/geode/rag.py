# SPDX-License-Identifier: Apache-2.0
"""GEODE → GMSExpertStore wiring: build a queryable graph-RAG store.

GEODE self-corrects an extracted graph (geometry catches contradictions, ENM
anchors catch numeric errors, provenance localizes them). This module turns that
corrected graph into a persisted :class:`~knowlytix.knowledge.store.GMSExpertStore`
— the same artifact the existing :class:`~knowlytix.knowledge.query.QueryEngine`
serves. The result is a graph-RAG system whose distinctive properties are
*self-corrected ingestion* and *geometry-calibrated answer confidence*.

Pipeline::

    build_rag_store(md_path, config)
      = GeodeLoop(...).run(md_path)            # propose → diagnose → repair
        -> corrected triples + diagnostics
      then store_from_triples(...)             # train production GMS + ENM
        -> saved GMSExpertStore

Query the result with the existing engine::

    from knowlytix.knowledge.query import QueryEngine
    res = build_rag_store("doc.md", config)
    engine = QueryEngine(res.store, llm, config)
    answer = engine.query("...")

Heavy imports (torch, core training, the store) are deferred to call time so
importing GEODE stays light.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowlytix.knowledge.geode.loop import GeodeLoop, Trainer, make_default_trainer

__all__ = ["RagBuildResult", "build_rag_store", "build_calibrated_rag_store",
           "store_from_triples", "DEFAULT_NOISE_RELATIONS", "fact_only_triples"]

Triple = tuple[str, str, str]

# Relations the regex ingester emits that are NOT answerable policy facts and
# pollute query binding if kept, so a fact-RAG store drops them by default:
#   * ``in_section`` -- organizational; its tail is a section-header entity
#     ("## Overdraft Fee Policy" -> ``overdraft_fee_policy``) with no fact edges,
#     which a query head then wrongly binds to instead of the real entity.
#   * ``is_functional`` -- a schema (single-valued) declaration, not a fact.
#   * ``fails / passes / is_weak / is_strong`` -- an opaque boolean-column encoding.
# Triple-mediated retrieval should bind questions only onto typed policy facts.
DEFAULT_NOISE_RELATIONS = frozenset({
    "in_section", "is_functional", "fails", "passes", "is_weak", "is_strong"})


def fact_only_triples(triples, *, drop_relations=DEFAULT_NOISE_RELATIONS):
    """Keep only fact-bearing triples: drop organizational / schema / boolean
    relations (see :data:`DEFAULT_NOISE_RELATIONS`) so query binding lands on
    real entities, never a fact-less section header."""
    drop = set(drop_relations)
    return [(h, r, t) for h, r, t in triples if r not in drop]


@dataclass
class RagBuildResult:
    """Outcome of :func:`build_rag_store`."""

    store: object                       # GMSExpertStore (kept loose to defer import)
    converged: bool
    iterations: int
    n_entities: int
    n_triples: int
    n_enm: int
    # GEODE self-correction diagnostics, passed through for audit.
    corrections: list[dict] = field(default_factory=list)
    anchor_violations: list[dict] = field(default_factory=list)
    duplicates_removed: list[dict] = field(default_factory=list)
    duplicates_flagged: list[dict] = field(default_factory=list)
    canonicalizations: list[dict] = field(default_factory=list)
    canonicalize_flagged: list[dict] = field(default_factory=list)
    # Artifacts written by build_calibrated_rag_store (paths relative to the store).
    artifacts: dict = field(default_factory=dict)


def _corrected_doc_graph(md_path: str, triples: list[Triple]):
    """Build a DocumentGraph carrying GEODE's *corrected* triples.

    The numeric ENM and phase encoders come from a fresh deterministic regex
    ingest of the same file (GEODE corrects relational triples, not the parsed
    exact numbers — numeric anchor violations are surfaced for review, never
    silently rewritten — so the ingest's ENM stays consistent). Only the triple
    set is swapped for the corrected one.
    """
    from knowlytix.benchmark.ingest import ingest_markdown

    dg = ingest_markdown(md_path, mode="regex")
    dg.triples = []
    dg._triple_set = set()
    for h, r, t in triples:
        dg.add_triple(h, r, t)
    return dg


def store_from_triples(md_path: str, triples: list[Triple], config,
                       device=None, *, drop_relations=DEFAULT_NOISE_RELATIONS):
    """Train + persist a :class:`GMSExpertStore` from corrected triples.

    This is the wiring core: it mirrors the first-document build path in
    :func:`knowlytix.knowledge.ingest._ingest_first` (train GMS → populate ENM →
    transport/compression/router) but sources its triples from GEODE rather than
    a raw extractor. The store is saved and returned, ready for
    :class:`~knowlytix.knowledge.query.QueryEngine`.

    By default it keeps only fact-bearing triples (:func:`fact_only_triples`):
    organizational/schema/boolean relations are dropped so query binding cannot
    land on a fact-less section-header entity. Pass ``drop_relations=()`` to keep
    everything.
    """
    import torch

    triples = fact_only_triples(triples, drop_relations=drop_relations)
    from knowlytix.core.graph.transport import RelationalTransport
    from knowlytix.core.memory.compression import CompressionMemory
    from knowlytix.core.memory.router import MemoryRouter
    from knowlytix.core.train_finstructbench import (
        GraphToGMS, populate_enm, train_gms,
    )
    from knowlytix.knowledge.store import GMSExpertStore

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    doc_graph = _corrected_doc_graph(md_path, triples)
    adapter = GraphToGMS(doc_graph)

    # Production GMS for serving — trained on the *corrected* graph with the
    # store's configured geometry (cap admissibility when loss_mode='cap'), so
    # load-time reconstruction in GMSExpertStore.load matches.
    model = train_gms(
        adapter, dev,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        num_neg=config.train.neg_samples,
        lr=config.train.lr,
        lr_riemannian=config.train.lr_riemannian,
        geometry=config.geometry,
        loss=config.loss,
        loss_mode=getattr(config, "loss_mode", "point"),
        cap=getattr(config, "cap", None),
        # Honor the store config's EmbeddingConfig (e.g. a GEODE embed-loop
        # v-vector warm-start via Mode B). Previously dropped, which silently
        # made config.embedding a no-op on the GEODE build path.
        embedding=getattr(config, "embedding", None),
    )
    model.eval()

    enm = populate_enm(doc_graph, adapter)
    transport = RelationalTransport(model)
    compression = CompressionMemory(k=config.memory.k, d=config.geometry.m)
    router = MemoryRouter(enm, compression)

    store = GMSExpertStore(config, dev)
    store.model = model
    store.adapter = adapter
    store.enm = enm
    store.doc_graph = doc_graph
    store.transport = transport
    store.compression = compression
    store.router = router
    with open(md_path, encoding="utf-8") as f:
        store.markdown = f.read()
    store.save()
    return store


def build_rag_store(md_path: str, config, *, device=None,
                    geode_trainer: Trainer | None = None,
                    llm=None, max_iters: int = 8,
                    residual_threshold: float | None = None) -> RagBuildResult:
    """Run GEODE self-correction over ``md_path``, then build a RAG store.

    Args:
        md_path: source markdown document.
        config: :class:`~knowlytix.knowledge.config.DocGMSConfig` for the
            production store (geometry, training, store_path, loss_mode).
        device: torch device (defaults to cuda if available).
        geode_trainer: trainer for GEODE's *correction* loop. Defaults to
            :func:`~knowlytix.knowledge.geode.loop.make_default_trainer`. This is
            distinct from the production training inside
            :func:`store_from_triples` — the loop trains a small GMS iteratively
            to detect/fix errors; the store trains once on the final clean graph.
        llm: optional GEODE actor (Qwen 3B, ``Callable[[str], str]``). The
            geometry stays authoritative; the actor only raises confidence.
        max_iters: GEODE loop iteration bound.
        residual_threshold: critic flag threshold (``None`` calibrates).

    Returns:
        :class:`RagBuildResult` with the saved store and GEODE diagnostics.
    """
    import torch

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = geode_trainer or make_default_trainer(dev)

    loop = GeodeLoop(trainer, device=dev, llm=llm, max_iters=max_iters,
                     residual_threshold=residual_threshold)
    res = loop.run(md_path)

    store = store_from_triples(md_path, res.triples, config, dev)

    return RagBuildResult(
        store=store,
        converged=res.converged,
        iterations=res.iterations,
        n_entities=store.adapter.num_entities,
        n_triples=len(store.doc_graph.triples),
        n_enm=len(store.doc_graph.enm),
        corrections=res.corrections,
        anchor_violations=res.anchor_violations,
        duplicates_removed=res.duplicates_removed,
        duplicates_flagged=res.duplicates_flagged,
    )


def build_calibrated_rag_store(
    md_path: str, config, *, device=None, llm=None,
    geode_trainer: Trainer | None = None, max_iters: int = 8,
    residual_threshold: float | None = None, canonicalize: bool = False,
    embed_sft: bool = True, embed_sft_config=None, embed_trainer_epochs: int = 150,
    embed_max_iters: int = 2, pool=None, relation_phraser=None,
    contradiction_epochs: int = 400) -> RagBuildResult:
    """The full GEODE build: self-correct, tune both encoders, calibrate the
    relevance gate, train the production store and write the coverage report.

    :func:`build_rag_store` does the minimum (GEODE loop → store). This adds the
    rest of the deployed pipeline so one call yields a complete, encoder-tuned,
    calibrated store, writing the standard artifacts into ``config.store_path``:

    1. **self-correct** — :class:`GeodeLoop` (optional ``canonicalize``).
    2. **tune $v$** — :class:`GeodeEmbedLoop` over the document; export name-keyed
       entity vectors to ``v_emb.pt`` and point ``config.embedding.v_vectors_path``
       at them (Mode-B warm-start), and save ``tuned_encoder/``.
    3. **tune $u$** — :func:`contradiction_sft` on per-relation phrasings
       (``relation_phraser(rel) -> list[str]``; defaults to the relation phrase
       alone) → ``contradiction_encoder/``.
    4. **calibrate** the relevance gate from the phrasings →
       ``relevance_calibration.json``.
    5. **train** the production store on the clean graph (picks up Mode-B vectors).
    6. **coverage** — region + entity/relation report → ``coverage_report.json``.

    Set ``embed_sft=False`` for the minimal build (equivalent to
    :func:`build_rag_store`). The judge/accept-gate calibration is a separate step
    (:func:`knowlytix.knowledge.rag.eval.calibrate_accept_threshold` +
    ``GMSJudge``) because it needs a labeled cohort, not just the document.
    """
    import json
    import torch
    from pathlib import Path

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = geode_trainer or make_default_trainer(dev)
    store_dir = Path(config.store_path)
    store_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict = {}

    # 1) self-correct.
    loop = GeodeLoop(trainer, device=dev, llm=llm, max_iters=max_iters,
                     residual_threshold=residual_threshold,
                     canonicalize=canonicalize)
    res = loop.run(md_path)
    kept = fact_only_triples(res.triples)

    # 2-4) encoder refinement + relevance calibration.
    if embed_sft:
        from knowlytix.embedding import EmbeddingSFTConfig
        from knowlytix.knowledge.geode.embed_loop import (
            EmbedLoopConfig, GeodeEmbedLoop, contradiction_sft)
        from knowlytix.knowledge.rag.relevance import calibrate_relevance_thresholds

        d_v = getattr(getattr(config, "geometry", None), "d_v", 256)
        sft = embed_sft_config or EmbeddingSFTConfig(
            rank=8, mode="full", out_dim=d_v, device=str(dev))
        eloop = GeodeEmbedLoop(
            make_default_trainer(dev, epochs=embed_trainer_epochs),
            EmbedLoopConfig(sft=sft, max_iters=embed_max_iters, use_geometry=True))
        eres = eloop.run(md_path, pool=pool)
        ent_names = sorted({e for h, _r, t in kept for e in (h, t)})
        v_emb = eres.ft.export_vectors(ent_names)
        torch.save(v_emb, store_dir / "v_emb.pt")
        if getattr(config, "embedding", None) is not None:
            config.embedding.v_vectors_path = str(store_dir / "v_emb.pt")
        eres.ft.save(store_dir / "tuned_encoder")
        artifacts["tuned_encoder"] = "tuned_encoder/"
        artifacts["v_vectors"] = "v_emb.pt"

        content_rels = sorted({r for _h, r, _t in kept
                               if r.startswith("has_") and r != "has_alias"})
        phraser = relation_phraser or (
            lambda rel: [(rel[4:] if rel.startswith("has_") else rel).replace("_", " ")])
        phrasings = {r: phraser(r) for r in content_rels}
        u_enc = contradiction_sft(phrasings, epochs=contradiction_epochs)
        u_enc.save(store_dir / "contradiction_encoder")
        artifacts["contradiction_encoder"] = "contradiction_encoder/"

        relcal = calibrate_relevance_thresholds(
            phrasings, eres.ft.encode, u_enc.encode)
        (store_dir / "relevance_calibration.json").write_text(
            json.dumps(relcal, indent=2) + "\n")
        artifacts["relevance_calibration"] = "relevance_calibration.json"

    # 5) train the production store (Mode-B vectors now wired into config.embedding).
    store = store_from_triples(md_path, kept, config, dev)

    # 6) coverage (graph_coverage needs the pre-filter graph, not the served store).
    from knowlytix.knowledge.rag import coverage_report, graph_coverage
    cov = coverage_report(store)
    gcov = graph_coverage(res.triples)
    (store_dir / "coverage_report.json").write_text(
        json.dumps({**cov.as_dict(), "graph": gcov.as_dict()}, indent=2) + "\n")
    artifacts["coverage_report"] = "coverage_report.json"

    return RagBuildResult(
        store=store, converged=res.converged, iterations=res.iterations,
        n_entities=store.adapter.num_entities,
        n_triples=len(store.doc_graph.triples),
        n_enm=len(store.doc_graph.enm),
        corrections=res.corrections, anchor_violations=res.anchor_violations,
        duplicates_removed=res.duplicates_removed,
        duplicates_flagged=res.duplicates_flagged,
        canonicalizations=res.canonicalizations,
        canonicalize_flagged=res.canonicalize_flagged,
        artifacts=artifacts,
    )
