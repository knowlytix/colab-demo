# SPDX-License-Identifier: Apache-2.0
"""Document ingestion into the GMSExpertStore.

First document: full pipeline (convert -> parse -> train -> populate).
Subsequent documents: incremental growth via learn.py.
"""

import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import torch

from knowlytix.benchmark.ingest import ingest_markdown
from knowlytix.core._license import require_license
from knowlytix.core.train_finstructbench import GraphToGMS, train_gms, populate_enm
from knowlytix.core.graph.transport import RelationalTransport
from knowlytix.core.memory.compression import CompressionMemory
from knowlytix.core.memory.router import MemoryRouter

from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.convert import convert_document
from knowlytix.knowledge.llm_backend import LLMBackend
from knowlytix.knowledge.store import GMSExpertStore


@dataclass
class IngestResult:
    """Result of a document ingestion."""
    document: str
    new_entities: int = 0
    new_triples: int = 0
    new_enm: int = 0
    training_epochs: int = 0
    is_first: bool = False
    elapsed_seconds: float = 0.0


def _llm_backend_to_callable(llm: LLMBackend):
    """Adapt an :class:`LLMBackend` to the ``Callable[[str], str]``
    contract that :func:`knowlytix.benchmark.ingest.ingest_markdown` expects.

    The hybrid / llm_only ingest prompts are self-contained (system-style
    instructions are inlined at the top), so we pass an empty system
    string and route the full prompt through ``backend.call``.
    """
    def _call(prompt: str) -> str:
        return llm.call(system="", user=prompt)

    return _call


def ingest_document(store: GMSExpertStore, document_path: str,
                     llm: LLMBackend | None, config: DocGMSConfig,
                     device: torch.device) -> IngestResult:
    """Ingest a document into the expert store.

    First document: full pipeline (convert, parse, train, populate).
    Subsequent documents: extract new triples/entities, expand via learning.

    Raises :class:`knowlytix.core.DemoModeError` when
    ``KNOWLYTIX_DEMO_MODE=1`` — ingestion is a licensed write path.
    """
    require_license("Document ingestion")
    t0 = time.time()
    result = IngestResult(document=os.path.basename(document_path))

    # Step 1: Convert to markdown
    print(f"\n{'='*60}")
    print(f"Ingesting: {document_path}")
    print(f"{'='*60}")

    md = convert_document(document_path, config.convert, llm)
    print(f"  Converted to markdown: {len(md)} chars")

    # Step 2: Parse markdown into DocumentGraph.
    #
    # ``config.ingest_mode`` selects regex / hybrid / llm_only. Hybrid
    # and llm_only require ``llm`` so we can wrap it as the
    # ``Callable[[str], str]`` that ``ingest_markdown`` expects. We
    # raise on misuse rather than silently falling back to regex —
    # silent degradation would hide that the LLM-augmented extraction
    # never ran, which is a governance hazard.
    ingest_mode = getattr(config, "ingest_mode", "hybrid")
    llm_callable = None
    if ingest_mode in ("hybrid", "llm_only"):
        if llm is None:
            raise ValueError(
                f"ingest_mode={ingest_mode!r} requires an LLMBackend. "
                "Pass llm=AnthropicBackend(...) (or any LLMBackend), or "
                "set DocGMSConfig.ingest_mode='regex' to disable LLM "
                "augmentation."
            )
        llm_callable = _llm_backend_to_callable(llm)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md)
        tmp_path = f.name
    try:
        doc_graph = ingest_markdown(
            tmp_path, mode=ingest_mode, llm_callable=llm_callable,
        )
    finally:
        os.unlink(tmp_path)

    stats = doc_graph.stats()
    print(f"  Parsed: {stats['enm_entries']} ENM, "
          f"{stats['triples']} triples, "
          f"{len(stats.get('phase_encoders', []))} phase encoders, "
          f"mode={ingest_mode!r}")

    if store.model is None:
        # First document: full training pipeline
        result.is_first = True
        _ingest_first(store, doc_graph, md, config, device, result)
    else:
        # Subsequent document: incremental growth
        _ingest_incremental(store, doc_graph, md, config, device, result)

    # Update metadata
    store._metadata.setdefault("documents", []).append({
        "name": result.document,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "triples": result.new_triples,
        "enm": result.new_enm,
    })
    store._metadata["total_ingestions"] = len(store._metadata["documents"])
    if store._metadata.get("created") is None:
        store._metadata["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save
    store.save()

    result.elapsed_seconds = time.time() - t0
    print(f"\n  Ingestion complete: {result.new_entities} entities, "
          f"{result.new_triples} triples, {result.new_enm} ENM entries "
          f"({result.elapsed_seconds:.1f}s)")
    return result


def _ingest_first(store: GMSExpertStore, doc_graph, markdown: str,
                   config: DocGMSConfig, device: torch.device,
                   result: IngestResult):
    """Full training pipeline for first document."""
    print("\n  First document — building GMS from scratch...")

    # Build adapter
    adapter = GraphToGMS(doc_graph)

    # Train GMS. ``loss_mode='cap'`` activates relation-conditioned
    # spherical-cap admissibility (paper.tex §7); the cap config is
    # propagated from ``DocGMSConfig.cap`` so callers (DOEHarnessConfig)
    # can set rho_max / use_diag without reaching into gms internals.
    loss_mode = getattr(config, "loss_mode", "point")
    cap_cfg = getattr(config, "cap", None)
    model = train_gms(
        adapter, device,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        num_neg=config.train.neg_samples,
        lr=config.train.lr,
        lr_riemannian=config.train.lr_riemannian,
        geometry=config.geometry,
        loss=config.loss,
        loss_mode=loss_mode,
        cap=cap_cfg,
    )
    model.eval()

    # Populate ENM
    print("\n  Populating ENM register...")
    enm = populate_enm(doc_graph, adapter)

    # Create transport, compression, router
    transport = RelationalTransport(model)
    m = config.geometry.m
    compression = CompressionMemory(k=config.memory.k, d=m)
    router = MemoryRouter(enm, compression)

    # Set store state
    store.model = model
    store.adapter = adapter
    store.enm = enm
    store.doc_graph = doc_graph
    store.transport = transport
    store.compression = compression
    store.router = router
    store.markdown = markdown

    result.new_entities = adapter.num_entities
    result.new_triples = len(doc_graph.triples)
    result.new_enm = len(doc_graph.enm)
    result.training_epochs = config.train.epochs


def _ingest_incremental(store: GMSExpertStore, new_doc_graph, new_markdown: str,
                         config: DocGMSConfig, device: torch.device,
                         result: IngestResult):
    """Incremental ingestion: add new triples and entities to existing store."""
    from knowlytix.knowledge.learn import RuntimeLearner
    from knowlytix.knowledge.extract import ExtractedTriple

    print("\n  Incremental ingestion — expanding existing store...")

    existing_entities = set(store.adapter.entity_to_idx.keys())
    existing_triples = set(store.doc_graph.triples)

    # Find new triples and entities
    new_triples = []
    new_entity_names = set()
    for h, r, t in new_doc_graph.triples:
        if (h, r, t) not in existing_triples:
            new_triples.append((h, r, t))
            if h not in existing_entities:
                new_entity_names.add(h)
            if t not in existing_entities:
                new_entity_names.add(t)

    print(f"  New triples: {len(new_triples)}")
    print(f"  New entities: {len(new_entity_names)}")

    # Add new triples to doc_graph
    for h, r, t in new_triples:
        store.doc_graph.add_triple(h, r, t)

    # Add new ENM entries
    new_enm_count = 0
    for key, entry in new_doc_graph.enm.items():
        existing = store.doc_graph.lookup(key.type, key.id)
        if existing is None:
            store.doc_graph.store_value(key.type, key.id, entry.value)
            from knowlytix.core.memory.enm import ENMKey as GmsENMKey
            enm_key = GmsENMKey(type=key.type, id=key.id)
            store.enm.write(enm_key, np.array(entry.value))
            new_enm_count += 1

    # Add new phase encoders
    for name, enc in new_doc_graph.phase_encoders.items():
        if name not in store.doc_graph.phase_encoders:
            store.doc_graph.add_phase_encoder(name, enc.v_min, enc.v_max)

    # If there are new entities, use RuntimeLearner to expand embeddings
    if new_entity_names:
        learner = RuntimeLearner(store, config.learn)
        extracted = [
            ExtractedTriple(head=h, relation=r, tail=t)
            for h, r, t in new_triples
            if h in new_entity_names or t in new_entity_names
        ]
        if extracted:
            learn_result = learner.learn_triples(extracted, device)
            print(f"  Learning result: accepted={learn_result.accepted}, "
                  f"entities={len(learn_result.new_entities)}")
    else:
        # Just rebuild adapter with expanded vocabulary
        store.adapter = GraphToGMS(store.doc_graph)

    # Append markdown
    store.markdown += f"\n\n---\n\n{new_markdown}"

    result.new_entities = len(new_entity_names)
    result.new_triples = len(new_triples)
    result.new_enm = new_enm_count
