# SPDX-License-Identifier: Apache-2.0
"""Build a trained GMS expert store from a list of (head, relation, tail) triples.

This is the one piece of glue the KAL+GMS demo needs: the public packages
expose a *document*-first ingest (``knowlytix.knowledge.ingest.ingest_document``,
markdown -> graph -> train), but no *triples*-first builder. KAL produces
triples, so we fill a ``DocumentGraph`` directly and replicate the training
wiring of ``knowlytix.knowledge.ingest._ingest_first``.

Candidate convenience API to propose upstream (a public
``build_store_from_triples`` in ``knowlytix.knowledge``) so consumers don't
carry this.

All imports are the **public** ``knowlytix.*`` namespace; this module does not
run inside knowly (knowly ships `knowly-gms`, namespace `gms.*`). Run it in a
licensed env with the gmsh-distributed wheels installed.
"""
from __future__ import annotations

import io
import sys
import time
from collections.abc import Iterable
from contextlib import redirect_stdout
from dataclasses import replace

import torch

# Public-package imports (mirror of knowlytix.knowledge.ingest._ingest_first).
from knowlytix.benchmark.graph import DocumentGraph
from knowlytix.core.graph.gkg import GeometricKnowledgeGraph
from knowlytix.core.graph.transport import RelationalTransport
from knowlytix.core.memory.compression import CompressionMemory
from knowlytix.core.memory.router import MemoryRouter
from knowlytix.core.train_finstructbench import GraphToGMS, populate_enm, train_gms
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.store import GMSExpertStore

Triple = tuple[str, str, str]


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _QuietTrainerStdout(io.TextIOBase):
    """Stdout proxy that mutes library chatter unfit for a persona-demo cell.

    Upstream prints we suppress:

    * ``  Epoch  N/M  Loss: ...`` — a 11-row wall of numbers the audience
      can't read; we capture first/last loss to compose a one-line summary
      after training.
    * ``  GMS entities: / relations: / triples:`` — duplicates the
      ``Training GMS:`` header line.
    * ``  Training data prepared:`` + its four indented sub-lines —
      only meaningful when ``lambda_tension > 0`` (Scene 4 etc.) where
      ``Agreement / Contradiction pairs`` actually drive training. When
      tension is off (Scene 3 / 5), the counts are noise; we detect this
      by reading the previously-emitted ``Loss:`` line and swallow the
      block on the same pass.
    """

    _GMS_COUNT_PREFIXES = ("GMS entities:", "GMS relations:", "GMS triples:")

    def __init__(self, underlying: object, *, tension_off: bool) -> None:
        self._under = underlying
        self._pending = ""
        self.losses: list[float] = []
        self._tension_off = tension_off
        self._in_prepared_block = False

    def write(self, s: str) -> int:
        self._pending += s
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._handle(line + "\n")
        return len(s)

    def _handle(self, line: str) -> None:
        stripped = line.lstrip()

        # Close out a swallowed "Training data prepared:" block once we hit
        # a line that isn't one of its 4-space-indented sub-rows.
        if self._in_prepared_block:
            if line.startswith("    "):
                return
            self._in_prepared_block = False

        # Always-drop: redundant GMS counts.
        if any(stripped.startswith(p) for p in self._GMS_COUNT_PREFIXES):
            return

        # Conditionally-drop: prepared block when tension training is off.
        if self._tension_off and stripped.startswith("Training data prepared:"):
            self._in_prepared_block = True
            return

        # Per-epoch loss table: capture loss for the summary, swallow line.
        if stripped.startswith("Epoch"):
            try:
                self.losses.append(float(line.split("Loss:")[1].split()[0]))
            except (IndexError, ValueError):
                pass
            return

        self._under.write(line)

    def flush(self) -> None:
        if self._pending:
            self._handle(self._pending)
            self._pending = ""
        self._under.flush()


def _build_adapter_and_train(
    doc_graph: DocumentGraph, config: DocGMSConfig, device: torch.device
) -> tuple[GraphToGMS, GeometricKnowledgeGraph]:
    """Construct the GMS adapter and train the model under one stdout filter.

    The licensed package prints diagnostic chatter at three spots:
    ``GraphToGMS.__init__``, ``prepare_training_data``, and the per-epoch
    loss line inside ``train_gms``. We cover all three under one
    ``_QuietTrainerStdout`` so the demo cell shows the four-line header +
    one-line summary, nothing more.
    """
    # Tension training (Scene 4 et al.) puts real signal into the "Training
    # data prepared:" block; without it the four counts are non-load-bearing
    # noise and the filter swallows the block.
    tension_off = getattr(config.loss, "lambda_tension", 0.0) == 0.0
    filt = _QuietTrainerStdout(sys.stdout, tension_off=tension_off)
    t0 = time.time()
    try:
        with redirect_stdout(filt):
            adapter = GraphToGMS(doc_graph)
            model = train_gms(
                adapter,
                device,
                epochs=config.train.epochs,
                batch_size=config.train.batch_size,
                num_neg=config.train.neg_samples,
                lr=config.train.lr,
                lr_riemannian=config.train.lr_riemannian,
                geometry=config.geometry,
                loss=config.loss,
                loss_mode=config.loss_mode,
                cap=config.cap,
            )
    finally:
        filt.flush()
    elapsed = time.time() - t0

    if filt.losses:
        first, last = filt.losses[0], filt.losses[-1]
        verdict = "converged" if last < first else "did not converge — investigate"
        print(
            f"  Trained {config.train.epochs} epochs in {elapsed:.1f}s — "
            f"loss {first:.2f} → {last:.2f} ({verdict}). Ready to query."
        )
    model.eval()
    return adapter, model


def build_store_from_triples(
    triples: Iterable[Triple],
    *,
    config: DocGMSConfig | None = None,
    device: torch.device | None = None,
    store_path: str | None = None,
) -> GMSExpertStore:
    """Train a ``GMSExpertStore`` from ``(head, relation, tail)`` triples.

    Mirrors ``knowlytix.knowledge.ingest._ingest_first`` (GraphToGMS ->
    train_gms -> populate_enm -> transport/compression/router -> assign onto
    the store), starting from a triples-filled ``DocumentGraph`` instead of a
    parsed document. Hyperparameters come from ``DocGMSConfig`` defaults.

    Keep the input small for the demo — training is O(~100s / ~3k triples) on
    CPU (plan gap G2).
    """
    config = config or DocGMSConfig()
    if store_path is not None:
        config = replace(config, store_path=store_path)  # don't mutate caller's config
    device = device or default_device()

    doc_graph = DocumentGraph()
    for head, relation, tail in triples:
        doc_graph.add_triple(head, relation, tail)

    if not doc_graph.triples:
        raise ValueError("No triples to train on — check the KAL source/loader.")

    adapter, model = _build_adapter_and_train(doc_graph, config, device)
    enm = populate_enm(doc_graph, adapter)
    compression = CompressionMemory(k=config.memory.k, d=config.geometry.m)

    store = GMSExpertStore(config, device)
    store.model = model
    store.adapter = adapter
    store.enm = enm
    store.doc_graph = doc_graph
    store.transport = RelationalTransport(model)
    store.compression = compression
    store.router = MemoryRouter(enm, compression)
    store.markdown = ""
    return store
