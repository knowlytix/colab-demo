# SPDX-License-Identifier: Apache-2.0
"""GEODE embedding loop — geometry-supervised, iterative encoder fine-tuning.

The base :class:`~knowlytix.knowledge.geode.loop.GeodeLoop` self-corrects the
*symbolic* graph (geometry repairs contradictory triples) but leaves the *text
encoder* frozen: the MiniLM that maps a customer phrase ("chargeback", "NSF
charge") onto a canonical policy entity at ingest- and retrieval-time never
learns the document's own vocabulary. This module closes that gap.

It is a closed loop over the encoder, with the geometry in the loop:

    ingest (regex / hybrid)  ->  train GMS  ->  read entity geometry
      ->  derive {surface text -> canonical entity} supervision
          (the document's alias structure, EXTENDED by the manifold:
           a surface the GMS places inside a policy's neighbourhood is
           labelled with that policy even when no explicit alias edge exists)
      ->  low-rank SFT the encoder (knowlytix.embedding.finetune_embedding)
      ->  use the tuned encoder to pseudo-label more of the unlabelled
          surface pool (self-training), feeding the next round's supervision
      ->  until the label set stops growing (converged)

The output is a :class:`~knowlytix.embedding.finetune.FineTunedEmbedding`
tuned to the document -- a drop-in encoder for the RAG binder, and a name-keyed
vector export for GMS ``EmbeddingConfig`` Mode B. The geometry *informs* the
supervision; the SFT *tunes* the encoder; the tuned encoder *improves* the next
ingest's entity resolution. Everything is local and reproducible (greedy /
seeded); the GMS trainer is injected so the loop is testable without a GPU.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

__all__ = ["EmbedLoopConfig", "EmbedLoopResult", "GeodeEmbedLoop",
           "graph_entity_labels", "geometry_entity_links", "contradiction_sft"]

# u-space (logical) base encoder -- NLI-trained, so it carries an
# entailment/contradiction prior the v-space (semantic) encoder does not.
_U_BASE = "sentence-transformers/nli-mpnet-base-v2"


def contradiction_sft(relation_phrasings: dict[str, list[str]], *,
                      base_model: str = _U_BASE, rank: int = 32,
                      epochs: int = 500, margin: float = 1.3,
                      drift_weight: float = 0.05, weight_decay: float = 1e-4,
                      seed: int = 0):
    """GEODE u-space contradiction encoder, via the knowlytix embedding SFT API.

    Thin convenience over :func:`knowlytix.embedding.finetune_contradiction`:
    builds a ``full``-mode contradiction config over the NLI (logical) base
    encoder and trains it from per-relation phrasings (same-relation consistent
    vs cross-relation contradictory). Returns a
    :class:`~knowlytix.embedding.FineTunedEmbedding` whose ``.encode`` is the
    tuned u-encoder for the relevance gate's contradiction veto (``full`` mode is
    mandatory -- a rotation preserves angles and cannot change tension).

    Unlike the prototype objective, contradiction *needs* the transform to move:
    a strong drift / weight-decay anchor pins the adapter near identity and the
    learned tension collapses back to raw cosine. The defaults here are light
    (``drift_weight=0.05``) so the contradiction signal can reshape u-space.
    """
    from knowlytix.embedding import EmbeddingSFTConfig, finetune_contradiction

    cfg = EmbeddingSFTConfig(rank=rank, mode="full", objective="contradiction",
                             encoder=base_model, epochs=epochs, margin=margin,
                             drift_weight=drift_weight, weight_decay=weight_decay,
                             seed=seed, device="cpu")
    return finetune_contradiction(relation_phrasings, cfg)


Triple = tuple[str, str, str]

# Relations whose tail is a surface form of the head entity (alias structure).
# A graph's alias edges are the explicit supervision; the geometry extends them.
_ALIAS_RELATIONS = ("has_alias",)
# Relations carrying a human-readable name for the head (used as a surface text).
_NAME_RELATIONS = ("has_policy_name", "has_name")


def _humanize(token: str) -> str:
    """``fee_reversal`` -> ``fee reversal`` -- a natural surface form of an id."""
    return token.replace("_", " ").replace("/", " ").strip()


def graph_entity_labels(
    triples: list[Triple],
    *,
    alias_relations: tuple[str, ...] = _ALIAS_RELATIONS,
    name_relations: tuple[str, ...] = _NAME_RELATIONS,
) -> dict[str, set[str]]:
    """Explicit supervision from the graph's alias structure.

    Returns ``{canonical_entity -> {surface texts}}``. A canonical entity is any
    head that owns an alias or a name edge; its surface set is its own humanized
    id, its name(s), and its alias tails. These are the document's *stated*
    synonyms -- the seed the geometry then extends.
    """
    alias_rel = set(alias_relations)
    name_rel = set(name_relations)
    labels: dict[str, set[str]] = {}
    for h, r, t in triples:
        if r in alias_rel:
            labels.setdefault(h, {_humanize(h)}).add(_humanize(t))
        elif r in name_rel:
            labels.setdefault(h, {_humanize(h)}).add(_humanize(t))
    return labels


def geometry_entity_links(
    triples: list[Triple],
    model,
    adapter,
    canonicals: list[str],
    *,
    relation: str | None = None,
    margin: float = 0.15,
) -> dict[str, set[str]]:
    """Geometry-discovered supervision: assign each surface entity to the
    canonical the trained GMS places it nearest to.

    For every entity in the graph that is not itself a canonical, score
    ``(canonical, relation, surface)`` for each canonical via the GMS geodesic
    (``model.score_triple``; lower = closer) and assign the surface to the
    nearest canonical -- but only when the nearest beats the runner-up by
    ``margin`` (a confident, unambiguous placement). This recovers alias->policy
    structure from the manifold itself, so a surface with a missing or noisy
    alias edge is still labelled correctly when the geometry is unambiguous.

    Returns ``{canonical -> {surface entities}}``. A no-op (empty) when the
    model is not trained or the relation is absent from the vocabulary.
    """
    import torch

    e2i = getattr(adapter, "entity_to_idx", None)
    r2i = getattr(adapter, "relation_to_idx", None)
    if not e2i or not r2i:
        return {}
    rel = relation
    if rel is None:
        rel = next((r for r in _ALIAS_RELATIONS if r in r2i), None)
    if rel is None or rel not in r2i:
        return {}

    canon = [c for c in canonicals if c in e2i]
    if len(canon) < 2:
        return {}
    dev = next(model.parameters()).device
    ri = torch.tensor([r2i[rel]], device=dev)
    surfaces = [e for e in e2i if e not in set(canonicals)]

    out: dict[str, set[str]] = {}
    for s in surfaces:
        si = torch.tensor([e2i[s]], device=dev)
        dists = []
        for c in canon:
            ci = torch.tensor([e2i[c]], device=dev)
            with torch.no_grad():
                d = float(model.score_triple(ci, ri, si).item())
            dists.append((d, c))
        dists.sort()
        if len(dists) >= 2 and (dists[1][0] - dists[0][0]) < margin:
            continue  # ambiguous -- the geometry does not commit
        out.setdefault(dists[0][1], set()).add(_humanize(s))
    return out


@dataclass
class EmbedLoopConfig:
    """Knobs for :class:`GeodeEmbedLoop`."""

    # SFT config is required (rank, mode, encoder, epochs, ...).
    sft: object  # EmbeddingSFTConfig
    max_iters: int = 4
    use_geometry: bool = True          # extend supervision with GMS-discovered links
    geometry_margin: float = 0.15      # min geodesic gap for a confident link
    # Self-training: pseudo-label unlabelled pool surfaces whose nearest-prototype
    # cosine clears this floor, and feed them into the next round's supervision.
    pseudo_label: bool = True
    pseudo_label_floor: float = 0.55
    ingest_mode: str = "regex"         # "regex" | "hybrid" | "llm_only"


@dataclass
class EmbedLoopResult:
    ft: object                          # FineTunedEmbedding (document-tuned encoder)
    labels: dict[str, set[str]]         # canonical -> surface texts (final)
    canonicals: list[str]
    iterations: int
    converged: bool
    history: list[dict] = field(default_factory=list)


class GeodeEmbedLoop:
    """Iterative, geometry-supervised encoder fine-tuning over a document.

    Args:
        trainer: ``list[Triple] -> (model, adapter, enm)`` (see
            :func:`knowlytix.knowledge.geode.loop.make_default_trainer`). Only
            used when ``config.use_geometry`` is on.
        config: :class:`EmbedLoopConfig`.
        llm: optional LLM backend for ``hybrid`` / ``llm_only`` ingest.
    """

    def __init__(self, trainer: Callable | None, config: EmbedLoopConfig,
                 *, llm=None):
        self.trainer = trainer
        self.cfg = config
        self.llm = llm

    # --- ingest -------------------------------------------------------------
    def _ingest(self, md_path: str) -> list[Triple]:
        from knowlytix.benchmark.ingest import ingest_markdown
        mode = self.cfg.ingest_mode
        if mode == "regex":
            return list(ingest_markdown(md_path, mode="regex").triples)
        # hybrid / llm_only need a str->str actor. Accept either a bare callable
        # or an LLMBackend (wrap its .call).
        actor = self.llm
        if actor is not None and not callable(actor) and hasattr(actor, "call"):
            backend = actor

            def actor(prompt, _b=backend):
                return _b.call(system="", user=prompt)
        return list(ingest_markdown(md_path, mode=mode,
                                    llm_callable=actor).triples)

    # --- SFT one round ------------------------------------------------------
    def _sft(self, labels: dict[str, set[str]]):
        from knowlytix.embedding import finetune_embedding
        rows = [{"text": txt, "label": canon}
                for canon, texts in labels.items() for txt in sorted(texts)]
        with tempfile.NamedTemporaryFile(
                "w", suffix=".jsonl", delete=False) as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            path = fh.name
        try:
            return finetune_embedding(path, self.cfg.sft)
        finally:
            Path(path).unlink(missing_ok=True)

    # --- run ----------------------------------------------------------------
    def run(self, md_path: str, *, pool: list[str] | None = None,
            eval_pairs: list[tuple[str, str]] | None = None,
            exclude_surfaces: set[str] | None = None) -> EmbedLoopResult:
        """Run the loop.

        Args:
            md_path: source document.
            pool: optional unlabelled surface phrases (customer vocabulary) the
                self-training step may pseudo-label and fold into supervision.
            eval_pairs: optional ``[(surface, canonical)]`` held-out probes;
                per-iteration nearest-prototype accuracy is recorded in history.
            exclude_surfaces: surface tails to drop from the ingested graph
                before any supervision -- so a held-out evaluation set is absent
                from both the SFT labels AND the geometry (a clean generalization
                test, not memorization).
        """
        triples = self._ingest(md_path)
        if exclude_surfaces:
            ex = {s.lower() for s in exclude_surfaces}
            triples = [(h, r, t) for h, r, t in triples if t.lower() not in ex]
        labels = graph_entity_labels(triples)
        canonicals = sorted(labels.keys())
        if len(canonicals) < 2:
            raise ValueError(
                f"need >= 2 canonical entities with alias/name edges; got "
                f"{canonicals}. Check the document's alias structure.")

        model = adapter = None
        history: list[dict] = []
        ft = None
        converged = False
        remaining_pool = list(pool or [])
        prev_sig = None

        for it in range(1, self.cfg.max_iters + 1):
            # 1) geometry: train GMS, extend supervision with confident links.
            n_geo = 0
            if self.cfg.use_geometry and self.trainer is not None:
                model, adapter, _ = self.trainer(triples)
                geo = geometry_entity_links(
                    triples, model, adapter, canonicals,
                    margin=self.cfg.geometry_margin)
                for c, surfs in geo.items():
                    before = len(labels.get(c, set()))
                    labels.setdefault(c, {_humanize(c)}).update(surfs)
                    n_geo += len(labels[c]) - before

            # 2) SFT the encoder on the current supervision.
            ft = self._sft(labels)

            # 3) evaluate held-out binding (if probes given).
            acc = None
            if eval_pairs:
                preds, _ = ft.classify([s for s, _ in eval_pairs])
                acc = sum(p == c for p, (_, c) in zip(preds, eval_pairs)) / len(
                    eval_pairs)

            # 4) self-training: pseudo-label confident pool surfaces.
            n_pseudo = 0
            if self.cfg.pseudo_label and remaining_pool:
                preds, scores = ft.classify(remaining_pool)
                keep = []
                for surf, p, sc in zip(remaining_pool, preds, scores):
                    conf = float(max(sc)) if hasattr(sc, "__iter__") else float(sc)
                    if p is not None and conf >= self.cfg.pseudo_label_floor:
                        labels.setdefault(p, set()).add(_humanize(surf))
                        n_pseudo += 1
                    else:
                        keep.append(surf)
                remaining_pool = keep

            history.append({
                "iter": it, "n_classes": len(labels),
                "n_labels": sum(len(v) for v in labels.values()),
                "geometry_links_added": n_geo, "pseudo_labeled": n_pseudo,
                "pool_remaining": len(remaining_pool), "heldout_acc": acc,
            })

            # 5) convergence: the supervision set stopped changing.
            sig = tuple(sorted((c, tuple(sorted(v))) for c, v in labels.items()))
            if sig == prev_sig:
                converged = True
                break
            prev_sig = sig

        return EmbedLoopResult(
            ft=ft, labels=labels, canonicals=canonicals,
            iterations=len(history), converged=converged, history=history)
