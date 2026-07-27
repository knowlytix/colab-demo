# SPDX-License-Identifier: Apache-2.0
"""GEODE closed loop — the orchestrator.

Ties the pieces into one self-correcting cycle:

    propose (regex ingest)  ->  integrate (train GMS, cap + path-consistency)
      ->  diagnose (CompositionCritic)  ->  localize (ProvenanceLedger)
      ->  repair (geometry predicts; optional Qwen-3B derivation/adjudication)
      ->  re-integrate  ->  until no flags (converged)

The trainer is injected (``make_default_trainer`` provides the validated
cap+path-consistency trainer) so the orchestration is testable without a GPU.
The actor LLM, when used, is Qwen 3B only (see :mod:`.agent_llm`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from knowlytix.knowledge.geode.anchor import AnchorChecker, enm_from_triples
from knowlytix.knowledge.geode.critic import CompositionCritic
from knowlytix.knowledge.geode.provenance import ProvenanceLedger

__all__ = ["GeodeLoop", "LoopResult", "make_default_trainer"]

Triple = tuple[str, str, str]
# trainer: list[Triple] -> (model, adapter, enm)
# The ENM register holds the triples' exact numeric values (integrity-checked);
# the anchors read it back instead of re-parsing numbers from text.
Trainer = Callable[[list[Triple]], tuple[object, object, object]]


@dataclass
class LoopResult:
    triples: list[Triple]
    converged: bool
    iterations: int
    corrections: list[dict] = field(default_factory=list)
    # External-anchor violations the geometry cannot catch (value errors with no
    # redundancy): detected against declared/ENM constraints, surfaced for review.
    anchor_violations: list[dict] = field(default_factory=list)
    # Spurious duplicate tails removed by cap+tension admissibility (one-to-many
    # tails are kept), and multi-tail groups too ambiguous to resolve (kept, for
    # review). Both audited with provenance.
    duplicates_removed: list[dict] = field(default_factory=list)
    duplicates_flagged: list[dict] = field(default_factory=list)
    # Co-referent entities/relations merged by canonicalization (geometry-near,
    # logic-vetoed), and surfaces too ambiguous to merge (kept, for review).
    canonicalizations: list[dict] = field(default_factory=list)
    canonicalize_flagged: list[dict] = field(default_factory=list)


def make_default_trainer(device=None, *, epochs: int = 300, lambda_path: float = 1.0,
                         cap_head_conditioned: bool = True):
    """Return a trainer that builds + trains the validated cap+path GMS.

    Heavy imports (torch, core training) are deferred to call time so importing
    GEODE stays light. ``cap_head_conditioned`` (default on) gives each head its
    own cap radius so one relation can be functional for one head and one-to-many
    for another -- the geometry the duplicate resolver reads.
    """
    def _trainer(triples: list[Triple]):
        import torch
        from knowlytix.core.config import GeometryConfig, CapLossConfig
        from knowlytix.core.train_finstructbench import GraphToGMS, train_gms

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _G:
            def __init__(self, t):
                self.triples = list(t)

        adapter = GraphToGMS(_G(triples))
        model = train_gms(adapter, dev, epochs=epochs, batch_size=64, num_neg=16,
                          geometry=GeometryConfig(d_v=64, d_u=64, m=32, d=32),
                          loss_mode="cap", cap=CapLossConfig(), seed=42,
                          lambda_path=lambda_path,
                          cap_head_conditioned=cap_head_conditioned)
        model.eval()
        # The exact-numeric register is part of the trained GMS: the geometric
        # model captures structure, the ENM captures the lossless numbers the
        # geometry can't (the anchors check against it). Built from the current
        # triples so it tracks the loop's corrections.
        enm = enm_from_triples(triples)
        return model, adapter, enm
    return _trainer


class GeodeLoop:
    """Self-correcting extraction loop over a regex-ingested document.

    Args:
        trainer: ``list[Triple] -> (model, adapter, enm)``. Use
            :func:`make_default_trainer` for the validated GMS trainer. The
            ``enm`` is an :class:`ExactNumericalMemory` register the anchors
            read for integrity-checked exact values.
        device: torch device for the critic (defaults to model's).
        llm: optional ``str -> str`` actor (Qwen 3B). When given, the actor
            independently derives a repair value and the loop applies it only
            when it agrees with the geometry's prediction; otherwise the
            geometry's prediction is authoritative.
        residual_threshold: critic flag threshold.
        max_iters: safety bound on loop iterations.
    """

    def __init__(self, trainer: Trainer, *, device=None,
                 llm: Callable[[str], str] | None = None,
                 residual_threshold: float | None = None, max_iters: int = 8,
                 check_anchors: bool = True, dedup: bool = True,
                 dedup_fpr: float = 0.05, canonicalize: bool = False,
                 canon_sim: float = 0.86, canon_margin: float = 0.05,
                 canon_contra_tau: float = 1.7,
                 v_encoder=None, u_encoder=None):
        self.trainer = trainer
        self.device = device
        self.llm = llm
        self.threshold = residual_threshold  # None => critic calibrates
        self.max_iters = max_iters
        self.check_anchors = check_anchors
        self.dedup = dedup                   # cap+tension duplicate resolution
        self.dedup_fpr = dedup_fpr
        # Entity/relation canonicalization (opt-in: it rewrites the vocabulary, so
        # it must be calibrated + validated per deployment before it changes a
        # shipped store). Runs before the critic so the critic sees unified vocab.
        self.canonicalize = canonicalize
        self.canon_sim = canon_sim
        self.canon_margin = canon_margin
        self.canon_contra_tau = canon_contra_tau
        self.v_encoder = v_encoder           # relation-phrase similarity (tuned enc)
        self.u_encoder = u_encoder           # relation-merge contradiction veto

    def _derive(self, triples, ent_h, asserted, suggestion) -> str:
        """Ask the actor to derive the value from the base-relation facts,
        with the suspect relation withheld; return its single-token answer."""
        if self.llm is None:
            return suggestion
        facts = "\n".join(f"- {h} -> {t}" for h, r, t in triples
                          if r != f"has_{asserted}")[:4000]
        prompt = (
            "Below are verified relationships (source of truth):\n"
            f"{facts}\n\n"
            f"A data entry claims the answer for '{ent_h}' is '{asserted}', but a "
            f"consistency check disputes it and suggests '{suggestion}'. Using "
            "ONLY the relationships above, reply with the single correct value on "
            "the last line."
        )
        ans = self.llm(prompt).strip()
        return ans.split()[-1].strip(".").lower() if ans else suggestion

    def run(self, md_path: str) -> LoopResult:
        from knowlytix.benchmark.ingest import ingest_markdown
        from knowlytix.knowledge.geode.dedup import resolve_duplicates

        dg = ingest_markdown(md_path, mode="regex")
        ledger = ProvenanceLedger(md_path)
        triples: list[Triple] = list(dg.triples)
        corrections: list[dict] = []
        dups_removed: list[dict] = []
        dups_flagged: list[dict] = []
        canon_merges: list[dict] = []
        canon_flagged: list[dict] = []

        enm = None
        for it in range(1, self.max_iters + 1):
            model, adapter, enm = self.trainer(triples)
            dev = self.device or next(model.parameters()).device

            # Canonicalize FIRST (before the critic) so the critic, dedup, and the
            # anchors all see a unified vocabulary -- otherwise split surfaces hide
            # the redundancy they rely on. A merge rewrites the triple set, so
            # retrain; merges are monotonic, so this converges.
            if self.canonicalize:
                from knowlytix.knowledge.geode.canonicalize import canonicalize_graph
                rewritten, merges, flagged = canonicalize_graph(
                    triples, model=model, adapter=adapter,
                    v_encoder=self.v_encoder, u_encoder=self.u_encoder,
                    sim_threshold=self.canon_sim, margin=self.canon_margin,
                    contra_tau=self.canon_contra_tau)
                canon_flagged = flagged
                if merges:
                    canon_merges.extend(merges)
                    triples = rewritten
                    continue

            critic = CompositionCritic(model, adapter, dev,
                                       residual_threshold=self.threshold)
            # Find a flag the geometry can actually IMPROVE: the predicted tail
            # must differ from the asserted one. A flag whose prediction equals its
            # current tail is a no-op (a just-corrected edge can re-flag against an
            # even tighter cluster); acting on it would loop forever, so it does not
            # count as a correction.
            top = geo = None
            for f in critic.flags():
                pred = critic.predict_tail(f.head_idx, f.rule)
                if pred != f.triple[2]:
                    top, geo = f, pred
                    break

            if top is None:
                # Composition critic satisfied (no actionable correction). Resolve
                # duplicate tails by cap+tension admissibility (keep legitimate
                # one-to-many, drop spurious); a removal changes the triple set, so
                # retrain (the fixed-point loop) -- removals are monotonic, so it
                # converges. Then run external anchors for value errors with no
                # geometric redundancy.
                if self.dedup:
                    kept, removed, flagged = resolve_duplicates(
                        model, adapter, triples, ledger, fpr_target=self.dedup_fpr)
                    dups_flagged = flagged
                    if removed:
                        dups_removed.extend(removed)
                        triples = kept
                        continue
                return LoopResult(triples, True, it, corrections,
                                  self._anchor_check(triples, enm, ledger),
                                  dups_removed, dups_flagged,
                                  canon_merges, canon_flagged)

            ent_h, rname, ent_t = top.triple
            prov = ledger.resolve(ent_h, rname, ent_t)
            derived = self._derive(triples, ent_h, ent_t, geo)
            chosen = geo  # geometry is authoritative; LLM agreement raises confidence
            corrections.append({
                "triple": top.triple, "residual": top.residual,
                "location": prov.location(), "method": prov.method,
                "geometry": geo, "actor": derived, "agree": derived == geo,
                "applied": chosen,
            })
            triples = [(ent_h, rname, chosen) if (a, b, c) == (ent_h, rname, ent_t)
                       else (a, b, c) for (a, b, c) in triples]

        # Did not converge: a correction was applied after the last train, so
        # rebuild the register from the final triples (cheap — no retrain) to
        # keep the anchors consistent with what we return.
        return LoopResult(triples, False, self.max_iters, corrections,
                          self._anchor_check(triples, enm_from_triples(triples),
                                             ledger),
                          dups_removed, dups_flagged,
                          canon_merges, canon_flagged)

    def _anchor_check(self, triples, enm, ledger) -> list[dict]:
        if not self.check_anchors:
            return []
        checker = AnchorChecker.from_enm(enm, ledger)
        out = []
        for v in checker.check_all(triples):
            out.append({
                "kind": v.kind, "message": v.message, "residual": v.residual,
                "locations": [p.location() for p in v.locations],
            })
        return out
