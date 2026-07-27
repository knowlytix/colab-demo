# SPDX-License-Identifier: Apache-2.0
"""Geometric critic for GEODE — label-free error detection.

The load-bearing detector is the **composition residual** (holonomy /
path-consistency): for an over-determined triple — an instance of a relation
``r3`` that the graph's structure shows equals ``r1 ∘ r2`` (e.g.
``skip_level = manages ∘ manages``) — measure the geodesic gap between the
composed-path prediction and the asserted tail. Clean edges close (~0); an edge
that contradicts the chain does not.

This is what catches errors that cap admissibility cannot: a single corrupted
triple is *absorbed* by the cap during training (the model fits it), but it
still violates the redundant composition. Cap detects out-of-distribution tails;
the composition residual detects internally-contradictory ones — complementary,
not redundant.

Limit (honest): a unique value with no redundant cross-check (e.g. a lone
revenue figure) is not detectable by geometry — it needs an external anchor
(ENM / a declared constraint / a redundant statement elsewhere).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from knowlytix.core.data.prepare import mine_relation_triangles
from knowlytix.core.geometry.sphere import geodesic_distance

__all__ = ["CompositionCritic", "Flag"]


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


@dataclass
class Flag:
    """A triple the critic judges inconsistent with the learned composition."""

    triple: tuple[str, str, str]
    head_idx: int
    tail_idx: int
    rule: tuple[int, int, int]   # (r1, r2, r3) with r3 = r1 ∘ r2
    residual: float              # geodesic gap; higher = more inconsistent


class CompositionCritic:
    """Flags triples violating a learned relation composition.

    Args:
        model: a trained ``GeometricKnowledgeGraph`` (cap + path-consistency).
        adapter: the ``GraphToGMS`` adapter whose vocab the model was built on.
        device: torch device the model lives on.
        residual_threshold: geodesic gap above which a triple is flagged. Leave
            ``None`` (default) to CALIBRATE it from the data instead of using a
            fixed constant (see :meth:`calibrated_threshold`) — a fixed cut does
            not adapt to a document's residual scale (clean residuals grow with
            chain length), so calibration is the default.
        fpr_target: target false-positive rate for calibration.
        exclude_relations: relation names whose triangles are structural noise
            (default excludes ``in_section``).
    """

    def __init__(self, model, adapter, device, *,
                 residual_threshold: float | None = None,
                 fpr_target: float = 0.05,
                 exclude_relations: tuple[str, ...] = ("in_section",)):
        self.model = model
        self.adapter = adapter
        self.device = device
        self._fixed = residual_threshold
        self.fpr_target = fpr_target
        self._exclude = {adapter.relation_to_idx.get(r) for r in exclude_relations}
        self._cached_threshold: float | None = None

    def composition_rules(self) -> list[tuple[int, int, int]]:
        """Relation-composition rules ``r3 = r1 ∘ r2`` mined from the graph."""
        rules = {(r1, r2, r3)
                 for (r1, r2, r3) in mine_relation_triangles(self.adapter)
                 if not (self._exclude & {r1, r2, r3}) and r3 != r1 and r3 != r2}
        return sorted(rules)

    def _path_point(self, h_idx: int, r1: int, r2: int) -> torch.Tensor:
        v_h = self.model.dual_emb.project_v(torch.tensor([h_idx], device=self.device))
        return self.model.apply_relation(
            self.model.apply_relation(v_h, torch.tensor([r1], device=self.device)),
            torch.tensor([r2], device=self.device))

    def _overdetermined(self) -> list[tuple[tuple[int, int, int], int, int]]:
        """Instances ``(rule, head, tail)`` of a composed relation r3 whose
        2-hop ``r1∘r2`` path ACTUALLY EXISTS in the graph.

        An r3 edge whose composition path is absent (e.g. a chain that ends
        before two hops) is not over-determined — there is no second source of
        truth — so it must not be scored. Scoring it would manufacture false
        positives from an ungrounded composition.
        """
        ad = self.adapter
        triples = list(zip(ad.heads.tolist(), ad.relations.tolist(),
                           ad.tails.tolist()))
        out = []
        for (r1, r2, r3) in self.composition_rules():
            adj1: dict[int, list[int]] = {}
            adj2: dict[int, list[int]] = {}
            for h, r, t in triples:
                if r == r1:
                    adj1.setdefault(h, []).append(t)
                if r == r2:
                    adj2.setdefault(h, []).append(t)
            has_path = {h for h, js in adj1.items() if any(j in adj2 for j in js)}
            for h, r, t in triples:
                if r == r3 and h in has_path:
                    out.append(((r1, r2, r3), h, t))
        return out

    def residuals(self) -> list[Flag]:
        """Composition residual for every over-determined triple, descending."""
        if getattr(self, "_cached_residuals", None) is not None:
            return self._cached_residuals
        ad = self.adapter
        out: list[Flag] = []
        with torch.no_grad():
            for (rule, h, t) in self._overdetermined():
                r1, r2, r3 = rule
                p = self._path_point(h, r1, r2)
                v_t = self.model.dual_emb.project_v(
                    torch.tensor([t], device=self.device))
                resid = float(geodesic_distance(p, v_t).cpu())
                out.append(Flag(
                    triple=(ad.idx_to_entity[h], ad.idx_to_relation[r3],
                            ad.idx_to_entity[t]),
                    head_idx=h, tail_idx=t, rule=rule, residual=resid))
        out.sort(key=lambda f: -f.residual)
        self._cached_residuals = out
        return out

    def _corruption_residuals(self, n_neg: int = 8, seed: int = 0) -> list[float]:
        """Synthetic 'definitely wrong' composition residuals: for each
        over-determined instance, swap the tail to random other entities and
        measure the gap. This UNCONTAMINATED reference (every sample is wrong by
        construction) is what the threshold is calibrated against, so it adapts
        to the document's residual scale instead of using a fixed constant.
        """
        rng = random.Random(seed)
        n = self.adapter.num_entities
        out: list[float] = []
        with torch.no_grad():
            for (rule, h, t) in self._overdetermined():
                r1, r2, _ = rule
                p = self._path_point(h, r1, r2)
                for _ in range(n_neg):
                    tp = rng.randrange(n)
                    if tp in (t, h):
                        continue
                    v = self.model.dual_emb.project_v(
                        torch.tensor([tp], device=self.device))
                    out.append(float(geodesic_distance(p, v).cpu()))
        return out

    def calibrated_threshold(self, n_neg: int = 8, seed: int = 0,
                             corruption_quantile: float = 0.10,
                             floor: float = 0.15, gap_ratio: float = 2.5) -> float:
        """Threshold at the separation between the clean cluster and the errors.

        An over-determined edge that satisfies the composition has a small
        residual (the model's composition-fit noise); a wrong tail is an outlier
        ABOVE that cluster. We find the lowest *significant multiplicative gap* in
        the sorted residuals -- where the next residual is at least ``gap_ratio``
        times the previous and above ``floor`` -- and put the bar in that gap
        (geometric mean). Everything above is flagged; everything below (the clean
        cluster) is not.

        This is robust to WHERE the cluster sits (it keys on the cluster-to-error
        *ratio*, not an absolute level), so it catches a subtle near-miss error
        without flagging a clean cluster that happens to fit loosely -- the failure
        of both a fixed cut and a random-corruption quantile (which sits at the
        random-entity distance ~rho_max and misses near-miss errors). When there is
        no significant gap the document is treated as clean (bar above the max), so
        the loop converges. Falls back to the corruption cloud when too few
        over-determined instances exist to see a gap.
        """
        resids = sorted(f.residual for f in self.residuals())
        if len(resids) >= 3:
            for lo, hi in zip(resids, resids[1:]):
                if hi > floor and hi >= gap_ratio * max(lo, 1e-6):
                    return float((lo * hi) ** 0.5)        # in the gap
            return float(resids[-1] + 1.0)                # no gap -> flag nothing
        corr = self._corruption_residuals(n_neg, seed)
        if len(corr) < 5:
            return 0.5
        return float(max(floor, _quantile(corr, corruption_quantile)))

    def effective_threshold(self) -> float:
        """The fixed threshold if one was given, else the calibrated one (cached)."""
        if self._fixed is not None:
            return self._fixed
        if self._cached_threshold is None:
            self._cached_threshold = self.calibrated_threshold()
        return self._cached_threshold

    def flags(self) -> list[Flag]:
        """Residual outliers above the (calibrated) threshold — triples to act on."""
        t = self.effective_threshold()
        return [f for f in self.residuals() if f.residual > t]

    def predict_tail(self, head_idx: int, rule: tuple[int, int, int]) -> str:
        """Tail the composed path ``r1 ∘ r2`` predicts (argmin geodesic)."""
        r1, r2, _ = rule
        with torch.no_grad():
            p = self._path_point(head_idx, r1, r2)
            allv = self.model.dual_emb.project_v(
                torch.arange(self.adapter.num_entities, device=self.device))
            d = geodesic_distance(p.expand(self.adapter.num_entities, -1), allv).cpu()
            d[head_idx] = float("inf")
            return self.adapter.idx_to_entity[int(d.argmin())]
