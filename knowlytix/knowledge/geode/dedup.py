# SPDX-License-Identifier: Apache-2.0
"""Admissibility-based duplicate resolution for GEODE extraction.

Exact-duplicate triples are already collapsed at ingest (``DocumentGraph`` dedups
``(h, r, t)``). What remains is the harder case: the same ``(head, relation)``
asserted with *several distinct tails*. That is **not** automatically an error --
GMS cap admissibility is built for one-to-many relations, where a relation maps a
head to an admissibility region that legitimately contains many tails. So we do
not blanket-dedupe; we let the geometry decide.

For each multi-tail ``(h, r)`` group, every tail is scored by the layered,
calibrated admissibility gate (head-conditioned cap + contradiction tension,
:func:`knowlytix.core.graph.admissibility.admissible`):

  * **all tails admissible**       -> legitimate one-to-many, keep all;
  * **some admissible, some not**  -> the inadmissible tails are spurious; drop
    them (audited, with provenance) -- the cap/ tension says they do not belong
    while real alternatives do;
  * **none admissible**            -> ambiguous; keep but flag for review (never
    destroy a whole group on geometry alone -- GEODE's honest-limit stance).

The two gates are applied cap-first, then tension-by-majority:

  1. **cap** drops tails that are v-far from the head's (head-conditioned,
     calibrated) cap -- semantic outliers;
  2. among the cap-admissible tails, **tension** builds the agreement graph
     (edge when ``tension <= tau_r``) and keeps the largest mutually-agreeing
     cluster, dropping the dissenters. This breaks a symmetry a pairwise gate
     cannot: a trained intruder contradicts the whole agreeing majority while
     each true tail contradicts only the intruder, so the intruder is the lone
     dissenting component. A *tie* for the largest cluster is ambiguous -- the
     group is flagged for review, never split on a guess. Tension is used only
     when it is calibrated as separating (``tau_r < 2.0``); otherwise this stage
     is skipped and the cap alone decides.

This runs *after* the composition critic, so relational errors that geometry can
*correct* are already fixed; this step removes the residual spurious duplicates
the critic cannot (a wrong extra tail with no compositional redundancy).
"""
from __future__ import annotations

import torch

from knowlytix.core.graph.admissibility import (
    _cap_ok,
    calibrate_cap_margins_per_head,
    calibrate_tension_threshold,
)


def _agreement_components(nodes, adj):
    """Connected components of the agreement graph (``adj`` = agreeing neighbours)."""
    seen: set = set()
    comps: list[list] = []
    for n in nodes:
        if n in seen:
            continue
        stack, comp = [n], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        comps.append(comp)
    return comps


def resolve_duplicates(model, adapter, triples, ledger=None, *,
                       fpr_target: float = 0.05):
    """Drop inadmissible tails from multi-tail ``(h, r)`` groups; keep one-to-many.

    Args:
        model: a cap-enabled ``GeometricKnowledgeGraph`` trained on ``triples``.
        adapter: the ``GraphToGMS`` whose vocab the model was built on.
        triples: current ``(head, relation, tail)`` list.
        ledger: optional ``ProvenanceLedger`` for source-span attribution.
        fpr_target: target false-positive rate for both calibrated gates.

    Returns:
        ``(kept_triples, removed, flagged)``. ``removed`` and ``flagged`` are
        lists of audit dicts (never silent).
    """
    if not getattr(model, "cap_enabled", False):
        return list(triples), [], []   # no cap -> no admissibility signal

    cap_m = calibrate_cap_margins_per_head(model, adapter, fpr_target=fpr_target)
    tau = calibrate_tension_threshold(model, adapter, fpr_target=fpr_target)
    e2i = adapter.entity_to_idx
    r2i = adapter.relation_to_idx
    dev = next(model.parameters()).device

    groups: dict[tuple[str, str], list[str]] = {}
    for h, r, t in triples:
        groups.setdefault((h, r), []).append(t)

    drop: set[tuple[str, str, str]] = set()
    removed: list[dict] = []
    flagged: list[dict] = []

    for (h, r), tails in groups.items():
        if len(tails) < 2:
            continue
        hi, ri = e2i.get(h), r2i.get(r)
        if hi is None or ri is None:
            continue

        # Stage 1 (cap): drop semantic outliers (tails v-far from the head's cap).
        cap = {}
        for t in tails:
            ti = e2i.get(t)
            if ti is None:
                continue
            cap[t] = _cap_ok(model, hi, ri, ti, cap_m.get((hi, ri), 0.0), dev)
        cap_inadm = [t for t, (ok, _d) in cap.items() if not ok]
        cap_adm = [t for t, (ok, _d) in cap.items() if ok]

        # Stage 2 (tension majority): among cap-admissible tails, keep the largest
        # mutually-agreeing cluster, drop dissenters. This breaks the symmetry a
        # pairwise gate cannot: a trained intruder contradicts the whole agreeing
        # majority while each true tail contradicts only the intruder. Skipped
        # when tension is inert (tau == 2.0) or fewer than 3 cap-admissible tails
        # (no majority to speak of).
        tau_r = tau.get(ri, 2.0)
        dissenters: list[str] = []
        tension_tie = False
        if tau_r < 2.0 and len(cap_adm) >= 3:
            idx = {t: e2i[t] for t in cap_adm}
            adj = {t: set() for t in cap_adm}
            with torch.no_grad():
                for i in range(len(cap_adm)):
                    for j in range(i + 1, len(cap_adm)):
                        a, b = cap_adm[i], cap_adm[j]
                        s = float(model.tension_energy_pairs(
                            torch.tensor([idx[a]], device=dev),
                            torch.tensor([idx[b]], device=dev)).item())
                        if s <= tau_r:          # agree
                            adj[a].add(b)
                            adj[b].add(a)
            comps = sorted(_agreement_components(cap_adm, adj), key=len, reverse=True)
            if len(comps) >= 2 and len(comps[0]) > len(comps[1]):
                majority = set(comps[0])
                dissenters = [t for t in cap_adm if t not in majority]
            elif len(comps) >= 2:               # tie -> ambiguous, do not guess
                tension_tie = True

        survivors = [t for t in cap_adm if t not in dissenters]
        removals = [(t, "cap_outlier") for t in cap_inadm] + \
                   [(t, "contradicts_majority") for t in dissenters]

        if survivors and removals:
            for t, why in removals:
                drop.add((h, r, t))
                rec = {"triple": (h, r, t), "reason": why,
                       "cap_distance": cap[t][1], "kept_tails": survivors}
                if ledger is not None:
                    rec["location"] = ledger.resolve(h, r, t).location()
                removed.append(rec)
        elif not survivors or tension_tie:
            flagged.append({"group": (h, r), "tails": tails,
                            "reason": "tension_tie" if tension_tie
                            else "all_tails_inadmissible"})
        # else: single agreeing cluster, all admissible -> one-to-many, keep all.

    kept = [tr for tr in triples if tr not in drop]
    return kept, removed, flagged
