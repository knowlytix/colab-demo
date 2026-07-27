# SPDX-License-Identifier: Apache-2.0
"""Post-retrieval cap/tension admissibility filter (the v/u gate, reusable).

The relevance gate (LLM) and answer verification (tension) already live in the
RAG pipeline; the **semantic admissibility** gate (the head-conditioned,
calibrated cap, optionally AND-ed with the contradiction-tension gate) is the
third orthogonal check. This module is its reusable, self-contained home: the
*logic* lives in knowlytix so every consumer shares one calibration, while a
*caller* (e.g. ``agentlab.search()``) decides when to invoke it. It deliberately
does not edit the pipeline/retriever, so it composes without collision; folding
it into ``Retriever`` as an opt-in option is the natural longer-term home.

A retrieved fact is dropped only when it is **scorable and inadmissible** -- an
unscorable fact (head/relation/tail not in the model's vocab) is kept, never
dropped on missing signal. When the model is not cap-trained the filter is a
no-op.
"""
from __future__ import annotations

from knowlytix.core.graph.admissibility import (
    admissible,
    calibrate_cap_margins_per_head,
    calibrate_tension_threshold,
)


def filter_admissible_facts(store, facts, *, cap_margins=None, tension_tau=None,
                            fpr_target: float = 0.05, use_tension: bool = True):
    """Split retrieved facts into ``(kept, dropped)`` by cap (+tension) admissibility.

    Args:
        store: a ``GMSExpertStore`` whose ``model`` is cap-trained.
        facts: iterable of objects with ``head``/``relation``/``tail`` attributes
            (e.g. :class:`~knowlytix.knowledge.rag.retrieve.RetrievedFact`).
        cap_margins / tension_tau: precomputed calibrations (calibrated on first
            use if ``None``; pass them in to amortize across queries).
        fpr_target: target false-positive rate for the calibrations.
        use_tension: AND the contradiction-tension gate with the cap (it is inert
            unless ``u`` carries contradiction structure, so this is safe).

    Returns:
        ``(kept, dropped)`` -- ``kept`` is a list of facts; ``dropped`` is a list
        of ``{"fact", "cap_ok", "tension_ok", "cap_distance", "tension"}`` audit
        dicts (never silent).
    """
    model = getattr(store, "model", None)
    adapter = getattr(store, "adapter", None)
    facts = list(facts)
    if model is None or adapter is None or not getattr(model, "cap_enabled", False):
        return facts, []

    if cap_margins is None:
        cap_margins = calibrate_cap_margins_per_head(model, adapter, fpr_target=fpr_target)
    if tension_tau is None:
        tension_tau = (calibrate_tension_threshold(model, adapter, fpr_target=fpr_target)
                       if use_tension else {})

    e2i = adapter.entity_to_idx
    r2i = adapter.relation_to_idx
    kept, dropped = [], []
    for f in facts:
        hi, ri, ti = e2i.get(f.head), r2i.get(f.relation), e2i.get(f.tail)
        if hi is None or ri is None or ti is None:
            kept.append(f)                       # unscorable -> keep (no signal)
            continue
        v = admissible(model, adapter, hi, ri, ti,
                       cap_margins=cap_margins, tension_tau=tension_tau)
        if v["admissible"]:
            kept.append(f)
        else:
            dropped.append({"fact": f, "cap_ok": v["cap_ok"],
                            "tension_ok": v["tension_ok"],
                            "cap_distance": v["cap_distance"], "tension": v["tension"]})
    return kept, dropped
