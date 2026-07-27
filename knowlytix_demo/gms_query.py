# SPDX-License-Identifier: Apache-2.0
"""Shared GMS query helper — multi-hop transport via chained link_predict.

Pure geometry (no LLM): compose ``store.link_predict`` over a relation chain.
Used by the governance lane, the cross-source lane, and the live-update demo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


class Hop(NamedTuple):
    """One step of a multi-hop walk."""

    head: str
    relation: str
    tail: str | None                  # None = chain broke here (no prediction)
    topk: list[tuple[str, float]]     # (entity, score); lower score = better


def multi_hop(
    store: GMSExpertStore, source: str, chain: list[str], top_k: int = 3
) -> tuple[str | None, list[Hop]]:
    """Walk ``chain`` from ``source`` via chained link_predict.

    Returns ``(answer, hops)``. ``answer`` is None if the chain broke; the last
    ``Hop`` then has ``tail is None`` and names where it broke (``head`` /
    ``relation``).
    """
    current = source
    hops: list[Hop] = []
    for rel in chain:
        preds = store.link_predict(current, rel, top_k=top_k)
        tail = preds[0][0] if preds else None
        hops.append(Hop(current, rel, tail, preds))
        if tail is None:
            return None, hops
        current = tail
    return current, hops


def is_path_valid(hops: list[Hop], triples: list[tuple[str, str, str]]) -> bool:
    """True iff every retrieved hop is a real edge in the training triples.

    ``multi_hop`` walks via ``link_predict``, which returns the geometric
    nearest tail even when no such edge exists in training — that's the
    hallucination floor we want to catch. This is the calibrated PASS/REJECT
    signal (gap **G4**) at its simplest: membership in the source triple set.
    An empty chain or any chain with a ``tail is None`` hop is never valid.
    """
    if not hops:
        return False
    triple_set = {(h, r, t) for h, r, t in triples}
    return all(
        hop.tail is not None and (hop.head, hop.relation, hop.tail) in triple_set
        for hop in hops
    )
