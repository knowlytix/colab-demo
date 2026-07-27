# SPDX-License-Identifier: Apache-2.0
"""Entity/relation canonicalization for GEODE extraction.

Ingest normalizes surface forms only *lexically* (``DocumentGraph`` lowercases
and collapses whitespace; relations are slugged ``has_<column>``). It does **not**
resolve co-reference: ``overdraft fee`` / ``od fee`` / ``nsf fee`` stay distinct
entities, and ``has_fee`` / ``has_fee_amount`` stay distinct relations. That
fragmentation silently disables the rest of GEODE -- the composition critic mines
triangles by *exact* relation identity, dedup groups by *exact* ``(h, r)``, and
the anchor duplicate-value check keys on ``(entity, relation)``; split vocabulary
means redundancy is never seen and contradictions are never caught. It also splits
a functional relation's tails across two caps and a real entity's facts across two
nodes.

This module merges co-referent **entities** and **relations** *before* the final
store is trained, the GEODE-native way -- the geometry proposes, the logic vetoes,
and ambiguity abstains:

  * **entities** -- propose a merge when two named subjects are v-near (semantic),
    VETO when they are u-far (logical contradiction / tension), and require the
    nearest match to beat the runner-up by a margin (unambiguous);
  * **relations** -- propose a merge when two relation phrases are v-near, VETO on
    a declared ``opposite_of`` or a *functional conflict* (some head asserts both
    with different tails -> they are distinct attributes, e.g. fee vs interest
    rate), again margin-gated.

The discipline matches :func:`resolve_duplicates`: only confident, unambiguous
merges are applied (audited, with the surfaces and scores); an ambiguous cluster
is **flagged for review, never merged on a guess**; nothing numeric is ever a
merge candidate (values are not entities). Merges are monotonic, so iterating to a
fixed point converges. Runs *before* the composition critic so the critic sees a
unified vocabulary.
"""
from __future__ import annotations

import re

import torch

__all__ = ["canonicalize_graph", "Merge"]

# Tension >= this is the GMS "strongly contradictory" band (core config
# ``tau_contra``); two surfaces this far apart in u-space are not the same thing.
_DEFAULT_CONTRA_TAU = 1.7
_NUM_RE = re.compile(r"^-?[\d,]*\.?\d+%?$")


class Merge(dict):
    """An audit record for one applied merge (a dict subclass for JSON-ability)."""


def _is_value(name: str) -> bool:
    """A numeric value-entity is an answer, never a merge candidate."""
    return bool(_NUM_RE.match(name.strip()))


def _rel_phrase(rel: str) -> str:
    return (rel[4:] if rel.startswith("has_") else rel).replace("_", " ").strip()


def _rel_key(rel: str) -> str:
    """Lexical relation key: strip the ``has_`` prefix so ``fee_amount`` and
    ``has_fee_amount`` collide (a safe, deterministic pre-merge)."""
    return rel[4:] if rel.startswith("has_") else rel


def _cos_tension(a, b):
    """GMS tension energy E = sqrt(2 - 2 cos) from already-normalized vectors."""
    cos = float((a * b).sum().clamp(-1.0, 1.0))
    return (2.0 - 2.0 * cos) ** 0.5


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def clusters(self):
        out: dict = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return [c for c in out.values() if len(c) > 1]


def _confident_pairs(names, emb, *, sim_threshold, margin, veto):
    """Mutual-nearest, margin-clearing, un-vetoed pairs; ambiguous nodes flagged.

    A pair ``(a, b)`` is accepted only if ``b`` is ``a``'s nearest neighbour and
    vice-versa, the similarity clears ``sim_threshold``, the gap to each side's
    runner-up clears ``margin`` (unambiguous), and ``veto(a, b)`` is false. A node
    whose top-2 neighbours are both above threshold but are not themselves a
    cluster is *ambiguous* -> returned as flagged, never merged.
    """
    n = len(names)
    if n < 2:
        return [], []
    sims = emb @ emb.t()
    sims.fill_diagonal_(-2.0)
    order = sims.argsort(dim=1, descending=True)

    # First pass: a node whose top-2 neighbours are both above threshold and
    # within ``margin`` of each other is ambiguous -- it cannot be assigned to one
    # cluster without guessing, so it is excluded from EVERY merge (not just its
    # own), and reported for review.
    ambiguous: set = set()
    for i in range(n):
        best = float(sims[i, int(order[i, 0])])
        second = float(sims[i, int(order[i, 1])]) if n > 2 else -2.0
        if best >= sim_threshold and second >= sim_threshold \
                and best - second < margin:
            ambiguous.add(i)

    accepted = {}
    for i in range(n):
        if i in ambiguous:
            continue
        j = int(order[i, 0])
        best = float(sims[i, j])
        if best < sim_threshold or j in ambiguous:
            continue
        if int(order[j, 0]) != i:                 # mutual nearest neighbour only
            continue
        if veto(i, j):
            continue
        a, b = (i, j) if i < j else (j, i)
        accepted[(a, b)] = best
    return [(a, b, s) for (a, b), s in accepted.items()], \
        sorted(names[i] for i in ambiguous)


def _entity_merges(triples, model, adapter, *, sim_threshold, margin, contra_tau):
    """Propose entity merges: v-near, u-tension-vetoed, margin-gated."""
    if model is None or adapter is None:
        return {}, [], []
    e2i = adapter.entity_to_idx
    dev = next(model.parameters()).device

    # Candidates = named subjects (head of >=1 triple), never numeric values.
    fact_counts: dict = {}
    for h, _r, _t in triples:
        fact_counts[h] = fact_counts.get(h, 0) + 1
    cands = sorted(e for e in fact_counts
                   if not _is_value(e) and e in e2i)
    if len(cands) < 2:
        return {}, [], []

    idx = torch.tensor([e2i[e] for e in cands], device=dev)
    with torch.no_grad():
        v = model.dual_emb.project_v(idx)                       # (N, m), unit
        v = torch.nn.functional.normalize(v, dim=-1)

    # Attribute signatures for the structural distinctness veto: two entities that
    # assert the SAME attribute with DIFFERENT values are distinct (the entity
    # analog of the relation functional-conflict veto). This is what separates
    # genuinely different products that merely share a fee from a true co-reference
    # -- v-proximity alone over-merges on a small/under-trained graph, but if their
    # shared attributes disagree they cannot be the same thing. Structural edges
    # (sections, booleans) are excluded so cross-section co-reference is allowed.
    attr: dict = {}
    for h, r, t in triples:
        if r.startswith("has_") and r not in ("has_alias", "has_name",
                                               "has_policy_name"):
            attr.setdefault(h, {}).setdefault(r, set()).add(t)

    def _shared(a, b):
        ra, rb = attr.get(a, {}), attr.get(b, {})
        keys = ra.keys() & rb.keys()
        agree = [r for r in keys if ra[r] == rb[r]]
        conflict = [r for r in keys if ra[r] != rb[r]]
        return agree, conflict

    def veto(i, j):
        a, b = cands[i], cands[j]
        agree, conflict = _shared(a, b)
        # (1) shared attribute with different values -> distinct, never merge.
        if conflict:
            return True
        # (2) REQUIRE positive corroboration: co-reference must be supported by at
        # least one shared attribute that AGREES. v-proximity alone is unreliable
        # on a small/under-trained GMS (it will place a fee product next to a
        # regulation); demanding an agreeing fact is what makes the merge safe and
        # is the evidence a true co-reference (same facts, different name) leaves.
        if not agree:
            return True
        # (3) u-space contradiction veto: do not merge what the logic separates.
        with torch.no_grad():
            ten = float(model.tension_energy_pairs(
                idx[i:i + 1], idx[j:j + 1]).item())
        return ten >= contra_tau

    pairs, ambiguous = _confident_pairs(
        cands, v, sim_threshold=sim_threshold, margin=margin, veto=veto)

    uf = _UnionFind(cands)
    scores: dict = {}
    for a, b, s in pairs:
        uf.union(cands[a], cands[b])
        scores[(cands[a], cands[b])] = s

    cmap, merges = {}, []
    for cluster in uf.clusters():
        # canonical = most-anchored surface (most facts), tie -> shortest name.
        rep = sorted(cluster, key=lambda e: (-fact_counts.get(e, 0), len(e), e))[0]
        for e in cluster:
            if e != rep:
                cmap[e] = rep
        merges.append(Merge(kind="entity", canonical=rep,
                            merged=sorted(c for c in cluster if c != rep),
                            max_similarity=round(max(
                                (scores.get((min(a, b), max(a, b)), 0.0)
                                 for a in cluster for b in cluster if a != b),
                                default=0.0), 4)))
    flagged = [{"kind": "entity", "surface": a, "reason": "ambiguous_nearest"}
               for a in ambiguous]
    return cmap, merges, flagged


def _relation_merges(triples, *, v_encoder, u_encoder, sim_threshold, margin,
                     contra_tau):
    """Propose relation merges: phrase-near, opposite/functional-conflict vetoed."""
    rels = sorted({r for _h, r, _t in triples})
    content = [r for r in rels if r not in ("in_section", "is_functional",
                                             "opposite_of", "passes", "fails",
                                             "is_weak", "is_strong")]
    if len(content) < 2:
        return {}, [], []

    # Hard vetoes from declared structure: opposite_of pairs and functional
    # conflicts (a head asserting two relations with different tails -> distinct).
    opposite = set()
    for h, r, t in triples:
        if r == "opposite_of":
            opposite.add(frozenset((h, t)))
    head_rel_tail: dict = {}
    for h, r, t in triples:
        head_rel_tail.setdefault(h, {}).setdefault(r, set()).add(t)
    conflict = set()
    for _h, rt in head_rel_tail.items():
        present = [r for r in rt if r in content]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                if rt[present[i]] != rt[present[j]]:
                    conflict.add(frozenset((present[i], present[j])))

    enc = v_encoder if v_encoder is not None else _default_encode
    phrases = [_rel_phrase(r) for r in content]
    v = torch.as_tensor(enc(phrases), dtype=torch.float32)
    v = torch.nn.functional.normalize(v, dim=-1)
    u = None
    if u_encoder is not None:
        u = torch.nn.functional.normalize(
            torch.as_tensor(u_encoder(phrases), dtype=torch.float32), dim=-1)

    def veto(i, j):
        pair = frozenset((content[i], content[j]))
        if pair in opposite or pair in conflict:
            return True
        if u is not None and _cos_tension(u[i], u[j]) >= contra_tau:
            return True            # u-space says fee vs interest rate, not synonyms
        return False

    pairs, ambiguous = _confident_pairs(
        content, v, sim_threshold=sim_threshold, margin=margin, veto=veto)

    counts: dict = {}
    for _h, r, _t in triples:
        counts[r] = counts.get(r, 0) + 1
    uf = _UnionFind(content)
    scores: dict = {}
    for a, b, s in pairs:
        uf.union(content[a], content[b])
        scores[(content[a], content[b])] = s

    rmap, merges = {}, []
    for cluster in uf.clusters():
        # canonical = most-used relation, tie -> the has_-prefixed form, then shortest.
        rep = sorted(cluster, key=lambda r: (-counts.get(r, 0),
                                             0 if r.startswith("has_") else 1,
                                             len(r), r))[0]
        for r in cluster:
            if r != rep:
                rmap[r] = rep
        merges.append(Merge(kind="relation", canonical=rep,
                            merged=sorted(c for c in cluster if c != rep),
                            max_similarity=round(max(
                                (scores.get((min(a, b), max(a, b)), 0.0)
                                 for a in cluster for b in cluster if a != b),
                                default=0.0), 4)))
    flagged = [{"kind": "relation", "surface": a, "reason": "ambiguous_nearest"}
               for a in ambiguous]
    return rmap, merges, flagged


def _default_encode(texts):
    from knowlytix.core.graph.encoders import encode_texts
    return encode_texts(texts)


def canonicalize_graph(triples, *, model=None, adapter=None,
                       v_encoder=None, u_encoder=None,
                       sim_threshold: float = 0.86, margin: float = 0.05,
                       contra_tau: float = _DEFAULT_CONTRA_TAU):
    """Merge co-referent entities and relations in a triple set.

    Args:
        triples: current ``(head, relation, tail)`` list.
        model, adapter: a trained ``GeometricKnowledgeGraph`` + its
            ``GraphToGMS``. Required for the (geometric) entity merge; entity
            merging is skipped without them. The model supplies v-space proximity
            and the u-space tension veto.
        v_encoder: ``list[str] -> array`` for relation-phrase similarity (the
            document-tuned encoder when available; defaults to MiniLM).
        u_encoder: optional u-space (contradiction) encoder for the relation
            merge veto (fee vs interest rate). ``opposite_of`` declarations and
            functional conflicts veto regardless.
        sim_threshold: minimum cosine to propose a merge.
        margin: minimum gap to the runner-up (ambiguous pairs abstain).
        contra_tau: tension at/above which a merge is vetoed.

    Returns:
        ``(rewritten_triples, merges, flagged)``. ``merges`` and ``flagged`` are
        audit records (never silent). ``rewritten_triples`` has every surface
        mapped to its canonical and exact duplicates collapsed (order preserved).
    """
    cmap, ent_merges, ent_flag = _entity_merges(
        triples, model, adapter, sim_threshold=sim_threshold, margin=margin,
        contra_tau=contra_tau)
    rmap, rel_merges, rel_flag = _relation_merges(
        triples, v_encoder=v_encoder, u_encoder=u_encoder,
        sim_threshold=sim_threshold, margin=margin, contra_tau=contra_tau)

    if not cmap and not rmap:
        return list(triples), [], ent_flag + rel_flag

    rewritten, seen = [], set()
    for h, r, t in triples:
        nh = cmap.get(h, h)
        nr = rmap.get(r, r)
        nt = cmap.get(t, t)
        key = (nh, nr, nt)
        if key in seen:
            continue
        seen.add(key)
        rewritten.append(key)
    return rewritten, ent_merges + rel_merges, ent_flag + rel_flag
