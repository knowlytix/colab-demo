# SPDX-License-Identifier: Apache-2.0
"""Question -> attribute relevance gate for triple-mediated RAG.

The cap (v-space) gates *graph validity* and tension (u-space) gates *logical
consistency*. Neither, on its own, sees a third failure mode: a query that
retrieves a **true, admissible** fact for an attribute the question never asked
about -- "what is the overdraft interest rate?" binding to ``has_fee_amount``
and returning the (correct) fee. That is a question<->attribute *relevance*
problem, one layer up, at query->relation binding.

There are two gates here.

:class:`RelevanceGate` (legacy) -- an LLM discrimination judgment over the head's
relations, abstaining on "none". It was introduced because raw embedding *cosine*
cannot gate this: a colloquial synonym ("charge") and an absent-but-adjacent
attribute ("interest rate") embed almost equally close to ``has_fee_amount`` in
any single space, so no similarity *floor* separates them. On a small policy the
LLM judge is accurate, but on a richer store (many typed relations, snake-case
names) a 3B model over-abstains, rejecting valid attributes -- and an LLM in the
loop is exactly what GEODE's geometry is meant to replace.

:class:`GeometricRelevanceGate` (preferred) -- the dual-space decision GEODE was
designed for, with no LLM at decision time. Relevance is *not* a single distance;
it is two questions answered in two spaces:

  * **accept (v-space, semantic):** which relation is the question closest to?
    A fine-tuned v-encoder picks the candidate relation by cosine.
  * **veto (u-space, logical):** is the asked attribute the *opposite* of that
    relation -- a distinct attribute that merely sounds money-adjacent? This is a
    *contradiction*, not a distance, and it lives in u-space. Symmetric tension
    ``2 sin(theta/2)`` is monotone in cosine, so it needs a u-encoder *fine-tuned
    for contradiction* (``full`` mode -- a rotation is angle-preserving and cannot
    move tension). Trained on same-relation (consistent) vs cross-relation
    (contradictory) pairs derived from the graph, it learns a tension that
    *generalizes to unseen attributes*: on the banking schema, held-out
    "interest rate" / "annual percentage rate" move from raw-NLI tension ~1.0
    (indistinguishable from synonyms) to >0.9 (clearly contradictory) while
    genuine synonyms ("monthly charge", "amount billed") fall to ~0.55. A query
    is relevant when v-space accepts a relation AND u-space tension to it stays
    below a calibrated contradiction threshold.

v proposes, u vetoes: the same separation of "is it valid / is it consistent"
the cap and tension gates already make, now applied to question<->attribute
relevance instead of a stored triple. The encoders are produced by the GEODE
embed loop (:mod:`knowlytix.knowledge.geode.embed_loop`) and supplied via
``RagConfig.encoder`` (v) and ``RagConfig.relevance_u_encoder`` (u).
"""
from __future__ import annotations

import difflib

_SYSTEM = (
    "You map a question to the ONE entity attribute it asks about.\n"
    "You are given an entity, a list of its known attributes, and a question.\n"
    "Reply with EXACTLY ONE attribute copied VERBATIM from the list, or the single "
    "word NONE.\n"
    "Output only that token -- never the entity name, never extra words.\n"
    "Pick the attribute whose MEANING matches what the question asks for: an "
    "'escalate to' question matches an escalation attribute (not a reporting one); "
    "a colloquial 'charge' or 'cost' is the fee amount.\n"
    "Reply NONE only when NO attribute in the list is what the question asks about "
    "(an interest rate or APR is NOT a fee; a minimum balance is NOT a maximum "
    "reversal)."
)


def _strip_has(rel: str) -> str:
    return rel[4:] if rel.startswith("has_") else rel


def _norm(s: str) -> str:
    """Lowercase, underscores/whitespace -> single spaces (for verbatim matching)."""
    return " ".join(s.lower().replace("_", " ").split())


class RelevanceGate:
    """LLM gate: does the question's asked attribute match a relation the bound
    head actually has? Returns the matched relation, or ``None`` to abstain."""

    def __init__(self, store, llm, *, encoder=None, prefilter: bool = True,
                 prefilter_floor: float = 0.25, max_tokens: int = 16,
                 exclude_relations: tuple[str, ...] = ("in_section",)):
        self.store = store
        self.llm = llm
        self._encoder = encoder
        self.prefilter = prefilter
        self.prefilter_floor = prefilter_floor
        self.max_tokens = max_tokens
        self.exclude = set(exclude_relations)

    def head_relations(self, head: str) -> list[str]:
        """Relations the graph actually asserts on ``head`` (content only)."""
        rels = {r for _h, r, _t in self.store.query_triples(head=head)
                if r not in self.exclude}
        return sorted(rels)

    def _max_cos(self, question: str, rels: list[str]) -> float:
        """Best cosine between the question and the head's relation phrases."""
        import torch

        if self._encoder is None:
            from knowlytix.core.graph.encoders import encode_texts
            self._encoder = encode_texts
        texts = [question] + [_strip_has(r).replace("_", " ") for r in rels]
        emb = torch.nn.functional.normalize(
            torch.as_tensor(self._encoder(texts), dtype=torch.float32), p=2, dim=-1)
        return float((emb[1:] @ emb[0]).max().item())

    def relevant_relation(self, question: str, head: str) -> tuple[str | None, str]:
        """Return ``(matched_relation, detail)``; ``matched_relation is None``
        means the question asks about an attribute the head does not hold -> abstain."""
        rels = self.head_relations(head)
        if not rels:
            return None, "head has no content relations"
        # Hybrid prefilter: reject (cheaply) only when nothing is even close. The
        # floor is conservative, so a plausibly-relevant question always reaches
        # the LLM judge -- the prefilter never decides a borderline case.
        if self.prefilter and self._max_cos(question, rels) < self.prefilter_floor:
            return None, "prefilter: unrelated to all head attributes"
        # Present attributes as readable phrases (snake_case -> words): the raw
        # relation name (e.g. has_exposure_limit_usd_mm) makes a small model both
        # over-abstain on suffixed names and mis-pick among adjacent ones.
        listing = ", ".join(_strip_has(r).replace("_", " ") for r in rels)
        user = (f"Entity: {head}\nKnown attributes: {listing}\n"
                f"Question: {question}\nAnswer:")
        ans = (self.llm.call(system=_SYSTEM, user=user,
                             max_tokens=self.max_tokens) or "").strip()
        # Map the reply back to a real relation by normalized (space-folded)
        # matching, tolerant of morphological variants the model emits ("escalate
        # to" for the relation "escalates to"): containment either way, or a high
        # sequence ratio. The LLM already made the hard relevance call; this only
        # canonicalizes its answer to an exact relation name. NONE / no match abstains.
        ans_n = _norm(ans)
        best_r, best_score = None, 0.0
        for r in rels:
            r_n = _norm(_strip_has(r))
            if not r_n:
                continue
            contained = r_n in ans_n or ans_n in r_n
            ratio = difflib.SequenceMatcher(None, ans_n, r_n).ratio()
            score = 1.0 if contained else ratio
            if score > best_score:
                best_r, best_score = r, score
        if best_r is not None and best_score >= 0.8:
            return best_r, f"matched {best_r} (llm={ans!r})"
        return None, f"none (llm={ans!r})"


def _tension(cos: float) -> float:
    """Tension energy ``2 sin(theta/2)`` from a cosine (unit vectors)."""
    import math

    return 2.0 * math.sin(math.acos(max(-1.0, min(1.0, cos))) / 2.0)


def calibrate_relevance_thresholds(relation_phrasings: dict[str, list[str]],
                                   v_encoder, u_encoder, *,
                                   accept_margin: float = 0.05,
                                   contra_margin: float = 0.02) -> dict:
    """Fit the geometric relevance gate's operating points from the store, the
    same discipline as the cap / tension / accept-threshold gates -- no constants.

    For each relation the build-time phrasings give a labelled cohort:
      * v-accept floor (global): the lowest v-cosine a genuine phrasing has to its
        own relation, minus a margin -- below it, nothing is close enough to accept.
      * u-veto cut (per relation): consistent tension = a phrasing of R vs R's own
        phrase (should pass); contradictory tension = a phrasing of a *different*
        relation vs R's phrase (should be vetoed). The cut is recall-first --
        ``max(consistent) + margin`` -- so a valid attribute is never vetoed; when
        the contradictory band overlaps (``min(contra) <= cut``) the residual leak
        is reported in ``overlap`` rather than sacrificing recall.

    Returns ``{tau_accept, default_tau_contra, tau_contra_per_relation, overlap}``.
    """
    import torch
    import torch.nn.functional as F

    def _vec(enc, texts):
        return F.normalize(torch.as_tensor(enc(texts), dtype=torch.float32), dim=-1)

    rels = [r for r, ws in relation_phrasings.items() if ws]
    phrase = {r: _strip_has(r).replace("_", " ") for r in rels}

    # v-accept floor: min cosine of any phrasing to its own relation phrase.
    accept_floor = 1.0
    for r in rels:
        zr = _vec(v_encoder, [phrase[r]])[0]
        zq = _vec(v_encoder, relation_phrasings[r])
        accept_floor = min(accept_floor, float((zq @ zr).min()))
    tau_accept = max(0.05, accept_floor - accept_margin)

    # u-veto per-relation cut.
    per_rel: dict[str, float] = {}
    overlap: dict[str, float] = {}
    cuts = []
    for r in rels:
        ur = _vec(u_encoder, [phrase[r]])[0]
        cons = [ _tension(float(z @ ur))
                 for z in _vec(u_encoder, relation_phrasings[r]) ]
        contra = []
        for r2 in rels:
            if r2 == r:
                continue
            for z in _vec(u_encoder, relation_phrasings[r2][:3]):
                contra.append(_tension(float(z @ ur)))
        cut = (max(cons) if cons else 0.0) + contra_margin
        per_rel[r] = round(cut, 4)
        cuts.append(cut)
        if contra and min(contra) <= cut:
            overlap[r] = round(min(contra), 4)
    default = round(sorted(cuts)[len(cuts) // 2], 4) if cuts else 1.0  # median
    return {"tau_accept": round(tau_accept, 4), "default_tau_contra": default,
            "tau_contra_per_relation": per_rel, "overlap": overlap}


class GeometricRelevanceGate:
    """LLM-free relevance: v-space accepts a relation, u-space tension vetoes a
    contradictory (absent-but-adjacent) attribute.

    Args:
        store: provides ``query_triples`` to list a head's asserted relations.
        v_encoder: ``list[str] -> vectors`` for the semantic accept step
            (the GEODE embed loop's v-encoder; falls back to MiniLM).
        u_encoder: ``list[str] -> vectors`` for the contradiction veto
            (the embed loop's ``full``-mode contradiction-tuned u-encoder).
            When ``None`` the veto is skipped (accept-only).
        tau_accept: min v-cosine to accept a candidate relation.
        tau_contra: max u-tension to the accepted relation; above it the asked
            attribute contradicts the relation -> abstain. Per-relation overrides
            may be supplied in ``tau_contra_per_relation``.
    """

    def __init__(self, store, v_encoder=None, u_encoder=None, *,
                 tau_accept: float = 0.30, tau_contra: float = 0.75,
                 tau_contra_per_relation: dict[str, float] | None = None,
                 exclude_relations: tuple[str, ...] = ("in_section",)):
        self.store = store
        self._v = v_encoder
        self._u = u_encoder
        self.tau_accept = tau_accept
        self.tau_contra = tau_contra
        self.tau_contra_per_relation = tau_contra_per_relation or {}
        self.exclude = set(exclude_relations)

    def head_relations(self, head: str) -> list[str]:
        rels = {r for _h, r, _t in self.store.query_triples(head=head)
                if r not in self.exclude}
        return sorted(rels)

    def _enc(self, which, texts):
        import torch
        enc = self._v if which == "v" else self._u
        if enc is None:
            from knowlytix.core.graph.encoders import encode_texts
            enc = encode_texts
        return torch.nn.functional.normalize(
            torch.as_tensor(enc(texts), dtype=torch.float32), p=2, dim=-1)

    def relevant_relation(self, question: str, head: str) -> tuple[str | None, str]:
        """``(matched_relation, detail)``; ``None`` abstains."""
        rels = self.head_relations(head)
        if not rels:
            return None, "head has no content relations"
        phrases = [_strip_has(r).replace("_", " ") for r in rels]

        # accept (v-space): nearest relation by cosine.
        v = self._enc("v", [question] + phrases)
        sims = (v[1:] @ v[0])
        bi = int(sims.argmax())
        best_r, best_cos = rels[bi], float(sims[bi])
        if best_cos < self.tau_accept:
            return None, f"v-accept: nothing >= tau_accept (best {best_cos:.2f})"

        # veto (u-space): contradiction tension to the accepted relation.
        if self._u is not None:
            u = self._enc("u", [question, phrases[bi]])
            ten = _tension(float(u[0] @ u[1]))
            tau_c = self.tau_contra_per_relation.get(best_r, self.tau_contra)
            if ten > tau_c:
                return None, (f"u-veto: tension {ten:.2f} > {tau_c:.2f} "
                              f"(asked attribute contradicts {best_r})")
            return best_r, f"matched {best_r} (v={best_cos:.2f} u-tension={ten:.2f})"
        return best_r, f"matched {best_r} (v={best_cos:.2f}, no u-veto)"
