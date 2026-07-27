# SPDX-License-Identifier: Apache-2.0
"""Contradiction detection (tension energy) and consistency checking (holonomy).

Every LLM-derived triple is verified against the GMS manifold before
being accepted into the expert store.
"""

from dataclasses import dataclass, field

from knowlytix.knowledge.config import VerifyConfig
from knowlytix.knowledge.extract import ScoredTriple
from knowlytix.knowledge.store import GMSExpertStore


@dataclass
class ContradictionResult:
    """Result of tension energy contradiction check."""
    entity_a: str
    entity_b: str
    tension_energy: float                # 0=agree, sqrt(2)=irrelevant, 2=contradict
    is_contradiction: bool               # E > tau_contra
    is_entailment: bool                  # E < tau_ent


@dataclass
class HolonomyResult:
    """Result of path holonomy consistency check."""
    path: list[str]                      # relation path
    direct: str                          # direct relation
    defect: float                        # holonomy defect
    is_consistent: bool                  # defect <= tau_path


@dataclass
class VerificationReport:
    """Combined verification report."""
    contradictions: list[ContradictionResult] = field(default_factory=list)
    inconsistencies: list[HolonomyResult] = field(default_factory=list)
    overall_consistent: bool = True
    confidence: float = 1.0              # 1.0=fully verified, 0.0=many issues
    detail: str = ""


# =============================================================================
# Tension energy — contradiction detection in u-space
# =============================================================================

def check_contradictions(entity_pairs: list[tuple[str, str]],
                          store: GMSExpertStore,
                          config: VerifyConfig) -> list[ContradictionResult]:
    """Check tension energy for each entity pair.

    Uses Clifford algebra tension energy: E = 2 * sin(theta/2)
      E ≈ 0    -> agreement (entailment)
      E ≈ √2   -> irrelevant (orthogonal)
      E ≈ 2    -> contradiction (anti-aligned)

    Thresholds must come from calibration (``GMSJudge.calibrate()``) per
    the VerifyConfig contract.  ``config.tau_contra`` is required; if it
    is ``None`` this function raises ``ValueError`` rather than
    falling back to a hardcoded default.  ``config.tau_ent`` is
    auxiliary (it only affects the ``is_entailment`` flag); when it is
    ``None`` that flag is set to ``False`` and the contradiction
    decision proceeds on ``tau_contra`` alone.
    """
    if config.tau_contra is None:
        raise ValueError(
            "check_contradictions requires a calibrated "
            "VerifyConfig.tau_contra; received None. Calibrate the "
            "store's judge (GMSJudge.calibrate()) and propagate the "
            "thresholds before invoking the contradiction gate."
        )
    results = []
    for a, b in entity_pairs:
        te = store.tension_energy(a, b)
        if te is None:
            continue
        is_entailment = (
            te < config.tau_ent
            if config.tau_ent is not None
            else False
        )
        results.append(ContradictionResult(
            entity_a=a,
            entity_b=b,
            tension_energy=te,
            is_contradiction=te > config.tau_contra,
            is_entailment=is_entailment,
        ))
    return results


# =============================================================================
# Holonomy — path consistency checking via rotor composition
# =============================================================================

def check_holonomy(scored_triples: list[ScoredTriple],
                    store: GMSExpertStore,
                    config: VerifyConfig) -> list[HolonomyResult]:
    """Find relation triangles and check holonomy defect.

    For each new triple (h, r_new, t):
      - Find existing paths from h to t via intermediate entities
      - Compute holonomy defect: ||R_path @ R_direct^{-1} - I||_F
      - Flag if defect > tau_path

    Mirrors :func:`check_contradictions`: ``config.tau_path`` is
    required. Operating without a calibrated threshold would mean
    comparing ``defect <= None`` (TypeError) or quietly accepting a
    constant — both are unsafe for high-stakes deployment, so this
    gate refuses uncalibrated invocation. Callers that may run
    pre-calibration must check ``config.tau_path is not None``
    before invoking.
    """
    results = []
    if config.tau_path is None:
        raise ValueError(
            "check_holonomy requires a calibrated "
            "VerifyConfig.tau_path; received None. Calibrate the "
            "store's judge (GMSJudge.calibrate()) and propagate the "
            "thresholds before invoking the holonomy gate."
        )
    if store.doc_graph is None or store.adapter is None:
        return results

    for st in scored_triples:
        if st.is_novel or st.head_match is None or st.tail_match is None:
            continue

        head = st.head_match
        tail = st.tail_match
        rel = st.triple.relation

        # Find existing 2-hop paths from head to tail
        # head --r1--> mid --r2--> tail
        out_from_head = {}
        for h, r, t in store.doc_graph.triples:
            if h == head:
                out_from_head.setdefault(r, set()).add(t)

        in_to_tail = {}
        for h, r, t in store.doc_graph.triples:
            if t == tail:
                in_to_tail.setdefault(r, set()).add(h)

        # Find intermediate entities
        for r1, mid_set in out_from_head.items():
            for r2, mid_set2 in in_to_tail.items():
                intermediates = mid_set & mid_set2
                if not intermediates:
                    continue

                # We have path: head --r1--> mid --r2--> tail
                # Compare with direct: head --rel--> tail
                defect = store.check_holonomy([r1, r2], rel)
                if defect is not None:
                    results.append(HolonomyResult(
                        path=[r1, r2],
                        direct=rel,
                        defect=defect,
                        is_consistent=defect <= config.tau_path,
                    ))
                    break  # one check per (r1, r2) pair is enough
            if results and not results[-1].is_consistent:
                break  # found an inconsistency, stop early

    return results


# =============================================================================
# Combined verification
# =============================================================================

def verify_answer(scored_triples: list[ScoredTriple],
                   store: GMSExpertStore,
                   config: VerifyConfig) -> VerificationReport:
    """Run full verification: contradictions + holonomy.

    1. For all pairs of (new triple entities, existing entities): tension energy
    2. For all new triples forming paths: holonomy defect
    3. Compute confidence score
    """
    if not scored_triples:
        return VerificationReport()

    # Gather entity pairs for contradiction checking
    entity_pairs = []
    for st in scored_triples:
        if st.head_match:
            # Check new triple's head against existing related entities
            related = store.query_triples(head=st.head_match)
            for _, _, t in related[:10]:  # limit to avoid explosion
                matched_tail = store.fuzzy_match_entity(t)
                if matched_tail and st.tail_match and matched_tail != st.tail_match:
                    entity_pairs.append((st.tail_match, matched_tail))
        if st.tail_match:
            related = store.query_triples(tail=st.tail_match)
            for h, _, _ in related[:10]:
                matched_head = store.fuzzy_match_entity(h)
                if matched_head and st.head_match and matched_head != st.head_match:
                    entity_pairs.append((st.head_match, matched_head))

    # Deduplicate
    entity_pairs = list(set(entity_pairs))

    # Run checks
    contradictions = check_contradictions(entity_pairs, store, config)
    inconsistencies = check_holonomy(scored_triples, store, config)

    # Compute confidence
    n_checks = len(contradictions) + len(inconsistencies)
    n_issues = (sum(1 for c in contradictions if c.is_contradiction) +
                sum(1 for h in inconsistencies if not h.is_consistent))

    has_contradictions = any(c.is_contradiction for c in contradictions)
    has_inconsistencies = any(not h.is_consistent for h in inconsistencies)
    overall_consistent = not has_contradictions and not has_inconsistencies

    confidence = 1.0 - (n_issues / max(n_checks, 1)) if n_checks > 0 else 1.0

    # Build detail string
    details = []
    if has_contradictions:
        for c in contradictions:
            if c.is_contradiction:
                details.append(
                    f"CONTRADICTION: {c.entity_a} <-> {c.entity_b} "
                    f"(tension={c.tension_energy:.3f})"
                )
    if has_inconsistencies:
        for h in inconsistencies:
            if not h.is_consistent:
                details.append(
                    f"INCONSISTENT PATH: {h.path} vs {h.direct} "
                    f"(defect={h.defect:.3f})"
                )

    return VerificationReport(
        contradictions=contradictions,
        inconsistencies=inconsistencies,
        overall_consistent=overall_consistent,
        confidence=confidence,
        detail="\n".join(details) if details else "All checks passed",
    )
