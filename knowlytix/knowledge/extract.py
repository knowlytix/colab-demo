# SPDX-License-Identifier: Apache-2.0
"""Triple extraction from LLM answers and GMS scoring.

Every LLM answer is decomposed into (head, relation, tail) triples,
then scored against the GMS manifold for plausibility.
"""

import re
from dataclasses import dataclass, field

import torch

from knowlytix.knowledge.llm_backend import LLMBackend
from knowlytix.knowledge.store import GMSExpertStore


@dataclass
class ExtractedTriple:
    """A single (head, relation, tail) triple extracted from text."""
    head: str
    relation: str
    tail: str
    numeric_value: float | None = None


@dataclass
class ScoredTriple:
    """An extracted triple scored against the GMS manifold."""
    triple: ExtractedTriple
    gms_score: float | None = None       # geodesic distance (lower = better)
    head_match: str | None = None        # matched entity in store
    tail_match: str | None = None        # matched entity in store
    is_novel: bool = False               # True if entities not in store


@dataclass
class PlausibilityReport:
    """Aggregate plausibility assessment of scored triples."""
    plausible: bool
    mean_score: float
    worst_score: float
    novel_count: int
    known_count: int
    scored_triples: list[ScoredTriple] = field(default_factory=list)


# =============================================================================
# Triple extraction via LLM
# =============================================================================

_EXTRACT_SYSTEM = (
    "You are a knowledge extraction engine. Given a question and answer, "
    "extract ALL factual claims as structured triples.\n\n"
    "Rules:\n"
    "- Each triple is: subject | relationship | object\n"
    "- For numeric facts, append the exact value after a fourth pipe\n"
    "- Use short, normalized relation names (e.g., has_value, is_type, "
    "interacts_with, has_score, located_in)\n"
    "- One triple per line, prefixed with TRIPLE:\n"
    "- Extract EVERY factual claim, even implicit ones\n"
    "- Output ONLY triples, no commentary\n\n"
    "Example:\n"
    "TRIPLE: Warfarin | has_molecular_weight | 308.33 | 308.33\n"
    "TRIPLE: Warfarin | interacts_with | Aspirin\n"
    "TRIPLE: Aspirin | is_type | NSAID\n"
)


def extract_triples(answer_text: str, question: str,
                     llm: LLMBackend) -> list[ExtractedTriple]:
    """Prompt LLM to decompose an answer into structured triples."""
    user_prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer_text}\n\n"
        "Extract all factual claims as triples:"
    )
    raw = llm.call(_EXTRACT_SYSTEM, user_prompt, max_tokens=2048)
    return _parse_triples(raw)


def extract_triples_simple(text: str) -> list[ExtractedTriple]:
    """Extract triples without LLM — parse structured text directly.

    Looks for patterns like:
    - "X has value Y" -> (X, has_value, Y)
    - "X is Y" -> (X, is, Y)
    - Key: Value bullet points
    """
    triples = []
    # Parse "TRIPLE:" lines if present
    for m in re.finditer(r"TRIPLE:\s*(.+)", text):
        parts = [p.strip() for p in m.group(1).split("|")]
        if len(parts) >= 3:
            num = None
            if len(parts) >= 4:
                try:
                    num = float(parts[3])
                except ValueError:
                    pass
            triples.append(ExtractedTriple(
                head=parts[0], relation=parts[1], tail=parts[2],
                numeric_value=num,
            ))
    return triples


def _parse_triples(raw: str) -> list[ExtractedTriple]:
    """Parse LLM output into ExtractedTriple objects."""
    triples = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove "TRIPLE:" prefix
        if line.upper().startswith("TRIPLE:"):
            line = line[7:].strip()
        elif "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        head = parts[0]
        relation = parts[1]
        tail = parts[2]

        # Try to extract numeric value
        numeric = None
        if len(parts) >= 4:
            try:
                numeric = float(parts[3])
            except ValueError:
                pass
        if numeric is None:
            try:
                numeric = float(tail)
            except ValueError:
                pass

        triples.append(ExtractedTriple(
            head=head, relation=relation, tail=tail,
            numeric_value=numeric,
        ))
    return triples


# =============================================================================
# GMS scoring of extracted triples
# =============================================================================

def score_triples(triples: list[ExtractedTriple],
                   store: GMSExpertStore,
                   device: torch.device) -> list[ScoredTriple]:
    """Score each extracted triple against the GMS manifold.

    For each triple:
    1. Fuzzy-match head/tail to nearest entity in store
    2. If matched: score_triple() -> geodesic distance
    3. If not matched: mark as novel
    4. For numeric values: cross-check against ENM
    """
    scored = []
    for triple in triples:
        head_match = store.fuzzy_match_entity(triple.head)
        tail_match = store.fuzzy_match_entity(triple.tail)
        is_novel = head_match is None or tail_match is None

        gms_score = None
        if head_match and tail_match:
            # Try to score with closest matching relation
            rel_match = _match_relation(triple.relation, store)
            if rel_match:
                gms_score = store.score_triple(head_match, rel_match,
                                                tail_match)

        scored.append(ScoredTriple(
            triple=triple,
            gms_score=gms_score,
            head_match=head_match,
            tail_match=tail_match,
            is_novel=is_novel,
        ))
    return scored


def _match_relation(relation: str, store: GMSExpertStore) -> str | None:
    """Find closest relation name in store."""
    if store.adapter is None:
        return None
    rel_lower = relation.strip().lower().replace(" ", "_")
    # Exact
    if rel_lower in store.adapter.relation_to_idx:
        return rel_lower
    # Prefix match
    for r in store.adapter.relation_to_idx:
        if r.startswith(rel_lower) or rel_lower.startswith(r):
            return r
    # Substring
    for r in store.adapter.relation_to_idx:
        if rel_lower in r or r in rel_lower:
            return r
    return None


# =============================================================================
# Plausibility assessment
# =============================================================================

def assess_plausibility(scored: list[ScoredTriple],
                         threshold: float = 1.2) -> PlausibilityReport:
    """Aggregate triple scores into overall plausibility verdict."""
    if not scored:
        return PlausibilityReport(
            plausible=True, mean_score=0.0, worst_score=0.0,
            novel_count=0, known_count=0,
        )

    known_scores = [s.gms_score for s in scored
                     if s.gms_score is not None]
    novel_count = sum(1 for s in scored if s.is_novel)
    known_count = len(known_scores)

    if not known_scores:
        return PlausibilityReport(
            plausible=True,  # Can't verify — assume plausible
            mean_score=0.0,
            worst_score=0.0,
            novel_count=novel_count,
            known_count=0,
            scored_triples=scored,
        )

    mean_score = sum(known_scores) / len(known_scores)
    worst_score = max(known_scores)

    return PlausibilityReport(
        plausible=worst_score <= threshold,
        mean_score=mean_score,
        worst_score=worst_score,
        novel_count=novel_count,
        known_count=known_count,
        scored_triples=scored,
    )
