# SPDX-License-Identifier: Apache-2.0
"""Scoring and LLM response parsing — domain-agnostic."""

import re
import numpy as np
from dataclasses import dataclass


@dataclass
class ScoreResult:
    correct: bool
    partial_score: float  # 0.0 to 1.0
    detail: str


def _norm_set_item(x: str) -> str:
    """Normalize a set item for comparison: lowercase, strip .0 suffix."""
    s = str(x).strip().lower()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _parse_multihop_str(text: str, ground_truth: tuple):
    """Try to parse a structured multi-hop answer string into a tuple.

    Expected formats:
        ENTITY: CaliforniaHousing, VALUE: 8, RELATED: 20640
        ENTITY: CaliforniaHousing, VALUE: 8.0, RELATED: a, b, c
    """
    text_clean = text.strip().strip("*").strip()

    # Try structured format
    entity_m = re.search(r"ENTITY:\s*([^,]+)", text_clean, re.IGNORECASE)
    value_m = re.search(r"VALUE:\s*([\d.eE+-]+)", text_clean, re.IGNORECASE)
    related_m = re.search(r"RELATED:\s*(.+)", text_clean, re.IGNORECASE)

    if entity_m and value_m:
        entity = entity_m.group(1).strip()
        try:
            value = float(value_m.group(1))
        except ValueError:
            return text  # can't parse, return original

        if related_m and isinstance(ground_truth[2], set):
            items = [s.strip().strip("'\"") for s in related_m.group(1).split(",")]
            items = {s for s in items if s and s.lower() not in ("none", "n/a", "")}
            return (entity, value, items)
        elif related_m and isinstance(ground_truth[2], dict):
            return (entity, value, {})  # partial match at best
        else:
            # No RELATED found — try with empty set
            return (entity, value, set())

    return text  # couldn't parse, return original string


def score_answer(answer, ground_truth, float_tol=1e-6) -> ScoreResult:
    """Score an answer against ground truth."""
    if answer is None:
        return ScoreResult(False, 0.0, "No answer")

    if isinstance(ground_truth, float):
        if isinstance(answer, (np.ndarray,)):
            answer = float(answer.item()) if answer.size == 1 else float(answer[0])
        if isinstance(answer, (int, float)):
            if abs(answer - ground_truth) < float_tol:
                return ScoreResult(True, 1.0, f"Exact: {answer}")
            elif abs(answer - ground_truth) < 0.01:
                return ScoreResult(False, 0.5, f"Close: {answer} vs {ground_truth}")
            else:
                return ScoreResult(False, 0.0, f"Wrong: {answer} vs {ground_truth}")

    if isinstance(ground_truth, bool):
        if isinstance(answer, bool) and answer == ground_truth:
            return ScoreResult(True, 1.0, f"Correct: {answer}")
        return ScoreResult(False, 0.0, f"Wrong: {answer} vs {ground_truth}")

    if isinstance(ground_truth, set):
        if isinstance(answer, set):
            # Set comparison is case-insensitive and tolerates ".0" suffix
            # (graph-derived ground truths are canonicalized to lowercase;
            # LLM answers are usually upper-case). Normalize both sides.
            a_norm = {_norm_set_item(x) for x in answer}
            g_norm = {_norm_set_item(x) for x in ground_truth}
            if a_norm == g_norm:
                return ScoreResult(True, 1.0, "Exact set match")
            if g_norm and a_norm:
                overlap = a_norm & g_norm
                partial = len(overlap) / len(g_norm)
                if overlap == g_norm:
                    return ScoreResult(True, 1.0, f"Superset: {answer}")
                return ScoreResult(False, partial,
                                   f"Partial {partial:.0%}: got {answer}, expected {ground_truth}")
        return ScoreResult(False, 0.0, f"Wrong: {answer} vs {ground_truth}")

    if isinstance(ground_truth, int):
        if isinstance(answer, (int, float, np.integer, np.floating)):
            if int(answer) == ground_truth:
                return ScoreResult(True, 1.0, f"Exact: {int(answer)}")
            return ScoreResult(False, 0.0, f"Wrong: {answer} vs {ground_truth}")

    if isinstance(ground_truth, str):
        ans_clean = str(answer).strip().strip("*").strip("_").strip().lower()
        gt_clean = ground_truth.strip().strip("*").strip("_").strip().lower()
        if ans_clean == gt_clean:
            return ScoreResult(True, 1.0, "Exact match")
        return ScoreResult(False, 0.0, f"Wrong: '{answer}' vs '{ground_truth}'")

    if isinstance(ground_truth, tuple):
        # If answer is a string, try to parse structured multi-hop format:
        #   "ENTITY: X, VALUE: Y, RELATED: a, b, c"
        if isinstance(answer, str) and len(ground_truth) == 3:
            answer = _parse_multihop_str(answer, ground_truth)

        if isinstance(answer, tuple) and len(answer) == len(ground_truth):
            all_ok = True
            for a, g in zip(answer, ground_truth):
                if isinstance(g, float):
                    try:
                        if abs(float(a) - g) > 1e-3:
                            all_ok = False
                            break
                    except (ValueError, TypeError):
                        all_ok = False
                        break
                elif isinstance(g, set):
                    a_set = a if isinstance(a, set) else {str(a).strip()}
                    # Normalize: strip .0 suffix, lowercase
                    a_norm = {_norm_set_item(x) for x in a_set}
                    g_norm = {_norm_set_item(x) for x in g}
                    if not a_norm >= g_norm:
                        all_ok = False
                        break
                elif isinstance(g, dict):
                    # Pattern 2: dict ground truth — check entity overlap
                    if not isinstance(a, dict):
                        all_ok = False
                        break
                elif isinstance(g, bool):
                    if a != g:
                        all_ok = False
                        break
                elif isinstance(g, str):
                    if str(a).strip().lower() != g.strip().lower():
                        all_ok = False
                        break
                else:
                    if a != g:
                        all_ok = False
                        break
            if all_ok:
                return ScoreResult(True, 1.0, f"Exact tuple: {answer}")
            return ScoreResult(False, 0.0, f"Partial tuple: {answer} vs {ground_truth}")
        return ScoreResult(False, 0.0, f"Wrong: {answer} vs {ground_truth}")

    return ScoreResult(False, 0.0, f"Cannot score: {type(answer)} vs {type(ground_truth)}")


# ============================================================================
# LLM RESPONSE PARSERS — one per answer_type
# ============================================================================

def parse_float(text):
    """Extract a float from LLM response."""
    m = re.search(r"ANSWER:\s*([\d.eE+-]+)", text)
    if m:
        return float(m.group(1))
    nums = re.findall(r"\d+\.\d+", text)
    return float(nums[-1]) if nums else None


def parse_int(text):
    """Extract an integer from LLM response."""
    m = re.search(r"ANSWER:\s*(\d+)", text)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\b(\d+)\b", text)
    reasonable = [int(n) for n in nums if 1 <= int(n) <= 99999]
    return reasonable[-1] if reasonable else None


def parse_bool(text):
    """Extract a boolean from LLM response."""
    m = re.search(r"ANSWER:\s*(true|false|yes|no)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower() in ("true", "yes")
    lower = text.lower()
    if "yes" in lower or "passes" in lower or "does pass" in lower:
        if "no" not in lower and "does not" not in lower:
            return True
    if "false" in lower or "does not" in lower or "fails" in lower:
        return False
    return None


_SET_ITEM_SPLITTER = re.compile(
    r",|;|\n|\s+and\s+|\s+or\s+|•|·", re.IGNORECASE
)
_BULLET_PREFIX = re.compile(r"^[-*•·]\s*")


def parse_set_str(text):
    """Extract a set of strings from an LLM response.

    Splits on commas, semicolons, newlines, the conjunctions "and"/"or",
    and common bullet markers; strips markdown emphasis (`**`, `*`, `_`)
    and list prefixes from each item. The "ANSWER:" payload is the
    preferred source but the parser falls back to the full text.
    """
    m = re.search(r"ANSWER:\s*(.+?)(?:\n\s*\n|\Z)", text, re.DOTALL)
    payload = m.group(1) if m else text
    pieces = _SET_ITEM_SPLITTER.split(payload)
    items: list[str] = []
    for raw in pieces:
        s = raw.strip()
        s = _BULLET_PREFIX.sub("", s)
        s = s.strip("*_'\" \t")
        if s and len(s) < 100:
            items.append(s)
    return set(items) if items else set()


def _strip_markdown(s: str) -> str:
    """Strip markdown bold/italic markers from an answer string."""
    return s.strip().strip("*").strip("_").strip()


def parse_str(text):
    """Extract a string answer."""
    m = re.search(r"ANSWER:\s*(.+)", text)
    return _strip_markdown(m.group(1)) if m else _strip_markdown(text)


# Registry
PARSERS = {
    "float": parse_float,
    "int": parse_int,
    "bool": parse_bool,
    "set_str": parse_set_str,
    "str": parse_str,
}


# ---------------------------------------------------------------------------
# Path-validity scorer
# ---------------------------------------------------------------------------

def score_path_validity(
    chain: list[tuple[str, str, str]],
    triples,
) -> ScoreResult:
    """Verify every (head, relation, tail) edge in a retrieved chain exists
    in the source store's triple set.

    This is the canonical complement to :func:`score_answer` for multi-hop
    benchmarks: ``score_answer`` checks whether the retrieved tail string
    matches the labeled ground truth (strict exact match); this scorer
    checks whether the retrieved CHAIN is a valid walk in the graph,
    regardless of which fork the bridge happened to label.

    For multi-valued relations, ``score_answer`` can mark a question wrong
    even when the model picked a different but graph-truthful tail. Use
    both scorers together — exact for strict reproducibility, path-validity
    for graph-truth coverage.

    Parameters
    ----------
    chain
        Ordered list of ``(head, relation, tail)`` triples representing the
        retrieved walk. For a 2-hop query ``source -> r1 -> mid -> r2 -> t``
        this is ``[(source, r1, mid), (mid, r2, t)]``.
    triples
        Iterable or set of ``(h, r, t)`` triples that exist in the store.
        Passing a ``set`` makes lookup O(1); a list works but is O(N) per hop.

    Returns
    -------
    ScoreResult
        ``correct`` is True iff every hop is a real edge.
        ``partial_score`` is the fraction of hops that are real edges.
        ``detail`` lists the failing hops, if any.
    """
    if not isinstance(triples, set):
        triples = set(triples)
    if not chain:
        return ScoreResult(correct=False, partial_score=0.0,
                           detail="empty chain")
    per_hop_valid: list[bool] = []
    failing: list[str] = []
    for h, r, t in chain:
        ok = (h, r, t) in triples
        per_hop_valid.append(ok)
        if not ok:
            failing.append(f"{h} --{r}--> {t}")
    n_ok = sum(per_hop_valid)
    n = len(chain)
    if n_ok == n:
        detail = f"all {n} hop(s) are real edges in the store"
    else:
        detail = f"{n_ok}/{n} hops valid; missing: " + "; ".join(failing)
    return ScoreResult(
        correct=(n_ok == n),
        partial_score=n_ok / n,
        detail=detail,
    )
