# SPDX-License-Identifier: Apache-2.0
"""Evaluation + calibration for the RAG pipeline (P4).

Two tools to make extraction reliability *measurable* and *tunable*:

* :func:`evaluate` — run a labeled question set through a pipeline and report
  bind-rate, decision correctness, answer accuracy, and abstention precision.
  Use it to quantify the lift from schema-grounded extraction / the repair loop.
* :func:`calibrate_bind_threshold` — fit the embedding-binding similarity
  threshold from labeled positive/negative term pairs (the literal calibration
  step), so synonyms bind and unrelated terms don't.
* :func:`calibrate_accept_threshold` — fit the retrieval-confidence accept gate
  (tau) on a labeled accept/abstain cohort by balanced accuracy under a
  false-accept ceiling, the operating point the pipeline reads.
* :func:`benchmark_query_parse` — isolate the parse→bind step and compare
  parsers (geometric vs LLM) on store-derived questions, scoring recovery of the
  asked ``(head, relation)`` and latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowlytix.knowledge.geode.provenance import value_canon
from knowlytix.knowledge.rag.query_triples import QueryTriple

__all__ = ["EvalCase", "EvalReport", "evaluate", "calibrate_bind_threshold",
           "calibrate_accept_threshold", "benchmark_query_parse"]


@dataclass
class EvalCase:
    question: str
    expected_answer: str | None = None        # substring/value expected in answer
    expect_decision: str | None = None        # "accept" | "abstain" (if known)


@dataclass
class CaseResult:
    case: EvalCase
    decision: str
    bound: bool
    answer_correct: bool | None
    decision_correct: bool | None


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    def _rate(self, pred) -> float:
        vals = [pred(r) for r in self.results]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def bind_rate(self) -> float:
        return self._rate(lambda r: r.bound)

    @property
    def accept_rate(self) -> float:
        return self._rate(lambda r: r.decision == "accept")

    @property
    def answer_accuracy(self) -> float:
        """Accuracy over cases that carried an expected_answer."""
        return self._rate(lambda r: r.answer_correct)

    @property
    def decision_accuracy(self) -> float:
        return self._rate(lambda r: r.decision_correct)

    def as_dict(self) -> dict:
        return {
            "n": len(self.results),
            "bind_rate": self.bind_rate,
            "accept_rate": self.accept_rate,
            "answer_accuracy": self.answer_accuracy,
            "decision_accuracy": self.decision_accuracy,
        }


def _answer_correct(ans, expected: str) -> bool:
    if expected is None:
        return None
    exp = value_canon(expected)
    if exp in value_canon(ans.answer):
        return True
    return any(exp == value_canon(f.tail) for f in ans.sources)


def evaluate(pipeline, cases: list[EvalCase]) -> EvalReport:
    """Run cases through ``pipeline.query`` and score them."""
    results: list[CaseResult] = []
    for case in cases:
        ans = pipeline.query(case.question)
        bound = any(b.bound for b in ans.bound_triples)
        answer_correct = (None if case.expected_answer is None
                          else _answer_correct(ans, case.expected_answer))
        decision_correct = (None if case.expect_decision is None
                            else ans.decision == case.expect_decision)
        results.append(CaseResult(case, ans.decision, bound,
                                  answer_correct, decision_correct))
    return EvalReport(results)


def calibrate_bind_threshold(
    binder, positives: list[tuple[str, str]], negatives: list[str], *,
    grid: tuple[float, ...] = tuple(i / 20 for i in range(1, 20)),
) -> tuple[float, float]:
    """Fit ``binder.bind_threshold`` to best separate positives from negatives.

    Args:
        binder: an embedding-mode :class:`TripleBinder`.
        positives: ``(term, expected_entity)`` that *should* bind.
        negatives: terms that should *not* bind to any entity.
        grid: candidate thresholds to sweep.

    Returns ``(best_threshold, accuracy)`` and sets it on the binder.
    """
    best_th, best_acc = binder.bind_threshold, -1.0
    total = len(positives) + len(negatives)
    for th in grid:
        binder.bind_threshold = th
        tp = sum(binder.bind(QueryTriple(t, "_", "?")).head == exp
                 for t, exp in positives)
        tn = sum(binder.bind(QueryTriple(t, "_", "?")).head is None
                 for t in negatives)
        acc = (tp + tn) / total if total else 0.0
        if acc > best_acc:
            best_acc, best_th = acc, th
    binder.bind_threshold = best_th
    return best_th, best_acc


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (k of n)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / d
    return max(0.0, center - half), min(1.0, center + half)


def calibrate_accept_threshold(
    records: list[tuple[int, bool, float]], *, max_false_accept: float = 0.05,
) -> dict:
    """Fit the accept-gate threshold ``tau`` on a labeled accept/abstain cohort.

    Each record is ``(label, abstained, confidence)``: ``label`` is 1 for an
    answerable (ACCEPT) question and 0 for an out-of-scope / no-such-attribute
    (ABSTAIN) question; ``abstained`` is whether a pre-threshold gate (bind-check
    or verify) already abstained (independent of ``tau``); ``confidence`` is the
    pipeline's retrieval confidence. Collect them by running the cohort through a
    pipeline with ``accept_threshold=0`` so its own gate does not pre-filter.

    The objective is balanced accuracy ``0.5*(recall + (1 - false_accept))`` so the
    gate cannot win by abstaining on everything; the ``max_false_accept`` ceiling is
    a tie-breaker, not a hard constraint (a bimodal signal can carry an irreducible
    leak). For a clean bimodal split ``tau`` is set to the midpoint of the gap
    between the highest abstained and the lowest accepted confidence. Returns a
    payload ready to persist as ``rag_gate_calibration.json``.
    """
    def decide(abstained: bool, conf: float, tau: float) -> int:
        return 0 if abstained else (1 if conf >= tau else 0)

    n = len(records)
    n_pos = sum(1 for lbl, _, _ in records if lbl == 1)
    n_neg = n - n_pos
    grid = sorted({round(c, 3) for _, _, c in records} | {0.0, 1.0001})

    def score(tau: float):
        tp = sum(1 for lbl, ab, c in records if lbl == 1 and decide(ab, c, tau))
        fa = sum(1 for lbl, ab, c in records if lbl == 0 and decide(ab, c, tau))
        recall = tp / n_pos if n_pos else 0.0
        false_accept = fa / n_neg if n_neg else 0.0
        bal = 0.5 * (recall + (1.0 - false_accept))
        acc = (tp + (n_neg - fa)) / n if n else 0.0
        return bal, acc, recall, false_accept, false_accept <= max_false_accept

    ranked = sorted(grid, key=lambda t: (score(t)[0], score(t)[4], t), reverse=True)
    tau = ranked[0]
    bal, acc, recall, false_accept, meets = score(tau)
    accepted = [c for lbl, ab, c in records if not ab and decide(ab, c, tau)]
    abstained = [c for _, ab, c in records if ab or not decide(ab, c, tau)]
    if accepted and abstained and min(accepted) > max(abstained):
        tau = round(0.5 * (min(accepted) + max(abstained)), 4)
        bal, acc, recall, false_accept, meets = score(tau)

    ci_lo, ci_hi = _wilson_ci(int(round(acc * n)), n)
    return {
        "method": "accept_threshold_balanced_acc_v2",
        "max_false_accept": max_false_accept,
        "false_accept_ceiling_met": meets,
        "accept_threshold": tau,
        "balanced_accuracy": round(bal, 4),
        "accuracy": round(acc, 4),
        "recall": round(recall, 4),
        "false_accept": round(false_accept, 4),
        "accuracy_ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "cohort_n": n, "n_pos": n_pos, "n_neg": n_neg,
    }


def _parse_question(head: str, relation: str) -> str:
    """A deterministic, gold-anchored question for one fact triple."""
    base = relation[4:] if relation.startswith("has_") else relation
    rel = base.replace("_", " ").strip()
    return f"What is the {rel} for {head.replace('_', ' ').replace('/', ' ')}?"


def benchmark_query_parse(store, parsers: dict, *, binder=None, n: int = 0) -> dict:
    """Compare query parsers on the parse→bind step, holding the binder constant.

    Reverses one templated, gold-anchored question per distinct ``(head, relation)``
    fact in ``store`` (``in_section`` excluded), runs each parser in ``parsers``
    (``{name: parser}``, each with ``.extract(question) -> list[QueryTriple]``),
    binds the result with the shared ``binder`` (defaults to an embedding
    ``TripleBinder`` over the store) and scores whether the asked
    ``(head, relation)`` was recovered. Returns per-parser ``parse_rate``,
    ``bind_rate``, ``recover_rate`` and latency, so a geometric parser and an LLM
    parser are measured on the same questions and the same binder.
    """
    import time
    import statistics
    from knowlytix.knowledge.rag.binding import TripleBinder

    binder = binder or TripleBinder(store, mode="embedding")
    golds, seen = [], set()
    for h, r, _t in store.triples:
        if r == "in_section" or (h, r) in seen:
            continue
        seen.add((h, r))
        golds.append((h, r, _parse_question(h, r)))
    if n:
        golds = golds[:n]

    results: dict = {}
    for name, parser in parsers.items():
        per, lat = [], []
        n_parse = n_bind = n_recover = 0
        for head, rel, q in golds:
            t0 = time.perf_counter()
            try:
                qts = parser.extract(q)
            except Exception:  # noqa: BLE001 - a parser failure is a data point
                qts = []
            lat.append(time.perf_counter() - t0)
            bound = [binder.bind(qt) for qt in qts]
            parsed = len(qts) > 0
            any_bound = any(b.bound for b in bound)
            recovered = any(b.bound and b.head == head and b.relation == rel
                            for b in bound)
            n_parse += parsed
            n_bind += any_bound
            n_recover += recovered
            per.append({"q": q, "gold": [head, rel], "parsed": parsed,
                        "bound": any_bound, "recovered": recovered,
                        "triples": [[b.head, b.relation, b.tail] for b in bound]})
        m = len(golds)
        results[name] = {
            "n": m,
            "parse_rate": round(n_parse / m, 4) if m else 0.0,
            "bind_rate": round(n_bind / m, 4) if m else 0.0,
            "recover_rate": round(n_recover / m, 4) if m else 0.0,
            "latency_ms_mean": round(1000 * statistics.mean(lat), 1) if lat else 0.0,
            "latency_ms_p50": round(1000 * statistics.median(lat), 1) if lat else 0.0,
            "examples": sorted(per, key=lambda r: r["recovered"])[:10],
        }
    return {"n_questions": len(golds), "results": results}
