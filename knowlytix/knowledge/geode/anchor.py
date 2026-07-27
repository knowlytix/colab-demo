# SPDX-License-Identifier: Apache-2.0
"""External anchors for GEODE — catch what the geometry cannot.

The geometric critic catches errors that violate REDUNDANCY (relation
composition) or learned structure. It is blind to an isolated value with no
cross-check: a lone revenue figure can be wrong and internally consistent, so no
amount of geometry flags it. That class of error needs an EXTERNAL anchor — an
exact, declared, or arithmetic constraint over the numbers themselves.

Two anchors, both grounded in exact (ENM-style) numeric facts:

* :class:`SumConstraint` — a declared aggregate equals the sum of its parts
  (e.g. total revenue = sum of division revenues). A corrupted part breaks the
  sum. :meth:`AnchorChecker.auto_sum_constraints` derives these from a "Total"/
  "Cumulative" row, the way financial tables already declare them.
* duplicate-value contradiction — the same ``(entity, relation)`` asserted with
  two different exact values anywhere in the document (precisely localizable).

Anchors DETECT (and, for sums, can't always localize to the single wrong part —
that escalates to review); they complement, not replace, the geometric critic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from knowlytix.core.memory.enm import ENMKey, ExactNumericalMemory
from knowlytix.knowledge.geode.provenance import Provenance, ProvenanceLedger

__all__ = [
    "numeric_facts_from_triples",
    "enm_from_triples",
    "SumConstraint",
    "AnchorViolation",
    "AnchorChecker",
]

_NUM_RE = re.compile(r"-?[\d,]*\.?\d+")
Key = tuple[str, str]  # (entity, relation)

# Fixed timestamp for GEODE-built ENM keys. ENMKey folds its timestamp into the
# hash, so a default time.time() would make a fresh key on every write and never
# collide. A constant makes (entity, relation) the true identity — the same fact
# written twice resolves to one key (last-write-wins), which is what we want: the
# loop rewrites a fact when it corrects it, and duplicate detection runs over the
# raw triples (see AnchorChecker.check_duplicates), not over the register.
_ENM_TS = 0.0


def numeric_facts_from_triples(triples) -> dict[Key, float]:
    """Parse ``(entity, relation) -> float`` from triples with numeric tails.

    Mirrors the exact values the ENM register holds; a tail that is not a clean
    number is skipped.
    """
    facts: dict[Key, float] = {}
    for h, r, t in triples:
        m = _NUM_RE.search(str(t).replace(",", ""))
        if m:
            try:
                facts[(h, r)] = float(m.group())
            except ValueError:
                pass
    return facts


def enm_from_triples(triples) -> ExactNumericalMemory:
    """Build a real :class:`ExactNumericalMemory` register from triple tails.

    The numeric tails are stored losslessly (IEEE 754 float64 + SHA-256
    integrity) under ``ENMKey(type=relation, id=entity)`` — the same core
    register the full GMS pipeline populates, rather than the plain dict
    :func:`numeric_facts_from_triples` returns. The anchors read exact values
    back through the register's integrity-checked path (see
    :meth:`AnchorChecker.from_enm`), so a silently corrupted value surfaces as an
    error instead of a wrong sum.
    """
    facts = numeric_facts_from_triples(triples)
    enm = ExactNumericalMemory(capacity=len(facts) + 100)
    for (ent, rel), value in facts.items():
        enm.write(ENMKey(type=rel, id=ent, timestamp=_ENM_TS), value)
    return enm


@dataclass
class SumConstraint:
    """Declared constraint: ``total == sum(parts)`` (within ``tol``)."""

    total: Key
    parts: list[Key]
    tol: float = 1e-6


@dataclass
class AnchorViolation:
    kind: str                       # "sum" | "duplicate"
    message: str
    residual: float                 # magnitude of the inconsistency
    locations: list[Provenance] = field(default_factory=list)


class AnchorChecker:
    """Verify external numeric anchors against the document's exact facts."""

    def __init__(self, numeric_facts: dict[Key, float],
                 ledger: ProvenanceLedger | None = None):
        self.facts = numeric_facts
        self.ledger = ledger

    @classmethod
    def from_enm(cls, enm: ExactNumericalMemory,
                 ledger: ProvenanceLedger | None = None) -> "AnchorChecker":
        """Build a checker whose facts are read from a real ENM register.

        Each value is pulled through :meth:`ExactNumericalMemory.read_exact`,
        which verifies the stored SHA-256 hash — so the sum/duplicate anchors run
        on integrity-checked exact numbers, not on values re-parsed from text. A
        corrupted entry raises rather than producing a wrong constraint result.
        """
        facts: dict[Key, float] = {}
        for key in enm.keys():
            value = enm.read_exact(key)  # raises on integrity failure
            if value is not None:
                facts[(key.id, key.type)] = float(value.reshape(-1)[0])
        checker = cls(facts, ledger)
        checker.enm = enm
        return checker

    def _loc(self, key: Key) -> Provenance | None:
        if self.ledger is None:
            return None
        ent, rel = key
        return self.ledger.resolve(ent, rel, "")

    def auto_sum_constraints(
        self, total_entities: tuple[str, ...] = ("total", "cumulative"),
        min_parts: int = 2,
    ) -> list[SumConstraint]:
        """Derive sum constraints from a declared total/cumulative row: for each
        relation that a total-entity reports, the other rows are its parts."""
        by_rel: dict[str, list[Key]] = {}
        for (ent, rel) in self.facts:
            by_rel.setdefault(rel, []).append((ent, rel))
        out: list[SumConstraint] = []
        for rel, keys in by_rel.items():
            totals = [k for k in keys if k[0] in total_entities]
            parts = [k for k in keys if k[0] not in total_entities]
            if totals and len(parts) >= min_parts:
                out.append(SumConstraint(total=totals[0], parts=parts))
        return out

    def check_sum(self, c: SumConstraint, *, rel_tol: float = 1e-3
                  ) -> AnchorViolation | None:
        if c.total not in self.facts:
            return None
        parts = [self.facts[p] for p in c.parts if p in self.facts]
        if len(parts) < 2:
            return None
        total = self.facts[c.total]
        s = sum(parts)
        tol = max(c.tol, rel_tol * abs(total))
        if abs(s - total) <= tol:
            return None
        locs = [p for p in (self._loc(c.total), *[self._loc(k) for k in c.parts])
                if p is not None]
        return AnchorViolation(
            kind="sum",
            message=(f"declared total {c.total[0]}.{c.total[1]}={total:g} != "
                     f"sum(parts)={s:g} (off by {s - total:g}); a part value is "
                     f"likely wrong (candidates: {[k[0] for k in c.parts]})"),
            residual=abs(s - total),
            locations=locs,
        )

    def check_duplicates(self, triples) -> list[AnchorViolation]:
        """Same (entity, relation) asserted with two different exact values."""
        seen: dict[Key, set[float]] = {}
        for h, r, t in triples:
            m = _NUM_RE.search(str(t).replace(",", ""))
            if not m:
                continue
            try:
                seen.setdefault((h, r), set()).add(float(m.group()))
            except ValueError:
                pass
        out: list[AnchorViolation] = []
        for key, vals in seen.items():
            if len(vals) > 1:
                loc = self._loc(key)
                out.append(AnchorViolation(
                    kind="duplicate",
                    message=(f"{key[0]}.{key[1]} asserted with conflicting exact "
                             f"values {sorted(vals)}"),
                    residual=max(vals) - min(vals),
                    locations=[loc] if loc else [],
                ))
        return out

    def check_all(self, triples, constraints: list[SumConstraint] | None = None
                  ) -> list[AnchorViolation]:
        cons = constraints if constraints is not None else self.auto_sum_constraints()
        viol = [v for c in cons if (v := self.check_sum(c)) is not None]
        viol += self.check_duplicates(triples)
        return viol
