# SPDX-License-Identifier: Apache-2.0
"""Coverage monitor — make triple-mediated retrieval's blind spots measurable.

Triple-mediated retrieval can only answer over text that was triplified at ingest
(``GEODE_RAG_DESIGN.md`` §3, §7). A document region with body text but zero
triples is a genuine blind spot: a question whose answer lives there will not
bind and the pipeline will (correctly) abstain. This module reports those regions
so the gap is *known* rather than silent — the honest, bank-grade alternative to
papering over it with distrusted dense retrieval.

A "region" is a markdown section (a header and the body until the next header).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from knowlytix.knowledge.geode.provenance import ProvenanceLedger

__all__ = ["RegionCoverage", "CoverageReport", "coverage_report",
           "GraphCoverage", "graph_coverage"]

_HEADER_RE = re.compile(r"^\s*#+\s")

# Schema-declaration relations whose *head* is a relation/value token, not a
# subject entity (``- has_status is_functional True`` -> ``(has_status,
# is_functional, True)``; ``- High opposite_of Low``). They are excluded from
# the entity universe so a declaration is never mistaken for an orphan entity.
_SCHEMA_RELATIONS = frozenset({"is_functional", "opposite_of"})


@dataclass
class RegionCoverage:
    title: str
    line_start: int          # 1-based, the header line
    line_end: int            # inclusive, line before the next header / EOF
    body_lines: int          # non-blank, non-header content lines
    triple_count: int        # content triples whose provenance lands here

    @property
    def covered(self) -> bool:
        return self.triple_count > 0

    @property
    def blind_spot(self) -> bool:
        """Has body text but no triples — unreachable by triple-mediation."""
        return self.body_lines > 0 and self.triple_count == 0


@dataclass
class CoverageReport:
    regions: list[RegionCoverage] = field(default_factory=list)
    unaligned_triples: int = 0   # triples whose provenance did not resolve

    @property
    def blind_spots(self) -> list[RegionCoverage]:
        return [r for r in self.regions if r.blind_spot]

    @property
    def coverage_ratio(self) -> float:
        """Fraction of content-bearing regions that have >=1 triple."""
        content = [r for r in self.regions if r.body_lines > 0]
        if not content:
            return 1.0
        return sum(r.covered for r in content) / len(content)

    def as_dict(self) -> dict:
        return {
            "coverage_ratio": self.coverage_ratio,
            "regions": [
                {"title": r.title, "lines": [r.line_start, r.line_end],
                 "body_lines": r.body_lines, "triples": r.triple_count,
                 "blind_spot": r.blind_spot}
                for r in self.regions
            ],
            "blind_spots": [r.title for r in self.blind_spots],
            "unaligned_triples": self.unaligned_triples,
        }


def _sections(text: str) -> list[tuple[str, int, int]]:
    """Return ``(title, start_line, end_line)`` per markdown header section."""
    lines = text.split("\n")
    heads = [(i + 1, ln) for i, ln in enumerate(lines) if _HEADER_RE.match(ln)]
    out: list[tuple[str, int, int]] = []
    for k, (start, ln) in enumerate(heads):
        end = (heads[k + 1][0] - 1) if k + 1 < len(heads) else len(lines)
        out.append((ln.lstrip("#").strip(), start, end))
    return out


def _body_lines(text: str, start: int, end: int) -> int:
    lines = text.split("\n")[start:end]  # exclude the header line itself
    return sum(1 for ln in lines if ln.strip() and not _HEADER_RE.match(ln))


def coverage_report(store, *, exclude_relations=("in_section",)) -> CoverageReport:
    """Compute per-section triple coverage for a built store.

    Content triples (excluding structural relations like ``in_section``) are
    bucketed by the source line their provenance resolves to.
    """
    text = getattr(store, "markdown", "") or ""
    if not text:
        return CoverageReport()
    ledger = ProvenanceLedger.from_text(text)
    sections = _sections(text)

    counts = [0] * len(sections)
    unaligned = 0
    for h, r, t in store.triples:
        if r in exclude_relations:
            continue
        line_no = ledger.resolve(h, r, t).line_no
        if line_no < 0:
            unaligned += 1
            continue
        for i, (_title, start, end) in enumerate(sections):
            if start <= line_no <= end:
                counts[i] += 1
                break

    regions = [
        RegionCoverage(title=title, line_start=start, line_end=end,
                       body_lines=_body_lines(text, start, end),
                       triple_count=counts[i])
        for i, (title, start, end) in enumerate(sections)
    ]
    return CoverageReport(regions=regions, unaligned_triples=unaligned)


# --- entity / relation completeness diagnostics ----------------------------
#
# Region coverage (above) answers "can a question about this section bind to
# anything?". These diagnostics answer the complementary ingest-completeness
# question the region view cannot see: are there named subjects the extractor
# left with no facts, relations represented by a single (possibly stray) edge,
# or relations the document *declares* in a schema bullet but never populates?
# None of this proves completeness -- there is no gold schema to diff against --
# but it makes the under-extraction signals measurable and auditable.


def _rel_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _structural_relations() -> frozenset:
    """Relations that are organizational, not facts *about* an entity.

    Anchored on GEODE's own fact-only filter (``DEFAULT_NOISE_RELATIONS``) so the
    two stay in sync, extended with the cross-reference / metadata edges
    (``opposite_of``, alias/name) that likewise carry no attribute value. Lazy
    import keeps the module light and avoids an import cycle with ``geode.rag``.
    """
    from knowlytix.knowledge.geode.rag import DEFAULT_NOISE_RELATIONS
    return DEFAULT_NOISE_RELATIONS | {
        "opposite_of", "has_alias", "has_name", "has_policy_name"}


@dataclass
class GraphCoverage:
    """Entity/relation completeness diagnostics for an extracted graph."""

    n_entities: int                       # distinct nodes (head or tail, excl. schema)
    n_subject_entities: int               # distinct triple heads (excl. schema)
    n_fact_entities: int                  # heads with >=1 fact (content) triple
    orphan_entities: list                 # heads reachable only via structural edges
    relation_fact_counts: dict            # content relation -> #facts (desc)
    singleton_relations: list             # content relations with exactly one fact
    declared_relations: list              # relations declared ``is_functional``
    unpopulated_declared: list            # declared but never used as a fact edge

    @property
    def orphan_ratio(self) -> float:
        """Fraction of subject entities that carry no fact."""
        if self.n_subject_entities == 0:
            return 0.0
        return len(self.orphan_entities) / self.n_subject_entities

    def as_dict(self) -> dict:
        return {
            "n_entities": self.n_entities,
            "n_subject_entities": self.n_subject_entities,
            "n_fact_entities": self.n_fact_entities,
            "orphan_ratio": self.orphan_ratio,
            "orphan_entities": self.orphan_entities,
            "relation_fact_counts": self.relation_fact_counts,
            "singleton_relations": self.singleton_relations,
            "declared_relations": self.declared_relations,
            "unpopulated_declared": self.unpopulated_declared,
        }


def graph_coverage(triples) -> GraphCoverage:
    """Compute entity/relation completeness diagnostics from a triple set.

    Pass the **full** graph *before* the fact-only noise filter --- the
    corrected loop output (``LoopResult.triples``) or a raw
    ``ingest_markdown(...).triples``. The *served* store
    (``store.triples``) is unsuitable: :func:`store_from_triples` strips the
    structural/schema edges (``in_section``, ``is_functional``, ...) that orphan
    and declared-relation detection depend on, so they would always come back
    empty on a built store.

    Structural relations (:func:`_structural_relations`) do not count as facts:
    a head reachable only through them is an *orphan* (a named subject with no
    attribute), and a relation declared ``is_functional`` but never used as a
    fact edge is *unpopulated* (declared schema the extractor did not fill).
    """
    structural = _structural_relations()
    nodes: set = set()
    head_entities: set = set()
    fact_heads: set = set()
    used_relations: set = set()
    declared: set = set()
    rel_counts: dict = {}

    for h, r, t in triples:
        if r in _SCHEMA_RELATIONS:
            if r == "is_functional":
                declared.add(h)   # the head of an is_functional bullet is the relation
            continue              # schema head/tail are tokens, not subject entities
        head_entities.add(h)
        nodes.add(h)
        nodes.add(t)
        used_relations.add(r)
        if r not in structural:
            fact_heads.add(h)
            rel_counts[r] = rel_counts.get(r, 0) + 1

    unpopulated = []
    for d in sorted(declared):
        slug = _rel_slug(d)
        if not (d in used_relations or slug in used_relations
                or f"has_{slug}" in used_relations):
            unpopulated.append(d)

    relation_fact_counts = dict(
        sorted(rel_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    singletons = sorted(r for r, c in rel_counts.items() if c == 1)

    return GraphCoverage(
        n_entities=len(nodes),
        n_subject_entities=len(head_entities),
        n_fact_entities=len(fact_heads),
        orphan_entities=sorted(head_entities - fact_heads),
        relation_fact_counts=relation_fact_counts,
        singleton_relations=singletons,
        declared_relations=sorted(declared),
        unpopulated_declared=unpopulated,
    )
