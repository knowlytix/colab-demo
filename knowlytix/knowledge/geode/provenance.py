# SPDX-License-Identifier: Apache-2.0
"""Span<->triple provenance for GEODE.

The extractor stores triples as canonicalized ``(h, r, t)`` tuples and discards
where each came from. GEODE's loop needs the reverse: when the geometric critic
flags a triple, point the actor back at the exact source span to re-read.

:class:`ProvenanceLedger` reconstructs that map post-hoc by re-aligning each
triple to the markdown source — table cells (precise char span), schema bullets,
and section headers.

Scope: this aligns regex-extracted *structured* triples (tables/bullets/
headers). Prose triples from LLM extraction have no structural anchor and need
spans captured at extraction time. On documents with recurring row labels
(e.g. financial scenario tables reusing "Cumulative"/"Total"), ``(entity,
relation)`` keys collide — robust provenance there requires table-scoped entity
IDs at extraction time, not post-hoc alignment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Provenance", "ProvenanceLedger", "is_consistent", "canon",
           "value_canon"]


def canon(s: str) -> str:
    """Match ``DocumentGraph._canon_entity``: collapse whitespace, lowercase."""
    return " ".join(str(s).split()).lower()


def value_canon(s: str) -> str:
    """Separator-insensitive canonical form for *comparing values* (not keys).

    Like :func:`canon` but also unifies space / underscore / hyphen, so a value
    an LLM emits as prose (``"dispute unit"``) compares equal to the graph's
    stored slug (``"dispute_unit"``). Use this ONLY to compare a generated value
    against a stored one (answer verification, eval scoring). For entity-key
    canonicalization use :func:`canon`, which must stay in lockstep with the
    graph's stored keys (``DocumentGraph._canon_entity``)."""
    return canon(re.sub(r"[_\-]+", " ", str(s)))


def _col_slug(header_cell: str) -> str:
    return header_cell.strip().replace(" ", "_").lower()


def _section_slug(header_text: str) -> str:
    tag = re.sub(r"^\d+\.\s*", "", header_text).strip()
    tag = tag.replace(" ", "_").lower()
    return re.sub(r"[^a-z0-9_]", "", tag)


_SCHEMA_RE = re.compile(
    r"^\s*[-*]\s*`?(?P<lhs>[^`\n]+?)`?\s+"
    r"(?P<rel>opposite_of|is_functional)\s+`?(?P<rhs>[^`\n]+?)`?\s*$"
)
_NUM_RE = re.compile(r"-?[\d,]*\.?\d+")


@dataclass
class Provenance:
    """Where an extracted triple came from in the source document."""

    head: str
    relation: str
    tail: str
    source_path: str
    line_no: int           # 1-based; -1 when unaligned
    char_start: int        # absolute file offset; -1 when unaligned
    char_end: int
    raw: str               # exact source substring (cell / line)
    method: str            # table_cell | schema_bullet | section_header | unaligned

    def location(self) -> str:
        return f"{self.source_path}:{self.line_no}:{self.char_start}-{self.char_end}"


def _line_index(text: str) -> list[tuple[int, int, str]]:
    out, off = [], 0
    for i, line in enumerate(text.split("\n"), start=1):
        out.append((i, off, line))
        off += len(line) + 1
    return out


def _cells_with_spans(line: str, line_start: int) -> list[tuple[str, int, int]]:
    segs, idx = [], 0
    for part in line.split("|"):
        seg_start, seg_end = idx, idx + len(part)
        idx = seg_end + 1
        stripped = part.strip()
        if stripped == "" and (seg_start == 0 or seg_end >= len(line)):
            continue
        lead = len(part) - len(part.lstrip())
        c_start = line_start + seg_start + lead
        segs.append((stripped, c_start, c_start + len(stripped)))
    return segs


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-+:?", c) for c in cells if c != "")


class ProvenanceLedger:
    """Maps each ``(h, r, t)`` triple to a source-span :class:`Provenance`."""

    def __init__(self, md_path: str):
        with open(md_path, encoding="utf-8") as f:
            self._init_from_text(f.read(), md_path)

    @classmethod
    def from_text(cls, text: str, source_path: str = "") -> "ProvenanceLedger":
        """Build a ledger from in-memory markdown (no file read).

        Lets the RAG pipeline align triples against ``store.markdown`` directly
        rather than depending on the on-disk ``documents/combined.md`` copy.
        """
        obj = cls.__new__(cls)
        obj._init_from_text(text, source_path)
        return obj

    def _init_from_text(self, text: str, source_path: str) -> None:
        self.path = source_path
        self.text = text
        self._lines = _line_index(self.text)
        self._table: dict[tuple[str, str], tuple[int, int, int, str]] = {}
        self._schema: dict[tuple[str, str], tuple[int, int, int, str]] = {}
        self._section: dict[str, tuple[int, int, int, str]] = {}
        self._index()

    def _index(self) -> None:
        block: list[tuple[int, int, str]] = []

        def flush(blk):
            rows = [(ln, st, _cells_with_spans(s, st)) for ln, st, s in blk]
            rows = [r for r in rows if not _is_separator([c[0] for c in r[2]])]
            if len(rows) < 2:
                return
            header = [c[0] for c in rows[0][2]]
            for ln, _st, cells in rows[1:]:
                if not cells:
                    continue
                row_entity = canon(cells[0][0])
                for ci in range(1, min(len(cells), len(header))):
                    rel = "has_" + _col_slug(header[ci])
                    txt, cs, ce = cells[ci]
                    self._table[(row_entity, rel)] = (ln, cs, ce, txt)

        for ln, st, s in self._lines:
            if s.strip().startswith("|"):
                block.append((ln, st, s))
            elif block:
                flush(block)
                block = []
        if block:
            flush(block)

        for ln, st, s in self._lines:
            m = _SCHEMA_RE.match(s)
            if m:
                lhs, rel = canon(m.group("lhs")), m.group("rel")
                lead = len(s) - len(s.lstrip())
                self._schema[(lhs, rel)] = (ln, st + lead, st + len(s.rstrip()),
                                            s.strip())
            h = s.lstrip()
            if h.startswith("#"):
                name = h.lstrip("#").strip()
                self._section[_section_slug(name)] = (ln, st, st + len(s.rstrip()),
                                                       s.strip())

    def resolve(self, head: str, relation: str, tail: str) -> Provenance:
        h, t = canon(head), canon(tail)
        key = (h, relation)
        if relation in ("opposite_of", "is_functional") and key in self._schema:
            ln, cs, ce, raw = self._schema[key]
            method = "schema_bullet"
        elif relation == "in_section" and t in self._section:
            ln, cs, ce, raw = self._section[t]
            method = "section_header"
        elif key in self._table:
            ln, cs, ce, raw = self._table[key]
            method = "table_cell"
        else:
            return Provenance(h, relation, t, self.path, -1, -1, -1, "", "unaligned")
        return Provenance(h, relation, t, self.path, ln, cs, ce, raw, method)

    def build(self, triples) -> dict[tuple[str, str, str], Provenance]:
        return {(h, r, t): self.resolve(h, r, t) for h, r, t in triples}

    def line(self, line_no: int) -> str:
        return self._lines[line_no - 1][2] if 1 <= line_no <= len(self._lines) else ""

    def table_block(self, line_no: int) -> str:
        """Full contiguous markdown table containing ``line_no`` (a bounded
        span to hand the actor for localized re-reasoning)."""
        n = len(self._lines)
        if not (1 <= line_no <= n):
            return ""
        lo = hi = line_no
        while lo > 1 and self._lines[lo - 2][2].strip().startswith("|"):
            lo -= 1
        while hi < n and self._lines[hi][2].strip().startswith("|"):
            hi += 1
        return "\n".join(self._lines[i - 1][2] for i in range(lo, hi + 1))


def is_consistent(p: Provenance) -> bool:
    """Does the resolved source span actually support the triple?"""
    if p.method == "unaligned":
        return False
    if p.method == "section_header":
        return _section_slug(p.raw.lstrip("#").strip()) == p.tail
    if p.method == "schema_bullet":
        return p.tail in canon(p.raw)
    if canon(p.raw) == p.tail:
        return True
    nums = _NUM_RE.findall(p.raw.replace(",", ""))
    try:
        return bool(nums) and abs(float(nums[0]) - float(p.tail)) < 1e-6
    except ValueError:
        return False
