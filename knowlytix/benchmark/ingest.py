# SPDX-License-Identifier: Apache-2.0
"""
Generic Markdown-to-Graph Ingester.

Parses any markdown document with tables into a DocumentGraph:
  - Every numeric cell becomes an ENM entry
  - Table structure becomes KG triples
  - Pass/fail, weak/strong columns become typed triples
  - Bullet-list key-value pairs become ENM entries
  - Cross-table entity references are detected automatically

Domain-agnostic: works with model validation reports, fair lending analyses,
stress test results, financial statements, etc.
"""

import re
import warnings

from knowlytix.benchmark.graph import DocumentGraph


# ============================================================================
# MARKDOWN PARSING
# ============================================================================

def parse_md_table(text: str) -> list[dict]:
    """Parse a markdown table into list of row dicts."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return []

    header_line = None
    sep_idx = 0
    for i, line in enumerate(lines):
        cleaned = line.replace("|", "").replace(" ", "").replace("-", "").replace(":", "")
        if "|" in line and cleaned == "":
            header_line = lines[i - 1] if i > 0 else None
            sep_idx = i
            break

    if header_line is None:
        return []

    def split_row(line):
        parts = line.split("|")
        cells = [c.strip() for c in parts]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        # Strip markdown bold (**...**) from cell values
        cells = [re.sub(r'\*\*(.+?)\*\*', r'\1', c).strip() for c in cells]
        return cells

    headers = split_row(header_line)

    # Detect unnamed index column
    has_index = (headers[0] == "" or headers[0].strip() == "")

    rows = []
    for line in lines[sep_idx + 1:]:
        if not line.strip() or "|" not in line:
            break
        cells = split_row(line)

        if has_index and len(cells) == len(headers):
            row = dict(zip(headers[1:], cells[1:]))
            row["_index"] = cells[0]
        elif len(cells) >= len(headers):
            row = dict(zip(headers, cells[:len(headers)]))
        elif len(cells) > 0:
            padded = cells + [""] * (len(headers) - len(cells))
            row = dict(zip(headers, padded))
        else:
            continue
        rows.append(row)

    return rows


def parse_sections(text: str) -> dict[str, str]:
    """Parse markdown into hierarchical sections."""
    sections = {}
    current_h2 = None
    current_h3 = None
    current_content = []

    def save():
        if current_h2:
            key = current_h2
            if current_h3:
                key = f"{current_h2}/{current_h3}"
            content = "\n".join(current_content).strip()
            if content:
                sections[key] = content

    for line in text.split("\n"):
        if line.startswith("## "):
            save()
            current_h2 = line[3:].strip()
            current_h3 = None
            current_content = []
        elif line.startswith("### "):
            if current_h2 and current_h3 is None:
                sections[current_h2] = "\n".join(current_content).strip()
            elif current_h3:
                save()
            current_h3 = line[4:].strip()
            current_content = []
        else:
            current_content.append(line)

    save()
    return sections


def parse_bullet_kvs(text: str) -> dict[str, str]:
    """Extract key-value pairs from bullet lists (- Key: Value)."""
    kvs = {}
    for m in re.finditer(r"^[-*]\s+(.+?):\s+(.+)$", text, re.MULTILINE):
        kvs[m.group(1).strip()] = m.group(2).strip()
    return kvs


def try_float(s: str) -> float | None:
    """Try to parse a string as float (plain numbers only)."""
    if not s or s.strip().lower() in ("nan", "", "-"):
        return None
    try:
        return float(s.strip())
    except (ValueError, TypeError):
        return None


def parse_financial_number(s: str) -> float | None:
    """Parse a financial-formatted number: $, %, commas, parenthetical negatives.

    Examples:
        "$1,250,000,000" -> 1250000000.0
        "13.2320%"       -> 13.232
        "$(187,345,672)" -> -187345672.0
        "+523.20"        -> 523.2
        "85%"            -> 85.0
        "N/A"            -> None
    """
    if not s:
        return None
    s = s.strip()
    if not s or s.lower() in ("nan", "", "-", "n/a", "—", "–"):
        return None

    # Try plain float first (fast path)
    try:
        return float(s)
    except (ValueError, TypeError):
        pass

    # Detect parenthetical negatives: $(123) or (123)
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    elif s.startswith("$(") and s.endswith(")"):
        negative = True
        s = s[2:-1].strip()

    # Strip currency symbols and whitespace
    s = s.replace("$", "").replace("€", "").replace("£", "").strip()

    # Strip percentage sign
    s = s.rstrip("%").strip()

    # Strip commas (thousand separators)
    s = s.replace(",", "")

    # Strip trailing multiplier 'x' (e.g., "1.06x")
    if s.endswith("x") and len(s) > 1:
        s = s[:-1].strip()

    # Strip parenthetical annotations (e.g., "3.2 (Pass)")
    paren_match = re.match(r'^([0-9.+\-]+)\s*\(.*\)\s*$', s)
    if paren_match:
        s = paren_match.group(1)

    # Strip leading +
    if s.startswith("+"):
        s = s[1:]

    if not s:
        return None

    try:
        val = float(s)
        return -val if negative else val
    except (ValueError, TypeError):
        return None


# ============================================================================
# COLUMN TYPE DETECTION
# ============================================================================

def _looks_like_year_column(col_name: str, values: list[str]) -> bool:
    """Check if a column contains year-like values (e.g., 2019, 2020)."""
    if not values:
        return False
    year_kw = {"year", "vintage", "period", "quarter"}
    if any(kw in col_name.lower() for kw in year_kw):
        year_count = sum(1 for v in values if re.match(r'^(19|20)\d{2}$', v.strip()))
        if year_count / len(values) > 0.5:
            return True
    return False


def classify_columns(rows: list[dict]) -> dict[str, str]:
    """Classify each column as: numeric, boolean, label, or entity.

    Returns dict of column_name -> type.
    """
    if not rows:
        return {}

    col_types = {}
    for col in rows[0].keys():
        if col.startswith("_"):
            continue

        values = [r.get(col, "").strip() for r in rows]
        non_empty = [v for v in values if v and v.lower() != "nan"]

        if not non_empty:
            col_types[col] = "empty"
            continue

        # Check boolean-like
        bool_values = {"true", "false", "yes", "no", "pass", "fail", "weak", "strong"}
        if all(v.lower() in bool_values for v in non_empty):
            col_types[col] = "boolean"
            continue

        # Year columns (e.g., "Vintage: 2019, 2020") should be labels, not numeric
        if _looks_like_year_column(col, non_empty):
            col_types[col] = "label"
            continue

        # Check numeric (use financial parser to handle $, %, commas)
        numeric_count = sum(1 for v in non_empty if parse_financial_number(v) is not None)
        if numeric_count / len(non_empty) > 0.7:
            col_types[col] = "numeric"
            continue

        # Otherwise it's a label/entity
        col_types[col] = "label"

    return col_types


def detect_entity_column(rows: list[dict], col_types: dict) -> str | None:
    """Find the primary entity/label column in a table."""
    label_cols = [c for c, t in col_types.items() if t == "label"]

    # Prefer columns named "Feature", "Name", "Entity", "Model", etc.
    entity_names = {"feature", "name", "entity", "model", "segment", "cluster",
                    "scenario", "group", "subgroup", "category", "variable",
                    "factor", "portfolio", "product", "metric", "effect"}
    for col in label_cols:
        if col.lower() in entity_names:
            return col

    # First label column
    if label_cols:
        return label_cols[0]

    # Check _index column (unnamed first column with non-numeric values)
    if rows and "_index" in rows[0]:
        index_values = [r.get("_index", "").strip() for r in rows]
        non_empty = [v for v in index_values if v]
        if non_empty:
            numeric_count = sum(1 for v in non_empty if try_float(v) is not None)
            if numeric_count / len(non_empty) <= 0.5:
                return "_index"

    return None


def detect_pass_fail_column(rows: list[dict], col_types: dict) -> str | None:
    """Find a pass/fail or weak/strong boolean column."""
    for col, ctype in col_types.items():
        if ctype == "boolean":
            return col
    return None


# ============================================================================
# PHASE ENCODER INFERENCE
# ============================================================================

# Common financial metric patterns and their ranges
METRIC_PATTERNS = {
    "AUC": (r"(?i)\bAUC\b", 0.0, 1.0),
    "accuracy": (r"(?i)\b(ACC|accuracy)\b", 0.0, 1.0),
    "AIR": (r"(?i)\bAIR\b", 0.0, 2.0),
    "ratio": (r"(?i)\bratio\b", 0.0, 5.0),
    "rate": (r"(?i)\b(rate|default.rate|approval.rate)\b", 0.0, 1.0),
    "importance": (r"(?i)\b(importance|score|weight)\b", 0.0, 1.0),
    "divergence": (r"(?i)\b(divergence|distance|JS|KL)\b", 0.0, 1.0),
    "loss": (r"(?i)\b(loss|logloss|brier)\b", 0.0, 2.0),
    "pvalue": (r"(?i)\bp.?value\b", 0.0, 1.0),
    "percentage": (r"(?i)\b(pct|percent|%)\b", 0.0, 100.0),
    "capital_ratio": (r"(?i)\b(CET1|Tier.?1|capital.ratio|leverage)\b", 0.0, 30.0),
}


def infer_phase_encoders(sections: dict[str, str]) -> dict[str, tuple[float, float]]:
    """Infer phase encoders from column names across all tables."""
    encoders = {}
    all_text = " ".join(sections.values())

    for name, (pattern, vmin, vmax) in METRIC_PATTERNS.items():
        if re.search(pattern, all_text):
            encoders[name] = (vmin, vmax)

    return encoders


# ============================================================================
# GENERIC INGESTION
# ============================================================================

def ingest_table(graph: DocumentGraph, section_name: str, subsection: str | None,
                 rows: list[dict], col_types: dict):
    """Ingest a single table into the graph."""
    if not rows:
        return

    entity_col = detect_entity_column(rows, col_types)
    pass_fail_col = detect_pass_fail_column(rows, col_types)
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    label_cols = [c for c, t in col_types.items() if t == "label" and c != entity_col]

    # Build a clean section tag for ENM categories
    section_tag = re.sub(r"^\d+\.\s*", "", section_name).strip()
    section_tag = re.sub(r"[^a-zA-Z0-9_\s]", "", section_tag).strip()
    section_tag = section_tag.replace(" ", "_").lower()

    if subsection:
        sub_tag = re.sub(r"[^a-zA-Z0-9_\s]", "", subsection).strip()
        sub_tag = sub_tag.replace(" ", "_").lower()

    for i, row in enumerate(rows):
        # Determine entity name
        if entity_col and row.get(entity_col, "").strip():
            entity = row[entity_col].strip()
        elif "_index" in row and row["_index"].strip() and try_float(row["_index"]) is None:
            # Use _index only if it's a meaningful label (not a bare number)
            entity = row["_index"].strip()
        elif subsection:
            entity = subsection
        else:
            entity = f"row_{i}"

        # Build composite entity ID for non-unique entities
        secondary_labels = []
        for lc in label_cols:
            val = row.get(lc, "").strip()
            if val:
                secondary_labels.append(val)

        if secondary_labels:
            entity_full = f"{entity}/{'/'.join(secondary_labels)}"
        else:
            entity_full = entity

        # Store numeric values as ENM entries
        for col in numeric_cols:
            val = parse_financial_number(row.get(col, ""))
            if val is not None:
                # ENM category = section_tag, entity_id = entity/column
                enm_id = f"{entity_full}/{col}" if len(numeric_cols) > 1 else entity_full
                graph.store_value(
                    section_tag, enm_id, val,
                    column=col, entity=entity,
                )

                # Triple: entity has_<column> value
                rel = f"has_{col.replace(' ', '_').lower()}"
                graph.add_triple(entity, rel, str(val))

        # Store boolean/pass-fail as triples
        if pass_fail_col:
            pf_val = row.get(pass_fail_col, "").strip().lower()
            if pf_val in ("true", "yes", "weak", "fail"):
                graph.add_triple(entity_full, "fails", f"{section_tag}_{entity_full}")
                graph.add_triple(entity_full, "is_weak", section_tag)
            elif pf_val in ("false", "no", "strong", "pass"):
                graph.add_triple(entity_full, "passes", f"{section_tag}_{entity_full}")
                graph.add_triple(entity_full, "is_strong", section_tag)

        # Label columns become typed triples
        for lc in label_cols:
            val = row.get(lc, "").strip()
            if val:
                rel = f"has_{lc.replace(' ', '_').lower()}"
                graph.add_triple(entity, rel, val)

        # Entity participates in section
        graph.add_triple(entity, "in_section", section_tag)


def ingest_bullets(graph: DocumentGraph, section_name: str, kvs: dict[str, str]):
    """Ingest bullet-list key-value pairs."""
    section_tag = re.sub(r"^\d+\.\s*", "", section_name).strip()
    section_tag = section_tag.replace(" ", "_").lower()
    section_tag = re.sub(r"[^a-z0-9_]", "", section_tag)

    for key, value in kvs.items():
        # Try to store as numeric
        num = try_float(value)
        if num is not None:
            graph.store_value(section_tag, key, num)
            graph.add_triple(section_tag, f"has_{key.replace(' ', '_').lower()}", str(num))
        else:
            graph.add_triple(section_tag, f"has_{key.replace(' ', '_').lower()}", value)


def _find_table_blocks(content: str) -> list[str]:
    """Extract contiguous blocks of markdown table lines from section content."""
    table_blocks = []
    current_block = []
    for line in content.split("\n"):
        if line.strip().startswith("|"):
            current_block.append(line)
        else:
            if current_block:
                table_blocks.append("\n".join(current_block))
                current_block = []
    if current_block:
        table_blocks.append("\n".join(current_block))
    return table_blocks


# ---------------------------------------------------------------------------
# Schema-declaration parser
# ---------------------------------------------------------------------------

# Bullets like  "- High opposite_of Low"  or "- has_status is_functional True"
# Tokens are non-whitespace words; left-side allows internal whitespace inside
# a single capture (so multi-word entity names like "Credit Decisioning" work
# when wrapped in backticks).  The pattern is intentionally narrow so it does
# not collide with prose bullets that happen to contain the keywords.
_SCHEMA_BULLET = re.compile(
    r"^\s*[-*]\s*"
    r"`?(?P<lhs>[^`\n]+?)`?\s+"
    r"(?P<rel>opposite_of|is_functional)\s+"
    r"`?(?P<rhs>[^`\n]+?)`?\s*$",
    re.MULTILINE,
)


def _ingest_schema_declarations(graph: DocumentGraph, text: str) -> None:
    """Parse schema-declaration bullets from the full markdown.

    Every line matching ``- <lhs> opposite_of <rhs>`` becomes a triple
    ``(lhs, opposite_of, rhs)``; every ``- <r> is_functional <anything>``
    becomes ``(r, is_functional, anything)`` (tail is preserved so the
    declaration is auditable but ``mine_tension_pairs`` only needs the
    relation match — see ``gms.data.prepare._read_schema_declarations``).
    """
    for m in _SCHEMA_BULLET.finditer(text):
        lhs = m.group("lhs").strip()
        rel = m.group("rel").strip()
        rhs = m.group("rhs").strip()
        if not lhs or not rhs:
            continue
        graph.add_triple(lhs, rel, rhs)


def ingest_markdown(
    path: str,
    document_entity: str | None = None,
    *,
    mode: str = "regex",
    llm_callable=None,
    # --- Deprecated arguments retained for one release ---
    llm_client=None,
    llm_model: str = "claude-sonnet-4-20250514",
) -> DocumentGraph:
    """Ingest any markdown document into a DocumentGraph.

    Args:
        path: Path to the markdown file.
        document_entity: Optional name for the document root entity.
        mode: Ingestion mode — one of:

            * ``"regex"`` (library default) — pure deterministic parser. No
              LLM calls. Reproducible byte-for-byte across hosts.
            * ``"hybrid"`` — regex backbone with LLM augmentation at three
              decision points (column classification, entity-column
              detection, prose relation extraction). Strict superset of
              regex on numeric values; recovers prose triples regex
              cannot reach. Requires ``llm_callable``.
            * ``"llm_only"`` — LLM extracts everything from each table and
              from prose; regex disabled. Recovers ≈ 12.5% of regex ENM
              entries with near-zero overlap (FinStructBench paper §6.5).
              Useful for paper reproduction, not production.

        llm_callable: A ``Callable[[str], str]`` that takes a fully
            formed prompt and returns the model's text response.
            Required when ``mode != "regex"``. Wrap any existing client
            or backend in a one-line lambda — for the
            :class:`~knowlytix.knowledge.llm_backend.LLMBackend` family,
            ``lambda p: backend.call(system="", user=p)`` is the
            canonical adapter (see
            ``knowlytix.doc.ingest._llm_backend_to_callable``).

        llm_client: *DEPRECATED.* Anthropic client. If supplied without
            ``llm_callable`` we adapt it on the fly and emit a
            ``DeprecationWarning``. Will be removed in a future release.
        llm_model: *DEPRECATED.* Model id used only with ``llm_client``.

    Returns:
        Populated :class:`DocumentGraph`.

    Raises:
        ValueError: when ``mode`` is unknown, or when ``mode != "regex"``
            and no callable / client was provided. Bank-grade callers
            should always handle this raise — silent fall-back to regex
            would hide that the LLM-augmented extraction never ran.
    """
    # --- Mode normalisation: legacy "default" -> "regex" with deprecation. ---
    if mode == "default":
        warnings.warn(
            "ingest_markdown(mode='default') is deprecated; use 'regex' "
            "for the no-LLM path. The 'default' alias will be removed in "
            "a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = "regex"

    if mode not in ("regex", "hybrid", "llm_only"):
        raise ValueError(
            f"Unknown ingest mode: {mode!r}. "
            "Use 'regex', 'hybrid', or 'llm_only'."
        )

    # --- Adapt deprecated (llm_client, llm_model) to llm_callable. ---
    if llm_callable is None and llm_client is not None:
        warnings.warn(
            "ingest_markdown(llm_client=...) is deprecated; pass "
            "llm_callable instead (a Callable[[str], str]). Adapting "
            "llm_client transparently for now.",
            DeprecationWarning,
            stacklevel=2,
        )
        from knowlytix.benchmark.llm_caller import call_llm as _benchmark_call

        def _adapted(prompt: str) -> str:
            # The hybrid prompts are self-contained; pass through with
            # no extra wrapping so the LLM seam is exactly the callable
            # contract.
            return _benchmark_call(
                llm_client, "", prompt, model=llm_model
            )

        llm_callable = _adapted

    # --- Enforce: hybrid / llm_only require an LLM callable. ---
    if mode in ("hybrid", "llm_only") and llm_callable is None:
        raise ValueError(
            f"ingest_markdown(mode={mode!r}) requires an llm_callable. "
            "Pass a Callable[[str], str] — e.g. wrap an LLMBackend with "
            "`lambda p: backend.call(system='', user=p)` — or "
            "set mode='regex' to disable LLM augmentation."
        )

    with open(path) as f:
        text = f.read()

    graph = DocumentGraph()
    sections = parse_sections(text)

    # Extract document title
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if title_match:
        graph.metadata["title"] = title_match.group(1).strip()
    if document_entity:
        graph.metadata["document_entity"] = document_entity
    graph.metadata["ingest_mode"] = mode

    # Infer and register phase encoders
    encoder_ranges = infer_phase_encoders(sections)
    for name, (vmin, vmax) in encoder_ranges.items():
        graph.add_phase_encoder(name, vmin, vmax)

    # Schema declarations.  Bullets of the form
    #
    #     - <token> opposite_of <token>
    #     - <token> is_functional <token-or-True>
    #
    # appearing in any section are picked up as schema triples that the
    # GMS training loop and the tension calibrator both consume
    # (mine_tension_pairs / label_tension).  Without these the trained
    # rotors never see contradiction signal and tension calibration
    # collapses to a 2-class agree-vs-unrelated fit ("DEGENERATE").
    _ingest_schema_declarations(graph, text)

    # Lazy import for LLM modes
    if mode in ("hybrid", "llm_only"):
        from knowlytix.benchmark.llm_ingest import (
            llm_classify_columns,
            llm_detect_entity_column,
            llm_extract_relations,
            llm_extract_table,
            llm_extract_bullets_and_prose,
        )

    # Process each section
    for section_key, content in sections.items():
        parts = section_key.split("/", 1)
        section_name = parts[0]
        subsection = parts[1] if len(parts) > 1 else None

        # Ingest tables
        table_blocks = _find_table_blocks(content)

        for table_text in table_blocks:
            rows = parse_md_table(table_text)
            if not rows:
                continue

            if mode == "llm_only":
                # LLM extracts everything from the table
                extracted = llm_extract_table(
                    llm_callable, section_name, subsection, table_text)

                for entry in extracted.get("enm", []):
                    try:
                        cat = str(entry.get("category", "unknown"))
                        eid = str(entry.get("entity_id", "unknown"))
                        val = float(entry.get("value", 0))
                        graph.store_value(cat, eid, val)
                    except (ValueError, TypeError):
                        continue

                for triple in extracted.get("triples", []):
                    if isinstance(triple, list) and len(triple) == 3:
                        h, r, t = str(triple[0]), str(triple[1]), str(triple[2])
                        if h and r and t:
                            graph.add_triple(h, r, t)

            else:
                # Default or hybrid: regex-based with optional LLM fallback
                col_types = classify_columns(rows)

                if mode == "hybrid":
                    headers = [c for c in rows[0].keys() if not c.startswith("_")]
                    label_count = sum(1 for t in col_types.values() if t == "label")
                    numeric_count = sum(1 for t in col_types.values() if t == "numeric")

                    if numeric_count == 0 or label_count == len(col_types):
                        llm_types = llm_classify_columns(
                            llm_callable, headers, rows)
                        if llm_types:
                            col_types.update(llm_types)

                    entity_col = detect_entity_column(rows, col_types)
                    if entity_col is None:
                        entity_col = llm_detect_entity_column(
                            llm_callable, headers, rows, col_types)
                        if entity_col and entity_col in col_types:
                            col_types[entity_col] = "label"

                ingest_table(graph, section_name, subsection, rows, col_types)

        # Ingest non-table content
        if mode == "llm_only":
            # LLM extracts from bullets and prose
            # Filter out table lines for the prose extraction
            non_table_lines = [l for l in content.split("\n")
                               if not l.strip().startswith("|")]
            non_table_text = "\n".join(non_table_lines).strip()
            if non_table_text:
                extracted = llm_extract_bullets_and_prose(
                    llm_callable, section_name, non_table_text)

                for entry in extracted.get("enm", []):
                    try:
                        cat = str(entry.get("category", "unknown"))
                        eid = str(entry.get("entity_id", "unknown"))
                        val = float(entry.get("value", 0))
                        graph.store_value(cat, eid, val)
                    except (ValueError, TypeError):
                        continue

                for triple in extracted.get("triples", []):
                    if isinstance(triple, list) and len(triple) == 3:
                        h, r, t = str(triple[0]), str(triple[1]), str(triple[2])
                        if h and r and t:
                            graph.add_triple(h, r, t)
        else:
            # Default and hybrid: regex bullet extraction
            kvs = parse_bullet_kvs(content)
            if kvs:
                ingest_bullets(graph, section_name, kvs)

            # Hybrid: also extract relations from prose
            if mode == "hybrid":
                triples = llm_extract_relations(
                    llm_callable, section_name, content)
                for head, relation, tail in triples:
                    graph.add_triple(head, relation, tail)
                    val = try_float(tail)
                    if val is not None:
                        section_tag = re.sub(r"^\d+\.\s*", "", section_name).strip()
                        section_tag = re.sub(r"[^a-zA-Z0-9_\s]", "", section_tag).strip()
                        section_tag = section_tag.replace(" ", "_").lower()
                        enm_id = f"{head}/{relation}"
                        graph.store_value(section_tag, enm_id, val)

    return graph
