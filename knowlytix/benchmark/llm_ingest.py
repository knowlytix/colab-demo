# SPDX-License-Identifier: Apache-2.0
"""LLM-assisted ingestion for hybrid and llm_only modes.

Provides LLM-based alternatives for the structural decisions and prose
extraction that the deterministic regex parser cannot reach:

  1. Column classification — when the row-sampling heuristic is ambiguous.
  2. Entity-column detection — when the name heuristic finds no match.
  3. Relation extraction from prose — non-tabular text the regex parser
     ignores.
  4. Whole-table extraction — used in ``llm_only`` mode to produce both
     ENM entries and triples directly from the markdown table.
  5. Bullet/prose extraction — used in ``llm_only`` mode for sections
     that have no tables.

The hybrid mode preserves the core invariant that the regex backbone
guarantees: numeric values are stored with SHA-256 integrity hashes and
verified on read. The LLM only contributes *structural* decisions and
prose-derived facts; values themselves never bypass the deterministic
extraction path.

LLM seam
--------

All five functions take an :data:`LLMCallable` (defined locally below)
— any ``Callable[[str], str]`` that takes a fully-formed prompt and
returns the model's text response. The caller is responsible for the
provider, model selection, retries, and rate limiting; this module only
formats prompts and parses responses.
"""

from __future__ import annotations

import json
import re
from typing import Callable

LLMCallable = Callable[[str], str]


# ============================================================================
# LLM-ASSISTED COLUMN CLASSIFICATION
# ============================================================================

_CLASSIFY_PROMPT = """\
You are analyzing a table from a financial document.

Table headers: {headers}

Sample rows (up to 5):
{sample_rows}

Classify each column into exactly one of these types:
- "entity": the primary identifier column (e.g., names of models, segments, portfolios)
- "numeric": columns containing numeric values (may include $, %, commas)
- "boolean": columns where every value is one of: pass, fail, yes, no, true, false, weak, strong
- "label": non-numeric, non-boolean string columns (e.g., categories, descriptions)

Return ONLY a JSON object mapping each column name to its type.
Example: {{"Feature": "entity", "AUC": "numeric", "Status": "boolean", "Group": "label"}}

JSON:"""


def llm_classify_columns(
    llm_callable: LLMCallable,
    headers: list[str],
    rows: list[dict],
) -> dict[str, str]:
    """Use the LLM to classify columns when heuristics are ambiguous.

    Args:
        llm_callable: prompt → response.
        headers: List of column names.
        rows: List of row dicts (a sample of up to 5 rows is shown).

    Returns:
        Dict mapping column name → type
        (one of ``"entity" | "numeric" | "boolean" | "label"``).
        Empty dict on parse failure (caller falls back to regex types).
    """
    sample = rows[:5]
    sample_text = "\n".join(
        "  " + " | ".join(r.get(h, "") for h in headers)
        for r in sample
    )
    prompt = _CLASSIFY_PROMPT.format(
        headers=headers,
        sample_rows=sample_text,
    )

    raw = llm_callable(prompt)

    # Extract JSON from response.
    match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            valid_types = {"entity", "numeric", "boolean", "label"}
            return {
                k: v for k, v in result.items()
                if k in headers and v in valid_types
            }
        except json.JSONDecodeError:
            pass

    return {}


# ============================================================================
# LLM-ASSISTED ENTITY-COLUMN DETECTION
# ============================================================================

_ENTITY_PROMPT = """\
You are analyzing a table from a financial document.

Table headers: {headers}
Column types: {col_types}

Sample rows (up to 5):
{sample_rows}

Which column contains the primary entity identifier — the column whose values
name the things being described in each row (e.g., model names, segment names,
portfolio names, metric names)?

Return ONLY the column name as a single string, with no quotes or explanation.

Column name:"""


def llm_detect_entity_column(
    llm_callable: LLMCallable,
    headers: list[str],
    rows: list[dict],
    col_types: dict[str, str],
) -> str | None:
    """Use the LLM to identify the entity column when the heuristic fails.

    Returns the chosen column name, or ``None`` if the response cannot be
    matched against any header (case-insensitive).
    """
    sample = rows[:5]
    sample_text = "\n".join(
        "  " + " | ".join(r.get(h, "") for h in headers)
        for r in sample
    )
    prompt = _ENTITY_PROMPT.format(
        headers=headers,
        col_types=col_types,
        sample_rows=sample_text,
    )

    raw = llm_callable(prompt)
    col_name = raw.strip().strip('"').strip("'")

    if col_name in headers:
        return col_name
    for h in headers:
        if h.lower() == col_name.lower():
            return h
    return None


# ============================================================================
# LLM-ASSISTED RELATION EXTRACTION FROM PROSE
# ============================================================================

_RELATION_PROMPT = """\
You are extracting structured facts from a financial document section.

Section name: {section_name}

Text:
{text}

Extract all factual relationships as (entity, relation, value) triples.
Focus on:
- Numeric facts: (entity_name, metric_name, numeric_value)
- Categorical facts: (entity_name, property, category_value)
- Status facts: (entity_name, status, pass/fail/approved/rejected)

Return ONLY a JSON array of triples. Each triple is [entity, relation, value].
Example: [["CET1", "capital_ratio", "12.5"], ["Model_A", "status", "approved"]]

If no structured facts can be extracted, return an empty array: []

JSON:"""


def llm_extract_relations(
    llm_callable: LLMCallable,
    section_name: str,
    text: str,
) -> list[tuple[str, str, str]]:
    """Use the LLM to extract triples from prose content.

    Skips text that is mostly tabular (those cells are handled by the
    regex parser) and very short text (≤ 50 chars).
    """
    lines = text.strip().split("\n")
    table_lines = sum(1 for l in lines if l.strip().startswith("|"))
    if table_lines / max(len(lines), 1) > 0.5:
        return []

    if len(text.strip()) < 50:
        return []

    prompt = _RELATION_PROMPT.format(
        section_name=section_name,
        text=text[:3000],
    )

    raw = llm_callable(prompt)

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            triples = []
            for item in result:
                if isinstance(item, list) and len(item) == 3:
                    h, r, t = str(item[0]), str(item[1]), str(item[2])
                    if h and r and t:
                        triples.append((h, r, t))
            return triples
        except json.JSONDecodeError:
            pass

    return []


# ============================================================================
# LLM-ONLY TABLE EXTRACTION
# ============================================================================

_TABLE_EXTRACT_PROMPT = """\
You are extracting structured data from a financial document table.

Section: {section_name}
{subsection_line}

Table (markdown):
{table_text}

Extract ALL data from this table as structured JSON with two arrays:

1. "enm": Array of exact numeric entries. Each entry is:
   {{"category": "<section_tag>", "entity_id": "<entity/column>", "value": <number>}}

   Rules for entity_id:
   - Identify the primary entity column (names/labels identifying each row)
   - For each numeric cell: entity_id = "entity_name/column_name"
   - If the entity has secondary label columns, use "entity/label1/label2/column"
   - If only one numeric column, entity_id = just the entity name

2. "triples": Array of [head, relation, tail] knowledge graph triples:
   - For each numeric cell: [entity, "has_<column_name_lowercase>", "value"]
   - For each label cell: [entity, "has_<column_name_lowercase>", "label_value"]
   - For boolean/status cells (pass/fail/yes/no/weak/strong):
     If pass/yes/strong: [entity, "passes", "section_entity"] and [entity, "is_strong", "section"]
     If fail/no/weak: [entity, "fails", "section_entity"] and [entity, "is_weak", "section"]
   - For every entity: [entity, "in_section", "section_tag"]

Use lowercase_with_underscores for section tags and relation names.
Strip markdown bold (**) from values.
Parse financial numbers: remove $, commas, %; handle parenthetical negatives $(X) = -X.

Return ONLY valid JSON. No explanation.

JSON:"""


def llm_extract_table(
    llm_callable: LLMCallable,
    section_name: str,
    subsection: str | None,
    table_text: str,
) -> dict:
    """Use the LLM to extract all structured data from a table (llm_only mode).

    Returns a dict with two lists:

      * ``"enm"``: ``[{"category", "entity_id", "value"}]`` — to be
        fed into :meth:`DocumentGraph.store_value`.
      * ``"triples"``: ``[[head, relation, tail]]`` — to be fed into
        :meth:`DocumentGraph.add_triple`.
    """
    sub_line = f"Subsection: {subsection}" if subsection else ""
    prompt = _TABLE_EXTRACT_PROMPT.format(
        section_name=section_name,
        subsection_line=sub_line,
        table_text=table_text[:4000],
    )

    raw = llm_callable(prompt)

    brace_start = raw.find("{")
    if brace_start == -1:
        return {"enm": [], "triples": []}

    depth = 0
    brace_end = brace_start
    for i in range(brace_start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                brace_end = i + 1
                break

    try:
        result = json.loads(raw[brace_start:brace_end])
        return {
            "enm": result.get("enm", []),
            "triples": result.get("triples", []),
        }
    except json.JSONDecodeError:
        return {"enm": [], "triples": []}


_BULLET_EXTRACT_PROMPT = """\
You are extracting structured data from a financial document section.

Section: {section_name}

Text:
{text}

Extract all key-value pairs and facts as structured JSON with two arrays:

1. "enm": Array of exact numeric entries found in bullet lists or prose.
   Each entry: {{"category": "<section_tag>", "entity_id": "<key>", "value": <number>}}

2. "triples": Array of [head, relation, tail] knowledge graph triples.
   - For numeric facts: [section_tag, "has_<key_lowercase>", "value"]
   - For categorical facts: [entity, "has_<property>", "category_value"]
   - For relationships: [entity, relation, target]

Use lowercase_with_underscores for tags and relations.

Return ONLY valid JSON. No explanation.

JSON:"""


def llm_extract_bullets_and_prose(
    llm_callable: LLMCallable,
    section_name: str,
    text: str,
) -> dict:
    """Use the LLM to extract structured data from non-table content (llm_only).

    Same return shape as :func:`llm_extract_table`.
    """
    if len(text.strip()) < 30:
        return {"enm": [], "triples": []}

    prompt = _BULLET_EXTRACT_PROMPT.format(
        section_name=section_name,
        text=text[:3000],
    )

    raw = llm_callable(prompt)

    brace_start = raw.find("{")
    if brace_start == -1:
        return {"enm": [], "triples": []}

    depth = 0
    brace_end = brace_start
    for i in range(brace_start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                brace_end = i + 1
                break

    try:
        result = json.loads(raw[brace_start:brace_end])
        return {
            "enm": result.get("enm", []),
            "triples": result.get("triples", []),
        }
    except json.JSONDecodeError:
        return {"enm": [], "triples": []}
