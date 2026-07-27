# SPDX-License-Identifier: Apache-2.0
"""Document-to-markdown conversion: PDF, TeX, and other formats.

Strategy:
  - TeX: regex for prose structure, LLM for tabular environments
  - PDF: OpenDataLoader (preferred) or PyMuPDF fallback, LLM for table-like pages
  - MD/TXT: passthrough
"""

import os
import re
import subprocess
import tempfile

from knowlytix.knowledge.config import ConvertConfig
from knowlytix.knowledge.llm_backend import LLMBackend


def _has_opendataloader() -> bool:
    """Check if opendataloader-pdf is available (requires Java 11+)."""
    try:
        import opendataloader_pdf  # noqa: F401
        # Also check Java is available
        result = subprocess.run(
            ["java", "-version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


# =============================================================================
# Format detection
# =============================================================================

def detect_format(path: str) -> str:
    """Detect document format from extension."""
    ext = os.path.splitext(path)[1].lower()
    return {
        ".pdf": "pdf",
        ".tex": "tex",
        ".latex": "tex",
        ".md": "md",
        ".markdown": "md",
        ".txt": "txt",
        ".docx": "docx",
    }.get(ext, "unknown")


# =============================================================================
# PDF conversion — OpenDataLoader backend (preferred)
# =============================================================================

def convert_pdf_opendataloader(path: str, config: ConvertConfig) -> str:
    """Convert PDF to markdown using OpenDataLoader.

    Uses XY-Cut++ reading order, layout-based table detection, and
    heading hierarchy extraction.  No LLM calls needed.

    Requires: ``pip install opendataloader-pdf`` and Java 11+.
    """
    import opendataloader_pdf

    pages_arg = None
    if config.max_pages and config.max_pages < 200:
        pages_arg = f"1-{config.max_pages}"

    with tempfile.TemporaryDirectory() as tmpdir:
        opendataloader_pdf.convert(
            input_path=path,
            output_dir=tmpdir,
            format="markdown",
            quiet=True,
            image_output="off",
            reading_order="xycut",
            pages=pages_arg,
        )
        # Read the generated markdown file
        md_files = [f for f in os.listdir(tmpdir) if f.endswith(".md")]
        if not md_files:
            raise RuntimeError(
                f"OpenDataLoader produced no markdown output for {path}")
        md_path = os.path.join(tmpdir, md_files[0])
        with open(md_path) as f:
            return f.read()


# =============================================================================
# PDF conversion — PyMuPDF backend (fallback)
# =============================================================================

def _extract_pdf_text(path: str, max_pages: int = 200) -> list[str]:
    """Extract text per page using PyMuPDF."""
    import pymupdf
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pages.append(page.get_text())
    doc.close()
    return pages


def _looks_like_table(text: str) -> bool:
    """Heuristic: does this text block look like a table?"""
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return False
    # Count lines with multiple numbers separated by whitespace
    numeric_lines = 0
    for line in lines:
        nums = re.findall(r"[\d.]+", line)
        if len(nums) >= 3:
            numeric_lines += 1
    return numeric_lines >= 2


def _extract_tables_with_llm(page_text: str, llm: LLMBackend,
                              chunk_size: int = 4000) -> str:
    """Use LLM to convert table-like text to markdown tables."""
    system = (
        "You are a document parser. Convert the following text into clean "
        "markdown. Preserve all numbers exactly. Convert any tabular data "
        "into proper markdown tables with | delimiters. Keep section headers "
        "as ## or ### headers. Output ONLY the markdown, no commentary."
    )
    # Process in chunks to stay within token limits
    if len(page_text) <= chunk_size:
        return llm.call(system, page_text, max_tokens=4096)

    chunks = []
    for i in range(0, len(page_text), chunk_size):
        chunk = page_text[i:i + chunk_size]
        chunks.append(llm.call(system, chunk, max_tokens=4096))
    return "\n\n".join(chunks)


def convert_pdf_to_markdown(path: str, config: ConvertConfig,
                             llm: LLMBackend | None = None) -> str:
    """Convert PDF to markdown.

    Backend selection (config.pdf_backend):
      - "auto": Use OpenDataLoader if available, else PyMuPDF + LLM.
      - "opendataloader": Use OpenDataLoader (error if unavailable).
      - "pymupdf": Use PyMuPDF + LLM (original behavior).
    """
    backend = config.pdf_backend

    if backend == "opendataloader" or (backend == "auto" and _has_opendataloader()):
        try:
            print("  PDF backend: OpenDataLoader")
            return convert_pdf_opendataloader(path, config)
        except Exception as e:
            if backend == "opendataloader":
                raise
            print(f"  OpenDataLoader failed ({e}), falling back to PyMuPDF")

    print("  PDF backend: PyMuPDF" + (" + LLM" if llm else ""))
    pages = _extract_pdf_text(path, config.max_pages)
    result_parts = []

    for i, page_text in enumerate(pages):
        if not page_text.strip():
            continue

        if llm and _looks_like_table(page_text):
            md = _extract_tables_with_llm(page_text, llm, config.chunk_size)
            result_parts.append(md)
        else:
            # Basic text cleanup: fix broken lines, add structure
            cleaned = _clean_pdf_text(page_text)
            result_parts.append(cleaned)

    return "\n\n".join(result_parts)


def _clean_pdf_text(text: str) -> str:
    """Basic cleanup of raw PDF extracted text."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        # Detect section headers (all caps or numbered)
        if re.match(r"^\d+\.?\s+[A-Z]", stripped) and len(stripped) < 100:
            cleaned.append(f"\n## {stripped}")
        elif re.match(r"^\d+\.\d+\.?\s+", stripped) and len(stripped) < 100:
            cleaned.append(f"\n### {stripped}")
        else:
            cleaned.append(stripped)
    return "\n".join(cleaned)


# =============================================================================
# TeX conversion
# =============================================================================

def convert_tex_to_markdown(path: str, config: ConvertConfig,
                             llm: LLMBackend | None = None) -> str:
    """Convert TeX/LaTeX to markdown.

    Stage 1: Extract tabular environments and convert via LLM (preserves numbers).
    Stage 2: Convert remaining prose with regex.
    Stage 3: Reassemble with proper markdown tables.
    """
    with open(path) as f:
        tex = f.read()

    # Extract body
    body_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}",
                           tex, re.DOTALL)
    body = body_match.group(1) if body_match else tex

    if llm:
        # Extract all tabular environments and convert via LLM
        md = _tex_to_md_with_llm_tables(body, llm)
    else:
        md = _basic_tex_to_md(body)

    return md


def _tex_to_md_with_llm_tables(body: str, llm: LLMBackend) -> str:
    """Convert TeX body to markdown: LLM for tables, regex for prose.

    1. Find all tabular/table environments in the TeX source
    2. Send each to LLM for precise markdown table conversion
    3. Replace in-place, then convert remaining prose with regex
    """
    # Find all tabular environments (possibly nested in table/center)
    # Match \begin{tabular}...\end{tabular} including surrounding table env
    table_pattern = re.compile(
        r"(\\begin\{(?:table|center)\}.*?)?"
        r"(\\begin\{tabular[x]?\}.*?\\end\{tabular[x]?\})"
        r"(.*?\\end\{(?:table|center)\})?",
        re.DOTALL,
    )

    # Also match standalone tabular
    tabular_pattern = re.compile(
        r"\\begin\{tabular[x]?\}.*?\\end\{tabular[x]?\}",
        re.DOTALL,
    )

    matches = list(tabular_pattern.finditer(body))
    print(f"  Found {len(matches)} tabular environments — converting via LLM...")

    # Convert each table via LLM
    system = (
        "Convert this LaTeX table into a clean markdown table. Rules:\n"
        "1. Use | delimiters and --- separator row\n"
        "2. Preserve ALL numbers exactly as they appear\n"
        "3. Include the table caption as a bold line above if present\n"
        "4. Output ONLY the markdown table, no commentary or code fences\n"
        "5. If there are multiple columns, keep them all\n"
        "6. Clean up LaTeX commands (\\textbf -> bold, etc)"
    )

    # Process in reverse to preserve positions
    replacements = []
    for match in matches:
        table_tex = match.group()
        # Get surrounding context for caption
        start = max(0, match.start() - 200)
        context_before = body[start:match.start()]
        caption_match = re.search(r"\\caption\{([^}]+)\}", context_before)
        if not caption_match:
            # Check inside the match or after
            caption_match = re.search(r"\\caption\{([^}]+)\}",
                                       body[match.start():match.end() + 200])

        user_msg = table_tex
        if caption_match:
            user_msg = f"Caption: {caption_match.group(1)}\n\n{table_tex}"

        try:
            md_table = llm.call(system, user_msg, max_tokens=2048)
            # Strip code fences if LLM wraps in them
            md_table = re.sub(r"^```\w*\n", "", md_table)
            md_table = re.sub(r"\n```\s*$", "", md_table)
            replacements.append((match.start(), match.end(), md_table))
        except Exception as e:
            print(f"    LLM table conversion error: {e}")

    # Apply replacements in reverse order
    result = body
    for start, end, md_table in reversed(replacements):
        result = result[:start] + "\n\n" + md_table + "\n\n" + result[end:]

    # Now convert the remaining prose with regex
    result = _basic_tex_to_md(result)
    return result


def _basic_tex_to_md(tex: str) -> str:
    """Minimal TeX to markdown conversion.

    Preserves lines that are already markdown tables (start with |).
    """
    # Strip preamble if present
    body_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}",
                           tex, re.DOTALL)
    text = body_match.group(1) if body_match else tex

    # Separate markdown table lines from TeX content
    # so the cleanup regex doesn't mangle them
    lines = text.split("\n")
    protected = {}  # line_idx -> original line
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and "|" in line[line.index("|")+1:]:
            protected[i] = line
            lines[i] = f"__MD_TABLE_LINE_{i}__"
    text = "\n".join(lines)

    # Section headers (handle nested braces for chapter/section)
    text = re.sub(r"\\chapter\{([^}]+)\}", r"\n# \1\n", text)
    text = re.sub(r"\\section\{([^}]+)\}", r"\n## \1\n", text)
    text = re.sub(r"\\subsection\{([^}]+)\}", r"\n### \1\n", text)
    text = re.sub(r"\\subsubsection\{([^}]+)\}", r"\n#### \1\n", text)
    text = re.sub(r"\\paragraph\{([^}]+)\}", r"\n**\1**\n", text)

    # Basic formatting
    text = re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{([^}]+)\}", r"*\1*", text)
    text = re.sub(r"\\emph\{([^}]+)\}", r"*\1*", text)
    text = re.sub(r"\\texttt\{([^}]+)\}", r"`\1`", text)

    # Remove common commands
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\ref\{[^}]*\}", "[ref]", text)
    text = re.sub(r"\\cite\{[^}]*\}", "[cite]", text)
    text = re.sub(r"\\maketitle", "", text)
    text = re.sub(r"\\begin\{abstract\}", "\n## Abstract\n", text)
    text = re.sub(r"\\end\{abstract\}", "", text)

    # Itemize/enumerate
    text = re.sub(r"\\begin\{(itemize|enumerate)\}", "", text)
    text = re.sub(r"\\end\{(itemize|enumerate)\}", "", text)
    text = re.sub(r"\\item\s*", "- ", text)

    # Math: keep inline as $...$, display as $$...$$
    text = re.sub(r"\\\[", "\n$$\n", text)
    text = re.sub(r"\\\]", "\n$$\n", text)
    text = re.sub(r"\\begin\{equation\*?\}", "\n$$\n", text)
    text = re.sub(r"\\end\{equation\*?\}", "\n$$\n", text)
    text = re.sub(r"\\begin\{align\*?\}", "\n$$\n", text)
    text = re.sub(r"\\end\{align\*?\}", "\n$$\n", text)

    # Remove remaining environments
    text = re.sub(r"\\begin\{[^}]*\}", "", text)
    text = re.sub(r"\\end\{[^}]*\}", "", text)

    # Clean stray commands
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"[{}]", "", text)

    # Restore protected markdown table lines
    for i, original in protected.items():
        text = text.replace(f"__MD_TABLE_LINE_{i}__", original)

    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fix_tex_tables_with_llm(md: str, llm: LLMBackend) -> str:
    """Use LLM to convert remaining \\begin{tabular} blocks to markdown tables."""
    pattern = r"\\begin\{tabular\}.*?\\end\{tabular\}"
    matches = list(re.finditer(pattern, md, re.DOTALL))
    if not matches:
        return md

    system = (
        "Convert this LaTeX tabular environment into a clean markdown table "
        "with | delimiters. Preserve all numbers exactly. Output ONLY the "
        "markdown table, no commentary."
    )
    for match in reversed(matches):  # reverse to preserve positions
        table_tex = match.group()
        table_md = llm.call(system, table_tex, max_tokens=2048)
        md = md[:match.start()] + table_md + md[match.end():]
    return md


# =============================================================================
# Top-level dispatcher
# =============================================================================

def convert_document(path: str, config: ConvertConfig,
                      llm: LLMBackend | None = None) -> str:
    """Convert any supported document to markdown.

    Args:
        path: Path to document (PDF, TeX, MD, TXT).
        config: Conversion configuration.
        llm: Optional LLM backend for table extraction.

    Returns:
        Markdown string.
    """
    fmt = detect_format(path)

    if fmt == "md":
        with open(path) as f:
            return f.read()

    if fmt == "txt":
        with open(path) as f:
            return f.read()

    if fmt == "pdf":
        return convert_pdf_to_markdown(path, config, llm)

    if fmt == "tex":
        return convert_tex_to_markdown(path, config, llm)

    # Unknown format: try reading as text
    try:
        with open(path) as f:
            return f.read()
    except UnicodeDecodeError:
        raise ValueError(f"Unsupported document format: {fmt} ({path})")
