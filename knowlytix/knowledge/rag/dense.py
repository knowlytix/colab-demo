# SPDX-License-Identifier: Apache-2.0
"""Dense fallback — the distrusted, opt-in last resort.

Triple-mediated retrieval is the only authoritative path (``GEODE_RAG_DESIGN.md``
§4): it is verifiable and provenance-grounded. Dense vector search over raw text
spans can reach prose the graph never triplified, but its hits are *not*
GMS-verified — so this path is **off by default**, and when it does answer the
result is flagged ``verified=False`` with a notice. Use it only when coverage
matters more than guaranteed verifiability.

The vector backend is pluggable (`VectorBackend`); the default is in-memory with
an injectable encoder (so it is offline-testable and needs no external DB).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = ["DenseSpan", "VectorBackend", "InMemoryVectorBackend", "build_spans"]


@dataclass
class DenseSpan:
    text: str
    location: str            # source_path:line_start-line_end
    line_start: int = -1
    line_end: int = -1


def build_spans(markdown: str, *, source_path: str = "") -> list[DenseSpan]:
    """Chunk markdown into paragraph spans with line-range provenance.

    Blank-line-separated blocks become spans; header-only blocks are skipped.
    """
    spans: list[DenseSpan] = []
    lines = markdown.split("\n")
    start = None
    buf: list[str] = []

    def flush(end_line: int) -> None:
        if not buf:
            return
        text = "\n".join(buf).strip()
        content = [ln for ln in buf if ln.strip() and not ln.lstrip().startswith("#")]
        if text and content:
            spans.append(DenseSpan(text=text,
                                   location=f"{source_path}:{start}-{end_line}",
                                   line_start=start, line_end=end_line))

    for i, ln in enumerate(lines, start=1):
        if ln.strip() == "":
            flush(i - 1)
            buf, start = [], None
        else:
            if start is None:
                start = i
            buf.append(ln)
    flush(len(lines))
    return spans


class VectorBackend(ABC):
    """Pluggable dense index. Implement to back the fallback with an external DB."""

    @abstractmethod
    def index(self, spans: list[DenseSpan]) -> None:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[tuple[DenseSpan, float]]:
        ...


class InMemoryVectorBackend(VectorBackend):
    """Cosine search over in-memory span embeddings (injectable encoder)."""

    def __init__(self, encoder=None):
        self._encoder = encoder
        self._spans: list[DenseSpan] = []
        self._emb = None

    def _encode(self, texts: list[str]):
        import torch

        if self._encoder is None:
            from knowlytix.core.graph.encoders import encode_texts
            self._encoder = encode_texts
        emb = torch.as_tensor(self._encoder(texts), dtype=torch.float32)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def index(self, spans: list[DenseSpan]) -> None:
        self._spans = list(spans)
        self._emb = self._encode([s.text for s in spans]) if spans else None

    def search(self, query: str, top_k: int) -> list[tuple[DenseSpan, float]]:
        if not self._spans or self._emb is None:
            return []
        q = self._encode([query])[0]
        sims = self._emb @ q
        k = min(top_k, len(self._spans))
        order = sims.argsort(descending=True)[:k]
        return [(self._spans[int(i)], float(sims[int(i)])) for i in order]
