"""``VectorNodeKnowledgeAdapter`` — read-only adapter for node-level
vector similarity search.

Per v0.9.0 plan (``docs/plans/plan_knowlytix_kal_v0_9.md``). Targets
paper §6.6 row 1 — "Bag-of-strings stores participating as
``search_similar_nodes`` providers" — flipping it from 🚧 to ✅. Pairs
with v0.7.0's ``JsonlKnowledgeAdapter`` (which closed §6.6 row 2,
read-only triple stores).

Shape: a paired ``.jsonl`` (one ``KALNode`` per line) and ``.npy``
(float32 array, shape ``(N, D)``, row-aligned) plus a required
``.encoder`` sidecar naming the model that produced the embeddings.
All three loaded eagerly at construction. Reads are O(N·D) brute-force
cosine over the in-memory normalized array — the lightweight option,
not the indexed one. Documented capacity envelope: ~50k vectors at
D=384 (~73 MB) on M-series hardware completes a search in ~2 ms; for
production-scale workloads, use Postgres + pgvector.

Capability profile:

- ``can_read=True``
- ``can_write=False`` / ``can_delete=False``
- ``supports_vector_search=True``    (the headline)
- ``supports_text_search=False``     (no inverted index)
- ``supports_provenance=True``       (nodes carry their own properties)
- ``supports_verification_metadata=False``  (no triple-level verification)
- ``supports_atomic_writes=False``

Direct-call error contract (per ``AdapterError`` docstring in
``knowly.kal.types``):

- ``query_triples`` / ``query_adjacent_triples`` →
  ``CapabilityNotSupportedError``. **Load-bearing**: today's federation
  router only pre-filters ``search_similar_nodes`` by
  ``supports_vector_search``; triple-shaped queries fan out to every
  registered adapter regardless of capability flag, so this raise is
  what surfaces the rejection as a per-adapter row in
  ``KALQueryResult.errors`` instead of silently corrupting the merged
  result.
- ``insert_triples`` / ``update_verification`` / ``delete_triples`` →
  ``CapabilityNotSupportedError`` ("read-only").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from knowlytix.kal.errors import CapabilityNotSupportedError
from knowlytix.kal.protocol import AdapterCapabilities
from knowlytix.kal.types import (
    KALNode,
    KALQuery,
    KALQueryResult,
    KALTriple,
    VerificationMetadata,
)

_CAPABILITIES = AdapterCapabilities(
    can_read=True,
    can_write=False,
    can_delete=False,
    supports_vector_search=True,
    supports_text_search=False,
    supports_provenance=True,
    supports_verification_metadata=False,
    supports_atomic_writes=False,
)


class VectorNodeKnowledgeAdapter:
    """Read-only ``KnowledgeAdapter`` for node-level cosine similarity."""

    def __init__(
        self,
        name: str,
        *,
        nodes_path: Path | str,
        embeddings_path: Path | str,
        encoder_path: Path | str,
    ) -> None:
        nodes_p = Path(nodes_path)
        embeddings_p = Path(embeddings_path)
        encoder_p = Path(encoder_path)
        self._name = name
        self._nodes: list[KALNode] = _load_nodes(nodes_p)
        embeddings = _load_embeddings(embeddings_p)
        self._dimension: int = _validate_alignment(
            self._nodes,
            embeddings,
            nodes_path=nodes_p,
            embeddings_path=embeddings_p,
        )
        self._normalized: NDArray[np.float32] = _normalize_rows(embeddings)
        self._encoder_id: str = _load_encoder_id(encoder_p)

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> AdapterCapabilities:
        return _CAPABILITIES

    @property
    def encoder_id(self) -> str:
        """Identifier of the model that produced the stored embeddings.

        Surfaced so consumers building heterogeneous federations can
        detect incommensurable embedding spaces — two adapters with
        the same metric (cosine) but different encoder models share
        the distance formula but not the axis basis, so merged
        results would silently scramble.
        """
        return self._encoder_id

    # --- Read operations ---

    async def query_triples(
        self,
        query: KALQuery,
        tenant_id: str | None = None,
    ) -> KALQueryResult:
        raise CapabilityNotSupportedError(
            "VectorNodeKnowledgeAdapter is a node-store, not a "
            "triple-store; use search_similar_nodes or get_node"
        )

    async def get_node(
        self,
        node_id: str,
        tenant_id: str | None = None,
    ) -> KALNode | None:
        for node in self._nodes:
            if node.id == node_id:
                return node
        return None

    async def query_adjacent_triples(
        self,
        node_id: str,
        predicate: str | None = None,
        direction: str = "outgoing",
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[KALTriple]:
        raise CapabilityNotSupportedError(
            "VectorNodeKnowledgeAdapter is a node-store, not a "
            "triple-store; adjacent-triple walks are not supported"
        )

    async def search_similar_nodes(
        self,
        query_vector: list[float],
        tenant_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[KALNode, float]]:
        if not query_vector:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (self._dimension,):
            raise ValueError(
                f"query vector dimension {query.shape[0]} does not "
                f"match stored embedding dimension {self._dimension}"
            )
        _check_finite(query, "query vector")
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return []
        query_normalized = query / query_norm
        # ``self._normalized`` is row-normalized; the matmul is then a
        # vector of cosine similarities. Theoretically in ``[-1, 1]``,
        # but float32 accumulation in a D-wide dot product can produce
        # values like ``1.0 + ulp`` for near-identical vectors —
        # subtracting from 1.0 then gives a tiny negative distance,
        # violating the documented ``[0, 2]`` range. Clip restores the
        # contract; the cost is one extra pass over an N-vector.
        similarities = self._normalized @ query_normalized
        distances = np.clip(1.0 - similarities, 0.0, 2.0)
        k = min(limit, distances.shape[0])
        if k <= 0:
            return []
        # ``argpartition`` runs in O(N); we then ``argsort`` only the
        # top-k subset. Cheaper than ``argsort(distances)[:k]`` for
        # ``k << N``.
        top_k_unordered = np.argpartition(distances, kth=k - 1)[:k]
        ranked = top_k_unordered[np.argsort(distances[top_k_unordered])]
        return [
            (self._nodes[int(i)], float(distances[int(i)])) for i in ranked
        ]

    # --- Write operations (rejected — read-only) ---

    async def insert_triples(
        self,
        triples: list[KALTriple],
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError(
            "VectorNodeKnowledgeAdapter is read-only"
        )

    async def update_verification(
        self,
        triple_id: str,
        verification: VerificationMetadata,
        tenant_id: str | None = None,
    ) -> bool:
        raise CapabilityNotSupportedError(
            "VectorNodeKnowledgeAdapter is read-only"
        )

    async def delete_triples(
        self,
        triple_ids: list[str],
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError(
            "VectorNodeKnowledgeAdapter is read-only"
        )

    async def quarantine_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError(
            "VectorNodeKnowledgeAdapter is read-only"
        )

    async def restore_triples_by_source(
        self,
        source: str,
        tenant_id: str | None = None,
    ) -> int:
        raise CapabilityNotSupportedError(
            "VectorNodeKnowledgeAdapter is read-only"
        )


def _load_nodes(path: Path) -> list[KALNode]:
    """Parse a ``.jsonl`` file into a list of ``KALNode``.

    Mirrors ``JsonlKnowledgeAdapter._load_jsonl``: eager parse so
    format errors surface at startup with the offending line number;
    blank lines tolerated. Additionally rejects duplicate node IDs —
    without this, ``get_node(id)`` would return the first match while
    ``search_similar_nodes`` could return both copies with different
    distances, producing a silent inconsistency.
    """
    nodes: list[KALNode] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                node = KALNode.model_validate_json(line)
            except ValidationError as exc:
                raise ValueError(
                    f"{path}: malformed KALNode at line {lineno}: {exc}"
                ) from exc
            if node.id in seen_ids:
                raise ValueError(
                    f"{path}: duplicate node id {node.id!r} at line "
                    f"{lineno}. Each node must appear exactly once so "
                    "search results and get_node lookups stay consistent."
                )
            seen_ids.add(node.id)
            nodes.append(node)
    return nodes


def _validate_alignment(
    nodes: list[KALNode],
    embeddings: NDArray[np.float32],
    *,
    nodes_path: Path,
    embeddings_path: Path,
) -> int:
    """Cross-check the loaded files; return the embedding dimension.

    Two rejections: an empty store (zero rows) is a configuration
    mistake, not a valid edge case; row counts must align 1:1 so a
    triple's index can address both the node and the embedding.
    """
    if embeddings.shape[0] == 0:
        raise ValueError(
            f"{embeddings_path}: embeddings file is empty (zero rows). "
            "A node-store with zero nodes is a configuration mistake, "
            "not a valid edge case."
        )
    if len(nodes) != embeddings.shape[0]:
        raise ValueError(
            f"node/embedding row count mismatch: "
            f"{len(nodes)} nodes in {nodes_path} vs "
            f"{embeddings.shape[0]} embeddings in {embeddings_path}. "
            "Rows must be aligned 1:1."
        )
    return int(embeddings.shape[1])


def _load_embeddings(path: Path) -> NDArray[np.float32]:
    """Load a ``.npy`` file as a 2D float32 array.

    ``allow_pickle=False`` blocks the numpy ``object``-array path that
    would otherwise call ``pickle.loads`` on the file contents — a
    well-known arbitrary-code-execution surface for adapters loading
    untrusted ``.npy``. Non-float dtypes (e.g. ``int8`` from a
    quantized embedding pipeline) are rejected explicitly per the plan
    — silent coercion of integer-quantized values to float32 would
    produce mathematically valid but semantically meaningless
    distances.
    """
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(
            f"{path}: embeddings must be 2D (N, D); got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(
            f"{path}: embeddings dtype must be floating-point; got "
            f"{array.dtype}. Quantized formats (int8 / int16) are not "
            f"supported in v0.9.0 — see plan §Out-of-scope."
        )
    coerced: NDArray[np.float32] = array.astype(np.float32, copy=False)
    _check_finite(coerced, str(path))
    return coerced


def _normalize_rows(
    embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Row-wise L2-normalize. Zero rows stay zero (no NaN from divide-by-zero)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return (embeddings / safe_norms).astype(np.float32, copy=False)


def _check_finite(array: NDArray[np.float32], description: str) -> None:
    """Raise if ``array`` contains any ``nan`` / ``inf`` values.

    Used at both load time (``_load_embeddings``) and query entry
    (``search_similar_nodes``). Without this, a single non-finite
    value poisons every downstream computation: ``norm`` returns
    nan, the zero-norm guard misses it (``nan == 0`` is False),
    division produces all-nan, ``argpartition`` on nan returns
    arbitrary indices, and the caller sees nan-distance rows.
    """
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{description} contains NaN or infinite values"
        )


def _load_encoder_id(path: Path) -> str:
    """Read + validate the encoder identity from the required sidecar."""
    encoder_id = path.read_text(encoding="utf-8").strip()
    if not encoder_id:
        raise ValueError(
            f"encoder file {path} is empty; must name the model used "
            f"to produce the embeddings (e.g. "
            f"'sentence-transformers/all-MiniLM-L6-v2'). The adapter "
            f"does not embed text — that's the consumer's "
            f"responsibility — but the encoder identity is pinned so "
            f"federation peers can detect mismatches."
        )
    return encoder_id
