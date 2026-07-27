"""Entity embedders for the KAL default storage backend.

Ports ``HashEntityEmbedder`` from
``knowly.modules.steering.inference``. The two implementations must
stay byte-equivalent (same SHA-256 seeding, same random-state
initialization, same unit-norm output) so federation between
knowly-backed and package-backed deployments produces identical
``cayley_embedding`` vectors for the same entity name.

``HashEntityEmbedder`` is the **default** the store wires in:
deterministic, no heavy deps (just ``hashlib`` + ``numpy``).
Users with richer needs inject a different ``EntityEmbedder`` impl at
``KalDefaultKGStore`` construction time — e.g. a sentence-transformer-
backed embedder that produces semantically-similar vectors for
similar names. The store stays agnostic; the embedder owns its
distance metric.

The 64-d default matches knowly's ``CAYLEY_DIM`` and the
``kal.kg_entities.cayley_embedding`` column dimension. Different
deployments target different dims via the ``dim`` constructor arg.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class EntityEmbedder(Protocol):
    """Protocol for entity embedding strategies.

    Implementations produce a unit vector for a given entity-name
    string. The dimension is the impl's choice; the store reads
    ``embedder.dim`` to know what column to fill.
    """

    dim: int

    def embed(self, text: str) -> np.ndarray:  # type: ignore[explicit-any]
        """Produce a unit vector for the given entity text."""
        ...


class DualEntityEmbedder(Protocol):
    """Protocol for GMS dual-embedding strategies.

    Implementations produce a ``(v, u)`` pair of unit vectors —
    Stiefel-projected dual embeddings used by GMS verification
    geometry (paper §6). Maps to ``kal.kg_entities.v_embedding`` and
    ``u_embedding`` columns; both are 128-d in the stock migration
    (``GMS_PROJECTED_DIM``).

    NO default impl ships in the package — production dual-embedders
    need ``torch`` + GMS infrastructure, which the package
    deliberately doesn't drag in as a hard dep. Users who want v/u
    fills inject their own embedder at ``KalDefaultKGStore``
    construction time (e.g. wrapping knowly's
    ``GMSEntityEmbedder``). When no dual-embedder is configured,
    ``v_embedding`` / ``u_embedding`` stay NULL — the store falls
    back to cayley-only embedding, which is the same behaviour
    knowly exhibits when ``[train]`` extras are missing.
    """

    dim: int

    def embed_dual(  # type: ignore[explicit-any, type-arg]
        self, text: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce a ``(v_embedding, u_embedding)`` pair.

        Both vectors are ``self.dim``-dimensional. Implementations may
        raise ``ImportError`` lazily (e.g. when a torch model fails to
        load) — the store catches that, treats it as "dual unavailable",
        and leaves v/u NULL for this call. The error is logged at
        DEBUG level so a repeating failure shows up in observability
        without spamming logs.
        """
        ...


class HashEntityEmbedder:
    """Deterministic hash-based entity embedder (no heavy deps).

    SHA-256 → seed → ``numpy.random.RandomState`` → randn(dim) →
    unit-normalize. Same name always produces the same vector; this
    is the "no embedding info available" fallback that gives the
    store SOMETHING to put in the embedding column rather than NULL.

    Two entities with similar names produce uncorrelated vectors —
    this is NOT a semantic embedder; it's a deterministic anchor.
    Callers that need similar-names-similar-vectors should inject a
    semantic embedder instead.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:  # type: ignore[explicit-any]
        """SHA-256 hash → seed → deterministic unit vector.

        Edge case: when the random vector happens to be all zeros
        (vanishingly rare for ``RandomState.randn``), unit-normalize
        would divide by zero — return the raw zero vector in that
        case to match knowly's behaviour at
        ``knowly.modules.steering.inference.HashEntityEmbedder``.
        """
        digest = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec  # type: ignore[no-any-return]
