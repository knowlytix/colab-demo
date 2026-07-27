# SPDX-License-Identifier: Apache-2.0
"""
DocumentGraph — Domain-agnostic structured memory for benchmark ground truth.

Stores:
  - ENM entries: exact numerical values with SHA-256 integrity
  - KG triples: (head, relation, tail) relationships
  - Phase encoders: threshold/inequality queries on numeric ranges
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ENMKey:
    """Structured key: (type, id)."""
    type: str
    id: str

    def __hash__(self):
        return hash((self.type, self.id))


@dataclass
class ENMEntry:
    """Stored numeric value with integrity hash."""
    key: ENMKey
    value: float
    hash: bytes  # SHA-256

    @staticmethod
    def compute_hash(value: float) -> bytes:
        arr = np.array(value, dtype=np.float64)
        return hashlib.sha256(arr.tobytes()).digest()


class PhaseEncoder:
    """Threshold checker and angle encoder for a named metric type.

    Carries two responsibilities:

    * ``check_inequality`` — used by the FinStructBench benchmark to grade
      threshold-style questions on the structured graph baseline.
    * ``encode`` — used by the GMS library and the testing harness
      (``knowlytix.harness.testing.judge``) to map a numeric value to an angle in
      :math:`[0, \\pi]`. This is the basis of the phase-encoding scheme
      documented in the GMS paper §4 ("Phase Encoding for Numeric
      Constraints"); removing it would break the geometric judge.
    """

    def __init__(self, v_min: float = 0.0, v_max: float = 1.0):
        self.v_min = v_min
        self.v_max = v_max

    def encode(self, value: float) -> float:
        """Map ``value`` to angle in :math:`[0, \\pi]`.

        Used by the geometric judge and the phase-encoding pipeline. The
        mapping is monotone and saturates at the endpoints.
        """
        t = (value - self.v_min) / (self.v_max - self.v_min + 1e-12)
        t = max(0.0, min(1.0, t))
        return t * np.pi

    def check_inequality(self, value: float, limit: float, op: str):
        """Check if value satisfies inequality vs limit.

        Args:
            op: "ge" (>=), "gt" (>), "le" (<=), "lt" (<)

        Returns:
            (satisfied: bool, margin: float)
        """
        ops = {
            "ge": lambda v, l: v >= l,
            "gt": lambda v, l: v > l,
            "le": lambda v, l: v <= l,
            "lt": lambda v, l: v < l,
        }
        fn = ops.get(op)
        if fn is None:
            return None, None
        return fn(value, limit), value - limit


class DocumentGraph:
    """Domain-agnostic graph store for structured document data.

    This is the ground-truth engine for the benchmark.
    Any correct graph implementation produces identical answers.
    """

    def __init__(self):
        self.enm: dict[ENMKey, ENMEntry] = OrderedDict()
        self.enm_meta: dict[ENMKey, dict[str, str]] = {}
        self.triples: list[tuple[str, str, str]] = []
        self._triple_set: set[tuple[str, str, str]] = set()
        self.phase_encoders: dict[str, PhaseEncoder] = {}
        self.metadata: dict[str, Any] = {}

    # --- ENM operations ---

    def store_value(self, category: str, entity_id: str, value: float,
                    *, column: str | None = None, entity: str | None = None):
        """Store an exact numeric value.

        Args:
            category: Section-derived ENM type (for graph baseline).
            entity_id: Composite entity ID (may include column name).
            value: The numeric value.
            column: Optional original column header name (kept in
                ``enm_meta`` for external verification by the hybrid
                ingestion path).
            entity: Optional original entity name without column suffix
                (kept in ``enm_meta``).
        """
        key = ENMKey(type=category, id=entity_id)
        entry = ENMEntry(
            key=key, value=value,
            hash=ENMEntry.compute_hash(value),
        )
        self.enm[key] = entry
        if column or entity:
            meta = self.enm_meta.setdefault(key, {})
            if column:
                meta["column"] = column
            if entity:
                meta["entity"] = entity

    def lookup(self, category: str, entity_id: str) -> float | None:
        """Exact key lookup."""
        key = ENMKey(type=category, id=entity_id)
        entry = self.enm.get(key)
        if entry is None:
            return None
        # Verify integrity
        expected = ENMEntry.compute_hash(entry.value)
        if expected != entry.hash:
            raise RuntimeError(f"Integrity check failed for {key}")
        return entry.value

    # --- Triple operations ---

    @staticmethod
    def _canon_entity(name: str) -> str:
        # Collapse casing/whitespace variants of the same entity.  Without this,
        # multi-hop BFS dead-ends when one triple stores a tail "Data profiling"
        # and the next triple's head is "data profiling" — _adj is keyed by the
        # literal string, so the walk lands on a node with no outgoing edges.
        # Holonomy stops discriminating for the same reason: each casing variant
        # gets its own embedding, so composed rotors steer toward an embedding
        # the chain never lands on.  Relations stay as-is (already snake_case).
        return " ".join(name.split()).lower()

    def add_triple(self, head: str, relation: str, tail: str):
        """Add a knowledge graph triple (deduplicated)."""
        head = self._canon_entity(head)
        tail = self._canon_entity(tail)
        triple = (head, relation, tail)
        if triple not in self._triple_set:
            self._triple_set.add(triple)
            self.triples.append(triple)

    def query_triples(self, head=None, relation=None, tail=None):
        """Pattern-match triples."""
        if head is not None:
            head = self._canon_entity(head)
        if tail is not None:
            tail = self._canon_entity(tail)
        results = []
        for h, r, t in self.triples:
            if head and h != head:
                continue
            if relation and r != relation:
                continue
            if tail and t != tail:
                continue
            results.append((h, r, t))
        return results

    # --- Phase operations ---

    def add_phase_encoder(self, name: str, v_min: float = 0.0, v_max: float = 1.0):
        """Register a phase encoder for a metric type."""
        self.phase_encoders[name] = PhaseEncoder(v_min, v_max)

    def check_threshold(self, encoder_name: str, value: float,
                        limit: float, op: str):
        """Phase-encoded threshold check."""
        enc = self.phase_encoders.get(encoder_name)
        if enc is None:
            return None, None
        return enc.check_inequality(value, limit, op)

    # --- Derived operations (used by generators) ---

    def find_contradictions(self):
        """Find entities that both pass and fail tests within the same group."""
        passes = {}
        fails = {}
        for h, r, t in self.triples:
            if r == "passes":
                passes.setdefault(h, set()).add(t)
            elif r == "fails":
                fails.setdefault(h, set()).add(t)

        contradictions = []
        for entity in set(passes.keys()) & set(fails.keys()):
            def extract_group(test_name):
                parts = test_name.split("_", 1)
                if len(parts) >= 2:
                    return parts[0] + "_" + parts[1].split(":")[0]
                return test_name

            pass_groups = {}
            for p in passes[entity]:
                g = extract_group(p)
                pass_groups.setdefault(g, []).append(p)

            fail_groups = {}
            for f in fails[entity]:
                g = extract_group(f)
                fail_groups.setdefault(g, []).append(f)

            for group in set(pass_groups.keys()) & set(fail_groups.keys()):
                for p in pass_groups[group]:
                    for f in fail_groups[group]:
                        contradictions.append((entity, p, f))

        return contradictions

    # --- Stats ---

    def stats(self) -> dict:
        """Return ingestion statistics."""
        enm_types = {}
        for k in self.enm:
            enm_types.setdefault(k.type, 0)
            enm_types[k.type] += 1

        relations = {}
        for _, r, _ in self.triples:
            relations.setdefault(r, 0)
            relations[r] += 1

        return {
            "enm_entries": len(self.enm),
            "enm_types": enm_types,
            "triples": len(self.triples),
            "relations": relations,
            "phase_encoders": list(self.phase_encoders.keys()),
        }
