# SPDX-License-Identifier: Apache-2.0
"""GEODE — self-supervised geometric extraction agent.

A closed loop that builds and *self-corrects* a GMS from documents without
human labels: a cheap actor proposes triples (regex backbone + a small local
LLM), the GMS geometry judges them as a label-free critic (plausibility, cap
admissibility, tension, holonomy), a calibrated gate accepts/rejects/reviews,
provenance points flags back to the source span, and confirmed structure feeds
the next round.

Design + validation: ``EXTRACTION_AGENT_DESIGN.md``. Key result: the geometry
provides label-free *correction* the LLM alone cannot — errors that violate
redundancy (relation composition) or declared constraints are caught and fixed;
isolated values with no cross-check are not (the honest limit).

Components
----------
* :class:`ProvenanceLedger` — map each triple to its exact source span.
* :class:`CompositionCritic` — flag triples that violate a learned relation
  composition (holonomy / path-consistency residual).
* :class:`GeodeLoop` — the orchestrator tying ingest → train → diagnose →
  localize → repair → reconverge.
* :func:`qwen_agent_callable` — the agent's ONLY runtime LLM (local Qwen 3B).
"""

from knowlytix.knowledge.geode.agent_llm import QWEN_3B, qwen_agent_callable
from knowlytix.knowledge.geode.anchor import (
    AnchorChecker,
    AnchorViolation,
    SumConstraint,
    enm_from_triples,
    numeric_facts_from_triples,
)
from knowlytix.knowledge.geode.canonicalize import Merge, canonicalize_graph
from knowlytix.knowledge.geode.critic import CompositionCritic, Flag
from knowlytix.knowledge.geode.dedup import resolve_duplicates
from knowlytix.knowledge.geode.loop import GeodeLoop, LoopResult, make_default_trainer
from knowlytix.knowledge.geode.provenance import (
    Provenance,
    ProvenanceLedger,
    is_consistent,
)
from knowlytix.knowledge.geode.rag import (
    RagBuildResult,
    build_calibrated_rag_store,
    build_rag_store,
    store_from_triples,
)

__all__ = [
    "ProvenanceLedger",
    "Provenance",
    "is_consistent",
    "CompositionCritic",
    "Flag",
    "AnchorChecker",
    "AnchorViolation",
    "SumConstraint",
    "enm_from_triples",
    "numeric_facts_from_triples",
    "GeodeLoop",
    "LoopResult",
    "make_default_trainer",
    "resolve_duplicates",
    "canonicalize_graph",
    "Merge",
    "RagBuildResult",
    "build_rag_store",
    "build_calibrated_rag_store",
    "store_from_triples",
    "qwen_agent_callable",
    "QWEN_3B",
]
