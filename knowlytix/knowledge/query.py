# SPDX-License-Identifier: Apache-2.0
"""Query engine: the GMS + LLM + verify + learn loop.

Routes questions to GMS subsystems, verifies LLM answers against the
geometric knowledge store, and optionally learns from verified answers.
"""

import re
from dataclasses import dataclass, field


from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.extract import (
    ScoredTriple, PlausibilityReport,
    extract_triples, score_triples, assess_plausibility,
)
from knowlytix.knowledge.learn import RuntimeLearner, LearnResult
from knowlytix.knowledge.llm_backend import LLMBackend
from knowlytix.knowledge.store import GMSExpertStore
from knowlytix.knowledge.verify import VerificationReport, verify_answer
from knowlytix.core.graph.search import GeometricSearch


@dataclass
class QueryResult:
    """Full result of an expert system query."""
    answer: str = ""
    source: str = ""                     # "GMS" or "External"
    source_detail: str = ""              # enm_lookup, triple, counting, web_search, llm, etc.
    confidence: float = 0.0              # from GMS geometry (geodesic / tension energy)
    confidence_method: str = ""          # how confidence was computed
    extracted_triples: list[ScoredTriple] = field(default_factory=list)
    plausibility: PlausibilityReport | None = None
    verification: VerificationReport | None = None
    learned: bool = False
    learn_result: LearnResult | None = None
    gms_context: str = ""
    web_sources: list = field(default_factory=list)
    triples_merged: int = 0


# =============================================================================
# System prompts
# =============================================================================

_SYSTEM_WITH_CONTEXT = (
    "You are an expert analyst backed by a geometric knowledge store. "
    "The provided context contains verified facts from the knowledge store. "
    "Use them as ground truth. For numeric values, prefer exact values from "
    "the knowledge store. If the context contradicts your knowledge, flag it. "
    "Be precise and specific."
)

_SYSTEM_NO_CONTEXT = (
    "You are an expert analyst. Answer questions precisely based on the "
    "provided document. Do NOT approximate or round values. "
    "Copy numbers exactly as they appear."
)


# =============================================================================
# Question classification
# =============================================================================

_QTYPE_PATTERNS = {
    "exact_recall": [
        r"exact value", r"what is the", r"how much", r"what was",
        r"report the", r"look up",
    ],
    "threshold": [
        r"above", r"below", r"exceeds?", r"threshold", r"greater than",
        r"less than", r"meets?", r"at least", r"at most",
    ],
    "contradiction": [
        r"contradict", r"conflict", r"inconsisten", r"disagree",
        r"both pass and fail",
    ],
    "multi_hop": [
        r"path", r"chain", r"leads? to", r"through", r"then find",
        r"step 1.*step 2", r"first.*then",
    ],
    "cross_reference": [
        r"both", r"intersection", r"and also", r"appear in.*and",
        r"common to",
    ],
    "counting": [
        r"how many", r"count", r"number of", r"total",
    ],
}


def classify_question(question: str) -> str:
    """Classify question type via keyword heuristics."""
    q_lower = question.lower()
    for qtype, patterns in _QTYPE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q_lower):
                return qtype
    return "open_ended"


# =============================================================================
# Query Engine
# =============================================================================

class QueryEngine:
    """Expert system query interface.

    The full query loop:
    1. GMS attempts structured answer
    2. If insufficient: LLM answers (with GMS context)
    3. Extract triples from LLM answer
    4. Score + verify (contradiction + holonomy)
    5. If verified + novel + auto_learn: grow GMS
    6. Return answer with full provenance
    """

    def __init__(self, store: GMSExpertStore, llm: LLMBackend,
                 config: DocGMSConfig):
        self.store = store
        self.llm = llm
        self.config = config
        self.device = store.device
        self.learner = RuntimeLearner(store, config.learn)

        # Geometric search from base GMS library
        self._geo_search: GeometricSearch | None = None
        if (store.model is not None and store.transport is not None
                and store.adapter is not None):
            self._geo_search = GeometricSearch(
                model=store.model,
                transport=store.transport,
                entity_to_idx=store.adapter.entity_to_idx,
                relation_to_idx=store.adapter.relation_to_idx,
                idx_to_entity=store.adapter.idx_to_entity,
                idx_to_relation=store.adapter.idx_to_relation,
                device=store.device,
            )
            if store.doc_graph is not None:
                self._geo_search.build_adjacency(store.doc_graph.triples)

    def query(self, question: str, mode: str = "gms_llm") -> QueryResult:
        """Query the expert system.

        Modes:
          gms_only   - structured GMS answer only
          llm_only   - LLM answer (still verified by GMS)
          gms_llm    - GMS first → LLM with GMS context → web agent
                       Three-tier fallback:
                       1. GMS structured answer (ENM, triples, counting, etc.)
                       2. LLM grounded in document + GMS context (verified by GMS)
                       3. Web agent for external knowledge (search, extract, verify, merge)
        """
        result = QueryResult()
        qtype = classify_question(question)

        if mode == "gms_only":
            return self._query_gms_only(question, qtype, result)

        if mode == "llm_only":
            return self._query_llm_only(question, result)

        # gms_llm: three-tier fallback

        # Tier 1: GMS structured answer
        gms_answer, gms_conf, gms_method, gms_detail = self._gms_answer(question, qtype)
        if gms_answer is not None and gms_conf > 0.5:
            result.answer = str(gms_answer)
            result.source = "GMS"
            result.source_detail = gms_detail
            result.confidence = gms_conf
            result.confidence_method = gms_method
            return result

        # Tier 2: LLM with GMS context (answers from ingested document)
        gms_context = self._gather_gms_context(question)
        llm_answer = self._llm_answer(question, gms_context=gms_context)
        if llm_answer and llm_answer.strip():
            # Detect refusal: LLM says it can't answer from the document
            if self._is_refusal(llm_answer):
                # Fall through to Tier 3
                pass
            else:
                result.answer = llm_answer
                result.source = "GMS"
                result.source_detail = "llm_grounded_in_document"
                result.gms_context = gms_context
                # Verify LLM answer against GMS
                self._process_llm_answer(question, llm_answer, result)
                # If verification produced a confidence, use it; otherwise moderate
                if result.confidence == 0.0:
                    result.confidence = 0.6
                result.confidence_method = "llm_verified_by_gms"
                return result

        # Tier 3: Web agent for external knowledge
        return self._query_web_agent(question, result)

    def query_batch(self, questions: list[str],
                     mode: str = "gms_llm") -> list[QueryResult]:
        """Batch query."""
        return [self.query(q, mode) for q in questions]

    # -----------------------------------------------------------------
    # Internal: GMS-only
    # -----------------------------------------------------------------

    def _query_gms_only(self, question: str, qtype: str,
                         result: QueryResult) -> QueryResult:
        """Attempt structured GMS answer."""
        ans, conf, method, detail = self._gms_answer(question, qtype)
        if ans is not None:
            result.answer = str(ans)
            result.source = "GMS"
            result.source_detail = detail
            result.confidence = conf
            result.confidence_method = method
        else:
            result.answer = ""
            result.source = "GMS"
            result.source_detail = "no_match"
            result.confidence = 0.0
            result.confidence_method = "none"
        return result

    def _gms_answer(self, question: str, qtype: str) -> tuple:
        """Attempt a GMS-only answer.

        Returns (answer, confidence, confidence_method, source_detail)
        or (None, 0, "", "") if not found.

        Confidence is computed from GMS geometry:
        - ENM: exact lossless recall (SHA-256 verified) → score via geodesic
        - Triples: geodesic distance d(R_r(v_h), v_t) on S^{m-1}
        - Counting: graph traversal (deterministic)
        - Contradiction: tension energy E = 2sin(θ/2)
        """
        q_lower = question.lower()

        if qtype == "exact_recall":
            r = self._gms_exact_recall(q_lower)
            if r[0] is not None:
                return r
        if qtype == "counting":
            r = self._gms_counting(q_lower)
            if r[0] is not None:
                return r
        if qtype == "contradiction":
            r = self._gms_contradiction(q_lower)
            if r[0] is not None:
                return r
        if qtype == "threshold":
            r = self._gms_threshold(q_lower)
            if r[0] is not None:
                return r

        # Fallback: try ENM lookup for any question
        r = self._gms_exact_recall(q_lower)
        if r[0] is not None:
            return r

        # Try triple-based answer
        r = self._gms_triple_search(q_lower)
        if r[0] is not None:
            return r

        return None, 0.0, "", ""

    def _gms_exact_recall(self, q_lower: str) -> tuple:
        """ENM lookup. Confidence = 1 - geodesic_distance for the entity pair."""
        if self.store.enm is None:
            return None, 0.0, "", ""
        for key in self.store.enm.keys():
            if key.id.lower() in q_lower or key.type.lower() in q_lower:
                val = self.store.lookup_enm(key.type, key.id)
                if val is not None:
                    # ENM is lossless (SHA-256 verified), but compute
                    # geometric confidence from entity embedding quality
                    conf = self._enm_geometric_confidence(key.type, key.id)
                    return (val, conf,
                            f"ENM lossless recall + geodesic={1-conf:.4f}",
                            f"enm:{key.type}/{key.id}")
        return None, 0.0, "", ""

    def _enm_geometric_confidence(self, category: str, entity_id: str) -> float:
        """Compute geometric confidence for an ENM entry.

        Uses average geodesic distance of triples involving the entity.
        Lower geodesic = better embedding = higher confidence.
        """
        if self.store.model is None or self.store.adapter is None:
            return 1.0  # ENM is exact regardless

        # Find the entity in the graph and score its triples
        parts = entity_id.split("/")
        entity = parts[0] if parts else entity_id
        matched = self.store.fuzzy_match_entity(entity)
        if not matched:
            return 1.0  # ENM value is exact, just can't score geometrically

        triples = self.store.query_triples(head=matched)
        if not triples:
            return 1.0

        scores = []
        for h, r, t in triples[:5]:
            s = self.store.score_triple(h, r, t)
            if s is not None:
                scores.append(s)

        if not scores:
            return 1.0

        # Convert geodesic distance to confidence: conf = exp(-mean_dist)
        import math
        mean_dist = sum(scores) / len(scores)
        return math.exp(-mean_dist)

    def _gms_counting(self, q_lower: str) -> tuple:
        """Count triples matching a pattern. Confidence from graph completeness."""
        if self.store.doc_graph is None:
            return None, 0.0, "", ""
        for rel in (self.store.adapter.relation_to_idx or {}):
            if rel.replace("_", " ") in q_lower:
                matches = self.store.query_triples(relation=rel)
                # Confidence: score a sample of matched triples geometrically
                conf = self._triple_set_confidence(matches[:10])
                return (len(matches), conf,
                        f"graph traversal, {len(matches)} matches, mean_geodesic={1-conf:.4f}",
                        f"counting:{rel}")
        return None, 0.0, "", ""

    def _gms_contradiction(self, q_lower: str) -> tuple:
        """Contradiction detection via tension energy."""
        if self.store.doc_graph is None:
            return None, 0.0, "", ""
        contras = self.store.find_contradictions()
        if not contras:
            return None, 0.0, "", ""

        # Confidence from mean tension energy of detected contradictions
        tensions = []
        for pair in contras:
            if len(pair) >= 2:
                te = self.store.tension_energy(pair[0], pair[1])
                if te is not None:
                    tensions.append(te)

        if tensions:
            mean_te = sum(tensions) / len(tensions)
            # Tension energy: 0=agree, √2=irrelevant, 2=contradict
            # Confidence that contradiction is real: te/2
            conf = mean_te / 2.0
        else:
            conf = 0.5

        if "how many" in q_lower:
            return (len(contras), conf,
                    f"tension energy, mean_E={mean_te:.4f}" if tensions else "graph count",
                    "contradiction:count")
        return (bool(contras), conf,
                f"tension energy, mean_E={mean_te:.4f}" if tensions else "graph detection",
                "contradiction:detect")

    def _gms_threshold(self, q_lower: str) -> tuple:
        """Threshold check using ENM values + geometric confidence."""
        if self.store.enm is None:
            return None, 0.0, "", ""
        for key in self.store.enm.keys():
            if key.id.lower() in q_lower or key.type.lower() in q_lower:
                val = self.store.lookup_enm(key.type, key.id)
                if val is not None:
                    import re
                    nums = re.findall(r'[-+]?\d*\.?\d+', q_lower)
                    if nums:
                        threshold = float(nums[-1])
                        v = float(val.item() if hasattr(val, 'item') else val)
                        conf = self._enm_geometric_confidence(key.type, key.id)
                        if "above" in q_lower or "greater" in q_lower or "exceeds" in q_lower or "at least" in q_lower:
                            return (v > threshold, conf,
                                    f"ENM={v}, threshold={threshold}, geodesic_conf={conf:.4f}",
                                    f"threshold:{key.type}/{key.id}")
                        if "below" in q_lower or "less than" in q_lower or "at most" in q_lower:
                            return (v < threshold, conf,
                                    f"ENM={v}, threshold={threshold}, geodesic_conf={conf:.4f}",
                                    f"threshold:{key.type}/{key.id}")
                        if "meets" in q_lower or "better than" in q_lower:
                            return (v >= threshold, conf,
                                    f"ENM={v}, threshold={threshold}, geodesic_conf={conf:.4f}",
                                    f"threshold:{key.type}/{key.id}")
        return None, 0.0, "", ""

    # Common words to skip during entity matching
    _STOP_WORDS = {
        "what", "which", "where", "when", "how", "does", "this",
        "that", "with", "from", "about", "have", "been", "also",
        "some", "they", "their", "there", "these", "those", "more",
        "most", "into", "over", "only", "than", "then", "very",
    }

    @staticmethod
    def _extract_query_words(q_lower: str) -> list[str]:
        """Extract content words from question, stripping punctuation."""
        import re
        words = re.findall(r'[a-z0-9_-]+', q_lower)
        return [
            w for w in words
            if len(w) > 3 and w not in QueryEngine._STOP_WORDS
        ]

    def _match_entities(self, q_lower: str) -> list[str]:
        """Find entities matching the question via word overlap.

        Same logic used by _gather_gms_context so GMS-only and GMS+LLM
        see the same knowledge.
        """
        if self.store.adapter is None:
            return []
        query_words = self._extract_query_words(q_lower)
        if not query_words:
            return []
        matched = []
        for ent in self.store.adapter.entity_to_idx:
            ent_lower = ent.lower()
            # Strip markdown formatting for matching
            ent_clean = ent_lower.replace("**", "").replace("*", "").strip()
            if ent_clean in q_lower or any(w in ent_clean for w in query_words):
                matched.append(ent)
                if len(matched) >= 20:
                    break
        return matched

    def _gms_triple_search(self, q_lower: str) -> tuple:
        """Multi-hop graph search connecting key concepts in the question.

        Strategy:
        1. Extract key concept words from question
        2. Group entities by which concept they match
        3. If multiple concept groups: BFS to find connecting paths
           (adaptive hop count — stops when concepts are bridged)
        4. If single concept: expand outward from matched entities
        5. Score paths using geodesic distance on S^{m-1}

        Example: "What is bivector do for steering?"
          → concepts: {bivector, steering}
          → bivector entities: [**Bivector**, Bivector B_l, ...]
          → steering entities: [Plane-specific, application_4..., ...]
          → BFS finds 3-hop path: Bivector → section → application_4 → steering
          → returns all triples along the path
        """
        if self.store.doc_graph is None or self.store.adapter is None:
            return None, 0.0, "", ""

        # Step 1: Extract concept words and group entities
        query_words = self._extract_query_words(q_lower)
        if not query_words:
            return None, 0.0, "", ""

        concept_groups = self._group_entities_by_concept(query_words)
        if not concept_groups:
            return None, 0.0, "", ""

        # Step 2: Find triples — strategy depends on concept count
        concepts = list(concept_groups.keys())
        all_entities_flat = set()
        for ents in concept_groups.values():
            all_entities_flat.update(ents)

        if len(concepts) >= 2:
            # Multi-concept: geometrically-scored path search
            path_triples, n_hops, scored_paths = self._find_connecting_paths(
                concept_groups, max_hops=5
            )
        else:
            path_triples = []
            n_hops = 0
            scored_paths = []

        # Step 3: Also collect direct triples for all matched entities
        seen = set()
        all_triples = []

        # Add path triples first (they bridge concepts)
        for triple in path_triples:
            if triple not in seen:
                seen.add(triple)
                all_triples.append(triple)

        # Add direct triples for matched entities
        for ent in list(all_entities_flat)[:15]:
            for h, r, t in self.store.query_triples(head=ent):
                key = (h, r, t)
                if key not in seen:
                    seen.add(key)
                    all_triples.append(key)
            for h, r, t in self.store.query_triples(tail=ent):
                key = (h, r, t)
                if key not in seen:
                    seen.add(key)
                    all_triples.append(key)

        if not all_triples:
            return None, 0.0, "", ""

        # Format results — skip in_section for readability
        results = []
        for h, r, t in all_triples:
            if r == "in_section":
                continue
            results.append(f"{h} | {r} | {t}")

        if not results:
            results = [f"{h} | {r} | {t}" for h, r, t in all_triples[:20]]

        conf = self._triple_set_confidence(all_triples)

        # Build method string with geometric path scores
        hop_info = f"{n_hops}-hop bridge" if n_hops > 0 else "direct"
        path_info = ""
        if scored_paths:
            best_geo = min(s[1] for s in scored_paths if s[1] < 999)  \
                if any(s[1] < 999 for s in scored_paths) else None
            n_scored = sum(1 for s in scored_paths if s[1] < 999)
            if best_geo is not None:
                path_info = f", {n_scored} paths scored, best_path_geo={best_geo:.4f}"

        return ("\n".join(results), conf,
                f"graph search ({hop_info}), {len(results)} facts, "
                f"concepts={concepts}, mean_geodesic={1-conf:.4f}{path_info}",
                f"triple_search:{','.join(concepts)}")

    def _group_entities_by_concept(self, query_words: list[str]) -> dict:
        """Group matched entities by which concept word they match.

        Collects all matching entities then ranks by name length
        (shorter = more likely a real graph node, not a definition string).
        Returns dict of concept_word -> list of matching entity names.
        """
        groups = {}
        for word in query_words:
            matched = []
            for ent in self.store.adapter.entity_to_idx:
                ent_clean = ent.lower().replace("**", "").replace("*", "").strip()
                if word in ent_clean:
                    matched.append(ent)
            if matched:
                # Sort by name length — shorter entities are more navigable
                matched.sort(key=len)
                groups[word] = matched[:20]
        return groups

    def _find_connecting_paths(self, concept_groups: dict,
                                max_hops: int = 5) -> tuple:
        """Geometrically-scored BFS connecting concept entity groups.

        Delegates to gms.graph.search.GeometricSearch for BFS + path scoring.
        Returns (list_of_triples_on_paths, n_hops_needed, scored_paths).
        """
        if self._geo_search is None or len(concept_groups) < 2:
            return [], 0, []

        result = self._geo_search.connect_concept_groups(
            concept_groups, max_hops=max_hops,
        )
        # Convert ScoredPath objects to legacy tuple format for compatibility
        scored_tuples = [
            (p.triples, p.geodesic, p.holonomy) for p in result.paths
        ]
        return result.all_triples, result.n_hops, scored_tuples

    def _triple_set_confidence(self, triples: list[tuple]) -> float:
        """Compute geometric confidence for a set of triples.

        Delegates to GeometricSearch.score_triple_set() which scores each
        triple via geodesic distance d(R_r(v_h), v_t) on S^{m-1},
        then returns conf = exp(-mean_distance).
        """
        if self._geo_search is not None:
            return self._geo_search.score_triple_set(triples)
        return 0.5

    # -----------------------------------------------------------------
    # Internal: Web Agent (GMS + web search + verify + merge)
    # -----------------------------------------------------------------

    def _query_web_agent(self, question: str,
                          result: QueryResult) -> QueryResult:
        """Use web agent: search → extract → verify → merge → answer."""
        from knowlytix.knowledge.web_agent import WebAgent

        agent = WebAgent(
            self.store, self.llm, self.config, self.device,
            max_sources=5, max_search_rounds=2,
        )
        web_result = agent.research(question)

        result.answer = web_result.answer
        if web_result.answer_source == "gms":
            result.source = "GMS"
            result.source_detail = "enriched_by_web"
        else:
            result.source = "External"
            result.source_detail = f"web:{web_result.answer_source}"
        result.confidence = web_result.confidence
        result.confidence_method = f"web_agent:{web_result.answer_source}"
        result.web_sources = web_result.sources
        result.triples_merged = web_result.triples_merged

        # If web agent merged triples, mark as learned
        if web_result.triples_merged > 0:
            result.learned = True
            result.learn_result = LearnResult(
                accepted=True,
                reason=(f"Web agent merged {web_result.triples_merged} triples "
                        f"from {len(web_result.sources)} sources"),
            )

        return result

    # -----------------------------------------------------------------
    # Internal: LLM-only (still verified by GMS, no web search)
    # -----------------------------------------------------------------

    def _query_llm_only(self, question: str,
                         result: QueryResult) -> QueryResult:
        """LLM answer, still verified by GMS."""
        llm_answer = self._llm_answer(question, gms_context=None)
        result.answer = llm_answer
        result.source = "llm"
        self._process_llm_answer(question, llm_answer, result)
        return result

    # -----------------------------------------------------------------
    # Internal: LLM interaction
    # -----------------------------------------------------------------

    def _gather_gms_context(self, question: str) -> str:
        """Build GMS context for LLM augmentation.

        Uses same multi-hop graph search as _gms_triple_search so
        GMS-only and GMS+LLM see the same knowledge.
        """
        if self.store.adapter is None:
            return ""

        context_parts = []
        q_lower = question.lower()

        # Multi-hop search: same as _gms_triple_search
        query_words = self._extract_query_words(q_lower)
        concept_groups = self._group_entities_by_concept(query_words) if query_words else {}

        all_entities_flat = set()
        for ents in concept_groups.values():
            all_entities_flat.update(ents)

        # Find connecting paths between concepts
        seen = set()
        if len(concept_groups) >= 2:
            path_triples, _, _ = self._find_connecting_paths(
                concept_groups, max_hops=5
            )
            for h, r, t in path_triples:
                if (h, r, t) not in seen:
                    seen.add((h, r, t))
                    if r != "in_section":
                        context_parts.append(f"FACT: {h} | {r} | {t}")

        # Direct triples for matched entities
        for ent in list(all_entities_flat)[:15]:
            for h, r, t in self.store.query_triples(head=ent):
                if (h, r, t) not in seen:
                    seen.add((h, r, t))
                    if r != "in_section":
                        context_parts.append(f"FACT: {h} | {r} | {t}")
            for h, r, t in self.store.query_triples(tail=ent):
                if (h, r, t) not in seen:
                    seen.add((h, r, t))
                    if r != "in_section":
                        context_parts.append(f"FACT: {h} | {r} | {t}")

        # Gather ENM values for matched entities
        if self.store.enm:
            for key in self.store.enm.keys():
                for ent in list(all_entities_flat)[:10]:
                    if ent.lower() in key.id.lower():
                        val = self.store.lookup_enm(key.type, key.id)
                        if val is not None:
                            context_parts.append(
                                f"EXACT VALUE: {key.type}/{key.id} = {val}"
                            )

        if not context_parts:
            return ""
        return "\n".join(context_parts[:50])

    def _llm_answer(self, question: str, gms_context: str | None) -> str:
        """Call LLM with optional GMS context."""
        if gms_context:
            system = _SYSTEM_WITH_CONTEXT
            user = (
                f"<knowledge_store>\n{gms_context}\n</knowledge_store>\n\n"
                f"<document>\n{self.store.markdown[:80000]}\n</document>\n\n"
                f"Question: {question}"
            )
        else:
            system = _SYSTEM_NO_CONTEXT
            user = (
                f"<document>\n{self.store.markdown[:80000]}\n</document>\n\n"
                f"Question: {question}"
            )
        return self.llm.call(system, user, max_tokens=2048)

    @staticmethod
    def _is_refusal(answer: str) -> bool:
        """Detect if the LLM refused to answer from the document."""
        lower = answer[:500].lower()
        refusal_phrases = [
            "cannot answer", "can't answer", "cannot find",
            "can't find", "not mentioned", "not in the document",
            "not in the provided", "does not mention",
            "does not contain", "doesn't mention",
            "no information about", "no mention of",
            "not available in", "not covered in",
            "outside the scope", "beyond the scope",
            "i must flag that",
        ]
        return any(phrase in lower for phrase in refusal_phrases)

    # -----------------------------------------------------------------
    # Internal: Process LLM answer (extract, score, verify, learn)
    # -----------------------------------------------------------------

    def _process_llm_answer(self, question: str, llm_answer: str,
                             result: QueryResult):
        """Extract triples, score, verify, optionally learn."""
        # Extract triples
        try:
            extracted = extract_triples(llm_answer, question, self.llm)
        except Exception:
            extracted = []

        if not extracted:
            result.confidence = 0.5  # Can't verify
            return

        # Score against GMS
        scored = score_triples(extracted, self.store, self.device)
        result.extracted_triples = scored

        # Plausibility
        plaus = assess_plausibility(
            scored, self.config.verify.min_plausibility
        )
        result.plausibility = plaus

        # Verify: contradiction + holonomy
        verification = verify_answer(
            scored, self.store, self.config.verify
        )
        result.verification = verification
        result.confidence = verification.confidence

        # Learn: if auto_learn and verified and has novel facts
        if (self.config.learn.auto_learn
                and verification.overall_consistent
                and any(s.is_novel for s in scored)):
            novel_triples = [s.triple for s in scored if s.is_novel]
            try:
                learn_result = self.learner.learn_triples(
                    novel_triples, self.device
                )
                result.learned = learn_result.accepted
                result.learn_result = learn_result
            except Exception as e:
                result.learn_result = LearnResult(
                    accepted=False, reason=f"Learning failed: {e}"
                )
