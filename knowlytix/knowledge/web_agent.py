# SPDX-License-Identifier: Apache-2.0
"""Web research agent: search → fetch → extract → build source GMS → merge.

Instead of LLM answering directly, the agent:
1. Searches the web for relevant sources
2. Extracts content from each source
3. Builds a per-source knowledge graph (triples + ENM)
4. Verifies source knowledge against the main GMS
5. Merges verified, non-contradictory knowledge into the main GMS
6. Answers from the enriched GMS

All knowledge is grounded in sources, not hallucinated.
"""

import time
from dataclasses import dataclass, field

import torch


from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.extract import (
    ExtractedTriple, ScoredTriple, extract_triples, score_triples,
)
from knowlytix.knowledge.learn import RuntimeLearner
from knowlytix.knowledge.llm_backend import LLMBackend
from knowlytix.knowledge.store import GMSExpertStore
from knowlytix.knowledge.verify import verify_answer


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class WebSource:
    """A web source fetched by the agent."""
    url: str = ""
    title: str = ""
    content: str = ""          # raw text/markdown from the source
    triples: list[ExtractedTriple] = field(default_factory=list)
    scored: list[ScoredTriple] = field(default_factory=list)
    is_consistent: bool = True
    n_novel: int = 0
    n_contradictions: int = 0


@dataclass
class WebSearchResult:
    """Result of a web agent research cycle."""
    query: str = ""
    sources: list[WebSource] = field(default_factory=list)
    total_triples_found: int = 0
    triples_merged: int = 0
    triples_rejected: int = 0
    answer: str = ""
    answer_source: str = ""    # "gms" if answered from enriched store, "synthesis" otherwise
    confidence: float = 0.0
    elapsed_seconds: float = 0.0


# =============================================================================
# Web Agent
# =============================================================================

class WebAgent:
    """Research agent that searches the web, extracts knowledge, and merges
    verified facts into the GMS expert store.

    Protocol:
    1. Generate search queries from the user question
    2. Call Anthropic web search to find sources
    3. For each source: extract triples via LLM
    4. Score each triple against the main GMS
    5. Verify: contradiction (tension energy) + consistency (holonomy)
    6. Merge verified, non-contradictory triples into GMS
    7. Re-attempt GMS answer from enriched store
    8. If still can't answer from GMS: synthesize from verified triples
    """

    def __init__(self, store: GMSExpertStore, llm: LLMBackend,
                 config: DocGMSConfig, device: torch.device,
                 max_sources: int = 5, max_search_rounds: int = 2):
        self.store = store
        self.llm = llm
        self.config = config
        self.device = device
        self.max_sources = max_sources
        self.max_search_rounds = max_search_rounds
        self.learner = RuntimeLearner(store, config.learn)

    def research(self, question: str) -> WebSearchResult:
        """Full research cycle: search → extract → verify → merge → answer."""
        t0 = time.time()
        result = WebSearchResult(query=question)

        # Step 1: Search the web
        print(f"\n  [WebAgent] Searching for: {question[:80]}...")
        sources = self._web_search(question)
        print(f"  [WebAgent] Found {len(sources)} sources")

        if not sources:
            result.elapsed_seconds = time.time() - t0
            return result

        # Step 2: Extract triples from each source
        for i, src in enumerate(sources):
            print(f"  [WebAgent] Extracting from source {i+1}/{len(sources)}: "
                  f"{src.title[:50]}...")
            self._extract_source_triples(src, question)
            result.total_triples_found += len(src.triples)

        result.sources = sources

        # Step 3: Score and verify each source's triples against main GMS
        for src in sources:
            if src.triples:
                self._verify_source(src)

        # Step 4: Merge verified, non-contradictory triples
        merged, rejected = self._merge_verified_sources(sources)
        result.triples_merged = merged
        result.triples_rejected = rejected

        if merged > 0:
            print(f"  [WebAgent] Merged {merged} triples into GMS "
                  f"(rejected {rejected})")
            # All Phase 1 writes done — start background training
            self.learner.flush_training()
        else:
            print(f"  [WebAgent] No new triples to merge "
                  f"(rejected {rejected})")

        # Step 5: Try to answer from enriched GMS
        answer, confidence, source_type = self._answer_from_enriched_gms(
            question, sources
        )
        result.answer = answer
        result.confidence = confidence
        result.answer_source = source_type
        result.elapsed_seconds = time.time() - t0

        print(f"  [WebAgent] Answer source: {source_type} "
              f"(confidence: {confidence:.2f}, {result.elapsed_seconds:.1f}s)")
        print(f"  [WebAgent] Training: {self.learner.training_status()}")

        return result

    # -----------------------------------------------------------------
    # Step 1: Web search via Anthropic
    # -----------------------------------------------------------------

    def _web_search(self, question: str) -> list[WebSource]:
        """Search the web using Anthropic's web search tool."""
        import anthropic

        from knowlytix.knowledge.llm_backend import AnthropicBackend

        client = anthropic.Anthropic()

        # The Anthropic SDK expects a bare Claude model name (no
        # ``anthropic/`` prefix). The LLMBackend's ``model_name`` is the
        # LiteLLM-format string; use ``bare_model_name`` when the
        # backend is an Anthropic subclass.
        if isinstance(self.llm, AnthropicBackend):
            sdk_model = self.llm.bare_model_name
        else:
            sdk_model = self.llm.model_name

        # Use Claude with web search tool to find and extract sources
        response = client.messages.create(
            model=sdk_model,
            max_tokens=4096,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": self.max_search_rounds,
            }],
            system=(
                "You are a research assistant. Search the web to find factual "
                "information relevant to the user's question. After searching, "
                "synthesize the key facts from the sources into a structured "
                "summary. For each fact, note which source it came from."
            ),
            messages=[{"role": "user", "content": question}],
        )

        # Parse response to extract sources and content
        sources = []
        synthesized_text = ""

        for block in response.content:
            block_type = type(block).__name__
            if block_type == "TextBlock":
                synthesized_text += block.text + "\n"
            elif getattr(block, "type", "") == "web_search_tool_result":
                # Extract source URLs and titles
                if hasattr(block, "content") and isinstance(block.content, list):
                    for item in block.content:
                        if hasattr(item, "url") and hasattr(item, "title"):
                            src = WebSource(
                                url=item.url,
                                title=getattr(item, "title", ""),
                                content=getattr(item, "page_content", ""),
                            )
                            sources.append(src)

        # Deduplicate by URL
        seen_urls = set()
        unique = []
        for src in sources:
            if src.url not in seen_urls:
                seen_urls.add(src.url)
                unique.append(src)

        # Store synthesized text in the first source if we have it
        # This is the LLM's synthesis of the search results
        if synthesized_text.strip() and unique:
            unique[0].content = synthesized_text.strip()

        return unique[:self.max_sources]

    # -----------------------------------------------------------------
    # Step 2: Extract triples from each source
    # -----------------------------------------------------------------

    def _extract_source_triples(self, source: WebSource, question: str):
        """Extract structured triples from a web source."""
        # Use the source content (or URL+title if no content)
        text = source.content if source.content else f"{source.title}\n{source.url}"
        if not text.strip():
            return

        try:
            triples = extract_triples(text, question, self.llm)
            source.triples = triples
        except Exception as e:
            print(f"    Extract error: {e}")

    # -----------------------------------------------------------------
    # Step 3: Score and verify source triples
    # -----------------------------------------------------------------

    def _verify_source(self, source: WebSource):
        """Score and verify a source's triples against the main GMS."""
        if not source.triples:
            return

        # Score against GMS manifold
        scored = score_triples(source.triples, self.store, self.device)
        source.scored = scored

        # Count novel triples
        source.n_novel = sum(1 for s in scored if s.is_novel)

        # Verify: contradiction + holonomy
        if scored:
            verification = verify_answer(scored, self.store, self.config.verify)
            source.is_consistent = verification.overall_consistent
            source.n_contradictions = sum(
                1 for c in verification.contradictions if c.is_contradiction
            )

    # -----------------------------------------------------------------
    # Step 4: Merge verified triples into main GMS
    # -----------------------------------------------------------------

    def _merge_verified_sources(self, sources: list[WebSource]) -> tuple[int, int]:
        """Merge verified, non-contradictory triples into the main GMS.

        Returns (merged_count, rejected_count).
        """
        merged = 0
        rejected = 0

        for source in sources:
            if not source.scored:
                continue

            # Only merge from consistent sources
            if not source.is_consistent:
                rejected += len(source.scored)
                print(f"    Rejected source (contradictions={source.n_contradictions}): "
                      f"{source.title[:40]}")
                continue

            # Collect novel triples for learning
            novel_triples = [s.triple for s in source.scored if s.is_novel]

            if not novel_triples:
                continue

            # Use RuntimeLearner to merge with contradiction gate
            try:
                learn_result = self.learner.learn_triples(novel_triples, self.device)
                if learn_result.accepted:
                    merged += len(novel_triples)
                    print(f"    Merged {len(novel_triples)} triples from: "
                          f"{source.title[:40]}")
                    if learn_result.reason:
                        print(f"      {learn_result.reason}")
                else:
                    rejected += len(novel_triples)
                    print(f"    Rejected {len(novel_triples)} triples: "
                          f"{learn_result.reason[:60]}")
            except Exception as e:
                rejected += len(novel_triples)
                print(f"    Learning error: {e}")

            # Also add known (non-novel) triples to doc_graph if not present
            for s in source.scored:
                if not s.is_novel and s.head_match and s.tail_match:
                    rel = s.triple.relation.strip().lower().replace(" ", "_")
                    triple = (s.head_match, rel, s.tail_match)
                    if triple not in self.store.doc_graph.triples:
                        self.store.doc_graph.add_triple(*triple)

        return merged, rejected

    # -----------------------------------------------------------------
    # Step 5: Answer from enriched GMS or synthesize
    # -----------------------------------------------------------------

    def _answer_from_enriched_gms(self, question: str,
                                    sources: list[WebSource]) -> tuple[str, float, str]:
        """Try GMS answer first, then synthesize from verified sources.

        Returns (answer, confidence, source_type).
        """
        from knowlytix.knowledge.query import classify_question, QueryEngine

        # First try GMS answer from enriched store
        engine = QueryEngine(self.store, self.llm, self.config)
        qtype = classify_question(question)
        gms_answer, gms_conf, _, _ = engine._gms_answer(question, qtype)

        if gms_answer is not None and gms_conf > 0.8:
            return str(gms_answer), gms_conf, "gms"

        # GMS can't answer directly — synthesize from verified source triples
        # Gather all verified triples as context
        verified_facts = []
        for src in sources:
            if src.is_consistent and src.scored:
                for s in src.scored:
                    t = s.triple
                    fact = f"{t.head} | {t.relation} | {t.tail}"
                    if t.numeric_value is not None:
                        fact += f" | {t.numeric_value}"
                    verified_facts.append(fact)

        if not verified_facts:
            # No verified facts — return empty
            return "", 0.0, "no_sources"

        # Build GMS context (existing knowledge)
        gms_context = engine._gather_gms_context(question)

        # Synthesize answer using LLM grounded in verified facts
        source_list = "\n".join(
            f"- {src.url}: {src.title}"
            for src in sources if src.is_consistent
        )
        verified_list = "\n".join(f"VERIFIED FACT: {f}" for f in verified_facts[:50])

        system = (
            "You are an expert analyst. Answer the question using ONLY the "
            "verified facts provided below. These facts have been extracted "
            "from web sources and verified against a geometric knowledge store "
            "for consistency. Do not add information beyond what is in the "
            "verified facts. Cite the source when possible."
        )
        user = (
            f"<verified_facts>\n{verified_list}\n</verified_facts>\n\n"
            f"<sources>\n{source_list}\n</sources>\n\n"
        )
        if gms_context:
            user += f"<existing_knowledge>\n{gms_context}\n</existing_knowledge>\n\n"
        user += f"Question: {question}"

        answer = self.llm.call(system, user, max_tokens=2048)
        return answer, 0.7, "synthesis"


# =============================================================================
# Convenience function for CLI
# =============================================================================

def web_research(store: GMSExpertStore, question: str, llm: LLMBackend,
                 config: DocGMSConfig, device: torch.device,
                 max_sources: int = 5) -> WebSearchResult:
    """Run a web research cycle and return the result."""
    agent = WebAgent(store, llm, config, device, max_sources=max_sources)
    return agent.research(question)
