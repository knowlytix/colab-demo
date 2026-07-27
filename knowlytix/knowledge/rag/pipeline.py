# SPDX-License-Identifier: Apache-2.0
"""RagPipeline — the triple-mediated graph-RAG orchestrator.

Online flow (``GEODE_RAG_DESIGN.md`` §5): extract query triples -> bind to graph
vocab -> bind-check -> GMS retrieve (+ provenance) -> synthesize -> decide
(accept / abstain / escalate). Dense fallback and answer self-verification are
opt-in extensions wired in later phases; the default is the verifiable triple
route only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowlytix.knowledge.geode.provenance import ProvenanceLedger
from knowlytix.knowledge.rag.assemble import Assembler
from knowlytix.knowledge.rag.binding import BoundTriple, TripleBinder
from knowlytix.knowledge.rag.config import RagConfig
from knowlytix.knowledge.rag.extraction import Extraction
from knowlytix.knowledge.rag.dense import (
    DenseSpan,
    InMemoryVectorBackend,
    build_spans,
)
from knowlytix.knowledge.rag.query_triples import (
    QueryTriple,
    QueryTripleExtractor,
    is_var,
    schema_from_store,
)
from knowlytix.knowledge.rag.retrieve import RetrievedFact, Retriever
from knowlytix.knowledge.rag.verify import AnswerVerifier

__all__ = ["RagAnswer", "RagPipeline"]


@dataclass
class RagAnswer:
    answer: str
    confidence: float
    decision: str                       # "accept" | "abstain" | "escalate"
    route: str                          # "triple" | "dense_fallback"
    verified: bool
    query_triples: list[QueryTriple] = field(default_factory=list)
    bound_triples: list[BoundTriple] = field(default_factory=list)
    sources: list[RetrievedFact] = field(default_factory=list)
    dense_sources: list[DenseSpan] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    notice: str | None = None

    def audit(self) -> dict:
        """Structured audit record of this answer (route, evidence, verdicts)."""
        return {
            "decision": self.decision,
            "route": self.route,
            "verified": self.verified,
            "confidence": self.confidence,
            "query_triples": [t.as_tuple() for t in self.query_triples],
            "bound_triples": [
                (b.head, b.relation, b.tail, b.bound) for b in self.bound_triples
            ],
            "sources": [
                {"triple": (f.head, f.relation, f.tail), "source": f.source,
                 "score": f.score, "location": f.location}
                for f in self.sources
            ],
            "verification": self.verification,
            "dense_sources": [s.location for s in self.dense_sources],
            "notice": self.notice,
        }


_ABSTAIN_MSG = "I cannot answer this from the available grounded evidence."


class RagPipeline:
    """Triple-mediated RAG over a :class:`GMSExpertStore`."""

    def __init__(self, store, rag: RagConfig, *, ledger: ProvenanceLedger | None = None):
        self.store = store
        self.rag = rag
        self.ledger = ledger or self._default_ledger(store)
        self._vocab = schema_from_store(store) if rag.ground_extraction else None
        if rag.query_parse_mode == "geometric":
            from knowlytix.knowledge.rag.query_triples import GeometricQueryParser
            self.extractor = GeometricQueryParser(store, encoder=rag.encoder,
                                                  max_hops=rag.max_hops)
        else:
            self.extractor = QueryTripleExtractor(rag.extract_llm(), vocab=self._vocab)
        self.binder = TripleBinder(store, mode=rag.binding, encoder=rag.encoder,
                                   bind_threshold=rag.bind_threshold,
                                   bind_margin=rag.bind_margin)
        self.retriever = Retriever(store, self.ledger, top_k=rag.top_k)
        self.assembler = Assembler(rag.llm)
        self.verifier = (AnswerVerifier(store, rag.verify_llm(), binder=self.binder)
                         if rag.verify_llm_output else None)
        self.relevance = None
        if rag.relevance_gate:
            if rag.relevance_mode == "geometric":
                from knowlytix.knowledge.rag.relevance import GeometricRelevanceGate
                self.relevance = GeometricRelevanceGate(
                    store, v_encoder=rag.encoder,
                    u_encoder=rag.relevance_u_encoder,
                    tau_accept=rag.relevance_tau_accept,
                    tau_contra=rag.relevance_tau_contra,
                    tau_contra_per_relation=rag.relevance_tau_contra_per_relation)
            else:
                from knowlytix.knowledge.rag.relevance import RelevanceGate
                self.relevance = RelevanceGate(
                    store, rag.extract_llm(), encoder=rag.encoder,
                    prefilter=rag.relevance_prefilter,
                    prefilter_floor=rag.relevance_prefilter_floor)
        self.dense = None
        if rag.dense_fallback:
            backend = rag.vector_backend or InMemoryVectorBackend(rag.encoder)
            backend.index(build_spans(store.markdown,
                                      source_path=store.store_path))
            self.dense = backend

        # Numeric-order route: cache the rankable ENM types (>= 2 scalar
        # entries) and, when an encoder is present, their phrase embeddings so a
        # superlative question can be matched to a type by cosine.
        self._enm_types: list[str] = []
        self._enm_type_emb = None
        if rag.numeric_order and store.enm is not None:
            from collections import Counter
            counts = Counter(k.type for k in store.enm.keys())
            self._enm_types = sorted(t for t, n in counts.items() if n >= 2)
            if self._enm_types and rag.encoder is not None:
                import torch
                emb = torch.as_tensor(
                    rag.encoder([t.replace("_", " ") for t in self._enm_types]),
                    dtype=torch.float32)
                self._enm_type_emb = torch.nn.functional.normalize(
                    emb, p=2, dim=-1)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_store(cls, store, rag: RagConfig) -> "RagPipeline":
        return cls(store, rag)

    @classmethod
    def build(cls, md_path: str, config, rag: RagConfig, *, device=None,
              **build_kwargs) -> "RagPipeline":
        """Run GEODE self-correction + train a store, then wrap it.

        ``build_kwargs`` are forwarded to
        :func:`knowlytix.knowledge.geode.rag.build_rag_store` (e.g. ``llm`` for
        the Qwen actor, ``max_iters``).
        """
        from knowlytix.knowledge.geode.rag import build_rag_store

        result = build_rag_store(md_path, config, device=device, **build_kwargs)
        return cls.from_store(result.store, rag)

    @staticmethod
    def _default_ledger(store) -> ProvenanceLedger | None:
        if getattr(store, "markdown", ""):
            return ProvenanceLedger.from_text(store.markdown,
                                              source_path=store.store_path)
        return None

    # -- extract (reusable parse⇄bind) --------------------------------------

    def extract(self, question: str) -> Extraction:
        """Parse ``question`` into graph-bound query triples — the reusable
        parse⇄bind half of :meth:`query`, with no retrieval or generation.

        Returns an :class:`Extraction` (the query triples plus their binding to
        the store's vocabulary). Use :attr:`Extraction.is_bound` to tell a
        grounded message from one that binds to nothing, and :attr:`Extraction.
        bound_facts` for the grounded ``(head, relation, tail)`` triples. Hand
        the result back into :meth:`query` to answer it without re-parsing.
        """
        qts, bound = self._extract_and_bind(question)
        return Extraction(query_triples=qts, bound_triples=bound)

    # -- query --------------------------------------------------------------

    def query(self, question: str, *, generate: bool = True,
              extraction: Extraction | None = None) -> RagAnswer:
        # ``generate=False`` is a retrieve-only path: run extract -> bind ->
        # relevance -> retrieve -> cap/accept gate, but SKIP the LLM assembler and
        # self-verifier. It returns the gated ``sources`` (the grounded graph
        # facts) with no prose answer -- for callers that route a query to its
        # governing graph entities (claim-decomposition extraction) rather than
        # needing a generated sentence. Same gates, no generation cost.
        # Numeric-order route first: a superlative/comparison question names no
        # anchor entity, so the triple route below cannot bind it. Resolve it by
        # ranking the relevant ENM type through u-space directed entailment.
        # A supplied ``extraction`` is reused verbatim: skip the numeric-order
        # route and the parse⇄bind loop so the message is parsed exactly once
        # across an extract()-then-query() flow. (Passing an extraction commits
        # to the triple route; a superlative resolves via numeric_order only on
        # the pipeline's own parse pass.)
        if extraction is None and self._enm_types:
            numeric = self._try_numeric_order(question)
            if numeric is not None:
                return self._finalize(numeric)

        if extraction is None:
            extraction = self.extract(question)
        qts, bound = extraction.query_triples, extraction.bound_triples

        # Bind-check: if nothing resolved to the graph, we have no grounded
        # path. Try the (distrusted, opt-in) dense fallback, else abstain.
        if not qts or not any(b.bound for b in bound):
            return self._finalize(
                self._fallback_or_abstain(question, qts, bound, "no_binding"))

        # Relevance gate: a bound relation can be graph-valid yet not the attribute
        # the question asked about (interest rate -> has_fee_amount). An LLM checks
        # the asked attribute against the head's real relations; "none" abstains.
        if self.relevance is not None:
            for b in bound:
                if (not b.bound or b.head is None or b.relation is None
                        or is_var(b.original.head)):
                    continue
                matched, _ = self.relevance.relevant_relation(question, b.head)
                if matched is None:
                    return self._finalize(self._abstain(qts, bound, reason="irrelevant"))

        retrieved = self.retriever.retrieve(bound)
        if not retrieved.matched:
            return self._finalize(
                self._fallback_or_abstain(question, qts, bound, "no_match"))

        if not generate:
            confidence = self._confidence(retrieved.facts)
            decision = ("accept" if confidence >= self.rag.accept_threshold
                        else "abstain")
            return self._finalize(RagAnswer(
                answer="", confidence=confidence, decision=decision,
                route="triple", verified=False, query_triples=qts,
                bound_triples=bound, sources=retrieved.facts,
                verification={}, notice=None))

        answer = self.assembler.assemble(question, retrieved.facts)
        confidence = self._confidence(retrieved.facts)

        verification: dict = {}
        notice: str | None = None
        if self.verifier is not None:
            answer, confidence, notice, verification = self._self_verify(
                question, answer, retrieved.facts, confidence)
            if verification.get("ok") is False and self.rag.on_verify_fail == "abstain":
                return self._finalize(self._abstain(
                    qts, bound, reason="verify_fail", verification=verification))

        decision = ("accept" if confidence >= self.rag.accept_threshold
                    else "abstain")
        return self._finalize(RagAnswer(
            answer=answer if decision == "accept" else _ABSTAIN_MSG,
            confidence=confidence,
            decision=decision,
            route="triple",
            verified=True,
            query_triples=qts,
            bound_triples=bound,
            sources=retrieved.facts,
            verification=verification,
            notice=notice,
        ))

    # -- numeric-order route ------------------------------------------------

    _HIGHEST_CUES = ("highest", "largest", "greatest", "maximum", "max ", "most ",
                     "biggest", "steepest", "top ", "dearest", "priciest")
    _LOWEST_CUES = ("lowest", "smallest", "least", "minimum", "min ", "fewest",
                    "cheapest", "lowest-cost", "bottom ")

    def _which_extreme(self, q: str) -> str | None:
        ql = f" {q.lower()} "
        hi = any(c in ql for c in self._HIGHEST_CUES)
        lo = any(c in ql for c in self._LOWEST_CUES)
        if hi == lo:            # neither, or ambiguous both -> not a superlative
            return None
        return "highest" if hi else "lowest"

    def _match_enm_type(self, question: str) -> str | None:
        """Pick the ENM type the question is about (encoder cosine, or substring
        fallback when no encoder is configured)."""
        if self._enm_type_emb is not None:
            import torch
            qv = torch.nn.functional.normalize(
                torch.as_tensor(self.rag.encoder([question]), dtype=torch.float32),
                p=2, dim=-1)[0]
            sims = self._enm_type_emb @ qv
            best = int(sims.argmax())
            if float(sims[best]) >= self.rag.numeric_order_floor:
                return self._enm_types[best]
            return None
        ql = question.lower()
        hits = [t for t in self._enm_types
                if t.replace("_", " ") in ql or t in ql]
        return hits[0] if len(hits) == 1 else None

    def _followon_relation(self, question: str, base_entity: str):
        """The asserted (head, rel, tail) triples of ``base_entity`` the question
        asks to list. If a relation matches the question, restrict to it; else
        return all of the entity's asserted triples (generic "its relationships").
        """
        triples = [(h, r, t) for h, r, t in self.store.query_triples(head=base_entity)
                   if r not in ("in_section", "has_alias")]
        if not triples or self.rag.encoder is None:
            return triples
        import torch
        rels = sorted({r for _, r, _ in triples})
        def phrase(r):
            return r[4:].replace("_", " ") if r.startswith("has_") else r.replace("_", " ")
        emb = torch.nn.functional.normalize(
            torch.as_tensor(self.rag.encoder([phrase(r) for r in rels]),
                            dtype=torch.float32), p=2, dim=-1)
        qv = torch.nn.functional.normalize(
            torch.as_tensor(self.rag.encoder([question]), dtype=torch.float32),
            p=2, dim=-1)[0]
        sims = emb @ qv
        best = int(sims.argmax())
        if float(sims[best]) >= self.rag.numeric_order_floor:
            return [(h, r, t) for h, r, t in triples if r == rels[best]]
        return triples

    def _try_numeric_order(self, question: str) -> RagAnswer | None:
        """Answer a superlative question by ranking an ENM type in u-space.

        Returns ``None`` (fall through to the triple route) unless the question
        is a superlative whose target ENM type is confidently identified.
        """
        import math
        which = self._which_extreme(question)
        if which is None:
            return None
        enm_type = self._match_enm_type(question)
        if enm_type is None:
            return None
        res = self.store.enm_extreme(enm_type, which)
        if res is None:
            return None

        # The values are exact, so the winner is ambiguous only when there is no
        # distinct runner-up (all entries equal -> margin 0). A near-but-distinct
        # runner-up is NOT a tie; the exact values already separate them.
        if res.margin <= self.rag.numeric_order_tie_margin:
            return RagAnswer(
                answer=_ABSTAIN_MSG, confidence=0.0, decision="abstain",
                route="numeric_order", verified=True,
                notice=(f"No distinct {which} '{enm_type}' value "
                        "(entries are equal); cannot single one out."))

        base = res.entity.split("/")[0]
        related = self._followon_relation(question, base)
        sources = [RetrievedFact(head=base, relation=enm_type,
                                 tail=f"{res.value:g}", score=0.0,
                                 confidence=1.0, source="enm")]
        sources += [RetrievedFact(head=h, relation=r, tail=t, score=0.0,
                                  confidence=1.0, source="triple")
                    for h, r, t in related]

        answer = (f"The {which} '{enm_type}' value is {base} "
                  f"at {res.value:g}.")
        if related:
            rels = sorted({r for _, r, _ in related})
            tails = ", ".join(sorted({t for _, _, t in related}))
            rel_label = "/".join(rels)
            answer += f" Its {rel_label} relationships: {tails}."

        confidence = 1.0 - math.exp(-res.margin)
        return RagAnswer(
            answer=answer, confidence=confidence, decision="accept",
            route="numeric_order", verified=True, sources=sources,
            notice=None)

    def _extract_and_bind(self, question: str):
        """Extract query triples and bind them; on a bind failure re-ask the
        extractor once (with the graph vocab + the terms that didn't match).

        This repair loop is the main extraction-reliability lever for small
        models: the first pass often guesses a relation/entity that doesn't
        exist; the retry, grounded in the actual schema, usually fixes it.
        """
        qts = self.extractor.extract(question)
        bound = [self.binder.bind(qt) for qt in qts]
        attempts = 0
        while (qts and not all(b.bound for b in bound)
               and attempts < self.rag.max_extract_retries):
            attempts += 1
            qts = self.extractor.extract(question, hint=self._repair_hint(bound))
            bound = [self.binder.bind(qt) for qt in qts]
        return qts, bound

    def _repair_hint(self, bound: list[BoundTriple]) -> str:
        failed: list[str] = []
        for b in bound:
            o = b.original
            if not is_var(o.head) and b.head is None:
                failed.append(o.head)
            if not is_var(o.relation) and b.relation is None:
                failed.append(o.relation)
            if not is_var(o.tail) and b.tail is None:
                failed.append(o.tail)
        rels = ", ".join((self._vocab or {}).get("relations", [])[:60])
        return (f"These terms did not match the knowledge graph: {failed}. "
                f"Re-express the query using ONLY these relations: {rels}.")

    def _fallback_or_abstain(self, question, qts, bound, reason: str) -> RagAnswer:
        """Dense fallback when enabled and not strict; otherwise abstain.

        Dense answers are quarantined: ``verified=False``, ``route=
        "dense_fallback"``, and a notice that they are not GMS-verified.
        """
        if self.dense is None or self.rag.strict_mode:
            return self._abstain(qts, bound, reason=reason)
        hits = self.dense.search(question, top_k=self.rag.top_k)
        if not hits:
            return self._abstain(qts, bound, reason=reason)
        spans = [s for s, _ in hits]
        answer = self.assembler.assemble_passages(question, [s.text for s in spans])
        return RagAnswer(
            answer=answer, confidence=float(hits[0][1]), decision="accept",
            route="dense_fallback", verified=False, query_triples=qts,
            bound_triples=bound, dense_sources=spans,
            notice="Answer from dense fallback; NOT GMS-verified.",
        )

    def _finalize(self, ans: RagAnswer) -> RagAnswer:
        """Emit the audit record (if a sink is configured) and return."""
        if self.rag.audit_sink is not None:
            self.rag.audit_sink(ans.audit())
        return ans

    def coverage(self):
        """Coverage report for this store — measurable triple-mediation blind spots."""
        from knowlytix.knowledge.rag.coverage import coverage_report
        return coverage_report(self.store)

    def _self_verify(self, question: str, answer: str,
                     facts: list[RetrievedFact], confidence: float):
        """Run GMS self-verification; apply ``on_verify_fail``.

        Returns ``(answer, confidence, notice, verification_dict)``. The
        ``abstain`` action is handled by the caller (it changes control flow).
        """
        report = self.verifier.verify(answer)
        if report.ok:
            return answer, confidence, None, report.as_dict()

        action = self.rag.on_verify_fail
        if action == "regenerate":
            answer = self.assembler.assemble(question, facts)
            report = self.verifier.verify(answer)
            if report.ok:
                return answer, confidence, None, report.as_dict()
        # annotate (and regenerate-still-failing): surface the contradiction,
        # downgrade confidence so the accept threshold can catch it.
        failed = ", ".join(f"{t[0]}.{t[1]}={t[2]}" for t in
                           report.as_dict()["failed"])
        notice = (f"GMS self-verification flagged unsupported claims: {failed}. "
                  "Answer not fully grounded.")
        return answer, 0.0, notice, report.as_dict()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _confidence(facts: list[RetrievedFact]) -> float:
        if not facts:
            return 0.0
        return min(f.confidence for f in facts)

    def _abstain(self, qts, bound, *, reason: str,
                 verification: dict | None = None) -> RagAnswer:
        notice = {
            "no_binding": "Question did not bind to known entities/relations.",
            "no_match": "No grounded facts matched the query.",
            "verify_fail": "Answer failed GMS self-verification (unsupported claims).",
            "irrelevant": "The entity does not hold the attribute the question asked about.",
        }[reason]
        return RagAnswer(
            answer=_ABSTAIN_MSG, confidence=0.0, decision="abstain",
            route="triple", verified=True, query_triples=qts,
            bound_triples=bound, notice=notice,
            verification=verification or {},
        )
