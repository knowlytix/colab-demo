# SPDX-License-Identifier: Apache-2.0
"""Closed-vocabulary geometric label classifier with a calibrated abstain.

Same principle as :class:`GeometricQueryParser` (``query_triples.py``): a small
instruct model asked to *emit* a label from a fixed taxonomy is unreliable -- it
invents off-taxonomy strings and, worse, over-commits to a concrete label on an
underspecified input. This classifier instead scores the text against the real
closed vocabulary by cosine over an embedding space and picks the nearest label
-- and **abstains** (returns ``abstain_label``) when the top score falls below a
calibrated threshold, rather than fabricating a concrete answer it cannot
support.

The abstain threshold is never hardcoded: :meth:`calibrate` fits it on a labeled
cohort to a false-accept ceiling (how often we may commit to a concrete label
when the truth is the abstain class), the smallest threshold that honors the
ceiling so recall on concrete labels is preserved. Persist and reload the fitted
threshold the same way the RAG gates persist their operating points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass
class GeometricLabelClassifier:
    """Assign a text to one of a fixed set of labels by nearest exemplar.

    Parameters
    ----------
    exemplars:
        ``{label: [exemplar phrase, ...]}`` -- the closed vocabulary. A label's
        score for a text is the max cosine over its exemplars (one clearly
        matching phrase is enough to claim the label).
    encoder:
        ``list[str] -> Tensor`` of L2-normalized row embeddings. Defaults to the
        graph's ``encode_texts`` (MiniLM) when None -- the same encoder family
        the store retrieval uses; pass a document-tuned encoder to match the
        GMS embedding space exactly.
    threshold:
        Abstain cut. ``classify`` returns ``abstain_label`` when the top label's
        score is below this. Fit it with :meth:`calibrate`; the 0.0 default
        abstains for nothing (every text gets its argmax label) and is only a
        safe pre-calibration placeholder.
    """

    exemplars: dict[str, list[str]]
    encoder: Callable[[list[str]], "object"] | None = None
    threshold: float = 0.0
    _owners: list[str] = field(default_factory=list, init=False, repr=False)
    _emb: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.encoder is None:
            from knowlytix.core.graph.encoders import encode_texts
            self.encoder = encode_texts
        owners, phrases = [], []
        for label, exs in self.exemplars.items():
            for ex in exs:
                owners.append(label)
                phrases.append(ex)
        if not phrases:
            raise ValueError("GeometricLabelClassifier needs at least one exemplar")
        self._owners = owners
        self._emb = self._encode(phrases)  # (n_exemplars, dim), normalized

    def _encode(self, texts: list[str]):
        import torch
        emb = torch.as_tensor(self.encoder(list(texts)), dtype=torch.float32)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def scores(self, text: str) -> dict[str, float]:
        """Per-label score = max cosine of ``text`` to that label's exemplars."""
        qv = self._encode([text])[0]
        sims = (self._emb @ qv).tolist()
        best: dict[str, float] = {}
        for owner, s in zip(self._owners, sims):
            if s > best.get(owner, float("-inf")):
                best[owner] = s
        return best

    def classify(self, text: str, *, abstain_label: str) -> tuple[str, float]:
        """Return ``(label, score)``: the highest-scoring label, or
        ``(abstain_label, top_score)`` when that top score is below threshold."""
        s = self.scores(text)
        if not s:
            return abstain_label, 0.0
        label, top = max(s.items(), key=lambda kv: kv[1])
        if top < self.threshold:
            return abstain_label, top
        return label, top

    def calibrate(
        self,
        examples: Sequence[tuple[str, str]],
        *,
        abstain_label: str,
        false_accept_ceiling: float = 0.10,
    ) -> float:
        """Fit and store the abstain threshold from a labeled cohort.

        A *false accept* is committing to a CONCRETE (non-abstain) label when the
        example's true label is ``abstain_label`` -- the over-commitment failure
        this classifier exists to prevent. When ``abstain_label`` is itself a
        reachable class (has exemplars), an abstain-class example whose argmax is
        already ``abstain_label`` is correct at any threshold and is never a false
        accept; only examples that argmax to a concrete label can be. We pick the
        smallest threshold whose false-accept rate over the cohort's abstain-class
        examples is <= the ceiling, so concrete-label recall is given up only as
        far as the ceiling forces. With no abstain-class examples the threshold
        stays 0.0 (nothing to protect against; treat the gate as uncalibrated).

        Returns the fitted threshold (also assigned to ``self.threshold``).
        """
        # For each abstain-class example, the argmax (label, score) over ALL labels.
        risky = []  # top scores of abstain-class examples that argmax to a CONCRETE label
        n = 0
        for text, label in examples:
            if label != abstain_label:
                continue
            n += 1
            s = self.scores(text)
            if not s:
                continue
            top_label, top = max(s.items(), key=lambda kv: kv[1])
            if top_label != abstain_label:
                risky.append(top)
        if n == 0:
            self.threshold = 0.0
            return self.threshold
        # A cut abstains a risky example when top < cut. Lowest cut whose remaining
        # concrete-accepts / n <= ceiling.
        for cut in [0.0] + sorted(set(risky)):
            false_accepts = sum(1 for t in risky if t >= cut)
            if false_accepts / n <= false_accept_ceiling:
                self.threshold = float(cut)
                return self.threshold
        self.threshold = float(max(risky)) + 1e-6
        return self.threshold
