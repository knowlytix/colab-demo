# SPDX-License-Identifier: Apache-2.0
"""Test the GEODE geometry-supervised iterative encoder SFT loop.

Demonstrates the loop's purpose: fine-tuning the text encoder on a document's
own alias/name structure (extended by the GMS geometry) makes UNSEEN customer
paraphrases bind to the right canonical policy better than the frozen base
encoder. The held-out probes are natural-language phrases that never appear in
supervision, so improvement on them measures generalization, not memorization.

Run directly for a verbose report::

    python -m knowlytix.knowledge.geode.tests.test_embed_loop
"""

from __future__ import annotations

from knowlytix.core.graph.encoders import encode_texts
from knowlytix.embedding import EmbeddingSFTConfig
from knowlytix.knowledge.geode.embed_loop import (
    EmbedLoopConfig, GeodeEmbedLoop, graph_entity_labels)
from knowlytix.knowledge.geode.loop import make_default_trainer

_DOC = "data/bank_policies.md"

# Held-out probes: natural customer phrasings, NONE of which appear in the
# document's alias table or names. (surface, expected canonical policy id).
_PROBES = [
    ("my balance went negative and you charged me", "overdraft"),
    ("there is a charge on my card I never authorized", "disputes"),
    ("please refund this charge as a courtesy", "fee_reversal"),
    ("I would like to shut down my checking account", "account_closure"),
    ("you exposed my social security number", "pii_handling"),
    ("this fee is unfair and abusive, escalate to compliance", "regulatory_escalation"),
]

# An unlabelled pool the self-training step may pseudo-label and fold in.
_POOL = [
    "bounced check charge", "dispute a transaction", "waive the overdraft fee",
    "leaked personal data", "regulation E claim", "goodwill credit request",
]

# Held-out ALIASES: one domain term per policy, dropped from supervision AND
# geometry, then used as the eval set. Matched-distribution generalization
# (short domain jargon -> policy), which is where a general encoder has headroom.
_HELDOUT_ALIASES = [
    ("insufficient funds", "overdraft"),
    ("transaction dispute", "disputes"),
    ("fee refund", "fee_reversal"),
    ("close savings", "account_closure"),
    ("account number leak", "pii_handling"),
    ("regulation z", "regulatory_escalation"),
]


def _frozen_baseline_acc(labels, probes, encoder):
    """Nearest-prototype accuracy with the FROZEN base encoder (no SFT):
    prototypes are the mean of each policy's supervised surface embeddings."""
    import torch
    import torch.nn.functional as F

    names = sorted(labels)
    protos = []
    for c in names:
        z = F.normalize(encode_texts(sorted(labels[c]), encoder), dim=-1)
        protos.append(F.normalize(z.mean(0), dim=-1))
    P = torch.stack(protos)
    zq = F.normalize(encode_texts([s for s, _ in probes], encoder), dim=-1)
    pred = (zq @ P.T).argmax(-1).tolist()
    return sum(names[i] == c for i, (_, c) in zip(pred, probes)) / len(probes)


def run_demo():
    encoder = "sentence-transformers/all-MiniLM-L6-v2"
    sft = EmbeddingSFTConfig(rank=8, mode="full", encoder=encoder, epochs=200,
                             drift_weight=0.5, val_split=0.2, seed=42, device="cpu")
    cfg = EmbedLoopConfig(sft=sft, max_iters=4, use_geometry=True,
                          geometry_margin=0.12, pseudo_label=True,
                          pseudo_label_floor=0.55, ingest_mode="regex")
    trainer = make_default_trainer(device="cpu", epochs=150)
    loop = GeodeEmbedLoop(trainer, cfg)

    exclude = {s for s, _ in _HELDOUT_ALIASES}

    # Frozen baseline: prototypes from the SAME (held-out-excluded) supervision
    # the loop starts from, so the only difference is the SFT.
    from knowlytix.benchmark.ingest import ingest_markdown
    triples = [(h, r, t) for h, r, t in ingest_markdown(_DOC, mode="regex").triples
               if t.lower() not in {e.lower() for e in exclude}]
    seed_labels = graph_entity_labels(triples)
    frozen_alias = _frozen_baseline_acc(seed_labels, _HELDOUT_ALIASES, encoder)
    frozen_sent = _frozen_baseline_acc(seed_labels, _PROBES, encoder)

    res = loop.run(_DOC, pool=list(_POOL), eval_pairs=_HELDOUT_ALIASES,
                   exclude_surfaces=exclude)
    tuned_alias = res.history[-1]["heldout_acc"]
    preds_sent, _ = res.ft.classify([s for s, _ in _PROBES])
    tuned_sent = sum(p == c for p, (_, c) in zip(preds_sent, _PROBES)) / len(_PROBES)

    print(f"\ncanonical policies: {res.canonicals}")
    print("\n== PRIMARY: held-out alias generalization (matched distribution) ==")
    print(f"  held-out aliases (dropped from supervision + geometry): {len(_HELDOUT_ALIASES)}")
    print(f"  FROZEN MiniLM  : {frozen_alias:.2f}")
    print(f"  DOC-TUNED      : {tuned_alias:.2f}")
    preds_a, _ = res.ft.classify([s for s, _ in _HELDOUT_ALIASES])
    for (s, gold), p in zip(_HELDOUT_ALIASES, preds_a):
        print(f"     [{'OK ' if p == gold else 'MISS'}] {p:24s} <- {s!r}")

    print("\n== SECONDARY: unseen full-sentence probes ==")
    print(f"  FROZEN MiniLM  : {frozen_sent:.2f}    DOC-TUNED : {tuned_sent:.2f}")

    print("\nper-iteration history:")
    for h in res.history:
        print(f"  iter {h['iter']}: classes={h['n_classes']} labels={h['n_labels']} "
              f"geo+{h['geometry_links_added']} pseudo+{h['pseudo_labeled']} "
              f"pool_left={h['pool_remaining']} heldout_acc={h['heldout_acc']:.2f}")
    print(f"\nconverged={res.converged} in {res.iterations} iters")
    return {"frozen_alias": frozen_alias, "tuned_alias": tuned_alias,
            "frozen_sent": frozen_sent, "tuned_sent": tuned_sent}, res


def test_loop_converges_and_does_not_regress_binding():
    m, res = run_demo()
    # The loop must converge (label set reaches a fixed point).
    assert res.converged, "loop did not converge"
    # Doc-tuning must not regress binding on either held-out set. (Small N, so we
    # assert no-regression + a floor rather than a brittle exact gain; the demo
    # prints the gain -- here, +0.17 on the hard full-sentence probes.)
    assert m["tuned_alias"] >= m["frozen_alias"], "regressed on held-out aliases"
    assert m["tuned_sent"] >= m["frozen_sent"], "regressed on sentence probes"
    assert m["tuned_alias"] >= 0.8 and m["tuned_sent"] >= 0.8, "binding too low"


if __name__ == "__main__":
    run_demo()
