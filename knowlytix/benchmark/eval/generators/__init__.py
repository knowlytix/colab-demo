# SPDX-License-Identifier: Apache-2.0
from knowlytix.benchmark.eval.generators.base import QuestionGenerator, GeneratedQuestion
from knowlytix.benchmark.eval.generators.exact_recall import ExactRecallGenerator
from knowlytix.benchmark.eval.generators.threshold import ThresholdGenerator
from knowlytix.benchmark.eval.generators.cross_reference import CrossReferenceGenerator
from knowlytix.benchmark.eval.generators.contradiction import ContradictionGenerator
from knowlytix.benchmark.eval.generators.multi_hop import MultiHopGenerator
from knowlytix.benchmark.eval.generators.counting import CountingGenerator


def default_generators():
    """Return one instance of each generator."""
    return [
        ExactRecallGenerator(),
        ThresholdGenerator(),
        CrossReferenceGenerator(),
        ContradictionGenerator(),
        MultiHopGenerator(),
        CountingGenerator(),
    ]
