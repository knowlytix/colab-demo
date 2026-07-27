# SPDX-License-Identifier: Apache-2.0
"""FinStructBench configuration.

Exposes scorer tolerance knobs via environment variables with the
``FINSTRUCTBENCH_`` prefix. Values are wheel-user-overridable without
touching source.

Example::

    export FINSTRUCTBENCH_FLOAT_TOL=1e-4
    export FINSTRUCTBENCH_CLOSE_THRESHOLD=0.02

    from knowlytix.benchmark.config import FinStructBenchSettings
    s = FinStructBenchSettings()

See ``distribution/config_prereq_plan.md`` §Domain 4 for design.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings

from knowlytix.core._settings_base import settings_config


class FinStructBenchSettings(BaseSettings):
    """Flat env-var-overrideable settings for the ``knowlytix.benchmark`` package.

    Env prefix: ``FINSTRUCTBENCH_``. Currently exposes the three scorer
    tolerances used by :func:`knowlytix.benchmark.scorers.score_answer` and
    related helpers. Additional knobs can be added here as the scorer
    layer evolves; upstream integer-range guards like the
    ``1 <= n <= 99999`` filter remain internal constants.
    """

    model_config = settings_config(env_prefix="FINSTRUCTBENCH_")

    # Absolute float-equality tolerance (default mirrors
    # scorers.score_answer's ``float_tol`` kwarg default).
    float_tol: float = 1e-6

    # "Close" threshold used when the exact float match fails; default
    # mirrors the 1% hardcode in scorers.score_answer.
    close_threshold: float = 0.01

    # Per-element tolerance when comparing tuples; mirrors the 1e-3
    # hardcode in the tuple-scoring path.
    tuple_element_tol: float = 1e-3
