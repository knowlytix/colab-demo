# SPDX-License-Identifier: Apache-2.0
"""DocGMS configuration.

Two coexisting APIs:

- **Nested dataclasses** (``DocGMSConfig`` + ``ConvertConfig``,
  ``VerifyConfig``, ``LearnConfig``) — legacy API used by the CLI,
  ingest, query, and MCP-server entry points.
- **Flat ``DocGMSSettings(BaseSettings)``** — env-var surface
  (``DOCGMS_*``) for wheel users to override limits, gates, and store
  paths without editing source. Mirrors a user-facing subset of the
  dataclass fields.

See ``distribution/config_prereq_plan.md`` §Domain 3 for design.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

from pydantic_settings import BaseSettings

from knowlytix.core._caps import Unlimited, apply_ceiling, get_numeric_cap
from knowlytix.core._settings_base import settings_config
from knowlytix.core.config import (
    CapLossConfig,
    EmbeddingConfig,
    GeometryConfig,
    LossConfig,
    MemoryConfig,
    TrainConfig,
)


@dataclass
class ConvertConfig:
    """Document-to-markdown conversion settings.

    ``llm_model`` default is ``None``: wheel users configure the model
    through ``GMS_LLM_MODEL`` (via :class:`gms.llm.LLMSettings`) rather
    than this dataclass. CLI/MCP entry points still assign to this
    field directly for per-invocation overrides. The backend factory
    (:func:`knowlytix.doc.llm_backend.create_backend`) raises an actionable
    error if the field is left unset at the point of use.
    """

    llm_model: str | None = None
    llm_backend: str = "anthropic"          # "anthropic" or "local"
    local_model_name: str = ""
    max_pages: int = 200
    chunk_size: int = 4000                  # chars per LLM chunk for table extraction
    pdf_backend: str = "auto"               # "auto", "opendataloader", or "pymupdf"


@dataclass
class VerifyConfig:
    """GMS verification thresholds.

    Every threshold defaults to ``None``. Thresholds must come from
    calibration (``GMSJudge.calibrate()``); verifiers will refuse to
    operate on fields left ``None``. No hardcoded numeric defaults --
    a bank-grade deployment treats any constant tucked into library
    code as a defect, since thresholds are properties of the store's
    training, not of the library.
    """
    tau_contra: float | None = None         # tension energy contradiction threshold
    tau_ent: float | None = None            # entailment threshold
    tau_path: float | None = None           # holonomy consistency threshold
    holonomy_alpha: float = 0.1             # decay for effective holonomy (not a threshold)
    min_plausibility: float | None = None   # max geodesic dist for plausible triple


@dataclass
class LearnConfig:
    """Runtime memory growth settings."""
    n_steps: int = 50                       # Riemannian GD steps for new entities
    lr_write: float = 5e-3
    freeze_existing: bool = True
    contradiction_gate: bool = True         # reject writes that introduce contradictions
    auto_learn: bool = True                 # auto-write verified novel facts from Q&A
    # Encoders for warm-starting newly-added entities. None -> package defaults
    # (MiniLM v / nli-mpnet u). Set to mirror the build's EmbeddingConfig so
    # incremental adds use the same encoders as the initial build.
    embedding: EmbeddingConfig | None = None


@dataclass
class DocGMSConfig:
    """Top-level configuration for DocGMS.

    The ``ingest_mode`` field selects how the markdown parser populates
    the :class:`~knowlytix.benchmark.graph.DocumentGraph`:

    * ``"hybrid"`` (default) — regex backbone + LLM augmentation at
      column classification, entity-column detection, and prose relation
      extraction. Strict superset of ``"regex"`` on numeric values
      (FinStructBench paper §6.5: 100% value match, ≈ 22% more ENM
      entries). Requires ``llm`` to be passed to
      :func:`~knowlytix.doc.ingest.ingest_document`; otherwise the call
      raises ``ValueError``.
    * ``"regex"`` — pure deterministic parser. No LLM calls. Set this
      when running offline / in tests / for byte-stable reproduction.
    * ``"llm_only"`` — LLM extracts everything; regex disabled. Recovers
      ≈ 12.5% of regex ENM entries with near-zero overlap. Useful for
      paper reproduction and as a debugging baseline; not for production.

    Bank-grade callers should set ``ingest_mode`` explicitly in code so
    a reviewer can tell which path produced any given store.
    """
    convert: ConvertConfig = field(default_factory=ConvertConfig)
    geometry: GeometryConfig = field(
        default_factory=lambda: GeometryConfig(d_v=256, d_u=256, m=128, d=128)
    )
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    loss: LossConfig = field(
        default_factory=lambda: LossConfig(
            lambda_geo=5.0, lambda_tension=0.0, lambda_path=0.0,
            lambda_logic=0.0, lambda_recon=0.0, lambda_neg=0.5, gamma=1.0,
        )
    )
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            lr=5e-3, lr_riemannian=2e-3, batch_size=256, epochs=200,
            neg_samples=32,
        )
    )
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    learn: LearnConfig = field(default_factory=LearnConfig)
    cap: CapLossConfig = field(default_factory=CapLossConfig)
    store_path: str = "docgms_store"
    ingest_mode: str = "hybrid"
    # "cap" trains relation-conditioned spherical-cap admissibility (paper §7);
    # "point" is the classical margin loss.
    loss_mode: Literal["point", "cap"] = "point"


class DocGMSSettings(BaseSettings):
    """Flat env-var-overrideable settings for the ``knowlytix.knowledge`` package.

    Env prefix: ``DOCGMS_``. Mirrors the user-facing subset of the legacy
    ``ConvertConfig`` / ``LearnConfig`` / ``DocGMSConfig`` dataclasses so
    wheel consumers can tune document-ingestion limits, safety gates,
    and store paths through environment variables only.

    The LLM model used for conversion is read from ``GMS_LLM_MODEL``
    (see :class:`gms.llm.LLMSettings`); it is deliberately not surfaced
    here to avoid two places defining the same setting.

    Example::

        export DOCGMS_MAX_PAGES=500
        export DOCGMS_CONTRADICTION_GATE=false
        export DOCGMS_STORE_PATH=/data/docgms

        from knowlytix.knowledge.config import DocGMSSettings
        s = DocGMSSettings()
    """

    model_config = settings_config(env_prefix="DOCGMS_")

    # From ConvertConfig
    max_pages: int = 200
    chunk_size: int = 4000

    # From LearnConfig
    n_steps: int = 50
    freeze_existing: bool = True
    contradiction_gate: bool = True
    auto_learn: bool = True

    # From DocGMSConfig
    store_path: Path = Path("./docgms_store")

    @classmethod
    def from_license(
        cls,
        caps: "Mapping[str, int | bool | Unlimited] | None" = None,
    ) -> "DocGMSSettings":
        """Return a :class:`DocGMSSettings` with license ceilings applied.

        **Precondition**: ``caps`` must be a validated caps dict (either
        passed explicitly or fetched via :func:`gms._caps.get_caps` when
        ``caps`` is ``None``).  A missing required claim raises ``KeyError``
        (fail-closed, Invariant 7).

        The ceiling rule itself lives in compiled ``knowlytix-core``: this method
        is the open-source hook that pulls caps from the compiled side via
        ``get_caps()``.  Clamps ``max_pages`` downward against
        ``caps["max_document_pages"]`` per Invariant 1.  An ``"unlimited"``
        cap is a no-op (Invariant 3).
        """
        if caps is None:
            # `get_caps()` returns an immutable MappingProxyType.  We never
            # mutate `caps` below (all reads go through `caps[key]`), so
            # skipping the dict copy saves an 8-key allocation on every
            # call — the Mapping protocol is all `get_numeric_cap` needs.
            from knowlytix.core._caps import get_caps

            caps = get_caps()

        inst = cls()
        clamped = apply_ceiling(
            "max_pages",
            inst.max_pages,
            get_numeric_cap(caps, "max_document_pages"),
        )
        if clamped == inst.max_pages:
            # Fast path: no clamp needed (UNLIMITED cap or env already
            # below cap) — return the env-loaded instance unchanged so
            # Invariant 3 (byte-identical to unlicensed) holds without
            # an extra model_copy allocation.
            return inst
        return inst.model_copy(update={"max_pages": clamped})

    @classmethod
    def from_legacy(cls, cfg: "DocGMSConfig") -> "DocGMSSettings":
        """Project a legacy :class:`DocGMSConfig` into flat settings.

        Useful for callers that have a ``DocGMSConfig`` in hand (e.g.
        the CLI) and want the flat view for handing off to PyPI-style
        consumers.
        """

        return cls(
            max_pages=cfg.convert.max_pages,
            chunk_size=cfg.convert.chunk_size,
            n_steps=cfg.learn.n_steps,
            freeze_existing=cfg.learn.freeze_existing,
            contradiction_gate=cfg.learn.contradiction_gate,
            auto_learn=cfg.learn.auto_learn,
            store_path=Path(cfg.store_path),
        )
