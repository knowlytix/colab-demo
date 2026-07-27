# SPDX-License-Identifier: Apache-2.0
"""DocGMS: Geometric Expert System with LLM-Augmented Learning.

Combines GMS (Geometric Memory Systems) as the structured knowledge back-end
with LLM as the reasoning/language front-end. Verified LLM outputs flow back
into GMS to grow the expert system over time.
"""

from knowlytix.knowledge.config import (
    ConvertConfig,
    DocGMSConfig,
    DocGMSSettings,
    LearnConfig,
    VerifyConfig,
)
from knowlytix.knowledge.ingest import IngestResult, ingest_document
from knowlytix.knowledge.query import QueryEngine, QueryResult
from knowlytix.knowledge.store import GMSExpertStore

# --- Licensed settings (module-level singleton) ------------------------------
# Applied once at first import; pulls caps via gms._caps.get_caps() which
# fails closed if gms.verify_license() has not yet populated the caps.
# Consumers should import `knowlytix.doc.settings` rather than constructing
# DocGMSSettings() directly so ceilings always bite.
settings: DocGMSSettings = DocGMSSettings.from_license()

__all__ = [
    "ConvertConfig",
    "DocGMSConfig",
    "DocGMSSettings",
    "GMSExpertStore",
    "IngestResult",
    "LearnConfig",
    "QueryEngine",
    "QueryResult",
    "VerifyConfig",
    "ingest_document",
    "settings",
]
from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("knowlytix-knowledge")
except _PkgNotFound:  # source tree (dev install is the `gms` distribution)
    __version__ = "0.0.0.dev0"
