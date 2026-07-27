"""KAL default storage — reference ``KGStore`` impl + ORM in the ``kal`` schema.

This package ships in ``knowlytix-kal`` from v0.3.0. The mirror script
(see ``scripts/MIRROR_MANIFEST.toml``) copies this directory verbatim
into ``knowlytix/kal/storage_default/`` in the standalone package, with
the usual import flips applied (``knowly.shared.database.Base`` →
``knowlytix.kal._db.KalBase``, etc.).

``KnowlyKGStore`` (the knowly-specific impl, wraps ``kg.*`` tables) lives
at ``knowly.modules.kg_kal_store`` and is excluded from the mirror — see
``[exclude]`` in the manifest. ``KalDefaultKGStore`` is the greenfield
reference impl ground truth — different schema, different SQL, same
``KGStore`` Protocol surface.
"""

from knowlytix.kal.storage_default.models import (
    KalEnmEntry,
    KalKgEntity,
    KalKgRelation,
    KalKgTriple,
    KalPhaseRange,
    KalRelationSynonym,
    KalStiefelProjection,
)
from knowlytix.kal.storage_default.store import KalDefaultKGStore
from knowlytix.kal.storage_default.verification_models import (
    KalKgEvidence,
    KalPolicyRule,
    KalTripleVerdict,
    KalVerificationRun,
)

__all__ = [
    "KalDefaultKGStore",
    "KalEnmEntry",
    "KalKgEntity",
    "KalKgEvidence",
    "KalKgRelation",
    "KalKgTriple",
    "KalPhaseRange",
    "KalPolicyRule",
    "KalRelationSynonym",
    "KalStiefelProjection",
    "KalTripleVerdict",
    "KalVerificationRun",
]
