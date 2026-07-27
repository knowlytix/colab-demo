# Nuitka re-export stub — see GMS-110 spike README for the rationale.
# `from ... import *` copies the names listed in __all__ but NOT the
# __all__/__version__ dunders themselves, so re-export them explicitly —
# otherwise knowlytix.core has no __all__/__version__ and the public-API
# contract + knowlytix-smoke + nuitka_compat_audit all fail.
from knowlytix.core.core import *  # noqa: F401,F403
from knowlytix.core.core import __all__, __version__  # noqa: F401
