"""
Path bootstrap for the verification scripts.

The baseline package requires every module to sit in one flat directory
(`setup_workspace.py` copies them). Rather than duplicate files, this puts the
baseline source directories and `src/` on sys.path, so the original modules
keep their flat import style (`import mindlin_q8`) while the repository stays
organised.

Import this first in any script that uses the models:

    import _bootstrap  # noqa: F401
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DIRS = (
    "baseline/01_core",
    "baseline/02_controllers",
    "baseline/03_analysis",
    "src",
)

for _d in _DIRS:
    _p = os.path.join(ROOT, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
