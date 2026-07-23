"""
_pkg_path.py
============
Import helper for the ``research_extensions`` package.

The base ``article_simulation_package`` keeps its modules in a flat, import-by-name
convention (``from plate_model import PlateModel``).  Importing this module once
puts the base package's source directories on ``sys.path`` so the extension
modules can reuse the *unmodified* plant, controllers and analysis code::

    import _pkg_path            # noqa: F401  (side-effect import)
    from plate_model import PlateModel
    from lqg_controller import LQGController

This keeps the original package untouched while the new modules live alongside
it in ``research_extensions/``.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.normpath(os.path.join(_HERE, "..", "article_simulation_package"))

_SUBDIRS = ["01_core", "02_controllers", "03_analysis", "04_figures", "05_main"]

for _d in _SUBDIRS:
    _p = os.path.join(_PKG, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Also expose the package root (for completeness).
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


def package_root():
    """Absolute path to the base ``article_simulation_package``."""
    return _PKG
