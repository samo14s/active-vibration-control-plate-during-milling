"""Shared utilities for the manuscript figure scripts.

Publication style, model caching, and the strategy roster evaluated in
every comparison figure.
"""
from __future__ import annotations

import pathlib
import pickle
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIGDIR = ROOT / "paper" / "figures"
RESDIR = ROOT / "results"
CACHE = RESDIR / "cache"
for d in (FIGDIR, RESDIR, CACHE):
    d.mkdir(parents=True, exist_ok=True)

# ---- publication style ---------------------------------------------------
plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.1,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# single-column ~ 3.5 in, double-column ~ 7.2 in (Elsevier)
SC = 3.5
DC = 7.2

COLORS = {
    "OL": "#7f7f7f",
    "DPD": "#9467bd",
    "FROZEN": "#d62728",
    "PS-LPV": "#1f77b4",
    "PS-LPV-SA": "#ff7f0e",
    "PS-LPV-DR": "#2ca02c",
}
LABELS = {
    "OL": "no control",
    "DPD": "delayed PD (single time-delay control)",
    "FROZEN": "best frozen $H_\\infty$ (non-scheduled)",
    "PS-LPV": "PS-LPV (performance-weighted)",
    "PS-LPV-SA": "PS-LPV-SA (saturation-aware, proposed)",
    "PS-LPV-DR": "PS-LPV-DR (proposed, full)",
}


def cached(name: str, builder, refresh: bool = False):
    """Pickle-cache expensive artifacts under results/cache."""
    f = CACHE / f"{name}.pkl"
    if f.exists() and not refresh:
        with open(f, "rb") as fh:
            return pickle.load(fh)
    obj = builder()
    with open(f, "wb") as fh:
        pickle.dump(obj, fh)
    return obj


def savefig(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    print(f"[fig] {name}")
    return FIGDIR / f"{name}.png"


def models(n_design: int = 3, n_full: int = 12, removals=(0.0,),
           anchor: bool | None = None):
    """Build (and cache) the modal models used across figures.

    The working model is the Mindlin FEM with its natural frequencies
    corrected to the measured EMA values (avc.modal.ANCHOR_MEASURED_FREQS).
    Pass anchor=False to obtain the raw, uncorrected FEM (used by the
    first-principles validation figures). USE_CLASSICAL_MODEL switches the
    whole stack to the classical Warburton model instead."""
    from avc.synthesis import USE_CLASSICAL_MODEL
    from avc.classical_plate import reduce_model_classical
    from avc.modal import ANCHOR_MEASURED_FREQS, reduce_model
    from avc.params import DEFAULT
    anc = ANCHOR_MEASURED_FREQS if anchor is None else anchor

    def rm(p, n, height_removed):
        if USE_CLASSICAL_MODEL:
            return reduce_model_classical(p, n, height_removed)
        return reduce_model(p, n, height_removed, anchor=anc)

    def build():
        out = {}
        for r in removals:
            out[r] = {
                "design": rm(DEFAULT, n_design, height_removed=r),
                "full": rm(DEFAULT, n_full, height_removed=r),
            }
        return out

    tag = ("cl" if USE_CLASSICAL_MODEL else ("fem_a" if anc else "fem_raw"))
    key = (f"models_{tag}_n{n_design}_{n_full}_r"
           + "_".join(f"{r:.4g}" for r in removals))
    return cached(key, build)
