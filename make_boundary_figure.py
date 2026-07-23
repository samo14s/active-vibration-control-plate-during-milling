#!/usr/bin/env python3
"""
Manuscript figure: the two stability-envelope sweeps that distinguish the
sliding-mode family --

  (a) milling-force-factor alpha4 stability boundary (settled RMS vs alpha4;
      a diverging controller is drawn at the ceiling with a marker), and
  (b) modal-frequency detuning band (settled RMS vs +/- % shift; mass removal
      is the +side), both at the paper's reduced damping (-20%).

Recomputes the sweeps from the plant + controllers so the figure is fully
reproducible.  Writes results/fig_boundary_detuning.png.

Run:  python make_boundary_figure.py
"""
from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import plant as P
from src.simulate import simulate
from src.controllers.smc import SMC
from src.controllers.tdsmc import TDSMC
from src.controllers.atdsmc import ATDSMC

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FAM = [("SMC", SMC, "#188038"), ("TD-SMC", TDSMC, "#7b5c00"),
       ("ATD-SMC", ATDSMC, "#8e24aa")]
CEIL = 1.0e3


def _rms(a): return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def settled(plant_kw, ctor, t_sim=0.35):
    r = simulate(P.MillingPlant(**plant_kw), ctor(), t_sim=t_sim, meas_noise=True)
    y = np.asarray(r["y"]) * 1e6
    if r["diverged"] or not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e4:
        return None
    n = len(y)
    return _rms(y[2 * n // 3:])


def main():
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))

    # (a) alpha4 boundary at -20% damping
    a4s = np.round(np.arange(3.0, 4.21, 0.1), 2)
    for name, ctor, col in FAM:
        ys, divx = [], []
        for a in a4s:
            v = settled(dict(alpha4_factor=float(a), dzeta=-0.2), ctor)
            if v is None:
                ys.append(np.nan); divx.append(a)
            else:
                ys.append(v)
        ax[0].semilogy(a4s, ys, "o-", color=col, lw=1.6, ms=4, label=name)
        if divx:
            ax[0].plot(divx, [CEIL * 0.85] * len(divx), "x", color=col, ms=7, mew=1.7)
    ax[0].axvspan(0.3, 2.9, color="gray", alpha=0.06)
    ax[0].text(2.9, 1.4, "paper band\n(≤2.9×)", fontsize=7, ha="right", color="gray")
    ax[0].set_ylim(1.0, CEIL)
    ax[0].set_xlabel(r"milling-force factor  $\alpha_4$  (× average)")
    ax[0].set_ylabel("settled RMS displacement (µm, log)")
    ax[0].set_title("(a) Cutting-force stability boundary  (−20% damping)\n"
                    "✗ = diverges;  ATD-SMC extends to ≈4.0× vs 3.6× / 3.1×")
    ax[0].legend(loc="upper left", fontsize=9)

    # (b) detuning band at -20% damping
    shifts = np.array([-8, -6, -4, -2, 0, 4, 8, 12, 17])
    for name, ctor, col in FAM:
        ys, divx = [], []
        for s in shifts:
            dmass = 1.0 / (1.0 + s / 100.0) ** 2 - 1.0
            v = settled(dict(dmass=dmass, dzeta=-0.2), ctor)
            if v is None:
                ys.append(np.nan); divx.append(s)
            else:
                ys.append(v)
        ax[1].semilogy(shifts, ys, "o-", color=col, lw=1.6, ms=4, label=name)
        if divx:
            ax[1].plot(divx, [CEIL * 0.85] * len(divx), "x", color=col, ms=7, mew=1.7)
    ax[1].axvline(0, color="gray", ls=":", lw=0.8)
    ax[1].set_ylim(1.0, CEIL)
    ax[1].set_xlabel("modal-frequency shift (%)   (mass removal → + side)")
    ax[1].set_ylabel("settled RMS displacement (µm, log)")
    ax[1].set_title("(b) Modal-detuning robustness  (−20% damping)\n"
                    "SMC collapses at both edges; ATD-SMC/TD-SMC hold a flat band")
    ax[1].legend(loc="upper center", fontsize=9)

    fig.tight_layout()
    out = os.path.join(RESULTS, "fig_boundary_detuning.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
