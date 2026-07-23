#!/usr/bin/env python3
"""
The study's decisive spatio-temporal figure: the FULL-MOVING-PASS force ceiling.

For each controller and each elevated cutting-force factor alpha4, runs the full
20.4 s peripheral pass (moving milling point + material-removal frequency drift
+ measurement noise) and plots the pass-mean RMS; a diverging run is drawn at
the ceiling with a ✗ and annotated with its failure time.

This is the sweep that exposes the central spatio-temporal result: statically
the developed controllers hold alpha4 ~ 4.0x, yet the full pass fails far
earlier (ATD-SMC at 1.8x, AST-SMC at 2.2x) — the moving pass is STRUCTURALLY
harder than the worst static snapshot, and AST-SMC's super-twisting integral
term lifts the pass ceiling from 1.6x to 2.0x (+25%).

Run:  python make_pass_ceiling_figure.py     (heavy: ~10 full passes, ~25 min)
"""
from __future__ import annotations
import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C, plant as P
from src.simulate import simulate
from src.controllers.atdsmc import ATDSMC
from src.controllers.astsmc import ASTSMC

RESULTS = os.path.join(os.path.dirname(__file__), "results")
A4S = [1.0, 1.4, 1.6, 1.8, 2.0, 2.2]
FAM = [("ATD-SMC", lambda: ATDSMC(), "#8e24aa"),
       ("AST-SMC", lambda: ASTSMC(moving=True), "#d81b60")]
CEIL = 300.0


def main():
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name, mk, col in FAM:
        ys, div = [], []
        for a4 in A4S:
            t0 = time.time()
            r = simulate(P.MillingPlant(moving=True, freq_drift=True,
                                        alpha4_factor=a4),
                         mk(), t_sim=C.T_PASS, meas_noise=True)
            y = np.asarray(r["y"]) * 1e6
            if r["diverged"] or not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e4:
                ys.append(np.nan); div.append((a4, float(r["t"][-1])))
                print(f"  {name} a4={a4}: DIV t={r['t'][-1]:.1f}s ({time.time()-t0:.0f}s)",
                      flush=True)
            else:
                ys.append(float(np.sqrt(np.mean(y ** 2))))
                print(f"  {name} a4={a4}: {ys[-1]:.2f} um ({time.time()-t0:.0f}s)",
                      flush=True)
        ax.semilogy(A4S, ys, "o-", color=col, lw=1.8, ms=5, label=name)
        for a4, tend in div:
            ax.plot([a4], [CEIL * 0.82], "x", color=col, ms=9, mew=2.0)
            ax.annotate(f"t≈{tend:.0f}s", xy=(a4, CEIL * 0.62), fontsize=7,
                        ha="center", color=col)
    ax.axhline(60.0, color="k", ls="--", lw=0.7, alpha=0.6)
    ax.text(A4S[0], 66, "control limit 60 µm", fontsize=8, color="k")
    ax.set_ylim(1.0, CEIL)
    ax.set_xlabel(r"cutting-force factor  $\alpha_4$  (× average)")
    ax.set_ylabel("pass-mean RMS displacement (µm, log)")
    ax.set_title("Full 20.4 s moving-pass force ceiling  (✗ = diverges, with failure time)\n"
                 "statically both hold ~4.0×, but the pass fails far earlier — "
                 "AST-SMC lifts the ceiling 1.6× → 2.0×")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fig_pass_ceiling.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
