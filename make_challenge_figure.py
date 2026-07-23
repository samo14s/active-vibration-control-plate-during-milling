#!/usr/bin/env python3
"""
Harder-than-paper challenge comparison for the SMC family: plain SMC, the (known)
combined TD-SMC, and the DEVELOPED ATD-SMC.  Runs src/challenges.py and plots the
settled RMS on every challenge (log scale; a diverging / saturating controller is
drawn at the ceiling with a red x) plus the survival count.

Run:  python make_challenge_figure.py
"""
from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import challenges as CH
from src.controllers.smc import SMC
from src.controllers.tdsmc import TDSMC
from src.controllers.atdsmc import ATDSMC
from src.controllers.astsmc import ASTSMC

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FAMILY = [("SMC", SMC, "#188038"),
          ("TD-SMC", TDSMC, "#7b5c00"),
          ("ATD-SMC (developed)", ATDSMC, "#8e24aa"),
          ("AST-SMC (developed)", ASTSMC, "#d81b60")]
CEIL = 3.0e3


def main():
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    names = [c[0] for c in CH.CHALLENGES]
    data = {}
    for lab, ctor, col in FAMILY:
        r = CH.evaluate(lambda **kw: ctor(**kw))
        data[lab] = (r, col)
        print(f"{lab:20s}: survived {r['n_survived']}/{r['n_total']}  score {r['score']:.0f}")

    fig, ax = plt.subplots(figsize=(14, 5.8))
    x = np.arange(len(names))
    nfam = len(FAMILY)
    w = 0.8 / nfam
    for i, (lab, ctor, col) in enumerate(FAMILY):
        r, _ = data[lab]
        vals, div = [], []
        for nm in names:
            m = r["per"][nm]
            if m["diverged"] or m["rms"] is None:
                vals.append(CEIL); div.append(True)
            else:
                vals.append(max(m["rms"], 0.3)); div.append(False)
        xb = x + (i - (nfam - 1) / 2.0) * w
        ax.bar(xb, vals, w, color=col, label=f"{lab}  ({r['n_survived']}/{r['n_total']})")
        for xi, v, d in zip(xb, vals, div):
            if d:
                ax.text(xi, CEIL * 1.05, "✗", ha="center", va="bottom",
                        color="#c5221f", fontsize=9, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.3, CEIL * 2.2)
    ax.axhline(CH.SURVIVE_UM, color="k", ls="--", lw=0.7, alpha=0.6)
    ax.text(len(names) - 0.5, CH.SURVIVE_UM * 1.1, "control limit 60 µm",
            fontsize=8, ha="right", color="k")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("settled RMS displacement (µm, log)")
    ax.set_title("Harder-than-paper challenges — SMC family "
                 "(✗ = diverges / rails; extreme α₄≥4.5 is beyond the piezo's "
                 "physical authority for all)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fig_challenges.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
