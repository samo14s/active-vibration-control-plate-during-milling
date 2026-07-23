#!/usr/bin/env python3
"""
Robustness to the paper's additive dynamic uncertainty (Du et al. Eq. 19, weight
W_Pau) for the sliding-mode family.

Left  (a): relative additive uncertainty |W_Pau(jw) / G(jw)| (standard robust-
           control form) -- negligible at the controlled modes but exceeding
           100% (uncertainty = plant) in the inter-modal troughs / high band
           toward the RHP zero.
Right (b): settled RMS vs uncertainty magnitude kappa (worst over Delta in
           {+1,-1,allpass}); a railed / diverged controller is drawn at the
           ceiling with a marker.  TD-SMC and ATD-SMC stay flat across the full
           weight (they meet the paper's ||W_Pau K S||<1); SMC and AST-SMC cross
           their robust-stability boundary between kappa=0.5 and 1.0.

Run:  python make_uncertainty_figure.py     (~4 min)
"""
from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import plant as P
from src import uncertainty as UNC
from src.controllers.smc import SMC
from src.controllers.tdsmc import TDSMC
from src.controllers.atdsmc import ATDSMC
from src.controllers.astsmc import ASTSMC

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FAM = [("SMC", SMC, "#188038"), ("TD-SMC", TDSMC, "#7b5c00"),
       ("ATD-SMC", ATDSMC, "#8e24aa"), ("AST-SMC", ASTSMC, "#d81b60")]
KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CEIL = 300.0


def main():
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))

    # (a) relative additive uncertainty |W_Pau / G| (standard robust-control form)
    pl = P.MillingPlant()
    A = pl.A
    Bu = np.concatenate([np.zeros(pl.n), pl.Hpe])
    Cy = pl.Cy.ravel()
    fhz = np.logspace(np.log10(200), np.log10(6000), 400)
    rel = []
    for fq in fhz:
        w = 2 * np.pi * fq
        G = Cy @ np.linalg.solve(1j * w * np.eye(2 * pl.n) - A, Bu)
        Wp = np.polyval(UNC.NUM_WPAU, 1j * w) / np.polyval(UNC.DEN_WPAU, 1j * w)
        rel.append(abs(Wp / G))
    ax[0].loglog(fhz, rel, color="#8e24aa", lw=1.8,
                 label=r"$|W_{Pau}(j\omega)/G(j\omega)|$")
    ax[0].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[0].text(210, 1.25, "100% (uncertainty = plant)", fontsize=8)
    for fn, lb in [(540, "mode 1"), (1068, "mode 2"), (2787, "mode 3"), (5171, "RHP zero")]:
        ax[0].axvline(fn, color="gray", ls=":", lw=0.7)
        ax[0].text(fn, ax[0].get_ylim()[0] * 1.6, lb, rotation=90, fontsize=7,
                   va="bottom", ha="right", color="gray")
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel("relative additive uncertainty (–)")
    ax[0].set_title("(a) Paper additive uncertainty, relative to the plant\n"
                    r"$|W_{Pau}/G|$ exceeds 100% in the inter-modal troughs / high band")
    ax[0].legend(fontsize=9, loc="upper left")

    # (b) settled RMS vs kappa
    for name, ctor, col in FAM:
        ys, div = [], []
        for kap in KAPPAS:
            o = UNC.worst_over_delta(lambda: P.MillingPlant(), lambda: ctor(), kappa=kap)
            if o is None or o[0] > CEIL:
                ys.append(np.nan); div.append(kap)
                print(f"  {name} kappa={kap}: DIV/rail", flush=True)
            else:
                ys.append(o[0])
                print(f"  {name} kappa={kap}: {o[0]:.1f} um / {o[1]:.0f} V", flush=True)
        ax[1].plot(KAPPAS, ys, "o-", color=col, lw=1.8, ms=5, label=name)
        for kap in div:
            ax[1].plot([kap], [CEIL * 0.82], "x", color=col, ms=9, mew=2.0)
    ax[1].axhline(60.0, color="k", ls="--", lw=0.7, alpha=0.6)
    ax[1].text(0.02, 66, "control limit 60 µm", fontsize=8)
    ax[1].set_yscale("log")
    ax[1].set_ylim(2.0, CEIL)
    ax[1].set_xlabel(r"uncertainty magnitude  $\kappa$  ($\kappa\!=\!1$ = full paper weight $W_{Pau}$)")
    ax[1].set_ylabel("settled RMS displacement (µm, log; worst Δ)")
    ax[1].set_title("(b) Robustness to the paper's additive uncertainty\n"
                    "TD-SMC / ATD-SMC stay flat; SMC / AST-SMC cross the "
                    "robust-stability boundary")
    ax[1].legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    out = os.path.join(RESULTS, "fig_uncertainty.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
