#!/usr/bin/env python3
"""Three standard representations of the SAME additive uncertainty weight W_Pau
(Du et al. Eq. 19), so the user can identify which one their Figure 8 uses.
Pure frequency-response math (no simulation). Writes results/fig_unc_repr_options.png.
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

# scaling used by the paper's mu-synthesis design (musyn.py / hinf.py)
SU, SY = 1.0e2, 1.0e-4

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def main():
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    pl = P.MillingPlant()
    A = pl.A
    Bu = np.concatenate([np.zeros(pl.n), pl.Hpe])
    Cy = pl.Cy.ravel()
    f = np.logspace(np.log10(100), np.log10(8000), 600)
    Wp, G = [], []
    for fq in f:
        w = 2 * np.pi * fq
        Wp.append(np.polyval(UNC.NUM_WPAU, 1j * w) / np.polyval(UNC.DEN_WPAU, 1j * w))
        G.append(Cy @ np.linalg.solve(1j * w * np.eye(2 * pl.n) - A, Bu))
    Wp = np.array(Wp); G = np.array(G)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (A) additive weight |W_Pau| in dB -- physical and mu-design-scaled (x SU/SY)
    ax[0].semilogx(f, 20 * np.log10(np.abs(Wp)), color="#c5221f", lw=1.8,
                   label=r"$|W_{Pau}|$ physical (m/V)")
    ax[0].semilogx(f, 20 * np.log10(np.abs(Wp) * (SU / SY)), color="#1a73e8", lw=1.8,
                   ls="--", label=r"$|W_{Pau}|\cdot(S_U/S_Y)$ (design units)")
    ax[0].set_title("(A) Additive weight, Bode magnitude (dB)")
    ax[0].set_ylabel("magnitude (dB)")

    # (B) relative / multiplicative-equivalent uncertainty |W_Pau / G|
    ax[1].loglog(f, np.abs(Wp / G), color="#8e24aa", lw=1.8)
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[1].text(120, 1.2, "100% (uncertainty = plant)", fontsize=8)
    ax[1].set_title(r"(B) Relative additive uncertainty  $|W_{Pau}/G|$")
    ax[1].set_ylabel("ratio (–)")

    # (C) the two absolute magnitudes (my current panel a)
    ax[2].loglog(f, np.abs(G), color="#1a73e8", lw=1.6, label=r"$|G|$")
    ax[2].loglog(f, np.abs(Wp), color="#c5221f", lw=1.6, ls="--", label=r"$|W_{Pau}|$")
    ax[2].set_title("(C) Absolute |G| and |W_Pau| (m/V) — my current panel")

    for a in ax:
        a.set_xlabel("frequency (Hz)")
        for fn in (540, 1068, 2787, 5171):
            a.axvline(fn, color="gray", ls=":", lw=0.6)
        if a.get_legend_handles_labels()[1]:
            a.legend(fontsize=8, loc="best")
    fig.suptitle("Which representation matches your Figure 8?  (same W_Pau, Eq. 19)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESULTS, "fig_unc_repr_options.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
