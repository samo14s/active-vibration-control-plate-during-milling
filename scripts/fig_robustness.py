"""Figures: robustness and material-removal scheduling.

fig09: Monte-Carlo dispersion of the closed-loop critical depth under
       cutting-coefficient dispersion (0.3-2.9x), +-10 % modal
       mass/stiffness, and up to -20 % damping, at the worst position.
fig10: critical depth vs material-removal state: removal-scheduled vs
       fresh-plate-scheduled vs frozen.
"""
import numpy as np

import matplotlib.pyplot as plt
from common import COLORS, DC, SC, savefig
from pipeline import REMOVALS, RPM_REF, X_EVAL, stage_d, stage_e


def main():
    D = stage_d()

    fig, ax = plt.subplots(figsize=(SC, 2.7))
    # PS-LPV-DR = PS-LPV (certified k_r = 0); not shown as a separate box
    strats = [s for s in D if s != "PS-LPV-DR"]
    data = [D[s] * 1e3 for s in strats]
    bp = ax.boxplot(data, tick_labels=strats, showfliers=True,
                    patch_artist=True, widths=0.5)
    for patch, s in zip(bp["boxes"], strats):
        patch.set_facecolor(COLORS[s])
        patch.set_alpha(0.6)
    ax.set_ylabel(rf"$a_{{\rm lim}}$ @ {RPM_REF/1e3:.1f} krpm, "
                  r"$x_T$ = 95 mm [mm]")
    ax.set_title("Monte-Carlo robustness (40 samples)", fontsize=8.5)
    for s, d in zip(strats, data):
        print(f"{s:10s} p5={np.percentile(d, 5):.3f} "
              f"median={np.median(d):.3f} p95={np.percentile(d, 95):.3f} mm")
    fig.tight_layout()
    savefig(fig, "fig09_robustness")

    # ---------------- fig10: removal scheduling -----------------------
    E = stage_e()
    fig, axs = plt.subplots(1, len(REMOVALS), figsize=(DC, 2.5),
                            sharey=True)
    xs_mm = np.array(X_EVAL) * 1e3
    for ax, r in zip(axs, REMOVALS):
        for key, lbl, c, ls in (
                ("sched", "PS-LPV (removal-scheduled)", COLORS["PS-LPV"], "-"),
                ("sched-fresh", "PS-LPV (fresh-plate gains)",
                 COLORS["PS-LPV"], "--"),
                ("frozen", "frozen $H_\\infty$", COLORS["FROZEN"], "-")):
            vals = [E[(key, x, r)] * 1e3 for x in X_EVAL]
            ax.plot(xs_mm, vals, ls, color=c, marker="o", ms=3, label=lbl)
        ax.set_title(f"removal {r*1e3:.1f} mm", fontsize=8.5)
        ax.set_xlabel(r"$x_T$ [mm]")
    axs[0].set_ylabel(r"$a_{\rm lim}$ [mm]")
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    savefig(fig, "fig10_removal")


if __name__ == "__main__":
    main()
