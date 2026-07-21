"""Figures: stability lobe diagrams.

fig04: open-loop SLDs at three positions (validation vs the reference
       paper's < 0.1 mm limits) on the full-order model.
fig05: closed-loop SLDs per strategy at start / middle / end positions.
fig06: critical depth at the reference speed vs tool position - the
       conservatism-gap headline figure.
"""
import numpy as np

import matplotlib.pyplot as plt
from common import COLORS, DC, LABELS, SC, savefig
from pipeline import RPM_REF, X_EVAL, stage_b
from pipeline import STRATEGIES as _ALL_STRATS

# PS-LPV-DR coincides exactly with PS-LPV (its certified delayed gain is
# zero at every scheduling point), so it is not drawn as a separate
# series; it would only duplicate the PS-LPV curve/bar in every panel.
STRATEGIES = tuple(s for s in _ALL_STRATS if s != "PS-LPV-DR")


def main():
    B = stage_b()
    rpms = B["rpms"]
    curves = B["curves"]

    # ---------------- fig04: open loop --------------------------------
    fig, ax = plt.subplots(figsize=(SC, 2.5))
    for x, ls in ((0.005, "-"), (0.05, "--"), (0.095, ":")):
        ax.plot(rpms / 1e3, curves[("OL", x)] * 1e3, ls, color="k",
                label=f"$x_T$ = {x*1e3:.0f} mm")
    ax.axhline(0.1, color="#d62728", lw=0.8, ls="-.",
               label="0.1 mm (ref. experimental range)")
    ax.axvline(RPM_REF / 1e3, color="#7f7f7f", lw=0.8)
    ax.set_xlabel("spindle speed [krpm]")
    ax.set_ylabel(r"$a_{\rm lim}$ [mm]")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "fig04_sld_openloop")

    # ---------------- fig05: closed loop ------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(DC, 2.8), sharey=True)
    for ax, x in zip(axs, (0.005, 0.05, 0.095)):
        for strat in STRATEGIES:
            ls = "--" if strat == "PS-LPV" else "-"
            ax.semilogy(rpms / 1e3, np.maximum(curves[(strat, x)], 1e-6) * 1e3,
                        ls, color=COLORS[strat], label=LABELS[strat],
                        lw=1.3 if strat.startswith("PS") else 1.0)
        ax.set_xlabel("spindle speed [krpm]")
        ax.set_title(f"$x_T$ = {x*1e3:.0f} mm", fontsize=8.5)
        ax.set_ylim(0.05, 6.0)
    axs[0].set_ylabel(r"$a_{\rm lim}$ [mm]")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, "fig05_sld_closedloop")

    # ---------------- fig06: a_lim vs position at RPM_REF -------------
    i_ref = int(np.argmin(np.abs(rpms - RPM_REF)))
    fig, axs = plt.subplots(1, 2, figsize=(DC, 2.6))
    xs_mm = np.array(X_EVAL) * 1e3
    ax = axs[0]
    for strat in STRATEGIES:
        vals = [curves[(strat, x)][i_ref] * 1e3 for x in X_EVAL]
        ax.plot(xs_mm, vals, "-o", ms=4, color=COLORS[strat],
                label=LABELS[strat])
    ax.set_xlabel(r"tool position $x_T$ [mm]")
    ax.set_ylabel(rf"$a_{{\rm lim}}$ @ {RPM_REF/1e3:.1f} krpm [mm]")
    ax.set_title("(a) critical depth along the pass", fontsize=8.5)
    ax.legend(fontsize=6)

    ax = axs[1]
    mins = {s: min(np.min(curves[(s, x)]) for x in X_EVAL)
            for s in STRATEGIES}
    worst = {s: min(curves[(s, x)][i_ref] for x in X_EVAL)
             for s in STRATEGIES}
    idx = np.arange(len(STRATEGIES))
    ax.bar(idx - 0.2, [worst[s] * 1e3 for s in STRATEGIES], 0.4,
           color=[COLORS[s] for s in STRATEGIES],
           label=f"worst position @ {RPM_REF/1e3:.1f} krpm")
    ax.bar(idx + 0.2, [mins[s] * 1e3 for s in STRATEGIES], 0.4,
           color=[COLORS[s] for s in STRATEGIES], alpha=0.45,
           label="worst position and speed (2-10 krpm)")
    ax.set_xticks(idx, STRATEGIES, fontsize=6.5, rotation=12)
    ax.set_ylabel(r"$a_{\rm lim}$ [mm]")
    ax.set_title("(b) worst-case guarantees", fontsize=8.5)
    ax.legend(fontsize=6)
    fig.tight_layout()
    savefig(fig, "fig06_alim_position")

    print("worst-position a_lim @ref rpm:",
          {s: round(worst[s] * 1e3, 3) for s in STRATEGIES})


if __name__ == "__main__":
    main()
