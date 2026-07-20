"""Figures: time-domain evaluation.

fig07: fixed-position runs at the reference condition (4.9 krpm,
       ap = 0.3 mm, x_T = 5 mm): displacement, voltage, PSD.
fig08: full moving pass at ap = 1.0 mm: scheduled vs frozen.
"""
import numpy as np

import matplotlib.pyplot as plt
from common import COLORS, DC, LABELS, savefig
from pipeline import STRATEGIES, stage_c


def main():
    C = stage_c()

    # ---------------- fig07: fixed position ---------------------------
    show = ("OL", "FROZEN", "PS-LPV", "PS-LPV-DR")
    fig, axs = plt.subplots(2, 2, figsize=(DC, 4.2))
    ax = axs[0, 0]
    for strat in show:
        d = C[("fixed", strat)]
        ax.plot(d["t"], d["w"] * 1e6, color=COLORS[strat], lw=0.7,
                alpha=0.9, label=LABELS[strat])
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$w(x_T)$ [$\mu$m]")
    ax.set_title("(a) milling-point displacement", fontsize=8.5)
    ax.legend(fontsize=6)

    ax = axs[0, 1]
    for strat in show:
        if strat == "OL":
            continue
        d = C[("fixed", strat)]
        ax.plot(d["t"], d["u"], color=COLORS[strat], lw=0.7, alpha=0.9,
                label=LABELS[strat])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("voltage [V]")
    ax.set_title("(b) control voltage", fontsize=8.5)
    ax.legend(fontsize=6)

    ax = axs[1, 0]
    for strat in show:
        f, S = C[("fixed", strat)]["psd"]
        ax.semilogy(f, np.sqrt(S) * 1e6, color=COLORS[strat], lw=0.9,
                    label=LABELS[strat])
    ax.set_xlim(0, 5000)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"ASD [$\mu$m/$\sqrt{\rm Hz}$]")
    ax.set_title("(c) displacement spectrum", fontsize=8.5)
    ax.legend(fontsize=6)

    ax = axs[1, 1]
    idx = np.arange(len(show))
    rms_w = [C[("fixed", s)]["rms_w"] * 1e6 for s in show]
    rms_u = [C[("fixed", s)]["rms_u"] for s in show]
    ax.bar(idx - 0.2, rms_w, 0.4, color=[COLORS[s] for s in show])
    ax.set_ylabel(r"RMS $w$ [$\mu$m]")
    ax2 = ax.twinx()
    ax2.bar(idx + 0.2, rms_u, 0.4, color=[COLORS[s] for s in show],
            alpha=0.45)
    ax2.set_ylabel("RMS voltage [V] (light)")
    ax2.grid(False)
    ax.set_xticks(idx, show, fontsize=6.5, rotation=12)
    ax.set_title("(d) steady-state metrics", fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "fig07_time_domain")

    # ---------------- fig08: moving pass ------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(DC, 4.0), sharex=True)
    for strat in ("FROZEN", "PS-LPV", "PS-LPV-DR"):
        d = C[("pass", strat)]
        axs[0].plot(d["x"] * 1e3, d["w"] * 1e6, color=COLORS[strat],
                    lw=0.6, alpha=0.85, label=LABELS[strat])
        axs[1].plot(d["x"] * 1e3, d["u"], color=COLORS[strat], lw=0.6,
                    alpha=0.85, label=LABELS[strat])
    axs[0].set_ylabel(r"$w(x_T)$ [$\mu$m]")
    axs[0].set_title(
        "full pass, $a_p$ = 1.0 mm "
        "(1.4x the frozen design's worst-position limit)",
        fontsize=8.5)
    axs[0].legend(fontsize=6.5)
    axs[1].set_ylabel("voltage [V]")
    axs[1].set_xlabel(r"tool position $x_T$ [mm]")
    fig.tight_layout()
    savefig(fig, "fig08_moving_pass")

    for s in STRATEGIES:
        if ("fixed", s) in C:
            d = C[("fixed", s)]
            print(f"{s:10s} rms_w={d['rms_w']*1e6:8.2f} um "
                  f"rms_u={d['rms_u']:6.2f} V")


if __name__ == "__main__":
    main()
