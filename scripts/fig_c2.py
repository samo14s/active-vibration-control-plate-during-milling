"""Figure 11: the differentiating fixed condition - end of the pass
(x_T = 95 mm, the frozen design's weakest position), a_p = 2.0 mm.

Open loop and delayed PD are unstable (loss-of-contact limit cycles),
the frozen design survives at high voltage and amplitude, the scheduled
controller is comfortable.
"""
import numpy as np

import matplotlib.pyplot as plt
from common import COLORS, DC, LABELS, savefig
from pipeline import stage_c2


def main():
    C = stage_c2()
    ap = 2.0e-3
    show = ("OL", "DPD", "FROZEN", "PS-LPV")

    fig, axs = plt.subplots(1, 3, figsize=(DC, 2.7))

    ax = axs[0]
    for strat in ("FROZEN", "PS-LPV"):
        d = C[(ap, strat)]
        lbl = (f"{LABELS[strat]}\n{d['rms_w']*1e6:.1f} µm RMS, "
               f"{d['peak_w']*1e6:.0f} µm peak")
        ax.plot(d["t"], d["w"] * 1e6, color=COLORS[strat], lw=0.5,
                alpha=0.9, label=lbl)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$w(x_T)$ [$\mu$m]")
    ax.set_ylim(-80, 80)
    ax.set_title("(a) milling-point displacement\n"
                 "(open loop & delayed PD diverge here)", fontsize=8)
    ax.legend(fontsize=5.6, loc="upper right")

    ax = axs[1]
    ax.axhspan(150, 230, color="0.85", alpha=0.7)
    ax.axhspan(-230, -150, color="0.85", alpha=0.7)
    ax.axhline(150, color="k", lw=0.6, ls="--")
    ax.axhline(-150, color="k", lw=0.6, ls="--")
    for strat in ("FROZEN", "PS-LPV"):
        d = C[(ap, strat)]
        lbl = (f"{LABELS[strat]}\n{d['rms_u']:.1f} V RMS, "
               f"{d['extras']['sat_frac']*100:.1f}% clipped")
        ax.plot(d["t"], d["u"], color=COLORS[strat], lw=0.5, alpha=0.9,
                label=lbl)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("control voltage [V]")
    ax.set_ylim(-215, 215)
    ax.text(0.012, -205, "amplifier saturation $\\pm$150 V", fontsize=6,
            ha="left", va="bottom", color="0.25")
    ax.set_title("(b) control voltage\n(frozen design saturates)",
                 fontsize=8)
    ax.legend(fontsize=5.6, loc="upper right")

    ax = axs[2]
    idx = np.arange(len(show))
    rms = []
    for s in show:
        d = C[(ap, s)]
        r = d["rms_w"] * 1e6
        rms.append(r)
    bars = ax.bar(idx, rms, 0.55, color=[COLORS[s] for s in show])
    for i, s in enumerate(show):
        if rms[i] > 50:                      # unstable limit cycle
            ax.text(i, 55, "unstable", ha="center", fontsize=6.5,
                    rotation=90, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(1, 1e7)
    ax.set_xticks(idx, show, fontsize=6.5, rotation=12)
    ax.set_ylabel(r"RMS $w$ [$\mu$m] (log)")
    ax.set_title("(c) all strategies", fontsize=8.5)

    fig.tight_layout()
    savefig(fig, "fig11_hard_condition")

    for s in show:
        d = C[(ap, s)]
        print(f"{s:8s} rms={d['rms_w']*1e6:12.2f} um  rms_u={d['rms_u']:6.1f} V"
              f"  peak_u={d['peak_u']:6.1f} V")


if __name__ == "__main__":
    main()
