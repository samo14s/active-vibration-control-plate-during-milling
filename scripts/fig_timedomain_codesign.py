"""Time-domain response at the hard end-of-pass condition (x_T = 95 mm,
a_p = 2 mm, 4900 rpm), comparing the frozen H-inf design, the naively
weighted PS-LPV, and the saturation-aware co-design PS-LPV-SA.

The story in one figure: the frozen design saturates and chatters; the
naive PS-LPV rejects the chatter but rides the +-150 V rail; PS-LPV-SA
achieves the SAME rejection with ~50 V of headroom to spare.
Output: paper/figures/fig17_timedomain_codesign.{pdf,png}
"""
import pathlib
import sys

import numpy as np

import matplotlib.pyplot as plt
from common import COLORS, DC, LABELS, savefig
from pipeline import RPM_REF, controller_for, models, stage_a

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from avc.simulate import simulate  # noqa: E402

V_MAX = 150.0
X_HARD = 0.095
AP = 2.0e-3
FS = 200e3
T = 0.5
SHOW = ("FROZEN", "PS-LPV", "PS-LPV-SA")
# steady-state display window (the campaign measures RMS over the last
# 0.1 s; show a short slice inside it so the oscillations are legible)
W0, W1 = 0.40, 0.44


def main():
    art = stage_a()
    mm = models()[0.0]["full"]

    runs = {}
    for s in SHOW:
        K = controller_for(s, art, X_HARD, latency=False)
        r = simulate(mm, RPM_REF, AP, controller=K, T=T, x0=X_HARD,
                     moving=False, fs=FS, seed=3, v_max=V_MAX)
        runs[s] = r
        print(f"{s:10s} rmsW={r.rms_w(0.1)*1e6:6.2f}um peakW={r.peak_w(0.1)*1e6:6.1f}um "
              f"peakU={r.peak_u(0.1):6.1f}V clip={r.extras['sat_frac']*100:5.2f}%",
              flush=True)

    fig, axs = plt.subplots(2, 1, figsize=(DC, 4.6), sharex=True)

    # ---- (a) milling-point displacement -------------------------------
    ax = axs[0]
    for s in SHOW:
        r = runs[s]
        m = (r.t >= W0) & (r.t <= W1)
        lbl = (f"{LABELS[s]}: {r.rms_w(0.1)*1e6:.1f} µm RMS")
        ax.plot(r.t[m] * 1e3, r.w_mill[m] * 1e6, color=COLORS[s], lw=0.8,
                label=lbl)
    ax.set_ylabel(r"milling-point $w(x_T)$ [$\mu$m]")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_title(r"(a) displacement --- $x_T$ = 95 mm, $a_p$ = 2 mm, "
                 "4900 rpm (frozen design chatters, both schedules reject)",
                 fontsize=8.5)
    ax.legend(fontsize=6.6, loc="upper right", ncol=1)

    # ---- (b) control voltage ------------------------------------------
    ax = axs[1]
    ax.axhspan(150, 210, color="0.85", alpha=0.7)
    ax.axhspan(-210, -150, color="0.85", alpha=0.7)
    ax.axhline(150, color="k", lw=0.6, ls="--")
    ax.axhline(-150, color="k", lw=0.6, ls="--")
    for s in SHOW:
        r = runs[s]
        m = (r.t >= W0) & (r.t <= W1)
        lbl = (f"{LABELS[s]}: {r.peak_u(0.1):.0f} V peak, "
               f"{r.extras['sat_frac']*100:.2f}% clipped")
        ax.plot(r.t[m] * 1e3, r.u[m], color=COLORS[s], lw=0.8, label=lbl)
    ax.set_ylabel("control voltage [V]")
    ax.set_xlabel("time [ms]")
    ax.set_ylim(-200, 200)
    ax.text(W0 * 1e3 + 0.3, -195, "amplifier saturation $\\pm$150 V",
            fontsize=6.2, va="bottom", color="0.25")
    ax.set_title("(b) control voltage (frozen & naive PS-LPV hit the rail; "
                 "PS-LPV-SA keeps $\\sim$50 V headroom)", fontsize=8.5)
    ax.legend(fontsize=6.6, loc="lower right", ncol=1)

    fig.tight_layout()
    savefig(fig, "fig17_timedomain_codesign")


if __name__ == "__main__":
    main()
