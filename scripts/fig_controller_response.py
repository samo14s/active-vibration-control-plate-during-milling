"""Controller time-domain response at the end-of-pass position
(x_T = 95 mm, a_p = 0.4 mm, 4900 rpm): open loop develops regenerative
chatter, the scheduled controller holds the milling point steady, and a
surface-step disturbance injected mid-run shows the controller's dynamic
rejection and its (modest) control voltage.

Output: paper/figures/fig19_controller_response.{pdf,png}
"""
import numpy as np

import matplotlib.pyplot as plt
from common import COLORS, DC, LABELS, savefig
from pipeline import RPM_REF, controller_for, models, stage_a
from avc.simulate import simulate

X = 0.095
AP = 0.4e-3
FS = 200e3
T = 0.30
T_INJ = 0.16          # surface-step disturbance time [s]
H_STEP = -20e-6       # step height [m] (negative = chip-thickening, binding)


def main():
    art = stage_a()
    mm = models()[0.0]["full"]

    r_ol = simulate(mm, RPM_REF, AP, controller=None, T=T, x0=X,
                    moving=False, fs=FS, seed=1, v_max=150.0)
    Kps = controller_for("PS-LPV", art, X, latency=False)
    r_cl = simulate(mm, RPM_REF, AP, controller=Kps, T=T, x0=X, moving=False,
                    fs=FS, seed=1, v_max=150.0, surf_step=(T_INJ, H_STEP))

    ol_pk = np.max(np.abs(r_ol.w_mill)) * 1e6
    cl_pk = np.max(np.abs(r_cl.w_mill)) * 1e6
    print(f"open loop : rms={r_ol.rms_w(0.05)*1e6:.1f} um, peak={ol_pk:.1f} um")
    print(f"PS-LPV    : rms={r_cl.rms_w(0.05)*1e6:.2f} um, peak={cl_pk:.1f} um, "
          f"peak V={np.max(np.abs(r_cl.u)):.0f}")

    fig, axs = plt.subplots(2, 1, figsize=(DC, 4.3), sharex=True)

    # (a) deflection: open loop vs controller (same axis)
    ax = axs[0]
    ax.plot(r_ol.t * 1e3, r_ol.w_mill * 1e6, color=COLORS["OL"], lw=0.5,
            label=f"open loop (no control): {r_ol.rms_w(0.05)*1e6:.0f} "
                  r"$\mu$m RMS chatter")
    ax.plot(r_cl.t * 1e3, r_cl.w_mill * 1e6, color=COLORS["PS-LPV"], lw=0.7,
            label=f"{LABELS['PS-LPV']}: {r_cl.rms_w(0.05)*1e6:.1f} "
                  r"$\mu$m RMS")
    ax.axvline(T_INJ * 1e3, color="0.45", lw=0.8, ls=":")
    ax.annotate(f"surface-step disturbance\n({H_STEP*1e6:.0f} $\\mu$m, "
                "one tooth period)", xy=(T_INJ * 1e3, H_STEP * 1e6),
                xytext=(T_INJ * 1e3 + 12, -34), fontsize=6.4, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    ax.set_ylabel(r"milling-point $w(x_T)$ [$\mu$m]")
    ax.axhline(0, color="0.6", lw=0.4)
    ax.set_title(r"(a) $x_T$ = 95 mm, $a_p$ = 0.4 mm, 4900 rpm: chatter "
                 "suppression and disturbance rejection", fontsize=8.5)
    ax.legend(fontsize=6.6, loc="upper right")

    # (b) control voltage
    ax = axs[1]
    ax.axhspan(150, 175, color="0.85", alpha=0.6)
    ax.axhspan(-175, -150, color="0.85", alpha=0.6)
    ax.axhline(150, color="k", lw=0.5, ls="--")
    ax.axhline(-150, color="k", lw=0.5, ls="--")
    ax.plot(r_cl.t * 1e3, r_cl.u, color=COLORS["PS-LPV"], lw=0.6)
    ax.axvline(T_INJ * 1e3, color="0.45", lw=0.8, ls=":")
    ax.text(3, 138, f"peak {np.max(np.abs(r_cl.u)):.0f} V of the "
            r"$\pm$150 V rail", fontsize=6.5, color="0.25", va="top")
    ax.set_ylabel("control voltage [V]")
    ax.set_xlabel("time [ms]")
    ax.set_ylim(-165, 165)
    ax.set_title("(b) control voltage (the controller's response)",
                 fontsize=8.5)

    fig.tight_layout()
    savefig(fig, "fig19_controller_response")


if __name__ == "__main__":
    main()
