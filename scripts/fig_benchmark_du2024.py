"""Direct benchmark against the reference published paper
Du et al., IJMS 274 (2024) 109257.

(a) actuator voltage-to-displacement FRF: present model (absolute, no fit)
    versus the digitized measured curve (their Fig. 12b);
(b) open-loop stable depth across their five tested spindle speeds: the
    present model's path-minimum critical depth at the reference's
    calibrated regenerative-coefficient scalings, versus their measured
    band (< 0.1 mm at four of five speeds, elevated at 5500 rpm);
(c) stable depth of cut benchmark: their measured open loop and their
    non-scheduled robust controller (0.6-0.8 mm), against the present
    best-frozen and scheduled (PS-LPV) designs on the same hardware.

Output: paper/figures/fig20_benchmark_du2024.{pdf,png}
"""
import pathlib

import numpy as np

import matplotlib.pyplot as plt
from common import COLORS, DC, savefig
from pipeline import X_EVAL, controller_for, models, stage_a
from avc import sld

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEEDS = (4300.0, 4900.0, 5500.0, 6100.0, 6700.0)   # ref. tested band


def path_min(mm, rpm, scale):
    return min(sld.critical_depth(mm, x, float(rpm), a4_scale=scale, m=40,
                                  ap_hi=5e-3) for x in X_EVAL)


def main():
    art = stage_a()
    mm = models()[0.0]["full"]                 # working (anchored) model
    from avc.params import DEFAULT
    from avc.modal import reduce_model
    mm_raw = reduce_model(DEFAULT, 12, 0.0, anchor=False)  # first-principles

    fig, axs = plt.subplots(1, 3, figsize=(DC, 2.9))

    # ---- (a) actuator FRF: model vs digitized measured ----------------
    ax = axs[0]
    d = np.genfromtxt(ROOT / "data" / "measured_frf_voltage_to_displacement.csv",
                      delimiter=",", names=True)
    f = np.linspace(20, 5000, 3000)
    H = mm.frf_voltage(f, DEFAULT.sensor.xs, DEFAULT.sensor.zs)
    # digitized magnitude is dB re 0.01 um/V (matches avc fig13 convention)
    meas_umV = 10.0 ** (d["mag_db"] / 20.0) * 0.01
    ax.semilogy(d["freq_hz"], meas_umV, ".", ms=3, color="k",
                label="measured (Du et al.\\ Fig.~12b)")
    ax.semilogy(f, np.abs(H) * 1e6, color=COLORS["PS-LPV"], lw=1.1,
                label="present model (no fit)")
    print(f"DC gain um/V:  measured={meas_umV[0]:.3f}  "
          f"model={abs(H[0])*1e6:.3f}")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$|w/u|$ [$\mu$m/V]")
    ax.set_title("(a) actuator FRF vs measured", fontsize=8.5)
    ax.legend(fontsize=6.2, loc="lower left")

    # ---- (b) open-loop stable depth across the tested speeds ----------
    ax = axs[1]
    for sc, c, mk in ((1.6, "#4c72b0", "o"), (2.9, "#c44e52", "s")):
        vals = [path_min(mm_raw, r, sc) * 1e3 for r in SPEEDS]
        ax.plot(np.array(SPEEDS) / 1e3, vals, marker=mk, ms=4, color=c,
                label=rf"model, ${sc}\,\bar\alpha_4$")
        print(f"path-min @{sc}a4:", [round(v, 3) for v in vals])
    ax.axhspan(0.0, 0.1, color="0.8", alpha=0.6)
    ax.text(4.35, 0.104, "measured $<$0.1 mm (4 of 5 speeds)", fontsize=6,
            color="0.3", va="bottom")
    ax.axvline(5.5, color="0.6", lw=0.8, ls=":")
    ax.text(5.52, 0.42, "5500 rpm\nanomaly", fontsize=6, color="0.3")
    ax.set_xlabel("spindle speed [krpm]")
    ax.set_ylabel(r"path-min $a_{\rm lim}$ [mm]")
    ax.set_ylim(0, 0.55)
    ax.set_title("(b) open-loop limit vs measured", fontsize=8.5)
    ax.legend(fontsize=6.2, loc="upper left")

    # ---- (c) stable depth benchmark bars ------------------------------
    ax = axs[2]
    froz = min(sld.critical_depth(mm, x, 4900.0,
                                  controller=controller_for("FROZEN", art, x,
                                                             latency=True),
                                  m=40, ap_hi=5e-3) for x in X_EVAL) * 1e3
    pslpv = min(sld.critical_depth(mm, x, 4900.0,
                                   controller=controller_for("PS-LPV", art, x,
                                                              latency=True),
                                   m=40, ap_hi=5e-3) for x in X_EVAL) * 1e3
    labels = ["measured\nopen loop", "Du et al.\nrobust ctrl.",
              "this work\nbest frozen", "this work\nPS-LPV"]
    vals = [0.1, 0.7, froz, pslpv]
    errs = [[0, 0.1, 0, 0], [0, 0.1, 0, 0]]     # 0.6-0.8 band on Du robust
    cols = ["0.55", "#8c8c8c", COLORS["FROZEN"], COLORS["PS-LPV"]]
    bars = ax.bar(range(4), vals, 0.6, color=cols, yerr=errs, capsize=3,
                  ecolor="0.3")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, f"{v:.2f}" if i > 1 else
                ("$<$0.1" if i == 0 else "0.6--0.8"), ha="center",
                fontsize=6.6)
    ax.set_xticks(range(4), labels, fontsize=6.3)
    ax.set_ylabel(r"stable depth $a_{\rm lim}$ [mm]")
    ax.set_ylim(0, 3.4)
    ax.set_title("(c) stable depth benchmark", fontsize=8.5)
    print(f"benchmark: frozen={froz:.3f} PS-LPV={pslpv:.3f} mm")

    fig.tight_layout()
    savefig(fig, "fig20_benchmark_du2024")


if __name__ == "__main__":
    main()
