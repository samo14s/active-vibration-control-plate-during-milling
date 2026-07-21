"""Validation of the classical modal (Warburton) model + Altintas-Budak
analytical stability lobes against the measured EMA and the Mindlin FEM.

(a) modal frequencies: classical (measured-anchored) vs FEM vs measured;
(b) voltage->sensor FRF: classical vs FEM;
(c) open-loop stability lobes at x=50 mm: classical Altintas-Budak
    zero-order vs semi-discretization (both on the classical model).
Output: paper/figures/fig18_classical_validation.{pdf,png}
"""
import numpy as np

import matplotlib.pyplot as plt
from common import DC, savefig
from pipeline import RPM_REF
from avc.params import DEFAULT
from avc.classical_plate import reduce_model_classical
from avc.modal import reduce_model
from avc import sld, analytical_lobes as al


def main():
    cl = reduce_model_classical(DEFAULT, 12, 0.0)
    fe = reduce_model(DEFAULT, 12, 0.0, anchor=False)   # raw first-principles FEM
    meas = np.array(DEFAULT.plate.f_measured)

    print("mode |  measured |  classical |   FEM")
    for i in range(5):
        print(f"  {i+1}  |  {meas[i]:7.1f}  |  {cl.freqs_hz[i]:8.1f}  | "
              f"{fe.freqs_hz[i]:7.1f}")

    fig, axs = plt.subplots(1, 3, figsize=(DC, 2.7))

    # (a) frequency comparison
    ax = axs[0]
    idx = np.arange(5)
    ax.bar(idx - 0.25, meas, 0.25, label="measured (EMA)", color="k")
    ax.bar(idx, cl.freqs_hz[:5], 0.25, label="classical (Warburton)",
           color="#1f77b4")
    ax.bar(idx + 0.25, fe.freqs_hz[:5], 0.25, label="Mindlin FEM",
           color="#d62728", alpha=0.8)
    ax.set_xticks(idx, [f"{i+1}" for i in idx])
    ax.set_xlabel("mode number")
    ax.set_ylabel("natural frequency [Hz]")
    ax.set_title("(a) modal frequencies", fontsize=8.5)
    ax.legend(fontsize=6)

    # (b) FRF voltage -> sensor
    ax = axs[1]
    f = np.linspace(200, 4500, 4000)
    Hc = cl.frf_voltage(f, DEFAULT.sensor.xs, DEFAULT.sensor.zs)
    He = fe.frf_voltage(f, DEFAULT.sensor.xs, DEFAULT.sensor.zs)
    ax.semilogy(f, np.abs(Hc) * 1e6, color="#1f77b4", lw=1.1,
                label="classical")
    ax.semilogy(f, np.abs(He) * 1e6, color="#d62728", lw=1.0, ls="--",
                label="Mindlin FEM")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$|w_s/u|$ [$\mu$m/V]")
    ax.set_title("(b) actuator FRF (at sensor)", fontsize=8.5)
    ax.legend(fontsize=6.5)

    # (c) open-loop SLD at x=50 mm: analytical vs semi-discretization
    ax = axs[2]
    rpms = np.linspace(2000, 10000, 200)
    env = al.lobe_envelope(cl, 0.05, rpms) * 1e3
    rpm_sd = np.linspace(2000, 10000, 60)
    sdc = np.array([sld.critical_depth(cl, 0.05, float(r), m=48, ap_hi=5e-3)
                    for r in rpm_sd]) * 1e3
    ax.plot(rpms / 1e3, env, color="#1f77b4", lw=1.1,
            label="Altintas--Budak (ZOA)")
    ax.plot(rpm_sd / 1e3, sdc, color="#2ca02c", lw=1.0, ls="--",
            label="semi-discretization")
    ax.axvline(RPM_REF / 1e3, color="0.6", lw=0.8)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("spindle speed [krpm]")
    ax.set_ylabel(r"$a_{\rm lim}$ [mm] ($x_T$=50 mm)")
    ax.set_title("(c) open-loop lobes (classical model)", fontsize=8.5)
    ax.legend(fontsize=6.5)

    fig.tight_layout()
    savefig(fig, "fig18_classical_validation")


if __name__ == "__main__":
    main()
