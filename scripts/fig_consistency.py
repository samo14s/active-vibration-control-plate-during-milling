"""Figure 12: consistency of the model with the published measurements
of the reference rig (Du et al., IJMS 274:109257, 2024).

The reference experiments report open-loop stability limits below
0.1 mm at most speeds in the 4.3-6.7 krpm band (higher near 5.5 krpm),
and their calibration treats the effective regenerative coefficient as
alpha_40 = 1.6 * mean with dispersion up to 2.9 * mean. The figure
shows the open-loop critical depth of the present model across their
tested band for the plain-average coefficient (a4_scale = 1.0) and for
their calibrated nominal (1.6) and upper (2.9) scalings, with their
reported experimental range shaded: the model brackets the measured
band once the published coefficient calibration is applied.
"""
import numpy as np

import matplotlib.pyplot as plt
from common import SC, cached, models, savefig

from avc import sld


XS = (0.005, 0.025, 0.05, 0.075, 0.095)


def main():
    mm12 = models(removals=(0.0,), anchor=False)[0.0]["full"]  # raw FEM: first-principles OL validation
    rpms = np.linspace(4300.0, 6700.0, 25)      # the reference tested band

    def build():
        # path-minimum limit: chatter anywhere along the pass ends it,
        # which is the quantity the reference experiments report
        out = {"rpms": rpms}
        for sc_a4 in (1.0, 1.6, 2.9):
            out[sc_a4] = np.array([
                min(sld.critical_depth(mm12, x, float(r), a4_scale=sc_a4,
                                       ap_hi=2e-3) for x in XS)
                for r in rpms])
        import dataclasses
        mm_lo = dataclasses.replace(mm12, zeta=mm12.zeta * 0.8)
        out["1.6_lowdamp"] = np.array([
            min(sld.critical_depth(mm_lo, x, float(r), a4_scale=1.6,
                                   ap_hi=2e-3) for x in XS)
            for r in rpms])
        return out

    curves = cached("consistency_pathmin", build)

    fig, ax = plt.subplots(figsize=(SC, 2.7))
    ax.axhspan(0.0, 0.1, color="#2ca02c", alpha=0.15,
               label="ref. measured band ($<0.1$ mm at 4/5 tested speeds)")
    styles = {1.0: ("-", r"$\bar{\alpha}_4$ (plain average)", "k"),
              1.6: ("-", r"$1.6\,\bar{\alpha}_4$ (ref. calibrated nominal)",
                    "#1f77b4"),
              2.9: ("--", r"$2.9\,\bar{\alpha}_4$ (ref. upper calibration)",
                    "#d62728")}
    for sc_a4, (ls, lbl, c) in styles.items():
        ax.semilogy(rpms / 1e3, curves[sc_a4] * 1e3, ls, color=c, label=lbl)
    ax.semilogy(rpms / 1e3, curves["1.6_lowdamp"] * 1e3, ":",
                color="#1f77b4", alpha=0.8,
                label=r"$1.6\,\bar{\alpha}_4$, $-20\%$ damping")
    for r in (4300, 4900, 5500, 6100, 6700):     # their tested speeds
        ax.axvline(r / 1e3, color="#7f7f7f", lw=0.6, alpha=0.5)
    ax.set_xlabel("spindle speed [krpm]")
    ax.set_ylabel(r"path-minimum open-loop $a_{\rm lim}$ [mm]")
    ax.set_ylim(0.04, 2.0)
    ax.legend(fontsize=5.8, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig12_consistency")

    tested = (4300, 4900, 5500, 6100, 6700)
    for k in (1.0, 1.6, 2.9, "1.6_lowdamp"):
        v = [curves[k][np.argmin(np.abs(rpms - r))] * 1e3 for r in tested]
        print(f"a4_scale {k} at tested speeds:", np.round(v, 3))


if __name__ == "__main__":
    main()
