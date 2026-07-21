"""Saturation-aware co-design selection figure.

Sweeps the control-effort weight multiplier gamma_u / gamma_u0
(gamma_u0 = 1/V_max, the nominal normalization) and reports, at the worst
scheduling position on the 12-mode plant at the production speed:

  (a) worst-position CERTIFIED saturation-free depth and LINEAR (Floquet)
      stability limit vs the weight. The naive normalization (multiplier 1)
      certifies BELOW the frozen H-inf design despite ~3x its linear limit
      -- the saturation inversion. Raising the weight reins in the voltage;
      the linear limit has a broad interior maximum near multiplier 24 that
      also reverses the inversion (certified depth ~1.4x frozen).
  (b) worst-position peak control voltage at a fixed hard cut (a_p = 2 mm):
      the naive design rides the +-150 V rail; the co-design leaves ~50 V
      of headroom, which is exactly the certified margin of panel (a).

The deployed PS-LPV-SA controller sits at the marked multiplier
(pipeline.WU_SA_MULT).
"""
import pathlib
import sys

import numpy as np

import matplotlib.pyplot as plt
from common import COLORS, DC, cached, savefig
from pipeline import (AP_HI, RPM_REF, WU_SA_MULT, X_EVAL, controller_for,
                      models, sa_schedule, stage_a)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from avc.satcert import certified_depth, lift_sampled  # noqa: E402
from avc.simulate import simulate  # noqa: E402
from avc.synthesis import deploy  # noqa: E402

V_MAX = 150.0
H_REQ = 20e-6
X_HARD = 0.095
AP_HARD = 2.0e-3
MULTS = (1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0)


def lin_depth(mm, x, K, ap_hi=AP_HI, tol=2e-5):
    def rho(ap):
        return lift_sampled(mm, x, ap, RPM_REF, K).rho()
    lo, hi = 0.0, ap_hi
    if rho(hi) < 1.0:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if rho(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return lo


def sweep():
    def build():
        mm = models()[0.0]["full"]
        cert, lin, peakV = [], [], []
        for mult in MULTS:
            S = sa_schedule(mult)
            nk = S.deployed_order()
            cw, lw = 1e9, 1e9
            for x in X_EVAL:
                K = deploy(S.at_theta(float(x), 0.0, deployed=False),
                           S.params, latency=False, n_keep=nk)
                ap_c, _ = certified_depth(mm, float(x), RPM_REF, K, V_MAX,
                                          H_REQ, ap_hi=AP_HI)
                cw = min(cw, ap_c)
                lw = min(lw, lin_depth(mm, float(x), K))
            Kh = deploy(S.at_theta(X_HARD, 0.0, deployed=False), S.params,
                        latency=False, n_keep=nk)
            r = simulate(mm, RPM_REF, AP_HARD, controller=Kh, T=0.5,
                         x0=X_HARD, moving=False, fs=250e3, seed=3, v_max=V_MAX)
            cert.append(cw)
            lin.append(lw)
            peakV.append(r.peak_u(0.1))
            print(f"[codesign] x{mult:4.0f}: cert={cw*1e3:.3f} lin={lw*1e3:.3f} "
                  f"peakV={r.peak_u(0.1):.1f}", flush=True)
        # frozen reference (worst position)
        art = stage_a()
        fc, fl = 1e9, 1e9
        for x in X_EVAL:
            Kf = controller_for("FROZEN", art, x, latency=False)
            ap_c, _ = certified_depth(mm, float(x), RPM_REF, Kf, V_MAX,
                                      H_REQ, ap_hi=AP_HI)
            fc = min(fc, ap_c)
            fl = min(fl, lin_depth(mm, float(x), Kf))
        Kfh = controller_for("FROZEN", art, X_HARD, latency=False)
        rf = simulate(mm, RPM_REF, AP_HARD, controller=Kfh, T=0.5, x0=X_HARD,
                      moving=False, fs=250e3, seed=3, v_max=V_MAX)
        return {"mults": np.array(MULTS), "cert": np.array(cert),
                "lin": np.array(lin), "peakV": np.array(peakV),
                "frozen_cert": fc, "frozen_lin": fl,
                "frozen_peakV": rf.peak_u(0.1)}
    return cached("codesign_sweep_v2", build)


def main():
    D = sweep()
    m = D["mults"]
    sac = COLORS["PS-LPV-SA"]
    naive = COLORS["PS-LPV"]
    froz = COLORS["FROZEN"]

    fig, axs = plt.subplots(1, 2, figsize=(DC, 2.9))

    # ---- (a) certified & linear depth vs weight -----------------------
    ax = axs[0]
    ax.semilogx(m, D["lin"] * 1e3, "-s", color=naive, ms=4,
                label="linear limit (Floquet)")
    ax.semilogx(m, D["cert"] * 1e3, "-o", color=sac, ms=4,
                label="certified saturation-free depth")
    ax.axhline(D["frozen_cert"] * 1e3, ls="--", color=froz, lw=1.0,
               label="frozen $H_\\infty$ certified")
    ax.axhline(D["frozen_lin"] * 1e3, ls=":", color=froz, lw=1.0,
               label="frozen $H_\\infty$ linear")
    ax.axvline(WU_SA_MULT, color="0.4", lw=0.8)
    # shade the inversion region (certified below frozen)
    inv = D["cert"] < D["frozen_cert"]
    if inv.any():
        ax.axvspan(m[0], m[inv][-1], color=froz, alpha=0.07)
        ax.text(m[0] * 1.05, 0.30, "naive\ninversion", fontsize=6.5,
                color=froz, va="bottom")
    ax.annotate("deployed\nPS-LPV-SA", xy=(WU_SA_MULT, D["cert"][m == WU_SA_MULT][0] * 1e3),
                xytext=(WU_SA_MULT * 1.1, 2.0), fontsize=6.5, color=sac,
                arrowprops=dict(arrowstyle="->", color=sac, lw=0.7))
    ax.set_xlabel(r"control-effort weight $\gamma_u / \gamma_{u0}$")
    ax.set_ylabel(r"worst-position $a_{\rm lim}$ [mm]")
    ax.set_title("(a) co-design selection", fontsize=8.5)
    ax.set_ylim(0, None)
    ax.legend(fontsize=6, loc="upper left")

    # ---- (b) actuator headroom at the hard cut ------------------------
    ax = axs[1]
    ax.axhspan(150, 170, color="0.85", alpha=0.7)
    ax.axhline(150, color="k", lw=0.7, ls="--")
    ax.semilogx(m, D["peakV"], "-o", color=sac, ms=4,
                label="PS-LPV-SA schedule")
    ax.axhline(D["frozen_peakV"], ls="--", color=froz, lw=1.0,
               label="frozen $H_\\infty$")
    ax.plot(m[0], D["peakV"][0], "s", color=naive, ms=7, mfc="none",
            mew=1.4, label="naive PS-LPV ($\\gamma_{u0}$)")
    ax.axvline(WU_SA_MULT, color="0.4", lw=0.8)
    ax.set_xlabel(r"control-effort weight $\gamma_u / \gamma_{u0}$")
    ax.set_ylabel(r"peak voltage @ $x_T$=95 mm, $a_p$=2 mm [V]")
    ax.set_ylim(40, 165)
    ax.set_title("(b) actuator headroom recovered", fontsize=8.5)
    ax.text(m[0] * 1.1, 152, "amplifier rail $\\pm$150 V", fontsize=6,
            color="0.25", va="bottom")
    ax.legend(fontsize=6, loc="lower left")

    fig.tight_layout()
    savefig(fig, "fig16_codesign")

    i = list(m).index(WU_SA_MULT)
    print(f"\nOperating point gamma_u = {WU_SA_MULT:g}/Vmax:")
    print(f"  certified worst = {D['cert'][i]*1e3:.3f} mm "
          f"({D['cert'][i]/D['frozen_cert']:.2f}x frozen, "
          f"{D['cert'][i]/D['cert'][0]:.2f}x naive)")
    print(f"  linear worst    = {D['lin'][i]*1e3:.3f} mm "
          f"({D['lin'][i]/D['frozen_lin']:.2f}x frozen)")
    print(f"  peak voltage    = {D['peakV'][i]:.1f} V "
          f"(naive {D['peakV'][0]:.1f} V, frozen {D['frozen_peakV']:.1f} V)")


if __name__ == "__main__":
    main()
