"""Figure: synthesis framework evidence.

(a) synthesis weights and the fitted additive-uncertainty overbound Wa
    vs the actual truncation residual;
(b) deployed controller FRFs at three scheduling positions;
(c) deployment certificates per position: small-gain spillover margin
    rs and exact sampled-loop radius (1 - rho, log).
"""
import numpy as np

import matplotlib.pyplot as plt
from common import DC, savefig
from pipeline import (P, X_EVAL, models, stage_a)

from avc.synthesis import (_sensor_frf, _siso_frf, _truncation_residual,
                           deploy, make_weights, rs_margin, sampled_rho)


def ctrl_frf(K, freqs):
    return _siso_frf(K.Ak, K.Bk, K.Ck, freqs, D=K.Dk)


def main():
    art = stage_a()
    S = art["sched"]
    Wa = art["Wa"]
    w = make_weights(P)
    freqs = np.geomspace(60, 2.0e4, 800)

    fig, axs = plt.subplots(1, 3, figsize=(DC, 2.5))

    ax = axs[0]
    theta_all = [(x, r) for x in S.x_grid for r in S.removal_grid]
    E = _truncation_residual(P, P.design.n_modes_design,
                             P.design.n_modes_full, theta_all, freqs)
    Wam = np.abs(_siso_frf(Wa[0], Wa[1], Wa[2], freqs, D=Wa[3]))
    ax.loglog(freqs, E * 1e6, "k", label=r"residual $\max_\theta|G_{12}-G_5|$")
    ax.loglog(freqs, Wam * 1e6, "--", color="#d62728", label=r"$W_a$ (fit)")
    Wfm = np.abs(_siso_frf(*w["Wf"][:3], freqs, D=w["Wf"][3]))
    Wum = np.abs(_siso_frf(*w["Wu"][:3], freqs, D=w["Wu"][3]))
    Wnm = np.abs(_siso_frf(*w["Wn"][:3], freqs, D=w["Wn"][3]))
    ax2 = ax.twinx()
    ax2.loglog(freqs, Wfm / Wfm.max(), ":", color="#1f77b4", alpha=0.8,
               label=r"$W_f$ (norm.)")
    ax2.loglog(freqs, Wum / Wum.max(), ":", color="#2ca02c", alpha=0.8,
               label=r"$W_u$ (norm.)")
    ax2.loglog(freqs, Wnm / Wnm.max(), ":", color="#9467bd", alpha=0.8,
               label=r"$W_n$ (norm.)")
    ax2.set_ylim(1e-3, 2)
    ax2.set_yticks([])
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"magnitude [$\mu$m/V]")
    ax.set_title("(a) uncertainty overbound and weights", fontsize=8.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower right")

    ax = axs[1]
    for x, c in ((0.005, "#1f77b4"), (0.05, "#2ca02c"), (0.095, "#d62728")):
        K = S.at_theta(x, 0.0)
        ax.loglog(freqs, np.abs(ctrl_frf(K, freqs)) * 1e-6,
                  color=c, label=f"$x_T$ = {x*1e3:.0f} mm")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$|K|$ [V/$\mu$m]")
    ax.set_title("(b) deployed scheduled controller", fontsize=8.5)
    ax.legend(fontsize=6.5)

    ax = axs[2]
    theta_all = [(x, 0.0) for x in X_EVAL]
    mm12 = models()[0.0]["full"]
    rs = [rs_margin(P, S.at_theta(x, 0.0), Wa, theta_all) for x in X_EVAL]
    sr = [1.0 - sampled_rho(P, S.at_theta(x, 0.0), mm12) for x in X_EVAL]
    xs_mm = np.array(X_EVAL) * 1e3
    ax.semilogy(xs_mm, rs, "-o", ms=4, color="#1f77b4",
                label=r"$\|W_a K S\|_\infty$ (< 1)")
    ax.semilogy(xs_mm, sr, "-s", ms=4, color="#2ca02c",
                label=r"$1-\rho_d$ (> 0)")
    ax.axhline(1.0, color="#d62728", lw=0.8, ls="--")
    ax.set_xlabel(r"tool position $x_T$ [mm]")
    ax.set_ylabel("certificate value")
    ax.set_title("(c) deployment certificates", fontsize=8.5)
    ax.legend(fontsize=6.5)

    fig.tight_layout()
    savefig(fig, "fig03_design_framework")


if __name__ == "__main__":
    main()
