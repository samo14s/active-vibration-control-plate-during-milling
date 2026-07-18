"""
make_figures.py
===============
Publication figures generated strictly from the computed results in ``results/``.
No value is hard-coded; every figure loads a JSON/NPZ produced by ``run_all.py``.
Output: 300-dpi PNG + PDF in ``figures/``.

  fig1_sld           OL / LQG / ADRC stability lobes (observer/ESO in the loop)
  fig2_robustness    tip vibration vs modal-frequency drift: LQG vs ADRC
  fig3_scenarios     RMS reduction vs LQG across 4 scenarios: 2-DOF and ADRC
  fig4_feedforward   role of the phase-aware feedforward (does not move stability)
  fig5_authority     feedback-authority vs stability design curve (supplementary)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 110, "savefig.bbox": "tight"})
C_OL, C_LQG, C_2D, C_ADRC = "#7f7f7f", "#2E8B57", "#c62828", "#1f4e8c"
RPM_OP, AP_OP = 4900, 0.3


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print("  wrote", name)


def load(name):
    return json.load(open(os.path.join(RES, name)))


# ---------------------------------------------------------------- Fig 1: SLD
def fig_sld():
    d = np.load(os.path.join(RES, "sld.npz"))
    s = load("sld_summary.json")
    RPM, ap = d["RPM"], d["ap"] * 1e3
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    for key, c in [("rho_OL", C_OL), ("rho_LQG", C_LQG), ("rho_ADRC", C_ADRC)]:
        ax.contour(RPM, ap, d[key], levels=[1.0], colors=c, linewidths=2.5)
    handles = [
        plt.Line2D([0], [0], color=C_OL, lw=2.5,
                   label=f"Open loop  ($a_{{p,crit}}$={s['ap_crit_OL_mm']:.2f} mm)"),
        plt.Line2D([0], [0], color=C_LQG, lw=2.5,
                   label=f"LQG, observer in loop  ({s['ap_crit_LQG_dyn_mm']:.2f} mm)"),
        plt.Line2D([0], [0], color=C_ADRC, lw=2.5,
                   label=f"ADRC, ESO in loop  ({s['ap_crit_ADRC_mm']:.2f} mm)"),
    ]
    ax.plot(RPM_OP, AP_OP, "*", color="gold", ms=17, mec="k", mew=1.1, zorder=5)
    handles.append(plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                              markeredgecolor="k", ms=13, label="operating point"))
    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel(r"Depth of cut $a_p$ (mm)")
    ax.set_title("Controlled stability-lobe diagram (stable region below each curve)")
    ax.set_ylim(0, ap.max())
    ax.legend(handles=handles, loc="upper right", framealpha=0.95, fontsize=9.5)
    save(fig, "fig1_sld")


# -------------------------------------------------- Fig 2: robustness sweep
def fig_robustness():
    r = load("robustness.json")["sweep"]
    df = np.array([x["freq_perturb"] for x in r]) * 100
    lqg = np.array([x["lqg_yrms"] for x in r])
    adrc = np.array([x["adrc_yrms"] for x in r])
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.semilogy(df, lqg, "-o", color=C_LQG, lw=1.8, ms=6, label="LQG (tip sensor + observer)")
    ax.semilogy(df, adrc, "-s", color=C_ADRC, lw=1.8, ms=6, label="ADRC (collocated)")
    # mark the LQG chatter point
    imax = int(np.argmax(lqg))
    if lqg[imax] > 10:
        ax.annotate("LQG chatters\n(loses stability)", xy=(df[imax], lqg[imax]),
                    xytext=(df[imax] + 4, lqg[imax] / 8), fontsize=9, color=C_LQG,
                    arrowprops=dict(arrowstyle="->", color=C_LQG))
    ax.set_xlabel("modal-frequency drift of the real plant (%)")
    ax.set_ylabel(r"tip vibration $y_{RMS}$ ($\mu$m, log scale)")
    ax.set_title("Robustness to varying dynamics: model-based LQG vs model-light ADRC")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, which="both", alpha=0.3)
    save(fig, "fig2_robustness")


# ----------------------------------------------- Fig 3: scenario RMS gains
def fig_scenarios():
    two = load("scenarios.json")["scenarios"]
    adrc = load("adrc.json")["scenarios"]
    names = [r["name"].split()[0] for r in two]
    g2 = [r["rms_gain_pct"] for r in two]
    ga = [r["rms_gain_pct"] for r in adrc]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    b1 = ax.bar(x - w / 2, g2, w, color=C_2D, ec="k", label="2-DOF (feedback + feedforward)")
    b2 = ax.bar(x + w / 2, ga, w, color=C_ADRC, ec="k", label="ADRC")
    for bars, vals in [(b1, g2), (b2, ga)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.1f}%",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("tip RMS reduction vs LQG (%)")
    ax.set_title("Vibration reduction relative to the LQG baseline")
    ax.set_ylim(0, max(ga) * 1.28)
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(loc="upper left", framealpha=0.95)
    save(fig, "fig3_scenarios")


# ------------------------------------------------- Fig 4: feedforward role
def fig_feedforward():
    ff = load("feedforward.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    tr = os.path.join(RES, "traces_S1.npz")
    if os.path.exists(tr):
        d = np.load(tr); t = d["t"] * 1e3
        w = (t >= 200) & (t <= 260)
        rL = np.sqrt(np.mean(d["y_lqg"] ** 2)) * 1e6
        rD = np.sqrt(np.mean(d["y_2d"] ** 2)) * 1e6
        ax.plot(t[w], d["y_lqg"][w] * 1e6, color=C_LQG, lw=1.0,
                label=f"LQG (feedback), RMS={rL:.3f}$\\mu$m")
        ax.plot(t[w], d["y_2d"][w] * 1e6, color=C_2D, lw=1.0,
                label=f"2-DOF (+feedforward), RMS={rD:.3f}$\\mu$m")
        ax.set_xlabel("time (ms)"); ax.set_ylabel(r"$y_p$ ($\mu$m)")
        ax.set_title("(a) S1 tip vibration (steady-state zoom)")
        ax.legend(loc="upper right", fontsize=8.5)
    ax = axes[1]
    fb, fd = ff["feedback_only"], ff["feedback_plus_ff"]
    labels = [r"$y_{RMS}$ ($\mu$m)", r"$y_{peak}$ ($\mu$m)", r"$u_{max}$ (V)"]
    fbv = [fb["y_rms"], fb["y_max"], fb["u_max"]]
    fdv = [fd["y_rms"], fd["y_max"], fd["u_max"]]
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, fbv, w, color=C_LQG, ec="k", label="feedback only")
    ax.bar(x + w / 2, fdv, w, color=C_2D, ec="k", label="+ feedforward")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(f"(b) role of feedforward  ($a_{{p,crit}}$ unchanged = {ff['ap_crit_mm']:.2f} mm)")
    ax.legend(fontsize=9)
    save(fig, "fig4_feedforward")


# ------------------------------------- Fig 5: authority design curve (suppl.)
def fig_authority():
    s = load("synthesis.json")
    cur = s["design_curve"]
    gn = np.array([r["gain_norm"] for r in cur])
    ac = np.array([r["ap_crit_mm"] for r in cur])
    uv = np.array([r["u_max_apeval"] for r in cur])
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(gn, ac, "-o", color="#444", lw=1.6, ms=5,
            label=r"static-feedback $a_{p,crit}(\|K\|)$ (ideal bound)")
    ax.scatter([s["lqg"]["gain_norm"]], [s["lqg"]["ap_crit_mm"]], s=140, marker="s",
               color=C_LQG, ec="k", zorder=5, label="LQG operating point")
    ax.set_xscale("log")
    ax.set_xlabel(r"feedback-gain norm $\|K\|$")
    ax.set_ylabel(r"static $a_{p,crit}$ (mm)", color="#444")
    ax.set_title("Feedback authority vs stability (static bound) and voltage cost")
    ax2 = ax.twinx()
    ax2.plot(gn, uv, "--^", color=C_2D, lw=1.3, ms=5,
             label=f"peak voltage at $a_p$={s['ap_eval_mm']:.1f} mm")
    ax2.axhline(s["u_budget"], color=C_2D, ls=":", lw=1.2)
    ax2.set_ylim(0, 165)
    ax2.text(gn.min(), s["u_budget"] + 3, f"{s['u_budget']:.0f} V budget", color=C_2D, fontsize=9)
    ax2.set_ylabel("peak actuator voltage (V)", color=C_2D)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8.5, framealpha=0.95)
    save(fig, "fig5_authority")


# ---------------------------------- Fig 6: placement co-design / metric inversion
def fig_placement():
    d = load("placement.json")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    # (a) plate map: patch positions coloured by class
    ax = axes[0]
    LPmm, HPmm = 100.0, 80.0
    pw, ph = d["patch_size_mm"]
    ax.add_patch(plt.Rectangle((0, 0), LPmm, HPmm, fc="#f2f2f2", ec="k", lw=1.5))
    ax.plot([0, LPmm], [HPmm, HPmm], color="#c62828", lw=3, alpha=0.6)
    ax.text(50, HPmm + 2.2, "tool path (top edge)", ha="center", fontsize=9,
            color="#c62828")
    ax.plot([0, LPmm], [0, 0], color="k", lw=4)
    ax.text(50, -5.5, "clamped edge", ha="center", fontsize=9)
    feas = {(round(f["px"] * 1e3), round(f["pz"] * 1e3)): f for f in d["feasible"]}
    for r in d["linear_map"]:
        x, z = r["px"] * 1e3, r["pz"] * 1e3
        key = (round(x), round(z))
        viable = r["signs"].startswith("--") and r["ap_crit_linear_mm"] > 0.1
        if not viable:
            ax.add_patch(plt.Rectangle((x, z), pw, ph, fill=False,
                                       ec="#bbbbbb", lw=1.0, ls=":"))
            ax.plot([x, x + pw], [z, z + ph], color="#bbbbbb", lw=0.8)
            ax.plot([x, x + pw], [z + ph, z], color="#bbbbbb", lw=0.8)
        else:
            f = feas.get(key)
            ax.add_patch(plt.Rectangle((x, z), pw, ph, fill=False,
                                       ec=C_ADRC, lw=1.8))
            if f:
                ax.text(x + pw / 2, z + ph / 2,
                        f"lin {f['ap_crit_linear_mm']:.1f}\nfeas {f['ap_feasible_mm']:.2f}",
                        ha="center", va="center", fontsize=8, color=C_ADRC,
                        fontweight="bold")
    ax.set_xlim(-4, 104); ax.set_ylim(-9, 90)
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    ax.set_title("(a) patch candidates\n(boxed = viable, crossed = sign-infeasible)", fontsize=10)
    # (b) inversion: linear vs feasible ranking
    ax = axes[1]
    fs = sorted(d["feasible"], key=lambda r: -r["ap_crit_linear_mm"])
    fs = [r for r in fs if r["ap_feasible_mm"] > 0.05]
    labels = [f"({r['px']*1e3:.0f},{r['pz']*1e3:.0f})" for r in fs]
    lin = [r["ap_crit_linear_mm"] for r in fs]
    fea = [r["ap_feasible_mm"] for r in fs]
    x = np.arange(len(fs)); w = 0.38
    ax.bar(x - w / 2, lin, w, color="#9e9e9e", ec="k", label="linear CL-SD boundary")
    ax.bar(x + w / 2, fea, w, color=C_ADRC, ec="k", label="voltage-feasible (150 V)")
    for xi, v in zip(x - w / 2, lin):
        ax.text(xi, v + 0.12, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, fea):
        ax.text(xi, v + 0.12, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(d["lqg_feasible_mm"], color=C_LQG, ls="--", lw=1.5)
    ax.text(len(fs) - 0.4, d["lqg_feasible_mm"] + 0.15,
            f"LQG feasible {d['lqg_feasible_mm']:.2f} mm", color=C_LQG,
            fontsize=8.5, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("patch position (x0, z0) mm — sorted by linear boundary")
    ax.set_ylabel(r"critical depth $a_p$ (mm)")
    ax.set_title("(b) metric inversion:\nlinear vs voltage-feasible ranking", fontsize=10)
    ax.legend(fontsize=9)
    save(fig, "fig6_placement")


if __name__ == "__main__":
    reg = [("sld.npz", fig_sld), ("robustness.json", fig_robustness),
           ("adrc.json", fig_scenarios), ("feedforward.json", fig_feedforward),
           ("synthesis.json", fig_authority), ("placement.json", fig_placement)]
    for f, fn in reg:
        if os.path.exists(os.path.join(RES, f)):
            fn()
    print("figures/ done.")
