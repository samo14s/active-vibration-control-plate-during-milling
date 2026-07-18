"""
make_figures.py
===============
Publication figures generated strictly from the computed results in ``results/``.
No value is hard-coded; every figure loads a JSON/NPZ produced by
``run_all.py``.  Output: 300-dpi PNG + PDF in ``figures/``.
"""
import os, sys, json
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
C_OL, C_LQG, C_FLO, C_2D = "#7f7f7f", "#2E8B57", "#1f4e8c", "#c62828"
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
    RPM, ap = d["RPM"], d["ap"] * 1e3
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for rho, c in [("rho_OL", C_OL), ("rho_LQG", C_LQG), ("rho_FLO", C_FLO)]:
        ax.contour(RPM, ap, d[rho], levels=[1.0], colors=c, linewidths=2.4)
    # legend proxies
    handles = [plt.Line2D([0], [0], color=C_OL, lw=2.4, label="Open loop"),
               plt.Line2D([0], [0], color=C_LQG, lw=2.4, label="LQG (CL-FDM)"),
               plt.Line2D([0], [0], color=C_FLO, lw=2.4,
                          label="Voltage-budget design (CL-FDM)")]
    ax.plot(RPM_OP, AP_OP, "*", color="gold", ms=18, mec="k", mew=1.2,
            label="operating point", zorder=5)
    handles.append(plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                              markeredgecolor="k", ms=13, label="operating point"))
    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel(r"Depth of cut $a_p$ (mm)")
    ax.set_title("Controlled stability-lobe diagram (stable region below each curve)")
    ax.set_ylim(0, ap.max())
    ax.legend(handles=handles, loc="upper right", framealpha=0.95)
    save(fig, "fig1_sld")


# --------------------------------------- Fig 2: authority design curve + voltage
def fig_pareto():
    s = load("synthesis.json")
    cur = s["design_curve"]
    gn = np.array([r["gain_norm"] for r in cur])
    ac = np.array([r["ap_crit_mm"] for r in cur])
    uv = np.array([r["u_max_apeval"] for r in cur])
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(gn, ac, "-o", color="#444", lw=1.6, ms=5, zorder=3,
            label=r"CL-FDM design curve $a_{p,\mathrm{crit}}(\|K\|)$")
    ax.scatter([s["lqg"]["gain_norm"]], [s["lqg"]["ap_crit_mm"]], s=150, marker="s",
               color=C_LQG, ec="k", zorder=5,
               label=f"LQG baseline: {s['lqg']['ap_crit_mm']:.2f} mm @ "
                     f"{s['lqg']['u_max_apeval']:.0f} V")
    ax.scatter([s["selected"]["gain_norm"]], [s["selected"]["ap_crit_mm"]], s=210,
               marker="*", color=C_FLO, ec="k", zorder=6,
               label=f"voltage-budget pick: {s['selected']['ap_crit_mm']:.2f} mm @ "
                     f"{s['selected']['u_max_apeval']:.0f} V")
    ax.set_xscale("log")
    ax.set_xlabel(r"Feedback-gain norm $\|K\|$")
    ax.set_ylabel(r"Critical depth $a_{p,\mathrm{crit}}$ (mm)", color="#444")
    ax.set_ylim(2.1, 5.9)
    ax.set_title("Feedback authority vs chatter stability (CL-FDM) and voltage cost")
    ax2 = ax.twinx()
    ax2.plot(gn, uv, "--^", color=C_2D, lw=1.3, ms=5, zorder=2,
             label=f"peak voltage at $a_p$={s['ap_eval_mm']:.1f} mm")
    ax2.axhline(s["u_budget"], color=C_2D, ls=":", lw=1.2)
    ax2.set_ylim(0, 165)                       # voltage curve stays low; budget line on top
    ax2.text(gn.min(), s["u_budget"] + 3, f"{s['u_budget']:.0f} V budget (never reached)",
             color=C_2D, fontsize=9)
    ax2.set_ylabel("peak actuator voltage (V)", color=C_2D)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8.5, framealpha=0.95)
    save(fig, "fig2_authority_curve")


# ----------------------------------------------------- Fig 3: scenario gains
def fig_scenarios():
    s = load("scenarios.json")["scenarios"]
    names = [r["name"].split()[0] for r in s]
    rms = [r["rms_gain_pct"] for r in s]
    peak = [r["peak_gain_pct"] for r in s]
    x = np.arange(len(s)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(x - w / 2, rms, w, color=C_2D, ec="k", label="RMS reduction")
    ax.bar(x + w / 2, peak, w, color="#ef9a9a", ec="k", label="peak reduction")
    for xi, v in zip(x - w / 2, rms):
        ax.text(xi, v + 0.1, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    for xi, v in zip(x + w / 2, peak):
        ax.text(xi, v + 0.1, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("vibration reduction vs LQG (%)")
    ax.set_title("Phase-aware feedforward: measured reduction (2-DOF vs LQG)")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend()
    save(fig, "fig3_scenarios")


# ------------------------------------------------- Fig 4: feedforward role + traces
def fig_feedforward():
    ff = load("feedforward.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    # (a) steady-state zoom of S1 tip vibration (dense oscillation is unreadable
    #     over the full 500 ms, so show a representative ~60 ms window)
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
    # (b) role bars
    ax = axes[1]
    fb, fd = ff["feedback_only"], ff["feedback_plus_ff"]
    labels = [r"$y_{RMS}$ ($\mu$m)", r"$y_{peak}$ ($\mu$m)", r"$u_{max}$ (V)"]
    fbv = [fb["y_rms"], fb["y_max"], fb["u_max"]]
    fdv = [fd["y_rms"], fd["y_max"], fd["u_max"]]
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, fbv, w, color=C_LQG, ec="k", label="feedback only")
    ax.bar(x + w / 2, fdv, w, color=C_2D, ec="k", label="+ feedforward")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(f"(b) role of feedforward  (a_p,crit unchanged = {ff['ap_crit_mm']:.2f} mm)")
    ax.legend(fontsize=9)
    save(fig, "fig4_feedforward")


if __name__ == "__main__":
    if os.path.exists(os.path.join(RES, "sld.npz")):
        fig_sld()
    if os.path.exists(os.path.join(RES, "synthesis.json")):
        fig_pareto()
    if os.path.exists(os.path.join(RES, "scenarios.json")):
        fig_scenarios()
    if os.path.exists(os.path.join(RES, "feedforward.json")):
        fig_feedforward()
    print("figures/ done.")
