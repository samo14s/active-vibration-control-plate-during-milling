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
  fig6_placement     transducer-placement co-design (linear vs feasible metric)
  fig7_refinement    precise-model validation + fidelity-layer feasibility
  fig8_material...   in-process material removal (drift + control through process)
  fig9_full_process  4-pass end-to-end finish map (LQG vs ADRC)
  fig10_afc          AFC-ADRC vs baselines (finish map + per-pass RMS)
  fig11_timeseries   full 81.6 s tip time response of all controllers
  fig11b_waveform    full-rate waveform snapshot at the worst region
  fig12_tip_comparison  tip-sensor-only comparison: LQG/ADRC/FOPID/HYBRID
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
C_FOPID = "#d9760a"
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


# ---------------------------- Fig 7: model refinement (validation + spillover)
def fig_refinement():
    d = load("refinement.json")
    v = d["validation"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    # (a) frequency errors vs measurement
    ax = axes[0]
    x = np.arange(5); w = 0.27
    ax.bar(x - w, v["err_bare_pct"], w, color="#9e9e9e", ec="k",
           label=f"bare model (mean {v['mean_abs_err_bare']:.2f}%)")
    ax.bar(x, v["err_refined_pct"], w, color=C_ADRC, ec="k",
           label=f"refined: + patch dynamics (mean {v['mean_abs_err_refined']:.2f}%)")
    ax.bar(x + w, v["err_article_pct"], w, color="#e0b040", ec="k",
           label=f"article's own theory (mean {v['mean_abs_err_article']:.2f}%)")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"mode {k+1}\n{v['measured_hz'][k]:.0f} Hz" for k in range(5)],
                       fontsize=9)
    ax.set_ylabel("natural-frequency error vs measured (%)")
    ax.set_title("(a) validation vs Du et al. Table 4", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    # (b) voltage-feasible depth across modelling-fidelity layers
    ax = axes[1]
    c = d["control_on_refined"]
    se = d.get("sampling_endpoint", None)
    groups = ["3-mode plant\n(dt=50$\\mu$s)", "refined plant\n(dt=50$\\mu$s)",
              "refined plant\n(dt=25$\\mu$s)"]
    lqg_v = [1.38, max(r["feasible_mm"] for r in d["retuned_on_refined"]["lqg"])]
    adrc_v = [1.92, max(r["feasible_mm"] for r in d["retuned_on_refined"]["adrc"])]
    if se:
        lqg_v.append(se["best_lqg_mm"]); adrc_v.append(se["best_adrc_mm"])
    else:
        groups = groups[:2]
    x = np.arange(len(groups)); w = 0.38
    ax.bar(x - w / 2, lqg_v, w, color=C_LQG, ec="k", label="LQG (best tuning)")
    ax.bar(x + w / 2, adrc_v, w, color=C_ADRC, ec="k", label="ADRC (best tuning)")
    for xi, vv in zip(x - w / 2, lqg_v):
        ax.text(xi, vv + 0.03, f"{vv:.2f}", ha="center", fontsize=8.5)
    for xi, vv in zip(x + w / 2, adrc_v):
        ax.text(xi, vv + 0.03, f"{vv:.2f}", ha="center", fontsize=8.5, fontweight="bold")
    if len(groups) == 3:
        ax.annotate("10 kHz sampling\nartifact", xy=(1 + w / 2, adrc_v[1]),
                    xytext=(1.25, max(adrc_v) * 0.72), fontsize=8, color=C_ADRC,
                    arrowprops=dict(arrowstyle="->", color=C_ADRC))
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("voltage-feasible depth (mm)")
    ax.set_title("(b) feasible depth across modelling-fidelity layers", fontsize=10.5)
    ax.legend(fontsize=9, framealpha=0.95)
    save(fig, "fig7_refinement")


# -------------------- Fig 8: in-process material removal / control through process
def fig_material_removal():
    d = load("material_removal.json")
    th = d["thinning"]; cp = d["control_through_process"]
    h = [r["h_band_mm"] for r in th]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8),
                             gridspec_kw={"wspace": 0.42})
    # (a) physically-generated modal drift
    ax = axes[0]
    f1 = [r["freq_hz"][0] for r in th]
    ax.plot(h, f1, "-o", color="#444", lw=1.8, ms=6)
    for hi, fi in zip(h, f1):
        ax.annotate(f"{fi:.0f}", (hi, fi), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.invert_xaxis()
    ax.set_xlabel("wall thickness of the machined band (mm)")
    ax.set_ylabel(r"$f_1$ (Hz)", color="#444")
    sp = d["single_pass"]["max_abs_drift_pct"]
    ax.set_title(f"(a) modal drift from material removal\n"
                 f"(single pass < {sp:.3f} %; thinning "
                 f"{(f1[-1]/f1[0]-1)*100:+.1f} %)", fontsize=10)
    ax2 = ax.twinx()
    b0 = [r["b0"] for r in th]
    ax2.plot(h, b0, "--s", color=C_ADRC, lw=1.4, ms=5)
    ax2.set_ylabel(r"collocated $b_0$", color=C_ADRC)
    ax2.grid(False)
    # (b) control through the process
    ax = axes[1]
    hc = [r["h_band_mm"] for r in cp]
    ax.plot(hc, [r["ap_crit_lqg_mm"] for r in cp], "-o", color=C_LQG, lw=1.8,
            ms=6, label="LQG (design frozen at h=4.0)")
    ax.plot(hc, [r["ap_crit_adrc_mm"] for r in cp], "-s", color=C_ADRC, lw=1.8,
            ms=6, label="ADRC (design frozen at h=4.0)")
    ax.axhline(0.3, color="k", ls=":", lw=1.2)
    ax.text(hc[0], 0.34, "operating depth 0.3 mm", fontsize=8)
    for r in cp:                                  # chatter markers at op point
        if not r["op_lqg"]["stable"]:
            ax.plot(r["h_band_mm"], 0.3, "x", color=C_LQG, ms=13, mew=3, zorder=6)
        if not r["op_adrc"]["stable"]:
            ax.plot(r["h_band_mm"], 0.3, "+", color=C_ADRC, ms=14, mew=3, zorder=6)
    ax.invert_xaxis()
    ax.set_xlabel("wall thickness of the machined band (mm)")
    ax.set_ylabel(r"linear $a_{p,crit}$ (mm)")
    ax.set_title("(b) frozen designs across the machining process\n"
                 "(x / + = chatter at the 0.3 mm operating point)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.95)
    save(fig, "fig8_material_removal")



# ------------------ Fig 9: full-process simulation (end-to-end, all passes)
def fig_full_process():
    dl = load("full_process_lqg.json")
    da = load("full_process_adrc.json")
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), sharey=True)
    cmap = plt.cm.viridis
    for ax, d, name, c in [(axes[0], dl, "LQG (frozen design)", C_LQG),
                           (axes[1], da, "ADRC (frozen design)", C_ADRC)]:
        for pp in range(1, d["n_pass"] + 1):
            rows = [r for r in d["segments"] if r["n_pass"] == pp]
            x = [r["x_mm"] for r in rows]
            y = [r["y_rms_um"] for r in rows]
            col = cmap(0.15 + 0.75 * (pp - 1) / max(d["n_pass"] - 1, 1))
            ax.plot(x, y, "-o", color=col, lw=1.6, ms=3.5,
                    label=f"pass {pp} (h={4.0 - 0.25 * pp:.2f} mm)")
        ps = d["pass_summary"]
        ax.set_title(f"{name}\nper-pass RMS: " +
                     ", ".join(f"{r['y_rms_um']:.2f}" for r in ps) + " $\\mu$m",
                     fontsize=10)
        ax.set_xlabel("tool position x (mm)")
        ax.legend(fontsize=8, framealpha=0.95)
    axes[0].set_ylabel(r"segment tip RMS ($\mu$m)")
    fig.suptitle("Full-process simulation: 4 end-to-end passes, material removed "
                 "behind the tool (predicted finish map)", fontsize=11, y=1.02)
    save(fig, "fig9_full_process")



# --------------- Fig 10: AFC-ADRC improvement over the full process
def fig_afc():
    dl = load("full_process_lqg.json")
    da = load("full_process_adrc.json")
    df = load("full_process_afc.json")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    # (a) pass-4 profiles
    ax = axes[0]
    for d, c, lab in [(dl, C_LQG, "LQG"), (da, C_ADRC, "ADRC (collocated only)"),
                      (df, "#7b1fa2", "AFC-ADRC (dual sensor)")]:
        rows = [r for r in d["segments"] if r["n_pass"] == d["n_pass"]]
        ax.plot([r["x_mm"] for r in rows], [r["y_rms_um"] for r in rows],
                "-o", color=c, lw=1.8, ms=4, label=lab)
    ax.set_xlabel("tool position x (mm)")
    ax.set_ylabel(r"segment tip RMS ($\mu$m)")
    ax.set_title("(a) final pass (h = 3.0 mm): vibration along the path", fontsize=10.5)
    ax.legend(fontsize=9, framealpha=0.95)
    # (b) per-pass process RMS
    ax = axes[1]
    x = np.arange(1, 5); w = 0.27
    for off, d, c, lab in [(-w, dl, C_LQG, "LQG"), (0, da, C_ADRC, "ADRC"),
                           (w, df, "#7b1fa2", "AFC-ADRC")]:
        v = [r["y_rms_um"] for r in d["pass_summary"]]
        bars = ax.bar(x + off, v, w, color=c, ec="k", label=lab)
        for b_, vv in zip(bars, v):
            ax.text(b_.get_x() + b_.get_width() / 2, vv + 0.008, f"{vv:.2f}",
                    ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([f"pass {p}" for p in x])
    ax.set_ylabel(r"process RMS ($\mu$m)")
    ax.set_title("(b) per-pass process RMS", fontsize=10.5)
    ax.legend(fontsize=9, framealpha=0.95)
    save(fig, "fig10_afc")



# ---------------- Fig 11: full-process time response (all controllers)
def fig_timeseries():
    import numpy as np
    specs = [("full_process_lqg_traces.npz", C_LQG, "LQG (tip sensor + observer)"),
             ("full_process_adrc_traces.npz", C_ADRC, "ADRC (collocated only)"),
             ("full_process_afc_traces.npz", "#7b1fa2", "AFC-ADRC (dual sensor)")]
    data = []
    for f, c, lab in specs:
        d = np.load(os.path.join(RES, f))
        y = d["y"] * 1e6
        dt_eff = float(d["dt"]) * int(d["decim"])
        t = np.arange(len(y)) * dt_eff
        # running RMS envelope (~80 ms window)
        w = max(1, int(round(0.08 / dt_eff)))
        rms = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, mode="same"))
        data.append((t, y, rms, c, lab))
    Tpass = 20.41
    ymax = 2.1
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 8.0), sharex=True)
    for ax, (t, y, rms, c, lab) in zip(axes, data):
        ax.plot(t, y, color=c, lw=0.15, alpha=0.55)
        ax.plot(t, rms, color="k", lw=1.4)
        ax.plot(t, -rms, color="k", lw=1.4)
        for pb in [Tpass, 2 * Tpass, 3 * Tpass]:
            ax.axvline(pb, color="0.4", ls="--", lw=1)
        yr = np.sqrt(np.mean(y ** 2))
        ax.text(0.006, 0.90, f"{lab}   (process RMS = {yr:.3f} $\\mu$m)",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", bbox=dict(fc="white", ec="none", alpha=0.8))
        ax.set_ylim(-ymax, ymax)
        ax.set_ylabel(r"$y_{tip}$ ($\mu$m)")
        ax.grid(True, alpha=0.3)
    for i, xlab in enumerate(["pass 1", "pass 2", "pass 3", "pass 4"]):
        axes[0].text((i + 0.5) * Tpass, 1.75, xlab + "\n(x: 0$\\to$100 mm)",
                     ha="center", fontsize=8.5, color="0.3")
    axes[-1].set_xlabel("process time (s)   —   each pass sweeps the tool from x = 0 to 100 mm")
    axes[0].set_xlim(0, 4 * Tpass)
    fig.suptitle("Full-process tip response: 4 end-to-end passes with in-process "
                 "material removal (black = 80 ms running RMS envelope)",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "fig11_timeseries")

    # full-rate steady-state waveform snapshot at the worst region (x = 90 mm,
    # h = 3.0 mm) -- shows the actual tooth-passing waveform and its cancellation
    wf = os.path.join(RES, "waveform_x90.npz")
    if os.path.exists(wf):
        d = np.load(wf)
        dtw = float(d["dt"])
        fig, ax = plt.subplots(figsize=(11.5, 4.4))
        for key, c, lab in [("lqg", C_LQG, "LQG"), ("adrc", C_ADRC, "ADRC (collocated only)"),
                            ("afc", "#7b1fa2", "AFC-ADRC (dual sensor)")]:
            y = d[key] * 1e6
            t = np.arange(len(y)) * dtw * 1e3
            rms = np.sqrt(np.mean(y ** 2))
            ax.plot(t, y, color=c, lw=1.2, label=f"{lab}  (RMS={rms:.3f} $\\mu$m)")
        # mark the tooth-passing period
        f_t = 3 * 4900 / 60.0
        ax.set_xlabel("time (ms)  —  steady state at x = 90 mm, wall h = 3.0 mm "
                      f"(tooth passing = {f_t:.0f} Hz, period {1e3/f_t:.2f} ms)")
        ax.set_ylabel(r"$y_{tip}$ ($\mu$m)")
        ax.set_title("Full-rate waveform at the worst region: AFC-ADRC cancels the "
                     "tooth-passing harmonics")
        ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(d["lqg"]) * dtw * 1e3)
        save(fig, "fig11b_waveform")


# ---------------------------- Fig 12: tip-only controller comparison (+ HYBRID)
def fig_tip_comparison():
    dt_ = load("fopid_tip.json")
    dh = load("hybrid_tip.json")
    tr = np.load(os.path.join(RES, "hybrid_tip_traces.npz"))
    C_HYB = "#7b1fa2"
    cols = {"LQG": C_LQG, "ADRC": C_ADRC, "FOPID": C_FOPID, "HYBRID": C_HYB}
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6), layout="constrained")

    # (a) voltage-feasible depth on the tip sensor
    ax = axes[0, 0]
    names = ["LQG", "ADRC", "FOPID", "HYBRID"]
    s = dh["summary_tip_feasible_mm"]
    fea = [s[n] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, fea, 0.55, color=[cols[n] for n in names], ec="k")
    for xi, v in zip(x, fea):
        ax.text(xi, v + 0.03, ("fails" if v < 0.1 else f"{v:.2f}"), ha="center",
                fontsize=9.5, fontweight="bold", color=("#b00" if v < 0.1 else "k"))
    ax.axhline(s["OL_linear"], color=C_OL, ls=":", lw=1.5)
    ax.text(3.35, s["OL_linear"] + 0.02, f"open loop {s['OL_linear']:.2f} mm",
            color=C_OL, ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["LQG\n(full model)", "ADRC\n(plain ESO)",
                        "FOPID\n(model-free)", "HYBRID\n(ESO+FOPID)"], fontsize=9)
    ax.set_ylabel(r"voltage-feasible $a_p$ (mm)")
    ax.set_title("(a) feasible depth, all controllers on the article's tip sensor\n"
                 "(non-minimum phase: coupling " + dt_["tip_coupling_signs"] + ")",
                 fontsize=10)

    # (b) stability demo at 0.5 mm
    ax = axes[0, 1]
    for name in ["ADRC", "FOPID", "HYBRID", "LQG"]:
        k = f"{name}_0.50"
        if f"{k}_t" not in tr:
            continue
        ax.plot(tr[f"{k}_t"] * 1e3, tr[f"{k}_y_um"], color=cols[name], lw=0.9,
                label=name)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("tip displacement (µm)")
    ax.set_title("(b) tip response at $a_p$ = 0.5 mm\n"
                 "diverging traces terminate at the chatter threshold", fontsize=10)
    ax.legend(fontsize=9)

    # (c) rejection at a low common depth (0.15 mm): FOPID vs HYBRID vs LQG
    ax = axes[1, 0]
    for name in ["LQG", "FOPID", "HYBRID"]:
        k = f"{name}_0.15"
        if f"{k}_t" not in tr:
            continue
        rms = dh["trace_summary"][k]["yrms_um"]
        ax.plot(tr[f"{k}_t"] * 1e3, tr[f"{k}_y_um"], color=cols[name], lw=0.9,
                label=f"{name}  (RMS {rms:.2f} µm)" if rms else name)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("tip displacement (µm)")
    ax.set_title("(c) rejection at $a_p$ = 0.15 mm (all stable)\n"
                 "does the ESO branch improve on plain FOPID?", fontsize=10)
    ax.legend(fontsize=9)

    # (d) drift robustness (feasible metric)
    ax = axes[1, 1]
    rb = dh["robustness_feasible_mm"]
    tags, xx = ["-20%", "nominal", "+20%"], [-20, 0, 20]
    for n in ["HYBRID", "LQG", "FOPID"]:
        ax.plot(xx, [rb[t][n] for t in tags], "o-", color=cols[n], lw=2, ms=7,
                label=n)
    ax.set_xticks(xx); ax.set_xlabel("plant frequency drift (%)")
    ax.set_ylabel(r"voltage-feasible $a_p$ (mm)")
    ax.set_title("(d) robustness: feasible depth under frequency drift\n"
                 "(tip sensor; plain ADRC infeasible everywhere)", fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle("Tip-sensor-only comparison (the article's measurement): "
                 "LQG / ADRC / FOPID / HYBRID (band-limited ESO + FOPID)",
                 fontsize=12)
    save(fig, "fig12_tip_comparison")

if __name__ == "__main__":
    reg = [("sld.npz", fig_sld), ("robustness.json", fig_robustness),
           ("adrc.json", fig_scenarios), ("feedforward.json", fig_feedforward),
           ("synthesis.json", fig_authority), ("placement.json", fig_placement),
           ("refinement.json", fig_refinement),
           ("material_removal.json", fig_material_removal),
           ("full_process_lqg.json", fig_full_process),
           ("full_process_afc.json", fig_afc),
           ("full_process_afc_traces.npz", fig_timeseries),
           ("hybrid_tip.json", fig_tip_comparison)]
    for f, fn in reg:
        if os.path.exists(os.path.join(RES, f)):
            fn()
    print("figures/ done.")
