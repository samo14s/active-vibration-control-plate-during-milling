#!/usr/bin/env python3
"""Build all manuscript figures from the saved campaign results."""
from __future__ import annotations

import json
import sys as _sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.0,
})

LBL = {"open": "No control", "cpd": "CPD", "vpd": "VPD",
       "pshlqg": "PSH-LQG (proposed)"}
CLR = {"open": "0.35", "cpd": "#c44e52", "vpd": "#dd8452",
       "pshlqg": "#4c72b0"}


def load_pass(name):
    return np.load(RES / f"pass_{name}.npz")


def fig_modes():
    d = np.load(RES / "model.npz")
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.2))
    for k, ax in enumerate(axes.ravel()):
        s = d["shapes"][k]
        m = np.max(np.abs(s)) or 1.0
        cs = ax.contourf(d["shapes_x"] * 1e3, d["shapes_y"] * 1e3, s / m,
                         levels=21, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"Mode {k+1}: {d['freqs'][k]:.0f} Hz")
        ax.set_aspect("equal")
        if k >= 3:
            ax.set_xlabel("x [mm]")
        if k % 3 == 0:
            ax.set_ylabel("y [mm]")
    fig.colorbar(cs, ax=axes, shrink=0.85, label="normalized deflection")
    fig.savefig(FIG / "fig_modes.png")
    plt.close(fig)


def fig_static():
    d = np.load(RES / "model.npz")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for V, c in (("500", "#ccb974"), ("900", "#c44e52"), ("1200", "#4c72b0")):
        ax.plot(d["stat_x"] * 1e3, np.array(d[f"stat_{V}"]) * 1e3,
                color=c, label=f"{V} V")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("top-edge deflection [mm]")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "fig_static.png")
    plt.close(fig)


def fig_force():
    d = np.load(RES / "model.npz")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    a1.plot(d["force_t"] * 1e3, d["force_F"], color="0.2", lw=0.8)
    a1.set_xlabel("time [ms]")
    a1.set_ylabel("normal force $F_n$ [N]")
    a1.grid(alpha=0.3)
    h = d["harm_w"] / (2 * np.pi)
    a2.stem(h[1:] , d["harm_F"][1:], basefmt=" ",
            linefmt="C0-", markerfmt="C0o")
    a2.set_xlabel("frequency [Hz]")
    a2.set_ylabel("|F harmonic| [N]")
    a2.grid(alpha=0.3)
    fig.savefig(FIG / "fig_force.png")
    plt.close(fig)


def fig_gains():
    t = json.loads((RES / "tuning.json").read_text())
    xg = np.array(t["xg"]) * 1e3
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    a1.plot(xg, t["kp_v"], "o-", ms=3, color=CLR["vpd"], label="VPD $k_p(x)$")
    a1.axhline(t["kp_c"], color=CLR["cpd"], ls="--", label="CPD $k_p$")
    a1.set_xlabel("cutter position x [mm]")
    a1.set_ylabel("$k_p$ [V/(m s$^{-2}$)]")
    a1.legend(frameon=False)
    a1.grid(alpha=0.3)
    a2.plot(xg, t["kd_v"], "o-", ms=3, color=CLR["vpd"], label="VPD $k_d(x)$")
    a2.axhline(t["kd_c"], color=CLR["cpd"], ls="--", label="CPD $k_d$")
    a2.set_xlabel("cutter position x [mm]")
    a2.set_ylabel("$k_d$ [V/(m s$^{-2}$)]")
    a2.legend(frameon=False)
    a2.grid(alpha=0.3)
    fig.savefig(FIG / "fig_gains.png")
    plt.close(fig)


def fig_passes_time():
    fig, axes = plt.subplots(4, 1, figsize=(6.6, 6.4), sharex=True)
    for ax, name in zip(axes, ["open", "cpd", "vpd", "pshlqg"]):
        d = load_pass(name)
        ax.plot(d["t"], d["w_cut"] * 1e3, color=CLR[name], lw=0.4)
        ax.set_ylabel("w$_c$ [mm]")
        ax.text(0.985, 0.92, LBL[name], transform=ax.transAxes,
                ha="right", va="top")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 11)
    axes[-1].set_xlabel("time [s]")
    fig.align_ylabels(axes)
    fig.savefig(FIG / "fig_passes_time.png")
    plt.close(fig)


def fig_points_159():
    idx = {"1": 1, "5": 5, "9": 9}
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 5.4), sharex=True)
    for ax, (lab, i) in zip(axes, idx.items()):
        for name in ["open", "cpd", "vpd", "pshlqg"]:
            d = load_pass(name)
            ax.plot(d["t"], d["w_points"][i] * 1e3, color=CLR[name],
                    lw=0.4, label=LBL[name])
        ax.set_ylabel(f"w(P{lab}) [mm]")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 11)
    axes[0].legend(frameon=False, ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, 1.35))
    axes[-1].set_xlabel("time [s]")
    fig.align_ylabels(axes)
    fig.savefig(FIG / "fig_points_159.png")
    plt.close(fig)


def fig_along_path():
    from avcp.metrics import along_path_rms
    from avcp.simulate import SimResult
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for name in ["open", "cpd", "vpd", "pshlqg"]:
        d = load_pass(name)
        res = SimResult(t=d["t"], x_cut=d["x_cut"], w_cut=d["w_cut"],
                        w_points=d["w_points"], acc=d["acc"], V=d["V"],
                        F=d["F"])
        cx, rms = along_path_rms(res, 10.0)
        a1.plot(cx * 1e3, rms * 1e6, color=CLR[name], label=LBL[name])
        # surface profile proxy
        sm = (d["surf_t"] > 0.5) & (d["surf_t"] < 10.0)
        sx, sw = d["surf_x"][sm], d["surf_w"][sm]
        a2.plot(sx * 1e3, (sw - np.mean(sw)) * 1e6, color=CLR[name], lw=0.5)
    a1.set_yscale("log")
    a1.set_xlabel("cutter position x [mm]")
    a1.set_ylabel("AC RMS of w$_c$ [$\\mu$m]")
    a1.legend(frameon=False, fontsize=7)
    a1.grid(alpha=0.3, which="both")
    a2.set_xlabel("cutter position x [mm]")
    a2.set_ylabel("surface profile proxy [$\\mu$m]")
    a2.grid(alpha=0.3)
    fig.savefig(FIG / "fig_along_path.png")
    plt.close(fig)


def fig_voltage_spectra():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for name in ["cpd", "vpd", "pshlqg"]:
        d = load_pass(name)
        a1.plot(d["t"], d["V"], color=CLR[name], lw=0.4, label=LBL[name])
    a1.axhline(1500, color="k", ls=":", lw=0.8)
    a1.axhline(-500, color="k", ls=":", lw=0.8)
    a1.set_xlabel("time [s]")
    a1.set_ylabel("actuator voltage [V]")
    a1.legend(frameon=False, fontsize=7)
    a1.set_xlim(0, 11)
    # spectrogram-like: spectra of w_cut in mid-pass window
    for name in ["open", "pshlqg"]:
        d = load_pass(name)
        m = (d["t"] > 4.0) & (d["t"] < 6.0)
        w = d["w_cut"][m]
        n = len(w)
        fr = np.fft.rfftfreq(n, 1e-4)
        W = np.abs(np.fft.rfft(w * np.hanning(n))) / n * 4
        a2.semilogy(fr, W * 1e6, color=CLR[name], lw=0.7, label=LBL[name])
    for k in range(1, 6):
        a2.axvline(466.67 * k, color="0.8", lw=0.5, zorder=0)
    a2.set_xlim(0, 2600)
    a2.set_xlabel("frequency [Hz]")
    a2.set_ylabel("|W$_c$| [$\\mu$m]")
    a2.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "fig_voltage_spectra.png")
    plt.close(fig)


def fig_frozen():
    d = json.loads((RES / "frozen.json").read_text())
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.array(d["x"]) * 1e3
    for name in ["cpd", "vpd", "pshlqg"]:
        ax.plot(x, d[name], color=CLR[name], label=LBL[name])
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("cutter position x [mm]")
    ax.set_ylabel("frozen spectral radius")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "fig_frozen.png")
    plt.close(fig)


def fig_sld():
    d = json.loads((RES / "sld.json").read_text())
    rpm = np.array(d["rpm"])
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6), sharey=True)
    for ax, pt in zip(axes, ["12", "60", "108"]):
        for name in ["open", "cpd", "vpd", "pshlqg"]:
            a = np.array(d[name][pt]) * 1e3
            ax.plot(rpm, a, color=CLR[name], label=LBL[name])
        ax.axhline(3.0, color="k", ls=":", lw=0.7)
        ax.axvline(7000, color="k", ls=":", lw=0.7)
        ax.set_yscale("log")
        ax.set_title(f"x = {pt} mm")
        ax.set_xlabel("spindle speed [rpm]")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("$a_{p,lim}$ [mm]")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.savefig(FIG / "fig_sld.png")
    plt.close(fig)


def fig_robust():
    rows = json.loads((RES / "robust.json").read_text())
    data = [[r["open"] / r[k] for r in rows] for k in ("cpd", "vpd", "pshlqg")]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    bp = ax.boxplot(data, tick_labels=["CPD", "VPD", "PSH-LQG"],
                    widths=0.5, patch_artist=True)
    for patch, c in zip(bp["boxes"], [CLR["cpd"], CLR["vpd"], CLR["pshlqg"]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("RMS reduction factor (open / controlled)")
    ax.grid(alpha=0.3, which="both")
    fig.savefig(FIG / "fig_robust.png")
    plt.close(fig)


def fig_sensitivity():
    d = np.load(RES / "frf_sens.npz")
    f = d["f"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, pt in ((a1, "60"), (a2, "12")):
        ratio = d[f"pshlqg_{pt}"] / d[f"open_{pt}"]
        ax.semilogy(f, ratio, color=CLR["pshlqg"], lw=0.8)
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        for k in range(1, 6):
            ax.axvline(466.67 * k, color="0.85", lw=0.5, zorder=0)
        ax.set_xlim(30, 2600)
        ax.set_xlabel("frequency [Hz]")
        ax.set_title(f"x = {pt} mm")
        ax.grid(alpha=0.3, which="both")
    a1.set_ylabel("|closed / open| FRF ratio")
    fig.savefig(FIG / "fig_sensitivity.png")
    plt.close(fig)


def fig_rpm():
    d = json.loads((RES / "rpm_sweep.json").read_text())
    rpms = sorted(int(k) for k in d)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    w = 220
    for i, key in enumerate(["open", "detuned", "adaptive"]):
        vals = [d[str(r)][key] * 1e6 for r in rpms]
        ax.bar(np.array(rpms) + (i - 1) * w, vals, width=w,
               label={"open": "No control", "detuned": "PSH-LQG detuned",
                      "adaptive": "PSH-LQG adaptive"}[key],
               color={"open": "0.5", "detuned": "#dd8452",
                      "adaptive": "#4c72b0"}[key])
    ax.set_xticks(rpms)
    ax.set_xlabel("spindle speed [rpm]")
    ax.set_ylabel("AC RMS of w$_c$ [$\\mu$m]")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(FIG / "fig_rpm.png")
    plt.close(fig)


def fig_bars():
    m = json.loads((RES / "metrics_passes.json").read_text())
    names = ["open", "cpd", "vpd", "pshlqg"]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
    for ax, key, lab in zip(
            axes, ["rms_ac_wc", "max_wc", "surf_std"],
            ["AC RMS [$\\mu$m]", "max |w$_c$| [$\\mu$m]",
             "surface std [$\\mu$m]"]):
        vals = [m[n].get(key, np.nan) * 1e6 for n in names]
        ax.bar(range(4), vals, color=[CLR[n] for n in names])
        ax.set_xticks(range(4))
        ax.set_xticklabels(["NC", "CPD", "VPD", "PSH-\nLQG"], fontsize=7)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, axis="y")
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=6.5)
    fig.savefig(FIG / "fig_bars.png")
    plt.close(fig)


ALL = {
    "modes": fig_modes, "static": fig_static, "force": fig_force,
    "gains": fig_gains, "passes": fig_passes_time, "p159": fig_points_159,
    "path": fig_along_path, "volt": fig_voltage_spectra,
    "frozen": fig_frozen, "sld": fig_sld, "robust": fig_robust,
    "rpm": fig_rpm, "bars": fig_bars, "sens": fig_sensitivity,
}

if __name__ == "__main__":
    which = _sys.argv[1:] or list(ALL)
    for name in which:
        try:
            ALL[name]()
            print(f"figure {name}: ok")
        except FileNotFoundError as e:
            print(f"figure {name}: missing data ({e})")
