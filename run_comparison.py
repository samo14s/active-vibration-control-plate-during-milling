#!/usr/bin/env python3
"""
Controller comparison for active vibration (milling-chatter) control of a
thin-walled cantilever plate.

Reproduces the milling-chatter scenario of

    J. Du et al., "Robust combined time delay control for milling chatter
    suppression of flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257,

and compares six control strategies on the SAME plant, actuator and sampling:

    PID (classical) | SMC | H-infinity | mu-synthesis | ADRC | MPC (predictive)

Outputs (written to results/):
    * fig_time_response.png    time-domain plate displacement
    * fig_control_voltage.png  control voltages
    * fig_spectrum.png         displacement spectra (chatter suppression)
    * fig_metrics_bars.png     RMS / voltage / energy / robustness bars
    * fig_robustness.png       performance across the uncertainty range
    * fig_activation.png       chatter build-up then control activation
    * metrics.csv / metrics.json  numerical results
"""
from __future__ import annotations
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src import plant as P
from src import metrics as M
from src.simulate import simulate
from src.controllers.base import ZeroController
from src.controllers.pid import PID
from src.controllers.smc import SMC
from src.controllers.hinf import Hinf
from src.controllers.musyn import MuSynthesis
from src.controllers.adrc import ADRC
from src.controllers.mpc import MPC

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

# controller lineup: (key, factory)   -- order controls plotting/legend order
CONTROLLERS = [
    ("PID",          PID),
    ("SMC",          SMC),
    ("H-infinity",   Hinf),
    ("mu-synthesis", MuSynthesis),
    ("ADRC",         ADRC),
    ("MPC",          MPC),
]

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True, "legend.framealpha": 0.9,
    "axes.prop_cycle": plt.cycler(color=[
        "#e8710a", "#188038", "#1a73e8", "#d01884", "#a142f4", "#00897b"]),
})


def build(key):
    for k, ctor in CONTROLLERS:
        if k == key:
            return ctor()
    raise KeyError(key)


# --------------------------------------------------------------------------- #
def run_all(plant_kwargs, t_sim=C.T_SIM, control_on_at=0.0, meas_noise=True):
    """Run every controller (and the uncontrolled baseline) on one plant."""
    runs = {}
    base = simulate(P.MillingPlant(**plant_kwargs), ZeroController(),
                    t_sim=t_sim, control_on_at=control_on_at, meas_noise=meas_noise)
    runs["No control"] = base
    for key, ctor in CONTROLLERS:
        c = ctor()
        runs[key] = simulate(P.MillingPlant(**plant_kwargs), c,
                             t_sim=t_sim, control_on_at=control_on_at,
                             meas_noise=meas_noise)
    return runs


def color_of(key):
    for k, ctor in CONTROLLERS:
        if k == key:
            return ctor().color if False else _COLOR[key]
    return "#333333"


_COLOR = {"PID": "#e8710a", "SMC": "#188038", "H-infinity": "#1a73e8",
          "mu-synthesis": "#d01884", "ADRC": "#a142f4", "MPC": "#00897b",
          "No control": "#9aa0a6"}


# --------------------------------------------------------------------------- #
def fig_time_response(runs):
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), height_ratios=[1, 1.3])
    b = runs["No control"]
    ax[0].plot(b["t"] * 1e3, b["y"] * 1e6, color=_COLOR["No control"], lw=0.8)
    ax[0].set_title("(a) Uncontrolled milling — regenerative chatter grows unbounded")
    ax[0].set_ylabel("displacement (µm)")
    ax[0].set_xlabel("time (ms)")
    for key, _ in CONTROLLERS:
        r = runs[key]
        ax[1].plot(r["t"] * 1e3, r["y"] * 1e6, color=_COLOR[key], lw=0.9, label=key)
    ax[1].set_title("(b) Controlled milling — plate displacement with each controller")
    ax[1].set_ylabel("displacement (µm)")
    ax[1].set_xlabel("time (ms)")
    ax[1].legend(ncol=3, fontsize=8, loc="upper right")
    ax[1].set_ylim(-40, 40)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_time_response.png"))
    plt.close(fig)


def fig_time_full(runs):
    """Full-horizon time response, one panel per controller, auto-scaled per
    panel so the entire trajectory is visible (no clipping)."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for a, (key, _) in zip(axes.ravel(), CONTROLLERS):
        r = runs[key]
        y_um = r["y"] * 1e6
        a.plot(r["t"] * 1e3, y_um, color=_COLOR[key], lw=0.6)
        m = M.compute(r)
        a.set_title(f"{key}   (RMS {m['rms_settled_um']:.2f} µm, "
                    f"peak {m['peak_um']:.1f} µm)", fontsize=9)
        a.set_ylabel("displacement (µm)", fontsize=8)
        lim = 1.1 * float(np.max(np.abs(y_um)))
        a.set_ylim(-lim, lim)
    for a in axes[-1, :]:
        a.set_xlabel("time (ms)")
    fig.suptitle("Full time response over the whole milling pass "
                 "(control active throughout) — per-panel vertical scale",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_time_full.png"))
    plt.close(fig)


def fig_control_voltage(runs):
    fig, axes = plt.subplots(3, 2, figsize=(10, 7.5), sharex=True)
    for a, (key, _) in zip(axes.ravel(), CONTROLLERS):
        r = runs[key]
        a.plot(r["t"] * 1e3, r["u"], color=_COLOR[key], lw=0.6)
        pk = float(np.max(np.abs(r["u"])))
        a.set_title(f"{key}   (peak {pk:.0f} V)", fontsize=9)
        a.set_ylabel("u (V)", fontsize=8)
        lim = max(20.0, 1.15 * pk)
        a.set_ylim(-lim, lim)
    for a in axes[-1, :]:
        a.set_xlabel("time (ms)")
    fig.suptitle("Control voltages (piezo actuator) — note the differing vertical scales",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_control_voltage.png"))
    plt.close(fig)


def fig_spectrum(runs):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    b = runs["No control"]
    # clip the uncontrolled record to before it blows the axes (use first 40%)
    nb = len(b["t"]); ib = int(0.45 * nb)
    fb, ab = M.spectrum(b["t"][:ib], b["y"][:ib])
    ax.semilogy(fb, ab * 1e6, color=_COLOR["No control"], lw=1.0,
                label="No control", alpha=0.9)
    for key, _ in CONTROLLERS:
        r = runs[key]
        f, a = M.spectrum(r["t"], r["y"])
        ax.semilogy(f, a * 1e6, color=_COLOR[key], lw=0.9, label=key)
    for fn, lbl in zip(C.MODE_FREQ_HZ[:2], ["mode 1", "mode 2"]):
        ax.axvline(fn, color="gray", ls=":", lw=0.7)
        ax.text(fn, ax.get_ylim()[1] * 0.4, lbl, rotation=90,
                va="top", ha="right", fontsize=7, color="gray")
    ax.set_xlim(0, 3500)
    ax.set_ylim(1e-3, None)
    ax.set_title("Displacement spectra — chatter peaks (≈540 & 1068 Hz) suppressed")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude (µm)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_spectrum.png"))
    plt.close(fig)


def fig_metrics_bars(met_nom, met_worst):
    keys = [k for k, _ in CONTROLLERS]
    colors = [_COLOR[k] for k in keys]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    x = np.arange(len(keys))

    rms = [met_nom[k]["rms_settled_um"] for k in keys]
    ax[0, 0].bar(x, rms, color=colors)
    ax[0, 0].set_title("(a) settled RMS displacement (nominal)")
    ax[0, 0].set_ylabel("µm")

    pv = [met_nom[k]["peak_volt"] for k in keys]
    ax[0, 1].bar(x, pv, color=colors)
    ax[0, 1].axhline(C.U_MAX, color="k", ls="--", lw=0.6, alpha=0.6)
    ax[0, 1].set_title("(b) peak control voltage (nominal)")
    ax[0, 1].set_ylabel("V")

    en = [met_nom[k]["control_energy"] for k in keys]
    ax[1, 0].bar(x, en, color=colors)
    ax[1, 0].set_title("(c) control energy ∫u²dt (nominal)")
    ax[1, 0].set_ylabel("V²·s")

    rob = [met_worst[k]["rms_settled_um"] / max(met_nom[k]["rms_settled_um"], 1e-9)
           for k in keys]
    ax[1, 1].bar(x, rob, color=colors)
    ax[1, 1].axhline(1.0, color="k", ls="--", lw=0.6, alpha=0.6)
    ax[1, 1].set_title("(d) robustness  RMS(worst-case)/RMS(nominal)\n(1.0 = perturbation-insensitive)")
    ax[1, 1].set_ylabel("ratio")

    for a in ax.ravel():
        a.set_xticks(x)
        a.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_metrics_bars.png"))
    plt.close(fig)


def fig_robustness(factors, freq_shifts_pct):
    """Two robustness sweeps:
    (a) milling-force coefficient alpha4 (chatter strength; paper: 0.3..2.9),
    (b) modal-frequency drift (material removal) — the demanding test that
        separates broadband dampers from narrowly tuned model-based designs."""
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))

    # (a) alpha4 sweep (frequency-preserving mass/stiffness perturbation)
    for key, ctor in CONTROLLERS:
        rmss = []
        for fac in factors:
            r = simulate(P.MillingPlant(alpha4_factor=fac, dmass=0.08,
                                        dstiff=0.08, dzeta=-0.2),
                         ctor(), t_sim=0.24, meas_noise=True)
            m = M.compute(r)
            rmss.append(m["rms_settled_um"] if not m["diverged"] else np.nan)
        ax[0].plot(factors, rmss, "o-", color=_COLOR[key], lw=1.4, ms=4, label=key)
    ax[0].axvspan(0.3, 2.9, color="gray", alpha=0.08)
    ax[0].axvline(C.ALPHA4_NOMINAL_FACTOR, color="gray", ls=":", lw=0.8)
    ax[0].set_title("(a) vs milling-force coefficient α₄\n(+8% mass/stiffness, −20% damping)")
    ax[0].set_xlabel("milling-force-coefficient factor  (× average α₄)")
    ax[0].set_ylabel("settled RMS displacement (µm)")
    ax[0].legend(ncol=2, fontsize=8)

    # (b) modal-frequency drift: dmass shifts every omega by 1/sqrt(1+dmass)
    for key, ctor in CONTROLLERS:
        rmss = []
        for pct in freq_shifts_pct:
            dmass = 1.0 / (1.0 + pct / 100.0) ** 2 - 1.0     # omega*(1+pct/100)
            r = simulate(P.MillingPlant(dmass=dmass, dzeta=-0.2),
                         ctor(), t_sim=0.22, meas_noise=True)
            m = M.compute(r)
            rmss.append(m["rms_settled_um"] if not m["diverged"] else np.nan)
        ax[1].plot(freq_shifts_pct, rmss, "o-", color=_COLOR[key], lw=1.4, ms=4, label=key)
    ax[1].axvline(0, color="gray", ls=":", lw=0.8)
    ax[1].set_title("(b) vs modal-frequency drift (material removal)\n(−20% damping); flat = frequency-robust")
    ax[1].set_xlabel("natural-frequency shift (%)")
    ax[1].set_ylabel("settled RMS displacement (µm)")
    ax[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_robustness.png"))
    plt.close(fig)


def fig_activation():
    """Let chatter start to build (control off), then switch control on at t_on
    while the amplitude is still recoverable, showing each controller damp it."""
    t_on = 0.035
    fig, ax = plt.subplots(len(CONTROLLERS), 1, figsize=(9, 10), sharex=True)
    for i, (key, ctor) in enumerate(CONTROLLERS):
        c = ctor()
        r = simulate(P.MillingPlant(), c, t_sim=0.22, control_on_at=t_on,
                     meas_noise=True)
        y_um = r["y"] * 1e6
        ax[i].axvspan(0, t_on * 1e3, color="gray", alpha=0.10)
        ax[i].plot(r["t"] * 1e3, y_um, color=_COLOR[key], lw=0.7)
        ax[i].axvline(t_on * 1e3, color="k", ls="--", lw=0.8)
        ax[i].set_ylabel(f"{key}\n(µm)", fontsize=8)
        ax[i].set_ylim(-30, 30)
        if i == 0:
            ax[i].text(t_on * 1e3 + 2, 22, "control ON", fontsize=8)
            ax[i].text(2, 22, "chatter\nbuilding", fontsize=7, color="gray", va="top")
    ax[0].set_title(f"Control activated at t = {t_on*1e3:.0f} ms — each controller "
                    f"damps the growing chatter")
    ax[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_activation.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    print("Running controller comparison (this takes ~1-2 min)...")

    print("  · nominal milling condition S ...")
    runs_nom = run_all(plant_kwargs=dict())
    print("  · worst-case perturbation (α₄=2.9, +10% mass/stiff, −20% damping) ...")
    runs_worst = run_all(plant_kwargs=dict(dmass=0.1, dstiff=0.1, dzeta=-0.2,
                                           alpha4_factor=2.9))

    met_nom = {k: M.compute(runs_nom[k]) for k in runs_nom}
    met_worst = {k: M.compute(runs_worst[k]) for k in runs_worst}

    print("  · figures ...")
    fig_time_response(runs_nom)
    fig_time_full(runs_nom)
    fig_control_voltage(runs_nom)
    fig_spectrum(runs_nom)
    fig_metrics_bars({k: met_nom[k] for k, _ in CONTROLLERS},
                     {k: met_worst[k] for k, _ in CONTROLLERS})
    fig_robustness(np.array([0.3, 0.8, 1.3, 1.6, 2.1, 2.6, 2.9]),
                   np.array([-8, -4, 0, 4, 8]))
    fig_activation()

    # ---- summary table ----
    rows = []
    header = ["controller", "RMS_nom_um", "peak_nom_um", "rmsV", "peakV",
              "energy", "domFreq_Hz", "RMS_worst_um", "robust_ratio", "diverged"]
    for key, _ in CONTROLLERS:
        mn, mw = met_nom[key], met_worst[key]
        rows.append([key,
                     round(mn["rms_settled_um"], 3), round(mn["peak_um"], 2),
                     round(mn["rms_volt"], 2), round(mn["peak_volt"], 1),
                     round(mn["control_energy"], 4), round(mn["dom_freq_hz"], 0),
                     round(mw["rms_settled_um"], 3),
                     round(mw["rms_settled_um"] / max(mn["rms_settled_um"], 1e-9), 2),
                     bool(mn["diverged"] or mw["diverged"])])

    # uncontrolled reference (metrics computed on the pre-divergence window)
    mn0 = met_nom["No control"]
    print("\n=== UNCONTROLLED baseline ===")
    print(f"  diverges: peak {mn0['peak_um']:.0f} µm, dominant chatter {mn0['dom_freq_hz']:.0f} Hz "
          f"(2nd mode ≈ {C.MODE_FREQ_HZ[1]:.0f} Hz)")

    print("\n=== CONTROLLER COMPARISON (nominal milling condition S) ===")
    print(f"{'controller':13s} {'RMS_um':>7s} {'peak_um':>8s} {'rmsV':>6s} "
          f"{'peakV':>6s} {'energy':>8s} {'RMS_worst':>10s} {'robust':>7s}")
    for r in rows:
        print(f"{r[0]:13s} {r[1]:7.3f} {r[2]:8.2f} {r[3]:6.2f} {r[4]:6.1f} "
              f"{r[5]:8.4f} {r[7]:10.3f} {r[8]:7.2f}")

    # ---- write CSV + JSON ----
    import csv
    with open(os.path.join(RESULTS, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump({"nominal": {k: met_nom[k] for k in met_nom},
                   "worst": {k: met_worst[k] for k in met_worst}}, f, indent=2)
    print(f"\nWrote figures + metrics.csv/json to {RESULTS}/")


if __name__ == "__main__":
    main()
