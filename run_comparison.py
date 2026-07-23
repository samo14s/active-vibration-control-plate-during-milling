#!/usr/bin/env python3
"""
Controller comparison for active vibration (milling-chatter) control of a
thin-walled cantilever plate.

Reproduces the milling-chatter scenario of

    J. Du et al., "Robust combined time delay control for milling chatter
    suppression of flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257,

and compares the SLIDING-MODE FAMILY on the SAME plant, actuator and sampling:

    SMC (plain state sliding mode)
      -> TD-SMC (the known combined sliding-mode + time-delay design; not novel)
      -> ST-SMC (the DEVELOPED controller: super-twisting reaching + a model-based
                 regenerative-force feedforward, PSO-tuned for a harder envelope)

The harder-than-paper stress suite lives in make_challenge_figure.py /
src/challenges.py; this script is the head-to-head at the nominal and paper
worst-case conditions.

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
from src.controllers.smc import SMC
from src.controllers.tdsmc import TDSMC
from src.controllers.stsmc import STSMC

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

# controller lineup: (key, factory)   -- order controls plotting/legend order
CONTROLLERS = [
    ("SMC",                SMC),      # plain state sliding mode
    ("TD-SMC",             TDSMC),    # known combined SMC + time-delay (not novel)
    ("ST-SMC (developed)", STSMC),    # developed: super-twisting + regen feedforward
]

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True, "legend.framealpha": 0.9,
    "axes.prop_cycle": plt.cycler(color=["#188038", "#7b5c00", "#8e24aa"]),
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


_COLOR = {"SMC": "#188038", "TD-SMC": "#7b5c00",
          "ST-SMC (developed)": "#8e24aa", "No control": "#9aa0a6"}


# --------------------------------------------------------------------------- #
def fig_time_response(runs):
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), height_ratios=[1, 1.3])
    b = runs["No control"]
    ax[0].plot(b["t"] * 1e3, b["y"] * 1e6, color=_COLOR["No control"], lw=0.8)
    ax[0].set_title("(a) Uncontrolled milling — regenerative chatter grows unbounded")
    ax[0].set_ylabel("displacement (µm)")
    ax[0].set_xlabel("time (ms)")
    excluded = []
    for key, _ in CONTROLLERS:
        r = runs[key]
        y_um = r["y"] * 1e6
        # skip controllers that fail at nominal (they rail off-scale and swamp the
        # overlay); they are shown individually in fig_time_full.png
        if float(np.max(np.abs(y_um))) > 100.0:
            excluded.append(key)
            continue
        ax[1].plot(r["t"] * 1e3, y_um, color=_COLOR[key], lw=0.9, label=key)
    ttl = "(b) Controlled milling — plate displacement with each controller"
    if excluded:
        ttl += f"\n({', '.join(excluded)} fails at nominal — see fig_time_full.png)"
    ax[1].set_title(ttl)
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
    n = len(CONTROLLERS)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.8 * n), sharex=True)
    for a, (key, _) in zip(np.atleast_1d(axes).ravel(), CONTROLLERS):
        r = runs[key]
        y_um = r["y"] * 1e6
        a.plot(r["t"] * 1e3, y_um, color=_COLOR[key], lw=0.6)
        m = M.compute(r)
        a.set_title(f"{key}   (RMS {m['rms_settled_um']:.2f} µm, "
                    f"peak {m['peak_um']:.1f} µm)", fontsize=9)
        a.set_ylabel("displacement (µm)", fontsize=8)
        lim = 1.1 * float(np.max(np.abs(y_um)))
        a.set_ylim(-lim, lim)
    np.atleast_1d(axes).ravel()[-1].set_xlabel("time (ms)")
    fig.suptitle("Full time response over the whole milling pass "
                 "(control active throughout) — per-panel vertical scale",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_time_full.png"))
    plt.close(fig)


def fig_control_voltage(runs):
    n = len(CONTROLLERS)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.4 * n), sharex=True)
    for a, (key, _) in zip(np.atleast_1d(axes).ravel(), CONTROLLERS):
        r = runs[key]
        a.plot(r["t"] * 1e3, r["u"], color=_COLOR[key], lw=0.6)
        pk = float(np.max(np.abs(r["u"])))
        a.set_title(f"{key}   (peak {pk:.0f} V)", fontsize=9)
        a.set_ylabel("u (V)", fontsize=8)
        lim = max(20.0, 1.15 * pk)
        a.set_ylim(-lim, lim)
    np.atleast_1d(axes).ravel()[-1].set_xlabel("time (ms)")
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
    excluded = []
    for key, _ in CONTROLLERS:
        r = runs[key]
        if float(np.max(np.abs(r["y"] * 1e6))) > 100.0:   # rails off-scale at nominal
            excluded.append(key)
            continue
        f, a = M.spectrum(r["t"], r["y"])
        ax.semilogy(f, a * 1e6, color=_COLOR[key], lw=0.9, label=key)
    for fn, lbl in zip(C.MODE_FREQ_HZ[:2], ["mode 1", "mode 2"]):
        ax.axvline(fn, color="gray", ls=":", lw=0.7)
        ax.text(fn, ax.get_ylim()[1] * 0.4, lbl, rotation=90,
                va="top", ha="right", fontsize=7, color="gray")
    ax.set_xlim(0, 3500)
    ax.set_ylim(1e-3, None)
    ttl = "Displacement spectra — chatter peaks (≈540 & 1068 Hz) suppressed"
    if excluded:
        ttl += f"  ({', '.join(excluded)} fails at nominal, omitted)"
    ax.set_title(ttl)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude (µm)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_spectrum.png"))
    plt.close(fig)


def fig_metrics_bars(met_nom, met_worst):
    keys = [k for k, _ in CONTROLLERS]
    colors = [_COLOR[k] for k in keys]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.6))
    x = np.arange(len(keys))

    # (a) nominal settled RMS — log scale
    rms = [max(met_nom[k]["rms_settled_um"], 1e-3) for k in keys]
    ax[0, 0].bar(x, rms, color=colors)
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("(a) settled RMS displacement (nominal)")
    ax[0, 0].set_ylabel("µm  (log)")
    for xi, k in zip(x, keys):
        if met_nom[k]["diverged"] or met_nom[k]["rms_settled_um"] > 100:
            ax[0, 0].text(xi, rms[keys.index(k)] * 1.15, "✗", ha="center",
                          fontsize=11, color="#c5221f")

    # (b) nominal peak control voltage
    pv = [met_nom[k]["peak_volt"] for k in keys]
    ax[0, 1].bar(x, pv, color=colors)
    ax[0, 1].axhline(C.U_MAX, color="k", ls="--", lw=0.6, alpha=0.6)
    ax[0, 1].text(len(keys) - 1, C.U_MAX * 0.92, "saturation", fontsize=7, ha="right")
    ax[0, 1].set_title("(b) peak control voltage (nominal)")
    ax[0, 1].set_ylabel("V")

    # (c) nominal control energy — log scale
    en = [max(met_nom[k]["control_energy"], 1e-4) for k in keys]
    ax[1, 0].bar(x, en, color=colors)
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("(c) control energy ∫u²dt (nominal)")
    ax[1, 0].set_ylabel("V²·s  (log)")

    # (d) paper worst case (α₄=2.9, −20% damping): settled RMS, log; diverged
    #     controllers drawn at the ceiling with a ✗ (they do not survive it).
    CEIL = 5.0e3
    wr = []
    for k in keys:
        wr.append(CEIL if met_worst[k]["diverged"]
                  else max(met_worst[k]["rms_settled_um"], 1e-3))
    ax[1, 1].bar(x, wr, color=colors)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylim(1.0, CEIL * 1.6)
    for xi, k in zip(x, keys):
        if met_worst[k]["diverged"]:
            ax[1, 1].text(xi, CEIL * 0.06, "✗ unstable", ha="center", rotation=90,
                          fontsize=8, color="white", va="bottom", fontweight="bold")
    ax[1, 1].set_title("(d) paper worst case  (α₄=2.9, −20% damping)\n"
                       "settled RMS; ✗ = diverges")
    ax[1, 1].set_ylabel("µm  (log)")

    for a in ax.ravel():
        a.set_xticks(x)
        a.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_metrics_bars.png"))
    plt.close(fig)


def fig_robustness(factors, freq_shifts_pct):
    """Two paper-faithful robustness sweeps (log scale so the failing controllers
    and the tight robust cluster are BOTH visible):
    (a) milling-force coefficient alpha4 (chatter strength; paper Eq. 23: 0.3..2.9)
        at the paper's reduced damping (−20 %), nominal geometry — the decisive
        test: only the robust/model-based designs stay stable across the range;
    (b) modal-frequency drift (material removal) at −20 % damping.
    A point is dropped (curve breaks) where that controller DIVERGES; a marker on
    the top rule flags the divergence explicitly."""
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    FLOOR, CEIL = 1.0, 5.0e3          # µm axis window

    def _plot(a, xs, curves_kwargs, xlabel, title):
        for key, ctor in CONTROLLERS:
            ys, div_x = [], []
            for xv, kw in curves_kwargs:
                r = simulate(P.MillingPlant(**kw), ctor(), t_sim=0.3, meas_noise=True)
                m = M.compute(r)
                if m["diverged"]:
                    ys.append(np.nan); div_x.append(xv)
                else:
                    ys.append(m["rms_settled_um"])
            a.semilogy(xs, ys, "o-", color=_COLOR[key], lw=1.4, ms=4, label=key)
            if div_x:                                   # mark divergences on the ceiling
                a.plot(div_x, [CEIL * 0.9] * len(div_x), "x",
                       color=_COLOR[key], ms=7, mew=1.6)
        a.set_ylim(FLOOR, CEIL)
        a.set_xlabel(xlabel); a.set_ylabel("settled RMS displacement (µm)")
        a.set_title(title); a.legend(ncol=2, fontsize=8)

    # (a) alpha4 sweep at nominal geometry, paper damping (−20 %)
    _plot(ax[0], factors,
          [(f, dict(alpha4_factor=f, dzeta=-0.2)) for f in factors],
          "milling-force-coefficient factor  (× average α₄)",
          "(a) vs milling-force coefficient α₄  (−20 % damping)\n"
          "✗ = diverges;  ST-SMC holds the widest force band before divergence")
    ax[0].axvspan(0.3, 2.9, color="gray", alpha=0.08)
    ax[0].axvline(C.ALPHA4_NOMINAL_FACTOR, color="gray", ls=":", lw=0.8)

    # (b) modal-frequency drift: dmass shifts every omega by ~(1+pct/100)
    _plot(ax[1], freq_shifts_pct,
          [(pct, dict(dmass=1.0 / (1.0 + pct / 100.0) ** 2 - 1.0, dzeta=-0.2))
           for pct in freq_shifts_pct],
          "natural-frequency shift (%)   (material removal → positive)",
          "(b) vs modal-frequency shift  (−20 % damping)\n"
          "material removal raises f (+side); flat = frequency-robust")
    ax[1].axvline(0, color="gray", ls=":", lw=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_robustness.png"))
    plt.close(fig)


def fig_activation():
    """Let chatter start to build (control off), then switch control on at t_on
    while the amplitude is still recoverable, showing each controller damp it."""
    t_on = 0.035
    fig, ax = plt.subplots(len(CONTROLLERS), 1, figsize=(9, 2.9 * len(CONTROLLERS)),
                           sharex=True)
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
    print("  · paper-style worst case (α₄=2.9 strong chatter force, −20% damping) ...")
    runs_worst = run_all(plant_kwargs=dict(dzeta=-0.2, alpha4_factor=2.9))

    met_nom = {k: M.compute(runs_nom[k]) for k in runs_nom}
    met_worst = {k: M.compute(runs_worst[k]) for k in runs_worst}

    print("  · figures ...")
    fig_time_response(runs_nom)
    fig_time_full(runs_nom)
    fig_control_voltage(runs_nom)
    fig_spectrum(runs_nom)
    fig_metrics_bars({k: met_nom[k] for k, _ in CONTROLLERS},
                     {k: met_worst[k] for k, _ in CONTROLLERS})
    fig_robustness(np.array([0.3, 0.8, 1.3, 1.8, 2.3, 2.9, 3.3, 3.7]),
                   np.array([-8, -4, 0, 4, 8, 12]))
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

    print("\n=== CONTROLLER COMPARISON (nominal condition S | paper worst case) ===")
    print(f"{'controller':13s} {'RMS_um':>7s} {'peak_um':>8s} {'rmsV':>6s} "
          f"{'peakV':>6s} {'energy':>8s} {'worst_case':>12s}")
    for key, _ in CONTROLLERS:
        mn, mw = met_nom[key], met_worst[key]
        worst = "DIVERGES" if mw["diverged"] else f"{mw['rms_settled_um']:8.2f} µm"
        print(f"{key:13s} {mn['rms_settled_um']:7.3f} {mn['peak_um']:8.2f} "
              f"{mn['rms_volt']:6.2f} {mn['peak_volt']:6.1f} "
              f"{mn['control_energy']:8.4f} {worst:>12s}")

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
