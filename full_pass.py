#!/usr/bin/env python3
"""
Full milling-PASS simulation (~20.4 s): the tool traverses the whole free upper
edge of the plate.  As it advances, the milling-point mode shape (hence the
regenerative coupling and the cutting excitation) varies with position, and the
modal frequencies drift upward from material removal -- the paper's "varying
dynamic characteristics".  The displacement sensor and the piezo actuator stay
fixed.  This shows each controller (designed once, at nominal) holding chatter
down across the entire pass.

Run:  python full_pass.py         (takes a few minutes; 6 controllers x 20.4 s)
Output: results/fig_full_pass.png
"""
from __future__ import annotations
import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src import plant as P
from src.simulate import simulate
from src.controllers.base import ZeroController
from src.controllers.pid import PID
from src.controllers.smc import SMC
from src.controllers.hinf import Hinf
from src.controllers.musyn import MuSynthesis
from src.controllers.adrc import ADRC
from src.controllers.mpc import MPC

RESULTS = os.path.join(os.path.dirname(__file__), "results")
CONTROLLERS = [("PID", PID), ("SMC", SMC), ("H-infinity", Hinf),
               ("mu-synthesis", MuSynthesis), ("ADRC", ADRC), ("MPC", MPC)]
_COLOR = {"PID": "#e8710a", "SMC": "#188038", "H-infinity": "#1a73e8",
          "mu-synthesis": "#d01884", "ADRC": "#a142f4", "MPC": "#00897b",
          "No control": "#9aa0a6"}


def windowed_rms(t, y_um, win=0.3):
    """Sliding-window RMS (non-overlapping) → (centres, rms)."""
    centres, vals = [], []
    n = int(win / (t[1] - t[0]))
    for i in range(0, len(t) - n, n):
        seg = y_um[i:i + n]
        centres.append(t[i] + win / 2)
        vals.append(float(np.sqrt(np.mean(seg ** 2))))
    return np.array(centres), np.array(vals)


def main():
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    vfeed = C.FEED_VELOCITY * 1e3          # mm/s
    print(f"Full pass: v_feed={vfeed:.2f} mm/s, length={C.PLATE['lp']*1e3:.0f} mm, "
          f"T_pass={C.T_PASS:.1f} s, dt=1/{C.FS:.0f}s ({int(C.T_PASS*C.FS)} steps/run)")

    results = {}
    for name, ctor in CONTROLLERS:
        t0 = time.time()
        r = simulate(P.MillingPlant(moving=True, freq_drift=True), ctor(),
                     t_sim=C.T_PASS, meas_noise=True)
        cen, wr = windowed_rms(r["t"], r["y"] * 1e6)
        # decimate the raw trace for plotting
        dec = max(1, len(r["t"]) // 6000)
        results[name] = dict(t=r["t"][::dec], y=r["y"][::dec] * 1e6,
                             cen=cen, wr=wr, diverged=r["diverged"])
        print(f"  {name:13s} done in {time.time()-t0:4.0f}s  "
              f"mean RMS {np.mean(wr):5.2f} µm  max {np.max(np.abs(r['y']))*1e6:6.1f} µm"
              f"  diverged={r['diverged']}")

    # uncontrolled (diverges quickly; simulate a short window to show it)
    ru = simulate(P.MillingPlant(moving=True, freq_drift=True), ZeroController(),
                  t_sim=1.0, meas_noise=True)

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.2, 1.2], hspace=0.35)

    # (a) varying dynamics: milling-point coupling phi_mill^2 vs tool position
    ax0 = fig.add_subplot(gs[0])
    xi = np.linspace(0, 1, 200)
    prof = np.array([P.edge_mode_profile(x) for x in xi])       # (200, n)
    pos_mm = xi * C.PLATE["lp"] * 1e3
    for i in range(prof.shape[1]):
        ax0.plot(xi * C.T_PASS, prof[:, i] ** 2, lw=1.5,
                 label=f"mode {i+1}")
    ax0.set_title("(a) Varying dynamics — relative milling-point modal coupling "
                  "φ_mill² as the tool advances")
    ax0.set_ylabel("coupling (×nominal)")
    ax0.set_xlabel("time (s)   [tool position 0 → 100 mm]")
    ax0.legend(fontsize=8, ncol=3)
    ax0.set_xlim(0, C.T_PASS)

    # (b) windowed RMS along the pass, per controller
    ax1 = fig.add_subplot(gs[1])
    for name, _ in CONTROLLERS:
        d = results[name]
        ax1.plot(d["cen"], d["wr"], lw=1.3, color=_COLOR[name], label=name)
    ax1.set_title("(b) Chatter suppression across the whole pass — "
                  "0.3 s windowed RMS displacement")
    ax1.set_ylabel("RMS displacement (µm)")
    ax1.set_xlabel("time (s)")
    ax1.legend(ncol=3, fontsize=8)
    ax1.set_xlim(0, C.T_PASS)

    # (c) representative full traces: best (H-infinity) vs uncontrolled start
    ax2 = fig.add_subplot(gs[2])
    d = results["H-infinity"]
    ax2.plot(d["t"], d["y"], lw=0.5, color=_COLOR["H-infinity"],
             label="H-infinity (controlled, full 20.4 s)")
    ax2.plot(ru["t"], ru["y"] * 1e6, lw=0.5, color=_COLOR["No control"],
             label="No control (diverges in <1 s)", alpha=0.8)
    ax2.set_title("(c) Full-pass displacement — controlled stays bounded, "
                  "uncontrolled diverges")
    ax2.set_ylabel("displacement (µm)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylim(-60, 60)
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, C.T_PASS)

    fig.suptitle(f"Milling of the whole free upper edge — full {C.T_PASS:.1f} s pass "
                 f"with varying dynamics & material removal", fontsize=12)
    out = os.path.join(RESULTS, "fig_full_pass.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
