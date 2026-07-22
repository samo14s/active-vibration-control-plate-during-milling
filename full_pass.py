#!/usr/bin/env python3
"""
Full milling-PASS simulation (~20.4 s): the tool traverses the whole free upper
edge (peripheral milling).  As it advances the milling-point mode shape (hence
the regenerative coupling and the cutting excitation) varies with position, and
the modal frequencies drift upward from material removal -- the paper's varying
dynamic characteristics.  The displacement sensor and piezo actuator stay fixed.

Produces two figures in results/:
  * fig14_responses.png  -- paper Fig. 14 style: displacement-vs-time (envelope)
                            and amplitude spectrum for every method.
  * fig_full_pass.png    -- summary: varying coupling, windowed RMS, sample trace.

Run:  python full_pass.py     (a few minutes; 6 controllers x 20.4 s at 25.6 kHz)
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
CONTROLLERS = [("PID", PID), ("SMC", SMC), ("H-infinity", Hinf),
               ("mu-synthesis", MuSynthesis), ("ADRC", ADRC), ("MPC", MPC)]
_COLOR = {"PID": "#e8710a", "SMC": "#188038", "H-infinity": "#1a73e8",
          "mu-synthesis": "#d01884", "ADRC": "#a142f4", "MPC": "#00897b",
          "No control": "#c9922b"}


def envelope(t, y_um, win=0.008):
    """Rolling (min,max) envelope over `win`-second windows."""
    n = max(2, int(win / (t[1] - t[0])))
    tc, ylo, yhi = [], [], []
    for i in range(0, len(t) - n, n):
        seg = y_um[i:i + n]
        tc.append(t[i] + win / 2); ylo.append(seg.min()); yhi.append(seg.max())
    return np.array(tc), np.array(ylo), np.array(yhi)


def windowed_rms(t, y_um, win=0.3):
    n = int(win / (t[1] - t[0]))
    cen, val = [], []
    for i in range(0, len(t) - n, n):
        cen.append(t[i] + win / 2)
        val.append(float(np.sqrt(np.mean(y_um[i:i + n] ** 2))))
    return np.array(cen), np.array(val)


def main():
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    print(f"Full pass: v_feed={C.FEED_VELOCITY*1e3:.2f} mm/s, T_pass={C.T_PASS:.1f} s, "
          f"{int(C.T_PASS*C.FS)} steps/run")

    data = {}
    for name, ctor in CONTROLLERS:
        t0 = time.time()
        r = simulate(P.MillingPlant(moving=True, freq_drift=True), ctor(),
                     t_sim=C.T_PASS, meas_noise=True)
        y_um = r["y"] * 1e6
        te, ylo, yhi = envelope(r["t"], y_um)
        # spectrum over the steady portion (after the 0.5 s start transient); a
        # diverged controller returns a truncated record, so fall back to whatever
        # (pre-divergence) data exists and never FFT an empty slice.
        i0 = int(0.5 * C.FS)
        if len(r["t"]) > i0 + 64:
            f, a = M.spectrum(r["t"][i0:], r["y"][i0:])
        elif len(r["t"]) > 64:
            f, a = M.spectrum(r["t"], r["y"])
        else:
            f, a = np.array([0.0, 1.0]), np.array([0.0, 0.0])
        cen, wr = windowed_rms(r["t"], y_um)
        dec = max(1, len(r["t"]) // 6000)
        data[name] = dict(te=te, ylo=ylo, yhi=yhi, f=f, a=a * 1e6, cen=cen, wr=wr,
                          tdec=r["t"][::dec], ydec=y_um[::dec],
                          peak=float(np.max(np.abs(y_um))), diverged=r["diverged"],
                          t_end=float(r["t"][-1]) if len(r["t"]) else 0.0)
        print(f"  {name:13s} {time.time()-t0:4.0f}s  meanRMS {np.mean(wr):5.2f} µm  "
              f"peak {data[name]['peak']:6.1f} µm  div={r['diverged']}")

    # uncontrolled: diverges fast -> short 0.2 s window (paper Fig 14a)
    ru = simulate(P.MillingPlant(moving=True, freq_drift=True), ZeroController(),
                  t_sim=0.2, meas_noise=True)
    yu = ru["y"] * 1e6
    fu, au = M.spectrum(ru["t"], ru["y"])
    data["No control"] = dict(t=ru["t"], y=yu, f=fu, a=au * 1e6,
                              peak=float(np.max(np.abs(yu))))

    ft = C.FT_S      # tooth-passing frequency 245 Hz

    # ================= FIGURE 1 : paper Fig. 14 style ==================
    order = ["No control"] + [k for k, _ in CONTROLLERS]
    nrow = len(order)
    fig, ax = plt.subplots(nrow, 2, figsize=(12, 2.0 * nrow))
    for i, name in enumerate(order):
        d = data[name]
        col = _COLOR[name]
        axt, axf = ax[i, 0], ax[i, 1]
        if name == "No control":
            axt.plot(d["t"], d["y"], color=col, lw=0.5)
            axt.set_xlim(0, 0.2)
            axt.set_title(f"{name} — diverges (chatter)", fontsize=9)
        elif d.get("diverged"):
            # unstable on the moving pass: show the envelope up to blow-up (capped
            # scale so the growth toward divergence is visible), flag it clearly.
            axt.fill_between(d["te"], d["ylo"], d["yhi"], color=col, alpha=0.85, lw=0)
            axt.set_xlim(0, C.T_PASS)
            axt.set_xticks([0, 5.1, 10.2, 15.3, 20.4])
            axt.set_ylim(-200, 200)
            axt.set_title(f"{name} — DIVERGES at t≈{d['t_end']:.1f}s (unstable on the "
                          f"moving pass)", fontsize=9, color="#c5221f")
        else:
            axt.fill_between(d["te"], d["ylo"], d["yhi"], color=col, alpha=0.85, lw=0)
            axt.set_xlim(0, C.T_PASS)
            axt.set_xticks([0, 5.1, 10.2, 15.3, 20.4])
            lim = 1.15 * max(np.max(np.abs(d["yhi"])), np.max(np.abs(d["ylo"])))
            axt.set_ylim(-lim, lim)
            axt.set_title(f"{name} — full pass (RMS {np.mean(d['wr']):.2f} µm)", fontsize=9)
        axt.set_ylabel("displ. (µm)", fontsize=8)

        # spectrum
        axf.plot(d["f"], d["a"], color=col, lw=0.8)
        axf.set_xlim(0, 1600)
        band = d["a"][(d["f"] > 100) & (d["f"] < 1600)]
        top = (float(np.max(band)) if band.size and np.max(band) > 0 else 1.0) * 1.25 + 1e-9
        axf.set_ylim(0, top)
        for k, lbl in [(1, "$f_t$"), (2, "$2f_t$"), (3, "$3f_t$")]:
            axf.axvline(k * ft, color="gray", ls=":", lw=0.6)
        if name == "No control":
            axf.axvline(C.MODE_FREQ_HZ[1], color="red", ls="--", lw=0.7)
            axf.annotate("$f_{c2}$ chatter", xy=(C.MODE_FREQ_HZ[1], top * 0.8),
                         fontsize=8, color="red", ha="center")
        axf.annotate("$f_t$", xy=(ft, top * 0.85), fontsize=8, ha="center", color="gray")
        axf.set_ylabel("amp. (µm)", fontsize=8)
    ax[-1, 0].set_xlabel("time (s)")
    ax[-1, 1].set_xlabel("frequency (Hz)")
    fig.suptitle("Milling responses over the full 20.4 s pass (paper Fig. 14 style) — "
                 "displacement envelope (left) and spectrum (right)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(os.path.join(RESULTS, "fig14_responses.png"))
    plt.close(fig)
    print("wrote results/fig14_responses.png")

    # ================= FIGURE 2 : summary ==================
    fig2 = plt.figure(figsize=(12, 8))
    gs = fig2.add_gridspec(2, 1, height_ratios=[0.8, 1.2], hspace=0.32)
    ax0 = fig2.add_subplot(gs[0])
    xi = np.linspace(0, 1, 200)
    prof = np.array([C.phi_mill(x) / C.PHI[:C.N_MODES] for x in xi])   # relative to ref
    for i in range(prof.shape[1]):
        ax0.plot(xi * C.T_PASS, prof[:, i] ** 2, lw=1.5, label=f"mode {i+1}")
    ax0.set_title("(a) Varying dynamics — milling-point modal coupling φ² vs tool position")
    ax0.set_ylabel("coupling (×nominal)"); ax0.set_xlabel("time (s)  [0 → 100 mm]")
    ax0.set_xlim(0, C.T_PASS); ax0.legend(ncol=3, fontsize=8)
    ax1 = fig2.add_subplot(gs[1])
    for name, _ in CONTROLLERS:
        d = data[name]
        lbl = name + (" (diverges)" if d["diverged"] else "")
        ax1.semilogy(d["cen"], np.maximum(d["wr"], 1e-2), lw=1.3,
                     color=_COLOR[name], label=lbl)
    ax1.set_title("(b) 0.3 s windowed RMS displacement across the whole pass "
                  "(log scale; SMC lowest — PID/ADRC diverge, curves shoot up & stop)")
    ax1.set_ylabel("RMS displacement (µm)"); ax1.set_xlabel("time (s)")
    ax1.set_xlim(0, C.T_PASS); ax1.set_ylim(1.0, 1e4); ax1.legend(ncol=3, fontsize=8)
    fig2.savefig(os.path.join(RESULTS, "fig_full_pass.png"), bbox_inches="tight")
    plt.close(fig2)
    print("wrote results/fig_full_pass.png")


if __name__ == "__main__":
    main()
