#!/usr/bin/env python3
"""
PSO tuning figure: (a) swarm convergence (best cost vs iteration) for each
controller, and (b) the full milling-force-coefficient sweep for the PSO-tuned
classical controllers compared with the robust family — showing which controllers
PSO can make robust across the whole alpha4 range and which are structurally
un-tunable on the non-minimum-phase plant.

Reads results/pso_<NAME>.json (written by tune_pso.py).  Convergence curves are
parsed from the run logs if present (path via --logs), else omitted gracefully.

Run:  python make_pso_figure.py
"""
from __future__ import annotations
import os, json, glob, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src import plant as P
from src import metrics as M
from src.simulate import simulate
from src import pso_postprocess as PP
from src import tuning_pso as T
from src.controllers.hinf import Hinf
from src.controllers.musyn import MuSynthesis
from src.controllers.mpc import MPC

RESULTS = os.path.join(os.path.dirname(__file__), "results")
_COLOR = {"PID": "#e8710a", "SMC": "#188038", "ADRC": "#a142f4",
          "H-infinity": "#1a73e8", "mu-synthesis": "#d01884", "MPC": "#00897b",
          "CTDC": "#b31412", "TDSMC": "#7b5c00"}
LOGDIR = None      # set via CLI; scratchpad path holding pso_<NAME>.log


def load(name):
    path = os.path.join(RESULTS, f"pso_{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)[name]


def _factory(name, gains):
    from src.controllers.pid import PID
    from src.controllers.smc import SMC
    from src.controllers.adrc import ADRC
    from src.controllers.ctdc import CTDC
    from src.controllers.tdsmc import TDSMC
    cls = {"PID": PID, "SMC": SMC, "ADRC": ADRC, "CTDC": CTDC, "TDSMC": TDSMC}[name]
    return lambda: cls(**gains)


def main():
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    names = ["PID", "SMC", "ADRC", "CTDC", "TDSMC"]
    data = {n: load(n) for n in names}

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) convergence
    plotted = False
    for n in names:
        if LOGDIR:
            its, gs = PP.parse_convergence(os.path.join(LOGDIR, f"pso_{n}.log"))
            if gs:
                ax[0].semilogy(its, gs, "o-", color=_COLOR[n], lw=1.6, ms=4, label=n)
                plotted = True
    ax[0].axhline(T.DIVERGE_PENALTY, color="gray", ls="--", lw=0.8)
    ax[0].text(0.5, T.DIVERGE_PENALTY * 1.1, "divergence-penalty floor",
               fontsize=8, color="gray")
    ax[0].set_title("(a) PSO convergence — robust cost vs iteration\n"
                    "(cost below the floor ⇒ stable on all cutting conditions)")
    ax[0].set_xlabel("PSO iteration"); ax[0].set_ylabel("best robust cost (log)")
    if plotted:
        ax[0].legend()
    else:
        ax[0].text(0.5, 0.5, "run logs not available", transform=ax[0].transAxes,
                   ha="center", color="gray")

    # (b) full alpha4 sweep for PSO-tuned classical + robust family
    FLOOR, CEIL = 1.0, 5.0e3
    # robust reference family (already-tuned)
    ref = [("H-infinity", Hinf), ("mu-synthesis", MuSynthesis), ("MPC", MPC)]
    for label, ctor in ref:
        xs, ys = PP.full_alpha4_sweep(lambda c=ctor: c())
        ys = [np.nan if v is None else v for v in ys]
        ax[1].semilogy(xs, ys, "s-", color=_COLOR[label], lw=1.2, ms=3,
                       alpha=0.55, label=f"{label} (robust ref)")
    # PSO-tuned classical
    for n in names:
        d = data.get(n)
        if not d:
            continue
        mk = _factory(n, d["gains"])
        xs, ys = PP.full_alpha4_sweep(mk)
        div_x = [xs[i] for i, v in enumerate(ys) if v is None]
        ys = [np.nan if v is None else v for v in ys]
        ax[1].semilogy(xs, ys, "o-", color=_COLOR[n], lw=1.8, ms=4,
                       label=f"{n} (PSO)")
        if div_x:
            ax[1].plot(div_x, [CEIL * 0.9] * len(div_x), "x", color=_COLOR[n],
                       ms=8, mew=1.8)
    ax[1].axvspan(0.3, 2.9, color="gray", alpha=0.06)
    ax[1].set_ylim(FLOOR, CEIL)
    ax[1].set_title("(b) PSO-tuned classical vs robust family — full α₄ sweep\n"
                    "(−20 % damping);  ✗ = diverges")
    ax[1].set_xlabel("milling-force-coefficient factor  (× average α₄)")
    ax[1].set_ylabel("settled RMS displacement (µm, log)")
    ax[1].legend(ncol=2, fontsize=8)

    fig.tight_layout()
    out = os.path.join(RESULTS, "fig_pso_tuning.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")

    # print a compact verdict table.  Recompute robustness HONESTLY from the
    # validation (stable AND actually controlling: <50 um and off the 150 V rail),
    # not the lenient "did not numerically diverge" flag.
    print("\n=== PSO tuning verdict ===")
    for n in names:
        d = data.get(n)
        if not d:
            print(f"  {n}: (no result)"); continue
        val = d["validation"]
        robust = all((not v["diverged"]) and v["rms_um"] is not None
                     and v["rms_um"] < 50.0 and v["peak_V"] < 147.0
                     for v in val.values())
        worst = max((v["rms_um"] if v["rms_um"] is not None else 9e9)
                    for v in val.values())
        gains = ", ".join(f"{k}={v:.4g}" for k, v in d["gains"].items())
        verdict = "ROBUST across α4" if robust else "NOT robust"
        print(f"  {n:5s}: {verdict:16s} worst-cond={worst:6.1f} um  cost={d['cost']:.1f}  [{gains}]")


if __name__ == "__main__":
    if "--logs" in sys.argv:
        LOGDIR = sys.argv[sys.argv.index("--logs") + 1]
    main()
