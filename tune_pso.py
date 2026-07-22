#!/usr/bin/env python3
"""
PSO tuning driver — re-tune the gain-based controllers (PID, SMC, ADRC) on the
paper's non-minimum-phase plant with a ROBUST cost (see src/tuning_pso.py).

Writes results/pso_tuning.json with, for each controller, the PSO-optimal gains,
the best cost, and a full-horizon validation across the robustness set.  Prints a
readable summary so the honest outcome (which controllers PSO can make robust,
and which are structurally un-tunable) is visible.

Run:  python tune_pso.py            (~15-25 min; 3 controllers x PSO swarm)
"""
from __future__ import annotations
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

from src import tuning_pso as T

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


def main(names=("PID", "SMC", "ADRC"), n_particles=14, n_iters=16, seed=0):
    all_out = {}
    for name in names:
        t0 = time.time()
        print(f"\n===== PSO tuning: {name} "
              f"({n_particles} particles x {n_iters} iters) =====", flush=True)
        best_kwargs, best_cost, val, hist = T.tune(name, n_particles, n_iters, seed)
        dt = time.time() - t0
        # genuinely robust = every condition stable AND actually controlling
        # (not merely "not diverged": a controller pinned to the 150 V rail with a
        # 100+ um residual is a failure even though it never numerically blows up).
        robust = all((not v["diverged"]) and v["rms_um"] is not None
                     and v["rms_um"] < 50.0 and v["peak_V"] < 0.98 * 150.0
                     for v in val.values())
        all_out[name] = dict(gains={k: float(v) for k, v in best_kwargs.items()},
                             cost=best_cost, robust=robust, validation=val,
                             history=[round(h, 2) for h in hist],
                             seconds=round(dt, 1))
        print(f"\n[{name}] done in {dt:.0f}s  cost={best_cost:.2f}  robust={robust}")
        print(f"[{name}] gains: " +
              ", ".join(f"{k}={v:.4g}" for k, v in best_kwargs.items()))
        for lab, v in val.items():
            s = "DIVERGES" if v["diverged"] else f"{v['rms_um']:6.2f} um ({v['peak_V']:.0f} V)"
            print(f"        {lab:8s}: {s}")

    # one file per invocation (so parallel single-controller runs don't clash)
    fname = "pso_tuning.json" if len(names) > 1 else f"pso_{names[0]}.json"
    with open(os.path.join(RESULTS, fname), "w") as f:
        json.dump(all_out, f, indent=2)

    print("\n================= PSO SUMMARY =================")
    for name, o in all_out.items():
        verdict = "ROBUST across alpha4 range" if o["robust"] else "still NOT robust (diverges somewhere)"
        print(f"  {name:5s}: {verdict}")
    print(f"\nWrote {os.path.join(RESULTS, 'pso_tuning.json')}")


if __name__ == "__main__":
    import sys
    names = tuple(sys.argv[1:]) if len(sys.argv) > 1 else ("PID", "SMC", "ADRC")
    main(names=names)
