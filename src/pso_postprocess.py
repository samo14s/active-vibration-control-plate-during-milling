"""
Post-process PSO tuning: validate the PSO-optimal gains on the FULL alpha4 sweep
and the moving 20.4 s pass (the real tests, beyond the 4 discrete cost
conditions), and build a PSO convergence figure from the run logs.

Used by make_pso_figure.py; kept importable so the numbers can be reused.
"""
from __future__ import annotations
import re
import numpy as np

from . import config as C
from . import plant as P
from . import metrics as M
from .simulate import simulate


def full_alpha4_sweep(make_ctrl, factors=None, dzeta=-0.2, t_sim=0.3):
    """Settled RMS (or None if diverged) across a fine alpha4 sweep at -20% damping."""
    if factors is None:
        factors = [0.3, 0.6, 0.9, 1.3, 1.6, 2.0, 2.3, 2.6, 2.9]
    out = []
    for f in factors:
        r = simulate(P.MillingPlant(alpha4_factor=f, dzeta=dzeta), make_ctrl(),
                     t_sim=t_sim, meas_noise=True)
        m = M.compute(r)
        out.append(None if m["diverged"] else m["rms_settled_um"])
    return list(factors), out


def moving_pass(make_ctrl, t_sim=None):
    """Run the full moving+drifting pass; return (meanRMS or None, diverged, t_end)."""
    t_sim = C.T_PASS if t_sim is None else t_sim
    r = simulate(P.MillingPlant(moving=True, freq_drift=True), make_ctrl(),
                 t_sim=t_sim, meas_noise=True)
    m = M.compute(r)
    y_um = r["y"] * 1e6
    n = len(y_um)
    mean_rms = None if m["diverged"] else float(np.sqrt(np.mean(y_um[n // 4:] ** 2)))
    return mean_rms, bool(m["diverged"]), float(r["t"][-1]) if n else 0.0


def parse_convergence(log_path):
    """Extract the gbest-vs-iteration series from a PSO run log."""
    its, gs = [], []
    pat = re.compile(r"iter\s+(\d+)/\d+\s+gbest=\s*([0-9.eE+-]+)")
    ini = re.compile(r"init\s+gbest=\s*([0-9.eE+-]+)")
    try:
        with open(log_path) as f:
            for line in f:
                mi = ini.search(line)
                if mi:
                    its.append(0); gs.append(float(mi.group(1))); continue
                mm = pat.search(line)
                if mm:
                    its.append(int(mm.group(1))); gs.append(float(mm.group(2)))
    except FileNotFoundError:
        pass
    return its, gs
