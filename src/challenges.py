"""
Harder-than-paper control-challenge suite for the SMC family.

Tuned (from an empirical failure-boundary probe) to the regime where the plain
SMC / TD-SMC start to break, so the DEVELOPED ATD-SMC can be measured on genuinely
harder problems than the paper's:

  * EXTREME cutting force alpha4 = 3.5 .. 5.5x  (the paper studies up to 2.9x;
    plain TD-SMC diverges already at 3.5x, plain SMC at ~4x).  The alpha4 >= 4.5x
    cases are near / beyond the finite piezo actuator's physical authority, so
    they cap EVERY controller — they are kept as an honest "ceiling" marker.
  * LARGE material-removal frequency rise +12% / +17% (paper full series; +17%
    breaks the plain SMC) and a NEGATIVE -4% shift (model-overestimation side:
    breaks the plain SMC and exposes the amplitude-schedule blind spot of the
    feedforward-scheduled designs).
  * TIGHT actuator limit u_max = 20 V.
  * COMBINED extreme (alpha4=4.0 at nominal frequency + -40% damping).
  * nominal / low-force guards against regression.

evaluate(make) runs every challenge and returns per-challenge metrics, an
aggregate score (lower = better, divergence penalised) and n_survived (stable AND
actually controlling: settled RMS < 60 um and off the actuator rail).
"""
from __future__ import annotations
import numpy as np

from . import plant as P
from .simulate import simulate


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def dmass_for_pct(pct):
    """dmass giving a +pct% modal-frequency rise (material removal)."""
    return 1.0 / (1.0 + pct / 100.0) ** 2 - 1.0


CHALLENGES = [
    ("nominal",  dict(),                              dict()),
    ("a4_0.3",   dict(alpha4_factor=0.3),             dict()),
    ("a4_2.9",   dict(alpha4_factor=2.9),             dict()),
    ("a4_3.5",   dict(alpha4_factor=3.5),             dict()),
    ("a4_4.5",   dict(alpha4_factor=4.5),             dict()),
    ("a4_5.5",   dict(alpha4_factor=5.5),             dict()),
    ("freq+12%", dict(dmass=dmass_for_pct(12)),       dict()),
    ("freq+17%", dict(dmass=dmass_for_pct(17)),       dict()),
    ("freq-4%",  dict(dmass=dmass_for_pct(-4)),       dict()),
    ("tightV",   dict(alpha4_factor=2.9),             dict(u_max=20.0)),
    ("combo",    dict(alpha4_factor=4.0, dzeta=-0.4), dict()),
]

DIVERGE_PENALTY = 3000.0
SURVIVE_UM = 60.0


def eval_one(make, plant_kw, ctrl_kw, t_sim=0.35):
    try:
        c = make(**ctrl_kw)
        r = simulate(P.MillingPlant(**plant_kw), c, t_sim=t_sim, meas_noise=True)
        y = np.asarray(r["y"]) * 1e6
        u = np.asarray(r["u"])
        n = len(y)
        if r["diverged"] or not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e4:
            return dict(rms=None, peakV=float(np.max(np.abs(u))) if len(u) else 0.0,
                        diverged=True, growth=None)
        mid = _rms(y[n // 3:2 * n // 3]); late = _rms(y[2 * n // 3:])
        return dict(rms=late, peakV=float(np.max(np.abs(u))), diverged=False,
                    growth=late / max(mid, 1e-3))
    except Exception as e:
        return dict(rms=None, peakV=0.0, diverged=True, growth=None, error=str(e))


def evaluate(make, t_sim=0.35, verbose=False):
    """make is a factory: make(**ctrl_kwargs) -> controller."""
    per = {}
    score = 0.0
    n_surv = 0
    for name, pk, ck in CHALLENGES:
        m = eval_one(make, pk, ck, t_sim=t_sim)
        per[name] = m
        if m["diverged"] or m["rms"] is None:
            score += DIVERGE_PENALTY
        else:
            g = max(0.0, (m["growth"] or 1.0) - 1.15)
            score += m["rms"] + 0.02 * m["peakV"] + 200.0 * g
            if m["rms"] < SURVIVE_UM and m["peakV"] < 0.98 * ck.get("u_max", 150.0):
                n_surv += 1
        if verbose:
            s = "DIVERGES" if (m["diverged"] or m["rms"] is None) else \
                f"{m['rms']:6.1f} um / {m['peakV']:3.0f} V"
            print(f"  {name:9s}: {s}")
    return dict(per=per, score=float(score), n_survived=n_surv, n_total=len(CHALLENGES))
