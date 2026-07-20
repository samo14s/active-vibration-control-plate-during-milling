"""Offline tuning of the regenerative-targeted delayed feedback gain.

For each scheduling grid point theta_k, the scalar gain k_r is chosen by
maximizing the closed-loop critical axial depth of cut computed by
semi-discretization of the complete delayed closed loop (plant +
controller + delayed tap), subject to a voltage-budget proxy:

    k_r(theta_k) = argmax_k  a_lim^CL(theta_k, k)
                   s.t.      |k| <= k_max

The voltage budget enters through k_max (voltage per meter of recirculated
displacement; k_max = beta * v_max / w_scale with w_scale the expected
vibration scale) and is re-checked a posteriori in time-domain simulation.
A robust golden-section-free approach is used: coarse scan + local refine,
because a_lim(k) can be multi-modal.
"""
from __future__ import annotations

import numpy as np

from .controller import Controller
from .modal import ModalModel
from . import sld


def tune_kr(mm: ModalModel, x_T: float, rpm: float, base: Controller,
            k_max: float, n_coarse: int = 13, n_refine: int = 8,
            m: int = 48, ap_hi: float = 5e-3) -> tuple[float, float]:
    """Return (k_r_opt, a_lim_opt) for one grid point.

    base must have c_wmill set (observer-structure controller).
    """
    def alim(k: float) -> float:
        ctrl = base.with_delayed_feedback(k) if k != 0.0 else base
        return sld.critical_depth(mm, x_T, rpm, controller=ctrl,
                                  m=m, ap_hi=ap_hi)

    # scan in order of increasing |k| and require a STRICT improvement to
    # move to a larger gain: when the closed loop already sits at the
    # search ceiling ap_hi, the tuning stays at the smallest gain that
    # attains it (voltage economy) instead of picking an arbitrary large
    # one.
    ks = np.array(sorted(np.linspace(-k_max, k_max, n_coarse), key=abs))
    vals = np.array([alim(k) for k in ks])
    tol_a = 1e-6
    i0 = 0
    for i in range(1, len(ks)):
        if vals[i] > vals[i0] + tol_a:
            i0 = i
    k_best, a_best = ks[i0], vals[i0]
    if a_best <= vals[np.argmin(np.abs(ks))] + tol_a:
        return 0.0, float(vals[np.argmin(np.abs(ks))])

    step = k_max / max(n_coarse - 1, 1)
    lo, hi = k_best - step, k_best + step
    lo, hi = max(lo, -k_max), min(hi, k_max)
    for _ in range(n_refine):
        cand = np.array([lo + (hi - lo) * f for f in (0.25, 0.5, 0.75)])
        cv = np.array([alim(k) for k in cand])
        for j in range(len(cand)):
            better = cv[j] > a_best + tol_a
            tie_smaller = (abs(cv[j] - a_best) <= tol_a
                           and abs(cand[j]) < abs(k_best))
            if better or tie_smaller:
                k_best, a_best = cand[j], cv[j]
        width = (hi - lo) * 0.5
        lo, hi = k_best - width / 2.0, k_best + width / 2.0
        lo, hi = max(lo, -k_max), min(hi, k_max)

    return float(k_best), float(a_best)


def tune_grid(models: dict, x_grid: np.ndarray, rpm: float,
              controllers: dict, k_max: float, **kw) -> dict:
    """Tune k_r over the scheduling grid.

    models: {removal: ModalModel}; controllers: {(x_T, removal): Controller}.
    Returns {(x_T, removal): (k_r, a_lim)}.
    """
    out = {}
    for removal, mm in models.items():
        for x_T in x_grid:
            base = controllers[(float(x_T), float(removal))]
            out[(float(x_T), float(removal))] = tune_kr(
                mm, float(x_T), rpm, base, k_max, **kw)
    return out
