"""Performance metrics for the milling-pass comparison."""
from __future__ import annotations

import numpy as np

from .simulate import SimResult


def pass_window(res: SimResult, t0: float, t1: float) -> np.ndarray:
    return (res.t >= t0) & (res.t <= t1)


def summarize(res: SimResult, pass_time: float,
              settle: float = 0.5) -> dict:
    """Global metrics over the milling pass (excluding entry transient)."""
    m = pass_window(res, settle, pass_time)
    w = res.w_cut[m]
    V = res.V[m]
    out = {
        "rms_wc": float(np.sqrt(np.mean(w ** 2))),
        "max_wc": float(np.max(np.abs(w))),
        "mean_wc": float(np.mean(w)),
        "rms_ac_wc": float(np.std(w)),           # AC part (vibration proper)
        "v_rms": float(np.sqrt(np.mean(V ** 2))),
        "v_peak": float(np.max(np.abs(V))),
        "sat_frac": float(np.mean((V <= -499.99) | (V >= 1499.99))),
    }
    if res.surf_w is not None and len(res.surf_w):
        sm = (res.surf_t >= settle) & (res.surf_t <= pass_time)
        out["surf_std"] = float(np.std(res.surf_w[sm]))
        out["surf_ptp"] = float(np.ptp(res.surf_w[sm]))
    return out


def along_path_rms(res: SimResult, pass_time: float, n_bins: int = 40,
                   settle: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """AC-RMS of the cutting-point deflection binned by cutter position."""
    m = pass_window(res, settle, pass_time)
    x, w = res.x_cut[m], res.w_cut[m]
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    cx, rms = [], []
    for i in range(n_bins):
        sel = (x >= edges[i]) & (x < edges[i + 1])
        if np.count_nonzero(sel) > 10:
            cx.append(0.5 * (edges[i] + edges[i + 1]))
            rms.append(np.std(w[sel]))
    return np.array(cx), np.array(rms)


def uniformity(cx: np.ndarray, rms: np.ndarray) -> float:
    """Coefficient of variation of the along-path RMS (lower = more even)."""
    return float(np.std(rms) / np.mean(rms))


def reduction_db(ref: float, val: float) -> float:
    return float(20.0 * np.log10(ref / val))
