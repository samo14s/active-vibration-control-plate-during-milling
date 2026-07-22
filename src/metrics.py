"""Performance metrics for a closed-loop milling simulation result."""

from __future__ import annotations
import numpy as np
from . import config as C


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def spectrum(t, y):
    """Single-sided amplitude spectrum of y (windowed)."""
    y = np.asarray(y) - np.mean(y)
    n = len(y)
    w = np.hanning(n)
    Y = np.fft.rfft(y * w)
    f = np.fft.rfftfreq(n, t[1] - t[0])
    amp = 2.0 / np.sum(w) * np.abs(Y)
    return f, amp


def dominant_frequency(t, y, fmin=100.0):
    f, amp = spectrum(t, y)
    m = f >= fmin
    if not np.any(m):
        return 0.0, 0.0
    k = np.argmax(amp[m])
    return float(f[m][k]), float(amp[m][k])


def compute(result, settle_frac=0.4, diverge_thresh_um=1e4):
    """
    Compute a metrics dict from a simulate() result.

    settle_frac : fraction of the record treated as the initial transient;
                  the "settled" RMS uses the remainder.
    """
    t, y, u = result["t"], result["y"], result["u"]
    diverged = bool(result.get("diverged", False))
    y_um = y * 1e6
    n = len(t)
    i0 = int(settle_frac * n)

    peak = float(np.max(np.abs(y_um)))
    settled = y_um[i0:]
    rms_settled = _rms(settled)
    rms_full = _rms(y_um)

    fdom, adom = dominant_frequency(t, y)

    diverged = diverged or (peak > diverge_thresh_um) or not np.all(np.isfinite(y_um))

    return dict(
        rms_settled_um=rms_settled,
        rms_full_um=rms_full,
        peak_um=peak,
        rms_volt=_rms(u),
        peak_volt=float(np.max(np.abs(u))),
        control_energy=float(np.trapezoid(np.asarray(u) ** 2, t)),
        dom_freq_hz=fdom,
        dom_amp_um=adom * 1e6,
        diverged=diverged,
    )
