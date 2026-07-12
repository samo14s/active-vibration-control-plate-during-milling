"""Real-time modal identification (Sec. 3.1 / 3.3).

Recursive least squares with exponential forgetting on an AR(2) model of
the wall vibration signal, after removal of the spindle-synchronous
content — the paper's "adaptive notch filters that eliminate spindle
rotation harmonics" (Sec. 3.2) — so that the estimator locks onto the
structural mode rather than the tooth-passing forcing.

The AR(2) poles map to the continuous natural frequency and damping ratio
of the dominant wall mode; their evolution tracks Eqs. (6)-(7) online.
"""

from __future__ import annotations

import math

import numpy as np


def remove_harmonics(y: np.ndarray, fs: float, f_rev: float,
                     n_harm: int = 28,
                     protect_hz: float | None = None) -> np.ndarray:
    """Least-squares removal of spindle-synchronous content from a window.

    All harmonics of the spindle rotation frequency ``f_rev`` are removed
    (tooth-passing lines AND runout sidebands), leaving the nonsynchronous
    residual that carries the structural / chatter information — the
    paper's "adaptive notch filters that eliminate spindle rotation
    harmonics" (Sec. 3.2).

    ``protect_hz`` guards the currently identified natural frequency: a
    harmonic falling within 6 % of it is left untouched so the notch never
    swallows the structural mode (the physical system decorrelates the two
    with brief spindle-speed perturbations, Sec. 3.1).
    """
    n = len(y)
    if n < 8 or f_rev <= 0.0:
        return y
    t = np.arange(n) / fs
    cols = [np.ones(n)]
    for h in range(1, n_harm + 1):
        f = h * f_rev
        if f >= 0.45 * fs:
            break
        if protect_hz is not None and abs(f - protect_hz) < 0.06 * protect_hz:
            continue
        cols.append(np.sin(2 * math.pi * f * t))
        cols.append(np.cos(2 * math.pi * f * t))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


class RLSModalIdentifier:
    """RLS AR(2) tracker:  y_k = a1 y_{k-1} + a2 y_{k-2} + e_k."""

    def __init__(self, fs: float, forgetting: float = 0.995,
                 f_lo: float = 500.0, f_hi: float = 3200.0):
        self.fs = fs
        self.lam = forgetting
        self.f_lo, self.f_hi = f_lo, f_hi
        self.theta = np.zeros(2)
        self.P = np.eye(2) * 1e4
        self.fn_hz: float | None = None
        self.zeta: float | None = None
        self.resid_rms: float = 0.0   # nonsynchronous RMS (chatter energy)
        self.n_accepts: int = 0       # count of accepted pole extractions
        self._last = np.zeros(2)
        self._warm = 0

    @property
    def uncertainty(self) -> float:
        """Normalised parameter covariance for gain scheduling, Eq. (25)."""
        return float(np.trace(self.P)) / 2.0

    def update(self, samples: np.ndarray, f_rev: float,
               protect_hz: float | None = None) -> None:
        y = remove_harmonics(np.asarray(samples, float), self.fs, f_rev,
                             protect_hz=(protect_hz if protect_hz is not None
                                         else self.fn_hz))
        s = np.std(y)
        # low-passed chatter-energy tracker (variance analysis, Sec. 3.2)
        self.resid_rms = 0.7 * self.resid_rms + 0.3 * float(s)
        if s < 1e-12:
            return
        y = y / s                                 # scale-invariant tracking
        for k in range(2, len(y)):
            phi = np.array([y[k - 1], y[k - 2]])
            denom = self.lam + phi @ self.P @ phi
            K = (self.P @ phi) / denom
            err = y[k] - self.theta @ phi
            self.theta = self.theta + K * err
            self.P = (self.P - np.outer(K, phi @ self.P)) / self.lam
            self._warm += 1
        self._extract()

    def _extract(self) -> None:
        if self._warm < 200:
            return
        a1, a2 = self.theta
        disc = a1 * a1 + 4.0 * a2                 # poles of z^2 - a1 z - a2
        if disc >= 0.0:
            return                                # real poles: no oscillator
        re = 0.5 * a1
        im = 0.5 * math.sqrt(-disc)
        r = math.hypot(re, im)
        if not (1e-6 < r < 1.0):
            return
        dt = 1.0 / self.fs
        sig = -math.log(r) / dt                   # zeta * wn
        wd = math.atan2(im, re) / dt              # damped frequency
        wn = math.hypot(sig, wd)
        fn = wn / (2.0 * math.pi)
        if not (self.f_lo <= fn <= self.f_hi):
            return
        zeta = sig / wn
        if not (1e-4 < zeta < 0.5):
            return
        # light smoothing: the paper reports 3.2 % identification accuracy
        if self.fn_hz is None:
            self.fn_hz, self.zeta = fn, zeta
        else:
            self.fn_hz = 0.85 * self.fn_hz + 0.15 * fn
            self.zeta = 0.90 * self.zeta + 0.10 * zeta
        self.n_accepts += 1
