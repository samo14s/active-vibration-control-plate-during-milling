"""Cross-part learning of the stability boundary (Sec. 3.3, Eq. (23)).

A Gaussian-process regression over (spindle speed, removed-volume
fraction) learns a multiplicative correction to the model-based stability
boundary from the outcomes observed on previous workpieces.  Where the
process ran stably close to the predicted boundary, the correction (and
hence the usable depth of cut) grows; where vibration was high, it
shrinks.

This implements the mechanism behind the paper's progressive improvement
across identical workpieces (+8 % MRR by the later parts).  In OUR
calibrated plant the realised cross-part gain is small (~0-2 %, within
trial-to-trial scatter): the binding constraint is usually the
forced-response cap rather than a conservatively-predicted stability
boundary, so there is little conservatism for the GP to reclaim.  See
results/REPORT.md, "Known deviations".
"""

from __future__ import annotations

import numpy as np


class BoundaryGP:
    def __init__(self, noise: float = 0.05,
                 len_n_rpm: float = 2500.0, len_vr: float = 0.25,
                 max_points: int = 400):
        self.noise = noise
        self.len_n = len_n_rpm
        self.len_vr = len_vr
        self.max_points = max_points
        self.X: list[tuple[float, float]] = []    # (n_rpm, Vr)
        self.y: list[float] = []                  # log-correction targets
        self._chol = None
        self._alpha = None
        self._calls = 0
        self._since_fit = 0

    # -- data collection ----------------------------------------------------
    def observe(self, n_rpm: float, Vr: float, ap_mm: float,
                ap_lim_mm: float, rms_um: float) -> None:
        """Record an outcome sample (subsampled to ~0.25 s cadence)."""
        self._calls += 1
        if self._calls % 25 != 0 or ap_lim_mm <= 0.1:
            return
        util = ap_mm / ap_lim_mm
        if rms_um < 18.0 and util > 0.5:
            target = np.log(1.12)                 # boundary was conservative
        elif rms_um > 28.0:
            target = np.log(0.85)                 # boundary was optimistic
        else:
            target = 0.0
        self.X.append((n_rpm, Vr))
        self.y.append(target)
        if len(self.X) > self.max_points:         # keep a bounded memory
            self.X = self.X[:: 2]
            self.y = self.y[:: 2]
        self._since_fit += 1
        if self._since_fit >= 40:                 # refit in batches
            self._chol = None
            self._since_fit = 0

    # -- GP machinery, Eq. (23) ----------------------------------------------
    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        dn = (A[:, None, 0] - B[None, :, 0]) / self.len_n
        dv = (A[:, None, 1] - B[None, :, 1]) / self.len_vr
        return np.exp(-0.5 * (dn * dn + dv * dv))

    def _fit(self) -> None:
        X = np.array(self.X)
        y = np.array(self.y)
        K = self._kernel(X, X) + self.noise ** 2 * np.eye(len(X))
        self._chol = np.linalg.cholesky(K)
        from scipy.linalg import solve_triangular
        t = solve_triangular(self._chol, y, lower=True)
        self._alpha = solve_triangular(self._chol.T, t, lower=False)
        self._X = X

    def correction(self, n_grid_rpm: np.ndarray,
                   Vr: float | None = None) -> np.ndarray:
        """Multiplicative boundary correction over a speed grid."""
        if not self.X:
            return np.ones_like(n_grid_rpm, dtype=float)
        if self._chol is None:
            self._fit()
        vr = Vr if Vr is not None else float(np.mean([x[1] for x in self.X]))
        Xs = np.column_stack([n_grid_rpm,
                              np.full(len(n_grid_rpm), vr)])
        ks = self._kernel(Xs, self._X)
        mu = ks @ self._alpha                     # predictive mean, Eq. (23)
        return np.clip(np.exp(mu), 0.75, 1.25)
