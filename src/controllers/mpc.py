"""
Model Predictive Control (linear, constrained).

A steady-state Kalman filter estimates the 3-mode modal state from the measured
displacement; each sample a finite-horizon optimal control problem is solved:

    min_U  sum_{k=1}^{Np} qy * y_k^2  +  sum_{k=0}^{Np-1} ( ru * u_k^2 + rdu * du_k^2 )
    s.t.   x_{k+1} = Ad x_k + Bd u_k ,   |u_k| <= u_max

The problem is condensed to the input sequence U and solved as a box-constrained
QP with a warm-started FISTA (accelerated projected-gradient) iteration — fast
enough to run every 39 us sample.  Only the first move u_0 is applied (receding
horizon).  The output penalty makes MPC act as a constrained optimal regulator
that injects damping into the lightly damped modes and respects the actuator
voltage limit explicitly — the feature that distinguishes it from the others.
"""

from __future__ import annotations
import numpy as np
from .base import Controller
from .. import config as C
from .. import design as D


class MPC(Controller):
    name = "MPC (predictive)"
    color = "#00897b"      # teal

    def __init__(self, Np=45, qy=4e12, ru=1.0e-3, rdu=2.0e-3,
                 kf_q=5e-3, kf_r=None, fista_iters=40, **kw):
        self.Np = Np
        self.qy, self.ru, self.rdu = qy, ru, rdu
        self.kf_q = kf_q
        self.kf_r = (C.MEAS_NOISE_STD ** 2) if kf_r is None else kf_r
        self.fista_iters = fista_iters
        super().__init__(**kw)
        self._design()

    def _design(self):
        pl = D.design_plant()                       # full 3-mode model
        self.A, self.Bu, self.Cy = pl["A"], pl["Bu"], pl["Cy"]
        self.Ad, self.Bd = D.c2d(self.A, self.Bu, self.dt)
        self.Cd = self.Cy
        self.nx = self.Ad.shape[0]
        self.L = D.kalman_gain(self.Ad, self.Cd, self.kf_q, self.kf_r)

        Np, nx = self.Np, self.nx
        Ad, Bd, Cd = self.Ad, self.Bd, self.Cd
        # prediction matrices for the OUTPUT: Y = Sx x0 + Su U
        Sx = np.zeros((Np, nx))
        Su = np.zeros((Np, Np))
        Apow = np.eye(nx)
        # precompute Cd Ad^{k}
        CAk = [Cd @ np.linalg.matrix_power(Ad, k + 1) for k in range(Np)]
        for i in range(Np):
            Sx[i, :] = CAk[i].ravel()
            for j in range(i + 1):
                Su[i, j] = (Cd @ np.linalg.matrix_power(Ad, i - j) @ Bd).item()
        self.Sx, self.Su = Sx, Su

        Qy = self.qy * np.eye(Np)
        # input-rate penalty matrix: du_k = u_k - u_{k-1}
        Dmat = np.eye(Np) - np.eye(Np, k=-1)
        Rdu = self.rdu * (Dmat.T @ Dmat)
        Ru = self.ru * np.eye(Np)
        self.H = Su.T @ Qy @ Su + Ru + Rdu
        self.H = 0.5 * (self.H + self.H.T)
        # gradient pieces: f = Su^T Qy Sx x0  - rdu*(coupling to previous u)
        self.FxT = Su.T @ Qy @ Sx                   # (Np, nx)
        self.Rdu = Rdu
        self.Dmat = Dmat
        # FISTA step size = 1 / Lipschitz (largest eigenvalue of H)
        self.Lip = float(np.max(np.linalg.eigvalsh(self.H)))
        self.step = 1.0 / self.Lip
        self._Uwarm = np.zeros(Np)

    def reset(self):
        super().reset()
        if hasattr(self, "nx"):
            self.xhat = np.zeros(self.nx)
        self._Uwarm = np.zeros(self.Np)
        self.u_prev = 0.0

    def _solve_qp(self, f):
        """Box-constrained QP: min 0.5 U'HU + f'U, |U|<=umax, via FISTA."""
        U = self._Uwarm.copy()
        Y = U.copy()
        t = 1.0
        umax = self.u_max
        H, step = self.H, self.step
        for _ in range(self.fista_iters):
            grad = H @ Y + f
            Un = np.clip(Y - step * grad, -umax, umax)
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            Y = Un + ((t - 1.0) / tn) * (Un - U)
            U, t = Un, tn
        self._Uwarm = U
        return U

    def _update(self, t, y):
        # linear term: includes previous-input coupling from the rate penalty
        f = self.FxT @ self.xhat
        # rate penalty coupling to u_prev: d0 = u0 - u_prev -> adds -rdu*u_prev to f[0]
        f = f.copy()
        f[0] -= self.rdu * self.u_prev
        U = self._solve_qp(f)
        u = float(np.clip(U[0], -self.u_max, self.u_max))
        # observer update
        yhat = float((self.Cd @ self.xhat).item())
        self.xhat = self.Ad @ self.xhat + self.Bd.flatten() * u + self.L.flatten() * (y - yhat)
        self.u_prev = u
        return u
