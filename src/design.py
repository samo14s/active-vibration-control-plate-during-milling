"""
Shared linear-design utilities: the nominal design plant, discretisation,
Kalman observer and LQR — reused by SMC, ADRC and MPC.

The nominal design plant is the lightly damped 3-mode modal model
    x' = A x + Bu u + Bd d ,   y = Cy x
with x = [q ; q'] in R^6.  Milling regeneration is treated as the disturbance d.
"""

from __future__ import annotations
import numpy as np
import scipy.linalg as sla

from . import config as C
from . import plant as P


def design_plant(n_modes=None):
    """Return the nominal continuous design plant matrices (dict).

    n_modes=None keeps all modes; n_modes=2 gives the paper's reduced model.
    """
    return P.design_state_space(n_modes=n_modes)


def c2d(A, B, dt, C_=None, D=None):
    """Zero-order-hold discretisation via the matrix exponential."""
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * dt
    M[:n, n:] = B * dt
    Md = sla.expm(M)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    if C_ is None:
        return Ad, Bd
    return Ad, Bd, C_, (np.zeros((C_.shape[0], m)) if D is None else D)


def lqr(A, B, Q, R):
    """Continuous-time LQR gain K (u = -K x)."""
    Pmat = sla.solve_continuous_are(A, B, Q, np.atleast_2d(R))
    K = np.linalg.solve(np.atleast_2d(R), B.T @ Pmat)
    return K, Pmat


def dlqr(Ad, Bd, Q, R):
    """Discrete-time LQR gain K (u = -K x)."""
    Pmat = sla.solve_discrete_are(Ad, Bd, Q, np.atleast_2d(R))
    K = np.linalg.solve(np.atleast_2d(R) + Bd.T @ Pmat @ Bd, Bd.T @ Pmat @ Ad)
    return K, Pmat


def kalman_gain(Ad, Cd, qn, rn):
    """
    Steady-state discrete Kalman predictor gain L for
        x[k+1] = Ad x[k] + Bd u[k] + w ,  y = Cd x + v
    with process cov Qn = qn * Bd Bd^T (input-driven) and meas cov Rn = rn.
    Returns L such that x_hat[k+1] = Ad x_hat + Bd u + L (y - Cd x_hat).
    """
    n = Ad.shape[0]
    Qn = qn * np.eye(n)
    Rn = np.atleast_2d(rn)
    Pmat = sla.solve_discrete_are(Ad.T, Cd.T, Qn, Rn)
    L = (Ad @ Pmat @ Cd.T) @ np.linalg.inv(Cd @ Pmat @ Cd.T + Rn)
    return L
