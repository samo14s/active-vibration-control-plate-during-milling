"""Semi-discretization stability analysis of the milling closed loop.

First-order semi-discretization (Insperger & Stepan, "Semi-Discretization
for Time-Delay Systems", Springer 2011) of the time-periodic
delay-differential closed loop formed by

* the reduced modal plate model (avc.modal.ModalModel),
* the regenerative milling force with periodic coefficient a4(t)
  (avc.milling.alpha4_time),
* optionally an output-feedback controller with a spindle-synchronized
  delayed state tap (avc.controller.Controller).

Sign conventions (verified against avc/milling.py, the authoritative
physical derivation)
---------------------------------------------------------------------
Plant and regenerative force, all signals SI:

    eta_dd + Z eta_d + W2 eta = bu u + bf Fy
    Fy(t) = a4(t) [w(t-tau) - w(t)]   (+ affine static/feed terms, which
                                       do not affect Floquet stability)
    w(t)  = bf^T eta(t)               (milling-point deflection [m])
    tau   = 60/(N_t rpm) [s],  a4(t) [N/m] signed (<= 0 for this
            down-milling engagement), period of a4 equal to tau.

Moving the current-time part of Fy to the left-hand side:

    eta_dd + Z eta_d + (W2 + a4 bf bf^T) eta = bu u + a4 bf w(t-tau).

ModalModel.abcd(x_T, alpha4=c) assembles the stiffness block
W2 - c bf bf^T, i.e. its ``alpha4`` argument is the coefficient of the
OPPOSITE difference, Fy = c (w(t) - w(t-tau)); it equals MINUS the
signed a4 of avc.milling. This module therefore calls

    abcd(x_T, alpha4=-a4_i)     ->  stiffness  W2 + a4_i bf bf^T

and adds the delayed scalar input

    + Bf * a4_i * w(t-tau)

which together realize exactly Fy = a4_i (w(t-tau) - w(t)). Sign sanity
check: a4 < 0 gives effective stiffness W2 - |a4| bf bf^T (the current
cut softens the structure) and a delayed-feedback gain a4 < 0. Passing
the signed milling a4 directly to abcd would flip the sign of the
current-time cutting stiffness.

Rank-one delay structure
------------------------
Only two SCALARS are delayed: the milling-point deflection
w(t-tau) = Cw x(t-tau) driving the plant, and (when the controller has a
delayed tap) s(t-tau) = gd1 @ xk(t-tau) driving the actuation. The
discrete state is therefore augmented with scalar histories instead of
full-state histories:

    z_k = [ x_k ; w_k, ..., w_{k-m} ; (s_k, ..., s_{k-m}) ]

with x = [eta; eta_dot; xk], so the dimension is (2n + nk) + (m+1) or
(2n + nk) + 2(m+1) instead of the (m+1)(2n+nk) of a full-state scheme.

Discretization
--------------
The tool has equal pitch, so the coefficient period equals the
regenerative delay tau; one period = m intervals, dt = tau/m. On
interval i the periodic coefficient is frozen at its interval mean a4_i
(sub-sampled mean of avc.milling.alpha4_time). The delayed scalar over
the interval is approximated by the linear interpolation of the two
adjacent history samples evaluated at the interval midpoint,
0.5*(w_{k-m} + w_{k-m+1}), held constant (first-order accurate). The
frozen linear ODE with constant input is advanced exactly via the matrix
exponential of the augmented block [[A, B], [0, 0]]*dt (zero-order hold
of the delayed-scalar input). The monodromy matrix is the ordered
product of the m interval maps; the loop is asymptotically stable iff
its spectral radius is below 1.

Benchmark hook
--------------
``transition_matrix`` and ``spectral_radius`` accept ``a4_fn``: a
callable t_array -> a4(t) [N/m] that OVERRIDES avc.milling.alpha4_time
(then ``ap`` does not influence the coefficient). Used by the turning
(constant-coefficient) analytic benchmark in tests/test_sld.py.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from . import milling

_N_SUB = 8  # a4(t) samples per interval used for the interval mean


def _interval_a4(params, ap, rpm, tau, m, a4_fn=None,
                 a4_scale=1.0) -> np.ndarray:
    """Interval means of a4(t) [N/m] over the m intervals of [0, tau]."""
    t = (np.arange(m * _N_SUB) + 0.5) * (tau / (m * _N_SUB))
    if a4_fn is not None:
        a4 = np.asarray(a4_fn(t), dtype=float)
    else:
        a4 = milling.alpha4_time(params, ap, rpm, t)
    return a4_scale * a4.reshape(m, _N_SUB).mean(axis=1)


def transition_matrix(mm, x_T, ap, rpm, controller=None, m=48,
                      a4_fn=None, a4_scale=1.0) -> np.ndarray:
    """Monodromy matrix of the periodic delayed closed loop.

    Parameters
    ----------
    mm : avc.modal.ModalModel (or any object exposing ``params`` with
        ``tooth_passing_delay(rpm)`` and ``abcd(x_T, alpha4)`` affine in
        alpha4 with the ModalModel sign convention).
    x_T : tool position along the feed path [m].
    ap : axial depth of cut [m].
    rpm : spindle speed [rev/min].
    controller : avc.controller.Controller or None (open loop).
    m : intervals per tooth-passing period (= delay resolution).
    a4_fn : optional override of milling.alpha4_time (see module doc).

    Returns
    -------
    Phi : (N, N) monodromy matrix over one period T = tau,
        N = 2n + nk + (m+1) (+ m+1 more when the controller has a
        delayed tap gd1).
    """
    tau = mm.params.tooth_passing_delay(rpm)
    dt = tau / m
    a4 = _interval_a4(mm.params, ap, rpm, tau, m, a4_fn, a4_scale)

    A0, Bu, Bf, Cs, Cw = mm.abcd(x_T, alpha4=0.0)
    Ady = mm.abcd(x_T, alpha4=1.0)[0] - A0   # dA/dalpha4 (A affine in alpha4)
    n2 = A0.shape[0]
    bu = np.asarray(Bu, float).reshape(n2)
    bf_col = np.asarray(Bf, float).reshape(n2)
    cs = np.asarray(Cs, float).reshape(n2)
    cw = np.asarray(Cw, float).reshape(n2)

    if controller is None:
        nk, tap = 0, False
    else:
        nk = controller.nk
        tap = controller.gd1 is not None
    na = n2 + nk

    # closed-loop A, alpha4-independent part:
    #   x_p_dot = A_p x_p + Bu (Ck+gd0) xk + Bu Dk Cs x_p + delayed inputs
    #   xk_dot  = Ak xk + Bk Cs x_p
    Abase = np.zeros((na, na))
    Abase[:n2, :n2] = A0
    Dy = np.zeros((na, na))
    Dy[:n2, :n2] = Ady

    Bw = np.zeros(na)            # delayed w column (scaled by a4_i)
    Bw[:n2] = bf_col
    cw_a = np.zeros(na)          # w_k = cw_a @ x_k
    cw_a[:n2] = cw

    if controller is not None:
        Ak = np.asarray(controller.Ak, float).reshape(nk, nk)
        Bk = np.asarray(controller.Bk, float).reshape(nk)
        Ck = np.asarray(controller.Ck, float).reshape(nk)
        Dk = float(np.asarray(controller.Dk, float).ravel()[0])
        ck_eff = Ck.copy()
        if controller.gd0 is not None:
            ck_eff += np.asarray(controller.gd0, float).reshape(nk)
        Abase[:n2, :n2] += Dk * np.outer(bu, cs)
        Abase[:n2, n2:] = np.outer(bu, ck_eff)
        Abase[n2:, :n2] = np.outer(Bk, cs)
        Abase[n2:, n2:] = Ak
    if tap:
        Bs = np.zeros(na)        # delayed controller-tap column
        Bs[:n2] = bu
        c_tap = np.zeros(na)     # s_k = c_tap @ x_k = gd1 @ xk
        c_tap[n2:] = np.asarray(controller.gd1, float).reshape(nk)

    nh = m + 1                   # stored history scalars per channel
    N = na + nh * (2 if tap else 1)
    iw = na                      # h_w[j] = w_{k-j} at index iw + j
    is_ = na + nh                # h_s[j] = s_{k-j} at index is_ + j
    shift = np.eye(m)

    Phi = np.eye(N)
    for i in range(m):
        # abcd's alpha4 = -a4 (sign flip documented in the module doc)
        Ai = Abase - a4[i] * Dy
        if tap:
            B = np.column_stack([a4[i] * Bw, Bs])
        else:
            B = (a4[i] * Bw)[:, None]
        nc = B.shape[1]
        Maug = np.zeros((na + nc, na + nc))
        Maug[:na, :na] = Ai * dt
        Maug[:na, na:] = B * dt
        E = expm(Maug)           # exact ZOH map of (Ai, B) over dt

        G = np.zeros((N, N))
        G[:na, :na] = E[:na, :na]
        q = E[:na, na]           # int_0^dt e^{Ai s} ds @ B[:, 0]
        G[:na, iw + m] += 0.5 * q        # w_{k-m}   = w(t_k - tau)
        G[:na, iw + m - 1] += 0.5 * q    # w_{k-m+1} = w(t_{k+1} - tau)
        if tap:
            q2 = E[:na, na + 1]
            G[:na, is_ + m] += 0.5 * q2
            G[:na, is_ + m - 1] += 0.5 * q2
        # history shift h^+[j] = h[j-1], and h^+[0] from the new state
        G[iw + 1: iw + nh, iw: iw + m] = shift
        G[iw, :] = cw_a @ G[:na, :]
        if tap:
            G[is_ + 1: is_ + nh, is_: is_ + m] = shift
            G[is_, :] = c_tap @ G[:na, :]
        Phi = G @ Phi
    return Phi


def spectral_radius(mm, x_T, ap, rpm, controller=None, m=48,
                    a4_fn=None, a4_scale=1.0) -> float:
    """Spectral radius of the monodromy matrix (stable iff < 1)."""
    Phi = transition_matrix(mm, x_T, ap, rpm, controller=controller, m=m,
                            a4_fn=a4_fn, a4_scale=a4_scale)
    return float(np.max(np.abs(np.linalg.eigvals(Phi))))


def critical_depth(mm, x_T, rpm, controller=None, m=48, ap_hi=5e-3,
                   tol=2e-6, a4_scale=1.0) -> float:
    """Critical axial depth of cut a_lim [m] at one spindle speed.

    Bisection on ap of the condition spectral_radius < 1. Returns ap_hi
    when still stable at ap_hi, and 0.0 when the loop is already
    unstable as ap -> 0 (e.g. a destabilizing controller). a4_scale
    multiplies the cutting coefficient (robustness Monte-Carlo hook).
    """
    if spectral_radius(mm, x_T, 0.0, rpm, controller, m,
                       a4_scale=a4_scale) >= 1.0:
        return 0.0
    if spectral_radius(mm, x_T, ap_hi, rpm, controller, m,
                       a4_scale=a4_scale) < 1.0:
        return float(ap_hi)
    lo, hi = 0.0, float(ap_hi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if spectral_radius(mm, x_T, mid, rpm, controller, m,
                           a4_scale=a4_scale) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sld_curve(mm, x_T, rpms, controller=None, m=48, ap_hi=5e-3,
              a4_scale=1.0) -> np.ndarray:
    """Stability lobe diagram: critical depth [m] for each rpm in rpms."""
    return np.array([critical_depth(mm, x_T, float(r), controller=controller,
                                    m=m, ap_hi=ap_hi, a4_scale=a4_scale)
                     for r in rpms])
