"""
closed_loop_sld.py
==================
Certified closed-loop stability lobe diagrams for actively controlled milling
of a flexible workpiece.

WHY THIS MODULE EXISTS
----------------------
The usual practice for reporting a "closed-loop" stability lobe diagram is to

    1. design a controller,
    2. read the closed-loop damping ratios zeta_cl off the closed-loop poles,
    3. substitute zeta_cl into the OPEN-LOOP single-mode lobe formula.

That substitution is not the closed loop. A dynamic output-feedback
controller (observer + gain, sampled, zero-order held) changes the DIMENSION
and the STRUCTURE of the monodromy operator: the observer states are extra
states driven by the delayed regenerative term through the measurement, and
the hold introduces an intersample response that no equivalent viscous
damping reproduces. This module builds the monodromy matrix of the ACTUAL
sampled closed loop, so the lobe is certified rather than inferred.

FORMULATION
-----------
Plant in modal coordinates (mass-normalised, n modes):

    q_ddot + Cp q_dot + Kp q + a4(t) Dp Dp^T q
        = a4(t) Dp Dp^T q(t - tau) + ft a3(t) Dp + H u(t)

With x = [q; q_dot] in R^{2n}:

    x_dot = A(t) x + Ad(t) x(t - tau) + B u(t)
    A(t)  = [[0, I], [-(Kp + a4(t) Dp Dp^T), -Cp]]
    Ad(t) = [[0, 0], [ a4(t) Dp Dp^T,          0]]
    B     = [0; H]

MULTI-MODE COUPLING: Dp Dp^T is a rank-one n x n matrix coupling every
retained mode through the single tool contact point. Taking one Floquet
radius per mode and keeping the maximum is NOT the spectral radius of the
coupled system; the full coupled n-mode system is propagated here.

Controller: sampled dynamic output feedback, timed exactly as the
time-domain simulator implements it (measure, then compute, then apply, so
the measurement entering step i is the plant state at step i and the
resulting command acts from step i+1 onwards):

    xhat_{k+1} = Ao xhat_k + Gu u_k + Gy C x_k
    u_{k+1}    = -K xhat_{k+1}                     (zero-order held)

MULTI-RATE: the controller has a fixed hardware sample period T_s, while the
semi-discretisation step must divide the tooth-passing period
tau = 60/(NT*RPM), which changes with spindle speed. The period is therefore
split into m_ctrl controller samples, each subdivided into `refine` plant
sub-steps, giving m_div = m_ctrl*refine plant steps of dt = tau/m_div, with
the command held constant across the sub-steps of one controller period.
Mesh convergence is obtained by raising `refine` WITHOUT changing the
controller rate; raising both together would silently change the system being
analysed.

Semi-discretisation (Insperger & Stepan, zeroth order), a4 frozen to a4_i on
plant interval i:

    x_{i+1} = Phi_i x_i + Gamma_i x_{i-m_div} + Psi_i u_i
    Phi_i   = exp(A_i dt)
    Gamma_i = (int_0^dt exp(A_i s) ds) Ad_i
    Psi_i   = (int_0^dt exp(A_i s) ds) B

Augmented state z_i = [x_i; x_{i-1}; ...; x_{i-m_div}; xhat_i], dimension
2n(m_div+1) + 2n. The monodromy matrix over one period is the ordered product
of the m_div step matrices; the loop is asymptotically stable iff its
spectral radius is < 1.

Passing ctrl_design=None recovers the open-loop lobe through exactly the same
code path, so open- and closed-loop lobes are strictly comparable.

Reference for the underlying discretisation:
    T. Insperger, G. Stepan, "Updated semi-discretization method for periodic
    delay-differential equations with discrete delay", Int. J. Numer. Meth.
    Engng 61 (2004) 117-141.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from milling_force import milling_force_coeffs


# ---------------------------------------------------------------------
# Discretisation helpers
# ---------------------------------------------------------------------
def _step_matrices(A, Ad, B, dt):
    """
    (Phi, Gamma, Psi) for one frozen-coefficient interval.

    A single augmented exponential yields S = int_0^dt exp(A s) ds without
    inverting A (A is singular for a zero-frequency mode and ill-conditioned
    for lightly damped high-frequency modes):

        expm([[A, I], [0, 0]] dt) = [[exp(A dt), S], [0, I]]
    """
    n2 = A.shape[0]
    Maug = np.zeros((2 * n2, 2 * n2))
    Maug[:n2, :n2] = A
    Maug[:n2, n2:] = np.eye(n2)
    E = expm(Maug * dt)
    Phi, S = E[:n2, :n2], E[:n2, n2:]
    return Phi, S @ Ad, S @ B


def discretize_controller(A, B, C, L_kal, Ts):
    """
    Zero-order-hold discretisation of the Kalman observer at sample period Ts.

    Returns (Ao, Gu, Gy) for xhat_{k+1} = Ao xhat_k + Gu u_k + Gy y_k, matching
    lqg_controller.discretize_observer but at an arbitrary Ts, so the
    controller can be re-timed onto the semi-discretisation grid.
    """
    n2 = A.shape[0]
    A_obs = A - L_kal @ C
    Maug = np.zeros((2 * n2, 2 * n2))
    Maug[:n2, :n2] = A_obs
    Maug[:n2, n2:] = np.eye(n2)
    E = expm(Maug * Ts)
    Ao, S = E[:n2, :n2], E[:n2, n2:]
    return Ao, S @ B, S @ L_kal


# ---------------------------------------------------------------------
# Monodromy
# ---------------------------------------------------------------------
def build_monodromy(omega_n, zeta_n, Dp, H_modal, D_obs,
                    a4_array, dt, m_div, refine=1, ctrl=None):
    """
    Monodromy matrix of the sampled closed loop over one tooth-passing period.

    Parameters
    ----------
    omega_n, zeta_n : (n,) modal frequencies [rad/s] and damping ratios
    Dp       : (n,) mode shapes at the tool contact point
    H_modal  : (n,) modal force per volt of the actuator [N/V]
    D_obs    : (n,) mode shapes at the sensor point
    a4_array : (m_div,) regenerative coefficient over one period
    dt       : plant sub-step, = tau / m_div
    m_div    : total plant sub-steps per period
    refine   : plant sub-steps per controller sample
    ctrl     : dict(Ao=, Gu=, Gy=, K=) or None for the open loop
    """
    n = len(omega_n)
    n2 = 2 * n

    Kp = np.diag(np.asarray(omega_n, float) ** 2)
    Cp = np.diag(2.0 * np.asarray(zeta_n, float) * np.asarray(omega_n, float))
    Dp = np.asarray(Dp, float).reshape(n)
    DpDpT = np.outer(Dp, Dp)

    B = np.zeros((n2, 1))
    B[n:, 0] = np.asarray(H_modal, float).reshape(n)
    C = np.zeros((1, n2))
    C[0, :n] = np.asarray(D_obs, float).reshape(n)

    closed = ctrl is not None
    N = n2 * (m_div + 1) + (n2 if closed else 0)
    s = n2 * (m_div + 1)                       # observer block offset

    if closed:
        Ao = np.asarray(ctrl['Ao'], float)
        Gu = np.asarray(ctrl['Gu'], float).reshape(n2, 1)
        Gy = np.asarray(ctrl['Gy'], float).reshape(n2, 1)
        Kf = np.asarray(ctrl['K'], float).reshape(1, n2)
        GyC = Gy @ C
        Ahat = Ao - Gu @ Kf        # xhat_{k+1} = Ahat xhat_k + Gy C x_k

    # The step matrix D_i is almost entirely a shift register: only the first
    # block row (the new plant state) and the observer block row carry
    # non-trivial entries. Forming D_i explicitly and doing a dense N x N
    # product would cost O(N^3) per step for a matrix that is O(n2 * N) dense.
    # Instead the two live block rows are updated directly and the delay line
    # is shifted, which is exact and ~three orders of magnitude cheaper.
    P = np.eye(N)
    top = slice(0, n2)
    tail = slice(n2 * m_div, n2 * (m_div + 1))
    obs = slice(s, s + n2)

    for i in range(m_div):
        alpha_i = float(a4_array[i])

        A = np.zeros((n2, n2))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -(Kp + alpha_i * DpDpT)
        A[n:, n:] = -Cp

        Ad = np.zeros((n2, n2))
        Ad[n:, :n] = alpha_i * DpDpT

        Phi_i, Gam_i, Psi_i = _step_matrices(A, Ad, B, dt)

        # both new block rows are formed from the OLD P before the shift
        new_top = Phi_i @ P[top] + Gam_i @ P[tail]
        if closed:
            new_top -= (Psi_i @ Kf) @ P[obs]
            if i % refine == 0:                      # controller sample instant
                new_obs = GyC @ P[top] + Ahat @ P[obs]
            else:                                    # observer holds
                new_obs = P[obs].copy()

        P[n2:n2 * (m_div + 1)] = P[0:n2 * m_div]     # advance the delay line
        P[top] = new_top
        if closed:
            P[obs] = new_obs

    return P


def spectral_radius(omega_n, zeta_n, Dp, H_modal, D_obs,
                    RPM, NT, RT, eta_h, phi_st, phi_ex,
                    ap, hp, k1, k2, kt,
                    Ts=5e-5, refine=1, ctrl_design=None):
    """
    Spectral radius of the sampled closed loop at one (RPM, a_p) point.

    Ts : controller sample period [s]. The tooth-passing period is split into
         m_ctrl = max(1, round(tau/Ts)) controller samples; the effective
         period tau/m_ctrl re-discretises the observer so that plant and
         controller are exactly co-timed.
    ctrl_design : dict(A=, B=, C=, K=, L=) continuous-time design, or None for
         the open loop.
    """
    tau = 60.0 / (NT * RPM)
    m_ctrl = max(1, int(round(tau / Ts)))
    m_div = m_ctrl * refine
    dt = tau / m_div
    Ts_eff = tau / m_ctrl
    Omega = 2.0 * np.pi * RPM / 60.0

    a4_array = np.empty(m_div)
    for i in range(m_div):
        _, a4 = milling_force_coeffs(i * dt, Omega, NT, RT, eta_h,
                                     phi_st, phi_ex, hp - ap, hp, k1, k2, kt)
        a4_array[i] = a4

    ctrl = None
    if ctrl_design is not None:
        Ao, Gu, Gy = discretize_controller(
            ctrl_design['A'], ctrl_design['B'], ctrl_design['C'],
            ctrl_design['L'], Ts_eff)
        ctrl = dict(Ao=Ao, Gu=Gu, Gy=Gy, K=ctrl_design['K'])

    Phi_T = build_monodromy(omega_n, zeta_n, Dp, H_modal, D_obs,
                            a4_array, dt, m_div, refine=refine, ctrl=ctrl)
    return float(np.max(np.abs(np.linalg.eigvals(Phi_T))))


def critical_depth(fn_rho, lo=1e-7, hi=1e-2, tol=1e-7):
    """Bisect a_p such that rho(a_p) = 1; fn_rho must be increasing in a_p."""
    if fn_rho(lo) >= 1.0:
        return lo
    if fn_rho(hi) < 1.0:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if fn_rho(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
