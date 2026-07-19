"""
fopid_control.py
================
Fractional-order PID (FOPID / PI^lambda D^mu) controller for chatter
suppression, embeddable in the same closed-loop semi-discretization (CL-SD)
monodromy as the LQG and ADRC designs so its stability-lobe diagram is genuine.

Controller
----------
The fractional PID transfer function from the measured displacement error
e = r - y (r = 0, so e = -y) to the piezo voltage u is

    C(s) = Kp + Ki s^{-lambda} + Kd s^{mu},     0 < lambda, mu < 1,

so u(s) = C(s) e(s) = -C(s) y(s).  Relative to an integer PID, the two extra
knobs (lambda, mu) give independent low- and high-frequency phase shaping, which
is attractive for a lightly-damped resonant workpiece: the fractional
differentiator adds damping to mode 1 with a gentler high-frequency gain slope
than a pure derivative (less spillover to the 3-4 kHz modes), and the fractional
integrator suppresses the low-frequency forced response without the full 90 deg
phase lag of an integer integrator.

Realization
-----------
The irrational operators s^{mu} and s^{-lambda} are approximated over the band
[wb, wh] by the Oustaloup recursive filter of order N (2N+1 poles/zeros):

    s^alpha ~ wh^alpha * prod_{k=-N}^{N} (s + wz_k)/(s + wp_k),
    wz_k = wb (wh/wb)^{(k+N+0.5(1+alpha))/(2N+1)},
    wp_k = wb (wh/wb)^{(k+N+0.5(1-alpha))/(2N+1)}.

Each biproper first-order section (s+wz)/(s+wp) = 1 + (wz-wp)/(s+wp) is realized
as a scalar state (A=-wp, B=1, C=wz-wp, D=1) and the sections are cascaded in
series; the two fractional branches are then summed in parallel with the
proportional feed-through.  This first-order-section cascade is numerically far
better conditioned than the companion form of the degree-(2N+1) polynomial,
whose coefficients span ~20 decades over this band.

The result is a strictly-causal LTI controller

    z_dot = Ac z + Bc y,   u = Cc z + Dc y

(feed-through Dc != 0 because the fractional operators are biproper).  This is
exactly the interface consumed by cl_fdm.ClosedLoopFDM.rho_dynamic (linear
a_p,crit with the controller inside the Floquet monodromy) and, after ZOH
discretization, by the Newmark time-domain feasibility metric.  No plant
observer and no state model are needed -- only the single measured signal y.

Author: contribution built on the model of
        Du et al., Int. J. Mech. Sci. 274 (2024) 109257.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


# ----------------------------------------------------------------------
# Oustaloup recursive approximation of s^alpha over [wb, wh]
# ----------------------------------------------------------------------
def oustaloup_zpk(alpha, wb, wh, N):
    """Zeros, poles (negative reals) and gain of the Oustaloup filter for
    s^alpha, valid for -1 < alpha < 1 over the band [wb, wh].

    Zeros use the (1-alpha) exponent and poles the (1+alpha) exponent, so that
    a positive alpha (differentiator) places each zero below its paired pole and
    the magnitude rises at +alpha*20 dB/dec across the band.  The overall gain is
    normalized numerically at the geometric-mean frequency wc = sqrt(wb wh) so
    that |G(j wc)| = wc^alpha exactly, independent of the exponent convention.
    """
    if not (-1.0 < alpha < 1.0):
        raise ValueError("Oustaloup requires -1 < alpha < 1")
    ks = np.arange(-N, N + 1)
    r = wh / wb
    wz = wb * r ** ((ks + N + 0.5 * (1.0 - alpha)) / (2 * N + 1))
    wp = wb * r ** ((ks + N + 0.5 * (1.0 + alpha)) / (2 * N + 1))
    zeros = -wz
    poles = -wp
    # numerical gain normalization at the band centre
    wc = np.sqrt(wb * wh)
    g_raw = np.prod((1j * wc - zeros) / (1j * wc - poles))
    gain = (wc ** alpha) / abs(g_raw)
    return zeros, poles, gain


def _zpk_to_ss_cascade(zeros, poles, gain):
    """Biproper (#zeros == #poles) real zpk -> (A, B, C, D) as a series
    cascade of first-order sections.  zeros, poles are negative reals."""
    zeros = np.asarray(zeros, float)
    poles = np.asarray(poles, float)
    assert zeros.size == poles.size
    A = np.zeros((0, 0)); B = np.zeros((0, 1))
    C = np.zeros((1, 0)); D = np.array([[1.0]])
    for z, p in zip(zeros, poles):
        wz, wp = -z, -p                     # positive corner frequencies
        Aa, Ba = np.array([[-wp]]), np.array([[1.0]])
        Ca, Da = np.array([[wz - wp]]), np.array([[1.0]])
        # series fold: existing (A,B,C,D) then new section (Aa,Ba,Ca,Da)
        na = A.shape[0]
        A_new = np.zeros((na + 1, na + 1))
        A_new[:na, :na] = A
        A_new[na:, :na] = Ba @ C
        A_new[na:, na:] = Aa
        B_new = np.zeros((na + 1, 1))
        B_new[:na] = B
        B_new[na:] = Ba @ D
        C_new = np.zeros((1, na + 1))
        C_new[:, :na] = Da @ C
        C_new[:, na:] = Ca
        D_new = Da @ D
        A, B, C, D = A_new, B_new, C_new, D_new
    C = C * gain
    D = D * gain
    return A, B, C, D


def _parallel(ssa, ssb):
    """Parallel connection (outputs summed, common input)."""
    Aa, Ba, Ca, Da = ssa
    Ab, Bb, Cb, Db = ssb
    na, nb = Aa.shape[0], Ab.shape[0]
    A = np.zeros((na + nb, na + nb))
    A[:na, :na] = Aa
    A[na:, na:] = Ab
    B = np.vstack([Ba, Bb])
    C = np.hstack([Ca, Cb])
    D = Da + Db
    return A, B, C, D


class FOPIDController:
    """Fractional-order PID controller  C(s) = Kp + Ki s^{-lambda} + Kd s^{mu}.

    Parameters
    ----------
    plate    : PlateModel (uses D_obs for the measured row, H_Pe_modal only to
               report b0; the controller itself needs no plant model)
    dt       : sample time [s]
    Kp,Ki,Kd : PID gains  (map y [m] -> u [V])
    lam, mu  : fractional integral / derivative orders in (0, 1)
    wb, wh   : Oustaloup band [rad/s]
    N        : Oustaloup order (2N+1 sections per fractional branch)
    sign     : overall feedback sign (+1: u = -C(s) y ; -1: u = +C(s) y).
               Chosen at design time so the loop is dissipative for the plant's
               collocated coupling sign.
    u_max    : actuator saturation [V]
    """

    def __init__(self, plate, dt, Kp, Ki, Kd, lam, mu,
                 wb=2 * np.pi * 5.0, wh=2 * np.pi * 15000.0, N=4,
                 sign=1.0, u_max=150.0, verbose=False):
        self.plate = plate
        self.dt = float(dt)
        self.n_x = 2 * plate.n_modes
        self.u_max = float(u_max)
        self.Kp, self.Ki, self.Kd = float(Kp), float(Ki), float(Kd)
        self.lam, self.mu = float(lam), float(mu)
        self.wb, self.wh, self.N = float(wb), float(wh), int(N)
        self.sign = float(sign)
        self.b0 = float(plate.D_obs @ plate.H_Pe_modal)
        self.needs_kstep = False

        # --- build C(s) = Kp + Ki s^{-lam} + Kd s^{mu} as one LTI (y -> u) ---
        zI, pI, kI = oustaloup_zpk(-self.lam, wb, wh, N)   # s^{-lambda}
        zD, pD, kD = oustaloup_zpk(self.mu, wb, wh, N)      # s^{mu}
        ssI = _zpk_to_ss_cascade(zI, pI, kI)
        ssD = _zpk_to_ss_cascade(zD, pD, kD)
        # scale branches by their gains Ki, Kd (multiply output matrices)
        ssI = (ssI[0], ssI[1], self.Ki * ssI[2], self.Ki * ssI[3])
        ssD = (ssD[0], ssD[1], self.Kd * ssD[2], self.Kd * ssD[3])
        A, B, C, D = _parallel(ssI, ssD)
        D = D + np.array([[self.Kp]])        # proportional feed-through

        # negative feedback with chosen loop sign: u = -sign * C(s) * y
        self.Ac = A
        self.Bc = B
        self.Cc = -self.sign * C
        self.Dc = -self.sign * D
        self.nz = A.shape[0]

        # ZOH discretization for the time-domain step()
        Ad = expm(A * self.dt)
        try:
            M = np.linalg.solve(A, Ad - np.eye(self.nz))
            Bd = M @ B
        except np.linalg.LinAlgError:
            Bd = B * self.dt
        self.Ad, self.Bd = Ad, Bd
        self.z = np.zeros(self.nz)

        if verbose:
            print(f"[FOPID] Kp={Kp:.3g} Ki={Ki:.3g} Kd={Kd:.3g} "
                  f"lam={lam:.2f} mu={mu:.2f} sign={sign:+.0f} "
                  f"nz={self.nz} b0={self.b0:+.4f}")

    # ------------------------------------------------------------------
    def reset(self):
        self.z = np.zeros(self.nz)

    # ------------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas, k_step=0):
        """One FOPID update.  x_hat_prev / u_prev are ignored (the controller
        keeps its own state and is a pure y -> u filter).  A zero-padded
        n_x-vector is returned for the Newmark solver's buffer."""
        u = float(self.Cc[0] @ self.z + self.Dc[0, 0] * y_meas)
        self.z = self.Ad @ self.z + self.Bd[:, 0] * y_meas
        u = float(np.clip(u, -self.u_max, self.u_max))
        packed = np.zeros(self.n_x)
        m = min(self.nz, self.n_x)
        packed[:m] = self.z[:m]
        return packed, u

    def discretize_observer(self):           # API compatibility
        pass

    # ------------------------------------------------------------------
    def export_lti(self):
        """Continuous LTI controller (y -> u) for closed-loop semi-discretization:
        z_dot = Ac z + Bc y,  u = Cc z + Dc y.  Returns (Ac, Bc, Cc, Dc)."""
        return self.Ac, self.Bc, self.Cc, self.Dc

    # ------------------------------------------------------------------
    def freqresp(self, w):
        """Controller frequency response C_fb(jw) = Cc (jwI - Ac)^-1 Bc + Dc
        (the realized y->u map, including the feedback sign)."""
        w = np.atleast_1d(np.asarray(w, float))
        out = np.empty(w.size, complex)
        I = np.eye(self.nz)
        for i, wi in enumerate(w):
            out[i] = (self.Cc @ np.linalg.solve(1j * wi * I - self.Ac, self.Bc)
                      + self.Dc)[0, 0]
        return out
