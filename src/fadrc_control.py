"""
fadrc_control.py
================
ADRC-FOPID (fractional-order ADRC) -- the strategy of thesis Chapter 2,
developed for the thin flexible plate.

Chapter-2 lineage (beam, one dominant mode):
    a third-order extended state observer (ESO) estimates (y, y', f_total),
    and the control law applies a FRACTIONAL PID to the estimated output
    error before cancelling the estimated total disturbance:

        u = [ -Kp z1 - Ki I^lambda z1 - Kd D^mu z1  -  z3 ] / b0        (r = 0)

    with the gains and fractional orders tuned by a population metaheuristic.

Developments required by the plate (this module):
    the tip measurement of the plate is non-minimum phase, which defeats the
    textbook ESO (Sec. 4.11 of the study).  The chapter-2 architecture is
    therefore carried over with four plate-specific developments, each LTI:
      1. SIGNED, FREE effective gain b0_eff (not the instantaneous D.H product)
         so the ESO can lock onto the mode-1 subplant whose tip coupling has a
         definite sign;
      2. band-limited observer (wo a free design parameter, not a fixed
         multiple of a control bandwidth);
      3. optional leakage eps on the disturbance state (bounds the estimate
         under saturation);
      4. optional first-order roll-off wf on the injected voltage (protects
         the unmodelled 3.4-4.1 kHz modes).
    The fractional operators I^lambda and D^mu are realized by the same
    validated Oustaloup machinery as fopid_control.py (N = 3, band kept below
    the Nyquist frequency), so the whole controller is a strictly-causal LTI
    system and embeds UNCHANGED in the closed-loop semi-discretization
    monodromy: its controlled stability boundaries are genuine.

Author: contribution built on the model of
        Du et al., Int. J. Mech. Sci. 274 (2024) 109257.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from fopid_control import oustaloup_zpk, _zpk_to_ss_cascade


def _zoh(A, B, dt):
    """Exact ZOH (Ad, Bd) via the augmented-matrix exponential (A may be
    singular -- the closed ESO has an exact integrator direction)."""
    n = A.shape[0]
    B = B.reshape(n, -1)
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * dt
    M[:n, n:] = B * dt
    E = expm(M)
    return E[:n, :n], E[:n, n:]


class FADRCController:
    """ADRC-FOPID of Chapter 2, developed for the plate (tip sensor).

    Parameters
    ----------
    plate    : PlateModel (D_obs = the measurement row; used for n_x only)
    dt       : sample time [s]
    Kp,Ki,Kd : fractional-PID gains applied to the ESO estimate z1
    lam, mu  : fractional integral / derivative orders in (0, 1)
    wo       : ESO bandwidth [rad/s] (triple pole -wo)
    b0_eff   : SIGNED effective input gain used by the ESO and the division
    eps_leak : leakage on the disturbance state z3 [1/s] (0 = none)
    wf       : optional roll-off corner on the injected voltage [rad/s]
    wb,wh,N  : Oustaloup band/order for I^lambda and D^mu
    u_max    : saturation [V]
    """

    def __init__(self, plate, dt, Kp, Ki, Kd, lam, mu, wo, b0_eff,
                 eps_leak=0.0, wf=None,
                 wb=2 * np.pi * 5.0, wh=2 * np.pi * 8000.0, N=3,
                 u_max=150.0, verbose=False):
        self.plate = plate
        self.dt = float(dt)
        self.n_x = 2 * plate.n_modes
        self.u_max = float(u_max)
        self.Kp, self.Ki, self.Kd = float(Kp), float(Ki), float(Kd)
        self.lam, self.mu = float(lam), float(mu)
        self.b0 = float(b0_eff)
        if abs(self.b0) < 1e-12:
            raise ValueError("b0_eff must be nonzero")
        self.wo = float(wo)
        self.eps_leak = float(eps_leak)
        self.wf = None if wf is None else float(wf)

        # ---- ESO (leaky) ----
        b1, b2, b3 = 3 * wo, 3 * wo ** 2, wo ** 3
        A_o = np.array([[-b1, 1.0, 0.0],
                        [-b2, 0.0, 1.0],
                        [-b3, 0.0, -self.eps_leak]])
        B_o = np.array([0.0, self.b0, 0.0])
        L_o = np.array([b1, b2, b3])
        self.A_o, self.B_o, self.L_o = A_o, B_o, L_o

        # ---- fractional branches on z1 (Oustaloup, series cascades) ----
        zI, pI, kI = oustaloup_zpk(-self.lam, wb, wh, N)     # I^lambda
        zD, pD, kD = oustaloup_zpk(self.mu, wb, wh, N)       # D^mu
        AI, BI, CI, DI = _zpk_to_ss_cascade(zI, pI, kI)
        AD, BD, CD, DD = _zpk_to_ss_cascade(zD, pD, kD)
        self.AI, self.BI = AI, BI
        self.CI, self.DI = self.Ki * CI, self.Ki * DI        # y_I = Ki I^lam z1
        self.AD, self.BD = AD, BD
        self.CD, self.DD = self.Kd * CD, self.Kd * DD        # y_D = Kd D^mu z1
        self.nI, self.nD = AI.shape[0], AD.shape[0]

        # ---- exported continuous augmented controller (y -> u) ----
        # states: [z_e(3), zI(nI), zD(nD), (z_lp)]
        # u_raw = -(Kp z1 + y_I + y_D + z3)/b0
        nz = 3 + self.nI + self.nD + (1 if self.wf is not None else 0)
        iE, iI = 0, 3
        iD = 3 + self.nI
        iL = 3 + self.nI + self.nD
        # row vector giving u_raw from the augmented state
        Cu = np.zeros((1, nz))
        Cu[0, iE + 0] = -(self.Kp + self.DI[0, 0] + self.DD[0, 0]) / self.b0
        Cu[0, iE + 2] = -1.0 / self.b0
        Cu[0, iI:iI + self.nI] = -self.CI[0] / self.b0
        Cu[0, iD:iD + self.nD] = -self.CD[0] / self.b0
        Ac = np.zeros((nz, nz))
        Bc = np.zeros((nz, 1))
        # fractional filters driven by z1
        Ac[iI:iI + self.nI, iI:iI + self.nI] = self.AI
        Ac[iI:iI + self.nI, iE + 0:iE + 1] = self.BI
        Ac[iD:iD + self.nD, iD:iD + self.nD] = self.AD
        Ac[iD:iD + self.nD, iE + 0:iE + 1] = self.BD
        if self.wf is None:
            # ESO sees the total (raw) voltage: u = Cu z
            Ac[:3, :] += np.outer(B_o, Cu.ravel())
            Ac[:3, :3] += A_o
            Bc[:3, 0] = L_o
            Cc = Cu.copy()
        else:
            # z_lp' = -wf z_lp + wf u_raw ; u = z_lp ; ESO sees z_lp
            Ac[:3, :3] = A_o
            Ac[:3, iL] = B_o
            Ac[iL, :] = self.wf * Cu.ravel()
            Ac[iL, iL] += -self.wf
            Bc[:3, 0] = L_o
            Cc = np.zeros((1, nz))
            Cc[0, iL] = 1.0
        Dc = np.zeros((1, 1))
        self.Ac, self.Bc, self.Cc, self.Dc = Ac, Bc, Cc, Dc
        self.nz = nz
        self.iE, self.iI, self.iD, self.iL = iE, iI, iD, iL

        # ---- discrete implementation blocks ----
        self.Ad_e, Bd_e = _zoh(A_o, np.column_stack([B_o, L_o]), dt)
        self.Bd_u, self.Bd_y = Bd_e[:, 0], Bd_e[:, 1]
        self.Ad_I, self.Bd_I = _zoh(self.AI, self.BI, dt)
        self.Ad_D, self.Bd_D = _zoh(self.AD, self.BD, dt)
        self.a_lp = None if self.wf is None else float(np.exp(-self.wf * dt))

        self.z_e = np.zeros(3)
        self.zI = np.zeros(self.nI)
        self.zD = np.zeros(self.nD)
        self.z_lp = 0.0
        self.needs_kstep = False
        if verbose:
            print(f"[FADRC] Kp={Kp:.3g} Ki={Ki:.3g} Kd={Kd:.3g} lam={lam:.2f} "
                  f"mu={mu:.2f} wo={wo:.0f} b0={self.b0:+.3g} eps={eps_leak:.3g} "
                  f"wf={wf} nz={nz}")

    # ------------------------------------------------------------------
    def reset(self):
        self.z_e = np.zeros(3)
        self.zI = np.zeros(self.nI)
        self.zD = np.zeros(self.nD)
        self.z_lp = 0.0

    # ------------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas, k_step=0):
        """One update: ESO advances with the previous APPLIED voltage
        (saturation-honest); fractional filters run on z1; fractional law."""
        self.z_e = (self.Ad_e @ self.z_e + self.Bd_u * u_prev
                    + self.Bd_y * y_meas)
        z1, z2, z3 = self.z_e
        y_I = float(self.CI[0] @ self.zI + self.DI[0, 0] * z1)
        y_D = float(self.CD[0] @ self.zD + self.DD[0, 0] * z1)
        self.zI = self.Ad_I @ self.zI + self.Bd_I[:, 0] * z1
        self.zD = self.Ad_D @ self.zD + self.Bd_D[:, 0] * z1
        u_raw = -(self.Kp * z1 + y_I + y_D + z3) / self.b0
        if self.a_lp is not None:
            self.z_lp = self.a_lp * self.z_lp + (1.0 - self.a_lp) * u_raw
            u_raw = self.z_lp
        u = float(np.clip(u_raw, -self.u_max, self.u_max))
        packed = np.zeros(self.n_x)
        m = min(3, self.n_x)
        packed[:m] = self.z_e[:m]
        return packed, u

    def discretize_observer(self):           # API compatibility
        pass

    # ------------------------------------------------------------------
    def export_lti(self):
        """Continuous LTI controller (y -> u) for the CL-SD monodromy."""
        return self.Ac, self.Bc, self.Cc, self.Dc

    # ------------------------------------------------------------------
    def freqresp(self, w):
        w = np.atleast_1d(np.asarray(w, float))
        out = np.empty(w.size, complex)
        I = np.eye(self.nz)
        for i, wi in enumerate(w):
            out[i] = (self.Cc @ np.linalg.solve(1j * wi * I - self.Ac, self.Bc)
                      + self.Dc)[0, 0]
        return out
