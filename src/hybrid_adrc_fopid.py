"""
hybrid_adrc_fopid.py
====================
HYBRID controller = band-limited ADRC (extended state observer) + fractional-
order PID, both acting on the SAME single tip measurement -- the article's
original sensor (Du et al. 2024 measure the workpiece tip displacement).

Why a hybrid on the tip sensor
------------------------------
The tip measurement is non-minimum phase (modal coupling D_tip.H_pe = -+--+,
right-half-plane zeros in u -> y_tip).  On this sensor the two parent
controllers fail in complementary ways (Sec. 4.11 / fopid_tip_study.py):

  * plain ADRC fails at EVERY bandwidth: a single lumped-disturbance ESO
    inverts the plant through its RHP zeros, so some subset of modes always
    sees positive feedback;
  * plain FOPID stays stable but is weak (0.20 mm): a fixed-structure law
    cannot push enough phase lead through the RHP zeros AND reject the
    mode-1 disturbance at the same time.

The hybrid splits the job by frequency band:

    u = u_ESO + u_FOPID,
    u_ESO   = ( -kp z1 - kd z2 - z3 ) / b0_eff,     kp = wc^2, kd = 2 wc,
    u_FOPID = -sigma ( Kp + Ki s^-lambda + Kd s^mu ) y,

with the ESO bandwidth wo and effective input gain b0_eff *free design
parameters*: a LOW-bandwidth ESO sees essentially the mode-1 subplant, whose
tip coupling has a definite (negative) sign, so with the matching signed
b0_eff its disturbance estimate z3 is coherent exactly where regenerative
chatter lives -- while the FOPID branch shapes the loop where the ESO must not
act.  Both branches are LTI, so the hybrid exports (Ac, Bc, Cc, Dc) and drops
into the closed-loop semi-discretization monodromy unchanged.

Augmented controller (y -> u), with ESO state z_e (3) and FOPID state z_f:

    z_e' = (A_o + B_o Ce) z_e + (B_o Cf) z_f + (L_o + B_o Df) y
    z_f' = A_f z_f + B_f y
    u    = Ce z_e + Cf z_f + Df y

(the ESO sees the TOTAL voltage, hence the B_o couplings).  The discrete
step() mirrors the deployed ADRC implementation instead: the ESO is updated
with the previous *applied* (saturated) voltage -- the same convention as
ADRCController, which keeps the estimate honest under saturation.

Author: contribution built on the model of
        Du et al., Int. J. Mech. Sci. 274 (2024) 109257.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from fopid_control import FOPIDController


class HybridADRCFOPID:
    """Band-limited ESO (ADRC core) + FOPID, single sensor, one voltage output.

    Parameters
    ----------
    plate   : PlateModel (D_obs = the tip row here)
    dt      : sample time [s]
    wc, wo  : ESO controller / observer bandwidths [rad/s]
    b0_eff  : signed effective input gain used by the ESO (free parameter;
              NOT forced to D_obs.H_pe -- on an NMP plant the useful value is
              the low-frequency/mode-1 coupling, opposite in sign to the
              instantaneous gain)
    Kp,Ki,Kd,lam,mu,sign : FOPID branch (see fopid_control.FOPIDController)
    wb,wh,N : Oustaloup realization (kept below Nyquist)
    u_max   : saturation [V]
    eps_leak: ESO disturbance-state leakage [1/s] (v2 improvement).  The plain
              ESO integrates the innovation into z3 without bound; on the NMP
              tip this rams the actuator into a saturated limit cycle at low
              depths.  A leak z3' += -eps_leak*z3 bounds the estimate (an LTI
              change: A_o[2,2] = -eps_leak).  0 recovers the v1 hybrid.
    wf      : optional first-order low-pass corner [rad/s] on the ESO branch
              output (v2 improvement): u_ESO is filtered before injection,
              rolling the mode-1 inversion off before the 3.4-4.1 kHz spillover
              modes.  None recovers the v1 hybrid (no filter state).
    """

    def __init__(self, plate, dt, wc, wo, b0_eff,
                 Kp, Ki, Kd, lam, mu, sign=1.0,
                 wb=2 * np.pi * 5.0, wh=2 * np.pi * 8000.0, N=3,
                 u_max=150.0, eps_leak=0.0, wf=None, verbose=False):
        self.plate = plate
        self.dt = float(dt)
        self.n_x = 2 * plate.n_modes
        self.u_max = float(u_max)
        self.wc, self.wo = float(wc), float(wo)
        self.b0 = float(b0_eff)
        if abs(self.b0) < 1e-9:
            raise ValueError("b0_eff must be nonzero")

        # ---- FOPID branch (reuse the validated realization) ----
        self.fopid = FOPIDController(plate, dt, Kp, Ki, Kd, lam, mu,
                                     wb=wb, wh=wh, N=N, sign=sign,
                                     u_max=u_max)
        Af, Bf, Cf, Df = self.fopid.export_lti()
        self.nzf = Af.shape[0]

        # ---- ESO branch (ADRCController structure + optional v2 leakage) ----
        kp_a, kd_a = wc ** 2, 2.0 * wc
        b1, b2, b3 = 3 * wo, 3 * wo ** 2, wo ** 3
        self.eps_leak = float(eps_leak)
        A_o = np.array([[-b1, 1.0, 0.0],
                        [-b2, 0.0, 1.0],
                        [-b3, 0.0, -self.eps_leak]])
        B_o = np.array([0.0, self.b0, 0.0])
        L_o = np.array([b1, b2, b3])
        Ce = np.array([[-kp_a / self.b0, -kd_a / self.b0, -1.0 / self.b0]])
        self.A_o, self.B_o, self.L_o, self.Ce = A_o, B_o, L_o, Ce
        self.kp_a, self.kd_a = kp_a, kd_a
        self.wf = None if wf is None else float(wf)

        # ZOH discretization of the ESO (u_prev and y as held inputs)
        Ad = expm(A_o * dt)
        try:
            M = np.linalg.solve(A_o, Ad - np.eye(3))
            Bd_u = M @ B_o
            Bd_y = M @ L_o
        except np.linalg.LinAlgError:
            Bd_u, Bd_y = B_o * dt, L_o * dt
        self.Ad_e, self.Bd_u, self.Bd_y = Ad, Bd_u, Bd_y

        # ---- exported continuous augmented controller ----
        if self.wf is None:
            # v1 structure: states [z_e(3), z_f]
            nz = 3 + self.nzf
            Ac = np.zeros((nz, nz))
            Ac[:3, :3] = A_o + np.outer(B_o, Ce.ravel())
            Ac[:3, 3:] = np.outer(B_o, Cf.ravel())
            Ac[3:, 3:] = Af
            Bc = np.zeros((nz, 1))
            Bc[:3, 0] = L_o + B_o * Df[0, 0]
            Bc[3:, :] = Bf
            Cc = np.zeros((1, nz))
            Cc[0, :3] = Ce
            Cc[0, 3:] = Cf
            Dc = Df.copy()
        else:
            # v2 structure: states [z_e(3), z_lp(1), z_f];  u = z_lp + u_FOPID,
            # z_lp' = -wf z_lp + wf Ce z_e  (ESO output low-passed before
            # injection), and the ESO sees the TOTAL voltage u.
            nz = 4 + self.nzf
            Ac = np.zeros((nz, nz))
            Ac[:3, :3] = A_o
            Ac[:3, 3] = B_o                      # u contribution: z_lp
            Ac[:3, 4:] = np.outer(B_o, Cf.ravel())
            Ac[3, :3] = self.wf * Ce.ravel()
            Ac[3, 3] = -self.wf
            Ac[4:, 4:] = Af
            Bc = np.zeros((nz, 1))
            Bc[:3, 0] = L_o + B_o * Df[0, 0]
            Bc[4:, :] = Bf
            Cc = np.zeros((1, nz))
            Cc[0, 3] = 1.0
            Cc[0, 4:] = Cf
            Dc = Df.copy()
        self.Ac, self.Bc, self.Cc, self.Dc = Ac, Bc, Cc, Dc
        self.nz = nz
        self.a_lp = None if self.wf is None else float(np.exp(-self.wf * dt))
        self.z_lp = 0.0

        self.z_e = np.zeros(3)
        self.needs_kstep = False
        if verbose:
            print(f"[HYBRID] wc={wc:.0f} wo={wo:.0f} b0_eff={self.b0:+.3g}  "
                  f"FOPID(lam={lam:.2f}, mu={mu:.2f}, sign={sign:+.0f})  nz={nz}")

    # ------------------------------------------------------------------
    def reset(self):
        self.z_e = np.zeros(3)
        self.z_lp = 0.0
        self.fopid.reset()

    # ------------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas, k_step=0):
        """One hybrid update.  ESO advances with the previous APPLIED voltage
        (saturation-honest, as in ADRCController); FOPID filters y."""
        self.z_e = (self.Ad_e @ self.z_e + self.Bd_u * u_prev
                    + self.Bd_y * y_meas)
        z1, z2, z3 = self.z_e
        u_e = (-self.kp_a * z1 - self.kd_a * z2 - z3) / self.b0
        if self.a_lp is not None:            # v2: low-pass the ESO injection
            self.z_lp = self.a_lp * self.z_lp + (1.0 - self.a_lp) * u_e
            u_e = self.z_lp
        # FOPID branch: output from current state, then advance (strictly causal)
        f = self.fopid
        u_f = float(f.Cc[0] @ f.z + f.Dc[0, 0] * y_meas)
        f.z = f.Ad @ f.z + f.Bd[:, 0] * y_meas
        u = float(np.clip(u_e + u_f, -self.u_max, self.u_max))
        packed = np.zeros(self.n_x)
        m = min(3, self.n_x)
        packed[:m] = self.z_e[:m]
        return packed, u

    def discretize_observer(self):           # API compatibility
        pass

    # ------------------------------------------------------------------
    def export_lti(self):
        """Continuous LTI controller (y -> u) for closed-loop semi-discretization:
        z' = Ac z + Bc y,  u = Cc z + Dc y."""
        return self.Ac, self.Bc, self.Cc, self.Dc

    # ------------------------------------------------------------------
    def freqresp(self, w):
        """Hybrid controller frequency response C(jw): y -> u."""
        w = np.atleast_1d(np.asarray(w, float))
        out = np.empty(w.size, complex)
        I = np.eye(self.nz)
        for i, wi in enumerate(w):
            out[i] = (self.Cc @ np.linalg.solve(1j * wi * I - self.Ac, self.Bc)
                      + self.Dc)[0, 0]
        return out
