"""
afc_adrc.py
===========
AFC-ADRC: collocated ADRC augmented with tip-driven Adaptive Feedforward
Cancellation (synchronous FxLMS harmonic comb).

Motivation (manuscript Secs. 4.9-4.10)
--------------------------------------
The full-process simulation showed that a single collocated sensor gives the
ESO an outstanding but *local* view: tip vibration is rejected 2x better than
LQG near the sensor yet degrades with tool-sensor distance, because the far-
side residual -- concentrated at the tooth-passing harmonics k*f_t -- is
barely observable from the patch corner.  Blending the tip signal into the
fast ESO loop is structurally impossible here (the tip is non-minimum-phase:
its mode-2/4/5 couplings are large and wrong-signed, and any right-side
w-sensor flips mode 2), so the tip sensor -- already present on the rig of
Du et al. -- is used OUTSIDE the fast loop:

    u = u_ADRC(y_col) + u_ff,
    u_ff(t) = sum_k  a_k cos(k theta(t)) + b_k sin(k theta(t)),

with theta locked to the tooth-passing period (spindle-synchronous, exactly
known on the machine) and the Fourier coefficients adapted by normalized
filtered-x LMS on the TIP error:

    a_k <- a_k - g_k * y_tip * |G_k| cos(k theta + phi_k) * dt,
    g_k = 2 / (tau_a * |G_k|^2),

where G_k is the *closed-loop* secondary path from the injection to the tip
at the k-th harmonic.  Two implementation details are essential (both were
found the hard way and are covered by regression tests in the study script):

1. The injection u_ff is a KNOWN input to the ESO (it sees the total applied
   voltage), so the secondary path must be computed with the augmented input
   matrix [B; B_o] -- plant AND observer channels.  Using [B; 0] yields
   ~90-280 deg phase errors and divergence.
2. The comb must be phase-locked to the actual tooth period (theta from the
   rounded n_per used by the discrete implementation); a 0.4 Hz frequency
   mismatch already spoils the cancellation.

Because u_ff is (slowly-adapted) feedforward, the fast-loop poles are those
of the plain ADRC: the CL-SD stability boundaries of the manuscript are
unchanged.  The adaptation itself is a slow outer loop (time constant tau_a
~ 0.3 s >> tooth period 4.1 ms); its convergence follows the standard FxLMS
condition |phase error of G_k| < 90 deg, which the study script checks under
machining-state and frequency-drift perturbations.

Author: contribution built on the model of
        Du et al., Int. J. Mech. Sci. 274 (2024) 109257.
"""
from __future__ import annotations

import numpy as np

from adrc_control import ADRCController


class AFCADRCController:
    """Collocated ADRC + spindle-synchronous FxLMS comb on the tip error.

    Parameters
    ----------
    design_plate : PlateModel of the design state, with D_obs = collocated row
                   and D_tip = tip row (e.g. via full_process_sim.refresh_rows)
    dt           : sample time [s]
    wc, wo       : ADRC bandwidths [rad/s]
    n_per        : tooth-passing period in samples (comb phase reference)
    K            : number of comb harmonics
    tau_a        : adaptation time constant [s]
    uff_max      : feedforward clamp [V]
    t_enable     : adaptation enable time [s] (let the ESO settle first)
    zeta         : modal damping list for the design model
    """

    dual = True                     # tells the simulator to pass both sensors

    def __init__(self, design_plate, dt, wc, wo, n_per,
                 K=5, tau_a=0.3, uff_max=30.0, t_enable=0.2,
                 b0=None, u_max=150.0, zeta=None, tau_leak=1.0):
        p = design_plate
        self.dt = dt
        self.n_per = int(n_per)
        self.K = int(K)
        self.uff_max = float(uff_max)
        self.k_enable = int(round(t_enable / dt))
        self.u_max = float(u_max)

        b0_col = float(p.D_obs @ p.H_Pe_modal) if b0 is None else float(b0)
        self.adrc = ADRCController(p, dt=dt, wc=wc, wo=wo, b0=b0_col,
                                   u_max=u_max)

        # --- closed-loop secondary path G_k : u_ff -> y_tip -----------------
        n = p.n_modes
        nx = 2 * n
        if zeta is None:
            zeta = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035][:n]
        A = np.zeros((nx, nx))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.diag(p.omega_n ** 2)
        A[n:, n:] = -np.diag(2 * np.array(zeta) * p.omega_n)
        B = np.zeros((nx, 1))
        B[n:, 0] = p.H_Pe_modal
        C_col = np.zeros((1, nx)); C_col[0, :n] = p.D_obs
        C_tip = np.zeros((1, nx)); C_tip[0, :n] = p.D_tip
        Ac, Bc, Cc, _ = self.adrc.export_lti()
        na = nx + 3
        Aa = np.zeros((na, na))
        Aa[:nx, :nx] = A
        Aa[:nx, nx:] = B @ Cc
        Aa[nx:, :nx] = Bc @ C_col
        Aa[nx:, nx:] = Ac
        Ba = np.zeros((na, 1))
        Ba[:nx] = B
        Ba[nx:, 0] = self.adrc.B_o          # the ESO sees the injection too
        Ca = np.zeros((1, na)); Ca[0, :nx] = C_tip

        f_t = 1.0 / (self.n_per * dt)       # effective tooth-passing frequency
        self.gain = np.zeros(self.K)
        self.phase = np.zeros(self.K)
        for k in range(1, self.K + 1):
            w = 2 * np.pi * f_t * k
            Gk = (Ca @ np.linalg.solve(1j * w * np.eye(na) - Aa, Ba))[0, 0]
            self.gain[k - 1] = abs(Gk)
            self.phase[k - 1] = np.angle(Gk)
        self.g_ad = 2.0 / (tau_a * self.gain ** 2)   # per-harmonic normalized
        self.sigma = 1.0 / tau_leak                  # leaky-LMS coefficient decay
        #   Leakage bounds the coefficients when the secondary-path phase error
        #   exceeds the FxLMS 90-deg condition (e.g. under strong frequency
        #   drift): cancellation degrades gracefully instead of pumping.

        self.a = np.zeros(self.K)
        self.b = np.zeros(self.K)
        self.ks = np.arange(1, self.K + 1)

    # ------------------------------------------------------------------
    def reset(self):
        self.adrc.reset()
        self.a[:] = 0.0
        self.b[:] = 0.0

    # ------------------------------------------------------------------
    def step_dual(self, u_prev, y_col, y_tip, k_step):
        """One update: ESO on the collocated signal, comb adapted on the tip."""
        z = (self.adrc.Ad @ self.adrc.z + self.adrc.Bd_u * u_prev
             + self.adrc.Bd_y * y_col)
        self.adrc.z = z
        u0 = (self.adrc.kp * (0.0 - z[0]) - self.adrc.kd * z[1] - z[2]) / self.adrc.b0

        u_ff = 0.0
        if k_step >= self.k_enable:
            th = 2 * np.pi * self.ks * (k_step % self.n_per) / self.n_per
            cth = np.cos(th); sth = np.sin(th)
            u_ff = float(np.clip(self.a @ cth + self.b @ sth,
                                 -self.uff_max, self.uff_max))
            xc = self.gain * np.cos(th + self.phase)
            xs = self.gain * np.sin(th + self.phase)
            self.a -= (self.g_ad * y_tip * xc + self.sigma * self.a) * self.dt
            self.b -= (self.g_ad * y_tip * xs + self.sigma * self.b) * self.dt
            np.clip(self.a, -self.uff_max, self.uff_max, out=self.a)
            np.clip(self.b, -self.uff_max, self.uff_max, out=self.b)

        return float(np.clip(u0 + u_ff, -self.u_max, self.u_max))
