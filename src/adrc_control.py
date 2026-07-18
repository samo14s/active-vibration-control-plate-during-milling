"""
adrc_control.py
===============
Active Disturbance Rejection Control (ADRC) for chatter suppression.

Motivation for thin-walled milling
----------------------------------
The dominant difficulties of milling flexible workpieces -- the regenerative
(delayed) cutting force, the intermittent periodic excitation, multi-mode
participation, and dynamics that vary along the tool path -- are exactly what
ADRC is built to handle.  Rather than modelling each of these, ADRC lumps
everything that is not the known control channel into a single *total
disturbance* f(t), estimates it online with an extended state observer (ESO),
and cancels it.  The only model information required is the input-output gain
b0; this makes ADRC intrinsically robust to the parameter variations that defeat
a fixed model-based design, which is the central challenge highlighted by
Du et al. (2024).

Design (linear ADRC, Gao 2003 bandwidth parameterisation)
---------------------------------------------------------
The plant seen from the piezo voltage u to the measured tip displacement
y = D_obs . q has relative degree two:

    y_ddot = f(t) + b0 u,      b0 = D_obs . H_pe .

A third-order linear ESO estimates (y, y_dot, f):

    z1_dot = z2 + beta1 (y - z1)
    z2_dot = z3 + b0 u + beta2 (y - z1)
    z3_dot =        beta3 (y - z1)          (z3 -> f, the total disturbance)

with observer gains placed at a triple pole -wo:
    beta1 = 3 wo,  beta2 = 3 wo^2,  beta3 = wo^3.

The control law cancels the estimated disturbance and imposes a double pole -wc:

    u = ( kp (r - z1) - kd z2 - z3 ) / b0,   kp = wc^2,  kd = 2 wc,   r = 0.

Because z3 already contains the regenerative and periodic cutting forces, a
single ADRC loop both extends the stability boundary and rejects the forced
vibration -- unlike the two-degree-of-freedom split (feedback + feedforward).

The controller is linear and time-invariant, so its true stability-lobe diagram
is obtained by embedding it (ESO included) in the closed-loop full-discretization
solver via :meth:`export_lti` and
:meth:`cl_fdm.ClosedLoopFDM.rho_dynamic`.

Author: contribution built on the model of
        Du et al., Int. J. Mech. Sci. 274 (2024) 109257.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


class ADRCController:
    """Linear ADRC (third-order ESO + PD-on-estimate control law).

    Parameters
    ----------
    plate : PlateModel with D_obs and H_Pe_modal set
    dt    : sample time [s]
    wc    : controller bandwidth [rad/s] (closed-loop double pole -wc)
    wo    : observer bandwidth  [rad/s] (ESO triple pole -wo)
    b0    : input-output gain; default D_obs . H_pe
    u_max : actuator voltage saturation [V]
    """

    def __init__(self, plate, dt, wc, wo, b0=None, u_max=150.0, verbose=False):
        self.plate = plate
        self.dt = dt
        self.n_x = 2 * plate.n_modes
        self.u_max = u_max
        self.wc = wc
        self.wo = wo
        self.b0 = float(plate.D_obs @ plate.H_Pe_modal) if b0 is None else float(b0)
        self.kp = wc ** 2
        self.kd = 2.0 * wc
        b1, b2, b3 = 3 * wo, 3 * wo ** 2, wo ** 3

        # Continuous ESO: z_dot = A_o z + B_o u + L_o y
        self.A_o = np.array([[-b1, 1.0, 0.0],
                             [-b2, 0.0, 1.0],
                             [-b3, 0.0, 0.0]])
        self.B_o = np.array([0.0, self.b0, 0.0])
        self.L_o = np.array([b1, b2, b3])

        # ZOH discretisation of the ESO (same style as the LQG observer)
        Ad = expm(self.A_o * dt)
        try:
            M = np.linalg.solve(self.A_o, (Ad - np.eye(3)))
            Bd_u = M @ self.B_o
            Bd_y = M @ self.L_o
        except np.linalg.LinAlgError:
            Bd_u = self.B_o * dt
            Bd_y = self.L_o * dt
        self.Ad, self.Bd_u, self.Bd_y = Ad, Bd_u, Bd_y

        self.z = np.zeros(3)
        self.needs_kstep = False
        if verbose:
            print(f"[ADRC] b0={self.b0:.4g}  wc={wc:.0f}  wo={wo:.0f}  "
                  f"kp={self.kp:.3g} kd={self.kd:.3g}")

    # ------------------------------------------------------------------
    def reset(self):
        self.z = np.zeros(3)

    # ------------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas, k_step=0):
        """One ADRC update.  x_hat_prev is ignored (ADRC keeps its own ESO
        state); a zero-padded 6-vector is returned so the Newmark solver can
        store it in its x_hat buffer."""
        # ESO update with the previous command and the current measurement
        self.z = self.Ad @ self.z + self.Bd_u * u_prev + self.Bd_y * y_meas
        z1, z2, z3 = self.z
        u0 = self.kp * (0.0 - z1) - self.kd * z2
        u = (u0 - z3) / self.b0
        u = float(np.clip(u, -self.u_max, self.u_max))
        packed = np.zeros(self.n_x)
        packed[:min(3, self.n_x)] = self.z[:min(3, self.n_x)]
        return packed, u

    def discretize_observer(self):     # API compatibility with LQGController
        pass

    # ------------------------------------------------------------------
    def export_lti(self):
        """Continuous LTI controller (y -> u) for closed-loop FDM.

        With r = 0 the control law is static in z: u = Cc z, and substituting it
        into the ESO gives z_dot = (A_o + B_o Cc) z + L_o y.  Returns
        (Ac, Bc, Cc, Dc) with Ac (3x3), Bc (3x1), Cc (1x3), Dc (1x1).
        """
        Cc = np.array([[-self.kp / self.b0, -self.kd / self.b0, -1.0 / self.b0]])
        Ac = self.A_o + np.outer(self.B_o, Cc.ravel())
        Bc = self.L_o.reshape(3, 1)
        Dc = np.zeros((1, 1))
        return Ac, Bc, Cc, Dc
