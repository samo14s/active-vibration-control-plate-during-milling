"""
Sliding-Mode Control (SMC) — output/boundary-layer formulation.

A model-inversion (equivalent-control) SMC is ill-conditioned for this plant: the
piezo actuator is weak relative to the modal restoring forces, so forcing the
state onto a stiff surface would demand enormous voltage.  Instead we use an
*output sliding-mode* design, which is the robust counterpart of the velocity
feedback that physically damps the plate.

A steady-state Kalman observer reconstructs the modal state from the measured
displacement y, giving estimates y_hat and y_dot_hat.  The sliding variable

        s = y_dot_hat + lam * y_hat

defines a stable first-order sliding motion (y_dot = -lam y on s = 0).  The
control enforces the reaching law  s_dot = -k s - eta sat(s/phi):

        u = -(1/b0) ( k s + eta sat(s/phi) )

The linear term k s injects damping (the nominal action); the saturated
switching term eta sat(s/phi) rejects the *matched* uncertainty — the
regenerative cutting force and parameter perturbations — giving SMC its
robustness.  The boundary layer phi replaces the discontinuous sign() to avoid
control chattering that a piezo stack could not follow.
"""

from __future__ import annotations
import numpy as np
from .base import Controller
from .. import config as C
from .. import design as D


class SMC(Controller):
    name = "SMC"
    color = "#188038"      # green

    def __init__(self, lam=1.0e4, k=9.0, eta=300.0, phi=0.05, b0=26.0,
                 kf_q=5e-2, kf_r=None, n_modes=3, **kw):
        self.lam = lam            # sliding-surface slope
        self.k = k                # linear reaching gain
        self.eta = eta            # switching (robustness) gain
        self.phi = phi            # boundary-layer half-width (on s)
        self.b0 = b0              # input gain  y'' ~ b0 u
        self.kf_q = kf_q
        self.kf_r = (C.MEAS_NOISE_STD ** 2) if kf_r is None else kf_r
        self.n_modes = n_modes
        super().__init__(**kw)
        self._design()

    def _design(self):
        pl = D.design_plant(n_modes=self.n_modes)
        self.A, self.Bu, self.Cy = pl["A"], pl["Bu"], pl["Cy"]
        n = pl["n"]
        # output rows for displacement and velocity at the sensor
        self.Cpos = pl["Cy"]                                   # [phi, 0]
        self.Cvel = np.hstack([np.zeros((1, n)), pl["phi"].reshape(1, -1)])  # [0, phi]
        self.Ad, self.Bd = D.c2d(self.A, self.Bu, self.dt)
        self.L = D.kalman_gain(self.Ad, self.Cy, self.kf_q, self.kf_r)
        self.nx = 2 * n

    def reset(self):
        super().reset()
        if hasattr(self, "nx"):
            self.xhat = np.zeros(self.nx)

    @staticmethod
    def _sat(x):
        return np.clip(x, -1.0, 1.0)

    def _update(self, t, y):
        xh = self.xhat
        y_hat = float((self.Cpos @ xh).item())
        yd_hat = float((self.Cvel @ xh).item())
        s = yd_hat + self.lam * y_hat
        u = -(self.k * s + self.eta * self._sat(s / self.phi)) / self.b0
        u = float(np.clip(u, -self.u_max, self.u_max))
        # observer update with applied (saturated) input
        yhat_meas = float((self.Cy @ xh).item())
        self.xhat = self.Ad @ xh + self.Bd.flatten() * u + self.L.flatten() * (y - yhat_meas)
        return u
