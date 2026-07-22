"""
Active Disturbance Rejection Control (linear ADRC / LADRC).

The plate is treated as a double integrator driven by a lumped "total
disturbance" f (which absorbs all modal dynamics, the regenerative cutting
force and the parameter perturbations) plus a known input gain b0:

        y'' = f + b0 u

A 3rd-order linear Extended State Observer (LESO) estimates (y, y', f):

        z' = (Ao - Lo Co) z + Bo u + Lo y ,   Lo = [3wo, 3wo^2, wo^3]

and the control cancels the estimated disturbance and imposes a critically
damped 2nd-order target of bandwidth wc:

        u = ( wc^2 (r - z1) - 2 wc z2 - z3 ) / b0

The LESO is discretised exactly (ZOH) so it stays stable at the high observer
bandwidth needed to keep up with the ~1 kHz chatter mode.
"""

from __future__ import annotations
import numpy as np
from .base import Controller
from .. import config as C
from .. import design as D


class ADRC(Controller):
    name = "ADRC"
    color = "#a142f4"      # purple

    def __init__(self, b0=26.0, wo=3.0e4, wc=4.0e3, **kw):
        self.b0 = b0
        self.wo = wo
        self.wc = wc
        super().__init__(**kw)
        self._design()

    def _design(self):
        wo = self.wo
        Ao = np.array([[0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 0.0, 0.0]])
        Bo = np.array([[0.0], [self.b0], [0.0]])
        Co = np.array([[1.0, 0.0, 0.0]])
        Lo = np.array([[3 * wo], [3 * wo ** 2], [wo ** 3]])
        # discretise observer  z' = (Ao - Lo Co) z + [Bo, Lo] [u; y]
        Aobs = Ao - Lo @ Co
        Bobs = np.hstack([Bo, Lo])            # inputs: u, y
        self.Phi, self.Gam = D.c2d(Aobs, Bobs, self.dt)
        self.Co = Co

    def reset(self):
        super().reset()
        self.z = np.zeros(3)

    def _update(self, t, y):
        z1, z2, z3 = self.z
        u0 = self.wc ** 2 * (0.0 - z1) - 2.0 * self.wc * z2
        u = (u0 - z3) / self.b0
        u = float(np.clip(u, -self.u_max, self.u_max))
        # observer step with applied (saturated) u and measured y
        inp = np.array([u, y])
        self.z = self.Phi @ self.z + self.Gam @ inp
        return u
