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

Honest limitation on the paper's non-minimum-phase geometry
-----------------------------------------------------------
LADRC treats the plant as a pure double integrator y'' = f + b0 u and cancels the
estimated total disturbance f — i.e. it *inverts* the plant.  With the piezo patch
at the left-lower corner and the sensor at the right-upper corner the transfer
function u -> y is NON-MINIMUM-PHASE (a right-half-plane zero at ~5 kHz).  Inverting
a non-minimum-phase plant produces an internally unstable controller: the LESO's
disturbance cancellation drives the RHP-zero dynamics and the loop diverges.  The
input gain even changes sign here (b0 = phi_sensor . Hpe = -1.29 < 0).  No choice of
(b0, wo, wc) stabilises this plant — LADRC is therefore reported as FAILING on the
paper's exact geometry, which is precisely the situation robust / model-based
control (H-infinity, mu-synthesis) is needed for.  b0 defaults to the physically
correct (negative) value so the model is faithful even though the loop cannot be
stabilised.

PSO tuning (src/tuning_pso.py) confirms this is structural, not a tuning problem:
optimising (b0, wo, wc) against the robust cost cannot find any stable loop — the
best swarm result still rides the 150 V actuator limit with a 100+ um residual at
every cutting condition (cost ~3960, far above the divergence-penalty floor).
Whereas PSO makes the STATE-feedback SMC robust across the whole alpha4 range,
it cannot rescue the plant-inverting ADRC on a non-minimum-phase plant.
"""

from __future__ import annotations
import numpy as np
from .base import Controller
from .. import config as C
from .. import design as D


def _nominal_b0():
    """Physically correct input gain b0 = phi_sensor . Hpe (u -> y'' at the sensor)."""
    pl = D.design_plant(n_modes=C.N_MODES)
    return float(pl["phi"] @ pl["Hpe"])


class ADRC(Controller):
    name = "ADRC"
    color = "#a142f4"      # purple

    def __init__(self, b0=None, wo=3.0e4, wc=4.0e3, **kw):
        self.b0 = _nominal_b0() if b0 is None else b0   # ~ -1.29 for paper geometry
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
