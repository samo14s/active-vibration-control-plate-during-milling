"""
TD-SMC — Time-Delay Sliding-Mode Combined Control (the NEW strategy).

Motivation.  The paper's combined method (RCTDC, controllers/ctdc.py) welds its
robust OUTPUT-feedback mu controller to an active time-delay term (Eq. 30)

        u_td(t) = K_Pp * y(t - tau) + K_Pd * ydot(t - tau)

that offsets part of the regenerative milling force at its source.  On the
paper's exact non-minimum-phase geometry, however, output feedback is capped by
the right-half-plane zero, so the mu base (and hence RCTDC) cannot cover the
very top of the milling-force band.  STATE feedback is not capped that way —
our regular-form sliding-mode controller (controllers/smc.py) is the strongest
base on this plant.

TD-SMC therefore combines the two ideas:

        u(t) = u_SMC(x_hat(t))  +  K_Pp * y_hat(t - tau) + K_Pd * ydot_hat(t - tau)

  * the sliding-mode part (equivalent control + boundary-layer reaching law on
    the pole-placed regular-form surface) supplies broadband stabilising
    damping through the Kalman-observed modal state;
  * the paper-style time-delay part cheaply removes the delayed (regenerative)
    component of the milling force, unloading the sliding mode exactly where
    chatter is born.

ALL six gains (zeta_s, ks, eta, phi, K_Pp, K_Pd) are tuned TOGETHER by PSO
against the robust cost of src/tuning_pso.py (milling-force band 0.3-2.9x at
reduced damping + modal-frequency rise), with the previously tuned SMC and the
zero-TD point seeded into the swarm — so the combined optimum can never be
worse than the plain SMC, and any improvement is genuinely earned by the
time-delay term.
"""

from __future__ import annotations
import numpy as np

from .smc import SMC
from .. import config as C


class TDSMC(SMC):
    name = "TD-SMC (new)"
    color = "#7b5c00"      # dark gold

    # defaults: joint PSO optimum on the paper-exact force model (cost 47.4 —
    # the best of all eight controllers): 0.7-7.3 um across the whole
    # alpha4 in [0.3, 2.9] band at only 5-24 V, and 3.7 um at +4% frequency rise.
    def __init__(self, zeta_s=0.3882, ks=1125.0, eta=254.1, phi=0.2414,
                 KPp=1.073e4, KPd=1.898, **kw):
        self.KPp = KPp            # delayed proportional gain [V/m]
        self.KPd = KPd            # delayed derivative gain   [V/(m/s)]
        super().__init__(zeta_s=zeta_s, ks=ks, eta=eta, phi=phi, **kw)

    def _design(self):
        super()._design()
        # velocity output row at the sensor (for ydot_hat), plus delay buffer
        from .. import design as D
        pl = D.design_plant(n_modes=self.n_modes)
        n = pl["n"]
        self.Cpos = pl["Cy"]
        self.Cvel = np.hstack([np.zeros((1, n)), pl["phi"].reshape(1, -1)])
        self.ndel = C.TAU_S / self.dt
        self.nbuf = int(np.ceil(self.ndel)) + 2

    def reset(self):
        super().reset()
        if hasattr(self, "nbuf"):
            self.buf_y = np.zeros(self.nbuf)
            self.buf_yd = np.zeros(self.nbuf)
            self.kstep = 0

    def _delayed(self, buf):
        idx = self.kstep - self.ndel
        if idx < 0:
            return 0.0
        i0 = int(np.floor(idx))
        fr = idx - i0
        return (1.0 - fr) * buf[i0 % self.nbuf] + fr * buf[(i0 + 1) % self.nbuf]

    def _update(self, t, y):
        xh = self.xhat
        # sliding-mode part (identical maths to SMC._update, unsaturated)
        s = float(self.Sn @ xh)
        u_eq = -float(self.SAn @ xh)
        u_sm = u_eq - (self.ks * s + self.eta * self._sat(s / self.phi))
        # time-delay part (paper Eq. 30 on the observer estimates)
        y_hat = float((self.Cpos @ xh).item())
        yd_hat = float((self.Cvel @ xh).item())
        self.buf_y[self.kstep % self.nbuf] = y_hat
        self.buf_yd[self.kstep % self.nbuf] = yd_hat
        u_td = self.KPp * self._delayed(self.buf_y) + self.KPd * self._delayed(self.buf_yd)
        u = float(np.clip(u_sm + u_td, -self.u_max, self.u_max))
        # observer update with the applied (saturated) input
        yhat_meas = float((self.Cy @ xh).item())
        self.xhat = self.Ad @ xh + self.Bd.flatten() * u + self.L.flatten() * (y - yhat_meas)
        self.kstep += 1
        return u
