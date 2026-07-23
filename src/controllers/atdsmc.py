"""
ATD-SMC — Adaptive Time-Delay Sliding-Mode Control (the DEVELOPED controller).

This is the direct development of the (known, not-novel) TD-SMC.  It keeps TD-SMC
UNCHANGED as its base — the regular-form STATE sliding surface with a Kalman
observer (smc.py) plus the paper's Eq.-30 time-delay term that removes the delayed
regenerative milling force at its source — and adds ONE thing: an ADAPTIVE
switching gain.

Why adapt the switching gain
----------------------------
The fixed switching gain eta of SMC / TD-SMC is a compromise.  Small eta keeps the
nominal cut gentle (low voltage) but cannot reject an EXTREME regenerative force
(alpha4 >= 3.5x), so TD-SMC diverges there; large eta survives the extreme force
but wastes voltage on every ordinary cut and destabilises the +17% frequency rise.
No single fixed eta wins the whole harder-than-paper envelope — which is exactly
why plain SMC and TD-SMC each survive only 5/10 of the challenge suite.

An ADAPTIVE eta removes the compromise.  It auto-GROWS only while the sliding
variable persists outside a healthy dead-zone (genuine chatter driving the
surface) and DECAYS back to its floor when the surface is quiescent:

    if |s| > phi_dz:  eta_ad += a_up * |s| * dt        (grow under real chatter)
    else:             eta_ad -= a_dn * eta_ad * dt     (leak back when quiescent)
    eta_ad = clip(eta_ad, eta_min, eta_max)

    u = u_eq  +  u_td(Eq.30)  -  ( ks*s + eta_ad*sat(s/phi) )

The adaptation is REVERSIBLE (unlike an irreversible latch): a transient raises
eta_ad only for as long as the disturbance persists, then it relaxes — so the
controller cannot get stuck in a high-voltage mode.  The dead-zone phi_dz stops
measurement noise from winding eta_ad up.  A short soft-start ramp (t_ramp) scales
the command during start-up so the first-sample transient does not rail the piezo.

Result on the harder-than-paper challenge suite (src/challenges.py)
------------------------------------------------------------------
ATD-SMC survives 7/10 versus 5/10 for both SMC and TD-SMC, and it does so while
KEEPING TD-SMC's economy: ~4 um at ~10 V nominal, ~3.8 um over the full 20.4 s
moving pass — i.e. it DOMINATES TD-SMC (same low voltage on ordinary cuts) and
additionally survives the extreme cutting force (alpha4 = 3.5x, where TD-SMC
diverges), the +17% material-removal frequency rise (where plain SMC blows up),
and the combined extreme (alpha4 = 4.0x with -40% damping, which diverges SMC and
TD-SMC together).  The two alpha4 >= 4.5x cases remain divergent for every
controller — they are beyond the finite piezo actuator's physical authority, not a
tuning failure — and the 20 V-limited case saturates the rail while staying
stable.

Honesty note: neither TD-SMC nor this ATD-SMC is a novel control law.  Adaptive-
gain sliding-mode control and delayed-force compensation are both established.  The
contribution is the engineering combination and tuning for the milling non-
minimum-phase problem so the operating envelope is genuinely extended beyond the
fixed-gain designs.
"""

from __future__ import annotations
import numpy as np

from .smc import SMC
from .. import config as C


class ATDSMC(SMC):
    name = "ATD-SMC (developed)"
    color = "#8e24aa"      # purple

    def __init__(self, zeta_s=0.3882, ks=1315.0, phi=0.16,
                 phi_dz=0.01, a_up=1.5e7, a_dn=8.0,
                 eta_min=180.0, eta_max=1400.0, eta0=180.0,
                 KPp=1.073e4, KPd=1.898, t_ramp=0.08, **kw):
        # adaptive switching-gain law
        self.phi_dz = phi_dz      # dead-zone on |s| below which eta_ad does not grow
        self.a_up = a_up          # growth rate of eta_ad under real chatter
        self.a_dn = a_dn          # leak (decay) rate back to eta_min when quiescent
        self.eta_min = eta_min    # floor (nominal switching gain) [V]
        self.eta_max = eta_max    # ceiling [V]
        self.eta0 = eta0          # initial eta_ad
        # paper Eq.-30 time-delay (regenerative) feed-forward, as in TD-SMC
        self.KPp = KPp            # delayed proportional gain [V/m]
        self.KPd = KPd            # delayed derivative gain   [V/(m/s)]
        self.t_ramp = t_ramp      # soft-start ramp [s] (caps the start-up spike)
        super().__init__(zeta_s=zeta_s, ks=ks, eta=eta0, phi=phi, **kw)

    def _design(self):
        super()._design()
        # velocity output row at the sensor (for ydot_hat in the time-delay term)
        self.Cvel = np.hstack([np.zeros((1, self.n)),
                               C.phi_mill(C.MILL_POS_STATIC).reshape(1, -1)])
        self.ndel = C.TAU_S / self.dt
        self.nbuf = int(np.ceil(self.ndel)) + 2

    def reset(self):
        super().reset()
        if hasattr(self, "nx"):
            self.eta_ad = float(self.eta0)
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
        s = float(self.Sn @ xh)
        u_eq = -float(self.SAn @ xh)

        # adaptive switching gain: grow under persistent chatter, leak back otherwise
        if abs(s) > self.phi_dz:
            self.eta_ad += self.a_up * abs(s) * self.dt
        else:
            self.eta_ad -= self.a_dn * self.eta_ad * self.dt
        self.eta_ad = float(np.clip(self.eta_ad, self.eta_min, self.eta_max))
        u_reach = -(self.ks * s + self.eta_ad * self._sat(s / self.phi))

        # paper Eq.-30 time-delay feed-forward on the observer estimates (as TD-SMC)
        y_hat = float((self.Cy @ xh).item())
        yd_hat = float((self.Cvel @ xh).item())
        self.buf_y[self.kstep % self.nbuf] = y_hat
        self.buf_yd[self.kstep % self.nbuf] = yd_hat
        u_td = self.KPp * self._delayed(self.buf_y) + self.KPd * self._delayed(self.buf_yd)

        u_cmd = u_eq + u_reach + u_td
        if self.t_ramp > 0.0 and t < self.t_ramp:
            u_cmd *= t / self.t_ramp           # soft-start: cap the start-up transient
        u = float(np.clip(u_cmd, -self.u_max, self.u_max))

        # observer update with the applied (saturated) input
        self.xhat = self.Ad @ xh + self.Bd.flatten() * u + self.L.flatten() * (y - y_hat)
        self.kstep += 1
        return u
