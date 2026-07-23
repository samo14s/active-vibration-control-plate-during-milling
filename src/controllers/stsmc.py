"""
ST-SMC — Super-Twisting Sliding-Mode Control (the DEVELOPED controller).

This is the enhancement of the (known, not-novel) TD-SMC: it keeps the strongest
base on this non-minimum-phase plant — the regular-form STATE sliding surface with
a Kalman observer (see smc.py) — but upgrades the reaching law and the force
compensation to reach control challenges HARDER than the paper's:

  * super-twisting reaching (a second-order sliding-mode algorithm):
        u_st = -k1*sqrt(|s|)*sign(s) - kl*s + w ,   w_dot = -k2*sign(s)
    which is CONTINUOUS (no boundary-layer sat / no chattering the piezo could not
    follow), gives finite-time convergence and higher sliding accuracy than the
    first-order boundary-layer reaching of the plain SMC;
  * a model-based regenerative-force FEEDFORWARD: the observer state q_hat and a
    tau-delay buffer reconstruct the delayed cutting excitation and cancel its
    projection onto the sliding variable at its source, so the reaching law is
    unloaded exactly where chatter is born — this is what pushes the achievable
    cutting-force ceiling toward the actuator's physical limit.

Harder-than-paper envelope (challenge suite in src/challenges.py):
  EXTREME cutting force alpha4 up to ~4x (the paper studies up to 2.9x; the plain
  SMC/TD-SMC diverge at 3.5-4x), a LARGE +17% material-removal frequency rise
  (which breaks the plain SMC), and a TIGHT 20 V actuator limit — all at once,
  while staying gentle (low voltage) at the nominal condition.

Honesty note: neither TD-SMC nor this ST-SMC is a novel control law — super-
twisting, delayed-force compensation and state SMC are all established.  The
contribution here is engineering: combining them for the milling non-minimum-phase
problem and PSO-tuning the six gains against the hard multi-condition cost so the
operating envelope is genuinely extended beyond the plain designs.  The extreme-
alpha4 ceiling (~4x) is set by the finite piezo authority, not by the tuning.

All six gains (zeta_s, k1, k2, kl, ff_fac, ff_kappa) are PSO-tuned on the hard
challenge suite (src/tuning_pso.py :: stsmc_spec), seeded with the plain SMC's
surface and the working super-twisting point.
"""

from __future__ import annotations
import numpy as np

from .smc import SMC
from .. import config as C


class STSMC(SMC):
    name = "ST-SMC (developed)"
    color = "#8e24aa"      # purple

    # defaults: optimum on the hard challenge suite (src/challenges.py).  A
    # strong feedforward (ff_fac=3) unloads the reaching law, so k1/kl stay
    # gentle -- 7/10 survived at only ~40 V nominal (vs 74 V for the aggressive
    # k1=1400/kl=2000 point), lower voltage on every challenge, and better on the
    # +17% frequency rise.  The two a4>=4.5 misses are beyond the piezo's
    # physical authority (they cap all controllers) and tightV saturates the
    # 20 V rail while staying stable -- the rest are all survived.
    def __init__(self, zeta_s=0.3644, k1=900.0, k2=5000.0, kl=1400.0,
                 ff_fac=3.0, ff_kappa=1.0, **kw):
        self.k1 = k1              # super-twisting sqrt(|s|) gain
        self.k2 = k2              # super-twisting integral (w) gain
        self.kl = kl              # extra linear -kl*s damping
        self.ff_fac = ff_fac      # cutting factor assumed by the feedforward
        self.ff_kappa = ff_kappa  # fraction (0..1.5) of the feedforward applied
        super().__init__(zeta_s=zeta_s, **kw)

    def _design(self):
        super()._design()
        self.phi_mill = C.phi_mill(C.MILL_POS_STATIC).ravel()[:self.n]
        self.a4_ff = C.CUT_GAIN * C.ALPHA4_BASE * self.ff_fac * (C.DUTY / 2.0)
        self.Sn_force = self.Sn.ravel()[self.n:]        # modal-force -> s projection
        self.ndel = C.TAU_S / self.dt
        self.nbuf = int(np.ceil(self.ndel)) + 2

    def reset(self):
        super().reset()
        if hasattr(self, "nbuf"):
            self.w = 0.0
            self.qbuf = np.zeros((self.nbuf, self.n))
            self.kstep = 0

    def _delayed_q(self):
        idx = self.kstep - self.ndel
        if idx < 0:
            return np.zeros(self.n)
        i0 = int(np.floor(idx)); fr = idx - i0
        return (1.0 - fr) * self.qbuf[i0 % self.nbuf] + fr * self.qbuf[(i0 + 1) % self.nbuf]

    def _update(self, t, y):
        xh = self.xhat
        s = float(self.Sn @ xh)
        u_eq = -float(self.SAn @ xh)

        # regenerative-force feedforward: cancel the delayed cutting force on s.
        # the plant force is NEGATIVE (qdd -= a4*phi_mill*proj); cancelling its
        # s-projection gives u_ff = +ff_kappa * a4_ff * (Sn_force . phi_mill) * proj.
        q = xh[:self.n]
        self.qbuf[self.kstep % self.nbuf] = q
        proj = float(self.phi_mill @ (q - self._delayed_q()))
        u_ff = self.ff_kappa * self.a4_ff * float(self.Sn_force @ self.phi_mill) * proj

        # super-twisting reaching (continuous, chatter-free)
        sg = np.sign(s)
        u_reach = -self.k1 * np.sqrt(abs(s)) * sg - self.kl * s + self.w

        u = float(np.clip(u_eq + u_ff + u_reach, -self.u_max, self.u_max))

        yhat = float((self.Cy @ xh).item())
        self.xhat = self.Ad @ xh + self.Bd.flatten() * u + self.L.flatten() * (y - yhat)
        self.w -= self.k2 * sg * self.dt          # integrate super-twisting state
        self.kstep += 1
        return u
