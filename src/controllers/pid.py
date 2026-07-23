"""
Classical controller: PID with filtered derivative and anti-windup.

For vibration regulation the reference is 0, so u = -(Kp y + Ki ∫y + Kd dy/dt).
The derivative (rate) term supplies the damping that the lightly damped plate
lacks; the proportional term retunes the effective stiffness; the (small)
integral term rejects quasi-static cutting-force bias.  A first-order filter on
the derivative (coefficient N) keeps it from amplifying sensor noise.

Gains are auto-tuned (see tuning.py) by minimising the common cost
J = ∫(y^2 + rho u^2) dt on the nominal closed loop, so the *classical* family
gets a genuinely fair, well-tuned baseline rather than a hand-picked strawman.
"""

from __future__ import annotations
import numpy as np
from .base import Controller
from .. import config as C


class PID(Controller):
    name = "PID (classical)"
    color = "#e8710a"      # orange

    # NOTE (paper NON-MINIMUM-PHASE geometry): the piezo (left-lower) and the
    # sensor (right-upper) sit on opposite sides, so u -> y has a right-half-plane
    # zero (~5 kHz).  The key to a working classical design is to keep the
    # derivative-filter bandwidth N BELOW that RHP zone: the rate feedback then
    # damps the 540/1068 Hz modes but rolls off before the non-minimum-phase
    # region, so it does not fight the RHP zero.
    #
    # These gains were PSO-tuned (src/tuning_pso.py) against the same robust cost
    # as the SMC.  PSO makes PID excellent in the MID cutting-force range
    # (alpha4 ~ 1.1..2.4x -> ~8 um at <20 V) and, unlike the hand-tuned version,
    # keeps it bounded at the high extreme (alpha4=2.9 -> ~20 um).  But PID is
    # OUTPUT feedback, and PSO could NOT make it robust across the WHOLE range:
    # at the low extreme (alpha4=0.3x) the high-gain loop rides the actuator limit
    # (~216 um at 150 V).  The right-half-plane zero caps the achievable output-
    # feedback gain, so no PID tuning covers the full range -- a structural limit
    # (contrast the STATE-feedback SMC, which PSO makes robust everywhere).  This
    # performance-vs-robustness gap is exactly why the paper adopted robust control.
    # defaults: PSO optimum on the PAPER-EXACT force model.  A LOW-gain PID now
    # holds the MID band (alpha4 ~ 0.7..2.4 -> 1.3-5 um at <10 V) but grows
    # slowly (2.4-2.5x per window third) at BOTH band edges (alpha4 = 0.3 and
    # 2.9) — bounded snapshots, divergent over the long pass.  The RHP zero
    # still caps output feedback; only the band centre is coverable.
    def __init__(self, Kp=1000.0, Ki=0.0, Kd=393.7, N=1360.0, **kw):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.N = N          # derivative-filter bandwidth [rad/s-ish]
        super().__init__(**kw)

    def reset(self):
        super().reset()
        self.integ = 0.0
        self.deriv = 0.0
        self.y_prev = 0.0
        self._first = True

    def _update(self, t, y):
        dt = self.dt
        if self._first:
            self.y_prev = y
            self._first = False
        # filtered derivative:  D = N*(y - y_prev) blended with previous D
        a = self.N * dt / (1.0 + self.N * dt)
        self.deriv = (1 - a) * self.deriv + a * (y - self.y_prev) / dt
        self.y_prev = y

        u_unsat = -(self.Kp * y + self.Ki * self.integ + self.Kd * self.deriv)
        u = float(np.clip(u_unsat, -self.u_max, self.u_max))
        # conditional anti-windup: integrate only if not saturating outward
        if not (abs(u_unsat) > self.u_max and np.sign(u_unsat) == np.sign(y) * -1):
            self.integ += y * dt
        return u
