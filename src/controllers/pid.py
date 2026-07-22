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
    # derivative-filter bandwidth N BELOW that RHP zone (N = 2000 rad/s ~ 320 Hz):
    # the rate feedback then damps the 540/1068 Hz modes but rolls off before the
    # non-minimum-phase region, so it does not fight the RHP zero.  With that
    # rolloff a high-gain PID regulates the NOMINAL plant and the paper's combined
    # worst-case extremely well (~5 um, better than the robust controllers).
    # Its weakness is robustness: the gains are tuned at the nominal cutting force,
    # and because the regenerative force is UNMATCHED (enters at the milling point,
    # not the sensor/actuator) the loop has no guaranteed margin against a large
    # change of the cutting-force coefficient alpha4 alone -- it degrades badly and
    # can diverge at the extremes of the alpha4 range.  That performance-vs-
    # robustness gap is exactly why the paper adopted robust mu-synthesis.
    def __init__(self, Kp=2.0e4, Ki=0.0, Kd=200.0, N=2000.0, **kw):
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
