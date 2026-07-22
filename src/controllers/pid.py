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

    def __init__(self, Kp=2.0e4, Ki=0.0, Kd=320.0, N=8000.0, **kw):
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
