"""
delayed_pd_controller.py
========================
Active time-delay (delayed PD) controller of Du et al. (2024), Eq. (30):

    u(t) = K_Pp * y(t - tau) + K_Pd * y_dot(t - tau)

Pure OUTPUT feedback on the delayed measurement — the article combines it with a
mu-synthesis robust controller; alone, the article reports it is insufficient at the
start/end positions. Here it serves as an additional model-based BASELINE between
open loop and LQG, with gains grid-tuned on the nominal scenario.

Implementation notes:
- y(t - tau) comes from an internal ring buffer of the (noisy) measurement.
- The delayed derivative uses an n_diff-sample backward difference (mild low-pass,
  needed because the 10 nm measurement noise makes a 1-step difference too noisy).
- The output is clipped to the same +/-u_max as the other controllers (symmetry).
- The `n_x = 2` dummy state exists only to satisfy the Newmark harness allocation.
"""
import numpy as np


class DelayedPDController:
    def __init__(self, dt: float, tau: float,
                 K_pp: float, K_pd: float,
                 u_max: float = 150.0, n_diff: int = 5):
        self.dt = dt
        self.tau = tau
        self.n_tau = int(np.round(tau / dt))
        self.K_pp = K_pp
        self.K_pd = K_pd
        self.u_max = u_max
        self.n_diff = int(n_diff)
        self.n_x = 2                      # dummy (harness allocates x_hat[n_x])
        self._buf = []

    def reset(self):
        """Clear the measurement buffer — call before every simulation run."""
        self._buf = []

    def step(self, x_hat_prev, u_prev, y_meas):
        self._buf.append(float(y_meas))
        i = (len(self._buf) - 1) - self.n_tau
        if i >= self.n_diff:
            y_d = self._buf[i]
            yd_d = (self._buf[i] - self._buf[i - self.n_diff]) / (self.n_diff * self.dt)
        else:                              # not enough history yet
            y_d, yd_d = 0.0, 0.0
        u = self.K_pp * y_d + self.K_pd * yd_d
        u = float(np.clip(u, -self.u_max, self.u_max))
        return np.zeros(2), u
