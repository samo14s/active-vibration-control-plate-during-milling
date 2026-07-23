"""
RCTDC — Robust Combined Time-Delay Control (the reference paper's full method).

The paper's proposed controller (Section 3.4) combines the mu-synthesis robust
controller with an ACTIVE TIME-DELAY controller, Eq. (30):

        u(t) = u_mu(t) + u_Pd(t) ,
        u_Pd(t) = K_Pp * y(t - tau) + K_Pd * ydot(t - tau) ,

where tau is the regenerative (tooth-passing) delay.  Rationale (paper Eq. 31):
the regenerative part of the milling force is  +alpha4(t) * Phi_m^T y_P(t-tau);
an actuator force proportional to the DELAYED output partially offsets that
delayed force term, so the disturbance the robust loop has to reject loses most
of its regenerative (chatter-driving) component and keeps mainly the forced
part.  The delayed-derivative term speeds up the response.  The paper reports
that the combination both improves the vibration level and REDUCES the average
control voltage (~22 %) versus the robust controller alone, because the robust
controller stabilises the loop while the time-delay term cheaply removes part
of the milling force at its source.

Implementation
--------------
* Robust base: the same mu-synthesis controller as controllers/musyn.py
  (design cached at module level — it is fixed; only the TD gains are tuned).
* Delayed signals: a steady-state Kalman observer (same design as the SMC's)
  reconstructs smooth y_hat and ydot_hat at the sensor from the noisy
  measurement; a ring buffer delays both by tau (linear interpolation between
  samples, tau/dt is not an integer).
* The two gains (K_Pp, K_Pd) are tuned by PSO against the same ROBUST cost as
  the other gain-based controllers (sweep of the milling-force coefficient over
  the paper band at reduced damping — see src/tuning_pso.py), i.e. the "ideal"
  tuning the combination admits.
"""

from __future__ import annotations
import numpy as np

from .base import Controller
from .. import config as C
from .. import design as D
from .musyn import design_musyn, FB_SIGN


_MU_CACHE = {}


def _mu_controller():
    """Cached discrete mu-synthesis base controller (A, B, C, D)."""
    if "K" not in _MU_CACHE:
        Kd = design_musyn()
        _MU_CACHE["K"] = (np.asarray(Kd.A), np.asarray(Kd.B),
                          np.asarray(Kd.C), np.asarray(Kd.D))
    return _MU_CACHE["K"]


class CTDC(Controller):
    name = "RCTDC (paper)"
    color = "#b31412"      # dark red

    # defaults: PSO optimum of (K_Pp, K_Pd) on the paper-exact force model.
    # The TD term extends the mu base's coverage of the milling-force band up to
    # ~2.0x (mu alone degrades from ~1.9x) and lowers its voltage, but the
    # OUTPUT-feedback mu base is still capped by the RHP zero, so RCTDC cannot
    # reach the extreme alpha4 = 2.4-2.9x — the structural limit the state-
    # feedback TD-SMC (tdsmc.py) removes.
    def __init__(self, KPp=5.0e4, KPd=10.48, kf_q=5e-2, kf_r=None,
                 n_modes=3, **kw):
        self.KPp = KPp            # delayed proportional gain [V/m]
        self.KPd = KPd            # delayed derivative gain   [V/(m/s)]
        self.kf_q = kf_q
        self.kf_r = (C.MEAS_NOISE_STD ** 2) if kf_r is None else kf_r
        self.n_modes = n_modes
        super().__init__(**kw)
        self._design()

    def _design(self):
        # robust base (mu-synthesis), fixed
        self.Am, self.Bm, self.Cm, self.Dm = _mu_controller()
        self.nxm = self.Am.shape[0]
        # Kalman observer for smooth y_hat / ydot_hat at the sensor
        pl = D.design_plant(n_modes=self.n_modes)
        A, Bu, Cy = pl["A"], pl["Bu"], pl["Cy"]
        n = pl["n"]
        self.Cy = Cy
        self.Cpos = Cy
        self.Cvel = np.hstack([np.zeros((1, n)), pl["phi"].reshape(1, -1)])
        self.Ad, self.Bd = D.c2d(A, Bu, self.dt)
        self.L = D.kalman_gain(self.Ad, Cy, self.kf_q, self.kf_r)
        self.nx = 2 * n
        # delay buffer: tau/dt samples (non-integer -> linear interpolation)
        self.ndel = C.TAU_S / self.dt
        self.nbuf = int(np.ceil(self.ndel)) + 2

    def reset(self):
        super().reset()
        if hasattr(self, "nxm"):
            self.xm = np.zeros(self.nxm)          # mu controller state
            self.xhat = np.zeros(self.nx)         # observer state
            self.buf_y = np.zeros(self.nbuf)      # ring buffers of y_hat, yd_hat
            self.buf_yd = np.zeros(self.nbuf)
            self.k = 0

    def _delayed(self, buf):
        """Value delayed by tau (linear interpolation in the ring buffer)."""
        idx = self.k - self.ndel
        if idx < 0:
            return 0.0                            # start-up: no history yet
        i0 = int(np.floor(idx))
        fr = idx - i0
        return (1.0 - fr) * buf[i0 % self.nbuf] + fr * buf[(i0 + 1) % self.nbuf]

    def _update(self, t, y):
        # --- robust mu part (identical to the LinearSSController wrapper with
        #     sign = FB_SIGN used by MuSynthesis:  e = sign*(0 - y)) ---
        e = FB_SIGN * (0.0 - y)
        u_mu = float((self.Cm @ self.xm).item() + float(self.Dm.reshape(-1)[0]) * e)
        # --- observer estimates of y and ydot at the sensor ---
        xh = self.xhat
        y_hat = float((self.Cpos @ xh).item())
        yd_hat = float((self.Cvel @ xh).item())
        self.buf_y[self.k % self.nbuf] = y_hat
        self.buf_yd[self.k % self.nbuf] = yd_hat
        # --- active time-delay part (paper Eq. 30) ---
        u_td = self.KPp * self._delayed(self.buf_y) + self.KPd * self._delayed(self.buf_yd)
        u = float(np.clip(u_mu + u_td, -self.u_max, self.u_max))
        # --- state updates (mu part driven by e, as in the base wrapper) ---
        self.xm = self.Am @ self.xm + self.Bm.ravel() * e
        yhat_meas = float((self.Cy @ xh).item())
        self.xhat = self.Ad @ xh + self.Bd.flatten() * u + self.L.flatten() * (y - yhat_meas)
        self.k += 1
        return u
