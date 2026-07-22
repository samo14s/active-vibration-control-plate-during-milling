"""Common controller interface used by every design in this package."""

from __future__ import annotations
import numpy as np
from .. import config as C


class Controller:
    """
    Base class.  A controller maps the (noisy) measured plate displacement y to a
    control voltage u, once per sample (dt = 1/FS).  Reference is 0 (regulation).

    Subclasses implement `_update(t, y)`; `update` handles saturation and logging.
    """
    name = "base"
    color = "#888888"

    def __init__(self, dt: float = C.DT, u_max: float = C.U_MAX):
        self.dt = dt
        self.u_max = u_max
        self.reset()

    def reset(self):
        self._last_u = 0.0

    def _update(self, t: float, y: float) -> float:
        raise NotImplementedError

    def update(self, t: float, y: float) -> float:
        u = self._update(t, y)
        u = float(np.clip(u, -self.u_max, self.u_max))
        self._last_u = u
        return u


class ZeroController(Controller):
    """No control (open loop) — the uncontrolled milling baseline."""
    name = "No control"
    color = "#9aa0a6"

    def _update(self, t, y):
        return 0.0


class LinearSSController(Controller):
    """
    Wrapper for a linear discrete state-space controller K(z) that maps the
    tracking error e = r - y (r = 0 for regulation) to the control voltage u:

        xk1 = Ad xk + Bd e
        u   = Cd xk + Dd e

    Used by the H-infinity and mu-synthesis designs, which both synthesise a
    continuous K(s) that is then discretised.
    """
    name = "linear-ss"

    def __init__(self, Ad, Bd, Cd, Dd, sign=+1.0, **kw):
        self.Ad = np.asarray(Ad, float)
        self.Bd = np.asarray(Bd, float).reshape(-1, 1)
        self.Cd = np.asarray(Cd, float).reshape(1, -1)
        self.Dd = float(np.asarray(Dd).reshape(-1)[0])
        self.sign = sign
        self._nx = self.Ad.shape[0]
        super().__init__(**kw)

    def reset(self):
        super().reset()
        self.xk = np.zeros(self._nx)

    def _update(self, t, y):
        e = self.sign * (0.0 - y)           # r = 0
        u = float((self.Cd @ self.xk).item() + self.Dd * e)
        self.xk = self.Ad @ self.xk + self.Bd.flatten() * e
        return u
