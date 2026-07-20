"""Shared controller interface used by synthesis, SLD, and simulation.

A controller is a continuous-time LTI system driven by the sensor signal
y_s and producing the actuator voltage u, optionally augmented with a
spindle-synchronized delayed tap on its own state (used for the
regenerative-targeted delayed feedback and for the delayed-PD baseline):

    xk_dot = Ak xk + Bk y_s
    u(t)   = Ck xk + Dk y_s + gd0 @ xk(t) + gd1 @ xk(t - tau)

with tau = tooth passing period (spindle-synchronized). gd0/gd1 = None
means no delayed tap. The closed-loop analysis layers (avc/sld.py,
avc/simulate.py) must implement exactly this contract.

Conventions:
- The H-infinity / LQG central controller has observer structure, so its
  state xk estimates the design-model state [eta; eta_dot] (n=3 modes).
  `c_wmill` (when set) maps xk to the estimated milling-point deflection
  w_hat = c_wmill @ xk; the regenerative-targeted delayed feedback is then
  gd0 = +k_r * c_wmill, gd1 = -k_r * c_wmill.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Controller:
    Ak: np.ndarray                      # (nk, nk)
    Bk: np.ndarray                      # (nk, 1)
    Ck: np.ndarray                      # (1, nk)
    Dk: np.ndarray                      # (1, 1)
    gd0: np.ndarray | None = None       # (nk,)
    gd1: np.ndarray | None = None       # (nk,)
    c_wmill: np.ndarray | None = None   # (nk,) w_hat map (observer structure)
    gamma: float | None = None          # achieved H-infinity level (if any)
    label: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return self.Ak.shape[0]

    def with_delayed_feedback(self, k_r: float) -> "Controller":
        """Return a copy with the regenerative-targeted delayed tap set."""
        if self.c_wmill is None:
            raise ValueError("controller has no c_wmill map")
        return Controller(
            Ak=self.Ak, Bk=self.Bk, Ck=self.Ck, Dk=self.Dk,
            gd0=+k_r * self.c_wmill, gd1=-k_r * self.c_wmill,
            c_wmill=self.c_wmill, gamma=self.gamma,
            label=self.label + f"+DR(k_r={k_r:.3g})",
            meta={**self.meta, "k_r": k_r},
        )
