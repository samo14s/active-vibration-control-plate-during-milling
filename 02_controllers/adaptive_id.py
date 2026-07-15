"""
adaptive_id.py
==============
Real-time-identification-SCHEDULED controllers for the multi-pass finishing
sequence: the controller's internal model is re-tuned to the modal frequencies
IDENTIFIED by the active piezo probe at each pass boundary (the non-cutting
transit), so it tracks the physically drifting, progressively thinned workpiece.

This is the closed-loop payoff of the material-removal FEM (01_core/material_removal)
and the probe identifier (03_analysis/realtime_id):

  transit probe -> f_hat (modal frequencies) -> rebuild controller gains for the
  upcoming pass -> cut.

Design choice (honest): the probe identifies modal FREQUENCIES cheaply (peak-pick
of the probe FRF); full mode SHAPES need spatial measurement and are NOT identified
online, so the scheduled controller uses the identified frequencies with the
pristine (nominal) mode shapes. The reshaping is second order (MAC > 0.999 over the
sequence), so frequency-only scheduling matches a true-frequency oracle to 3
decimals — see main_realtime_id.py.

WHEN ID HELPS (the finding this class exists to demonstrate): it rescues the
high-performance but fragile regulators (LQG, and the HRC layer) that DIVERGE once
the plant drifts past their fixed-design margin. The architecturally robust
ESO-ADRC already survives the whole sequence WITHOUT identification (its
disturbance states absorb the drift), and naive frequency-only retuning can even
hurt it — so identification and disturbance-observer robustness are complementary,
not redundant.
"""
import numpy as np
from types import SimpleNamespace

from lqg_controller import LQGController
from adrc_controller import ESO_ADRC_Controller, ESO_ADRC_HRC_Controller


def _design_view(freqs3, zeta3, D_obs3, H_Pe3):
    """Build a 3-mode controller design view from modal frequencies (Hz) and the
    (nominal) mode-shape projections."""
    om = 2 * np.pi * np.asarray(freqs3, float)
    return SimpleNamespace(
        n_modes=3, Mp=np.eye(3), Kp=np.diag(om**2),
        Cp=np.diag(2 * np.asarray(zeta3, float) * om),
        omega_n=om, freq_n=om / (2 * np.pi),
        D_obs=np.asarray(D_obs3, float).copy(),
        H_Pe_modal=np.asarray(H_Pe3, float).copy())


class IDScheduledController:
    """
    A controller whose internal model is re-tuned to identified modal frequencies
    at each pass boundary. `kind` selects the base law:
      'lqg'  -> LQGController (grid-searched weights per retune)
      'adrc' -> ESO_ADRC_Controller
      'hrc'  -> ESO_ADRC_HRC_Controller (with tooth-harmonic guard: drops a
                resonant line when an identified mode is within `harm_guard` of it)

    The pristine mode-shape projections (D_obs, H_Pe_modal) and modal damping are
    held fixed (frequency-only ID); `f3_ratio` extends the 2 identified frequencies
    to the third design mode by its pristine ratio to mode 2.
    """

    def __init__(self, kind, dt, D_obs3, H_Pe3, zeta3, f_nominal3,
                 w_tooth=None, kalman_V=1e-12, u_max=150.0,
                 lqg_wq=(1e10, 1e12, 1e14, 1e16), lqg_wqd=(1e4, 1e6, 1e8),
                 adrc_design=(1e14, 1e8, 1e4),
                 hrc_design=(1e16, 1e8, 3e3), hrc_n_harm=4, hrc_g=150.0,
                 hrc_lam=5.0, harm_guard=0.03, verbose=False):
        self.kind = kind
        self.dt = float(dt)
        self.D_obs3 = np.asarray(D_obs3, float)[:3].copy()
        self.H_Pe3 = np.asarray(H_Pe3, float)[:3].copy()
        self.zeta3 = np.asarray(zeta3, float)[:3].copy()
        self.f_nom3 = np.asarray(f_nominal3, float)[:3].copy()
        self.f3_ratio = self.f_nom3[2] / self.f_nom3[1]
        self.w_tooth = w_tooth
        self.kalman_V = kalman_V
        self.u_max = u_max
        self.lqg_wq, self.lqg_wqd = list(lqg_wq), list(lqg_wqd)
        self.adrc_design = adrc_design
        self.hrc_design, self.hrc_n_harm = hrc_design, hrc_n_harm
        self.hrc_g, self.hrc_lam, self.harm_guard = hrc_g, hrc_lam, harm_guard
        self.verbose = verbose
        self.history_fhat = []
        self.history_lines = []
        self.retune(self.f_nom3[:2])          # start at the nominal design

    # ------------------------------------------------------------------
    def retune(self, f_hat2):
        """Rebuild the base controller from the two identified frequencies."""
        f_hat2 = np.asarray(f_hat2, float)
        f3 = f_hat2[1] * self.f3_ratio
        freqs3 = np.array([f_hat2[0], f_hat2[1], f3])
        view = _design_view(freqs3, self.zeta3, self.D_obs3, self.H_Pe3)
        if self.kind == 'lqg':
            c = LQGController(view, dt=self.dt, verbose=False,
                              kalman_V=self.kalman_V, u_max=self.u_max)
            c.optimize_weights(w_q_list=self.lqg_wq, w_qd_list=self.lqg_wqd, w_r=1.0)
            c.discretize_observer()
            lines = None
        elif self.kind == 'adrc':
            c = ESO_ADRC_Controller(view, self.dt, *self.adrc_design,
                                    kalman_V=self.kalman_V, u_max=self.u_max,
                                    verbose=False)
            lines = None
        elif self.kind == 'hrc':
            f_tooth = self.w_tooth / (2 * np.pi)
            lines = [h for h in range(1, self.hrc_n_harm + 1)
                     if all(abs(h * f_tooth - f) / f > self.harm_guard
                            for f in freqs3[:2])]
            if not lines:
                lines = [1]
            c = ESO_ADRC_HRC_Controller(
                view, self.dt, *self.hrc_design, n_harm=self.hrc_n_harm,
                harmonics=lines, g_base=self.hrc_g, lam=self.hrc_lam,
                w_tooth=self.w_tooth, kalman_V=self.kalman_V, u_max=self.u_max,
                verbose=False)
        else:
            raise ValueError(f"unknown kind {self.kind!r}")
        self._c = c
        self.n_x = c.n_x
        self.history_fhat.append(freqs3.copy())
        self.history_lines.append(lines)
        return c

    # delegate the per-step interface to the current base controller
    def step(self, z_prev, u_prev, y_meas):
        return self._c.step(z_prev, u_prev, y_meas)

    def controller_realization(self):
        return self._c.controller_realization()
