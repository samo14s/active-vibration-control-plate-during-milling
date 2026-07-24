"""
digital_twin.py
===============
Physics-guided predictive–corrective DIGITAL TWIN for in-process thin-wall
milling, enabling robust model-based active chatter control.

Motivation
----------
Model-based controllers (LQG, and especially the regeneration-cancelling
RC-SAC) are powerful but FRAGILE to model error: a ~9 % error in the modal
frequency (from Young's-modulus / boundary-condition uncertainty, or the
material-removal drift) detunes them and destroys the deep-cut stability they
were designed for. Fixed-model control therefore fails in practice; blind
online identification (FFT of the sensor) is reactive, noisy, and cannot see a
mode that control has already suppressed.

The twin
--------
The digital twin fuses TWO information sources:

  1. FEED-FORWARD (physics):   the FEM in-process model predicts the
     DETERMINISTIC evolution of the modal parameters along the known
     material-removal trajectory (frequency trend, mode shapes, the
     force / sensor / actuator projections).  ->  `predict(removal)`

  2. FEEDBACK (data):   a short, FEM-GUIDED active probe — a chirp swept over a
     NARROW band around the FEM-predicted mode (not a blind wide search) — is
     injected through the piezo; the receptance peak calibrates the scalar
     model-error `theta = (f_true/f_FEM)^2` once.  ->  `calibrate(u, y, dt)`

The corrected model  `omega_true(s) ~= sqrt(theta) * omega_FEM(s)`  is then
available for the WHOLE pass from a SINGLE calibration (the property error is
constant; the removal trend is predicted), so the model-based controller stays
valid without continuous re-identification.

This module is model-only (no controller); pair `corrected_model(removal)` with
`RCSACController` / `LQGController`.
"""
import copy
import numpy as np
from plate_model import PlateModel
from mindlin_q8 import shape_at_point_M


class MillingDigitalTwin:
    """
    Predictive–corrective digital twin of the cantilever plate being milled.

    Parameters mirror `PlateModel` plus the sensor / cutter geometry and the
    piezo patch. `removal` is the cumulative free-end material removed (m),
    i.e. the deterministic scheduling coordinate (H_P_effective = H_P - removal).
    """

    def __init__(self, lp, hp, bp, rho, E, nu,
                 x_tool, z_tool, x_obs, z_obs, patch,
                 N1=30, N2=24, n_modes=3, zeta_modes=None, kappa=5.0/6.0):
        self.lp, self.hp, self.bp = lp, hp, bp
        self.rho, self.E, self.nu, self.kappa = rho, E, nu, kappa
        self.N1, self.N2, self.n_modes = N1, N2, n_modes
        self.zeta_modes = (np.array(zeta_modes) if zeta_modes is not None
                           else np.array([0.0031, 0.0017, 0.0027])[:n_modes])
        self.x_tool, self.z_tool = x_tool, z_tool
        self.x_obs, self.z_obs = x_obs, z_obs
        self.patch = patch                     # dict for add_piezo_patch
        self.theta_cal = 1.0                   # frequency-scale correction (uncalibrated)
        self._cache = {}
        # nominal (intact) FEM prediction, for the calibration band
        self._fem0 = self.predict(0.0)
        self.f_fem_intact = self._fem0.freq_n.copy()

    # ------------------------------------------------------------------
    def predict(self, removal):
        """FEM in-process model at `removal` (deterministic physics feed-forward)."""
        key = round(float(removal), 9)
        p = self._cache.get(key)
        if p is None:
            p = PlateModel(self.lp, self.hp - removal, self.bp, self.rho, self.E,
                           self.nu, N1=self.N1, N2=self.N2, n_modes=self.n_modes,
                           zeta_modes=list(self.zeta_modes), kappa=self.kappa,
                           verbose=False)
            p.set_observation(self.x_obs, self.z_obs)
            p.add_piezo_patch(**self.patch)
            p.Dp_tool = shape_at_point_M(self.x_tool, self.z_tool, p.N1, p.N2,
                                         p.lex, p.ley, p.node_id, p.ndof)[p.DOFf] @ p.V
            p.omega_fem = p.omega_n.copy()      # pristine FEM frequencies (never mutated)
            self._cache[key] = p
        return p

    # ------------------------------------------------------------------
    def design_probe(self, dt, T_probe=0.10, amp=20.0, band=(0.80, 1.22)):
        """FEM-guided chirp probe: swept sine over band·f_FEM(mode 1) [volts]."""
        ns = int(round(T_probe / dt)) + 1
        t = np.arange(ns) * dt
        f0, f1 = band[0] * self.f_fem_intact[0], band[1] * self.f_fem_intact[0]
        u = amp * np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * T_probe) * t**2))
        self._probe_band = band
        return u

    def calibrate(self, u_probe, y_probe, dt):
        """
        Estimate the scalar model-error theta from the probe receptance:
        pick the |Y/U| peak in the FEM-predicted band -> f_true -> theta.
        Returns (theta_cal, f_true).
        """
        ns = len(u_probe)
        w = np.hanning(ns)
        NFFT = 2 ** int(np.ceil(np.log2(ns)))
        Y = np.fft.rfft(y_probe * w, NFFT)
        U = np.fft.rfft(u_probe * w, NFFT)
        f = np.fft.rfftfreq(NFFT, dt)
        H = np.abs(Y / (U + 1e-15))
        b0, b1 = self._probe_band
        band = (f > b0 * self.f_fem_intact[0]) & (f < b1 * self.f_fem_intact[0])
        f_true = f[band][np.argmax(H[band])]
        self.theta_cal = (f_true / self.f_fem_intact[0]) ** 2
        self.f_true_intact = f_true
        return self.theta_cal, f_true

    # ------------------------------------------------------------------
    def corrected_model(self, removal, theta=None):
        """
        FEM in-process prediction at `removal`, corrected by the calibrated
        scalar theta (uniform frequency scaling — property-error model).
        Returns a fresh PlateModel-like COPY with scaled omega_n / Kp / Cp
        (the cached prediction is never mutated). Pass theta=1.0 for the
        uncalibrated FEM-nominal model. Mode shapes / projections are FEM
        (unchanged by a uniform property error).
        """
        p = self.predict(removal)
        th = self.theta_cal if theta is None else theta
        s = np.sqrt(th)
        pc = copy.copy(p)                       # shallow copy; reassign scaled arrays
        pc.omega_n = p.omega_fem * s
        pc.freq_n = pc.omega_n / (2 * np.pi)
        pc.Kp = np.diag(pc.omega_n ** 2)
        pc.Cp = np.diag(2 * self.zeta_modes[:self.n_modes] * pc.omega_n)
        return pc

    # ------------------------------------------------------------------
    def predicted_frequency(self, removal, corrected=True):
        """Mode-1 frequency along the removal trajectory (trend, optionally
        with the calibrated offset)."""
        f = self.predict(removal).omega_fem[0] / (2 * np.pi)
        return f * (np.sqrt(self.theta_cal) if corrected else 1.0)
