"""
adaptive_palf_lqg_controller.py
===============================
A-PALF-LQG : ADAPTIVE Phase-Aware Learned Feedforward + LQG, with ONLINE SYSTEM-
PARAMETER UPDATING during material removal.

Motivation. As the tool removes material, the plate's modal frequencies drift
(Du et al. 2024 measured +9 %/+17 % over a machining series), and a FIXED model-based
controller loses performance and eventually stability (this package's measured margin
is ~-9 % frequency mismatch at a_p = 0.3 mm). This controller therefore updates its
plant-parameter estimate at EVERY control step and reschedules its feedback gain and
observer accordingly.

Identifiability — why a probe is used. Under STABLE cutting all signals are
tooth-passing-periodic, and the periodic milling force is unknown: innovations of a
model bank are then dominated by the common unmodelled disturbance and do NOT
discriminate the plant parameters (verified in this package's diagnostics). The
classical remedy is PERSISTENT EXCITATION: a low-amplitude multisine PROBE is
injected through the piezo at frequencies chosen AWAY from the tooth-passing
harmonics. At those frequencies the milling force has no content, so the measured
ratio  H(jw_p) = Y(jw_p)/U(jw_p)  is a clean plant FRF sample.

Strategy:
  1. OFFLINE: a grid of frequency-scale factors theta (Kp ~ theta^2, Cp ~ theta —
     the material-removal signature) defines a model bank. For each theta: LQR gain,
     Kalman observer (discretised), and the plant FRF G(jw_p, theta) at the probe
     frequencies are pre-computed.
  2. ONLINE, EVERY STEP: a sliding DFT (O(1)/step per frequency) updates Y(jw_p) and
     U(jw_p) over the last W samples; the parameter estimate is
        theta_hat = argmin_i sum_p |Y_p - G(jw_p,theta_i) U_p|^2
                                   / (|Y_p|^2 + |G U_p|^2)
     (balanced, resonance-safe normalization). The score refreshes at every step —
     the per-step parameter update.
  3. GAIN SCHEDULING: the active model (LQR gain + observer state from a bank of
     parallel observers) switches to theta_hat, with a dwell window and a hysteresis
     margin to prevent chattering. The phase-locked learned feedforward and the CLF
     voltage governor are inherited from PALF-LQG unchanged (the feedforward is
     phase-indexed, not model-indexed — exactly the part that survives drift).

Honest costs, stated: the probe adds a small extra vibration and voltage (amplitude
`probe_amp` per line, default 0.4 V — report the nominal-scenario cost); the tracker
lags the true drift by roughly half the DFT window; theta resolution is the grid
spacing. All three are quantified by `main_adaptive_removal.py`.
"""
import numpy as np
from scipy.linalg import solve_continuous_are, expm

from palf_lqg_controller import PALF_LQG_Controller


class AdaptivePALF_LQG_Controller(PALF_LQG_Controller):
    def __init__(self, plate, dt,
                 base_w_q=1e14, base_w_qd=1e8, base_w_r=1.0,
                 ff_lr=0.01, ff_max=30.0, ff_alpha=1.0,
                 n_per=82, safety_alpha=5.0,
                 kalman_V=1e-12, u_max=150.0,
                 theta_grid=(0.86, 0.90, 0.94, 0.98, 1.00, 1.04, 1.08, 1.12, 1.16),
                 probe_bins=(42, 48, 54, 60, 66), probe_amp=0.4,
                 window_periods=25, dwell=None, switch_margin=0.8,
                 verbose=True):
        """
        Anti-leakage design. The identification window is W = window_periods * n_per
        steps — an INTEGER number of tooth-passing periods — so every milling-force
        harmonic completes an integer number of cycles in the window; the probe lines
        are placed ON DFT BINS of that window (f_p = bin / (W*dt)) that are NOT
        multiples of window_periods. DFT-bin orthogonality then makes the (huge)
        forced-vibration harmonics leak EXACTLY ZERO into the probe lines.

        probe_bins  : DFT bin indices of the probe lines. With n_per=82, dt=50 µs,
            window_periods=25 (W=2050, bin ~9.76 Hz): 42/48/54/60/66 ->
            410/468/527/585/644 Hz, straddling the mode-1 range spanned by
            theta_grid (448–604 Hz).
        probe_amp   : amplitude per probe line (V). Small vs the ~6 V RMS control.
        switch_margin : hysteresis — switch models only if the best FRF-matching
            score is below switch_margin * (score of the active model).
        """
        super().__init__(plate, dt,
                         base_w_q=base_w_q, base_w_qd=base_w_qd, base_w_r=base_w_r,
                         ff_lr=ff_lr, ff_max=ff_max, ff_alpha=ff_alpha,
                         n_per=n_per, safety_alpha=safety_alpha,
                         kalman_V=kalman_V, u_max=u_max, verbose=False)
        self.theta_grid = np.asarray(theta_grid, float)
        self.W = int(window_periods) * int(n_per)
        self.probe_bins = np.asarray(probe_bins, int)
        self.probe_freqs = self.probe_bins / (self.W * dt)     # exactly on DFT bins
        self.probe_amp = float(probe_amp)
        self.dwell = int(dwell) if dwell is not None else int(n_per)
        self.switch_margin = float(switch_margin)

        # Control bank: LQR gain + discrete Kalman observer per theta
        self._bank = [self._design_for_theta(th, base_w_q, base_w_qd, base_w_r,
                                             kalman_V)
                      for th in self.theta_grid]

        # Plant FRF u->y at the probe lines, per theta (identification reference)
        w_p = 2 * np.pi * self.probe_freqs
        self._G_ref = np.array([[self._plant_frf(mdl['A'], w) for w in w_p]
                                for mdl in self._bank])          # (n_theta, n_p)

        # Sliding-DFT twiddles at the probe lines
        self._twd = np.exp(1j * w_p * self.dt)                   # e^{+j w dt}
        self.reset_adaptation()

        if verbose:
            print(f"[A-PALF] theta grid {list(self.theta_grid)}; probe "
                  f"{list(self.probe_freqs)} Hz @ {self.probe_amp} V; "
                  f"W={self.W} steps ({self.W*self.dt*1e3:.0f} ms), "
                  f"dwell={self.dwell}, margin={self.switch_margin}")

    # ------------------------------------------------------------------
    def _plant_frf(self, A, w):
        """Plant-only FRF u -> y at angular frequency w (no observer, no force)."""
        n2 = self.n_x
        return complex((self.C @ np.linalg.solve(1j * w * np.eye(n2) - A,
                                                 self.B))[0, 0])

    def _design_for_theta(self, theta, w_q, w_qd, w_r, kalman_V):
        """LQR gain + discretised Kalman observer for the theta-scaled model."""
        n = self.n_modes
        Kp = self.plate.Kp * theta**2
        Cp = self.plate.Cp * theta
        A = np.zeros((self.n_x, self.n_x))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.linalg.solve(self.plate.Mp, Kp)
        A[n:, n:] = -np.linalg.solve(self.plate.Mp, Cp)
        B, C = self.B, self.C

        M_yp = np.outer(self.plate.D_obs, self.plate.D_obs)
        Q = np.block([[w_q * M_yp + 1e-3 * np.eye(n), np.zeros((n, n))],
                      [np.zeros((n, n)), w_qd * np.eye(n) + 1e-3 * np.eye(n)]])
        R = np.array([[w_r]])
        P = solve_continuous_are(A, B, Q, R)
        K_lqr = np.linalg.solve(R, B.T @ P)

        W = 1e-6 * np.eye(self.n_x)
        V = np.array([[kalman_V]])
        P_kal = solve_continuous_are(A.T, C.T, W, V)
        L = P_kal @ C.T @ np.linalg.inv(V)
        A_obs = A - L @ C
        A_obs_d = expm(A_obs * self.dt)
        G_u = np.linalg.solve(A_obs, (A_obs_d - np.eye(self.n_x)) @ B)
        G_y = np.linalg.solve(A_obs, (A_obs_d - np.eye(self.n_x)) @ L)
        return dict(theta=theta, A=A, K_lqr=K_lqr,
                    A_obs_d=A_obs_d, G_u=G_u.flatten(), G_y=G_y.flatten())

    # ------------------------------------------------------------------
    def reset_adaptation(self):
        """Reset observer bank, DFT buffers, and selection — call before each run."""
        nb = len(self.theta_grid)
        n_p = len(self.probe_freqs)
        self._xb = [np.zeros(self.n_x) for _ in range(nb)]
        self._i_active = int(np.argmin(np.abs(self.theta_grid - 1.0)))
        self._k_last_switch = 0
        self._y_buf = np.zeros(self.W)
        self._u_buf = np.zeros(self.W)
        self._Sy = np.zeros(n_p, dtype=complex)   # sliding DFT of y at probe lines
        self._Su = np.zeros(n_p, dtype=complex)   # sliding DFT of u at probe lines
        self.history_theta = []
        self.history_u_lqg = []
        self.history_u_ff = []
        self.history_u_total = []
        self.history_phase = []
        self.history_safety = []

    # ------------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas, k_step=0):
        W = self.W

        # 1. Sliding DFT update (O(1)/step/line): the PER-STEP parameter data update.
        #    S <- e^{jw dt} (S + x_new - x_old)
        i_buf = k_step % W
        y_old = self._y_buf[i_buf]
        u_old = self._u_buf[i_buf]
        self._Sy = self._twd * (self._Sy + (y_meas - y_old))
        self._Su = self._twd * (self._Su + (u_prev - u_old))
        self._y_buf[i_buf] = y_meas
        self._u_buf[i_buf] = u_prev

        # 2. FRF matching over the model grid (valid once the window is full).
        #    Balanced normalization per line:
        #        e = |Y - G U|^2 / (|Y|^2 + |G U|^2)  in [0, ~1]
        #    — bounded and resonance-safe on both sides (a line sitting on the TRUE
        #    resonance has |Y| large; one sitting on a MODEL resonance has |G U|
        #    large; either way the error stays well-conditioned).
        if k_step >= W:
            GU = self._G_ref * self._Su[None, :]
            num = np.abs(self._Sy[None, :] - GU)**2
            den = np.abs(self._Sy[None, :])**2 + np.abs(GU)**2 + 1e-30
            score = np.sum(num / den, axis=1)
            if k_step - self._k_last_switch >= self.dwell:
                i_best = int(np.argmin(score))
                if (i_best != self._i_active
                        and score[i_best]
                            < self.switch_margin * score[self._i_active]):
                    self._i_active = i_best
                    self._k_last_switch = k_step

        # 3. Parallel control-observer bank (state continuity across switches)
        for i, mdl in enumerate(self._bank):
            self._xb[i] = (mdl['A_obs_d'] @ self._xb[i]
                           + mdl['G_u'] * u_prev + mdl['G_y'] * y_meas)

        mdl = self._bank[self._i_active]
        x_hat = self._xb[self._i_active]
        self.history_theta.append(float(mdl['theta']))

        # 4. Scheduled LQG feedback on the active model
        u_lqg = float(np.squeeze(-mdl['K_lqr'] @ x_hat))
        u_lqg = np.clip(u_lqg, -self.u_max, self.u_max)
        self.history_u_lqg.append(u_lqg)

        # 5. Phase-locked learned feedforward (phase-indexed — unchanged by drift)
        phase = 2 * np.pi * (k_step % self.n_per) / self.n_per
        self.history_phase.append(phase)
        u_ff, _, _ = self.ff_nn.forward(self._zero_state, phase)
        u_ff = self.ff_alpha * u_ff
        self.history_u_ff.append(u_ff)

        # 6. Combine + governor, then add the identification probe (multisine at
        #    non-harmonic lines — the persistent excitation that keeps the plant
        #    parameters identifiable during stable cutting)
        u_proposed = np.clip(u_lqg + u_ff, -self.u_max, self.u_max)
        u_safe, violated = self.safety.filter_action(x_hat, u_proposed, u_lqg)
        t_k = k_step * self.dt
        u_probe = self.probe_amp * np.sum(
            np.sin(2 * np.pi * self.probe_freqs * t_k))
        u_out = float(np.clip(u_safe + u_probe, -self.u_max, self.u_max))

        self.history_u_total.append(u_out)
        self.history_safety.append(1 if violated else 0)

        return x_hat, u_out
