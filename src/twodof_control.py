"""
twodof_control.py
=================
Two-degree-of-freedom (2-DOF) controller for active chatter suppression:

    u(t) = -K x_hat(t)            (feedback: sets the stability boundary)
           + alpha * u_ff(phi, x_hat)   (phase-aware feedforward: rejects the
                                         periodic forced excitation)

Design rationale (honest separation of roles)
---------------------------------------------
* The **feedback** term is the only part that moves the closed-loop poles and
  therefore the only part that shifts the chatter stability boundary.  Its gain
  ``K`` may be supplied externally (e.g. an LQR gain, or a gain synthesised
  directly against the Floquet criterion in :mod:`floquet_synthesis`); if none
  is given an LQR gain is built internally.  The corresponding controlled
  stability-lobe diagram must be evaluated with :class:`cl_fdm.ClosedLoopFDM`.

* The **feedforward** term is a small phase-aware network trained (by iterative
  simulation) to anticipate and cancel the *periodic* cutting force at the
  tooth-passing frequency.  Being a feedforward signal it does **not** change
  the closed-loop poles and hence **cannot** extend the stability boundary; its
  benefit is a modest, consistent reduction of the forced vibration and of the
  peak actuator voltage inside the stable region.  Reporting it as a stability
  improvement (as is sometimes done with an ad-hoc damping multiplier) is
  incorrect -- this module keeps the two effects strictly separate.

A Lyapunov inequality filter keeps the combined command inside the region where
the feedback part guarantees decay, and an optional online adapter raises a
robustness weight when the estimated frequency drifts.

This is a faithful, honestly-labelled reimplementation of the "residual
correction" feedforward controller in the source package, with the feedback
gain exposed so it can be chosen by Floquet-direct synthesis.

Author: contribution built on the model of
        Du et al., Int. J. Mech. Sci. 274 (2024) 109257.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are, expm, solve_lyapunov
from collections import deque


# ---------------------------------------------------------------------------
class PhaseAwareFeedforward:
    """Small network u_ff(phase, x_hat) -> bounded feedforward voltage.

    Inputs are the (scaled, tanh-squashed) modal state estimate plus the
    cos/sin encoding of the tooth-passing phase; the single output is squashed
    to +/- u_ff_max.  Trained by gradient descent on a feedforward target.
    """

    def __init__(self, n_x, n_hidden=16, lr=0.01, seed=42, u_ff_max=30.0):
        self.n_x = n_x
        self.n_input = n_x + 2
        self.n_hidden = n_hidden
        self.lr = lr
        self.u_ff_max = u_ff_max
        self.n_modes = n_x // 2
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (n_hidden, self.n_input))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.1, (1, n_hidden))
        self.b2 = np.zeros(1)
        self.scale_pos = 1e6
        self.scale_vel = 1e3

    def _normalize_input(self, x, phase):
        x_n = np.zeros(self.n_x)
        x_n[:self.n_modes] = x[:self.n_modes] * self.scale_pos
        x_n[self.n_modes:] = x[self.n_modes:] * self.scale_vel
        x_n = np.tanh(x_n)
        return np.concatenate([x_n, [np.cos(phase), np.sin(phase)]])

    def forward(self, x, phase):
        inp = self._normalize_input(x, phase)
        h = np.tanh(self.W1 @ inp + self.b1)
        z = self.W2 @ h + self.b2
        u_ff = self.u_ff_max * np.tanh(z / self.u_ff_max)
        return float(u_ff.flatten()[0]), h, inp

    def backward(self, x, phase, target_u):
        u, h, inp = self.forward(x, phase)
        error_n = (u - target_u) / self.u_ff_max
        d_z = error_n * (1.0 - (u / self.u_ff_max) ** 2)
        d_W2 = d_z * h.reshape(1, -1)
        d_b2 = np.array([d_z])
        d_h = self.W2.flatten() * d_z * (1 - h ** 2)
        d_W1 = np.outer(d_h, inp)
        d_b1 = d_h
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        return float((u - target_u) ** 2)


# ---------------------------------------------------------------------------
class LyapunovSafetyFilter:
    """Blend the proposed command toward the pure-feedback command whenever the
    Lyapunov decay inequality V_dot + alpha V <= 0 would otherwise be violated."""

    def __init__(self, A, B, P_lyap, alpha=10.0):
        self.A = A
        self.B = B.flatten() if B.ndim > 1 else B
        self.P = P_lyap
        self.alpha = alpha
        self.n_violations = 0
        self.n_calls = 0

    def filter_action(self, x, u_proposed, u_fallback):
        self.n_calls += 1
        x_dot = self.A @ x + self.B * u_proposed
        V = x.T @ self.P @ x
        V_dot = 2.0 * x.T @ self.P @ x_dot
        if V_dot + self.alpha * V <= 0:
            return u_proposed, False
        x_PB = float(x @ self.P @ self.B)
        V_dot_fb = 2.0 * x.T @ self.P @ (self.A @ x + self.B * u_fallback)
        if abs(x_PB * (u_proposed - u_fallback)) > 1e-15:
            beta = np.clip((-V_dot_fb - self.alpha * V) /
                           (2.0 * x_PB * (u_proposed - u_fallback)), 0.0, 1.0)
        else:
            beta = 0.0
        self.n_violations += 1
        return u_fallback + beta * (u_proposed - u_fallback), True


# ---------------------------------------------------------------------------
class OnlineFreqAdapter:
    """Raises a robustness weight when the estimated frequency drifts."""

    def __init__(self, omega_nominal, gamma=0.0005):
        self.omega_nom = float(omega_nominal[0])
        self.omega_hat = self.omega_nom
        self.gamma = gamma
        self.error_buffer = deque(maxlen=100)

    def update(self, x_meas, x_pred):
        self.error_buffer.append(x_meas[0] - x_pred[0])
        if len(self.error_buffer) > 20:
            mean_err = np.mean(self.error_buffer)
            if abs(mean_err) > 1e-6:
                self.omega_hat += self.gamma * np.sign(mean_err) * self.omega_nom
                self.omega_hat = np.clip(self.omega_hat,
                                         0.7 * self.omega_nom, 1.3 * self.omega_nom)
        mismatch = abs(self.omega_hat - self.omega_nom) / self.omega_nom
        return 1.0 + 5.0 * mismatch


# ---------------------------------------------------------------------------
class TwoDOFController:
    """Feedback + phase-aware feedforward controller.

    Parameters
    ----------
    plate       : PlateModel (with D_obs and H_Pe_modal set)
    dt          : sample time [s]
    K_feedback  : optional (1 x 2n) feedback gain.  If None an LQR gain is built
                  from (base_w_q, base_w_qd, base_w_r).
    ff_lr, ff_max, ff_alpha : feedforward learning rate, output bound and gain
    alpha4_periodic, n_per  : periodic regenerative coefficient over one period
    safety_alpha            : Lyapunov filter margin
    enable_adaptation       : enable the online frequency adapter
    u_max                   : actuator voltage saturation [V]
    """

    def __init__(self, plate, dt,
                 K_feedback=None,
                 base_w_q=1e14, base_w_qd=1e8, base_w_r=1.0,
                 ff_lr=0.01, ff_max=30.0, ff_alpha=1.0,
                 alpha4_periodic=None, n_per=82,
                 safety_alpha=5.0, enable_adaptation=True,
                 u_max=150.0, verbose=True):
        self.plate = plate
        self.dt = dt
        self.verbose = verbose
        self.n_modes = plate.n_modes
        self.n_x = 2 * plate.n_modes
        self.u_max = u_max
        self.ff_alpha = ff_alpha
        self.alpha4_periodic = alpha4_periodic
        self.n_per = n_per
        self.needs_kstep = True             # ask the solver for the step index

        self._build_base(K_feedback, base_w_q, base_w_qd, base_w_r)

        self.ff_nn = PhaseAwareFeedforward(self.n_x, n_hidden=16,
                                           lr=ff_lr, u_ff_max=ff_max)
        try:
            A_cl = self.A - self.B @ self.K_lqr
            self.P_lyap = solve_lyapunov(A_cl.T, -np.eye(self.n_x))
        except Exception:
            self.P_lyap = np.eye(self.n_x)
        self.safety = LyapunovSafetyFilter(self.A, self.B, self.P_lyap,
                                           alpha=safety_alpha)
        self.enable_adaptation = enable_adaptation
        self.rls = OnlineFreqAdapter(plate.omega_n)
        self.x_prev = np.zeros(self.n_x)
        self.u_prev = 0.0
        self._reset_histories()

    # ------------------------------------------------------------------
    def _reset_histories(self):
        self.history_u_lqg = []
        self.history_u_ff = []
        self.history_u_total = []
        self.history_phase = []
        self.history_safety = []

    def _build_base(self, K_feedback, w_q, w_qd, w_r):
        n = self.n_modes
        A = np.zeros((self.n_x, self.n_x))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.linalg.solve(self.plate.Mp, self.plate.Kp)
        A[n:, n:] = -np.linalg.solve(self.plate.Mp, self.plate.Cp)
        B = np.zeros((self.n_x, 1))
        B[n:, 0] = np.linalg.solve(self.plate.Mp, self.plate.H_Pe_modal)
        C = np.zeros((1, self.n_x))
        C[0, :n] = self.plate.D_obs
        self.A, self.B, self.C = A, B, C

        if K_feedback is not None:
            self.K_lqr = np.asarray(K_feedback, float).reshape(1, self.n_x)
        else:
            M_yp = np.outer(self.plate.D_obs, self.plate.D_obs)
            Q = np.block([[w_q * M_yp + 1e-3 * np.eye(n), np.zeros((n, n))],
                          [np.zeros((n, n)), w_qd * np.eye(n) + 1e-3 * np.eye(n)]])
            R = np.array([[w_r]])
            P = solve_continuous_are(A, B, Q, R)
            self.K_lqr = np.linalg.solve(R, B.T @ P)

        # observer (Kalman) discretisation
        self.A_d = expm(A * self.dt)
        try:
            self.B_d = np.linalg.solve(A, (self.A_d - np.eye(self.n_x)) @ B)
        except np.linalg.LinAlgError:
            self.B_d = B * self.dt
        W_kal = 1e-6 * np.eye(self.n_x)
        V_kal = np.array([[1e-12]])
        P_kal = solve_continuous_are(A.T, C.T, W_kal, V_kal)
        self.L_kal = P_kal @ C.T @ np.linalg.inv(V_kal)
        A_obs = A - self.L_kal @ C
        self.A_obs_d = expm(A_obs * self.dt)
        try:
            self.G_u = np.linalg.solve(A_obs, (self.A_obs_d - np.eye(self.n_x)) @ B)
            self.G_y = np.linalg.solve(A_obs, (self.A_obs_d - np.eye(self.n_x)) @ self.L_kal)
        except np.linalg.LinAlgError:
            self.A_obs_d = np.eye(self.n_x) + A_obs * self.dt
            self.G_u = B * self.dt
            self.G_y = self.L_kal * self.dt

    # ------------------------------------------------------------------
    def pretrain_iterative_simulation(self, simulator, alpha3_t, alpha4_t, kp_idx,
                                      n_iterations=30, n_epochs_per_iter=15,
                                      verbose=True):
        """Iterative-learning pre-training of the feedforward network.

        Each iteration simulates the current closed loop, then nudges the
        feedforward target toward the correction that would have reduced the
        observed vibration; the network is retrained on the updated targets.
        """
        rng = np.random.default_rng(42)
        history = []
        n_sim_steps = min(simulator.nstep, 4 * self.n_per)
        for it in range(n_iterations):
            self._reset_histories()
            res = simulator.simulate(alpha3_t, alpha4_t, kp_idx, controller=self,
                                     progress=False,
                                     stop_at_time=n_sim_steps * self.dt)
            y_sim = res['y'][:n_sim_steps]
            y_rms = np.sqrt(np.mean(y_sim ** 2)) * 1e6
            n_collected = min(len(self.history_phase), n_sim_steps)

            X_train, Phase_train, U_target = [], [], []
            B_flat = self.B.flatten()
            gain_dc = float(self.plate.D_obs[0] * B_flat[self.n_modes])
            gain_sign = np.sign(gain_dc) if abs(gain_dc) > 1e-30 else 1.0
            K_correction = 1e6
            eta = 0.3
            for k in range(n_collected):
                u_ff_curr = self.history_u_ff[k]
                u_corr = -K_correction * gain_sign * y_sim[k]
                u_t = np.clip(u_ff_curr + eta * u_corr,
                              -self.ff_nn.u_ff_max, self.ff_nn.u_ff_max)
                X_train.append(np.zeros(self.n_x))
                Phase_train.append(self.history_phase[k])
                U_target.append(u_t)
            X_train = np.array(X_train)
            Phase_train = np.array(Phase_train)
            U_target = np.array(U_target)

            for _ in range(n_epochs_per_iter):
                idx = rng.permutation(len(X_train))
                for j in idx[:min(len(idx), 500)]:
                    self.ff_nn.backward(X_train[j], Phase_train[j], U_target[j])
            errs = [(self.ff_nn.forward(X_train[j], Phase_train[j])[0] - U_target[j]) ** 2
                    for j in range(min(len(X_train), 200))]
            rmse = np.sqrt(np.mean(errs)) if errs else 0.0
            history.append(dict(iter=it + 1, y_rms=y_rms, rmse=rmse))
            if verbose:
                print(f"   iter {it+1:2d}: y_rms={y_rms:.4f}um  ff_rmse={rmse:.3f}V")
        return history

    # ------------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas, k_step=0):
        # observer
        x_hat = (self.A_obs_d @ x_hat_prev
                 + self.G_u.flatten() * u_prev
                 + self.G_y.flatten() * y_meas)
        if self.enable_adaptation:
            x_pred = self.A_d @ self.x_prev + self.B_d.flatten() * self.u_prev
            self.rls.update(x_hat, x_pred)
        # feedback
        u_fb = float(np.clip(np.squeeze(-self.K_lqr @ x_hat), -self.u_max, self.u_max))
        self.history_u_lqg.append(u_fb)
        # phase-aware feedforward
        phase = 2 * np.pi * (k_step % self.n_per) / self.n_per
        self.history_phase.append(phase)
        u_ff, _, _ = self.ff_nn.forward(x_hat, phase)
        u_ff *= self.ff_alpha
        self.history_u_ff.append(u_ff)
        # combine + safety
        u_proposed = np.clip(u_fb + u_ff, -self.u_max, self.u_max)
        u_safe, violated = self.safety.filter_action(x_hat, u_proposed, u_fb)
        u_safe = float(np.clip(u_safe, -self.u_max, self.u_max))
        self.history_u_total.append(u_safe)
        self.history_safety.append(1 if violated else 0)
        self.x_prev = x_hat.copy()
        self.u_prev = u_safe
        return x_hat, u_safe

    def discretize_observer(self):    # API compatibility with LQGController
        pass
