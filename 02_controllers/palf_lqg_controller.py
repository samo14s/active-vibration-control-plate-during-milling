"""
palf_lqg_controller.py
======================
PALF-LQG : Phase-Aware Learned Feedforward + LQG.

    u(t) = u_LQG(x_hat(t))  +  alpha * u_FF(phi(t))

- u_LQG : reactive LQG feedback (LQR gain + Kalman observer).
- u_FF  : a small neural feedforward INDEXED ON THE TOOTH-PASSING PHASE phi,
          trained by iterative learning (ILC) over closed-loop simulation runs.
          Because the training data is a function of the tooth-passing phase, the
          learned object is a *learned periodic (repetitive-control-like) map*
          u_FF(phi); it does not depend on the state at deployment (see step()).
- Safety : a heuristic control-Lyapunov-style voltage governor evaluated on the
           NOMINAL delay-free model. This is a soft actuator guard, NOT a stability
           certificate for the true delayed, time-periodic closed loop.

Honesty notes (see docs/AUDIT_FINDINGS.md and docs/CONTRIBUTION.md):
  * This controller is NOT "MPC" (no receding-horizon optimization) and the network
    is NOT "deep" (one hidden layer, 16 tanh units, ~161 parameters). It was renamed
    from the earlier, over-claiming "DARC-MPC v3".
  * The feedforward changes the periodic forcing, not the closed-loop poles; it does
    not by itself move the regenerative chatter stability boundary.
"""

import numpy as np
from scipy.linalg import solve_continuous_are, expm, solve_lyapunov


# ====================================================================
# Phase-indexed feedforward network (small MLP)
# ====================================================================
class PhaseFeedforwardNN:
    """
    Learns a periodic feedforward voltage u_FF(phi) from the tooth-passing phase.

    Input : [cos(phi), sin(phi)]  (the state channel is kept in the layout for
            interface compatibility but is fed zeros at train and deploy time, so
            the trained map is purely periodic — see PALF_LQG_Controller.step).
    Output: u_FF in [-u_FF_max, +u_FF_max].
    """
    def __init__(self, n_x, n_hidden=16, lr=0.01, seed=42, u_FF_max=30.0):
        self.n_x = n_x
        self.n_input = n_x + 2  # state channel (unused) + cos/sin phase
        self.n_hidden = n_hidden
        self.lr = lr
        self.u_FF_max = u_FF_max
        self.n_modes = n_x // 2

        rng = np.random.default_rng(seed)
        # Small init (conservative feedforward start)
        self.W1 = rng.normal(0, 0.1, (n_hidden, self.n_input))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.1, (1, n_hidden))
        self.b2 = np.zeros(1)

        # State scaling (only relevant if the state channel is ever activated)
        self.scale_pos = 1e6
        self.scale_vel = 1e3

    def _normalize_input(self, x, phase):
        x_n = np.zeros(self.n_x)
        x_n[:self.n_modes] = x[:self.n_modes] * self.scale_pos
        x_n[self.n_modes:] = x[self.n_modes:] * self.scale_vel
        x_n = np.tanh(x_n)
        phase_enc = np.array([np.cos(phase), np.sin(phase)])
        return np.concatenate([x_n, phase_enc])

    def forward(self, x, phase):
        inp = self._normalize_input(x, phase)
        h = np.tanh(self.W1 @ inp + self.b1)
        z = self.W2 @ h + self.b2
        u_FF = self.u_FF_max * np.tanh(z / self.u_FF_max)
        return float(u_FF.flatten()[0]), h, inp

    def backward(self, x, phase, target_u):
        u, h, inp = self.forward(x, phase)

        error_n = (u - target_u) / self.u_FF_max
        d_u_dz = 1.0 - (u / self.u_FF_max)**2
        d_z = error_n * d_u_dz

        d_W2 = d_z * h.reshape(1, -1)
        d_b2 = np.array([d_z])

        d_h = self.W2.flatten() * d_z * (1 - h**2)
        d_W1 = np.outer(d_h, inp)
        d_b1 = d_h

        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2

        return float((u - target_u)**2)


# ====================================================================
# Heuristic CLF voltage governor (NOT a stability certificate)
# ====================================================================
class CLFVoltageGovernor:
    """
    Soft actuator guard: blends a proposed voltage toward the LQG fallback when the
    proposed voltage would increase a quadratic V = x'Px on the NOMINAL, delay-free,
    disturbance-free LQR closed loop. This is a heuristic governor evaluated on the
    state estimate, not a Lyapunov stability proof for the true delayed periodic
    plant (which includes the milling force and the regenerative delay term).
    """
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
        margin = V_dot + self.alpha * V

        if margin <= 0:
            return u_proposed, False

        x_PB = float(x @ self.P @ self.B)
        V_dot_fb = 2.0 * x.T @ self.P @ (self.A @ x + self.B * u_fallback)
        if abs(x_PB * (u_proposed - u_fallback)) > 1e-15:
            target = -V_dot_fb - self.alpha * V
            denom = 2.0 * x_PB * (u_proposed - u_fallback)
            beta = np.clip(target / denom, 0.0, 1.0)
        else:
            beta = 0.0

        u_safe = u_fallback + beta * (u_proposed - u_fallback)
        self.n_violations += 1
        return u_safe, True


# ====================================================================
# PALF-LQG controller
# ====================================================================
class PALF_LQG_Controller:
    """
    Phase-Aware Learned Feedforward added residually to an LQG loop:

        u(t) = u_LQG(x_hat) + alpha * u_FF(phi)

    The LQG loop is always present (reactive baseline). The feedforward only adds a
    small, phase-locked correction learned by iterative learning in the simulator.
    """
    def __init__(self, plate, dt,
                 base_w_q=1e14, base_w_qd=1e8, base_w_r=1.0,
                 # feedforward network
                 ff_lr=0.01, ff_max=30.0, ff_alpha=1.0,
                 # phase indexing (tooth-passing period, in integration steps)
                 n_per=82,
                 # heuristic safety governor
                 safety_alpha=5.0,
                 # Kalman measurement-noise variance assumption (m^2)
                 kalman_V=1e-12,
                 # actuator limit
                 u_max=150.0,
                 verbose=True):

        self.plate = plate
        self.dt = dt
        self.verbose = verbose
        self.n_modes = plate.n_modes
        self.n_x = 2 * plate.n_modes
        self.u_max = u_max
        self.ff_alpha = ff_alpha
        self.n_per = n_per

        # Build LQG base (feedback gain + Kalman observer)
        self._build_base(base_w_q, base_w_qd, base_w_r, kalman_V)

        # Phase feedforward network
        self.ff_nn = PhaseFeedforwardNN(self.n_x, n_hidden=16,
                                        lr=ff_lr, u_FF_max=ff_max)

        # Heuristic CLF voltage governor on the nominal closed loop
        try:
            A_cl = self.A - self.B @ self.K_lqr
            self.P_lyap = solve_lyapunov(A_cl.T, -np.eye(self.n_x))
        except Exception:
            self.P_lyap = np.eye(self.n_x)
        self.safety = CLFVoltageGovernor(self.A, self.B, self.P_lyap,
                                         alpha=safety_alpha)

        # Deployment feeds a zero state to the feedforward (learned map is periodic)
        self._zero_state = np.zeros(self.n_x)

        # Histories
        self.history_u_lqg = []
        self.history_u_ff = []
        self.history_u_total = []
        self.history_phase = []
        self.history_safety = []

        if verbose:
            self._print_summary()

    def _build_base(self, w_q, w_qd, w_r, kalman_V=1e-12):
        n = self.n_modes
        A = np.zeros((self.n_x, self.n_x))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.linalg.solve(self.plate.Mp, self.plate.Kp)
        A[n:, n:] = -np.linalg.solve(self.plate.Mp, self.plate.Cp)
        B = np.zeros((self.n_x, 1))
        B[n:, 0] = np.linalg.solve(self.plate.Mp, self.plate.H_Pe_modal)
        C = np.zeros((1, self.n_x))
        C[0, :n] = self.plate.D_obs
        self.A = A; self.B = B; self.C = C

        # LQR base — SAME Q/R construction as LQGController so that, given identical
        # (w_q, w_qd, w_r), the feedback gain is identical to the standalone LQG
        # baseline (fair, apples-to-apples comparison).
        M_yp = np.outer(self.plate.D_obs, self.plate.D_obs)
        Q_top = w_q * M_yp + 1e-3 * np.eye(n)
        Q_bot = w_qd * np.eye(n) + 1e-3 * np.eye(n)
        Q = np.block([[Q_top, np.zeros((n, n))],
                      [np.zeros((n, n)), Q_bot]])
        R = np.array([[w_r]])
        P = solve_continuous_are(A, B, Q, R)
        self.K_lqr = np.linalg.solve(R, B.T @ P)

        # Kalman observer
        W_kal = 1e-6 * np.eye(self.n_x)
        V_kal = np.array([[kalman_V]])
        P_kal = solve_continuous_are(A.T, C.T, W_kal, V_kal)
        self.L_kal = P_kal @ C.T @ np.linalg.inv(V_kal)

        A_obs = A - self.L_kal @ C
        self.A_obs_d = expm(A_obs * self.dt)
        try:
            self.G_u = np.linalg.solve(A_obs,
                                       (self.A_obs_d - np.eye(self.n_x)) @ B)
            self.G_y = np.linalg.solve(A_obs,
                                       (self.A_obs_d - np.eye(self.n_x)) @ self.L_kal)
        except np.linalg.LinAlgError:
            self.A_obs_d = np.eye(self.n_x) + A_obs * self.dt
            self.G_u = B * self.dt
            self.G_y = self.L_kal * self.dt

    def _print_summary(self):
        print(f"\n{'='*70}")
        print(f"  PALF-LQG : Phase-Aware Learned Feedforward + LQG")
        print(f"{'='*70}")
        print(f"  Law       : u = u_LQG(x_hat) + alpha * u_FF(phi)")
        print(f"  u_FF       : {self.ff_nn.n_hidden} hidden tanh units, "
              f"lr={self.ff_nn.lr} (phase-only learned map)")
        print(f"  alpha (FF) : {self.ff_alpha}")
        print(f"  u_FF max   : +/-{self.ff_nn.u_FF_max}V")
        print(f"  ||K_LQR||  : {np.linalg.norm(self.K_lqr):.2e}")

    def pretrain_iterative_simulation(self, simulator, alpha3_t, alpha4_t, kp_idx,
                                      n_iterations=5, n_epochs_per_iter=30,
                                      verbose=True):
        """
        Iterative Learning Control via simulation.

        Each iteration:
          1. Simulate the current controller (LQG + alpha*u_FF).
          2. Collect (phase, y) along the run.
          3. Form a corrected feedforward target that would have reduced y:
             u_target = u_FF_current + eta * (-K_corr * sign(gain) * y).
          4. Train the phase network toward u_target.

        IMPORTANT: this must be called ONCE on the nominal (design) scenario, then the
        weights FROZEN. Do not re-train per evaluation scenario — that would score the
        controller on its own training data (see docs/AUDIT_FINDINGS.md).
        """
        if verbose:
            print(f"\n--- ILC feedforward pretraining (nominal scenario) ---")
            print(f"   iterations: {n_iterations}, epochs/iter: {n_epochs_per_iter}")

        rng = np.random.default_rng(42)
        all_iter_history = []

        # A few tooth-passing periods are enough to capture the periodic target
        n_sim_steps = min(simulator.nstep, 4 * self.n_per)

        for it in range(n_iterations):
            self.history_u_lqg = []
            self.history_u_ff = []
            self.history_u_total = []
            self.history_phase = []
            self.history_safety = []

            # 1. Simulate with the current controller
            res = simulator.simulate(alpha3_t, alpha4_t, kp_idx,
                                     controller=self,
                                     progress=False,
                                     stop_at_time=n_sim_steps * self.dt)

            y_sim = res['y'][:n_sim_steps]
            y_rms = np.sqrt(np.mean(y_sim**2)) * 1e6

            if verbose:
                print(f"\n   Iter {it+1}: y_RMS sim = {y_rms:.4f} um")

            # 2-3. Build corrected feedforward targets from the run
            n_collected = min(len(self.history_phase), n_sim_steps)
            X_train, Phase_train, U_target_train = [], [], []

            B_flat = self.B.flatten()
            gain_dc = float(self.plate.D_obs[0] * B_flat[self.n_modes])
            gain_sign = np.sign(gain_dc) if abs(gain_dc) > 1e-30 else 1.0
            K_correction = 1e6   # 1 V correction per 1 um vibration (heuristic)
            eta = 0.3            # iterative-learning step

            for k in range(n_collected):
                phase = self.history_phase[k]
                u_ff_curr = self.history_u_ff[k]
                y_k = y_sim[k]
                u_correction = -K_correction * gain_sign * y_k
                u_target = np.clip(u_ff_curr + eta * u_correction,
                                   -self.ff_nn.u_FF_max, self.ff_nn.u_FF_max)
                # Phase-only learned map: state channel trained on zeros
                X_train.append(self._zero_state.copy())
                Phase_train.append(phase)
                U_target_train.append(u_target)

            X_train = np.array(X_train)
            Phase_train = np.array(Phase_train)
            U_target_train = np.array(U_target_train)

            # 4. Train the phase network
            for epoch in range(n_epochs_per_iter):
                idx = rng.permutation(len(X_train))
                for j in idx[:min(len(idx), 500)]:
                    self.ff_nn.backward(X_train[j], Phase_train[j],
                                        U_target_train[j])

            errs = []
            for j in range(min(len(X_train), 200)):
                u_p, _, _ = self.ff_nn.forward(X_train[j], Phase_train[j])
                errs.append((u_p - U_target_train[j])**2)
            rmse = np.sqrt(np.mean(errs)) if errs else float('nan')

            all_iter_history.append({
                'iter': it+1, 'y_rms': y_rms, 'rmse': rmse,
                'u_target_std': float(U_target_train.std()) if len(U_target_train) else 0.0,
            })

            if verbose:
                print(f"     RMSE NN: {rmse:.3f}V, "
                      f"u_target std: {U_target_train.std():.2f}V")

        return all_iter_history

    def step(self, x_hat_prev, u_prev, y_meas, k_step=0):
        """
        One control step:
          1. Kalman observer update.
          2. LQG feedback (always present).
          3. Phase feedforward correction u_FF(phi).
          4. Combine + heuristic safety governor.
        """
        # 1. Kalman observer
        x_hat = (self.A_obs_d @ x_hat_prev
                 + self.G_u.flatten() * u_prev
                 + self.G_y.flatten() * y_meas)

        # 2. LQG feedback (reactive baseline)
        u_lqg = float(np.squeeze(-self.K_lqr @ x_hat))
        u_lqg = np.clip(u_lqg, -self.u_max, self.u_max)
        self.history_u_lqg.append(u_lqg)

        # 3. Phase-locked learned feedforward (periodic map, phase-only)
        phase = 2 * np.pi * (k_step % self.n_per) / self.n_per
        self.history_phase.append(phase)
        u_ff, _, _ = self.ff_nn.forward(self._zero_state, phase)
        u_ff = self.ff_alpha * u_ff
        self.history_u_ff.append(u_ff)

        # 4. Combine + heuristic safety governor
        u_proposed = np.clip(u_lqg + u_ff, -self.u_max, self.u_max)
        u_safe, violated = self.safety.filter_action(x_hat, u_proposed, u_lqg)
        u_safe = float(np.clip(u_safe, -self.u_max, self.u_max))

        self.history_u_total.append(u_safe)
        self.history_safety.append(1 if violated else 0)

        return x_hat, u_safe
