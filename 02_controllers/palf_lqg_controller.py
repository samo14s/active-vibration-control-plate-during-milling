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

    def _closed_loop_frf(self, omega_arr):
        """
        FRF G(jw) from an ADDITIVE feedforward voltage (entering where u does) to the
        measured displacement y, for the DESIGN closed loop (plant model + LQR gain +
        Kalman observer; no delay/regeneration). Used by the model-inverse ILC.

            xi = [x; x_hat]
            x_dot    = A x - B K x_hat + B u_ff
            x_hat_dot= L C x + (A - B K - L C) x_hat + B u_ff
            y        = C x
        """
        n2 = self.n_x
        A = self.A
        B = self.B.reshape(n2, 1)
        C = self.C.reshape(1, n2)
        K = self.K_lqr.reshape(1, n2)
        L = self.L_kal.reshape(n2, 1)
        Acl = np.block([[A, -B @ K],
                        [L @ C, A - B @ K - L @ C]])
        Bcl = np.vstack([B, B])
        Ccl = np.hstack([C, np.zeros_like(C)])
        G = np.empty(len(omega_arr), dtype=complex)
        I = np.eye(2 * n2)
        for i, w in enumerate(omega_arr):
            G[i] = (Ccl @ np.linalg.solve(1j * w * I - Acl, Bcl))[0, 0]
        return G

    def pretrain_iterative_simulation(self, simulator, alpha3_t, alpha4_t, kp_idx,
                                      n_iterations=10, n_epochs_per_iter=30,
                                      n_harmonics=10, eta_ilc=0.5,
                                      verbose=True):
        """
        Frequency-domain MODEL-INVERSE Iterative Learning Control.

        The residual y after LQG is dominated by the tooth-passing-periodic forced
        response. Each ILC trial:
          1. Simulate the current controller (LQG + alpha*u_FF) on the design scenario
             long enough to reach the periodic steady state.
          2. DFT the steady-state residual over an integer number of tooth periods to
             get the harmonic coefficients Y_h (h = 0..n_harmonics).
          3. Model-inverse update of the feedforward harmonics:
                 U_h <- U_h - eta * Y_h / G(j h w_tau)
             where G is the design closed-loop FRF from u_ff to y (_closed_loop_frf).
             Because the update iterates against the TRUE simulated plant (5 modes +
             regenerative delay), plant/model mismatch at the harmonics is absorbed by
             the iteration (standard frequency-domain ILC contraction argument).
          4. Rebuild the periodic target u_target(phi) from the harmonics and train
             the phase network on it.

        The best-so-far harmonic set (lowest steady-state y_RMS) is kept, so a late
        divergent trial cannot degrade the frozen controller.

        IMPORTANT: call ONCE on the nominal (design) scenario, then FREEZE the weights.
        Do not re-train per evaluation scenario (docs/AUDIT_FINDINGS.md).
        """
        if verbose:
            print(f"\n--- Model-inverse ILC feedforward pretraining (nominal) ---")
            print(f"   trials: {n_iterations}, harmonics: {n_harmonics}, "
                  f"eta: {eta_ilc}")

        rng = np.random.default_rng(42)
        n_per = self.n_per
        H = int(min(n_harmonics, n_per // 2 - 1))

        # Simulation horizon: settle + FFT windows (periodic steady state)
        P_settle, P_fft = 6, 4
        n_sim_steps = min(simulator.nstep, (P_settle + P_fft) * n_per)
        n_fft = P_fft * n_per

        # Design closed-loop FRF at the tooth-passing harmonics (tiled period n_per*dt)
        w_tau = 2 * np.pi / (n_per * self.dt)
        omega_h = np.arange(1, H + 1) * w_tau
        G_h = self._closed_loop_frf(omega_h)
        G_0 = self._closed_loop_frf(np.array([1e-6]))[0].real   # DC gain
        g_floor = 1e-3 * np.max(np.abs(G_h))                     # ill-conditioned guard

        U = np.zeros(H + 1, dtype=complex)     # current FF harmonics (U[0] = DC)
        best = dict(y_rms=np.inf, U=U.copy())
        all_iter_history = []
        k_idx = np.arange(n_per)
        phases = 2 * np.pi * k_idx / n_per

        def _target_from_harmonics(Uh):
            u_t = np.full(n_per, Uh[0].real)
            for h in range(1, H + 1):
                u_t += 2.0 * np.real(Uh[h] * np.exp(1j * h * 2 * np.pi * k_idx / n_per))
            return np.clip(u_t, -self.ff_nn.u_FF_max, self.ff_nn.u_FF_max)

        def _fit_nn(u_t):
            for _ in range(n_epochs_per_iter):
                for j in rng.permutation(n_per):
                    self.ff_nn.backward(self._zero_state, phases[j], u_t[j])

        for it in range(n_iterations):
            self.history_u_lqg = []
            self.history_u_ff = []
            self.history_u_total = []
            self.history_phase = []
            self.history_safety = []

            # 1. Simulate current controller to periodic steady state
            res = simulator.simulate(alpha3_t, alpha4_t, kp_idx,
                                     controller=self, progress=False,
                                     stop_at_time=n_sim_steps * self.dt)
            n_got = res['stop_idx'] + 1
            if n_got < n_sim_steps:            # diverged trial: keep best, stop
                if verbose:
                    print(f"   Trial {it+1}: diverged — reverting to best trial")
                break
            y_ss = res['y'][n_sim_steps - n_fft:n_sim_steps]
            y_rms = np.sqrt(np.mean(y_ss**2)) * 1e6

            # 2. Harmonics of the steady-state residual (DFT bins h*P_fft are exact)
            Yf = np.fft.fft(y_ss) / n_fft
            Y_h = np.array([Yf[h * P_fft] for h in range(H + 1)])

            if y_rms < best['y_rms']:
                best = dict(y_rms=y_rms, U=U.copy())

            # 3. Model-inverse harmonic update
            U_new = U.copy()
            U_new[0] -= eta_ilc * (Y_h[0].real / G_0 if abs(G_0) > 1e-30 else 0.0)
            for h in range(1, H + 1):
                if np.abs(G_h[h - 1]) > g_floor:
                    U_new[h] -= eta_ilc * Y_h[h] / G_h[h - 1]
            U = U_new

            # 4. Rebuild periodic target and refit the phase network
            u_t = _target_from_harmonics(U)
            _fit_nn(u_t)

            errs = [(self.ff_nn.forward(self._zero_state, phases[j])[0] - u_t[j])**2
                    for j in range(n_per)]
            rmse = float(np.sqrt(np.mean(errs)))
            all_iter_history.append({'iter': it + 1, 'y_rms': y_rms, 'rmse': rmse,
                                     'u_target_std': float(u_t.std())})
            if verbose:
                print(f"   Trial {it+1}: y_RMS(ss) = {y_rms:.4f} um | "
                      f"NN fit RMSE {rmse:.3f} V | u_t std {u_t.std():.2f} V")

        # Freeze on the BEST harmonic set (guards against a late bad trial)
        if best['y_rms'] < np.inf:
            u_t = _target_from_harmonics(best['U'])
            _fit_nn(u_t)
            if verbose:
                print(f"   Frozen at best trial: y_RMS(ss) = {best['y_rms']:.4f} um")

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
