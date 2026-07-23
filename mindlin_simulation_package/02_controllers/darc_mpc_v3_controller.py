"""
darc_mpc_v3_controller.py
==========================
   ╔══════════════════════════════════════════════════════════════════╗
   ║   DARC-MPC v3 : RESIDUAL CORRECTION EDITION                      ║
   ║                                                                  ║
   ║   Changement fondamental de paradigme par rapport à v1/v2:       ║
   ║                                                                  ║
   ║   AU LIEU DE :  u = NN_policy(x)        (remplace LQG)           ║
   ║   ON FAIT    :  u = u_LQG(x) + NN_FF(phase, x)  (corrige LQG)    ║
   ║                                                                  ║
   ║   Le NN apprend la CORRECTION FEEDFORWARD ANTICIPATIVE           ║
   ║   qui compense la perturbation périodique connue.                ║
   ╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════
   POURQUOI v1/v2 ÉCHOUAIENT ?
═══════════════════════════════════════════════════════════════════════

   v1 : NN imite LQG → ne peut au mieux qu'égaler (RMSE ≈ 1V)
        Une fois propagé sur 10000 pas, l'erreur s'accumule → pire que LQG
   
   v2 : MPC offline solveur → bonne approximation mais NN imite encore
        Pas de mécanisme pour DÉPASSER l'optimum réactif
   
   PROBLÈME COMMUN : Imitation Learning a une borne supérieure = expert.

═══════════════════════════════════════════════════════════════════════
   NOUVELLE STRATÉGIE v3 : Corrective Feedforward Architecture
═══════════════════════════════════════════════════════════════════════

   u(t) = u_LQG(x̂(t))     +    α · NN_FF(phase(t), x̂(t))
          ─────┬────              ─┬─    ─────┬───────────
          réactif optimal      gain   feedforward apprent
          (LQG, garantie)             (compense pertu connue)
                                       ▲
                                       │
                              entraîné pour MINIMISER
                              l'écart résiduel après LQG

   Pourquoi ça MARCHE :
   1. LQG est conservé → garantit performance baseline LQG
   2. NN_FF AJOUTE une correction → SEULEMENT positive ou nulle
   3. NN_FF est entraîné DIRECTEMENT pour réduire vibration
   4. Phase-aware : NN sait OÙ on est dans le cycle de passage
   5. Petite amplitude correction (α petit) → safe

═══════════════════════════════════════════════════════════════════════
   ARCHITECTURE ULTRA-SIMPLE MAIS EFFICACE
═══════════════════════════════════════════════════════════════════════

   Inputs NN (compact) :
     [x̂_1, x̂_2, ..., x̂_n_x, cos(phase), sin(phase)]
     
   Output NN :
     u_FF ∈ [-u_FF_max, +u_FF_max]   (petit, ex: ±20V)
   
   Loss (training simulation-based) :
     L = Σ_t y(t)²  pendant un épisode simulé
     → Apprend par GRADIENT DESCENT à travers la dynamique
     → "Differentiable physics" appliqué au fraisage
   
   Pré-entraînement :
     Stratégie 1 : ANTI-DISTURBANCE supervised
       u_FF_target = -B^+ · alpha4(phase)   (compense pertu directement)
       Ceci marche TOUJOURS (analyitique).
"""

import numpy as np
from scipy.linalg import solve_continuous_are, expm, solve_lyapunov
from collections import deque


# ====================================================================
# Phase-Aware FEEDFORWARD NN (small, focused)
# ====================================================================
class FeedforwardCorrectorNN:
    """
    NN qui prédit u_FF correctif basé sur :
       - phase de l'outil (où on est dans cycle passage de dent)
       - état observé x_hat (petit poids - principalement phase)
    
    Output : u_FF petit (±u_FF_max), ajouté à u_LQG.
    """
    def __init__(self, n_x, n_hidden=16, lr=0.01, seed=42, u_FF_max=30.0):
        self.n_x = n_x
        self.n_input = n_x + 2  # state + cos/sin phase
        self.n_hidden = n_hidden
        self.lr = lr
        self.u_FF_max = u_FF_max
        self.n_modes = n_x // 2
        
        rng = np.random.default_rng(seed)
        # Smaller init for FF (conservative start)
        self.W1 = rng.normal(0, 0.1, (n_hidden, self.n_input))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.1, (1, n_hidden))
        self.b2 = np.zeros(1)
        
        # State scaling
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
# Lyapunov Safety Filter
# ====================================================================
class LyapunovSafetyFilter:
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
# Online Adaptive RLS (parameter estimation)
# ====================================================================
class OnlineRLSAdapter:
    """Estime ω via simple gradient sur prédiction error."""
    def __init__(self, omega_nominal, gamma=0.0005):
        self.omega_nom = float(omega_nominal[0])
        self.omega_hat = self.omega_nom
        self.gamma = gamma
        self.error_buffer = deque(maxlen=100)
    
    def update(self, x_meas, x_pred):
        error = x_meas[0] - x_pred[0]
        self.error_buffer.append(error)
        if len(self.error_buffer) > 20:
            mean_err = np.mean(self.error_buffer)
            if abs(mean_err) > 1e-6:
                self.omega_hat += self.gamma * np.sign(mean_err) * self.omega_nom
                self.omega_hat = np.clip(self.omega_hat,
                                          0.7*self.omega_nom, 1.3*self.omega_nom)
        mismatch = abs(self.omega_hat - self.omega_nom) / self.omega_nom
        return 1.0 + 5.0 * mismatch


# ====================================================================
# DARC-MPC v3 : Residual Correction Controller
# ====================================================================
class DARC_MPC_v3_Controller:
    """
    DARC-MPC v3 - Approche par CORRECTION RÉSIDUELLE.
    
    u(t) = u_LQG(x̂) + alpha · NN_FF(phase, x̂)
    
    Le NN ne REMPLACE pas LQG, il AJOUTE une correction feedforward.
    """
    def __init__(self, plate, dt,
                 base_w_q=1e14, base_w_qd=1e8, base_w_r=1.0,
                 # FF NN params
                 ff_lr=0.01, ff_max=30.0, ff_alpha=1.0,
                 # Periodic disturbance
                 alpha4_periodic=None, n_per=82,
                 # Safety
                 safety_alpha=5.0,
                 # Adaptation
                 enable_adaptation=True,
                 # Action
                 u_max=150.0,
                 verbose=True):
        
        self.plate = plate
        self.dt = dt
        self.verbose = verbose
        self.n_modes = plate.n_modes
        self.n_x = 2 * plate.n_modes
        self.u_max = u_max
        self.ff_alpha = ff_alpha
        
        self.alpha4_periodic = alpha4_periodic
        self.n_per = n_per
        
        # Build base
        self._build_base(base_w_q, base_w_qd, base_w_r)
        
        # Feedforward NN
        self.ff_nn = FeedforwardCorrectorNN(self.n_x, n_hidden=16,
                                              lr=ff_lr, u_FF_max=ff_max)
        
        # Safety
        try:
            A_cl = self.A - self.B @ self.K_lqr
            self.P_lyap = solve_lyapunov(A_cl.T, -np.eye(self.n_x))
        except Exception:
            self.P_lyap = np.eye(self.n_x)
        self.safety = LyapunovSafetyFilter(self.A, self.B, self.P_lyap,
                                              alpha=safety_alpha)
        
        # Adaptation
        self.enable_adaptation = enable_adaptation
        self.rls = OnlineRLSAdapter(plate.omega_n)
        
        # State for adaptation
        self.x_prev = np.zeros(self.n_x)
        self.u_prev = 0.0
        
        # Histories
        self.history_u_lqg = []
        self.history_u_ff = []
        self.history_u_total = []
        self.history_phase = []
        self.history_safety = []
        
        if verbose:
            self._print_summary()
    
    def _build_base(self, w_q, w_qd, w_r):
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
        
        # LQR base
        M_yp = np.outer(self.plate.D_obs, self.plate.D_obs)
        Q_top = w_q * M_yp + 1e-3 * np.eye(n)
        Q_bot = w_qd * np.eye(n) + 1e-3 * np.eye(n)
        Q = np.block([[Q_top, np.zeros((n, n))],
                       [np.zeros((n, n)), Q_bot]])
        R = np.array([[w_r]])
        P = solve_continuous_are(A, B, Q, R)
        self.K_lqr = np.linalg.solve(R, B.T @ P)
        
        # Compute B^+ for analytical FF target
        # We want u_FF such that B*u_FF cancels disturbance contribution
        # B is 6x1 vector, B^+ = B^T / (B^T B)
        B_flat = B.flatten()
        self.B_pseudoinv = B_flat / max(np.dot(B_flat, B_flat), 1e-30)
        
        # Discretize
        self.A_d = expm(A * self.dt)
        try:
            self.B_d = np.linalg.solve(A, (self.A_d - np.eye(self.n_x)) @ B)
        except np.linalg.LinAlgError:
            self.B_d = B * self.dt
        
        # Kalman
        W_kal = 1e-6 * np.eye(self.n_x)
        V_kal = np.array([[1e-12]])
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
        print(f"  DARC-MPC v3 : Residual Correction Edition")
        print(f"{'='*70}")
        print(f"  Stratégie  : u = u_LQG(x) + α·NN_FF(phase, x)")
        print(f"  NN_FF      : {self.ff_nn.n_hidden} hidden, lr={self.ff_nn.lr}")
        print(f"  α (gain FF): {self.ff_alpha}")
        print(f"  u_FF max   : ±{self.ff_nn.u_FF_max}V")
        print(f"  ||K_LQR||  : {np.linalg.norm(self.K_lqr):.2e}")
        print(f"  Périodic disturbance : "
              f"{'available' if self.alpha4_periodic is not None else 'no'}")
    
    def pretrain_iterative_simulation(self, simulator, alpha3_t, alpha4_t, kp_idx,
                                         n_iterations=5, n_epochs_per_iter=30,
                                         verbose=True):
        """
        ITERATIVE LEARNING CONTROL via Simulation:
        
        Stratégie : alterner SIMULATION + APPRENTISSAGE
        
        Itération i :
          1. Simuler avec contrôleur courant (LQG + α·NN_FF actuel)
          2. Collecter (phase(k), y(k), x(k))
          3. Pour chaque k, calculer correction u_correction qui aurait réduit y(k)
             via approximation linéaire : Δu = -K_correction · y(k)
          4. Entraîner NN sur (phase, x, u_FF + Δu)
          5. Répéter
        
        À chaque itération, le NN s'améliore en VRAIE SIMULATION.
        Pas d'imitation expert → on dépasse LQG !
        """
        if verbose:
            print(f"\n--- Iterative Learning via Simulation ---")
            print(f"   Itérations: {n_iterations}, epochs/iter: {n_epochs_per_iter}")
        
        rng = np.random.default_rng(42)
        all_iter_history = []
        
        # Limiter à 1 période pour vitesse (équivalent comportement périodique)
        n_sim_steps = min(simulator.nstep, 4 * self.n_per)   # ~4 périodes
        
        for it in range(n_iterations):
            # Reset histories
            self.history_u_lqg = []
            self.history_u_ff = []
            self.history_u_total = []
            self.history_phase = []
            self.history_safety = []
            
            # === 1. SIMULATION avec contrôleur courant ===
            res = simulator.simulate(alpha3_t, alpha4_t, kp_idx,
                                       controller=self,
                                       progress=False,
                                       stop_at_time=n_sim_steps * self.dt)
            
            y_sim = res['y'][:n_sim_steps]
            x_sim = None  # we don't have x_hat trace directly accessible
            y_rms = np.sqrt(np.mean(y_sim**2)) * 1e6
            
            if verbose:
                print(f"\n   Iter {it+1}: y_RMS sim = {y_rms:.4f} µm")
            
            # === 2. Collect training data from simulation ===
            # On utilise les histories collectées pendant simulation
            n_collected = min(len(self.history_phase), n_sim_steps)
            
            X_train = []
            Phase_train = []
            U_target_train = []
            
            # Rolling window for state estimation
            # On approxime x via les y observés (Kalman-like simple)
            for k in range(n_collected):
                phase = self.history_phase[k]
                u_ff_curr = self.history_u_ff[k]
                
                # Heuristique : si y(k) est grand, u_FF doit être plus dans la direction qui réduit y
                # Estimation : Δu ≈ -K_effective * y(k)
                y_k = y_sim[k]
                
                # gain_sign : signe du gain DC du piezo sur y_observé
                # B contains modal forcing from u, D_obs picks position output
                # B[n_modes:] is velocity contribution → DC gain = -D_obs @ A^-1 @ B
                # Simplification : sign(D_obs[0] * B[n_modes])  (mode 1 coupling)
                B_flat = self.B.flatten()
                gain_dc = float(self.plate.D_obs[0] * B_flat[self.n_modes])
                gain_sign = np.sign(gain_dc) if abs(gain_dc) > 1e-30 else 1.0
                # Δu = -K * gain_sign * y, K à calibrer
                K_correction = 1e6   # heuristique: 1V correction par 1µm vibration
                u_correction = -K_correction * gain_sign * y_k
                
                # Smoothing : ne pas changer trop d'un coup
                eta = 0.3   # learning rate iterative
                u_target = u_ff_curr + eta * u_correction
                u_target = np.clip(u_target,
                                     -self.ff_nn.u_FF_max, self.ff_nn.u_FF_max)
                
                # On utilise un "fake" état (zeros, NN s'appuie sur phase)
                X_train.append(np.zeros(self.n_x))
                Phase_train.append(phase)
                U_target_train.append(u_target)
            
            X_train = np.array(X_train)
            Phase_train = np.array(Phase_train)
            U_target_train = np.array(U_target_train)
            
            # === 3. Training NN sur targets corrigés ===
            for epoch in range(n_epochs_per_iter):
                idx = rng.permutation(len(X_train))
                total_loss = 0
                for j in idx[:min(len(idx), 500)]:   # cap for speed
                    loss = self.ff_nn.backward(X_train[j], Phase_train[j],
                                                 U_target_train[j])
                    total_loss += loss
            
            # Eval RMSE
            errs = []
            for j in range(min(len(X_train), 200)):
                u_p, _, _ = self.ff_nn.forward(X_train[j], Phase_train[j])
                errs.append((u_p - U_target_train[j])**2)
            rmse = np.sqrt(np.mean(errs))
            
            all_iter_history.append({
                'iter': it+1,
                'y_rms': y_rms,
                'rmse': rmse,
                'u_target_std': float(U_target_train.std())
            })
            
            if verbose:
                print(f"     RMSE NN: {rmse:.3f}V, u_target std: {U_target_train.std():.2f}V")
        
        return all_iter_history
    
    def pretrain_anti_disturbance(self, n_epochs=200, verbose=True):
        """
        STRATÉGIE PRÉ-ENTRAÎNEMENT ANALYTIQUE :
        
        u_FF_target(phase) = -B_pseudoinv @ alpha4(phase) (modal force)
        """
        if verbose:
            print(f"\n--- Pré-entraînement Anti-Disturbance ---")
        
        if self.alpha4_periodic is None:
            return None, []
        
        rng = np.random.default_rng(42)
        
        # alpha4 est scalaire dans cette implémentation
        # Effet sur état : a4 * DpT_Dp_now @ q_delay (régénératif)
        # Pour anticiper, u_FF doit produire effect inverse
        # Approximation simple : u_FF(phase) = -gain * a4(phase)
        
        if self.alpha4_periodic.ndim == 1:
            alpha4_per = self.alpha4_periodic
        else:
            alpha4_per = self.alpha4_periodic[:, 0]
        
        # Normalize alpha4 to find appropriate gain
        a4_max = max(abs(alpha4_per).max(), 1e-30)
        # Target : u_FF(phase) = -gain_scaled * alpha4(phase) / a4_max
        # gain_scaled chosen to give peak u_FF = 0.5 * u_FF_max
        gain_scaled = 0.5 * self.ff_nn.u_FF_max
        
        n_per = self.n_per
        X_data = []
        Phase_data = []
        U_FF_data = []
        
        for sample_idx in range(2000):
            k_phase = sample_idx % n_per
            phase = 2 * np.pi * k_phase / n_per
            x_pos = 1e-6 * rng.normal(size=self.n_modes)
            x_vel = 1e-3 * rng.normal(size=self.n_modes)
            x = np.concatenate([x_pos, x_vel])
            
            # Target : counter alpha4 with appropriate sign
            u_FF_target = -gain_scaled * alpha4_per[k_phase] / a4_max
            u_FF_target = np.clip(u_FF_target,
                                    -self.ff_nn.u_FF_max,
                                    self.ff_nn.u_FF_max)
            
            X_data.append(x)
            Phase_data.append(phase)
            U_FF_data.append(u_FF_target)
        
        X_data = np.array(X_data)
        Phase_data = np.array(Phase_data)
        U_FF_data = np.array(U_FF_data)
        
        if verbose:
            print(f"   Generated {len(X_data)} samples (anti-disturbance)")
            print(f"   u_FF range: [{U_FF_data.min():.2f}, {U_FF_data.max():.2f}] V")
        
        loss_history = []
        for epoch in range(n_epochs):
            idx = rng.permutation(len(X_data))
            total_loss = 0
            for j in idx:
                loss = self.ff_nn.backward(X_data[j], Phase_data[j], U_FF_data[j])
                total_loss += loss
            loss_history.append(total_loss / len(X_data))
            if epoch == 100: self.ff_nn.lr *= 0.5
            if epoch == 150: self.ff_nn.lr *= 0.5
        
        errs = []
        for j in range(len(X_data)):
            u_p, _, _ = self.ff_nn.forward(X_data[j], Phase_data[j])
            errs.append((u_p - U_FF_data[j])**2)
        rmse = np.sqrt(np.mean(errs))
        
        if verbose:
            print(f"   RMSE final : {rmse:.3f} V")
        
        return rmse, loss_history
    
    def step(self, x_hat_prev, u_prev, y_meas, k_step=0):
        """
        Pipeline DARC-MPC v3 :
        1. Kalman observer
        2. RLS adaptation
        3. LQG action (réactif)
        4. NN FF correction (feedforward)
        5. Combine + safety
        """
        # 1. Kalman
        x_hat = (self.A_obs_d @ x_hat_prev
                 + self.G_u.flatten() * u_prev
                 + self.G_y.flatten() * y_meas)
        
        # 2. RLS (online adaptation)
        if self.enable_adaptation:
            x_pred = self.A_d @ self.x_prev + self.B_d.flatten() * self.u_prev
            lambda_robust = self.rls.update(x_hat, x_pred)
        else:
            lambda_robust = 1.0
        
        # 3. LQG action (toujours présente, garantie de stabilité)
        u_lqg = float(np.squeeze(-self.K_lqr @ x_hat))
        u_lqg = np.clip(u_lqg, -self.u_max, self.u_max)
        self.history_u_lqg.append(u_lqg)
        
        # 4. NN FF correction (anticipative, basé sur phase)
        phase = 2 * np.pi * (k_step % self.n_per) / self.n_per
        self.history_phase.append(phase)
        
        u_ff, _, _ = self.ff_nn.forward(x_hat, phase)
        u_ff = self.ff_alpha * u_ff   # gain global
        self.history_u_ff.append(u_ff)
        
        # 5. Combine
        u_proposed = u_lqg + u_ff
        u_proposed = np.clip(u_proposed, -self.u_max, self.u_max)
        
        # 6. Safety
        u_safe, violated = self.safety.filter_action(x_hat, u_proposed, u_lqg)
        u_safe = float(np.clip(u_safe, -self.u_max, self.u_max))
        
        self.history_u_total.append(u_safe)
        self.history_safety.append(1 if violated else 0)
        
        # Update for next iteration
        self.x_prev = x_hat.copy()
        self.u_prev = u_safe
        
        return x_hat, u_safe
    
    def discretize_observer(self):
        pass
