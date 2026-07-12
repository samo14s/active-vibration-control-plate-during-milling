"""
darc_controller.py
==================
DARC : Deep Anticipative Residual Control

Loi de commande à trois couches ADDITIVES :

    u(t) = u_LQG(x̂)  +  u_FF(φ)  +  u_NN(x̂, φ)

  1. BASE LQG        : identique octet pour octet au baseline LQGController
                       (mêmes A, B, C, mêmes poids LQR, même Kalman, même ZOH)
                       → l'apport des couches 2-3 est mesuré équitablement.
  2. FEEDFORWARD     : table périodique u_FF indexée par la phase de broche
     ANTICIPATIF       (encodeur), conçue par MODÈLE INVERSE de la boucle
                       fermée aux harmoniques de passage de dent
                       (design_periodic_feedforward).  Conçu à partir du
                       modèle NOMINAL de coupe (K_T nominal) : le contrôleur
                       ne reçoit JAMAIS les coefficients réels perturbés.
  3. RÉSIDU NEURONAL : petit perceptron (16 neurones) entraîné par
     (« Deep »)        apprentissage itératif (ILC) à absorber le résidu que
                       le modèle inverse linéaire laisse.  PROTOCOLE PROPRE :
                       épisodes d'entraînement et de VALIDATION avec des
                       réalisations de bruit distinctes ; la sélection du
                       meilleur checkpoint se fait sur l'épisode de
                       VALIDATION, jamais sur l'épisode d'évaluation.

HONNÊTETÉ DE LA MÉTHODE (vs l'ancien « DARC-MPC v3 ») :
  - Il n'y a PAS de MPC ici (aucun horizon de prédiction, aucune optimisation
    en ligne, aucun QP sous contraintes) : l'ancien nom « DARC-MPC »
    représentait mal la méthode et a été abandonné.
  - L'ancien estimateur « RLS adaptatif » (sortie jamais utilisée) et le
    filtre de sécurité de Lyapunov (court-circuité dans toutes les
    expériences rapportées) étaient du code mort : supprimés.
  - Le feedforward suppose une synchronisation de phase broche PARFAITE
    (indexation modulo n_per).  La sensibilité à une erreur de phase réelle
    doit être étudiée séparément (le FF mal phasé peut INJECTER de l'énergie).
"""

import numpy as np
from scipy.linalg import solve_continuous_are, expm


# ====================================================================
# Réseau feedforward « phase-aware » (petit, résiduel)
# ====================================================================
class FeedforwardCorrectorNN:
    """
    NN qui prédit une correction u_NN à partir de :
       - la phase de passage de dent (cos φ, sin φ)
       - l'état estimé x̂ (poids faible — surtout la phase)

    Sortie : u_NN bornée (±u_FF_max), ajoutée à u_LQG + u_FF.
    """
    def __init__(self, n_x, n_hidden=16, lr=0.01, seed=42, u_FF_max=30.0):
        self.n_x = n_x
        self.n_input = n_x + 2  # état + cos/sin phase
        self.n_hidden = n_hidden
        self.lr = lr
        self.u_FF_max = u_FF_max
        self.n_modes = n_x // 2

        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (n_hidden, self.n_input))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.1, (1, n_hidden))
        self.b2 = np.zeros(1)

        # Mise à l'échelle de l'état
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
# DARC : LQG + feedforward inverse-modèle + résidu neuronal
# ====================================================================
class DARCController:
    """
    DARC — Deep Anticipative Residual Control.

    u(t) = u_LQG(x̂) + ff_alpha · [ u_FF(φ) + u_NN(x̂, φ) ]

    Le NN ne REMPLACE pas le LQG : il ajoute une correction résiduelle
    par-dessus le feedforward modèle-inverse.
    """
    def __init__(self, plate, dt,
                 base_w_q=1e14, base_w_qd=1e8, base_w_r=1.0,
                 # NN
                 ff_lr=0.01, ff_max=30.0, ff_alpha=1.0,
                 # Période de passage de dent (en pas)
                 n_per=82,
                 # Saturation de commande
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

        # Base LQG (identique au baseline)
        self._build_base(base_w_q, base_w_qd, base_w_r)

        # Réseau résiduel
        self.ff_nn = FeedforwardCorrectorNN(self.n_x, n_hidden=16,
                                            lr=ff_lr, u_FF_max=ff_max)

        # Feedforward périodique (modèle inverse) — design_periodic_feedforward()
        self.u_ff_periodic = None
        # Correction résiduelle NN active seulement après entraînement
        self.use_nn_residual = False

        # Historiques (diagnostics)
        self.history_u_lqg = []
        self.history_u_ff = []
        self.history_u_total = []
        self.history_phase = []

        if verbose:
            self._print_summary()

    # ----------------------------------------------------------------
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

        # Discrétisation
        self.A_d = expm(A * self.dt)
        try:
            self.B_d = np.linalg.solve(A, (self.A_d - np.eye(self.n_x)) @ B)
        except np.linalg.LinAlgError:
            self.B_d = B * self.dt

        # Kalman (mêmes covariances que le baseline LQGController)
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

    # ----------------------------------------------------------------
    def _print_summary(self):
        print(f"\n{'='*70}")
        print(f"  DARC : Deep Anticipative Residual Control")
        print(f"{'='*70}")
        print(f"  Loi        : u = u_LQG(x̂) + u_FF(φ) + u_NN(x̂, φ)")
        print(f"  NN résiduel: {self.ff_nn.n_hidden} neurones cachés, "
              f"lr={self.ff_nn.lr}")
        print(f"  α (gain FF): {self.ff_alpha}")
        print(f"  u_NN max   : ±{self.ff_nn.u_FF_max}V")
        print(f"  ||K_LQR||  : {np.linalg.norm(self.K_lqr):.2e}")
        print(f"  (pas de MPC, pas d'adaptation en ligne : voir docstring)")

    # ----------------------------------------------------------------
    def design_periodic_feedforward(self, ft, a3_period, Dp,
                                    scale=1.0, n_harm=24, ff_max=None):
        """
        Conçoit le FEEDFORWARD PÉRIODIQUE (modèle inverse) qui annule la
        réponse de la plaque à l'excitation périodique de passage de dent.

        Pour chaque harmonique h de la fréquence de passage de dent :
            U_FF(ω_h) = - G_wy(ω_h) · W(ω_h) / G_uy(ω_h)
        avec  W = FFT(f_t · a3)  (excitation périodique modale NOMINALE),
              G_wy(ω) = C (jωI - A_cl)^{-1} B_d  (perturbation -> sortie, BF),
              G_uy(ω) = C (jωI - A_cl)^{-1} B     (commande   -> sortie, BF),
              A_cl = A - B·K_lqr  (boucle fermée LQG NOMINALE).

        IMPORTANT (protocole honnête) : ``a3_period`` doit être la séquence
        NOMINALE (K_T nominal, engagement commandé) — le modèle de coupe dont
        un contrôleur réel disposerait — PAS la séquence réelle du procédé
        simulé.  L'écart modèle/réalité (p.ex. K_T +30 %) doit dégrader le FF,
        c'est précisément ce qu'un scénario de robustesse mesure.

        Hypothèse : synchronisation de phase broche parfaite (indexation
        modulo n_per).  `scale` (<1) permet de doser l'annulation.
        """
        n = self.n_modes
        A_cl = self.A - self.B @ self.K_lqr
        B_d = np.zeros((self.n_x, 1));  B_d[n:, 0] = np.asarray(Dp)   # M = I
        Bu = self.B
        C = self.C
        n_per = self.n_per
        w = ft * np.asarray(a3_period[:n_per], dtype=float)
        W = np.fft.rfft(w)
        Om = 2 * np.pi / (n_per * self.dt)                # pulsation fondamentale
        Im = np.eye(self.n_x)
        U = np.zeros(len(W), dtype=complex)
        for h in range(1, min(n_harm, len(W) - 1) + 1):
            Mh = (1j * h * Om) * Im - A_cl
            Gwy = complex((C @ np.linalg.solve(Mh, B_d))[0, 0])
            Guy = complex((C @ np.linalg.solve(Mh, Bu))[0, 0])
            if abs(Guy) > 1e-30:
                U[h] = -Gwy * W[h] / Guy
        U *= scale
        u_ff = np.fft.irfft(U, n=n_per)
        if ff_max is not None:
            u_ff = np.clip(u_ff, -ff_max, ff_max)
        self.u_ff_periodic = u_ff
        if self.verbose:
            print(f"[DARC] feedforward périodique conçu : "
                  f"|u_ff|max = {np.max(np.abs(u_ff)):.2f} V, "
                  f"RMS = {np.sqrt(np.mean(u_ff**2)):.2f} V")
        return u_ff

    # ----------------------------------------------------------------
    def train_nn_residual(self, simulator, alpha3_t, alpha4_t, kp_idx,
                          alpha4_2_t=None, alpha4_3_t=None,
                          n_iter=20, eta=0.4, K_corr=2e6, n_epochs=12,
                          sensor_noise=0.0, sensor_floor=0.0,
                          train_seed=100, val_seed=200,
                          verbose=False):
        """
        Apprentissage itératif (ILC) du résidu par le NN, PAR-DESSUS le
        feedforward modèle-inverse — PROTOCOLE TRAIN/VALIDATION PROPRE.

        - ``simulator`` et les séquences alpha* doivent décrire le MONDE
          NOMINAL du contrôleur (plaque nominale, K_T nominal) : le NN
          s'entraîne dans le modèle dont le contrôleur dispose, jamais sur
          le procédé réel perturbé de l'évaluation.
        - Chaque itération s'entraîne sur un épisode avec une réalisation de
          bruit capteur DIFFÉRENTE (graine train_seed + it).
        - La sélection du meilleur checkpoint se fait sur un épisode de
          VALIDATION séparé (graine val_seed, fixe), JAMAIS sur l'épisode
          d'évaluation : aucune fuite du test vers la sélection de modèle.
        - Le RMS rapporté par cette fonction est un RMS de VALIDATION ; la
          performance d'évaluation est mesurée ensuite, indépendamment.
        """
        import copy
        if self.u_ff_periodic is None:
            raise RuntimeError("Concevez d'abord le feedforward inverse "
                               "(design_periodic_feedforward).")
        self.use_nn_residual = True
        B_flat = self.B.flatten()
        g_sign = np.sign(self.plate.D_obs[0] * B_flat[self.n_modes]) or 1.0
        rng = np.random.default_rng(0)

        def _episode(seed):
            for h in ('history_u_lqg', 'history_u_ff', 'history_u_total',
                      'history_phase'):
                setattr(self, h, [])
            res = simulator.simulate(alpha3_t, alpha4_t, kp_idx, controller=self,
                                     alpha4_2_t=alpha4_2_t, alpha4_3_t=alpha4_3_t,
                                     sensor_noise=sensor_noise,
                                     sensor_floor=sensor_floor,
                                     sensor_rng=np.random.default_rng(seed),
                                     progress=False)
            i_end = res['stop_idx']
            n = min(i_end + 1, res['y'].shape[0], res['x_hat'].shape[1])
            y = res['y'][:n]; xh = res['x_hat'][:, :n]
            return float(np.sqrt(np.mean(y**2))), y, xh, n

        # Référence : NN initial, RMS sur l'épisode de VALIDATION
        best_val_rms, _, _, _ = _episode(val_seed)
        best_W = copy.deepcopy((self.ff_nn.W1, self.ff_nn.b1,
                                self.ff_nn.W2, self.ff_nn.b2))
        hist = [best_val_rms]
        for it in range(n_iter):
            # --- épisode d'ENTRAÎNEMENT (réalisation de bruit propre) ---
            train_rms, y, xh, n = _episode(train_seed + it)
            # cibles : sortie NN courante + correction proportionnelle à -y
            X, Ph, T = [], [], []
            for k in range(1, n):
                ph = 2 * np.pi * (k % self.n_per) / self.n_per
                x = xh[:, k]
                nn_out, _, _ = self.ff_nn.forward(x, ph)
                tgt = np.clip(nn_out - eta * K_corr * g_sign * y[k],
                              -self.ff_nn.u_FF_max, self.ff_nn.u_FF_max)
                X.append(x); Ph.append(ph); T.append(tgt)
            X = np.array(X); Ph = np.array(Ph); T = np.array(T)
            for ep in range(n_epochs):
                idx = rng.permutation(len(X))
                for j in idx[:min(len(idx), 800)]:
                    self.ff_nn.backward(X[j], Ph[j], T[j])
            # --- épisode de VALIDATION (sélection de checkpoint) ---
            val_rms, _, _, _ = _episode(val_seed)
            hist.append(val_rms)
            if val_rms < best_val_rms:
                best_val_rms = val_rms
                best_W = copy.deepcopy((self.ff_nn.W1, self.ff_nn.b1,
                                        self.ff_nn.W2, self.ff_nn.b2))
            if verbose:
                print(f"[DARC-NN] iter {it+1}/{n_iter}  "
                      f"train={train_rms*1e6:.4f}  val={val_rms*1e6:.4f} um "
                      f"(best val {best_val_rms*1e6:.4f})", flush=True)
        # restaurer le meilleur réseau (au sens de la VALIDATION)
        self.ff_nn.W1, self.ff_nn.b1, self.ff_nn.W2, self.ff_nn.b2 = best_W
        if verbose:
            print(f"[DARC-NN] meilleur résidu (validation) : "
                  f"y_rms={best_val_rms*1e6:.4f} um", flush=True)
        return hist

    # ----------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas, k_step=0):
        """
        Pipeline DARC :
        1. Observateur de Kalman
        2. Retour LQG (réactif)
        3. Feedforward périodique (table indexée par la phase) + résidu NN
        4. Somme + saturation
        """
        # 1. Kalman
        x_hat = (self.A_obs_d @ x_hat_prev
                 + self.G_u.flatten() * u_prev
                 + self.G_y.flatten() * y_meas)

        # 2. Retour LQG
        u_lqg = float(np.squeeze(-self.K_lqr @ x_hat))
        u_lqg = np.clip(u_lqg, -self.u_max, self.u_max)
        self.history_u_lqg.append(u_lqg)

        # 3. Feedforward anticipatif (phase de passage de dent) + NN résiduel
        phase = 2 * np.pi * (k_step % self.n_per) / self.n_per
        self.history_phase.append(phase)

        u_ff_base = 0.0
        if self.u_ff_periodic is not None:
            u_ff_base += float(self.u_ff_periodic[k_step % self.n_per])
        if self.use_nn_residual:
            u_nn, _, _ = self.ff_nn.forward(x_hat, phase)
            u_ff_base += u_nn
        u_ff = self.ff_alpha * u_ff_base
        self.history_u_ff.append(u_ff)

        # 4. Somme + saturation
        u_total = float(np.clip(u_lqg + u_ff, -self.u_max, self.u_max))
        self.history_u_total.append(u_total)

        return x_hat, u_total
