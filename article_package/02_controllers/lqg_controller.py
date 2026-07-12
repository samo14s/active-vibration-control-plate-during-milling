"""
lqg_controller.py
=================
Conception et optimisation d'un contrôleur LQG :
   - Filtre de Kalman pour estimer x_hat à partir de y_obs scalaire
   - LQR pour calculer u = -K * x_hat
   - Optimisation des poids Q, R par recherche sur grille
"""
import numpy as np
from scipy.linalg import solve_continuous_are, expm


class LQGController:
    """
    LQG controller pour le système :
        x = [q; q_dot]    de dimension 2*n_modes
        ẋ = A x + B u + w
        y = C x + v
    """

    def __init__(self, plate, dt: float, verbose: bool = True):
        """
        plate : instance de PlateModel (avec D_obs et H_Pe_modal calculés).
        """
        self.plate = plate
        self.dt = dt
        self.verbose = verbose
        self.n_modes = plate.n_modes
        self.n_x = 2 * plate.n_modes

        self._build_state_space()
        self._design_kalman()

    # ---------------------------------------------------------------
    def _build_state_space(self):
        n = self.n_modes
        A = np.zeros((2*n, 2*n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.linalg.solve(self.plate.Mp, self.plate.Kp)
        A[n:, n:] = -np.linalg.solve(self.plate.Mp, self.plate.Cp)

        B = np.zeros((2*n, 1))
        B[n:, 0] = np.linalg.solve(self.plate.Mp, self.plate.H_Pe_modal)

        C = np.zeros((1, 2*n))
        C[0, :n] = self.plate.D_obs

        self.A = A
        self.B = B
        self.C = C

        if self.verbose:
            ev = np.linalg.eigvals(A)
            print(f"[LQG] Pôles boucle ouverte ({2*n} états) :")
            for e in ev:
                zeta_pct = -np.real(e)/np.abs(e)*100 if np.abs(e) > 0 else 0
                print(f"   {np.real(e):+8.2f} {np.imag(e):+8.1f}j   "
                      f"(zeta = {zeta_pct:5.2f}%)")

    # ---------------------------------------------------------------
    def _design_kalman(self,
                        W_kal: float = 1e-6,
                        V_kal: float = 1e-12):
        """Filtre de Kalman avec scipy.linalg.solve_continuous_are."""
        W = W_kal * np.eye(self.n_x)
        V = np.array([[V_kal]])

        # Riccati pour Kalman :
        #   A^T P + P A^T - P C^T V^-1 C P + W = 0  =>  L = P C^T V^-1
        # mais solve_continuous_are utilise (A, B, Q, R) :
        #   A^T P + P A - P B R^-1 B^T P + Q = 0
        # On résout le problème dual : (A=A^T, B=C^T, Q=W, R=V).
        P = solve_continuous_are(self.A.T, self.C.T, W, V)
        L = P @ self.C.T @ np.linalg.inv(V)
        self.L_kal = L

        if self.verbose:
            print(f"[LQG] ||L_kal|| = {np.linalg.norm(L):.3e}")

    # ---------------------------------------------------------------
    def optimize_weights(self,
                          w_q_list=None, w_qd_list=None, w_r: float = 1.0,
                          gain_norm_max: float = 1e8):
        """
        Recherche sur grille des meilleurs poids LQR. Critère :
        maximiser |min(Re(λ_CL))| sous contrainte ||K_lqr|| < gain_norm_max.

        Q = blkdiag( w_q * D_obs.T D_obs , w_qd * I )
        R = w_r
        """
        if w_q_list  is None: w_q_list  = [1e10, 1e12, 1e14, 1e16]
        if w_qd_list is None: w_qd_list = [1e4, 1e6, 1e8]

        if self.verbose:
            print(f"[LQG] Optimisation grille "
                  f"({len(w_q_list)} x {len(w_qd_list)})...")

        n = self.n_modes
        n_x = self.n_x
        M_yp = np.outer(self.plate.D_obs, self.plate.D_obs)

        best_min_re = 0.0
        best = None

        for w_q in w_q_list:
            for w_qd in w_qd_list:
                Q_top    = w_q  * M_yp + 1e-3 * np.eye(n)
                Q_bottom = w_qd * np.eye(n) + 1e-3 * np.eye(n)
                Q = np.block([[Q_top,            np.zeros((n, n))],
                              [np.zeros((n, n)), Q_bottom]])
                R = np.array([[w_r]])

                try:
                    P = solve_continuous_are(self.A, self.B, Q, R)
                    K = np.linalg.solve(R, self.B.T @ P)
                    ev_cl = np.linalg.eigvals(self.A - self.B @ K)
                    min_re = float(np.min(np.real(ev_cl)))

                    if (min_re < 0
                        and np.linalg.norm(K) < gain_norm_max
                        and min_re < best_min_re):
                        best_min_re = min_re
                        best = (K, w_q, w_qd, ev_cl)
                except Exception:
                    pass

        if best is None:
            raise RuntimeError("Aucune combinaison Q-R n'a donné un système stable.")

        K, w_q_opt, w_qd_opt, ev_cl = best
        self.K_lqr = K
        self.w_q = w_q_opt
        self.w_qd = w_qd_opt
        self.w_r = w_r
        self.ev_cl = ev_cl
        self.min_re_cl = best_min_re

        if self.verbose:
            print(f"[LQG] Poids optimaux : w_q = {w_q_opt:.1e}, "
                  f"w_qd = {w_qd_opt:.1e}, w_r = {w_r:.1f}")
            print(f"[LQG] Pôles boucle fermée :")
            for e in ev_cl:
                zeta_pct = -np.real(e)/np.abs(e)*100 if np.abs(e) > 0 else 0
                print(f"   {np.real(e):+8.2f} {np.imag(e):+8.1f}j   "
                      f"(zeta = {zeta_pct:5.2f}%)")

    # ---------------------------------------------------------------
    def discretize_observer(self):
        """
        Pré-calcul des matrices ZOH pour l'observateur :
            x_hat[k] = A_obs_d @ x_hat[k-1] + G_u u[k-1] + G_y y[k]
        """
        A_obs = self.A - self.L_kal @ self.C
        A_obs_d = expm(A_obs * self.dt)

        # Calcul stable même si A_obs est mal conditionnée
        try:
            G_u = np.linalg.solve(A_obs, (A_obs_d - np.eye(self.n_x)) @ self.B)
            G_y = np.linalg.solve(A_obs, (A_obs_d - np.eye(self.n_x)) @ self.L_kal)
        except np.linalg.LinAlgError:
            # Fallback Euler
            A_obs_d = np.eye(self.n_x) + A_obs * self.dt
            G_u = self.B * self.dt
            G_y = self.L_kal * self.dt

        self.A_obs_d = A_obs_d
        self.G_u = G_u
        self.G_y = G_y

        # Matrices ZOH du MODELE pour la compensation d'un retard de
        # mesure CONNU (latence capteur datasheet) : avant le calcul de
        # la commande, l'estimee est propagee de `delay_comp_steps` pas
        # pour que u agisse sur l'etat COURANT et non l'etat retarde.
        Ad = expm(self.A * self.dt)
        try:
            Bd = np.linalg.solve(self.A, (Ad - np.eye(self.n_x)) @ self.B)
        except np.linalg.LinAlgError:
            Bd = self.B * self.dt
        self.Ad_plant = Ad
        self.Bd_plant = Bd
        if not hasattr(self, "delay_comp_steps"):
            self.delay_comp_steps = 0

    # ---------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas):
        """Une itération de l'observateur Kalman + commande LQR.

        Si ``delay_comp_steps`` > 0 (retard capteur connu, en pas dt),
        l'estimée est propagée en avant par le modèle avant le calcul de
        la commande (compensation prédictive standard)."""
        x_hat = self.A_obs_d @ x_hat_prev + self.G_u.flatten() * u_prev \
              + self.G_y.flatten() * y_meas
        n_cmp = getattr(self, "delay_comp_steps", 0)
        if n_cmp > 0:
            x_fwd = x_hat
            for _ in range(n_cmp):
                x_fwd = (self.Ad_plant @ x_fwd
                         + self.Bd_plant.flatten() * u_prev)
            u = float(np.squeeze(-self.K_lqr @ x_fwd))
        else:
            u = float(np.squeeze(-self.K_lqr @ x_hat))
        return x_hat, u
