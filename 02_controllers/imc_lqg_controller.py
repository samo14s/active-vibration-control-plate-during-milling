"""
imc_lqg_controller.py
=====================
LQG à MODÈLE INTERNE (disturbance-accommodating LQG) :
baseline feedback pour la perturbation périodique de passage de dent.

Motivation (voir docs/AUDIT_SCIENTIFIQUE.md, constat A6) : l'affirmation
« un retour ne peut pas rejeter une perturbation périodique » est fausse dès
que la PÉRIODE est connue — et l'encodeur broche est déjà supposé disponible
pour le feedforward du DARC.  Par le principe du modèle interne
(Francis & Wonham 1976 ; Johnson 1971 ; commande répétitive, Hara 1988), un
observateur augmenté d'états d'oscillateurs aux harmoniques de passage de
dent ESTIME la perturbation en ligne, et le retour l'annule asymptotiquement.

Structure :
   - Perturbation modélisée  w(t) = Σ_h s_h(t),  ḋ_h = [[0,ω_h],[-ω_h,0]] d_h
     (d_h = [s_h ; c_h]), entrant par B_d = [0 ; Dp] (direction outil).
   - Filtre de KALMAN AUGMENTÉ sur [x ; d_1 ; … ; d_H]  →  estime x̂ ET d̂_h.
   - Commande  u = -K_lqr x̂  +  Σ_h Re{ γ̄_h · (ŝ_h + j ĉ_h) }
     où γ_h = -G_wy(jω_h)/G_uy(jω_h) est la solution EXACTE des équations du
     régulateur (Davison/Francis) : mêmes gains d'inversion que le
     feedforward du DARC, mais pilotés par l'ESTIMATION EN LIGNE au lieu
     d'une table pré-calculée sur le modèle de coupe nominal.

Conséquences attendues (et vérifiées par main_imc_baseline.py) :
   - contrairement au feedforward pré-calculé, ce baseline n'a besoin
     d'AUCUN modèle de coupe (pas de K_T, pas de a3(t)) — seulement de la
     période de dent (encodeur) ;
   - il s'adapte donc en ligne à K_T inconnu (scénario S4) ;
   - en revanche il repose sur le CAPTEUR de déplacement (comme le LQG).

Ce contrôleur N'UTILISE PAS le même vecteur d'état externe que LQG/DARC
(état augmenté 2n+2H) : il maintient son état en interne — appeler reset()
avant chaque simulation.
"""
import numpy as np
from scipy.linalg import solve_continuous_are, expm


class IMCLQGController:
    """
    LQG + modèle interne harmonique (disturbance-accommodating LQG).

    u = -K_lqr x̂ + Σ_h Re{ γ̄_h ẑ_h },  ẑ_h = ŝ_h + j ĉ_h estimés par le
    Kalman augmenté.  Interface step(x_hat_prev, u_prev, y_meas) compatible
    NewmarkSimulator (l'état passé en argument est ignoré ; l'état augmenté
    est interne -> reset() obligatoire avant chaque run).
    """

    def __init__(self, plate, dt, f_fund, Dp_dist,
                 n_harm=8,
                 w_q=1e14, w_qd=1e8, w_r=1.0,
                 W_x=1e-6, q_dist=1e0, V_meas=1e-12,
                 u_max=150.0, verbose=True):
        """
        f_fund  : fréquence FONDAMENTALE de passage de dent réellement
                  simulée [Hz] (1/(n_per·dt) si la période est quantifiée).
                  C'est la SEULE information « procédé » requise (encodeur).
        Dp_dist : direction modale de la perturbation (forme propre au point
                  outil, nominale) — direction de B_d.
        n_harm  : nombre d'harmoniques modélisées par le modèle interne.
        q_dist  : densité de bruit de process des états de perturbation
                  (agilité de la poursuite ; plus grand = poursuite rapide,
                  plus sensible au bruit).
        """
        self.plate = plate
        self.dt = dt
        self.n_modes = plate.n_modes
        self.n_x = 2 * plate.n_modes
        self.n_harm = n_harm
        self.u_max = u_max
        self.verbose = verbose

        n = self.n_modes
        # --- plante ---
        A = np.zeros((self.n_x, self.n_x))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.linalg.solve(plate.Mp, plate.Kp)
        A[n:, n:] = -np.linalg.solve(plate.Mp, plate.Cp)
        B = np.zeros((self.n_x, 1))
        B[n:, 0] = np.linalg.solve(plate.Mp, plate.H_Pe_modal)
        C = np.zeros((1, self.n_x))
        C[0, :n] = plate.D_obs
        self.A, self.B, self.C = A, B, C

        Bd = np.zeros(self.n_x)
        Bd[n:] = np.asarray(Dp_dist)
        self.Bd = Bd

        # --- LQR (même base que LQGController / DARC) ---
        M_yp = np.outer(plate.D_obs, plate.D_obs)
        Q_top = w_q * M_yp + 1e-3 * np.eye(n)
        Q_bot = w_qd * np.eye(n) + 1e-3 * np.eye(n)
        Q = np.block([[Q_top, np.zeros((n, n))],
                      [np.zeros((n, n)), Q_bot]])
        R = np.array([[w_r]])
        P = solve_continuous_are(A, B, Q, R)
        self.K_lqr = np.linalg.solve(R, B.T @ P)

        # --- modèle interne : oscillateurs aux harmoniques ---
        self.omega_h = 2 * np.pi * f_fund * np.arange(1, n_harm + 1)
        n_d = 2 * n_harm
        n_a = self.n_x + n_d
        A_aug = np.zeros((n_a, n_a))
        A_aug[:self.n_x, :self.n_x] = A
        for h in range(n_harm):
            i0 = self.n_x + 2 * h
            w_h = self.omega_h[h]
            A_aug[i0, i0 + 1] = w_h
            A_aug[i0 + 1, i0] = -w_h
            # w(t) = Σ s_h -> la composante s_h pousse la plante par Bd
            A_aug[:self.n_x, i0] = Bd
        B_aug = np.vstack([B, np.zeros((n_d, 1))])
        C_aug = np.hstack([C, np.zeros((1, n_d))])
        self.n_a = n_a
        self.A_aug, self.B_aug, self.C_aug = A_aug, B_aug, C_aug

        # --- Kalman augmenté ---
        W_aug = np.zeros((n_a, n_a))
        W_aug[:self.n_x, :self.n_x] = W_x * np.eye(self.n_x)
        W_aug[self.n_x:, self.n_x:] = q_dist * np.eye(n_d)
        V = np.array([[V_meas]])
        P_kal = solve_continuous_are(A_aug.T, C_aug.T, W_aug, V)
        self.L_kal = P_kal @ C_aug.T @ np.linalg.inv(V)

        # --- gains d'annulation (équations du régulateur, exactes) ---
        #   γ_h = -G_wy(jω_h)/G_uy(jω_h),  u_d = Σ Re{γ̄_h (ŝ_h + j ĉ_h)}
        A_cl = A - B @ self.K_lqr
        Im = np.eye(self.n_x)
        self.gamma_h = np.zeros(n_harm, dtype=complex)
        Bd_col = Bd.reshape(-1, 1)
        for h in range(n_harm):
            Mh = (1j * self.omega_h[h]) * Im - A_cl
            Gwy = complex((C @ np.linalg.solve(Mh, Bd_col))[0, 0])
            Guy = complex((C @ np.linalg.solve(Mh, B))[0, 0])
            if abs(Guy) > 1e-30:
                self.gamma_h[h] = -Gwy / Guy

        # --- discrétisation ZOH de l'observateur augmenté ---
        A_obs = A_aug - self.L_kal @ C_aug
        self.A_obs_d = expm(A_obs * dt)
        try:
            self.G_u = np.linalg.solve(A_obs, (self.A_obs_d - np.eye(n_a)) @ B_aug)
            self.G_y = np.linalg.solve(A_obs, (self.A_obs_d - np.eye(n_a)) @ self.L_kal)
        except np.linalg.LinAlgError:
            self.G_u = B_aug * dt
            self.G_y = self.L_kal * dt

        self.reset()

        if verbose:
            print(f"[IMC-LQG] modèle interne : {n_harm} harmoniques "
                  f"({f_fund:.1f} ... {f_fund*n_harm:.0f} Hz), "
                  f"état augmenté {n_a}, ||K||={np.linalg.norm(self.K_lqr):.2e}")

    # ----------------------------------------------------------------
    def reset(self):
        """État augmenté interne — À APPELER avant chaque simulation."""
        self.x_aug = np.zeros(self.n_a)

    # ----------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas):
        """
        Une itération : Kalman augmenté + retour LQR + annulation de la
        perturbation estimée.  x_hat_prev (état externe du solveur) est
        IGNORÉ — l'état augmenté est interne (voir reset()).
        Retourne (x̂_plante, u) pour compatibilité solveur.
        """
        self.x_aug = (self.A_obs_d @ self.x_aug
                      + self.G_u.flatten() * u_prev
                      + self.G_y.flatten() * y_meas)
        x_hat = self.x_aug[:self.n_x]

        u = float(np.squeeze(-self.K_lqr @ x_hat))
        for h in range(self.n_harm):
            i0 = self.n_x + 2 * h
            z_h = complex(self.x_aug[i0], self.x_aug[i0 + 1])
            u += float(np.real(np.conj(self.gamma_h[h]) * z_h))
        u = float(np.clip(u, -self.u_max, self.u_max))
        return x_hat, u
