"""
smc_dob_controller.py
=====================
STSMC+DOB : super-twisting sliding-mode augmentation of the LQG baseline, with
the super-twisting integral acting as a built-in disturbance observer.

⚠ RÔLE DANS L'ÉTUDE — LIRE AVANT USAGE (voir docs/AUDIT_SCIENTIFIQUE.md §8).
Ce contrôleur est le « COIN ROBUSTE » de la carte d'architectures : il vise la
robustesse (temps fini, sans modèle de coupe, garantie de stabilité) plutôt que
la performance de pointe.  Le RÉSULTAT MESURÉ, honnête et documenté, est que sur
CETTE plaque très faiblement amortie (ζ = 0.17–0.31 %), l'augmentation
super-twisting/DOB à large bande **RÉCUPÈRE la performance du LQG mais ne la
dépasse pas** — elle ne réduit pas la vibration forcée sous le niveau LQG et
n'étend pas la frontière de stabilité.  Ce n'est pas un défaut de réglage mais
un fait STRUCTUREL (voir la « conclusion fondamentale » ci-dessous), et c'est
la contribution scientifique de ce volet.

CONCLUSION FONDAMENTALE (le vrai apport) :
  Sur une pièce faiblement amortie, la 2e harmonique de passage de dent
  (2·245 = 490 Hz) tombe SUR le mode 1 (521 Hz).  Il en résulte un CONFLIT
  structurel entre :
    (a) l'estimation/annulation ROBUSTE EN LIGNE de la perturbation (DOB, AFC,
        modèle interne adaptatif) — instable en phase dans la région résonante ;
    (b) l'annulation SÉLECTIVE en fréquence qui, seule, bat le LQG (IMC/DARC).
  Un contrôle robuste à large bande (SMC/DOB) ne peut pas rejeter sélectivement
  la perturbation périodique quasi-résonante : il récupère l'optimum du retour.
  L'annulation sélective (IMC/DARC) exige un MODÈLE (hors ligne) et paie sa
  fragilité.  C'est la forme rigoureuse (quasi-impossibilité) de la
  « fragilité directionnelle » révélée par l'étude d'architectures.

Ce que ce contrôleur APPORTE néanmoins, honnêtement :
  - garantie de STABILITÉ (jamais de divergence comme IMC-LQG en S3) ;
  - SANS modèle de coupe (le DOB super-twisting est model-free sur K_T) ;
  - convergence en TEMPS FINI (super-twisting) vers la surface, robustesse
    Lyapunov à l'incertitude appariée — au prix de la performance de pointe.

CONCEPTION :
  u = u_LQG(x̂)  +  u_st(s) ,   s = λ·δ̂ + δ̇̂  (δ̂ = Dpᵀ q̂ : déplacement outil)
  super-twisting :  u_st = -(1/b_s)[ k1·|s|^½·tanh(s/φ) + w ],  ẇ = -k2·tanh(s/φ)
  b_s = Dpᵀ H_Pe (couplage collocation), tanh = couche-limite (dt = 50 µs).
  Le terme intégral w converge vers la perturbation appariée -> c'est le DOB.
  La base LQG (u_LQG = -K x̂, mêmes poids w_q=1e14) est l'ADN de comparaison.

Interface : step(x_hat_prev, u_prev, y_meas, k_step) — état INTERNE, appeler
reset() avant chaque simulation.
"""
import numpy as np
from scipy.linalg import solve_continuous_are, expm


class SMCDOBController:
    """Augmentation super-twisting (+DOB intégral) du LQG — le coin robuste."""

    def __init__(self, plate, dt, Dp_dist, n_per,
                 # base LQG nominale (= base LQG/DARC)
                 w_q=1e14, w_qd=1e8, w_r=1.0,
                 # surface 1er ordre + gains super-twisting (petits -> récupère LQG)
                 lam=2*np.pi*250.0, st_k1=1.0, st_k2=50.0, bl=1.0e-6,
                 # observateur d'état (Kalman, comme LQG)
                 W_kal=1e-6, V_kal=1e-12,
                 u_max=150.0, verbose=True):
        self.plate = plate
        self.dt = dt
        self.n = plate.n_modes
        self.n_x = 2 * plate.n_modes
        self.n_per = n_per
        self.u_max = u_max
        self.verbose = verbose
        n = self.n

        A = np.zeros((self.n_x, self.n_x))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.linalg.solve(plate.Mp, plate.Kp)
        A[n:, n:] = -np.linalg.solve(plate.Mp, plate.Cp)
        B = np.zeros((self.n_x, 1))
        B[n:, 0] = np.linalg.solve(plate.Mp, plate.H_Pe_modal)
        C = np.zeros((1, self.n_x))
        C[0, :n] = plate.D_obs
        self.A, self.B, self.C = A, B, C

        # retour LQG nominal (mêmes poids que la base)
        M_yp = np.outer(plate.D_obs, plate.D_obs)
        Q = np.block([[w_q*M_yp + 1e-3*np.eye(n), np.zeros((n, n))],
                      [np.zeros((n, n)), w_qd*np.eye(n) + 1e-3*np.eye(n)]])
        R = np.array([[w_r]])
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P).reshape(1, self.n_x)

        # surface 1er ordre sur le déplacement outil
        self.Dp = np.asarray(Dp_dist).reshape(n)
        self.lam = lam
        self.bs = float(self.Dp @ plate.H_Pe_modal)     # b_s = Dpᵀ H_Pe
        if abs(self.bs) < 1e-30:
            raise ValueError("b_s = Dpᵀ H_Pe ~ 0 : actionneur non colocalisé.")
        self.k1 = st_k1;  self.k2 = st_k2;  self.bl = bl

        # Kalman (même conception que LQGController)
        W = W_kal * np.eye(self.n_x)
        V = np.array([[V_kal]])
        P_kal = solve_continuous_are(A.T, C.T, W, V)
        self.L_kal = P_kal @ C.T @ np.linalg.inv(V)
        A_obs = A - self.L_kal @ C
        self.A_obs_d = expm(A_obs * dt)
        try:
            self.G_u = np.linalg.solve(A_obs, (self.A_obs_d - np.eye(self.n_x)) @ B)
            self.G_y = np.linalg.solve(A_obs, (self.A_obs_d - np.eye(self.n_x)) @ self.L_kal)
        except np.linalg.LinAlgError:
            self.G_u = B * dt
            self.G_y = self.L_kal * dt

        self.reset()
        if verbose:
            print(f"[STSMC+DOB] ||K||={np.linalg.norm(self.K):.2e}, "
                  f"b_s=Dp·H={self.bs:.3e}, lam={lam/(2*np.pi):.0f}Hz, "
                  f"k1={self.k1:g}, k2={self.k2:g}, bl={self.bl:.0e}")

    def reset(self):
        self.x_hat = np.zeros(self.n_x)
        self.w_st = 0.0          # intégrale super-twisting = estimation DOB
        self.d_hat = 0.0         # perturbation appariée estimée (diagnostic)

    def step(self, x_hat_prev, u_prev, y_meas, k_step=0):
        dt = self.dt
        self.x_hat = (self.A_obs_d @ self.x_hat
                      + self.G_u.flatten() * u_prev
                      + self.G_y.flatten() * y_meas)
        x = self.x_hat
        q = x[:self.n];  qd = x[self.n:]

        # surface 1er ordre sur le déplacement outil
        s = self.lam * float(self.Dp @ q) + float(self.Dp @ qd)
        sig = np.tanh(s / self.bl)

        # super-twisting (le terme intégral w = observateur de perturbation)
        self.w_st += dt * (-self.k2 * sig)
        self.d_hat = self.w_st / self.bs                 # perturbation appariée est.
        u_lqg = -float((self.K @ x).item())
        u_st = -(self.k1 * np.sqrt(abs(s)) * sig + self.w_st) / self.bs

        u = float(np.clip(u_lqg + u_st, -self.u_max, self.u_max))
        return x.copy(), u
