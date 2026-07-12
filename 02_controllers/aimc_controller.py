"""
aimc_controller.py
==================
AIMC : Adaptive Internal-Model Cancellation — annulation harmonique
ADAPTATIVE PAR IDENTIFICATION (identification-adaptive), PAS par adaptation
directe des coefficients (FxLMS).

LA LACUNE VISÉE (état de l'art) :
  - Les commandes actives de broutement pour parois minces traitent la
    variation des dynamiques par un design robuste FIXE (pire cas) ou par la
    manipulation de la vitesse de broche — le contrôleur lui-même n'est PAS
    ré-identifié en cours d'usinage.
  - L'identification en cours d'usinage (in-process identification) existe,
    mais alimente la re-planification des paramètres de coupe et le recalcul
    des lobes de stabilité — PAS le chemin d'annulation active.
  - L'annulation harmonique adaptative (FxLMS, LMS hybride, combs) existe en
    fraisage, mais TOUJOURS sur une dynamique nominale/fixe : on adapte les
    COEFFICIENTS, pas le MODÈLE.
  => Contribution : fermer la boucle entre l'identification en ligne des
     dynamiques variables de la paroi mince et le CHEMIN D'ANNULATION
     lui-même (les gains d'annulation modèle-basés sont re-synthétisés à
     partir du modèle identifié).

POURQUOI L'ADAPTATION *INDIRECTE* (par identification) ET PAS FxLMS DIRECT :
  Résultat mesuré de cette étude (docs/ETUDE_ARCHITECTURES.md §4) : sur une
  plaque faiblement amortie dont la 2e harmonique de dent tombe sur le mode 1,
  toute adaptation DIRECTE des coefficients d'annulation (AFC/FxLMS) est
  instable en phase dans la région résonante (l'erreur de chemin secondaire y
  dépasse 90° dès quelques % de détuning) — nos essais AFC divergent, et
  IMC-LQG à gains FIXES diverge sous détuning −15 % (S3 : 15.8 µm).
  L'adaptation INDIRECTE contourne le conflit : la boucle d'IDENTIFICATION
  (bancs de filtres, stable par construction) est séparée de la boucle de
  COMMANDE ; les gains d'annulation sont recalculés ALGÉBRIQUEMENT à partir du
  modèle identifié — jamais adaptés par gradient dans la région résonante.

ARCHITECTURE (MMAE + resynthèse + engagement sûr) :
  1. BANC de N modèles candidats, grille de détuning structurel
     ρ_i ∈ [−20 %, +10 %] (ω_i = (1+ρ_i)·ω_nom, ζ constants).  Pour chaque
     candidat : filtre de Kalman AUGMENTÉ des harmoniques de dent (comme
     IMC-LQG) -> le filtre dont le modèle est juste explique la réponse
     mesurée avec ses états harmoniques ; les autres laissent un résidu.
  2. MMAE (multiple-model adaptive estimation) : probabilité a posteriori
     récursive p_i ∝ exp(−½ ν_i²/S) avec oubli exponentiel -> suit une
     dérive EN COURS d'usinage (enlèvement de matière).
  3. RESYNTHÈSE : les gains d'annulation γ_h(ρ) = −G_wy/G_uy et le filtre
     augmenté de chaque candidat sont précalculés HORS LIGNE sur la grille.
  4. MÉLANGE BAYÉSIEN (pas de commutation, pas de seuil) : la commande
     d'annulation est la moyenne a posteriori des annulations candidates,
        u_canc = Σ_i p_i · Σ_h Re{γ̄_h(ρ_i) ẑ_h^i},
     et l'état plante du retour LQG est la moyenne a posteriori des
     estimées.  Auto-régularisant : au démarrage les états harmoniques
     ẑ_h sont nuls (annulation ≈ 0) et se construisent PENDANT que la
     vraisemblance se concentre ; un candidat mal phasé perd son poids
     exponentiellement vite.  Le retour LQG nominal n'est JAMAIS commuté.

INFORMATION UTILISÉE (protocole honnête) : capteur de déplacement + période
de dent (encodeur) + la grille de modèles nominale.  AUCUN accès au détuning
réel, à K_T réel, ni à la dérive simulée.

Interface : step(x_hat_prev, u_prev, y_meas, k_step) — état INTERNE,
appeler reset() avant chaque simulation.
"""
import numpy as np
from scipy.linalg import solve_continuous_are, expm


class _CandidateModel:
    """Un modèle candidat : plante détunée + filtre augmenté + gains γ_h."""
    __slots__ = ('rho', 'A_obs_d', 'G_u', 'G_y', 'gamma_h', 'n_x', 'n_a')


class AIMCController:
    """Annulation harmonique adaptative par identification (MMAE)."""

    def __init__(self, plate, dt, f_fund, Dp_dist,
                 rho_grid=(-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10),
                 n_harm=8,
                 w_q=1e14, w_qd=1e8, w_r=1.0,
                 W_x=1e-6, q_dist=1e3, V_meas=1e-12,
                 # MMAE : oubli + échelle de vraisemblance
                 forget=0.995, S_lik=1e-14,
                 u_max=150.0, verbose=True):
        """
        f_fund  : fréquence fondamentale de passage de dent simulée [Hz]
                  (1/(n_per·dt) — information encodeur).
        Dp_dist : direction modale de la perturbation (forme au point outil,
                  NOMINALE).
        rho_grid: grille de détunings candidats (fraction de ω nominal).
        forget  : oubli exponentiel des vraisemblances (0.995 @ 20 kHz
                  ≈ fenêtre 10 ms) -> suit une dérive en cours d'usinage.
        S_lik   : variance d'innovation de référence pour la vraisemblance
                  (≈ variance du bruit capteur ; (0.1 µm)² = 1e-14).
        """
        self.plate = plate
        self.dt = dt
        self.n = plate.n_modes
        self.n_x = 2 * plate.n_modes
        self.n_harm = n_harm
        self.u_max = u_max
        self.verbose = verbose
        self.rho_grid = np.asarray(rho_grid, dtype=float)
        self.N = len(self.rho_grid)
        self.forget = forget
        n = self.n

        # --- plante NOMINALE : B, C (indépendants du détuning) ---
        B = np.zeros((self.n_x, 1))
        B[n:, 0] = np.linalg.solve(plate.Mp, plate.H_Pe_modal)
        C = np.zeros((1, self.n_x))
        C[0, :n] = plate.D_obs
        self.B, self.C = B, C
        Bd = np.zeros((self.n_x, 1))
        Bd[n:, 0] = np.asarray(Dp_dist)

        omega_nom = plate.omega_n.copy()
        zeta = np.diag(plate.Cp) / (2.0 * omega_nom)     # ζ modaux

        # --- retour LQG NOMINAL (jamais commuté ; base commune de l'étude) ---
        A_nom = self._build_A(omega_nom, zeta)
        M_yp = np.outer(plate.D_obs, plate.D_obs)
        Q = np.block([[w_q*M_yp + 1e-3*np.eye(n), np.zeros((n, n))],
                      [np.zeros((n, n)), w_qd*np.eye(n) + 1e-3*np.eye(n)]])
        R = np.array([[w_r]])
        P = solve_continuous_are(A_nom, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P).reshape(1, self.n_x)

        # --- harmoniques de dent (encodeur) ---
        self.omega_h = 2*np.pi*f_fund*np.arange(1, n_harm+1)
        n_d = 2*n_harm
        self.n_a = self.n_x + n_d

        # --- banc de candidats : filtre augmenté + gains γ_h par détuning ---
        self.models = []
        V = np.array([[V_meas]])
        for rho in self.rho_grid:
            m = _CandidateModel()
            m.rho = float(rho)
            m.n_x = self.n_x
            m.n_a = self.n_a
            A_i = self._build_A(omega_nom*(1.0+rho), zeta)
            # système augmenté (mêmes oscillateurs harmoniques pour tous)
            A_aug = np.zeros((self.n_a, self.n_a))
            A_aug[:self.n_x, :self.n_x] = A_i
            for h in range(n_harm):
                i0 = self.n_x + 2*h
                w_h = self.omega_h[h]
                A_aug[i0, i0+1] = w_h
                A_aug[i0+1, i0] = -w_h
                A_aug[:self.n_x, i0] = Bd.flatten()
            B_aug = np.vstack([B, np.zeros((n_d, 1))])
            C_aug = np.hstack([C, np.zeros((1, n_d))])
            W_aug = np.zeros((self.n_a, self.n_a))
            W_aug[:self.n_x, :self.n_x] = W_x*np.eye(self.n_x)
            W_aug[self.n_x:, self.n_x:] = q_dist*np.eye(n_d)
            P_k = solve_continuous_are(A_aug.T, C_aug.T, W_aug, V)
            L = P_k @ C_aug.T @ np.linalg.inv(V)
            A_obs = A_aug - L @ C_aug
            m.A_obs_d = expm(A_obs*dt)
            try:
                m.G_u = np.linalg.solve(A_obs, (m.A_obs_d - np.eye(self.n_a)) @ B_aug).flatten()
                m.G_y = np.linalg.solve(A_obs, (m.A_obs_d - np.eye(self.n_a)) @ L).flatten()
            except np.linalg.LinAlgError:
                m.G_u = (B_aug*dt).flatten()
                m.G_y = (L*dt).flatten()
            # gains d'annulation γ_h(ρ) : équations du régulateur sur la
            # boucle fermée (A_i - B K) — resynthèse modèle-basée
            A_cl = A_i - B @ self.K
            Im = np.eye(self.n_x)
            m.gamma_h = np.zeros(n_harm+1, dtype=complex)
            for h in range(1, n_harm+1):
                Mh = 1j*self.omega_h[h-1]*Im - A_cl
                Gwy = complex((C @ np.linalg.solve(Mh, Bd))[0, 0])
                Guy = complex((C @ np.linalg.solve(Mh, B))[0, 0])
                if abs(Guy) > 1e-30:
                    m.gamma_h[h] = -Gwy/Guy
            self.models.append(m)

        # variance d'innovation de référence (échelle du bruit capteur)
        self.S_ref = max(S_lik, 1e-18)

        self.reset()
        if verbose:
            print(f"[AIMC] banc MMAE : {self.N} candidats rho="
                  f"{np.array2string(self.rho_grid*100, precision=0)}% , "
                  f"{n_harm} harmoniques, mélange bayésien (oubli {forget})")

    # ----------------------------------------------------------------
    def _build_A(self, omega, zeta):
        n = self.n
        A = np.zeros((self.n_x, self.n_x))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.diag(omega**2)
        A[n:, n:] = -np.diag(2.0*zeta*omega)
        return A

    # ----------------------------------------------------------------
    def reset(self):
        self.x_bank = [np.zeros(self.n_a) for _ in range(self.N)]
        self.logp = np.zeros(self.N)          # log-vraisemblances (relatives)
        # traces de diagnostic
        self.trace_rho = []                    # détuning estimé (moyenne post.)
        self.trace_conf = []                   # max p_i

    # ----------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas, k_step=0):
        # 1. banc de filtres augmentés + log-vraisemblances récursives
        nu2 = np.empty(self.N)
        for i, m in enumerate(self.models):
            xa = self.x_bank[i]
            y_pred = float(self.C[0, :] @ xa[:self.n_x])
            nu = y_meas - y_pred
            nu2[i] = nu*nu
            self.x_bank[i] = m.A_obs_d @ xa + m.G_u*u_prev + m.G_y*y_meas
        # MMAE : oubli exponentiel, normalisation log-somme
        self.logp = self.forget*self.logp - 0.5*nu2/self.S_ref
        self.logp -= self.logp.max()
        p = np.exp(self.logp)
        p /= p.sum()

        # 2. MÉLANGE BAYÉSIEN : état plante et annulation moyennés par p_i
        x_mix = np.zeros(self.n_x)
        u_canc = 0.0
        for i, m in enumerate(self.models):
            if p[i] < 1e-4:
                continue
            xa = self.x_bank[i]
            x_mix += p[i] * xa[:self.n_x]
            uc = 0.0
            for h in range(1, self.n_harm+1):
                i0 = self.n_x + 2*(h-1)
                z_h = complex(xa[i0], xa[i0+1])
                uc += float(np.real(np.conj(m.gamma_h[h]) * z_h))
            u_canc += p[i] * uc

        u = -float((self.K @ x_mix).item()) + u_canc
        u = float(np.clip(u, -self.u_max, self.u_max))

        self.trace_rho.append(float(p @ self.rho_grid))
        self.trace_conf.append(float(p.max()))
        return x_mix.copy(), u
