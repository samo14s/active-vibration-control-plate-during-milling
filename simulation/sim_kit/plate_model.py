"""
plate_model.py
==============
Construction de la plaque mince encastrée :
   - assemblage Kirchhoff Q4
   - conditions aux limites (encastrement bord inférieur)
   - réduction modale
   - couplage piézoélectrique
   - pré-calcul Dp(x) pour outil mobile
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from kirchhoff_q4 import (
    stiffness_matrix_K, mass_matrix_K,
    shape_at_point, laplace_n_patch
)


class PlateModel:
    """
    Modèle complet de la plaque encastrée pour fraisage périphérique.

    Attributes
    ----------
    n_modes : nombre de modes conservés
    M, K : matrices modales (n_modes x n_modes), I & diag(omega²)
    C : amortissement modal diag(2*zeta*omega)
    omega_n, freq_n : pulsations et fréquences propres
    V : vecteurs propres (ndof_free x n_modes), normalisés en masse
    Dp_array : array (n_modes, n_pos) — Dp(x) le long de la trajectoire
    DpT_Dp_array : array (n_modes, n_modes, n_pos)
    D_obs : (n_modes,) — fonction de forme au point d'observation
    H_Pe_modal : (n_modes,) — vecteur force modale par volt
    """

    def __init__(self,
                 lp: float, hp: float, bp: float,
                 rho: float, E: float, nu: float,
                 N1: int = 30, N2: int = 24,
                 n_modes: int = 3,
                 zeta_modes=None,
                 verbose: bool = True):
        self.lp = lp;  self.hp = hp;  self.bp = bp
        self.rho = rho;  self.E = E;  self.nu = nu
        self.N1 = N1;  self.N2 = N2
        self.n1 = N1 + 1;  self.n2 = N2 + 1
        self.lex = lp / N1;  self.ley = hp / N2
        self.ndof = 3 * self.n1 * self.n2
        self.n_modes = n_modes
        self.verbose = verbose

        if zeta_modes is None:
            zeta_modes = [0.0031, 0.0017, 0.0027]
        self.zeta_modes = np.array(zeta_modes[:n_modes])

        self._assemble()
        self._apply_bc()
        self._modal_analysis()

    # ---------------------------------------------------------------
    def _assemble(self):
        if self.verbose:
            print("[PlateModel] Assemblage FEM...")
        Ke = stiffness_matrix_K(self.E, self.nu, self.lex, self.ley, self.bp)
        Me = mass_matrix_K(self.rho,        self.lex, self.ley, self.bp)

        nelem = self.N1 * self.N2
        n1 = self.n1
        rows = np.zeros(nelem * 144, dtype=np.int64)
        cols = np.zeros_like(rows)
        Kdat = np.zeros(nelem * 144)
        Mdat = np.zeros(nelem * 144)
        idx = 0

        for J in range(1, self.N2 + 1):
            for I in range(1, self.N1 + 1):
                DofE = np.r_[
                    3*(J-1)*n1 + 3*(I-1) + np.arange(6),
                    3*J*n1     + 3*(I-1) + np.arange(3, 6),
                    3*J*n1     + 3*(I-1) + np.arange(3),
                ]
                for a in range(12):
                    for b in range(12):
                        rows[idx] = DofE[a]
                        cols[idx] = DofE[b]
                        Kdat[idx] = Ke[a, b]
                        Mdat[idx] = Me[a, b]
                        idx += 1

        self.Kg = csr_matrix((Kdat, (rows, cols)),
                             shape=(self.ndof, self.ndof))
        self.Mg = csr_matrix((Mdat, (rows, cols)),
                             shape=(self.ndof, self.ndof))
        if self.verbose:
            print(f"[PlateModel] ndof = {self.ndof}, nelem = {nelem}")

    # ---------------------------------------------------------------
    def _apply_bc(self):
        # Encastrement : bloque w, ∂w/∂x, ∂w/∂y au bord inférieur (J=1)
        DOFb = np.arange(3 * self.n1)
        all_dofs = np.arange(self.ndof)
        DOFf = np.setdiff1d(all_dofs, DOFb)
        self.DOFf = DOFf
        self.DOFb = DOFb
        self.M_free = self.Mg[DOFf, :][:, DOFf]
        self.K_free = self.Kg[DOFf, :][:, DOFf]

    # ---------------------------------------------------------------
    def _modal_analysis(self):
        if self.verbose:
            print(f"[PlateModel] Analyse modale ({self.n_modes} modes)...")
        # eigsh résout Kx = lambda*M x ; on demande les + petits modes (sigma=0)
        w2, V = eigsh(self.K_free.tocsc(), k=self.n_modes, M=self.M_free.tocsc(),
                      sigma=0, which='LM')
        order = np.argsort(np.real(w2))
        w2 = np.real(w2[order])
        V  = np.real(V[:, order])

        self.omega_n = np.sqrt(w2)
        self.freq_n  = self.omega_n / (2 * np.pi)

        # Normalisation de masse : V'M V = I
        for k in range(self.n_modes):
            scale = np.sqrt(V[:, k] @ (self.M_free @ V[:, k]))
            V[:, k] /= scale
        self.V = V

        # Matrices modales
        self.Mp = np.eye(self.n_modes)
        self.Kp = np.diag(self.omega_n**2)
        self.Cp = np.diag(2 * self.zeta_modes * self.omega_n)

        if self.verbose:
            print("[PlateModel] Fréquences propres (Hz) :")
            for k in range(self.n_modes):
                print(f"   Mode {k+1} : {self.freq_n[k]:8.2f}")

    # ---------------------------------------------------------------
    def precompute_Dp(self, zp_pos: float, n_pos: int = 2001):
        """
        Pré-calcule Dp(x) en n_pos positions le long de zp_pos fixe.
        Utilisé pour outil mobile : lookup rapide pendant l'intégration.
        """
        if self.verbose:
            print(f"[PlateModel] Pré-calcul Dp(x) : {n_pos} positions...")

        xp_array = np.linspace(0, self.lp, n_pos)
        Dp_array = np.zeros((self.n_modes, n_pos))
        DpT_Dp_array = np.zeros((self.n_modes, self.n_modes, n_pos))

        for kp in range(n_pos):
            Nv = shape_at_point(xp_array[kp], zp_pos,
                                self.N1, self.N2,
                                self.lex, self.ley, self.n1)
            Dp_kp = Nv[self.DOFf] @ self.V          # (n_modes,)
            Dp_array[:, kp] = Dp_kp
            DpT_Dp_array[:, :, kp] = np.outer(Dp_kp, Dp_kp)

        self.xp_array = xp_array
        self.Dp_array = Dp_array
        self.DpT_Dp_array = DpT_Dp_array
        self.zp_pos = zp_pos

    # ---------------------------------------------------------------
    def set_observation(self, x_obs: float, z_obs: float):
        """Définit le point de mesure du capteur."""
        N_obs = shape_at_point(x_obs, z_obs,
                               self.N1, self.N2,
                               self.lex, self.ley, self.n1)
        self.D_obs = N_obs[self.DOFf] @ self.V       # (n_modes,)
        self.x_obs = x_obs
        self.z_obs = z_obs

    # ---------------------------------------------------------------
    def add_piezo_patch(self,
                        xP1: float, xP2: float,
                        zP1: float, zP2: float,
                        d31: float, h_Pa: float,
                        E_Pe: float, nu_Pe: float,
                        rho_Pe: float = 7450.0,
                        structural: bool = True):
        """
        Ajoute un patch piézoélectrique :

        1. (structural=True, défaut) ajoute la raideur et la masse du patch
           aux matrices K et M de la plaque, puis refait l'analyse modale.
           Ceci corrige un défaut de la version originale, qui ne calculait
           que H_Pe_modal : les fréquences propres étaient celles de la
           plaque NUE alors que les expériences de l'article (Fig.11-12,
           Table 4) portent sur la plaque AVEC patch collé.

           Raideur : rigidité de flexion composite (section transformée,
           plan neutre décalé) ; les éléments couverts par le patch sont
           multipliés par D_comp/D_plaque.
           Masse : rho_Pe * h_Pa ajouté par unité de surface
           (rho_Pe ~ 7450 kg/m3, céramique PZT typique — non fourni
           dans Table 2 de l'article).

        2. Calcule H_Pe_modal : force modale induite par 1 V (Eq. 15) :
              m_piezo = -E_Pe * d31 * (bp + h_Pa) / (2*(1 - nu_Pe))
              H_Pe(x) = m_piezo * grad2 N(x) intégré sur le patch

        Si precompute_Dp() / set_observation() ont déjà été appelés, ils
        sont automatiquement relancés avec les nouveaux modes.
        """
        if structural:
            self._add_patch_structure(xP1, xP2, zP1, zP2,
                                      h_Pa, E_Pe, nu_Pe, rho_Pe)

        m_piezo = -E_Pe * d31 * (self.bp + h_Pa) / (2 * (1 - nu_Pe))
        g_lap = laplace_n_patch(xP1, xP2, zP1, zP2,
                                self.N1, self.N2,
                                self.lex, self.ley,
                                self.n1, self.ndof)

        H_Pe_phys = m_piezo * g_lap
        self.H_Pe_modal = self.V.T @ H_Pe_phys[self.DOFf]    # (n_modes,)
        self.patch = dict(xP1=xP1, xP2=xP2, zP1=zP1, zP2=zP2,
                          m_piezo=m_piezo, structural=structural)

        if self.verbose:
            print(f"[PlateModel] ||H_Pe_modal|| = {np.linalg.norm(self.H_Pe_modal):.3e} N/V")

    # ---------------------------------------------------------------
    def _add_patch_structure(self, xP1, xP2, zP1, zP2,
                             h_Pa, E_Pe, nu_Pe, rho_Pe):
        """Ajoute dK, dM du patch aux matrices globales et refait les modes."""
        from scipy.sparse import csr_matrix

        # rigidité de flexion : plaque seule vs composite plaque+patch
        E1 = self.E / (1 - self.nu**2)
        E2 = E_Pe / (1 - nu_Pe**2)
        h1, h2 = self.bp, h_Pa
        z1, z2 = 0.0, (h1 + h2) / 2                    # plan moyen de chaque couche
        zbar = (E1*h1*z1 + E2*h2*z2) / (E1*h1 + E2*h2)  # plan neutre composite
        D_plate = E1 * h1**3 / 12
        D_comp = (E1 * (h1**3/12 + h1*(z1 - zbar)**2)
                  + E2 * (h2**3/12 + h2*(z2 - zbar)**2))
        rK = D_comp / D_plate - 1.0        # surcroît de raideur relatif
        rM = (rho_Pe * h_Pa) / (self.rho * self.bp)   # surcroît de masse relatif

        Ke = stiffness_matrix_K(self.E, self.nu, self.lex, self.ley, self.bp)
        Me = mass_matrix_K(self.rho, self.lex, self.ley, self.bp)

        rows, cols, Kd, Md = [], [], [], []
        n1 = self.n1
        n_cov = 0
        for J in range(1, self.N2 + 1):
            for I in range(1, self.N1 + 1):
                xc = (I - 0.5) * self.lex
                zc = (J - 0.5) * self.ley
                if not (xP1 - 1e-9 <= xc <= xP2 + 1e-9
                        and zP1 - 1e-9 <= zc <= zP2 + 1e-9):
                    continue
                n_cov += 1
                DofE = np.r_[
                    3*(J-1)*n1 + 3*(I-1) + np.arange(6),
                    3*J*n1     + 3*(I-1) + np.arange(3, 6),
                    3*J*n1     + 3*(I-1) + np.arange(3),
                ]
                for a in range(12):
                    for b in range(12):
                        rows.append(DofE[a]); cols.append(DofE[b])
                        Kd.append(rK * Ke[a, b])
                        Md.append(rM * Me[a, b])

        dK = csr_matrix((Kd, (rows, cols)), shape=(self.ndof, self.ndof))
        dM = csr_matrix((Md, (rows, cols)), shape=(self.ndof, self.ndof))
        self.Kg = self.Kg + dK
        self.Mg = self.Mg + dM

        if self.verbose:
            print(f"[PlateModel] patch structurel : {n_cov} éléments couverts, "
                  f"dD/D = {rK*100:.1f} %, dm/m = {rM*100:.1f} % (local)")

        # nouvelle base modale
        self._apply_bc()
        self._modal_analysis()

        # relance des pré-calculs dépendants des modes
        if hasattr(self, 'zp_pos'):
            self.precompute_Dp(self.zp_pos, self.Dp_array.shape[1])
        if hasattr(self, 'x_obs'):
            self.set_observation(self.x_obs, self.z_obs)

    # ---------------------------------------------------------------
    def calibrate_frequencies(self, f_targets_hz):
        """
        Recalage modal (model updating) : ajuste les fréquences propres du
        modèle sur des valeurs cibles (p.ex. Table 4 de l'article), sans
        toucher aux déformées modales. Seuls Kp, Cp, omega_n, freq_n sont
        mis à l'échelle ; V, Dp_array, D_obs, H_Pe_modal sont inchangés.

        Justifié par : écart résiduel formulation EF (Hermite Q4) vs
        Chebyshev-Ritz de l'article + couche de colle du patch non modélisée.
        """
        f_t = np.asarray(f_targets_hz, float)[:self.n_modes]
        if self.verbose:
            for k in range(len(f_t)):
                print(f"[PlateModel] calibrage mode {k+1} : "
                      f"{self.freq_n[k]:8.2f} Hz -> {f_t[k]:8.2f} Hz "
                      f"({(f_t[k]/self.freq_n[k]-1)*100:+.2f} %)")
        self.freq_scale = f_t / self.freq_n[:len(f_t)]
        self.omega_n = 2*np.pi*f_t
        self.freq_n = f_t.copy()
        self.Kp = np.diag(self.omega_n**2)
        self.Cp = np.diag(2 * self.zeta_modes * self.omega_n)

    # ---------------------------------------------------------------
    def get_Dp_at(self, kp: int):
        """Renvoie (Dp, DpT_Dp) à l'indice de position kp."""
        return self.Dp_array[:, kp], self.DpT_Dp_array[:, :, kp]
