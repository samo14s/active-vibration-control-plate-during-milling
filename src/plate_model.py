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
        # eigsh résout Kx = lambda*M x ; on demande les + petits modes (sigma=0).
        # A fixed start vector v0 makes ARPACK deterministic: without it the
        # random start yields eigenvectors that differ (in sign, and near
        # degeneracies in shape) from build to build, so a gain synthesised on
        # one PlateModel instance would not transfer to another.
        v0 = np.ones(self.M_free.shape[0])
        w2, V = eigsh(self.K_free.tocsc(), k=self.n_modes, M=self.M_free.tocsc(),
                      sigma=0, which='LM', v0=v0)
        order = np.argsort(np.real(w2))
        w2 = np.real(w2[order])
        V  = np.real(V[:, order])

        self.omega_n = np.sqrt(w2)
        self.freq_n  = self.omega_n / (2 * np.pi)

        # Normalisation de masse : V'M V = I
        for k in range(self.n_modes):
            scale = np.sqrt(V[:, k] @ (self.M_free @ V[:, k]))
            V[:, k] /= scale
        # Canonical sign convention: eigsh (ARPACK) returns eigenvectors with an
        # arbitrary sign, which changes the sign of the modal shape Dp and the
        # actuator vector H_Pe from one build to the next.  A feedback gain
        # computed on one PlateModel instance would then be mis-applied (wrong
        # sign) to another instance.  Fix the sign so the largest-magnitude
        # component of each mode is positive, making modal shapes -- and hence
        # any gain synthesised from them -- reproducible across builds.
        for k in range(self.n_modes):
            imax = np.argmax(np.abs(V[:, k]))
            if V[imax, k] < 0:
                V[:, k] = -V[:, k]
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
                        E_Pe: float, nu_Pe: float):
        """
        Ajoute un patch piézoélectrique et calcule H_Pe_modal :
        force modale induite par 1 V appliqué au patch.

        Modèle de couplage en flexion (Eq. 15) :
            m_piezo = -E_Pe * d31 * (bp + h_Pa) / (2*(1 - nu_Pe))
            H_Pe(x) = m_piezo * ∇²N(x) intégré sur le patch
        """
        m_piezo = -E_Pe * d31 * (self.bp + h_Pa) / (2 * (1 - nu_Pe))
        g_lap = laplace_n_patch(xP1, xP2, zP1, zP2,
                                self.N1, self.N2,
                                self.lex, self.ley,
                                self.n1, self.ndof)

        H_Pe_phys = m_piezo * g_lap
        self.H_Pe_modal = self.V.T @ H_Pe_phys[self.DOFf]    # (n_modes,)
        self.patch = dict(xP1=xP1, xP2=xP2, zP1=zP1, zP2=zP2,
                          m_piezo=m_piezo)

        if self.verbose:
            print(f"[PlateModel] ||H_Pe_modal|| = {np.linalg.norm(self.H_Pe_modal):.3e} N/V")

    # ---------------------------------------------------------------
    def get_Dp_at(self, kp: int):
        """Renvoie (Dp, DpT_Dp) à l'indice de position kp."""
        return self.Dp_array[:, kp], self.DpT_Dp_array[:, :, kp]
