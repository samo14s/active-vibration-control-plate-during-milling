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
import copy as _copy
from types import SimpleNamespace
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
                        E_Pe: float, nu_Pe: float):
        """
        Ajoute un patch piézoélectrique et calcule H_Pe_modal (force modale par volt).

        Coupling coefficient from Du et al. (2024), Eq. (15) — P2 fix. The induced
        bending moment per volt is m_piezo = -C_P0 * d31 / h_Pa, with

            P_M  = -(E_Pe/E_P) * (1 - nu_P^2)/(1 - nu_Pe^2)
                    * 3*h_Pa*bp*(bp + h_Pa) / (0.5*bp^3 + 4*h_Pa^3 + 3*bp*h_Pa^2)
            C_P0 = -(1/6) * (1 + nu_Pe)/(1 - nu_P) * E_P * bp^2 * P_M
                    / (1 + nu_P - (1 + nu_Pe)*P_M)

        (The previous code used the simplified constant
         -E_Pe*d31*(bp+h_Pa)/(2*(1-nu_Pe)), which is ~16 % larger.) The spatial part
        g_lap = integral of the Laplacian of the shape functions over the patch is,
        by the divergence theorem, the article's Eq. (14) bracket of patch-edge slope
        line integrals — that structure is unchanged; only the scalar prefactor is
        corrected here.
        """
        E_P = self.E
        nu_P = self.nu
        bp = self.bp
        P_M = (-(E_Pe / E_P) * (1 - nu_P**2) / (1 - nu_Pe**2)
               * 3 * h_Pa * bp * (bp + h_Pa)
               / (0.5 * bp**3 + 4 * h_Pa**3 + 3 * bp * h_Pa**2))
        C_P0 = (-(1.0 / 6.0) * (1 + nu_Pe) / (1 - nu_P) * E_P * bp**2 * P_M
                / (1 + nu_P - (1 + nu_Pe) * P_M))
        m_piezo = -C_P0 * d31 / h_Pa

        g_lap = laplace_n_patch(xP1, xP2, zP1, zP2,
                                self.N1, self.N2,
                                self.lex, self.ley,
                                self.n1, self.ndof)

        H_Pe_phys = m_piezo * g_lap
        self.H_Pe_modal = self.V.T @ H_Pe_phys[self.DOFf]    # (n_modes,)
        self.patch = dict(xP1=xP1, xP2=xP2, zP1=zP1, zP2=zP2,
                          m_piezo=m_piezo, C_P0=C_P0, P_M=P_M)

        if self.verbose:
            print(f"[PlateModel] C_P0 = {C_P0:.4e}, m_piezo = {m_piezo:.4e}; "
                  f"||H_Pe_modal|| = {np.linalg.norm(self.H_Pe_modal):.3e} N/V")

    # ---------------------------------------------------------------
    def get_Dp_at(self, kp: int):
        """Renvoie (Dp, DpT_Dp) à l'indice de position kp."""
        return self.Dp_array[:, kp], self.DpT_Dp_array[:, :, kp]

    # ---------------------------------------------------------------
    def truncated_view(self, n_keep: int):
        """
        Lightweight modal view of the FIRST n_keep modes, for controller design.

        Because it reuses this plate's own eigenvectors (D_obs, H_Pe_modal share the
        same mode signs), a controller designed on the view is sign-consistent with
        any full-order plant derived from the SAME eigensolve (see perturbed_copy).
        This is what lets us design a 3-mode controller and simulate a 5-mode plant
        WITHOUT an inverse crime and without a feedback-sign mismatch.
        """
        n = int(min(n_keep, self.n_modes))
        return SimpleNamespace(
            n_modes=n,
            Mp=np.eye(n),
            Kp=np.diag(self.omega_n[:n]**2),
            Cp=np.diag(2.0 * self.zeta_modes[:n] * self.omega_n[:n]),
            omega_n=self.omega_n[:n].copy(),
            freq_n=self.freq_n[:n].copy(),
            D_obs=self.D_obs[:n].copy(),
            H_Pe_modal=self.H_Pe_modal[:n].copy(),
        )

    # ---------------------------------------------------------------
    def perturbed_copy(self, freq_perturb: float = 0.0):
        """
        Return a copy with modal frequencies (and damping) rescaled by
        (1 + freq_perturb). Mode SHAPES (V, D_obs, H_Pe_modal, Dp_array) are SHARED
        and unchanged — the article models material-removal drift as a stiffness
        perturbation, not a mode reshaping. Using a copy instead of a fresh eigensolve
        keeps mode signs identical to this plate (and hence to any truncated_view of
        it), which is essential for feedback-sign consistency.

        The returned copy still points at THIS plate's Dp_array; call precompute_Dp on
        it afterwards if a different tool height (zp_pos) is needed — that rebinds the
        copy's Dp_array only, using the shared (sign-consistent) eigenvectors.
        """
        new = _copy.copy(self)
        if abs(freq_perturb) > 1e-12:
            fac = 1.0 + freq_perturb
            new.omega_n = self.omega_n * fac
            new.freq_n = new.omega_n / (2 * np.pi)
            new.Kp = np.diag(new.omega_n**2)
            new.Cp = np.diag(2.0 * self.zeta_modes * new.omega_n)
        return new
