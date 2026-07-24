"""
plate_model.py  (Mindlin edition)
=================================
Construction de la plaque encastrée avec la théorie de **Reissner-Mindlin** :
   - assemblage de l'élément Serendip 8 nœuds (mindlin_q8.py)
   - conditions aux limites (encastrement du bord inférieur : w = theta_x = theta_y = 0)
   - réduction modale
   - couplage piézoélectrique (analogie de moment cohérente Mindlin)
   - pré-calcul Dp(x) pour outil mobile

La classe `PlateModel` conserve EXACTEMENT la même interface publique que la
version Kirchhoff du package de l'article (constructeur, attributs modaux Mp,
Kp, Cp, omega_n, freq_n, V, et méthodes precompute_Dp / set_observation /
add_piezo_patch / get_Dp_at). Le reste du package (solveur de Newmark,
contrôleurs LQG / DARC-MPC, analyse de stabilité FDM, figures) fonctionne donc
sans aucune modification — seule la théorie de plaque sous-jacente change
(Kirchhoff -> Mindlin).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from mindlin_q8 import (
    stiffness_matrix_M, mass_matrix_M,
    build_node_map, elem_dofs_M, clamped_edge_dofs,
    shape_at_point_M, piezo_moment_load_M,
)


class PlateModel:
    """
    Modèle complet de la plaque encastrée (Mindlin) pour fraisage périphérique.

    Attributes
    ----------
    n_modes : nombre de modes conservés
    Mp, Kp : matrices modales (n_modes x n_modes), I & diag(omega²)
    Cp : amortissement modal diag(2*zeta*omega)
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
                 kappa: float = 5.0 / 6.0,
                 verbose: bool = True):
        self.lp = lp;  self.hp = hp;  self.bp = bp
        self.rho = rho;  self.E = E;  self.nu = nu
        self.kappa = kappa                       # facteur de correction de cisaillement
        self.N1 = N1;  self.N2 = N2
        self.lex = lp / N1;  self.ley = hp / N2
        self.n_modes = n_modes
        self.verbose = verbose

        # Maillage Serendip 8 nœuds
        self.node_id, self.ntot = build_node_map(N1, N2)
        self.ndof = 3 * self.ntot

        if zeta_modes is None:
            zeta_modes = [0.0031, 0.0017, 0.0027]
        self.zeta_modes = np.array(zeta_modes[:n_modes])

        self._assemble()
        self._apply_bc()
        self._modal_analysis()

    # ---------------------------------------------------------------
    def _assemble(self):
        if self.verbose:
            print("[PlateModel-Mindlin] Assemblage FEM (Serendip Q8)...")
        Ke = stiffness_matrix_M(self.E, self.nu, self.kappa,
                                self.lex, self.ley, self.bp)
        Me = mass_matrix_M(self.rho, self.lex, self.ley, self.bp)

        nelem = self.N1 * self.N2
        rows = np.zeros(nelem * 576, dtype=np.int64)   # 24*24 = 576
        cols = np.zeros_like(rows)
        Kdat = np.zeros(nelem * 576)
        Mdat = np.zeros(nelem * 576)
        idx = 0

        for J in range(1, self.N2 + 1):
            for I in range(1, self.N1 + 1):
                DofE = elem_dofs_M(I, J, self.node_id)
                for a in range(24):
                    da = DofE[a]
                    for b in range(24):
                        rows[idx] = da
                        cols[idx] = DofE[b]
                        Kdat[idx] = Ke[a, b]
                        Mdat[idx] = Me[a, b]
                        idx += 1

        self.Kg = csr_matrix((Kdat, (rows, cols)),
                             shape=(self.ndof, self.ndof))
        self.Mg = csr_matrix((Mdat, (rows, cols)),
                             shape=(self.ndof, self.ndof))
        if self.verbose:
            print(f"[PlateModel-Mindlin] ndof = {self.ndof}, nelem = {nelem}, "
                  f"nnodes = {self.ntot}")

    # ---------------------------------------------------------------
    def _apply_bc(self):
        # Encastrement du bord inférieur (z = 0, iy = 0) :
        # w = theta_x = theta_y = 0 sur toute l'arête.
        DOFb = clamped_edge_dofs(self.node_id, edge="bottom")
        all_dofs = np.arange(self.ndof)
        DOFf = np.setdiff1d(all_dofs, DOFb)
        self.DOFf = DOFf
        self.DOFb = DOFb
        self.M_free = self.Mg[DOFf, :][:, DOFf]
        self.K_free = self.Kg[DOFf, :][:, DOFf]

    # ---------------------------------------------------------------
    def _modal_analysis(self):
        if self.verbose:
            print(f"[PlateModel-Mindlin] Analyse modale ({self.n_modes} modes)...")
        # Kx = lambda*M x ; plus petits modes (sigma=0, shift-invert)
        w2, V = eigsh(self.K_free.tocsc(), k=self.n_modes,
                      M=self.M_free.tocsc(), sigma=0, which='LM')
        order = np.argsort(np.real(w2))
        w2 = np.real(w2[order])
        V = np.real(V[:, order])

        self.omega_n = np.sqrt(np.abs(w2))
        self.freq_n = self.omega_n / (2 * np.pi)

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
            print("[PlateModel-Mindlin] Fréquences propres (Hz) :")
            for k in range(self.n_modes):
                print(f"   Mode {k+1} : {self.freq_n[k]:8.2f}")

    # ---------------------------------------------------------------
    def precompute_Dp(self, zp_pos: float, n_pos: int = 2001):
        """
        Pré-calcule Dp(x) en n_pos positions le long de zp_pos fixe.
        Utilisé pour outil mobile : lookup rapide pendant l'intégration.
        """
        if self.verbose:
            print(f"[PlateModel-Mindlin] Pré-calcul Dp(x) : {n_pos} positions...")

        xp_array = np.linspace(0, self.lp, n_pos)
        Dp_array = np.zeros((self.n_modes, n_pos))
        DpT_Dp_array = np.zeros((self.n_modes, self.n_modes, n_pos))

        for kp in range(n_pos):
            Nv = shape_at_point_M(xp_array[kp], zp_pos,
                                  self.N1, self.N2, self.lex, self.ley,
                                  self.node_id, self.ndof)
            Dp_kp = Nv[self.DOFf] @ self.V            # (n_modes,)
            Dp_array[:, kp] = Dp_kp
            DpT_Dp_array[:, :, kp] = np.outer(Dp_kp, Dp_kp)

        self.xp_array = xp_array
        self.Dp_array = Dp_array
        self.DpT_Dp_array = DpT_Dp_array
        self.zp_pos = zp_pos

    # ---------------------------------------------------------------
    def set_observation(self, x_obs: float, z_obs: float):
        """Définit le point de mesure du capteur."""
        N_obs = shape_at_point_M(x_obs, z_obs,
                                 self.N1, self.N2, self.lex, self.ley,
                                 self.node_id, self.ndof)
        self.D_obs = N_obs[self.DOFf] @ self.V        # (n_modes,)
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

        Modèle de couplage en flexion (analogie de moment, cohérent Mindlin) :
            m_piezo = -E_Pe * d31 * (bp + h_Pa) / (2*(1 - nu_Pe))
            H_Pe(x) = m_piezo * ∫_patch Bf^T [1,1,0]^T dA

        C'est l'analogue Mindlin exact du couplage Kirchhoff de l'article
        (`m_piezo * ∫∇²N dA`) : même moment de flexion physique appliqué,
        mais formulé de façon cohérente avec les déformations Bf de Mindlin.
        """
        m_piezo = -E_Pe * d31 * (self.bp + h_Pa) / (2 * (1 - nu_Pe))
        g_mom = piezo_moment_load_M(xP1, xP2, zP1, zP2,
                                    self.N1, self.N2, self.lex, self.ley,
                                    self.node_id, self.ndof)

        H_Pe_phys = m_piezo * g_mom
        self.H_Pe_modal = self.V.T @ H_Pe_phys[self.DOFf]    # (n_modes,)
        self.patch = dict(xP1=xP1, xP2=xP2, zP1=zP1, zP2=zP2,
                          m_piezo=m_piezo)

        if self.verbose:
            print(f"[PlateModel-Mindlin] ||H_Pe_modal|| = "
                  f"{np.linalg.norm(self.H_Pe_modal):.3e} N/V")

    # ---------------------------------------------------------------
    def get_Dp_at(self, kp: int):
        """Renvoie (Dp, DpT_Dp) à l'indice de position kp."""
        return self.Dp_array[:, kp], self.DpT_Dp_array[:, :, kp]
