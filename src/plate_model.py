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
                        E_Pe: float, nu_Pe: float,
                        rho_Pe: float = 7800.0,
                        include_dynamics: bool = False):
        """
        Ajoute un patch piézoélectrique et calcule H_Pe_modal :
        force modale induite par 1 V appliqué au patch.

        Modèle de couplage en flexion (Eq. 15) :
            m_piezo = -E_Pe * d31 * (bp + h_Pa) / (2*(1 - nu_Pe))
            H_Pe(x) = m_piezo * ∇²N(x) intégré sur le patch

        include_dynamics=True ajoute en plus la MASSE et la RAIDEUR du patch
        au modèle EF (couche collée sur une face, section composite avec axe
        neutre déplacé), puis relance l'analyse modale et les pré-calculs
        (Dp, observation) qui en dépendent.  rho_Pe : masse volumique du
        matériau piézo (PZT ~ 7800 kg/m³).  L'inertie rotatoire de la couche
        déportée est négligée (hypothèse plaque mince standard).
        """
        if include_dynamics:
            self._add_patch_dynamics(xP1, xP2, zP1, zP2, h_Pa, E_Pe, nu_Pe, rho_Pe)

        m_piezo = -E_Pe * d31 * (self.bp + h_Pa) / (2 * (1 - nu_Pe))
        g_lap = laplace_n_patch(xP1, xP2, zP1, zP2,
                                self.N1, self.N2,
                                self.lex, self.ley,
                                self.n1, self.ndof)

        H_Pe_phys = m_piezo * g_lap
        self.H_Pe_modal = self.V.T @ H_Pe_phys[self.DOFf]    # (n_modes,)
        self.patch = dict(xP1=xP1, xP2=xP2, zP1=zP1, zP2=zP2,
                          m_piezo=m_piezo, include_dynamics=include_dynamics)

        if self.verbose:
            print(f"[PlateModel] ||H_Pe_modal|| = {np.linalg.norm(self.H_Pe_modal):.3e} N/V")

    # ---------------------------------------------------------------
    def _add_patch_dynamics(self, xP1, xP2, zP1, zP2, h_Pa, E_Pe, nu_Pe, rho_Pe):
        """Masse + raideur de la couche piézo collée sur une face.

        Section composite par unité de surface (plaque [-h/2, h/2], patch
        [h/2, h/2+hp]) : l'axe neutre se déplace de

            z_bar = E2' hp zc2 / (E1' h + E2' hp),   zc2 = (h+hp)/2,

        avec Ei' = Ei/(1-nu_i²).  L'incrément de rigidité de flexion se
        décompose en un terme de transport de la plaque (motif de Poisson
        nu_plaque) et le terme propre du patch (motif nu_patch) :

            dD1 = E1' h z_bar²                (transport plaque)
            dD2 = E2' (hp³/12 + hp (zc2 - z_bar)²)   (patch autour de l'axe)

        Chaque terme est assemblé via la matrice de raideur élémentaire
        standard avec un module équivalent E_eq tel que
        E_eq h³ / 12(1-nu²) = dD.  La masse ajoutée rho_Pe*hp est assemblée
        avec la matrice de masse cohérente (rho_eq = rho_Pe hp / h).
        L'analyse modale est ensuite relancée; Dp et l'observation sont
        recalculés s'ils avaient été définis.
        """
        h, hp = self.bp, h_Pa
        E1p = self.E / (1 - self.nu ** 2)
        E2p = E_Pe / (1 - nu_Pe ** 2)
        zc2 = (h + hp) / 2.0
        z_bar = (E2p * hp * zc2) / (E1p * h + E2p * hp)
        dD1 = E1p * h * z_bar ** 2
        dD2 = E2p * (hp ** 3 / 12.0 + hp * (zc2 - z_bar) ** 2)

        E_eq1 = 12.0 * (1 - self.nu ** 2) * dD1 / h ** 3
        E_eq2 = 12.0 * (1 - nu_Pe ** 2) * dD2 / h ** 3
        rho_eq = rho_Pe * hp / h

        dKe = (stiffness_matrix_K(E_eq1, self.nu, self.lex, self.ley, h)
               + stiffness_matrix_K(E_eq2, nu_Pe, self.lex, self.ley, h))
        dMe = mass_matrix_K(rho_eq, self.lex, self.ley, h)

        # éléments entièrement couverts par le patch (le patch de l'article
        # est aligné sur la grille; un recouvrement partiel est pondéré par
        # la fraction de surface couverte)
        Kg = self.Kg.tolil()
        Mg = self.Mg.tolil()
        n1 = self.n1
        for J in range(1, self.N2 + 1):
            for I in range(1, self.N1 + 1):
                xe_lo, xe_hi = (I - 1) * self.lex, I * self.lex
                ye_lo, ye_hi = (J - 1) * self.ley, J * self.ley
                dx = max(0.0, min(xe_hi, xP2) - max(xe_lo, xP1))
                dy = max(0.0, min(ye_hi, zP2) - max(ye_lo, zP1))
                frac = (dx * dy) / (self.lex * self.ley)
                if frac < 1e-12:
                    continue
                DofE = np.r_[
                    3 * (J - 1) * n1 + 3 * (I - 1) + np.arange(6),
                    3 * J * n1 + 3 * (I - 1) + np.arange(3, 6),
                    3 * J * n1 + 3 * (I - 1) + np.arange(3),
                ]
                Kg[np.ix_(DofE, DofE)] += frac * dKe
                Mg[np.ix_(DofE, DofE)] += frac * dMe
        self.Kg = Kg.tocsr()
        self.Mg = Mg.tocsr()
        self._apply_bc()
        self._modal_analysis()
        # recompute anything that depends on the mode shapes
        if hasattr(self, 'zp_pos'):
            self.precompute_Dp(self.zp_pos, n_pos=self.Dp_array.shape[1])
        if hasattr(self, 'x_obs'):
            self.set_observation(self.x_obs, self.z_obs)
        if self.verbose:
            print(f"[PlateModel] patch dynamics added: z_bar={z_bar*1e3:.3f} mm, "
                  f"dD1={dD1:.2f}, dD2={dD2:.2f} N·m, rho_eq·h={rho_eq*h:.2f} kg/m²")

    # ---------------------------------------------------------------
    def remove_material(self, x1: float, x2: float,
                        z1: float, z2: float,
                        depth: float):
        """Enlèvement de matière en cours d'usinage (fraisage périphérique).

        Amincit la plaque de ``depth`` (réduction d'épaisseur, p.ex. la prise
        radiale a_e) sur la bande [x1,x2] x [z1,z2].  Pour chaque élément
        couvert, la raideur et la masse élémentaires sont remplacées par
        celles de l'épaisseur restante (K ~ h³, M ~ h), pondérées par la
        fraction de surface couverte (recouvrement partiel).  L'analyse
        modale et les pré-calculs dépendants (Dp, observation) sont relancés.

        Hypothèse standard « piecewise-frozen » : à un état d'usinage donné,
        la dynamique est celle de la géométrie courante (l'enlèvement est
        beaucoup plus lent que les vibrations).  Les appels sont cumulables
        (l'épaisseur locale est suivie par élément dans self.h_elem).
        """
        if not hasattr(self, 'h_elem'):
            self.h_elem = np.full((self.N2, self.N1), self.bp, dtype=float)

        Kg = self.Kg.tolil()
        Mg = self.Mg.tolil()
        n1 = self.n1
        n_mod = 0
        for J in range(1, self.N2 + 1):
            for I in range(1, self.N1 + 1):
                xe_lo, xe_hi = (I - 1) * self.lex, I * self.lex
                ze_lo, ze_hi = (J - 1) * self.ley, J * self.ley
                dx = max(0.0, min(xe_hi, x2) - max(xe_lo, x1))
                dz = max(0.0, min(ze_hi, z2) - max(ze_lo, z1))
                frac = (dx * dz) / (self.lex * self.ley)
                if frac < 1e-12:
                    continue
                h_old = self.h_elem[J - 1, I - 1]
                h_new = max(h_old - depth, 1e-5)
                if abs(h_new - h_old) < 1e-12:
                    continue
                dKe = (stiffness_matrix_K(self.E, self.nu, self.lex, self.ley, h_new)
                       - stiffness_matrix_K(self.E, self.nu, self.lex, self.ley, h_old))
                dMe = (mass_matrix_K(self.rho, self.lex, self.ley, h_new)
                       - mass_matrix_K(self.rho, self.lex, self.ley, h_old))
                DofE = np.r_[
                    3 * (J - 1) * n1 + 3 * (I - 1) + np.arange(6),
                    3 * J * n1 + 3 * (I - 1) + np.arange(3, 6),
                    3 * J * n1 + 3 * (I - 1) + np.arange(3),
                ]
                Kg[np.ix_(DofE, DofE)] += frac * dKe
                Mg[np.ix_(DofE, DofE)] += frac * dMe
                # suivre l'épaisseur seulement si l'élément est (quasi)
                # entièrement couvert; sinon l'amincissement partiel est déjà
                # représenté par la pondération frac
                if frac > 0.999:
                    self.h_elem[J - 1, I - 1] = h_new
                n_mod += 1
        self.Kg = Kg.tocsr()
        self.Mg = Mg.tocsr()
        self._apply_bc()
        self._modal_analysis()
        if hasattr(self, 'zp_pos'):
            self.precompute_Dp(self.zp_pos, n_pos=self.Dp_array.shape[1])
        if hasattr(self, 'x_obs'):
            self.set_observation(self.x_obs, self.z_obs)
        if hasattr(self, 'patch'):        # re-project actuator on new modes
            m_piezo = self.patch['m_piezo']
            g_lap = laplace_n_patch(self.patch['xP1'], self.patch['xP2'],
                                    self.patch['zP1'], self.patch['zP2'],
                                    self.N1, self.N2, self.lex, self.ley,
                                    self.n1, self.ndof)
            self.H_Pe_modal = self.V.T @ (m_piezo * g_lap)[self.DOFf]
        if self.verbose:
            print(f"[PlateModel] material removed on {n_mod} elements; "
                  f"f1 = {self.freq_n[0]:.2f} Hz")

    # ---------------------------------------------------------------
    def get_Dp_at(self, kp: int):
        """Renvoie (Dp, DpT_Dp) à l'indice de position kp."""
        return self.Dp_array[:, kp], self.DpT_Dp_array[:, :, kp]
