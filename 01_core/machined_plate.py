"""
machined_plate.py
=================
NMODÉLISATION PHYSIQUE de la paroi AVEC ENLÈVEMENT DE MATIÈRE — solveur modal
de Rayleigh-Ritz à ÉPAISSEUR SPATIALEMENT VARIABLE B(X,Z).

Objet : au lieu d'appliquer un facteur scalaire k_scale aux matrices modales
(qui suppose un amincissement UNIFORME de toute la paroi -> ω_i ∝ B, identique
pour tous les modes), on RÉ-DÉRIVE le modèle modal à partir de la GÉOMÉTRIE
réellement usinée : la matière est retirée localement (bande usinée), donc
l'épaisseur B(X,Z) est non uniforme, les fréquences se décalent de manière
DIFFÉRENTE selon les modes, et les formes propres elles-mêmes évoluent.

Méthode (Rayleigh-Ritz sur la base de fonctions de poutre du modèle nominal) :
   base  W_a(X,Z) = φ_{m_a}(Z/H_P)·ψ_{n_a}(X/L_P)     (encastrée-libre × libre-libre)
   masse M_ab = ∫∫ ρ·B(X,Z)·W_a·W_b dX dZ
   raideur K_ab = ∫∫ D(X,Z)·[ W_a,xx W_b,xx + W_a,yy W_b,yy
                    + ν(W_a,xx W_b,yy + W_a,yy W_b,xx) + 2(1−ν) W_a,xy W_b,xy ] dX dZ
   avec D(X,Z) = E·B(X,Z)³/(12(1−ν²)),  x≡X (largeur L_P),  y≡Z (hauteur H_P).
   Problème aux valeurs propres généralisé  K φ = ω² M φ.

Cohérence avec le modèle nominal :
 - à épaisseur UNIFORME B₀, on retrouve les fréquences analytiques de Ritz
   (proches du quotient de Rayleigh mono-terme de plate_model) ; on applique
   les MÊMES facteurs de calibration (vers les ancres FEM) que plate_model ;
 - un amincissement UNIFORME B₀→αB₀ redonne EXACTEMENT ω_i ∝ α pour tous les
   modes (loi de plaque, vérifiée) ;
 - un amincissement LOCAL décale les modes différemment (test honnête de la
   grille à paramètre unique ρ de l'AIMC).

Limites explicites : base de Ritz finie (les formes propres restent dans le
sous-espace des produits de poutre) ; pas de rétroaction géométrique
contrainte-matière ; ζ modaux supposés constants.
"""
import numpy as np

from plate_model import phi_cf, psi_ff, PACKAGE_MODAL_FREQS_HZ


class MachinedPlate:
    """Modèle modal de plaque à épaisseur variable (Rayleigh-Ritz)."""

    def __init__(self, lp, hp, bp, rho, E, nu,
                 ritz_modes=((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (1, 4)),
                 n_target=3, n_gauss=40,
                 calib_freqs=PACKAGE_MODAL_FREQS_HZ, verbose=True):
        self.lp = lp; self.hp = hp; self.bp = bp
        self.rho = rho; self.E = E; self.nu = nu
        self.ritz_modes = list(ritz_modes)
        self.Nb = len(self.ritz_modes)
        self.n_target = n_target
        self.verbose = verbose

        # Gauss-Legendre 2D sur [0,1]² (mappé sur [0,L_P]×[0,H_P])
        xg, wg = np.polynomial.legendre.leggauss(n_gauss)
        u = 0.5*(xg + 1.0); w = 0.5*wg
        self._zx = u * lp                       # grille X (largeur)
        self._zz = u * hp                       # grille Z (hauteur)
        self._wx = w * lp; self._wz = w * hp    # poids (avec Jacobien)
        # pré-évaluation des fonctions de base et dérivées aux points de Gauss
        self._precompute_basis()

        # assemblage nominal (épaisseur uniforme B0) + calibration
        M0, K0 = self._assemble(np.full((n_gauss, n_gauss), bp))
        w2, V = self._geneig(M0, K0)
        self.omega_nom_ritz = np.sqrt(w2[:n_target])
        # calibration par mode vers les ancres FEM (comme plate_model)
        if calib_freqs is not None:
            self.calib = (2*np.pi*np.asarray(calib_freqs[:n_target])
                          / self.omega_nom_ritz)
        else:
            self.calib = np.ones(n_target)
        self.M0, self.K0 = M0, K0
        if verbose:
            f_ritz = self.omega_nom_ritz/(2*np.pi)
            print(f"[MachinedPlate] Ritz {self.Nb} fonctions, "
                  f"f_nom={np.round(f_ritz,1)} Hz, calib={np.round(self.calib,3)}")

    # ------------------------------------------------------------------
    def _precompute_basis(self):
        """Valeurs de W_a et dérivées (xx, yy, xy, 0) aux points de Gauss 2D."""
        ng = len(self._zx)
        chi = self._zz / self.hp                # Z/H_P (vecteur)
        zet = self._zx / self.lp                # X/L_P (vecteur)
        self._W = []; self._Wxx = []; self._Wyy = []; self._Wxy = []
        for (m, n) in self.ritz_modes:
            ph0 = phi_cf(m, chi, 0); ph1 = phi_cf(m, chi, 1)/self.hp
            ph2 = phi_cf(m, chi, 2)/self.hp**2
            ps0 = psi_ff(n, zet, 0); ps1 = psi_ff(n, zet, 1)/self.lp
            ps2 = psi_ff(n, zet, 2)/self.lp**2
            # grilles 2D (X = axe 0, Z = axe 1) : W(x,z) = ps(x)·ph(z)
            self._W.append(np.outer(ps0, ph0))
            self._Wxx.append(np.outer(ps2, ph0))     # ∂²/∂X²
            self._Wyy.append(np.outer(ps0, ph2))     # ∂²/∂Z²
            self._Wxy.append(np.outer(ps1, ph1))     # ∂²/∂X∂Z
        self._dA = np.outer(self._wx, self._wz)       # poids 2D (Jacobien inclus)

    # ------------------------------------------------------------------
    def _assemble(self, B_grid):
        """Matrices de Ritz M, K pour un champ d'épaisseur B(X,Z) (grille ng×ng)."""
        nu = self.nu
        D = self.E * B_grid**3 / (12.0*(1.0 - nu**2))     # rigidité de flexion
        rhoB = self.rho * B_grid
        dA = self._dA
        Nb = self.Nb
        M = np.zeros((Nb, Nb)); K = np.zeros((Nb, Nb))
        for a in range(Nb):
            for b in range(a, Nb):
                Mab = np.sum(rhoB * self._W[a]*self._W[b] * dA)
                bend = (self._Wxx[a]*self._Wxx[b] + self._Wyy[a]*self._Wyy[b]
                        + nu*(self._Wxx[a]*self._Wyy[b] + self._Wyy[a]*self._Wxx[b])
                        + 2*(1-nu)*self._Wxy[a]*self._Wxy[b])
                Kab = np.sum(D * bend * dA)
                M[a, b] = M[b, a] = Mab
                K[a, b] = K[b, a] = Kab
        return M, K

    # ------------------------------------------------------------------
    def _geneig(self, M, K):
        """Valeurs/vecteurs propres généralisés K φ = ω² M φ (tri croissant)."""
        from scipy.linalg import eigh
        w2, V = eigh(K, M)                       # symétrique généralisé, M>0
        idx = np.argsort(w2)
        return w2[idx], V[:, idx]

    # ------------------------------------------------------------------
    def modal_from_thickness(self, B_grid):
        """
        Ré-dérive le modèle modal pour un champ d'épaisseur B(X,Z) donné.

        Returns dict :
          omega   : (n_target,) pulsations calibrées [rad/s]
          coeff   : (Nb, n_target) coefficients de Ritz mass-normalisés
                    (forme propre i = Σ_a coeff[a,i]·W_a, ρ∫∫B W² = 1)
        """
        M, K = self._assemble(B_grid)
        w2, V = self._geneig(M, K)
        omega_raw = np.sqrt(np.abs(w2[:self.n_target]))
        omega = omega_raw * self.calib          # calibration par mode (nominale)
        # mass-normalisation : φᵀ M φ = 1
        coeff = V[:, :self.n_target].copy()
        for i in range(self.n_target):
            nrm = np.sqrt(coeff[:, i] @ M @ coeff[:, i])
            coeff[:, i] /= max(nrm, 1e-300)
        return dict(omega=omega, coeff=coeff, M=M)

    # ------------------------------------------------------------------
    def nominal_basis(self):
        """Coefficients de Ritz des modes NOMINAUX (base fixe Φ₀, Nb×n_target)."""
        return self.modal_from_thickness(
            np.full((len(self._zx), len(self._zx)), self.bp))['coeff']

    def modal_matrices(self, B_grid, coeff0):
        """
        Matrices modales dans la BASE FIXE des modes nominaux Φ₀ (méthode
        rigoureuse pour une structure à géométrie variable) :
            Mp = Φ₀ᵀ M(B) Φ₀ ,  Kp = C·[Φ₀ᵀ K(B) Φ₀]·C ,  C = diag(calib)
        À B = B₀ : Mp = I, Kp = diag(ω_calib²).  Pour B ≠ B₀ les matrices
        deviennent PLEINES (les modes nominaux ne sont plus propres) — c'est
        le couplage physique induit par l'usinage.  Les projections (Dp,
        D_obs, H_Pe) restent celles de la base NOMINALE (fixe), cohérent avec
        des contrôleurs conçus sur le modèle nominal.

        Returns (Mp, Kp, omega_inst) : matrices 3×3 et pulsations instantanées
        (eig de (Mp,Kp), calibrées).
        """
        from scipy.linalg import eigh
        M, K = self._assemble(B_grid)
        C = np.diag(self.calib)
        Mp = coeff0.T @ M @ coeff0
        Kp = C @ (coeff0.T @ K @ coeff0) @ C
        Mp = 0.5*(Mp + Mp.T); Kp = 0.5*(Kp + Kp.T)
        w2 = eigh(Kp, Mp, eigvals_only=True)
        omega_inst = np.sqrt(np.abs(np.sort(w2)))
        return Mp, Kp, omega_inst

    def damping_matrix(self, Mp, Kp, zeta):
        """
        Matrice d'amortissement modal préservant les ζ sur les modes
        INSTANTANÉS de (Mp,Kp) : Cp = Mp Ψ diag(2ζ_iω_i) Ψᵀ Mp, où Ψ sont les
        modes propres généralisés M-orthonormés (ζ constants dans le temps).
        """
        from scipy.linalg import eigh
        w2, Psi = eigh(Kp, Mp)            # Ψᵀ Mp Ψ = I, Ψᵀ Kp Ψ = diag(w2)
        om = np.sqrt(np.abs(w2))
        zc = np.asarray(zeta)[:len(om)]
        Cmodal = np.diag(2.0*zc*om)       # dans la base propre instantanée
        MPsi = Mp @ Psi
        return MPsi @ Cmodal @ MPsi.T

    # ------------------------------------------------------------------
    def shape_at(self, coeff_i, X, Z, dX=0, dZ=0):
        """Forme propre i (combinaison de Ritz) évaluée en (X,Z) ou sa dérivée."""
        chi = np.asarray(Z)/self.hp; zet = np.asarray(X)/self.lp
        out = np.zeros(np.broadcast(np.asarray(X), np.asarray(Z)).shape)
        for a, (m, n) in enumerate(self.ritz_modes):
            ph = phi_cf(m, chi, dZ)/self.hp**dZ
            ps = psi_ff(n, zet, dX)/self.lp**dX
            out = out + coeff_i[a]*ps*ph
        return out
