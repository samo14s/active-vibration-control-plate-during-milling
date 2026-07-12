"""
plate_model.py
==============
Plaque mince encastrée — NMODÉLISATION NON LINÉAIRE (Nasiri & Moradi, MSSP 224
(2025) 112198).

Cette version REMPLACE l'ancienne extraction modale par éléments finis
(Kirchhoff Q4) par la modélisation analytique de l'article :

   - Champ de déplacement de Von Kármán (Eq. 18-19)         → non-linéarité géométrique
   - Principe d'Hamilton + équations constitutives du piézo  (Eq. 21-29)
   - Méthode de Galerkin / sommation modale (Eq. 32-35)      → réduction en NDDE
   - Coefficient cubique de Von Kármán λ (terme +λ·η³, Eq. 30/36)
   - Couplage piézoélectrique B_piezo·∫∫∇²W   (Eq. 30-31)
   - Amortissement de process (process damping) ζ_p (Eq. 44-46)

Les FORMES PROPRES sont les produits poutre encastrée-libre × poutre libre-libre
   W_mn(X,Z) = φ_m(Z/H_P) · ψ_n(X/L_P)                       (Eq. 33-35)
avec, pour cette plaque (encastrée en Z=0, libre ailleurs) :
   - direction Z (longueur de la console H_P)  : encastrée-libre  → φ_m
   - direction X (largeur L_P, sens d'avance)  : libre-libre      → ψ_n

═══════════════════════════════════════════════════════════════════════════════
  CONSERVATION DES VALEURS PRINCIPALES
═══════════════════════════════════════════════════════════════════════════════
Les fréquences propres analytiques de Galerkin pour cette géométrie
(528.1 / 1165.3 / 3042.0 Hz pour les modes (1,1)=flexion, (1,2)=torsion,
(1,3)=flexion en largeur) sont CALIBRÉES sur les valeurs FEM d'origine du
package  [521.06, 1069.95, 2733.02] Hz  (paramètre ``freq_calib``), de sorte
que les valeurs principales (fréquences, amortissements, dimensions, matériau,
piézo) sont conservées et que les contrôleurs LQG / DARC restent inchangés.
NOTE : les facteurs de calibration valent [0.987, 0.918, 0.898] — la base de
Rayleigh mono-terme SURESTIME les modes 2-3 de 8-10 % ; les formes propres
(donc D_obs, Dp, H_Pe) restent celles de la base analytique, ce qui doit être
assumé comme un modèle hybride dans le texte de la thèse.

L'INTERFACE PUBLIQUE est strictement préservée :
   Mp, Kp, Cp, omega_n, freq_n, n_modes, V (formes),
   D_obs, H_Pe_modal, get_Dp_at(), precompute_Dp(), set_observation(),
   add_piezo_patch(), xp_array, Dp_array, DpT_Dp_array
Nouveautés (modélisation non linéaire) :
   lam_modal (coeff. cubiques λ_i), set_process_damping(), zeta_p
"""
import numpy as np


# ===========================================================================
#  Fréquences propres FEM d'origine — « valeurs principales » à conserver
#  (extraites du modèle Kirchhoff-Q4 initial : modes 1-flexion, 2-torsion,
#   3-flexion en largeur).
# ===========================================================================
PACKAGE_MODAL_FREQS_HZ = [521.06, 1069.95, 2733.02]


# ===========================================================================
#  Fonctions propres de poutre (base analytique de Galerkin, Eq. 34-35)
# ===========================================================================
# Racines de cosh(α)cos(α) = -1  (encastrée-libre)
_ALPHA_CF = np.array([1.8751040687, 4.6940911330, 7.8547574382,
                      10.9955407349, 14.1371683910])
# Racines de cosh(β)cos(β) = +1   (libre-libre, modes élastiques n>=3)
_BETA_FF = np.array([4.7300407449, 7.8532046241, 10.9956078380,
                     14.1371654913])


def _sigma_cf(a):
    return (np.cosh(a) + np.cos(a)) / (np.sinh(a) + np.sin(a))


def _sigma_ff(b):
    return (np.cosh(b) - np.cos(b)) / (np.sinh(b) - np.sin(b))


def phi_cf(m, chi, order=0):
    """
    Mode encastré-libre φ_m(χ), χ∈[0,1] (Eq. 34) et ses dérivées
    (order = 0, 1, 2 → φ, φ', φ'').
    """
    a = _ALPHA_CF[m - 1]
    s = _sigma_cf(a)
    ch, co = np.cosh(a * chi), np.cos(a * chi)
    sh, si = np.sinh(a * chi), np.sin(a * chi)
    if order == 0:
        return (ch - co) - s * (sh - si)
    if order == 1:
        return a * ((sh + si) - s * (ch - co))
    if order == 2:
        return a * a * ((ch + co) - s * (sh + si))
    raise ValueError("order ∈ {0,1,2}")


def psi_ff(n, zeta, order=0):
    """
    Mode libre-libre ψ_n(ζ), ζ∈[0,1] (Eq. 35) et ses dérivées.
    n=1 : translation rigide (1) ; n=2 : rotation rigide (√3(2ζ-1)) ;
    n>=3 : modes élastiques.
    """
    if n == 1:
        if order == 0:
            return np.ones_like(zeta)
        return np.zeros_like(zeta)
    if n == 2:
        if order == 0:
            return np.sqrt(3.0) * (2.0 * zeta - 1.0)
        if order == 1:
            return np.sqrt(3.0) * 2.0 * np.ones_like(zeta)
        return np.zeros_like(zeta)
    b = _BETA_FF[n - 3]
    s = _sigma_ff(b)
    ch, co = np.cosh(b * zeta), np.cos(b * zeta)
    sh, si = np.sinh(b * zeta), np.sin(b * zeta)
    if order == 0:
        return (ch + co) - s * (sh + si)
    if order == 1:
        return b * ((sh - si) - s * (ch + co))
    if order == 2:
        return b * b * ((ch - co) - s * (sh - si))
    raise ValueError("order ∈ {0,1,2}")


# ===========================================================================
#  PlateModel — assemblage modal analytique (Galerkin / Von Kármán)
# ===========================================================================
class PlateModel:
    """
    Modèle modal analytique de la plaque encastrée pour fraisage périphérique,
    suivant Nasiri & Moradi (MSSP 2025).

    La signature du constructeur est conservée (compatibilité descendante).
    ``N1``/``N2`` ne servent plus de maillage FEM mais de résolution
    d'intégration de Galerkin (points de Gauss).
    """

    def __init__(self,
                 lp: float, hp: float, bp: float,
                 rho: float, E: float, nu: float,
                 N1: int = 30, N2: int = 24,
                 n_modes: int = 3,
                 zeta_modes=None,
                 freq_calib=PACKAGE_MODAL_FREQS_HZ,
                 n_gauss: int = 48,
                 verbose: bool = True):
        self.lp = lp;  self.hp = hp;  self.bp = bp
        self.rho = rho;  self.E = E;  self.nu = nu
        self.N1 = N1;  self.N2 = N2
        self.n_modes = n_modes
        self.verbose = verbose

        # Rigidités de plaque / membrane (Eq. 25, 28)
        self.D_plate = E * bp**3 / (12.0 * (1.0 - nu**2))      # rigidité de flexion
        self.rho_h = rho * bp                                  # masse / surface
        self.c11 = E * bp / (1.0 - nu**2)                      # raideurs membranaires
        self.c12 = nu * E * bp / (1.0 - nu**2)
        self.c66 = E * bp / (2.0 * (1.0 + nu))

        if zeta_modes is None:
            zeta_modes = [0.0031, 0.0017, 0.0027]
        self.zeta_modes = np.array(zeta_modes[:n_modes], dtype=float)
        self.zeta_p = np.zeros(n_modes)                       # process damping

        # Points de Gauss-Legendre sur [0,1] (intégrales de Galerkin)
        xg, wg = np.polynomial.legendre.leggauss(n_gauss)
        self._xi = 0.5 * (xg + 1.0)
        self._wi = 0.5 * wg

        self._select_modes()
        self._calibrate_frequencies(freq_calib)
        self._build_modal_matrices()
        self._compute_vonkarman_cubic()

    # ---------------------------------------------------------------
    def _beam_int(self, f_order, g_order, m, n, kind):
        """∫₀¹ f^(f_order) g^(g_order) — produit de deux fonctions poutre."""
        if kind == 'phi':
            f = phi_cf(m, self._xi, f_order)
            g = phi_cf(n, self._xi, g_order)
        else:
            f = psi_ff(m, self._xi, f_order)
            g = psi_ff(n, self._xi, g_order)
        return float(np.sum(self._wi * f * g))

    def _omega_mn(self, m, n):
        """
        Pulsation propre analytique du mode (m,n) par quotient de Rayleigh
        de l'énergie de flexion de plaque isotrope (Eq. 25, 30 linéarisée).
        """
        L1 = self.hp      # direction encastrée-libre (φ) : longueur console
        L2 = self.lp      # direction libre-libre   (ψ) : largeur
        I_phi = self._beam_int(0, 0, m, m, 'phi')
        I_psi = self._beam_int(0, 0, n, n, 'psi')
        M = self.rho_h * L1 * L2 * I_phi * I_psi              # masse modale phys.

        d2ph = self._beam_int(2, 2, m, m, 'phi')              # ∫(φ'')²
        d2ps = self._beam_int(2, 2, n, n, 'psi')              # ∫(ψ'')²
        d1ph = self._beam_int(1, 1, m, m, 'phi')              # ∫(φ')²
        d1ps = self._beam_int(1, 1, n, n, 'psi')              # ∫(ψ')²
        cph = self._beam_int(2, 0, m, m, 'phi')               # ∫φ''φ
        cps = self._beam_int(2, 0, n, n, 'psi')               # ∫ψ''ψ

        U = self.D_plate * L1 * L2 * (
            d2ph * I_psi / L1**4
            + I_phi * d2ps / L2**4
            + 2.0 * self.nu * cph * cps / (L1**2 * L2**2)
            + 2.0 * (1.0 - self.nu) * d1ph * d1ps / (L1**2 * L2**2))
        return np.sqrt(U / M), M

    def _select_modes(self):
        """Énumère (m,n), trie par fréquence, conserve les n_modes plus bas."""
        cand = []
        for m in range(1, 5):
            for n in range(1, 6):
                w, M = self._omega_mn(m, n)
                cand.append((w, m, n, M))
        cand.sort(key=lambda c: c[0])
        cand = cand[:self.n_modes]
        self.modes_mn = [(m, n) for (_, m, n, _) in cand]
        self.omega_analytical = np.array([w for (w, _, _, _) in cand])
        self.M_phys = np.array([M for (_, _, _, M) in cand])
        if self.verbose:
            print("[PlateModel] Modes de Galerkin retenus (m,n) :",
                  self.modes_mn)
            print("[PlateModel] Fréquences analytiques (Hz) :",
                  np.round(self.omega_analytical / (2 * np.pi), 2))

    def _calibrate_frequencies(self, freq_calib):
        """
        Calibre les pulsations sur les valeurs principales conservées du
        package (modèle FEM d'origine).  N'affecte QUE la raideur de flexion
        linéaire ; les formes propres et la non-linéarité restent géométriques.
        """
        if freq_calib is not None and len(freq_calib) >= self.n_modes:
            self.omega_n = 2 * np.pi * np.array(freq_calib[:self.n_modes],
                                                dtype=float)
            self.freq_calibrated = True
        else:
            self.omega_n = self.omega_analytical.copy()
            self.freq_calibrated = False
        self.freq_n = self.omega_n / (2 * np.pi)
        # Facteur de calibration appliqué à chaque mode
        # ([0.987, 0.918, 0.898] pour cette plaque : la base mono-terme
        #  surestime les modes 2-3 de 8-10 %)
        self.calib_factor = self.omega_n / self.omega_analytical

    # ---------------------------------------------------------------
    def _W_normalized(self, m_idx, X, Z, dX=0, dZ=0):
        """
        Forme propre normalisée en masse  Ŵ_i(X,Z) = φ ψ / √M_phys,
        ou une de ses dérivées (dX, dZ ∈ {0,1,2}).
        ρ·h·∫∫ Ŵ² dA = 1  →  matrice de masse modale = I.
        """
        m, n = self.modes_mn[m_idx]
        chi = np.asarray(Z) / self.hp
        zeta = np.asarray(X) / self.lp
        ph = phi_cf(m, chi, dZ) / self.hp**dZ
        ps = psi_ff(n, zeta, dX) / self.lp**dX
        return ph * ps / np.sqrt(self.M_phys[m_idx])

    def _build_modal_matrices(self):
        """Matrices modales — Mp = I, Kp = diag(ω²), Cp = diag(2ζω)."""
        n = self.n_modes
        self.Mp = np.eye(n)
        self.Kp = np.diag(self.omega_n**2)
        self.Cp = np.diag(2.0 * self.zeta_modes * self.omega_n)
        # Forme propre « vecteur » conservée pour compat (valeur au coin libre)
        self.V = None
        if self.verbose:
            print("[PlateModel] Fréquences propres conservées (Hz) :")
            for k in range(n):
                print(f"   Mode {k+1} {self.modes_mn[k]} : "
                      f"{self.freq_n[k]:8.2f}  (calib x{self.calib_factor[k]:.3f})")

    # ---------------------------------------------------------------
    def _compute_vonkarman_cubic(self):
        """
        Coefficient cubique de Von Kármán λ_i (terme +λ_i·η_i³ de l'Eq. 30/36),
        par la FORME ÉNERGÉTIQUE variationnellement cohérente (u₀ = v₀ = 0).

        Énergie membranaire de Von Kármán avec ε_x = ½w_x², ε_y = ½w_y²,
        γ_xy = w_x·w_y et w = η·Ŵ_i :  U_m(η) = (λ_i/4)·η⁴, d'où le terme
        modal ∂U_m/∂η = λ_i·η³ avec

            λ_i = ∫∫ [ c11/2·Ŵ_x⁴ + c22/2·Ŵ_y⁴
                       + (c12 + 2c66)·Ŵ_x²·Ŵ_y² ] dA     ( > 0 par construction )

        NOTE de modélisation : l'hypothèse u₀ = v₀ = 0 (pas de détente
        membranaire) rend ce λ_i une BORNE SUPÉRIEURE du durcissement réel ;
        la résolution du problème membranaire (fonction d'Airy, comme dans
        l'article) donnerait un coefficient plus faible pour une plaque à
        trois bords libres.  L'ancienne implémentation projetait la forme
        forte S[Ŵ] (flux de bord parasites → λ NÉGATIF, signe corrigé par
        abs()) ; elle est remplacée par cette forme énergétique, positive et
        variationnellement justifiée.
        (x ≡ X, largeur L_P ; y ≡ Z, longueur H_P.)
        """
        c11, c12, c22, c66 = self.c11, self.c12, self.c11, self.c66
        XX = self._xi * self.lp                      # grille physique X
        ZZ = self._xi * self.hp                      # grille physique Z
        # Grilles 2D
        Xg, Zg = np.meshgrid(XX, ZZ, indexing='ij')
        Wx2, Wz2 = np.meshgrid(self._wi, self._wi, indexing='ij')
        dA = (self.lp * self.hp) * Wx2 * Wz2         # poids × Jacobien

        lam = np.zeros(self.n_modes)
        for i in range(self.n_modes):
            w_x = self._W_normalized(i, Xg, Zg, 1, 0)    # ∂/∂X
            w_y = self._W_normalized(i, Xg, Zg, 0, 1)    # ∂/∂Z
            integrand = (0.5 * c11 * w_x**4
                         + 0.5 * c22 * w_y**4
                         + (c12 + 2.0 * c66) * (w_x**2) * (w_y**2))
            lam[i] = float(np.sum(integrand * dA))

        self.lam_modal = lam
        if self.verbose:
            print("[PlateModel] Coeff. cubiques Von Karman lambda_i "
                  "(forme energetique, u0=v0=0) :",
                  np.array2string(self.lam_modal, precision=3))

    # ---------------------------------------------------------------
    def precompute_Dp(self, zp_pos: float, n_pos: int = 2001):
        """
        Pré-calcule Dp_i(X) = Ŵ_i(X, zp_pos) en n_pos positions le long de la
        trajectoire d'avance (X ∈ [0, L_P], Z = zp_pos fixe, près du bord libre).
        """
        if self.verbose:
            print(f"[PlateModel] Pré-calcul Dp(X) : {n_pos} positions "
                  f"(Z = {zp_pos*1e3:.2f} mm)...")
        xp_array = np.linspace(0.0, self.lp, n_pos)
        Dp_array = np.zeros((self.n_modes, n_pos))
        for i in range(self.n_modes):
            Dp_array[i, :] = self._W_normalized(i, xp_array,
                                                np.full_like(xp_array, zp_pos))
        DpT_Dp_array = np.einsum('ik,jk->ijk', Dp_array, Dp_array)

        self.xp_array = xp_array
        self.Dp_array = Dp_array
        self.DpT_Dp_array = DpT_Dp_array
        self.zp_pos = zp_pos

    # ---------------------------------------------------------------
    def set_observation(self, x_obs: float, z_obs: float):
        """Point de mesure du capteur (par défaut le coin libre)."""
        self.D_obs = np.array([self._W_normalized(i, x_obs, z_obs)
                               for i in range(self.n_modes)])
        self.x_obs = x_obs
        self.z_obs = z_obs

    # ---------------------------------------------------------------
    def add_piezo_patch(self,
                        xP1: float, xP2: float,
                        zP1: float, zP2: float,
                        d31: float, h_Pa: float,
                        E_Pe: float, nu_Pe: float):
        """
        Patch piézoélectrique → force modale par volt H_Pe_modal (Eq. 30-31).

            B_piezo  = -E_Pe d31 (h_Pa + b_P) / (2 (1 - ν_Pe))          (Eq. 31)
            H_Pe_i   = B_piezo · ∫∫_patch ∇²Ŵ_i dA

        avec ∫∫_patch ∇²Ŵ dA = (1/√M_i)·[ (1/L_P²) ∫φ dχ · (ψ'(ζ₂)-ψ'(ζ₁))
                                           + (1/H_P²) (φ'(χ₂)-φ'(χ₁)) · ∫ψ dζ ]·L_P·H_P
        """
        B_piezo = -E_Pe * d31 * (h_Pa + self.bp) / (2.0 * (1.0 - nu_Pe))

        chi1, chi2 = zP1 / self.hp, zP2 / self.hp
        zeta1, zeta2 = xP1 / self.lp, xP2 / self.lp
        # Sous-grilles de Gauss sur les bornes du patch
        chi_g = chi1 + (chi2 - chi1) * self._xi
        w_chi = (chi2 - chi1) * self._wi
        zeta_g = zeta1 + (zeta2 - zeta1) * self._xi
        w_zeta = (zeta2 - zeta1) * self._wi

        H = np.zeros(self.n_modes)
        for i in range(self.n_modes):
            m, n = self.modes_mn[i]
            int_phi = float(np.sum(w_chi * phi_cf(m, chi_g, 0)))
            int_psi = float(np.sum(w_zeta * psi_ff(n, zeta_g, 0)))
            dpsi_edge = psi_ff(n, np.array([zeta2]), 1)[0] \
                - psi_ff(n, np.array([zeta1]), 1)[0]
            dphi_edge = phi_cf(m, np.array([chi2]), 1)[0] \
                - phi_cf(m, np.array([chi1]), 1)[0]
            lap_int = self.lp * self.hp * (
                int_phi * dpsi_edge / self.lp**2
                + dphi_edge * int_psi / self.hp**2)
            H[i] = B_piezo * lap_int / np.sqrt(self.M_phys[i])

        self.H_Pe_modal = H
        self.patch = dict(xP1=xP1, xP2=xP2, zP1=zP1, zP2=zP2, B_piezo=B_piezo)
        if self.verbose:
            print(f"[PlateModel] B_piezo = {B_piezo:.3e} ,  "
                  f"H_Pe_modal = {np.array2string(H, precision=4)}")

    # ---------------------------------------------------------------
    def set_process_damping(self, Omega: float, Gamma: float):
        """
        Amortissement de process — OUTIL OPTIONNEL, NON UTILISÉ.

        AVERTISSEMENT : cette fonction n'est appelée par AUCUN script du
        package ; TOUS les résultats rapportés ont ζ_p = 0 (amortissement
        structurel pur).  Ne pas présenter le process damping comme une
        caractéristique du modèle simulé.  De plus, la forme
            ζ_p,i = Γ / (2 Ω M ω_i)
        réduit les Eq. 44-46 de l'article à un scalaire Γ non dérivé
        (unités non spécifiées) ; une utilisation sérieuse exigerait de
        dériver Γ de la géométrie de dépouille et de la vitesse de coupe.
        """
        if Gamma <= 0.0 or Omega <= 0.0:
            self.zeta_p = np.zeros(self.n_modes)
        else:
            self.zeta_p = Gamma / (2.0 * Omega * self.omega_n)
        self.Cp = np.diag(2.0 * (self.zeta_modes + self.zeta_p) * self.omega_n)
        if self.verbose:
            print(f"[PlateModel] zeta_p (process damping) = "
                  f"{np.array2string(self.zeta_p*100, precision=3)} %")

    # ---------------------------------------------------------------
    def get_Dp_at(self, kp: int):
        """Renvoie (Dp, DpT_Dp) à l'indice de position kp (compat)."""
        return self.Dp_array[:, kp], self.DpT_Dp_array[:, :, kp]
