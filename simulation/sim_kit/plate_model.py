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
    shape_at_point, laplace_n_patch, matrix_der_K
)


def shear_lag_efficiency(hx: float, hz: float,
                         E_b: float, nu_b: float, h_b: float,
                         E_p: float, nu_p: float, h_p: float,
                         G_adh: float, t_adh: float, alpha: float = 6.0):
    """
    Rendement de transfert de deformation d'un patch colle — theorie du
    shear lag, Crawley & de Luis, AIAA J. 25(10):1373-1385, 1987.

    Le patch n'est pas soude a la plaque : il lui est colle par une couche
    d'epaisseur t_adh et de module de cisaillement G_adh. Cette couche ne
    transmet le cisaillement que sur une longueur caracteristique 1/Gamma
    depuis chaque bord, si bien que la deformation dans le patch vaut

        eps(x) = eps_parfait * [1 - cosh(Gamma x)/cosh(Gamma l)]

    et non eps_parfait partout. Avec

        Gamma^2 = (G_adh/t_adh) * (1/(E_p' h_p) + alpha/(E_b' h_b))

    ou E' = E/(1-nu^2) et alpha = 6 pour la flexion. La moyenne de eps sur le
    patch, rapportee au cas soude, vaut

        eta(Gamma l) = 1 - tanh(Gamma l)/(Gamma l)

    -> 1 quand Gamma l >> 1 (collage parfait), -> (Gamma l)^2/3 quand la colle
    devient molle. Pour un patch rectangulaire on prend la forme separable
    eta_2D = eta(Gamma hx) * eta(Gamma hz) sur les DEMI-dimensions : c'est une
    approximation (la solution 2D exacte n'est pas separable), acceptable ici
    car 1/Gamma est petit devant le patch.

    Parameters
    ----------
    hx, hz : DEMI-longueurs du patch selon x et z [m]
    h_b, h_p : epaisseurs plaque et patch [m]
    G_adh, t_adh : module de cisaillement [Pa] et epaisseur [m] de la colle

    Returns
    -------
    (eta, Gamma) : rendement dans [0, 1] et parametre de shear lag [1/m]
    """
    Eb = E_b/(1.0 - nu_b**2)
    Ep = E_p/(1.0 - nu_p**2)
    Gamma = np.sqrt((G_adh/t_adh)*(1.0/(Ep*h_p) + alpha/(Eb*h_b)))
    def _eta(x):
        return 1.0 - np.tanh(x)/x if x > 1e-12 else 0.0
    return float(_eta(Gamma*hx)*_eta(Gamma*hz)), float(Gamma)


def _bilinear_B(xi, eta, lex, ley):
    """Matrice B membranaire (3x8) d'un Q4 bilineaire, ordre de noeuds
    identique a l'element de flexion : (-1,-1), (+1,-1), (+1,+1), (-1,+1)."""
    dNx = np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])/4*(2/lex)
    dNy = np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])/4*(2/ley)
    B = np.zeros((3, 8))
    B[0, 0::2] = dNx
    B[1, 1::2] = dNy
    B[2, 0::2] = dNy
    B[2, 1::2] = dNx
    return B


def _Qbar(E, nu):
    """Matrice de rigidite reduite en contraintes planes (3x3)."""
    return E/(1 - nu**2)*np.array([[1.0, nu, 0.0],
                                   [nu, 1.0, 0.0],
                                   [0.0, 0.0, (1 - nu)/2]])


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
        if getattr(self, '_ident_ratio', None) is not None:
            # H_Pe identifie est defini PAR RAPPORT a D_obs : deplacer le
            # capteur sans le reappliquer laisserait les deux incoherents.
            self.set_identified_coupling(self._ident_ratio)

    # ---------------------------------------------------------------
    def set_identified_coupling(self, ratio):
        """Impose la REPARTITION MODALE de H_Pe identifiee sur la Fig. 12(b).

        La fonction de transfert tension -> capteur vaut

            G_b(w) = somme_i  r_i / (w_i^2 - w^2 + 2 j z_i w_i w),
            r_i = D_obs,i * H_Pe,i

        Ses POLES sont les frequences mesurees (deja calees) et ses ZEROS sont
        les creux de la Fig. 12(b). Poles et zeros determinent les r_i a une
        constante multiplicative pres — c'est exactement ce que `ratio` porte.
        On en deduit H_Pe = r / D_obs : la fonction de transfert modelisee
        devient celle qui a ete mesuree.

        La CONSTANTE, elle, n'est pas identifiee : le niveau absolu des figures
        de l'article n'est pas fiable (defaut F9 — la Fig. 12(a) demande +12 dB
        et la Fig. 12(b) +6.8 dB pour se superposer au meme modele, ce qu'aucun
        modele ne peut satisfaire a la fois). On conserve donc la NORME du
        couplage elements finis et on ne corrige que sa repartition ; le niveau
        reste balaye par `gain_H` (facteurs 0.5 et 2 dans run_demo --full).

        Le signe global est lui aussi conventionnel (il depend de la face sur
        laquelle le patch est colle) : on le choisit pour que H reste du meme
        cote que le couplage elements finis, ce qui evite un saut de signe sans
        contenu physique entre les deux modes de fonctionnement.
        """
        r = np.asarray(ratio, float).ravel()
        if r.shape != (self.n_modes,):
            raise ValueError(f"ratio doit avoir {self.n_modes} composantes")
        if not hasattr(self, 'D_obs'):
            raise RuntimeError("set_observation() doit preceder "
                               "set_identified_coupling()")
        H_fem = np.asarray(getattr(self, 'H_Pe_fem', self.H_Pe_modal), float)
        H = r/np.asarray(self.D_obs, float).ravel()
        H *= np.linalg.norm(H_fem)/np.linalg.norm(H)
        if H @ H_fem < 0:
            H = -H
        self.H_Pe_fem = H_fem
        self.H_Pe_modal = H
        self._ident_ratio = r

    # ---------------------------------------------------------------
    def add_piezo_patch(self,
                        xP1: float, xP2: float,
                        zP1: float, zP2: float,
                        d31: float, h_Pa: float,
                        E_Pe: float, nu_Pe: float,
                        rho_Pe: float = 7450.0,
                        structural: bool = True,
                        G_adh: float = None, t_adh: float = None,
                        alpha_lag: float = 6.0,
                        membrane: bool = True):
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

        3. Couche de colle (G_adh, t_adh). Sans elle, le patch est suppose
           SOUDE a la plaque : transfert de deformation parfait. Une colle
           reelle ne transmet le cisaillement que sur ~1/Gamma depuis chaque
           bord, d'ou un rendement eta < 1 (shear_lag_efficiency, theorie de
           Crawley & de Luis). Ce MEME eta reduit deux choses a la fois :
             - la part COMPOSITE du raidissement (le patch ne peut porter sa
               contrainte axiale decalee que si la colle la transmet) ; sa
               flexion propre E_p h_p^3/12, elle, ne demande aucun collage et
               n'est pas reduite ;
             - le couplage piezoelectrique H_Pe, dans les memes proportions.
           Un seul parametre physique, deux effets : ce ne sont pas deux
           coefficients d'ajustement independants.
           G_adh = None (defaut) restitue le collage parfait.

        4. Couplage MEMBRANE-FLEXION (membrane=True, defaut). Un patch colle
           d'un seul cote applique aussi une resultante membranaire, qui se
           rabat en flexion parce que le stratifie plaque+patch est non
           symetrique. Voir _membrane_load_correction. Effet mesure : -11 %
           sur H_Pe pour les modes 1, 2, 4, 5 et -29 % sur le mode 3. NE change
           PAS la signature des creux de la Fig. 12(b) : l'effet est quasi
           scalaire, et un scalaire se simplifie dans les rapports de residus
           qui fixent les zeros.

        Si precompute_Dp() / set_observation() ont déjà été appelés, ils
        sont automatiquement relancés avec les nouveaux modes.
        """
        if G_adh is None or t_adh is None:
            eta_bond, Gamma_bond = 1.0, np.inf
        else:
            eta_bond, Gamma_bond = shear_lag_efficiency(
                0.5*(xP2 - xP1), 0.5*(zP2 - zP1),
                self.E, self.nu, self.bp, E_Pe, nu_Pe, h_Pa,
                G_adh, t_adh, alpha_lag)
        self.eta_bond = eta_bond
        self.Gamma_bond = Gamma_bond
        if self.verbose and np.isfinite(Gamma_bond):
            print(f"[PlateModel] colle : 1/Gamma = {1e3/Gamma_bond:.3f} mm, "
                  f"rendement de transfert eta = {eta_bond:.4f}")

        if structural:
            self._add_patch_structure(xP1, xP2, zP1, zP2,
                                      h_Pa, E_Pe, nu_Pe, rho_Pe, eta_bond)

        m_piezo = -eta_bond * E_Pe * d31 * (self.bp + h_Pa) / (2 * (1 - nu_Pe))
        g_lap = laplace_n_patch(xP1, xP2, zP1, zP2,
                                self.N1, self.N2,
                                self.lex, self.ley,
                                self.n1, self.ndof)

        H_Pe_phys = m_piezo * g_lap

        # Couplage membrane-flexion du patch colle d'un seul cote : sous
        # tension il applique aussi une resultante membranaire, qui se rabat en
        # flexion parce que le stratifie est non symetrique.
        self.membrane_coupling = bool(membrane)
        if membrane:
            # M_E = N_E (bp + h_Pa)/2, donc N_E porte le signe OPPOSE de
            # m_piezo : ce dernier est deja -M_E (convention de B_b).
            n_E = eta_bond * E_Pe * d31 / (1 - nu_Pe)
            N_E = np.array([n_E, n_E, 0.0])
            corr, _ = self._membrane_load_correction(xP1, xP2, zP1, zP2,
                                                     h_Pa, E_Pe, nu_Pe, N_E)
            if self.verbose:
                r = (np.linalg.norm(corr[self.DOFf])
                     / max(np.linalg.norm(H_Pe_phys[self.DOFf]), 1e-30))
                print(f"[PlateModel] couplage membrane-flexion : "
                      f"|correction|/|flexion| = {r:.3f} en ddl bruts")
            H_Pe_phys = H_Pe_phys + corr
        self.H_Pe_modal = self.V.T @ H_Pe_phys[self.DOFf]    # (n_modes,)
        self.patch = dict(xP1=xP1, xP2=xP2, zP1=zP1, zP2=zP2,
                          m_piezo=m_piezo, structural=structural)

        if self.verbose:
            print(f"[PlateModel] ||H_Pe_modal|| = {np.linalg.norm(self.H_Pe_modal):.3e} N/V")

    # ---------------------------------------------------------------
    def _add_patch_structure(self, xP1, xP2, zP1, zP2,
                             h_Pa, E_Pe, nu_Pe, rho_Pe, eta_bond=1.0):
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
        # Le patch apporte sa flexion propre SANS aucun collage ; l'effet
        # composite (plan neutre decale) demande, lui, que la colle transmette
        # la contrainte axiale. Seul ce second terme est reduit par eta_bond.
        D_free = E2 * h2**3 / 12
        rK_free = D_free / D_plate
        rK_perfect = D_comp / D_plate - 1.0
        rK = rK_free + eta_bond*(rK_perfect - rK_free)
        rM = (rho_Pe * h_Pa) / (self.rho * self.bp)   # surcroît de masse relatif

        Ke = stiffness_matrix_K(self.E, self.nu, self.lex, self.ley, self.bp)
        Me = mass_matrix_K(self.rho, self.lex, self.ley, self.bp)

        # Chaque élément est pondéré par sa FRACTION DE RECOUVREMENT avec le
        # rectangle du patch, et non retenu ou rejeté selon la position de son
        # centre. Le test au centre quantifiait l'empreinte sur la grille
        # (19.44 x 61.33 mm au lieu de 20 x 60), alors que laplace_n_patch, lui,
        # intègre exactement sur le rectangle : raideur/masse ajoutées et
        # couplage piézoélectrique ne portaient pas sur la même aire.
        rows, cols, Kd, Md = [], [], [], []
        n1 = self.n1
        n_cov = 0
        area_cov = 0.0
        a_elem = self.lex * self.ley
        for J in range(1, self.N2 + 1):
            for I in range(1, self.N1 + 1):
                ox = min(I*self.lex, xP2) - max((I-1)*self.lex, xP1)
                oz = min(J*self.ley, zP2) - max((J-1)*self.ley, zP1)
                if ox <= 1e-12 or oz <= 1e-12:
                    continue
                frac = (ox * oz) / a_elem
                n_cov += 1
                area_cov += ox * oz
                DofE = np.r_[
                    3*(J-1)*n1 + 3*(I-1) + np.arange(6),
                    3*J*n1     + 3*(I-1) + np.arange(3, 6),
                    3*J*n1     + 3*(I-1) + np.arange(3),
                ]
                for a in range(12):
                    for b in range(12):
                        rows.append(DofE[a]); cols.append(DofE[b])
                        Kd.append(frac * rK * Ke[a, b])
                        Md.append(frac * rM * Me[a, b])

        dK = csr_matrix((Kd, (rows, cols)), shape=(self.ndof, self.ndof))
        dM = csr_matrix((Md, (rows, cols)), shape=(self.ndof, self.ndof))
        self.Kg = self.Kg + dK
        self.Mg = self.Mg + dM

        if self.verbose:
            a_exact = (xP2 - xP1)*(zP2 - zP1)
            print(f"[PlateModel] patch structurel : {n_cov} éléments touchés, "
                  f"aire couverte {area_cov*1e6:.2f} mm2 "
                  f"(exacte {a_exact*1e6:.2f}, ecart {area_cov/a_exact-1:+.2%}), "
                  f"dD/D = {rK*100:.1f} % (collage parfait {rK_perfect*100:.1f} %, "
                  f"decolle {rK_free*100:.1f} %), dm/m = {rM*100:.1f} % (local)")

        # nouvelle base modale
        self._apply_bc()
        self._modal_analysis()

        # relance des pré-calculs dépendants des modes
        if hasattr(self, 'zp_pos'):
            self.precompute_Dp(self.zp_pos, self.Dp_array.shape[1])
        if hasattr(self, 'x_obs'):
            self.set_observation(self.x_obs, self.z_obs)

    # ---------------------------------------------------------------
    def _membrane_load_correction(self, xP1, xP2, zP1, zP2,
                                  h_Pa, E_Pe, nu_Pe, N_E):
        """
        Correction du vecteur de charge piezo due au couplage MEMBRANE-FLEXION.

        Un patch colle d'UN SEUL COTE n'applique pas qu'un moment : sous
        tension il applique aussi une resultante membranaire N_E dans son plan.
        Le stratifie plaque+patch etant NON SYMETRIQUE, sa matrice de couplage
        B = Qbar_p * h_p (h_plaque + h_p) / 2 est non nulle sur la zone du
        patch, et cette resultante se rabat en flexion. Une formulation de
        Kirchhoff en flexion pure ne peut pas le representer : elle n'a pas de
        ddl membranaires.

        Les modes membranaires de cette plaque sont vers 33 kHz, tres au-dessus
        des 4.1 kHz utiles ; on condense donc les ddl (u, v) en STATIQUE :

            [K_mm  K_mb][u]   [f_m]
            [K_mb^T K_bb][w] = [f_b]
            =>  (K_bb - K_mb^T K_mm^-1 K_mb) w = f_b - K_mb^T K_mm^-1 f_m

        Cette methode ne renvoie que la correction de CHARGE, second terme du
        membre de droite. La correction de RAIDEUR, elle, est deja prise en
        compte : _add_patch_structure utilise la rigidite de section
        transformee, qui est exactement le resultat de la meme condensation
        (verifie a 0.002 % pres, cf. VERIFICATION.md section 7).

        Returns
        -------
        (correction, info) : vecteur (ndof,) a AJOUTER a la charge de flexion,
        et un dict de diagnostic.
        """
        from scipy.sparse.linalg import splu

        n1 = self.n1
        nnod = n1*self.n2
        Q_pl = _Qbar(self.E, self.nu)
        Q_pa = _Qbar(E_Pe, nu_Pe)
        A_bare = Q_pl*self.bp
        A_pat = Q_pl*self.bp + Q_pa*h_Pa
        B_pat = Q_pa*(h_Pa*(self.bp + h_Pa)/2)

        gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        rm, cm, vm = [], [], []
        rc, cc, vc = [], [], []
        f_m = np.zeros(2*nnod)
        for J in range(1, self.N2 + 1):
            for I in range(1, self.N1 + 1):
                xl, xh = (I-1)*self.lex, I*self.lex
                zl, zh = (J-1)*self.ley, J*self.ley
                ox = min(xh, xP2) - max(xl, xP1)
                oz = min(zh, zP2) - max(zl, zP1)
                cov = ox > 1e-12 and oz > 1e-12
                nodes = [(J-1)*n1 + (I-1), (J-1)*n1 + I, J*n1 + I, J*n1 + (I-1)]
                mdof = np.array([[2*n, 2*n+1] for n in nodes]).ravel()
                bdof = np.array([[3*n, 3*n+1, 3*n+2] for n in nodes]).ravel()

                # rigidite membranaire : integrale sur TOUT l'element, avec la
                # part du patch ponderee par le recouvrement
                frac = (ox*oz)/(self.lex*self.ley) if cov else 0.0
                Am = A_bare + frac*(A_pat - A_bare)
                Ke_mm = np.zeros((8, 8))
                for i in range(2):
                    for j in range(2):
                        Bm = _bilinear_B(gp[i], gp[j], self.lex, self.ley)
                        Ke_mm += (self.lex/2)*(self.ley/2)*Bm.T @ Am @ Bm
                for a in range(8):
                    for b in range(8):
                        rm.append(mdof[a]); cm.append(mdof[b])
                        vm.append(Ke_mm[a, b])
                if not cov:
                    continue

                # couplage et charge membranaire : integres EXACTEMENT sur
                # l'intersection patch/element, comme laplace_n_patch
                a_, b_ = max(xl, xP1), min(xh, xP2)
                c_, d_ = max(zl, zP1), min(zh, zP2)
                xm, xr = (a_+b_)/2, (b_-a_)/2
                zm, zr = (c_+d_)/2, (d_-c_)/2
                Ke_mb = np.zeros((8, 12))
                fe_m = np.zeros(8)
                for i in range(2):
                    for j in range(2):
                        xg = xm + xr*gp[i]
                        zg = zm + zr*gp[j]
                        xi = 2*(xg - xl)/self.lex - 1
                        et = 2*(zg - zl)/self.ley - 1
                        Bm = _bilinear_B(xi, et, self.lex, self.ley)
                        Bb = matrix_der_K(xi, et, self.lex, self.ley)
                        w = xr*zr
                        Ke_mb += w*Bm.T @ B_pat @ Bb
                        fe_m += w*Bm.T @ N_E
                for a in range(8):
                    for b in range(12):
                        rc.append(mdof[a]); cc.append(bdof[b])
                        vc.append(Ke_mb[a, b])
                f_m[mdof] += fe_m

        K_mm = csr_matrix((vm, (rm, cm)), shape=(2*nnod, 2*nnod))
        K_mb = csr_matrix((vc, (rc, cc)), shape=(2*nnod, self.ndof))
        # encastrement : u = v = 0 sur le bord inferieur
        mfree = np.setdiff1d(np.arange(2*nnod), np.arange(2*n1))
        y = splu(K_mm[mfree, :][:, mfree].tocsc()).solve(f_m[mfree])
        corr = -(K_mb[mfree, :].T @ y)
        return np.asarray(corr).ravel(), dict(n_mfree=len(mfree))

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
