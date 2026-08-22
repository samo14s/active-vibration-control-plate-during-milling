"""
chebyshev_plate.py — Remodelisation COMPLETE de la plaque du papier
===================================================================
Du, Liu, Dai & Long, "Robust combined time delay control for milling chatter
suppression of flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257.

Cette remodelisation suit la METHODE DU PAPIER (Section 2 + Annexe A) et non
les elements finis du fichier milling_plant.py fourni :

  * deplacement   y_P(x,z,t) = Y_P(x~, z~) q_y(t),  Eq. (7)-(8),
    ou Y_P est le produit de Kronecker de polynomes de Chebyshev de 1re espece
    (PX x PZ fonctions), en coordonnees non-dimensionnelles
        x~ = 2x/l_P - 1,   z~ = 2z/h_P - 1 ;
  * energie de flexion de Kirchhoff -> K_PD (Eq. A.2) ;
  * energie cinetique avec inertie de rotation I2 -> M_PD (Eq. A.1) ;
  * bord inferieur encastre par PENALISATION -> K_kappa (Eq. A.4, ressorts
    k_w en translation et k_r en rotation le long de z = 0). Le terme K_lambda
    du principe variationnel modifie (Eq. A.3) sert dans le papier a stabiliser
    le calcul avec des conditions relachees ; la penalisation seule converge
    vers les memes frequences (verifie par balayage de k_w, k_r) et suffit ici.

Donnees (Tableau 1) : AL6061, l_P = 100 mm, h_P = 80 mm, b_P = 4 mm,
rho = 2830 kg/m^3, E = 69 GPa, nu = 0.33.

Sorties principales :
    freq_n  : frequences propres [Hz] — a comparer aux "theoretical natural
              frequency" du Tableau 4 : 537 / 1101 / 2805 / 3423 / 4254 Hz ;
    V       : modes normalises en masse (V^T M V = I), donc D = Y V est en
              1/sqrt(kg) et a4 * D^T D est bien homogene a des rad^2/s^2 ;
    D_at    : deformee modale en tout point (x, z) ;
    calibrate_frequencies : recale les frequences sur les valeurs mesurees
              (Tableau 4 : 540 / 1068 / 2787 / 3351 / 4122 Hz) sans toucher
              aux deformees — meme demarche que le papier qui mesure les
              amortissements et cale le modele avant synthese.
"""
import numpy as np
from numpy.polynomial import chebyshev as _C
from scipy.linalg import eigh


# ---------------------------------------------------------------------------
# Polynomes de Chebyshev de premiere espece : valeurs et derivees 0..2
# ---------------------------------------------------------------------------
def cheb_matrix(P, u, deriv=0):
    """(P, len(u)) : T_p^{(deriv)}(u) pour p = 0..P-1."""
    u = np.atleast_1d(np.asarray(u, float))
    out = np.zeros((P, u.size))
    for p in range(P):
        c = np.zeros(p + 1)
        c[p] = 1.0
        cd = _C.chebder(c, m=deriv) if deriv else c
        out[p, :] = _C.chebval(u, cd)
    return out


class ChebyshevPlate:
    """Plaque de Kirchhoff encastree-libre-libre-libre par Chebyshev-Ritz."""

    def __init__(self, lp=0.100, hp=0.080, bp=0.004,
                 rho=2830.0, E=69e9, nu=0.33,
                 PX=14, PZ=14, n_modes=5,
                 kw=1e12, kr=1e8, ngauss=64,
                 zeta_modes=(0.0031, 0.0017, 0.0027, 0.0056, 0.0035),
                 rotary=True, verbose=False):
        self.lp, self.hp, self.bp = lp, hp, bp
        self.rho, self.E, self.nu = rho, E, nu
        self.PX, self.PZ = PX, PZ
        self.n_modes = n_modes
        self.zeta_modes = np.asarray(zeta_modes[:n_modes], float)
        self.DP = E * bp**3 / (12.0 * (1.0 - nu**2))     # rigidite de flexion
        self.verbose = verbose
        self.rotary = rotary
        self._ngauss = ngauss
        self._assemble(kw, kr, ngauss)
        self._modal()

    # -- assemblage separable (la base est un produit tensoriel) ------------
    def _grams(self, P, ngauss):
        ug, wg = np.polynomial.legendre.leggauss(ngauss)
        B = [cheb_matrix(P, ug, d) for d in range(3)]
        G = {}
        for a in range(3):
            for b in range(3):
                G[(a, b)] = (B[a] * wg) @ B[b].T
        return G

    def _assemble(self, kw, kr, ngauss):
        lp, hp, nu = self.lp, self.hp, self.nu
        Gx = self._grams(self.PX, ngauss)
        Gz = self._grams(self.PZ, ngauss)
        cx, cz = 2.0 / lp, 2.0 / hp          # d/dx = cx d/dx~, d/dz = cz d/dz~
        Aj = (lp / 2.0) * (hp / 2.0)         # jacobien de surface

        # K_PD, Eq. (A.2)
        K = self.DP * Aj * (
            cx**4 * np.kron(Gx[(2, 2)], Gz[(0, 0)])
            + cz**4 * np.kron(Gx[(0, 0)], Gz[(2, 2)])
            + nu * cx**2 * cz**2 * (np.kron(Gx[(2, 0)], Gz[(0, 2)])
                                    + np.kron(Gx[(0, 2)], Gz[(2, 0)]))
            + 2.0 * (1.0 - nu) * cx**2 * cz**2 * np.kron(Gx[(1, 1)], Gz[(1, 1)])
        )

        # M_PD, Eq. (A.1) : I0 = rho b, I2 = rho b^3 / 12 (inertie de rotation)
        I0 = self.rho * self.bp
        I2 = self.rho * self.bp**3 / 12.0 if self.rotary else 0.0
        M = I0 * Aj * np.kron(Gx[(0, 0)], Gz[(0, 0)])
        if I2:
            M += I2 * Aj * (cx**2 * np.kron(Gx[(1, 1)], Gz[(0, 0)])
                            + cz**2 * np.kron(Gx[(0, 0)], Gz[(1, 1)]))

        # K_kappa, Eq. (A.4) : penalisation de l'encastrement z = 0 (z~ = -1)
        phi0 = cheb_matrix(self.PZ, -1.0, 0)[:, 0]
        phi1 = cheb_matrix(self.PZ, -1.0, 1)[:, 0]
        K += (lp / 2.0) * (kw * np.kron(Gx[(0, 0)], np.outer(phi0, phi0))
                           + kr * cz**2 * np.kron(Gx[(0, 0)],
                                                  np.outer(phi1, phi1)))
        self.K, self.M = 0.5 * (K + K.T), 0.5 * (M + M.T)

    def _modal(self):
        w2, V = eigh(self.K, self.M)
        keep = w2 > 1.0
        w2, V = w2[keep][:self.n_modes], V[:, keep][:, :self.n_modes]
        y_corner = self.Y_row(self.lp, self.hp)
        for k in range(self.n_modes):                     # masse-normalisation
            V[:, k] /= np.sqrt(V[:, k] @ self.M @ V[:, k])
            # signe deterministe : deplacement positif au coin superieur droit
            if y_corner @ V[:, k] < 0:
                V[:, k] = -V[:, k]
        self.V = V
        self.omega_n = np.sqrt(w2)
        self.freq_n = self.omega_n / (2 * np.pi)
        self.Kp = np.diag(self.omega_n**2)
        self.Cp = np.diag(2 * self.zeta_modes * self.omega_n)
        self.Mp = np.eye(self.n_modes)
        if self.verbose:
            print("frequences Chebyshev-Ritz (Hz) :",
                  np.round(self.freq_n, 1))

    # -- patch piezoelectrique (Section 3 du papier, Eqs. 14-15) -------------
    def _sub_grams(self, P, u1, u2, ngauss=48):
        """Grams 1D sur le sous-intervalle [u1, u2] de [-1, 1], et vecteurs
        v_d = int_{u1}^{u2} T^{(d)} du (d = 0, 1, 2)."""
        ug, wg = np.polynomial.legendre.leggauss(ngauss)
        um = 0.5 * (u1 + u2) + 0.5 * (u2 - u1) * ug
        wm = 0.5 * (u2 - u1) * wg
        B = [cheb_matrix(P, um, d) for d in range(3)]
        G = {(a, b): (B[a] * wm) @ B[b].T for a in range(3) for b in range(3)}
        v = [B[d] @ wm for d in range(3)]
        return G, v

    @staticmethod
    def shear_lag_efficiency(hx, hz, E_b, nu_b, h_b, E_p, nu_p, h_p,
                             G_adh, t_adh, alpha=6.0):
        """Rendement de transfert d'un patch colle (Crawley & de Luis 1987) :
        eta(G l) = 1 - tanh(G l)/(G l) par demi-dimension, forme separable."""
        Eb = E_b / (1.0 - nu_b**2)
        Ep = E_p / (1.0 - nu_p**2)
        Gam = np.sqrt((G_adh / t_adh) * (1.0 / (Ep * h_p) + alpha / (Eb * h_b)))
        f = lambda x: 1.0 - np.tanh(x) / x if x > 1e-12 else 0.0
        return float(f(Gam * hx) * f(Gam * hz)), float(Gam)

    def add_piezo_patch(self, x1=0.040, x2=0.100, z1=0.0, z2=0.020,
                        d31=175e-12, hPa=0.7e-3, EPe=63e9, nuPe=0.35,
                        rhoPe=7450.0, G_adh=1.0e9, t_adh=30e-6,
                        structural=True):
        """Colle le patch QDA60-20-0.7 (Tableau 2) :

        Geometrie par defaut : coin bas DROIT, 60 mm le long de x, 20 mm le
        long de z. Le papier ne dit pas comment la pastille est orientee et se
        contredit sur le cote ; c'est la signature des zeros de la Fig. 12(b)
        qui tranche — voir verification/19_patch_orientation.py, ou cette
        geometrie est la SEULE des quatre admissibles a reproduire
        l'occupation mesuree (1, 1, 0, 1).

        1. raidissement/masse composites ajoutes a K, M sur le rectangle du
           patch (rapports rK, rM de la section transformee, reduits par le
           rendement de collage eta) puis re-analyse modale ;
        2. couplage H_Pe = m_pz * int_patch lap(Y) dA — c'est exactement
           l'Eq. (14) du papier ecrite par le theoreme de la divergence —
           avec le moment par volt m_pz = -eta E_Pe d31 (b_P + h_Pa)/(2(1-nu_Pe)).
           Le coefficient C_P0 de l'Eq. (15) est aussi calcule et stocke
           (self.CP0) a titre de reference papier, avec les deux signes moins
           de l'equation imprimee.
        """
        eta, Gam = (1.0, np.inf) if G_adh is None else \
            self.shear_lag_efficiency(0.5 * (x2 - x1), 0.5 * (z2 - z1),
                                      self.E, self.nu, self.bp,
                                      EPe, nuPe, hPa, G_adh, t_adh)
        self.eta_bond = eta

        # C_P0, Eq. (15) du papier — VALEUR DE REFERENCE, non utilisee par le
        # couplage (H_Pe est construit a partir de m_pz plus bas).
        #
        # LES DEUX SIGNES MOINS SONT CEUX DE L'ARTICLE. Ce commentaire a
        # longtemps dit le contraire : il presentait le signe devant P_M comme
        # un « ecart assume par rapport a l'equation imprimee » et concluait
        # que l'Eq. (15) etait « tres probablement mal transcrite dans le
        # papier ». C'etait faux, et l'erreur venait d'ici : la couche texte du
        # PDF rend le glyphe moins par une espace, la transcription de
        # verification/15_piezo_coupling_eq14_15.py l'avait donc perdu, et la
        # lecture P_M > 0 qui en resultait etait effectivement pathologique
        # (denominateur a 5.6 % d'un pole, sensibilite +18.7, 1.2 N par volt
        # pour une pastille PZT de 0.7 mm). La page est rendue en image dans
        # verification/figures/eq15_imprimee.png : les deux signes s'y lisent.
        #
        # Le code n'a jamais bouge et n'avait pas a bouger : l'Eq. (15) telle
        # qu'imprimee donne -C_P0 d31/hPa = -0.033387 N/V, soit exactement ce
        # que calcule la ligne ci-dessous, et 5.8 % du moment equivalent
        # classique avec collage reel. Seule la description etait fausse.
        PM = -(EPe / self.E) * ((1 - self.nu**2) / (1 - nuPe**2)) \
            * (3 * hPa * self.bp * (self.bp + hPa)) \
            / (0.5 * self.bp**3 + 4 * hPa**3 + 3 * self.bp * hPa**2)
        self.CP0 = -(1.0 / 6.0) * ((1 + nuPe) / (1 - self.nu)) \
            * (self.E * self.bp**2 * PM) / (1 + self.nu - (1 + nuPe) * PM)

        xt1, xt2 = 2 * x1 / self.lp - 1, 2 * x2 / self.lp - 1
        zt1, zt2 = 2 * z1 / self.hp - 1, 2 * z2 / self.hp - 1
        Gxp, vx = self._sub_grams(self.PX, xt1, xt2)
        Gzp, vz = self._sub_grams(self.PZ, zt1, zt2)
        cx, cz = 2.0 / self.lp, 2.0 / self.hp
        Aj = (self.lp / 2.0) * (self.hp / 2.0)

        if structural:
            # rapports de section transformee (identiques a milling_plant.py)
            E1 = self.E / (1 - self.nu**2)
            E2 = EPe / (1 - nuPe**2)
            h1, h2 = self.bp, hPa
            zb = (E2 * h2 * (h1 + h2) / 2) / (E1 * h1 + E2 * h2)
            D_comp = (E1 * (h1**3 / 12 + h1 * zb**2)
                      + E2 * (h2**3 / 12 + h2 * ((h1 + h2) / 2 - zb)**2))
            D_plate = E1 * h1**3 / 12
            rK_free = (E2 * h2**3 / 12) / D_plate
            rK_perfect = D_comp / D_plate - 1.0
            rK = rK_free + eta * (rK_perfect - rK_free)
            rM = (rhoPe * hPa) / (self.rho * self.bp)
            self.rK_patch, self.rM_patch = rK, rM

            dK = rK * self.DP * Aj * (
                cx**4 * np.kron(Gxp[(2, 2)], Gzp[(0, 0)])
                + cz**4 * np.kron(Gxp[(0, 0)], Gzp[(2, 2)])
                + self.nu * cx**2 * cz**2 * (np.kron(Gxp[(2, 0)], Gzp[(0, 2)])
                                             + np.kron(Gxp[(0, 2)], Gzp[(2, 0)]))
                + 2 * (1 - self.nu) * cx**2 * cz**2
                * np.kron(Gxp[(1, 1)], Gzp[(1, 1)]))
            I0p = rM * self.rho * self.bp
            dM = I0p * Aj * np.kron(Gxp[(0, 0)], Gzp[(0, 0)])
            self.K = self.K + dK
            self.M = self.M + dM
            self._modal()

        # H_Pe = m_pz * int_patch (Y_xx + Y_zz) dA  (equiv. Eq. 14)
        m_pz = -eta * EPe * d31 * (self.bp + hPa) / (2 * (1 - nuPe))
        lap = Aj * (cx**2 * np.kron(vx[2], vz[0])
                    + cz**2 * np.kron(vx[0], vz[2]))
        self.H_Pe_modal = m_pz * (lap @ self.V)
        self.m_piezo = m_pz
        self.patch = dict(x1=x1, x2=x2, z1=z1, z2=z2, hPa=hPa,
                          EPe=EPe, nuPe=nuPe, d31=d31)
        return self.H_Pe_modal

    def set_observation(self, x_obs, z_obs):
        self.D_obs = self.D_row(x_obs, z_obs)
        self.x_obs, self.z_obs = x_obs, z_obs
        return self.D_obs

    # -- evaluation ----------------------------------------------------------
    def Y_row(self, x, z):
        """Vecteur ligne Y_P(x~, z~), Eq. (8)."""
        xt = 2.0 * x / self.lp - 1.0
        zt = 2.0 * z / self.hp - 1.0
        return np.kron(cheb_matrix(self.PX, xt, 0)[:, 0],
                       cheb_matrix(self.PZ, zt, 0)[:, 0])

    def D_row(self, x, z):
        """Deformee modale au point (x, z) : D_P = Y_P U_P (n_modes,)."""
        return self.Y_row(x, z) @ self.V if hasattr(self, 'V') \
            else self.Y_row(x, z)

    def D_top_edge(self, n_pos=201):
        """D_P le long du bord superieur z = h_P (positions 0..l_P)."""
        xs = np.linspace(0.0, self.lp, n_pos)
        D = np.array([self.Y_row(x, self.hp) @ self.V for x in xs])
        return xs, D                                     # (n_pos, n_modes)

    def calibrate_frequencies(self, f_targets_hz):
        """Recale les frequences sur les valeurs mesurees, deformees intactes."""
        f = np.asarray(f_targets_hz, float)[:self.n_modes]
        self.freq_scale = f / self.freq_n[:len(f)]
        self.omega_n = 2 * np.pi * f
        self.freq_n = f.copy()
        self.Kp = np.diag(self.omega_n**2)
        self.Cp = np.diag(2 * self.zeta_modes * self.omega_n)


if __name__ == "__main__":
    ref_theo = [537, 1101, 2805, 3423, 4254]
    ref_meas = [540, 1068, 2787, 3351, 4122]
    for (kw, kr) in [(1e10, 1e6), (1e11, 1e7), (1e12, 1e8), (1e13, 1e9)]:
        p = ChebyshevPlate(kw=kw, kr=kr)
        err = 100 * (p.freq_n / np.array(ref_theo) - 1)
        print(f"kw={kw:.0e} kr={kr:.0e} :",
              np.round(p.freq_n, 1), " ecart% vs theo:", np.round(err, 2))
