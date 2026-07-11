"""
von_karman_rom.py
=================
Modèle réduit GÉOMÉTRIQUEMENT NON-LINÉAIRE (von Kármán) de la plaque mince.

────────────────────────────────────────────────────────────────────────
   Verrou traité : le contrôle actif des vibrations en fraisage des
   pièces flexibles s'appuie EXCLUSIVEMENT sur des modèles LINÉAIRES
   (Kirchhoff/Mindlin + réduction modale linéaire).  Or, en régime
   agressif / proche du broutement, l'amplitude de vibration de la paroi
   mince devient une fraction non négligeable de son épaisseur : la
   non-linéarité géométrique de von Kármán (couplage membrane-flexion)
   raidit la structure, BORNE la croissance du broutement en un CYCLE
   LIMITE (bifurcation de Hopf) que le modèle linéaire — qui prédit une
   divergence exponentielle — ne peut décrire.
────────────────────────────────────────────────────────────────────────

Méthode (cohérente EF, dans l'esprit des coques intelligentes non
linéaires de Mallek-Jrad-Wali-Dammak 2019) :

  * la flexion transverse w reste décrite par l'élément de Kirchhoff Q4
    existant (modes linéaires φ_r, pulsations ω_r) ;
  * on AJOUTE un élément membranaire Q4 (contrainte plane, ddl u,v) sur
    le MÊME maillage, encastré en membrane sur le bord inférieur
    (cohérent avec l'encastrement transverse du cantilever) ;
  * la déformation membranaire de von Kármán  θ = { ½w,x² , ½w,y² ,
    w,x w,y }  induite par w charge le problème membranaire ; on condense
    statiquement les ddl membranaires (inertie dans le plan négligée) :
        K_m a = −f_θ(w)         (a quadratique en w)
        N     = D_m (B_m a + θ) (résultantes de contrainte, quadr. en w)
        f_w   = ∫ G(w)ᵀ N dA     (force transverse non linéaire, CUBIQUE)
  * projection sur les modes → tenseur de raideur cubique modal Γ :
        F_nl,r(q) = Σ_ijk Γ_rijk q_i q_j q_k .

Le modèle réduit non linéaire s'écrit alors (coordonnées modales
normées en masse) :
        q̈_r + 2ζ_rω_r q̇_r + ω_r² q_r + Σ_ijk Γ_rijk q_i q_j q_k = F_ext,r .

Aucune hypothèse de matériau : AL6061 homogène (le tenseur Γ dépend
linéairement de E et cubiquement des pentes modales).
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from kirchhoff_q4 import shape_function_K


# ====================================================================
#  Élément membranaire Q4 bilinéaire (contrainte plane)
# ====================================================================
def _membrane_dN(xi, eta, lex, ley):
    """Dérivées physiques des fonctions bilinéaires : dN/dx, dN/dy (4,)."""
    dNdxi = np.array([-(1-eta),  (1-eta),  (1+eta), -(1+eta)]) / 4.0
    dNdet = np.array([-(1-xi),  -(1+xi),   (1+xi),   (1-xi)]) / 4.0
    dNdx = dNdxi * (2.0/lex)
    dNdy = dNdet * (2.0/ley)
    return dNdx, dNdy


def _membrane_B(xi, eta, lex, ley):
    """Matrice B membranaire (3x8) dans l'ordre nodal [n1,n2,n3,n4]."""
    dNdx, dNdy = _membrane_dN(xi, eta, lex, ley)
    B = np.zeros((3, 8))
    for a in range(4):
        B[0, 2*a]   = dNdx[a]
        B[1, 2*a+1] = dNdy[a]
        B[2, 2*a]   = dNdy[a]
        B[2, 2*a+1] = dNdx[a]
    return B


# ====================================================================
#  Modèle réduit von Kármán
# ====================================================================
class VonKarmanROM:
    """
    Construit le tenseur cubique Γ pour un PlateModel donné, et fournit
    la force modale non linéaire F_nl(q) et sa jacobienne.
    """

    # Gauss 3x3
    _GP = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    _GW = np.array([5/9, 8/9, 5/9])

    def __init__(self, plate, membrane_bc="cantilever", verbose=True):
        self.plate = plate
        self.n_modes = plate.n_modes
        self.verbose = verbose
        self.membrane_bc = membrane_bc

        self.N1, self.N2 = plate.N1, plate.N2
        self.n1, self.n2 = plate.n1, plate.n2
        self.lex, self.ley = plate.lex, plate.ley
        self.h = plate.bp
        self.detJ = self.lex*self.ley/4.0

        # membrane constitutive (plane stress) × thickness
        E, nu = plate.E, plate.nu
        self.D_m = E*self.h/(1-nu**2) * np.array([[1.0, nu,  0.0],
                                                  [nu,  1.0, 0.0],
                                                  [0.0, 0.0, (1-nu)/2]])

        self.n_node = self.n1*self.n2
        self.n_mdof = 2*self.n_node

        # full transverse mode shapes (nodal, BC dofs = 0)
        self.Phi_full = np.zeros((plate.ndof, self.n_modes))
        self.Phi_full[plate.DOFf, :] = plate.V
        # transverse (w) dofs only — index %3==0 (the others are slopes
        # w,x / w,y which are numerically larger and must NOT be used to
        # gauge the physical deflection amplitude)
        self.w_dofs = np.arange(0, plate.ndof, 3)

        self._precompute_element_bending_ops()
        self._assemble_membrane_stiffness()
        self._extract_cubic_tensor()

    # ----------------------------------------------------------------
    def _node_id(self, I, J):
        """0-indexed membrane node id for 1-indexed grid node (I,J)."""
        return (J-1)*self.n1 + (I-1)

    def _elem_nodes(self, I, J):
        """4 nodes of element (I,J) in local order matching Kirchhoff."""
        return [(I, J), (I+1, J), (I+1, J+1), (I, J+1)]

    def _bending_dofe(self, I, J):
        n1 = self.n1
        return np.r_[3*(J-1)*n1 + 3*(I-1) + np.arange(6),
                     3*J*n1     + 3*(I-1) + np.arange(3, 6),
                     3*J*n1     + 3*(I-1) + np.arange(3)]

    def _membrane_dofe(self, I, J):
        dofe = np.zeros(8, dtype=int)
        for a, (Ia, Ja) in enumerate(self._elem_nodes(I, J)):
            nid = self._node_id(Ia, Ja)
            dofe[2*a] = 2*nid
            dofe[2*a+1] = 2*nid+1
        return dofe

    # ----------------------------------------------------------------
    def _precompute_element_bending_ops(self):
        """Per Gauss point : bx,by (12,) mapping bending dofs → w,x / w,y."""
        self._gp_list = []
        for ig in range(3):
            for jg in range(3):
                xi, eta = self._GP[ig], self._GP[jg]
                wgt = self._GW[ig]*self._GW[jg]
                _, derN = shape_function_K(xi, eta, self.lex, self.ley)
                bx = derN[0, :]*(2.0/self.lex)     # ∂w/∂x = bx·w_e
                by = derN[1, :]*(2.0/self.ley)     # ∂w/∂y = by·w_e
                Bm = _membrane_B(xi, eta, self.lex, self.ley)
                self._gp_list.append((wgt, bx, by, Bm))

    # ----------------------------------------------------------------
    def _assemble_membrane_stiffness(self):
        if self.verbose:
            print("[vK-ROM] Assemblage raideur membranaire...")
        Km = np.zeros((self.n_mdof, self.n_mdof))
        # element membrane stiffness (identical for all rectangular elems)
        Ke = np.zeros((8, 8))
        for (wgt, bx, by, Bm) in self._gp_list:
            Ke += wgt * (Bm.T @ self.D_m @ Bm) * self.detJ
        for J in range(1, self.N2+1):
            for I in range(1, self.N1+1):
                dofe = self._membrane_dofe(I, J)
                Km[np.ix_(dofe, dofe)] += Ke

        # in-plane boundary conditions
        fixed = set()
        for I in range(1, self.n1+1):
            nid = self._node_id(I, 1)               # bottom edge clamped
            fixed.add(2*nid); fixed.add(2*nid+1)
        if self.membrane_bc == "immovable":
            # all four edges immovable in-plane (clamped-plate validation)
            for I in range(1, self.n1+1):
                for J in (1, self.n2):
                    nid = self._node_id(I, J)
                    fixed.update((2*nid, 2*nid+1))
            for J in range(1, self.n2+1):
                for I in (1, self.n1):
                    nid = self._node_id(I, J)
                    fixed.update((2*nid, 2*nid+1))
        elif self.membrane_bc == "wall":
            # bottom clamped + two VERTICAL side edges in-plane restrained,
            # top edge free — models a thin rib/wall flanked by thicker
            # stock (typical aerospace pocket wall).  Membrane can no longer
            # relax sideways ⇒ strong von Kármán stretching (hardening).
            for J in range(1, self.n2+1):
                for I in (1, self.n1):
                    nid = self._node_id(I, J)
                    fixed.update((2*nid, 2*nid+1))
        self.m_fixed = np.array(sorted(fixed), dtype=int)
        self.m_free = np.array([d for d in range(self.n_mdof)
                                if d not in fixed], dtype=int)
        Kmff = Km[np.ix_(self.m_free, self.m_free)]
        self.Km_full = Km
        self._Km_lu = lu_factor(Kmff)
        if self.verbose:
            print(f"[vK-ROM] membrane dofs: {self.n_mdof} "
                  f"({len(self.m_free)} libres), BC={self.membrane_bc}")

    # ----------------------------------------------------------------
    def _membrane_solve(self, f_theta):
        """Solve K_m a = −f_theta with the in-plane BC. Returns full a."""
        a = np.zeros(self.n_mdof)
        a[self.m_free] = lu_solve(self._Km_lu, -f_theta[self.m_free])
        return a

    # ----------------------------------------------------------------
    def nonlinear_force_physical(self, W_full):
        """
        Force transverse non linéaire (pleine, ndof) pour un champ
        transverse nodal W_full.  Exactement cubique en W.
        """
        f_theta = np.zeros(self.n_mdof)
        # pass 1 : assemble membrane load f_theta(w)
        gp = self._gp_list
        elem_cache = []
        for J in range(1, self.N2+1):
            for I in range(1, self.N1+1):
                bdof = self._bending_dofe(I, J)
                mdof = self._membrane_dofe(I, J)
                w_e = W_full[bdof]
                fte = np.zeros(8)
                per_gp = []
                for (wgt, bx, by, Bm) in gp:
                    wx = bx @ w_e
                    wy = by @ w_e
                    theta = np.array([0.5*wx*wx, 0.5*wy*wy, wx*wy])
                    fte += wgt*(Bm.T @ (self.D_m @ theta))*self.detJ
                    per_gp.append((wgt, bx, by, Bm, wx, wy, theta))
                f_theta[mdof] += fte
                elem_cache.append((I, J, bdof, mdof, w_e, per_gp))

        a = self._membrane_solve(f_theta)

        # pass 2 : transverse nonlinear force f_w
        f_w = np.zeros(self.plate.ndof)
        for (I, J, bdof, mdof, w_e, per_gp) in elem_cache:
            a_e = a[mdof]
            fwe = np.zeros(12)
            for (wgt, bx, by, Bm, wx, wy, theta) in per_gp:
                eps_L = Bm @ a_e
                N = self.D_m @ (eps_L + theta)          # (3,)
                # G^T N with G = [[wx*bx],[wy*by],[wx*by+wy*bx]]
                GT_N = (N[0]*wx)*bx + (N[1]*wy)*by + N[2]*(wx*by + wy*bx)
                fwe += wgt*GT_N*self.detJ
            f_w[bdof] += fwe
        return f_w

    # ----------------------------------------------------------------
    def _modal_force(self, q):
        """Physical assembly path → modal nonlinear force (n_modes,)."""
        W = self.Phi_full @ q
        f_w = self.nonlinear_force_physical(W)
        return self.Phi_full.T @ f_w

    # ----------------------------------------------------------------
    def _extract_cubic_tensor(self):
        """
        F_nl,r(q) is an exact homogeneous cubic form.  Fit the 10 cubic
        monomial coefficients per output by least squares over random q.
        The fit residual (printed) validates cubic-exactness ⇒ correctness.
        """
        if self.verbose:
            print("[vK-ROM] Extraction du tenseur cubique Γ...")
        n = self.n_modes
        # cubic monomials index list (i<=j<=k)
        monos = []
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    monos.append((i, j, k))
        self._monos = monos
        n_mono = len(monos)

        rng = np.random.default_rng(0)
        n_s = max(3*n_mono, 40)
        # amplitude scale: modal coords giving ~ a thickness of tip motion
        phi_w_max = np.max(np.abs(self.Phi_full[self.w_dofs, :]))
        q_scale = self.h / max(phi_w_max, 1e-30)
        Q = rng.normal(size=(n_s, n)) * q_scale
        M = np.zeros((n_s, n_mono))
        F = np.zeros((n_s, n))
        for s in range(n_s):
            F[s] = self._modal_force(Q[s])
            for m, (i, j, k) in enumerate(monos):
                M[s, m] = Q[s, i]*Q[s, j]*Q[s, k]
        C, *_ = np.linalg.lstsq(M, F, rcond=None)   # (n_mono, n)
        resid = np.linalg.norm(M @ C - F) / max(np.linalg.norm(F), 1e-30)
        self.coeff = C.T                              # (n_out, n_mono)
        self.cubic_residual = float(resid)
        if self.verbose:
            print(f"[vK-ROM] résidu cubique relatif = {resid:.2e} "
                  f"(→ 0 confirme la forme cubique exacte)")

    # ----------------------------------------------------------------
    def force(self, q):
        """Modal nonlinear force F_nl(q) (n_modes,) — fast tensor form."""
        mono = np.array([q[i]*q[j]*q[k] for (i, j, k) in self._monos])
        return self.coeff @ mono

    def jacobian(self, q):
        """dF_nl/dq (n_modes, n_modes) — analytic."""
        n = self.n_modes
        dmono = np.zeros((len(self._monos), n))
        for m, (i, j, k) in enumerate(self._monos):
            # d/dq_s of q_i q_j q_k
            for s in range(n):
                t = 0.0
                if s == i: t += q[j]*q[k]
                if s == j: t += q[i]*q[k]
                if s == k: t += q[i]*q[j]
                dmono[m, s] = t
        return self.coeff @ dmono

    # ----------------------------------------------------------------
    def backbone_single_mode(self, mode=0, amp_ratio=None):
        """
        Rapport de fréquence non linéaire ω_nl/ω_l pour un mode isolé,
        par la méthode du bilan harmonique à 1 terme (Duffing) :
            ω_nl/ω_l = sqrt(1 + (3/4) (Γ_rrrr / ω_r²) A_q²)
        où A_q est l'amplitude modale correspondant à une flèche crête
        A_w = amp_ratio · h au point d'amplitude maximale du mode.
        """
        if amp_ratio is None:
            amp_ratio = np.linspace(0, 1.0, 11)
        amp_ratio = np.atleast_1d(amp_ratio)
        r = mode
        # self-cubic coefficient Γ_rrrr
        idx = self._monos.index((r, r, r))
        gamma_rrrr = self.coeff[r, idx]
        omega_r = self.plate.omega_n[r]
        phi_max = np.max(np.abs(self.Phi_full[self.w_dofs, r]))   # w dofs only
        A_q = (amp_ratio*self.h)/phi_max
        ratio = np.sqrt(np.maximum(
            1.0 + 0.75*gamma_rrrr/omega_r**2 * A_q**2, 0.0))
        return amp_ratio, ratio, gamma_rrrr
