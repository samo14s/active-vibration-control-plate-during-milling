"""
milling_force.py
================
Forces de coupe de fraisage — MODÉLISATION NON LINÉAIRE
(Nasiri & Moradi, MSSP 224 (2025) 112198, Eqs. 3-6, 16-17, 39).

L'ancien modèle linéarisé (coefficients alpha3/alpha4 de Lehmann-Engin) est
CONSERVÉ comme ossature linéaire (valeurs principales du package, K_T = 925 MPa)
et ÉTENDU à la forme polynomiale non linéaire de l'article :

   dF_t(φ) = [ξ1 h³ + ξ2 h² + ξ3 h + ξ4] dz          (Eq. 4, tangentiel)
   dF_r(φ) = [δ1 h³ + δ2 h² + δ3 h + δ4] dz          (Eq. 4, radial)

avec l'épaisseur de copeau régénérative (Eq. 5) :
   h(φ) = [c_f sinφ + (v - v0)] · g(φ)

La force dynamique (régénérative) qui pilote le chatter est un polynôme de la
variation de déplacement régénératif Δ = w(outil,t) - w(outil,t-τ) :

   F_dyn(Δ) = -( a4·Δ + a4_2·Δ² + a4_3·Δ³ )

où  a4   = coefficient régénératif LINÉAIRE du package (conservé, ∝ K_T),
    a4_2 = a4 · (ξ2/ξ3)   (terme quadratique, ratio article Table 2),
    a4_3 = a4 · (ξ1/ξ3)   (terme cubique,    ratio article Table 2).

Aux amplitudes nominales (µm), les termes Δ², Δ³ sont négligeables → le modèle
se réduit à l'ossature linéaire du package (résultats principaux conservés).
Aux grandes amplitudes (chatter mm), la non-linéarité de coupe s'active comme
dans l'article, et la SÉPARATION outil-pièce (Eq. 6, §4.1) borne la force.
"""
import numpy as np


# ===========================================================================
#  Coefficients de force de coupe non linéaires de l'article (Table 2)
#  Convertis en SI :  h en mètres.
#     ξ1 = 6765 N/mm³ = 6.765e12 N/m³  ;  ξ2 = -4910 N/mm² = -4.91e9 N/m²
#     ξ3 = 2840 N/mm  = 2.84e6  N/m    ;  ξ4 = 132 N
#     δ1 = 12740 N/mm³ ; δ2 = -7452 N/mm² ; δ3 = 1674 N/mm ; δ4 = 246 N
# ===========================================================================
XI = dict(xi1=6.765e12, xi2=-4.910e9, xi3=2.840e6, xi4=132.0)
DELTA = dict(d1=1.274e13, d2=-7.452e9, d3=1.674e6, d4=246.0)

# Ratios de forme de la non-linéarité de coupe (tangentielle), en SI
RATIO_QUAD = XI['xi2'] / XI['xi3']          #  ξ2/ξ3  [1/m]   ≈ -1729
RATIO_CUB = XI['xi1'] / XI['xi3']           #  ξ1/ξ3  [1/m²]  ≈  2.382e6


# ===========================================================================
#  Ossature LINÉAIRE — coefficients alpha3 (forçage moyen) / alpha4
#  (régénératif linéaire).  STRICTEMENT CONSERVÉE (compat descendante).
# ===========================================================================
def milling_force_coeffs(t: float,
                         Omega: float, NT: int, RT: float, eta: float,
                         phi_st: float, phi_ex: float,
                         za_low: float, za_high: float,
                         k1: float, k2: float, kt: float):
    """
    Calcule (alpha3, alpha4) — coefficients de force de fraisage à l'instant t.
    Intègre les fonctions trigonométriques sur la portion engagée [phi_st,
    phi_ex] et la profondeur axiale [za_low, za_high] (Eqs. 3-4 linéarisées).
    """
    ss = 0.0
    sc = 0.0
    cc = 0.0
    te = np.tan(eta)

    for j in range(NT):
        theta0    = Omega*t + 2*np.pi*j/NT
        theta_max = theta0 - za_low *te/RT
        theta_min = theta0 - za_high*te/RT

        k_min = int(np.floor((theta_min - phi_ex)/(2*np.pi)))
        k_max = int(np.ceil ((theta_max - phi_st)/(2*np.pi)))

        for kk in range(k_min, k_max + 1):
            phi_lo = phi_st + 2*np.pi*kk
            phi_hi = phi_ex + 2*np.pi*kk

            th_lo = max(theta_min, phi_lo)
            th_hi = min(theta_max, phi_hi)

            if th_hi <= th_lo:
                continue

            z2 = (theta0 - th_lo)*RT/te
            z1 = (theta0 - th_hi)*RT/te
            z1 = max(z1, za_low)
            z2 = min(z2, za_high)

            if z2 <= z1:
                continue

            theta1 = theta0 - z1*te/RT
            theta2 = theta0 - z2*te/RT

            ss += (z2 - z1)/2 \
                + RT/(2*te) * np.sin(theta2 - theta1) * np.cos(theta2 + theta1)
            sc += RT/(2*te) * np.sin(theta1 - theta2) * np.sin(theta1 + theta2)
            cc += (z2 - z1)/2 \
                - RT/(2*te) * np.sin(theta2 - theta1) * np.cos(theta2 + theta1)

    alpha3 =  k2*kt*ss - k1*kt*sc
    alpha4 =  k2*kt*sc - k1*kt*cc
    return alpha3, alpha4


def precompute_alpha_periodic(dt: float, n_per: int, n_total: int,
                              Omega: float, NT: int, RT: float, eta: float,
                              phi_st: float, phi_ex: float,
                              za_low: float, za_high: float,
                              k1: float, k2: float, kt: float):
    """
    Calcule alpha3(t), alpha4(t) sur une période tau (n_per pas) puis duplique.
    Exploite la périodicité tau = 60/(NT*RPM) pour gros gain en vitesse.
    (Ossature LINÉAIRE conservée — valeurs principales du package.)
    """
    alpha3_per = np.zeros(n_per)
    alpha4_per = np.zeros(n_per)
    for k in range(n_per):
        alpha3_per[k], alpha4_per[k] = milling_force_coeffs(
            k*dt, Omega, NT, RT, eta,
            phi_st, phi_ex, za_low, za_high, k1, k2, kt
        )

    nrep = n_total // n_per + 2
    alpha3_t = np.tile(alpha3_per, nrep)[:n_total]
    alpha4_t = np.tile(alpha4_per, nrep)[:n_total]
    return alpha3_t, alpha4_t


# ===========================================================================
#  Extension NON LINÉAIRE de l'article
# ===========================================================================
def nonlinear_regen_coeffs(alpha4_t, ratio_quad=RATIO_QUAD, ratio_cub=RATIO_CUB):
    """
    Construit les coefficients régénératifs quadratique et cubique à partir du
    coefficient linéaire conservé alpha4 et des ratios de l'article (Table 2) :
        a4_2 = alpha4 · (ξ2/ξ3)   [N/m²  →  force ∝ Δ²]
        a4_3 = alpha4 · (ξ1/ξ3)   [N/m³  →  force ∝ Δ³]
    """
    a4 = np.asarray(alpha4_t)
    return a4 * ratio_quad, a4 * ratio_cub


def precompute_nonlinear_periodic(dt, n_per, n_total,
                                  Omega, NT, RT, eta, phi_st, phi_ex,
                                  za_low, za_high, k1, k2, kt,
                                  ratio_quad=RATIO_QUAD, ratio_cub=RATIO_CUB):
    """
    Renvoie (alpha3_t, alpha4_t, alpha4_2_t, alpha4_3_t) :
      - alpha3_t, alpha4_t  : ossature linéaire (identique à
        precompute_alpha_periodic, valeurs principales conservées) ;
      - alpha4_2_t, alpha4_3_t : coefficients quadratique/cubique de la coupe
        non linéaire (article Eq. 4/16, Table 2).
    """
    alpha3_t, alpha4_t = precompute_alpha_periodic(
        dt, n_per, n_total, Omega, NT, RT, eta, phi_st, phi_ex,
        za_low, za_high, k1, k2, kt)
    alpha4_2_t, alpha4_3_t = nonlinear_regen_coeffs(alpha4_t,
                                                    ratio_quad, ratio_cub)
    return alpha3_t, alpha4_t, alpha4_2_t, alpha4_3_t


def tool_wear_edge_force(VB: float, mu: float, H_brinell: float, ac: float,
                         kt: float = None, kte_ref: float = XI['xi4']):
    """
    Terme de bord dû au BLEU/usure en dépouille (flank wear, Eq. 39) :
        F_edge = mu · ac · H_brinell · VB        (∝ largeur d'usure VB)
    équivalent à ξ4/δ4 dans l'article (force statique de bord).  Renvoie une
    force constante [N] ajoutée au forçage moyen.  VB=0 → pas d'usure (nominal
    du package conservé).  Si les données (mu,H) sont absentes, on utilise la
    valeur de référence ξ4 de l'article mise à l'échelle par VB/VB_ref.
    """
    if VB <= 0.0:
        return 0.0
    if mu and H_brinell:
        return mu * ac * H_brinell * VB
    # repli : mise à l'échelle de ξ4 (réf. VB=0.15 mm de l'article, Table 2)
    return kte_ref * (VB / 0.15e-3)


def chip_separation_factor(delta, w_tool, h_sep, w_sep):
    """
    Facteur de séparation outil-pièce g ∈ {0,1} (Eq. 6, §4.1) :
      - séparation si la variation régénérative rend l'épaisseur de copeau
        négative au-delà de l'avance (Δ < -h_sep) → la dent sort du copeau ;
      - séparation si l'amplitude normale dépasse l'engagement (|w| > w_sep).
    Renvoie 0 (hors coupe) ou 1 (en coupe).  Aux amplitudes nominales (µm,
    h_sep ≈ avance ~ 20 µm), g = 1 → ossature linéaire conservée.
    """
    if h_sep is not None and delta < -h_sep:
        return 0.0
    if w_sep is not None and abs(w_tool) > w_sep:
        return 0.0
    return 1.0
