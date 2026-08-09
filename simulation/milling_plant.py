"""
================================================================================
milling_plant.py — LE PROCEDE SEUL, EN UN SEUL FICHIER, SANS AUCUN CORRECTEUR
================================================================================
Plaque mince encastree + patch piezoelectrique colle + capteur + efforts de
coupe avec retard regeneratif. Rien d'autre. Aucune loi de commande n'est
ecrite ici, et aucune n'est presupposee : le point de branchement est unique,
documente en PARTIE 6, et accepte aussi bien une fonction d'une ligne qu'un
observateur complet.

Ce fichier est AUTONOME. Il ne depend que de numpy et scipy. Il regroupe, sans
les modifier numeriquement, le contenu de kirchhoff_q4.py, plate_model.py,
milling_force.py, newmark_solver.py et simulation_base.py. L'equivalence avec
cette base a cinq fichiers est verifiee a l'execution par
`verification/16_standalone_equivalence.py`.

--------------------------------------------------------------------------------
LE MODELE
--------------------------------------------------------------------------------
Fraisage peripherique d'une paroi mince, Eq. (13) de Du, Liu, Dai & Long,
"Robust combined time delay control for milling chatter suppression of flexible
workpieces", Int. J. Mech. Sci. 274 (2024) 109257 :

    M q" + C q' + (K + a4 D^T D) q(t) - a4 D^T D q(t-tau) = ft a3 D^T + H_Pe u

  q      coordonnees modales (5 modes)
  D(x)   deformee modale au point de contact outil/piece, qui SE DEPLACE
  tau    periode de passage de dent, 60/(N_teeth * rpm)
  a3, a4 coefficients d'effort de coupe, periodiques de periode tau
  H_Pe   vecteur de couplage piezoelectrique, en N/V
  u      tension appliquee au patch, en V   <-- C'EST LA SEULE ENTREE DE COMMANDE

Le terme en q(t-tau) est le couplage regeneratif : c'est lui qui rend le
broutement possible, et c'est lui que toute commande cherche a mater.

--------------------------------------------------------------------------------
CE QUI EST VALIDE, ET CE QUI NE L'EST PAS
--------------------------------------------------------------------------------
Le modele est cale sur deux courbes experimentales numerisees de l'article
(reception au choc, Fig. 12a ; transfert tension -> deplacement, Fig. 12b).
Ces courbes valident la FORME de la reponse, PAS son niveau.

VALIDE
  * cinq resonances mesurees   : 536.4 / 1071.8 / 2778.6 / 3358.7 / 4122.6 Hz,
    predites par le modele EF a +0.4 / +1.5 / -1.9 / +0.8 / +0.7 % pres, puis
    recalees exactement (calibrate_frequencies).
  * quatre antiresonances mesurees : 734 / 2161 / 3001 / 3805 Hz
    modele                         : 731 / 2123 / 3009 / 3938 Hz
    Un modele a TROIS modes ne donne que deux antiresonances, dont une fausse
    de 11 % : d'ou les cinq modes retenus.
  * la REPARTITION MODALE de H_Pe, par les quatre creux de la Fig. 12(b) :
      mesures : 788 / 1493 / 2913 / 3609 Hz
      modele  : 787 / 1495 / 2907 / 3613 Hz     (ecart max 0.19 %)
    Elle n'est pas PREDITE par les elements finis, elle y est IDENTIFIEE : le
    H_Pe elements finis ne produit qu'un seul creux, a 2818 Hz, et s'ecarte de
    14.3 dB RMS de la Fig. 12(b) contre 5.7 dB pour la version identifiee.
    Voir H_IDENT plus bas ; coupling='fem' restitue la version EF.
  * amortissements modaux mesures (Tableau 4) : 0.31 / 0.17 / 0.27 / 0.56 / 0.35 %.

NON VALIDE : LE NIVEAU ABSOLU, DONC LE GAIN ACTIONNEUR
  Le plateau basse frequence de la Fig. 12(a), lu avec la reference annoncee
  (1 um/N), vaut 43.8 um/N ; une plaque de Kirchhoff aux dimensions du Tableau 1
  donne 6.92 um/N. Facteur 6.34. Ce n'est PAS une erreur du modele : les memes
  K et M reproduisent les cinq frequences a 2 % pres, or la souplesse est K^-1 ;
  une raideur 6.34 fois trop faible donnerait f1 = 214 Hz, pas 540 Hz. Et les
  propres lobes de l'article (Fig. 13, 0.03-0.05 mm) s'accordent avec la valeur
  RAIDE. Conclusion : l'echelle en dB de la Fig. 12 est inexploitable, et comme
  H_Pe ne se calibre que par le rapport des deux courbes, SON NIVEAU RESTE NON
  VALIDE.

  Toute conclusion de performance en boucle fermee herite de cette incertitude
  et doit etre accompagnee d'un balayage :

      MillingPlant(gain_H=0.5) ... MillingPlant(gain_H=2.0)

  en SYNTHETISANT le correcteur sur la plaque nominale et en le SIMULANT sur la
  plaque perturbee. C'est un vrai essai d'erreur de modele, et il est severe :
  aucun des correcteurs essayes dans ce depot ne survivait a gain_H = 2 avant
  d'y etre explicitement contraint.

--------------------------------------------------------------------------------
CE QUI EST FIGE, ET POURQUOI
--------------------------------------------------------------------------------
  * dt = tau/82 EXACTEMENT. Le retard regeneratif doit tomber sur la grille
    d'echantillonnage, sinon q(t-tau) demande une interpolation qui fausse la
    stabilite. dt n'est donc PAS un reglage libre — et c'est aussi la periode
    d'echantillonnage imposee au correcteur.
  * cinq modes. Justifie par les antiresonances. La limite libre est identique
    a 1, 2, 3 ou 5 modes (les modes 4 et 5 sont fortement couples au patch mais
    quasiment pas excites par la coupe) : le nombre de modes ne change pas la
    frontiere libre, mais il change ce que l'observateur peut voir.
  * patch QDA60-20-0.7 VERTICAL 20 x 60 mm au coin inferieur gauche, capteur au
    coin superieur droit oppose (Fig. 11 de l'article).
  * patch COLLE, pas soude : epoxy G_adh = 1 GPa, t_adh = 30 um, d'ou
    1/Gamma = 0.874 mm et un rendement de transfert eta = 0.886 qui reduit A LA
    FOIS le raidissement composite (58.3 % -> 51.7 %) et H_Pe (-11.4 %). Un
    seul parametre physique, deux effets. G_adh=None restitue le collage
    parfait.
  * horizon T = 0.60 s et critere sigma <= 0.05 /s (voir CRITERE DE STABILITE).

--------------------------------------------------------------------------------
CRITERE DE STABILITE — A LIRE AVANT DE COMPARER QUOI QUE CE SOIT
--------------------------------------------------------------------------------
Une coupe est declaree stable si le TAUX DE CROISSANCE EXPONENTIEL sigma de la
reponse, mesure sur la DERNIERE MOITIE d'un releve de 0.60 s, ne depasse pas
SIGMA_MAX = 0.05 /s.

Ce n'est pas un detail d'implementation, c'est la definition de la grandeur
publiee. Le critere precedent — rapport des RMS entre les deux moities de la
fenetre — etait AVEUGLE a un mode qui croit lentement sous une reponse forcee
plus grande que lui : le rapport reste proche de 1 tant que le mode instable
n'a pas rattrape le regime force. Sur 144 cas dont la verite a ete etablie a
T = 1.60 s, il se trompait 8 fois et TOUJOURS dans le sens dangereux, en
declarant stable ce qui diverge ensuite. Le critere sur sigma se trompe 0 fois
sur les memes 144 cas.

Deux corrections etaient necessaires, aucune ne suffit seule :
  1. sigma se mesure sur la DERNIERE MOITIE. La reponse forcee d'une plaque a
     zeta = 0.17 % met ~0.3 s a s'etablir, et tant qu'elle monte, une droite
     ajustee sur log(RMS) lit cette montee comme une croissance : sigma vaut
     ~1 /s meme en boucle ouverte parfaitement stable.
  2. UN SEUL horizon, 0.60 s. Le verdict DEPEND de l'horizon ; deux horizons,
     c'est deux criteres — et c'est exactement ce qui avait produit un resultat
     faux dans ce depot.

Consequence pratique : une limite de stabilite n'a de sens qu'accompagnee de
son horizon. Ne comparez jamais un chiffre obtenu ici a un chiffre obtenu avec
d'autres reglages d'evaluation.

--------------------------------------------------------------------------------
UTILISATION
--------------------------------------------------------------------------------
    from milling_plant import MillingPlant, check_model

    check_model()                       # une fois, verifie la physique
    plant = MillingPlant()

    # boucle ouverte
    r = plant.run(None, rpm=4900, ap=0.25e-3)
    print(r['stable'], r['rms_um'], r['sigma'])

    # avec VOTRE correcteur (voir PARTIE 6 pour les formes acceptees)
    r = plant.run(lambda y, t: -1e5*y, rpm=4900, ap=0.25e-3)

    # limite de passe stable, en metres
    lim = plant.stability_limit(None, rpm=4900)
    lim = plant.stability_limit(lambda dt, tau: MonCorrecteur(dt, tau), rpm=4900)

Executer `python milling_plant.py` lance la verification puis un balayage en
boucle ouverte.
================================================================================
"""
import inspect

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, splu, spsolve

__all__ = ['MillingPlant', 'PlateModel', 'NewmarkIntegrator',
           'ControllerTemplate', 'check_model']

# ============================================================================
# PARTIE 0 — CONSTANTES FIGEES
# ============================================================================
PLATE_L, PLATE_H, PLATE_T = 0.100, 0.080, 0.004        # m
RHO, YOUNG, POISSON = 2830.0, 69e9, 0.33               # AL6061
MESH_N1, MESH_N2 = 36, 30
N_MODES = 5
F_MEASURED = [536.4, 1071.8, 2778.6, 3358.7, 4122.6]   # Hz, Fig. 12(a)
ZETA_MODES = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]  # Tableau 4

# Le patch est COLLE, pas soude. Ni G_adh ni t_adh ne sont donnes par
# l'article : valeurs d'un epoxy structural courant. Les mettre a None
# restitue le collage parfait.
PATCH = dict(x1=0.0, x2=0.020, z1=0.0, z2=0.060,
             d31=175e-12, thickness=0.7e-3, E=63e9, nu=0.35,
             G_adh=1.0e9, t_adh=30e-6)
SENSOR_XZ = (0.100, 0.080)                             # coin superieur droit
V_MAX = 150.0                                          # V, borne amplificateur

# Repartition modale du couplage piezoelectrique, IDENTIFIEE sur la Fig. 12(b).
#
# H_Pe etait la seule grandeur modale encore prise telle quelle des elements
# finis : les frequences sont recalees sur la mesure, les amortissements sont
# mesures, et D_obs est valide par la Fig. 12(a) — une FRF au point d'impact,
# dont les residus valent D_i^2 — a 30 % pres. H_Pe, lui, n'avait jamais ete
# confronte a une mesure, et ne la reproduisait pas (un seul creux a 2825 Hz
# contre quatre mesures ; 14.3 dB RMS d'ecart).
#
# La Fig. 12(b) donne pourtant exactement ce qu'il faut : pour
# G_b(w) = somme_i r_i/(w_i^2 - w^2 + 2j z_i w_i w) avec r_i = D_obs,i H_Pe,i,
# ses POLES sont les frequences mesurees et ses ZEROS les creux mesures — et
# poles + zeros determinent les r_i A UNE CONSTANTE PRES. Les valeurs
# ci-dessous sont ces r_i/r_1, calcules exactement comme racines d'un polynome,
# pas ajustes par un optimiseur :
#
#     r_i/r_1 identifie      1   0.959   1.646   3.728   10.111  -> 5.7 dB RMS
#     r_i/r_1 elements finis 1  -1.538   0.164   3.625   -4.035  -> 14.3 dB
#
# Le NIVEAU n'est PAS identifie (l'echelle en dB des figures est inexploitable) :
# set_identified_coupling conserve la NORME du couplage EF et ne corrige que la
# repartition. Le niveau se balaie par gain_H.
H_IDENT = (1.0, 0.95943, 1.64565, 3.72754, 10.11124)
COUPLING = 'ident'                                     # 'ident' | 'fem'

N_TEETH, D_TOOL = 3, 0.010                             # fraise 3 dents, 10 mm
HELIX = np.deg2rad(35.0)
RAKE = np.deg2rad(15.0)
KT, KN, MU_C = 925e6, 0.26, 0.20
AE_NOM, FZ_NOM = 0.1e-3, 0.02e-3                       # m, m/dent
STEPS_PER_TOOTH = 82                                   # dt = tau/82, FIGE

SPEEDS_DEFAULT = [3000, 4200, 4900, 6000, 7200]        # tr/min
AP_TEST = 0.25e-3                                      # m
SIGMA_MAX = 0.05                                       # 1/s, voir en-tete
T_RUN = 0.60                                           # s, horizon unique
T_LIMIT = 0.60                                         # s, idem — un seul

# Convention de signe des efforts de coupe. L'article se contredit :
#   * son Eq. (13) s'ecrit  M q" + C q' + (K + a4 DtD) q(t) - a4 DtD q(t-tau)
#                            = ft a3 Dt + H u ;
#   * mais en enchainant ses propres Eqs. (1), (2), (5), (10) et (A.4) — avec
#     f_y = -F_y et y_r = -y_P(t) + y_P(t-tau) — on obtient les memes termes
#     TOUS DE SIGNE INVERSE, soit a3 -> -a3, a4 -> -a4.
#
# Ce n'est pas un detail : les deux conventions donnent des lobes quasiment
# complementaires. L'Eq. (13) telle que publiee est celle qui reproduit les
# donnees de l'article — les pics de sa Fig. 13(b) a ~3600 et ~5500 tr/min, la
# divergence du point S (4900 tr/min, 0.30 mm) en Fig. 14(a), et sa Fig. 18. La
# convention derivee inverse ce motif et rendrait le point S stable. L'erreur
# de signe est donc dans les equations intermediaires de l'article, pas dans son
# Eq. (13). force_sign=-1 reproduit la convention derivee, pour verifier qu'un
# resultat ne depend pas de ce choix.
FORCE_SIGN = +1.0


# ============================================================================
# PARTIE 1 — ELEMENT DE PLAQUE MINCE KIRCHHOFF Q4, 12 DDL
#
# Quatre noeuds, chacun portant (w, dw/dx, dw/dy). Fonctions de forme :
# polynomes d'Hermite bicubiques. Toutes les derivees sont analytiques.
# ============================================================================
def shape_function_K(xi, eta, lex, ley):
    """(N, derN) : N = (12,) fonctions de forme ; derN = (5, 12) derivees
    d/dxi, d/deta, d2/dxi2, d2/deta2, d2/dxi deta."""
    N = np.array([
        -((xi-1)*(eta-1)*(xi**2 + xi + eta**2 + eta - 2))/8,
        -ley*((xi-1)*(eta-1)**2*(eta+1))/16,
        lex*((xi-1)**2*(xi+1)*(eta-1))/16,
        ((xi+1)*(eta-1)*(xi**2 - xi + eta**2 + eta - 2))/8,
        ley*((xi+1)*(eta-1)**2*(eta+1))/16,
        lex*((xi-1)*(xi+1)**2*(eta-1))/16,
        ((xi+1)*(eta+1)*(-xi**2 + xi - eta**2 + eta + 2))/8,
        ley*((xi+1)*(eta-1)*(eta+1)**2)/16,
        -lex*((xi-1)*(xi+1)**2*(eta+1))/16,
        ((xi-1)*(eta+1)*(xi**2 + xi + eta**2 - eta - 2))/8,
        -ley*((xi-1)*(eta-1)*(eta+1)**2)/16,
        -lex*((xi-1)**2*(xi+1)*(eta+1))/16,
    ])

    derN = np.zeros((5, 12))
    derN[0, :] = [
        -((eta-1)*(eta**2 + eta + 3*xi**2 - 3))/8,
        -ley*((eta-1)**2*(eta+1))/16,
        lex*((eta-1)*(3*xi**2 - 2*xi - 1))/16,
        ((eta-1)*(eta**2 + eta + 3*xi**2 - 3))/8,
        ley*((eta-1)**2*(eta+1))/16,
        lex*((eta-1)*(3*xi**2 + 2*xi - 1))/16,
        ((eta+1)*(-eta**2 + eta - 3*xi**2 + 3))/8,
        ley*((eta-1)*(eta+1)**2)/16,
        -lex*((eta+1)*(3*xi**2 + 2*xi - 1))/16,
        ((eta+1)*(eta**2 - eta + 3*xi**2 - 3))/8,
        -ley*((eta-1)*(eta+1)**2)/16,
        -lex*((eta+1)*(3*xi**2 - 2*xi - 1))/16,
    ]
    derN[1, :] = [
        -((xi-1)*(xi**2 + xi + 3*eta**2 - 3))/8,
        -ley*((xi-1)*(3*eta**2 - 2*eta - 1))/16,
        lex*((xi-1)**2*(xi+1))/16,
        ((xi+1)*(xi**2 - xi + 3*eta**2 - 3))/8,
        ley*((xi+1)*(3*eta**2 - 2*eta - 1))/16,
        lex*((xi-1)*(xi+1)**2)/16,
        ((xi+1)*(-xi**2 + xi - 3*eta**2 + 3))/8,
        ley*((xi+1)*(3*eta**2 + 2*eta - 1))/16,
        -lex*((xi+1)**2*(xi-1))/16,
        ((xi-1)*(xi**2 + xi + 3*eta**2 - 3))/8,
        -ley*((xi-1)*(3*eta**2 + 2*eta - 1))/16,
        -lex*((xi-1)**2*(xi+1))/16,
    ]
    derN[2, :] = [
        -(3*xi*(eta-1))/4, 0.0, lex*((3*xi-1)*(eta-1))/8,
        (3*xi*(eta-1))/4, 0.0, lex*((3*xi+1)*(eta-1))/8,
        -(3*xi*(eta+1))/4, 0.0, -lex*((3*xi+1)*(eta+1))/8,
        (3*xi*(eta+1))/4, 0.0, -lex*((3*xi-1)*(eta+1))/8,
    ]
    derN[3, :] = [
        -(3*eta*(xi-1))/4, -ley*((3*eta-1)*(xi-1))/8, 0.0,
        (3*eta*(xi+1))/4, ley*((3*eta-1)*(xi+1))/8, 0.0,
        -(3*eta*(xi+1))/4, ley*((3*eta+1)*(xi+1))/8, 0.0,
        (3*eta*(xi-1))/4, -ley*((3*eta+1)*(xi-1))/8, 0.0,
    ]
    derN[4, :] = [
        -(3*xi**2 + 3*eta**2 - 4)/8,
        -ley*(3*eta**2 - 2*eta - 1)/16,
        lex*(3*xi**2 - 2*xi - 1)/16,
        (3*xi**2 + 3*eta**2 - 4)/8,
        ley*(3*eta**2 - 2*eta - 1)/16,
        lex*(3*xi**2 + 2*xi - 1)/16,
        (-3*xi**2 - 3*eta**2 + 4)/8,
        ley*(3*eta**2 + 2*eta - 1)/16,
        -lex*(3*xi**2 + 2*xi - 1)/16,
        (3*xi**2 + 3*eta**2 - 4)/8,
        -ley*(3*eta**2 + 2*eta - 1)/16,
        -lex*(3*xi**2 - 2*xi - 1)/16,
    ]
    return N, derN


def jacobian_K(xi, eta, lex, ley):
    """Jacobienne d'un Q4 rectangulaire — constante, donc analytique."""
    return np.array([[lex/2, 0.0], [0.0, ley/2]])


def matrix_der_K(xi, eta, lex, ley):
    """Matrice B (3x12) des courbures : [-d2N/dx2, -d2N/dy2, -2 d2N/dxdy]."""
    _, derN = shape_function_K(xi, eta, lex, ley)
    J_invT = np.linalg.inv(jacobian_K(xi, eta, lex, ley)).T
    B = np.zeros((3, 12))
    B[0, :] = -J_invT[0, 0]**2 * derN[2, :]
    B[1, :] = -J_invT[1, 1]**2 * derN[3, :]
    B[2, :] = -2 * J_invT[0, 0] * J_invT[1, 1] * derN[4, :]
    return B


def stiffness_matrix_K(E, nu, lex, ley, h):
    """Raideur elementaire 12x12, quadrature de Gauss 3x3."""
    Hf = E*h**2/(12*(1 - nu**2))*np.array([[1.0, nu, 0.0],
                                           [nu, 1.0, 0.0],
                                           [0.0, 0.0, (1-nu)/2]])
    xi = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    w = np.array([5/9, 8/9, 5/9])
    K = np.zeros((12, 12))
    for i in range(3):
        for j in range(3):
            B = matrix_der_K(xi[i], xi[j], lex, ley)
            Ja = jacobian_K(xi[i], xi[j], lex, ley)
            K += h*w[i]*w[j]*B.T @ Hf @ B*np.linalg.det(Ja)
    return K


def mass_matrix_K(rho, lex, ley, h, rotary=True):
    """Masse elementaire 12x12, quadrature de Gauss 5x5 (exacte).

        M = I0 int N N^T dA + I2 int (N,x N,x^T + N,y N,y^T) dA
        I0 = rho h ,  I2 = rho h^3/12

    Le second terme est l'inertie de rotation — les termes I2 de l'Eq. (A.1) de
    l'article. rotary=False restitue le Kirchhoff pur.
    """
    xi, w = np.polynomial.legendre.leggauss(5)
    I0 = rho*h
    I2 = rho*h**3/12.0 if rotary else 0.0
    M = np.zeros((12, 12))
    for i in range(5):
        for j in range(5):
            N, derN = shape_function_K(xi[i], xi[j], lex, ley)
            dJ = np.linalg.det(jacobian_K(xi[i], xi[j], lex, ley))
            M += w[i]*w[j]*dJ*I0*np.outer(N, N)
            if I2:
                Nx = derN[0, :]*(2.0/lex)
                Ny = derN[1, :]*(2.0/ley)
                M += w[i]*w[j]*dJ*I2*(np.outer(Nx, Nx) + np.outer(Ny, Ny))
    return M


def shape_at_point(x_g, y_g, N1, N2, lex, ley, n1):
    """Vecteur ligne N_glob (ndof,) tel que w(x_g, y_g) = N_glob @ U."""
    ndof = 3*n1*(N2 + 1)
    N_glob = np.zeros(ndof)
    I = max(1, min(int(np.floor(x_g/lex)) + 1, N1))
    J = max(1, min(int(np.floor(y_g/ley)) + 1, N2))
    x0, y0 = (I - 1)*lex, (J - 1)*ley
    xi = max(-1.0, min(1.0, 2*(x_g - x0)/lex - 1))
    eta = max(-1.0, min(1.0, 2*(y_g - y0)/ley - 1))
    N_loc, _ = shape_function_K(xi, eta, lex, ley)
    N_glob[_dofs_of_element(I, J, n1)] = N_loc
    return N_glob


def _dofs_of_element(I, J, n1):
    """Ddl globaux de l'element (I, J), dans l'ordre de noeuds de l'element."""
    return np.r_[3*(J-1)*n1 + 3*(I-1) + np.arange(6),
                 3*J*n1 + 3*(I-1) + np.arange(3, 6),
                 3*J*n1 + 3*(I-1) + np.arange(3)]


def laplace_n_patch(xP1, xP2, zP1, zP2, N1, N2, lex, ley, n1, ndof):
    """g = int_patch grad^2 N dA, pour le couplage piezoelectrique (Eq. 15).

    Integration de Gauss 2x2 sur l'INTERSECTION EXACTE patch/element, pas sur
    les elements entiers : le patch ne tombe pas sur la grille.
    """
    g = np.zeros(ndof)
    I_min = max(1, int(np.floor(xP1/lex)) + 1)
    I_max = min(N1, int(np.ceil(xP2/lex)))
    J_min = max(1, int(np.floor(zP1/ley)) + 1)
    J_max = min(N2, int(np.ceil(zP2/ley)))
    xg2 = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    for J in range(J_min, J_max + 1):
        for I in range(I_min, I_max + 1):
            xe_lo, xe_hi = (I-1)*lex, I*lex
            ye_lo, ye_hi = (J-1)*ley, J*ley
            x_lo, x_hi = max(xe_lo, xP1), min(xe_hi, xP2)
            y_lo, y_hi = max(ye_lo, zP1), min(ye_hi, zP2)
            if x_hi <= x_lo or y_hi <= y_lo:
                continue
            xm, xh = (x_lo + x_hi)/2, (x_hi - x_lo)/2
            ym, yh = (y_lo + y_hi)/2, (y_hi - y_lo)/2
            DofE = _dofs_of_element(I, J, n1)
            for ig in range(2):
                for jg in range(2):
                    xi_loc = 2*(xm + xh*xg2[ig] - xe_lo)/lex - 1
                    eta_loc = 2*(ym + yh*xg2[jg] - ye_lo)/ley - 1
                    B_loc = matrix_der_K(xi_loc, eta_loc, lex, ley)
                    g[DofE] += -(B_loc[0, :] + B_loc[1, :])*(xh*yh)
    return g


def _bilinear_B(xi, eta, lex, ley):
    """B membranaire (3x8) d'un Q4 bilineaire, meme ordre de noeuds que
    l'element de flexion : (-1,-1), (+1,-1), (+1,+1), (-1,+1)."""
    dNx = np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])/4*(2/lex)
    dNy = np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])/4*(2/ley)
    B = np.zeros((3, 8))
    B[0, 0::2] = dNx
    B[1, 1::2] = dNy
    B[2, 0::2] = dNy
    B[2, 1::2] = dNx
    return B


def _Qbar(E, nu):
    """Raideur reduite en contraintes planes (3x3)."""
    return E/(1 - nu**2)*np.array([[1.0, nu, 0.0],
                                   [nu, 1.0, 0.0],
                                   [0.0, 0.0, (1 - nu)/2]])


def shear_lag_efficiency(hx, hz, E_b, nu_b, h_b, E_p, nu_p, h_p,
                         G_adh, t_adh, alpha=6.0):
    """Rendement de transfert d'un patch COLLE — Crawley & de Luis, AIAA J.
    25(10):1373-1385, 1987.

    La colle ne transmet le cisaillement que sur une longueur caracteristique
    1/Gamma depuis chaque bord, si bien que

        eps(x) = eps_parfait * [1 - cosh(Gamma x)/cosh(Gamma l)]

    avec Gamma^2 = (G_adh/t_adh)(1/(E_p' h_p) + alpha/(E_b' h_b)), E' = E/(1-nu^2)
    et alpha = 6 en flexion. La moyenne sur le patch, rapportee au cas soude :

        eta(Gamma l) = 1 - tanh(Gamma l)/(Gamma l)

    -> 1 quand la colle est raide, -> (Gamma l)^2/3 quand elle est molle. Pour
    un rectangle on prend la forme separable eta = eta(Gamma hx) eta(Gamma hz)
    sur les DEMI-dimensions : approximation acceptable ici car 1/Gamma = 0.87 mm
    est petit devant le patch (20 x 60 mm).

    hx, hz : DEMI-longueurs du patch [m]. Retour : (eta, Gamma).
    """
    Eb = E_b/(1.0 - nu_b**2)
    Ep = E_p/(1.0 - nu_p**2)
    Gamma = np.sqrt((G_adh/t_adh)*(1.0/(Ep*h_p) + alpha/(Eb*h_b)))

    def _eta(x):
        return 1.0 - np.tanh(x)/x if x > 1e-12 else 0.0
    return float(_eta(Gamma*hx)*_eta(Gamma*hz)), float(Gamma)


# ============================================================================
# PARTIE 2 — LA PLAQUE : ASSEMBLAGE, MODES, PATCH PIEZOELECTRIQUE
# ============================================================================
class PlateModel:
    """Plaque encastree en bas, libre sur les trois autres bords.

    Apres construction :
        omega_n, freq_n   pulsations et frequences propres
        V                 modes normalises en masse (V^T M V = I)
        Mp, Kp, Cp        matrices modales : I, diag(w^2), diag(2 z w)
        D_obs             (n_modes,) deformee au point de mesure
        Dp_array          (n_modes, n_pos) deformee le long de la trajectoire
        DpT_Dp_array      (n_modes, n_modes, n_pos)
        H_Pe_modal        (n_modes,) force modale par volt [N/V]
    """

    def __init__(self, lp, hp, bp, rho, E, nu, N1=30, N2=24, n_modes=3,
                 zeta_modes=None, verbose=True):
        self.lp, self.hp, self.bp = lp, hp, bp
        self.rho, self.E, self.nu = rho, E, nu
        self.N1, self.N2 = N1, N2
        self.n1, self.n2 = N1 + 1, N2 + 1
        self.lex, self.ley = lp/N1, hp/N2
        self.ndof = 3*self.n1*self.n2
        self.n_modes = n_modes
        self.verbose = verbose
        if zeta_modes is None:
            zeta_modes = [0.0031, 0.0017, 0.0027]
        self.zeta_modes = np.array(zeta_modes[:n_modes])
        self._assemble()
        self._apply_bc()
        self._modal_analysis()

    # -- assemblage ----------------------------------------------------------
    def _assemble(self):
        if self.verbose:
            print("[PlateModel] assemblage EF...")
        Ke = stiffness_matrix_K(self.E, self.nu, self.lex, self.ley, self.bp)
        Me = mass_matrix_K(self.rho, self.lex, self.ley, self.bp)
        nelem = self.N1*self.N2
        rows = np.zeros(nelem*144, dtype=np.int64)
        cols = np.zeros_like(rows)
        Kdat = np.zeros(nelem*144)
        Mdat = np.zeros(nelem*144)
        idx = 0
        for J in range(1, self.N2 + 1):
            for I in range(1, self.N1 + 1):
                DofE = _dofs_of_element(I, J, self.n1)
                for a in range(12):
                    for b in range(12):
                        rows[idx] = DofE[a]
                        cols[idx] = DofE[b]
                        Kdat[idx] = Ke[a, b]
                        Mdat[idx] = Me[a, b]
                        idx += 1
        self.Kg = csr_matrix((Kdat, (rows, cols)), shape=(self.ndof, self.ndof))
        self.Mg = csr_matrix((Mdat, (rows, cols)), shape=(self.ndof, self.ndof))
        if self.verbose:
            print(f"[PlateModel] ndof = {self.ndof}, nelem = {nelem}")

    def _apply_bc(self):
        """Encastrement : w, dw/dx, dw/dy bloques sur le bord inferieur."""
        self.DOFb = np.arange(3*self.n1)
        self.DOFf = np.setdiff1d(np.arange(self.ndof), self.DOFb)
        self.M_free = self.Mg[self.DOFf, :][:, self.DOFf]
        self.K_free = self.Kg[self.DOFf, :][:, self.DOFf]

    def _modal_analysis(self):
        if self.verbose:
            print(f"[PlateModel] analyse modale ({self.n_modes} modes)...")
        w2, V = eigsh(self.K_free.tocsc(), k=self.n_modes,
                      M=self.M_free.tocsc(), sigma=0, which='LM')
        order = np.argsort(np.real(w2))
        w2 = np.real(w2[order])
        V = np.real(V[:, order])
        self.omega_n = np.sqrt(w2)
        self.freq_n = self.omega_n/(2*np.pi)
        for k in range(self.n_modes):            # normalisation en masse
            V[:, k] /= np.sqrt(V[:, k] @ (self.M_free @ V[:, k]))
        # CONVENTION DE SIGNE. ARPACK part d'un vecteur aleatoire : sans cette
        # normalisation, le SIGNE de chaque mode change d'une construction a
        # l'autre, y compris dans le meme processus. C'est sans consequence sur
        # la reponse simulee — tout observable est un produit D_i H_i ou D_i^2,
        # invariant par ce changement, verifie a 1e-11 pres — mais c'est un
        # piege des qu'un correcteur est SYNTHETISE sur un objet et SIMULE sur
        # un autre : H et D_obs changeraient de signe entre les deux, et le
        # retour se retrouverait en opposition de phase sur ces modes-la.
        # On fixe donc le signe : la composante de plus grand module est
        # positive. Choix arbitraire mais deterministe et independant du
        # maillage.
        for k in range(self.n_modes):
            if V[np.argmax(np.abs(V[:, k])), k] < 0:
                V[:, k] = -V[:, k]
        self.V = V
        self.Mp = np.eye(self.n_modes)
        self.Kp = np.diag(self.omega_n**2)
        self.Cp = np.diag(2*self.zeta_modes*self.omega_n)
        if self.verbose:
            for k in range(self.n_modes):
                print(f"   mode {k+1} : {self.freq_n[k]:8.2f} Hz")

    # -- points d'interet ----------------------------------------------------
    def precompute_Dp(self, zp_pos, n_pos=2001):
        """Pre-calcule Dp(x) en n_pos positions le long de z = zp_pos.

        L'outil SE DEPLACE pendant la passe ; recalculer la deformee modale a
        chaque pas couterait plus cher que l'integration elle-meme.
        """
        if self.verbose:
            print(f"[PlateModel] pre-calcul Dp(x) : {n_pos} positions...")
        xp_array = np.linspace(0, self.lp, n_pos)
        Dp_array = np.zeros((self.n_modes, n_pos))
        DpT_Dp_array = np.zeros((self.n_modes, self.n_modes, n_pos))
        for kp in range(n_pos):
            Nv = shape_at_point(xp_array[kp], zp_pos, self.N1, self.N2,
                                self.lex, self.ley, self.n1)
            Dp_kp = Nv[self.DOFf] @ self.V
            Dp_array[:, kp] = Dp_kp
            DpT_Dp_array[:, :, kp] = np.outer(Dp_kp, Dp_kp)
        self.xp_array = xp_array
        self.Dp_array = Dp_array
        self.DpT_Dp_array = DpT_Dp_array
        self.zp_pos = zp_pos

    def get_Dp_at(self, kp):
        """(Dp, Dp^T Dp) a l'indice de position kp."""
        return self.Dp_array[:, kp], self.DpT_Dp_array[:, :, kp]

    def set_observation(self, x_obs, z_obs):
        """Point de mesure du capteur."""
        N_obs = shape_at_point(x_obs, z_obs, self.N1, self.N2,
                               self.lex, self.ley, self.n1)
        self.D_obs = N_obs[self.DOFf] @ self.V
        self.x_obs, self.z_obs = x_obs, z_obs
        if getattr(self, '_ident_ratio', None) is not None:
            # H_Pe identifie est defini PAR RAPPORT a D_obs : deplacer le
            # capteur sans le reappliquer laisserait les deux incoherents.
            self.set_identified_coupling(self._ident_ratio)

    # -- couplage piezoelectrique -------------------------------------------
    def set_identified_coupling(self, ratio):
        """Impose la REPARTITION MODALE de H_Pe identifiee sur la Fig. 12(b).

        `ratio` porte les r_i = D_obs,i H_Pe,i a une constante pres ; on en
        deduit H_Pe = r/D_obs, de sorte que la fonction de transfert modelisee
        tension -> capteur devienne celle qui a ete mesuree.

        La CONSTANTE n'est pas identifiee : le niveau absolu des figures de
        l'article n'est pas exploitable (la Fig. 12(a) demande +12 dB et la
        Fig. 12(b) +6.8 dB pour se superposer au meme modele — aucun modele ne
        peut satisfaire les deux). On conserve donc la NORME du couplage
        elements finis et on ne corrige que la repartition ; le niveau reste
        balaye par gain_H.

        Le signe global est lui aussi conventionnel — il depend de la face sur
        laquelle le patch est colle — et on le choisit du meme cote que le
        couplage EF, pour eviter un saut de signe sans contenu physique.
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

    def add_piezo_patch(self, xP1, xP2, zP1, zP2, d31, h_Pa, E_Pe, nu_Pe,
                        rho_Pe=7450.0, structural=True,
                        G_adh=None, t_adh=None, alpha_lag=6.0, membrane=True):
        """Colle un patch piezoelectrique et calcule son couplage.

        1. structural=True ajoute la RAIDEUR et la MASSE du patch aux matrices
           globales, puis refait l'analyse modale. Sans cela les frequences
           propres seraient celles de la plaque NUE, alors que les mesures de
           l'article portent sur la plaque AVEC patch colle.
        2. H_Pe(x) = m_piezo * grad^2 N(x) integre sur le patch (Eq. 15), avec
           m_piezo = -eta E_Pe d31 (bp + h_Pa)/(2(1 - nu_Pe)).
        3. La couche de COLLE (G_adh, t_adh) donne un rendement eta < 1 qui
           reduit A LA FOIS la part composite du raidissement et H_Pe, dans les
           memes proportions. Un seul parametre physique, deux effets : ce ne
           sont pas deux coefficients d'ajustement independants. G_adh=None
           restitue le collage parfait.
        4. Couplage MEMBRANE-FLEXION (membrane=True) : un patch colle d'un seul
           cote applique aussi une resultante membranaire, qui se rabat en
           flexion car le stratifie est non symetrique. Voir
           _membrane_load_correction. Effet : -11 % sur H_Pe pour les modes
           1, 2, 4, 5 et -29 % sur le mode 3. NE change PAS la signature des
           creux : l'effet est quasi scalaire, et un scalaire se simplifie dans
           les rapports de residus qui fixent les zeros.
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
                  f"rendement eta = {eta_bond:.4f}")

        if structural:
            self._add_patch_structure(xP1, xP2, zP1, zP2,
                                      h_Pa, E_Pe, nu_Pe, rho_Pe, eta_bond)

        m_piezo = -eta_bond*E_Pe*d31*(self.bp + h_Pa)/(2*(1 - nu_Pe))
        g_lap = laplace_n_patch(xP1, xP2, zP1, zP2, self.N1, self.N2,
                                self.lex, self.ley, self.n1, self.ndof)
        H_Pe_phys = m_piezo*g_lap

        self.membrane_coupling = bool(membrane)
        if membrane:
            # M_E = N_E (bp + h_Pa)/2, donc N_E porte le signe OPPOSE de
            # m_piezo : ce dernier est deja -M_E (convention de B_b).
            n_E = eta_bond*E_Pe*d31/(1 - nu_Pe)
            corr, _ = self._membrane_load_correction(
                xP1, xP2, zP1, zP2, h_Pa, E_Pe, nu_Pe,
                np.array([n_E, n_E, 0.0]))
            if self.verbose:
                r = (np.linalg.norm(corr[self.DOFf])
                     / max(np.linalg.norm(H_Pe_phys[self.DOFf]), 1e-30))
                print(f"[PlateModel] membrane-flexion : |corr|/|flexion| "
                      f"= {r:.3f} en ddl bruts")
            H_Pe_phys = H_Pe_phys + corr

        self.H_Pe_modal = self.V.T @ H_Pe_phys[self.DOFf]
        self.patch = dict(xP1=xP1, xP2=xP2, zP1=zP1, zP2=zP2,
                          m_piezo=m_piezo, structural=structural)
        if self.verbose:
            print(f"[PlateModel] ||H_Pe_modal|| = "
                  f"{np.linalg.norm(self.H_Pe_modal):.3e} N/V")

    def _add_patch_structure(self, xP1, xP2, zP1, zP2,
                             h_Pa, E_Pe, nu_Pe, rho_Pe, eta_bond=1.0):
        """Ajoute dK, dM du patch et refait les modes."""
        E1 = self.E/(1 - self.nu**2)
        E2 = E_Pe/(1 - nu_Pe**2)
        h1, h2 = self.bp, h_Pa
        z1, z2 = 0.0, (h1 + h2)/2
        zbar = (E1*h1*z1 + E2*h2*z2)/(E1*h1 + E2*h2)     # plan neutre composite
        D_plate = E1*h1**3/12
        D_comp = (E1*(h1**3/12 + h1*(z1 - zbar)**2)
                  + E2*(h2**3/12 + h2*(z2 - zbar)**2))
        # Le patch apporte sa flexion propre SANS aucun collage ; l'effet
        # composite (plan neutre decale) demande, lui, que la colle transmette
        # la contrainte axiale. Seul ce second terme est reduit par eta_bond.
        D_free = E2*h2**3/12
        rK_free = D_free/D_plate
        rK_perfect = D_comp/D_plate - 1.0
        rK = rK_free + eta_bond*(rK_perfect - rK_free)
        rM = (rho_Pe*h_Pa)/(self.rho*self.bp)

        Ke = stiffness_matrix_K(self.E, self.nu, self.lex, self.ley, self.bp)
        Me = mass_matrix_K(self.rho, self.lex, self.ley, self.bp)

        # Chaque element est pondere par sa FRACTION DE RECOUVREMENT avec le
        # rectangle du patch, et non retenu ou rejete selon la position de son
        # centre : le test au centre quantifiait l'empreinte sur la grille
        # (19.44 x 61.33 mm au lieu de 20 x 60) alors que laplace_n_patch
        # integre exactement sur le rectangle. Raideur ajoutee et couplage
        # piezoelectrique ne portaient pas sur la meme aire.
        rows, cols, Kd, Md = [], [], [], []
        n_cov, area_cov = 0, 0.0
        a_elem = self.lex*self.ley
        for J in range(1, self.N2 + 1):
            for I in range(1, self.N1 + 1):
                ox = min(I*self.lex, xP2) - max((I-1)*self.lex, xP1)
                oz = min(J*self.ley, zP2) - max((J-1)*self.ley, zP1)
                if ox <= 1e-12 or oz <= 1e-12:
                    continue
                frac = (ox*oz)/a_elem
                n_cov += 1
                area_cov += ox*oz
                DofE = _dofs_of_element(I, J, self.n1)
                for a in range(12):
                    for b in range(12):
                        rows.append(DofE[a])
                        cols.append(DofE[b])
                        Kd.append(frac*rK*Ke[a, b])
                        Md.append(frac*rM*Me[a, b])
        self.Kg = self.Kg + csr_matrix((Kd, (rows, cols)),
                                       shape=(self.ndof, self.ndof))
        self.Mg = self.Mg + csr_matrix((Md, (rows, cols)),
                                       shape=(self.ndof, self.ndof))
        if self.verbose:
            a_exact = (xP2 - xP1)*(zP2 - zP1)
            print(f"[PlateModel] patch structurel : {n_cov} elements, aire "
                  f"{area_cov*1e6:.2f} mm2 (exacte {a_exact*1e6:.2f}, ecart "
                  f"{area_cov/a_exact-1:+.2%}), dD/D = {rK*100:.1f} % "
                  f"(colle parfait {rK_perfect*100:.1f} %, decolle "
                  f"{rK_free*100:.1f} %), dm/m = {rM*100:.1f} %")
        self._apply_bc()
        self._modal_analysis()
        if hasattr(self, 'zp_pos'):
            self.precompute_Dp(self.zp_pos, self.Dp_array.shape[1])
        if hasattr(self, 'x_obs'):
            self.set_observation(self.x_obs, self.z_obs)

    def _membrane_load_correction(self, xP1, xP2, zP1, zP2,
                                  h_Pa, E_Pe, nu_Pe, N_E):
        """Correction de charge due au couplage MEMBRANE-FLEXION.

        Un patch colle d'UN SEUL COTE n'applique pas qu'un moment : sous
        tension il applique aussi une resultante membranaire N_E dans son plan.
        Le stratifie plaque+patch etant NON SYMETRIQUE, sa matrice de couplage
        B = Qbar_p h_p (h_plaque + h_p)/2 est non nulle sur la zone du patch, et
        cette resultante se rabat en flexion. Une formulation de Kirchhoff en
        flexion pure ne peut pas le representer : elle n'a pas de ddl
        membranaires.

        Les modes membranaires de cette plaque sont vers 33 kHz, tres au-dessus
        des 4.1 kHz utiles ; on condense donc (u, v) en STATIQUE :

            [K_mm  K_mb][u]   [f_m]
            [K_mb^T K_bb][w] = [f_b]
            =>  (K_bb - K_mb^T K_mm^-1 K_mb) w = f_b - K_mb^T K_mm^-1 f_m

        Cette methode ne renvoie que la correction de CHARGE — le second terme.
        La correction de RAIDEUR est deja prise en compte par
        _add_patch_structure, qui utilise la rigidite de section transformee :
        c'est exactement le resultat de la meme condensation, verifie a
        0.002 % pres.
        """
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

                # rigidite membranaire sur TOUT l'element, part du patch
                # ponderee par le recouvrement
                frac = (ox*oz)/(self.lex*self.ley) if cov else 0.0
                Am = A_bare + frac*(A_pat - A_bare)
                Ke_mm = np.zeros((8, 8))
                for i in range(2):
                    for j in range(2):
                        Bm = _bilinear_B(gp[i], gp[j], self.lex, self.ley)
                        Ke_mm += (self.lex/2)*(self.ley/2)*Bm.T @ Am @ Bm
                for a in range(8):
                    for b in range(8):
                        rm.append(mdof[a])
                        cm.append(mdof[b])
                        vm.append(Ke_mm[a, b])
                if not cov:
                    continue

                # couplage et charge : integres EXACTEMENT sur l'intersection
                # patch/element, comme laplace_n_patch
                a_, b_ = max(xl, xP1), min(xh, xP2)
                c_, d_ = max(zl, zP1), min(zh, zP2)
                xm, xr = (a_+b_)/2, (b_-a_)/2
                zm, zr = (c_+d_)/2, (d_-c_)/2
                Ke_mb = np.zeros((8, 12))
                fe_m = np.zeros(8)
                for i in range(2):
                    for j in range(2):
                        xi = 2*(xm + xr*gp[i] - xl)/self.lex - 1
                        et = 2*(zm + zr*gp[j] - zl)/self.ley - 1
                        Bm = _bilinear_B(xi, et, self.lex, self.ley)
                        Bb = matrix_der_K(xi, et, self.lex, self.ley)
                        w = xr*zr
                        Ke_mb += w*Bm.T @ B_pat @ Bb
                        fe_m += w*Bm.T @ N_E
                for a in range(8):
                    for b in range(12):
                        rc.append(mdof[a])
                        cc.append(bdof[b])
                        vc.append(Ke_mb[a, b])
                f_m[mdof] += fe_m

        K_mm = csr_matrix((vm, (rm, cm)), shape=(2*nnod, 2*nnod))
        K_mb = csr_matrix((vc, (rc, cc)), shape=(2*nnod, self.ndof))
        mfree = np.setdiff1d(np.arange(2*nnod), np.arange(2*n1))
        y = splu(K_mm[mfree, :][:, mfree].tocsc()).solve(f_m[mfree])
        return np.asarray(-(K_mb[mfree, :].T @ y)).ravel(), dict(n_mfree=len(mfree))

    # -- recalage ------------------------------------------------------------
    def calibrate_frequencies(self, f_targets_hz):
        """Recale les frequences propres sur les valeurs mesurees, SANS toucher
        aux deformees. Seuls Kp, Cp, omega_n, freq_n changent ; V, Dp_array,
        D_obs, H_Pe_modal sont inchanges.

        Justifie par l'ecart residuel entre la formulation EF (Hermite Q4) et le
        Chebyshev-Ritz de l'article.
        """
        f_t = np.asarray(f_targets_hz, float)[:self.n_modes]
        if self.verbose:
            for k in range(len(f_t)):
                print(f"[PlateModel] mode {k+1} : {self.freq_n[k]:8.2f} -> "
                      f"{f_t[k]:8.2f} Hz ({(f_t[k]/self.freq_n[k]-1)*100:+.2f} %)")
        self.freq_scale = f_t/self.freq_n[:len(f_t)]
        self.omega_n = 2*np.pi*f_t
        self.freq_n = f_t.copy()
        self.Kp = np.diag(self.omega_n**2)
        self.Cp = np.diag(2*self.zeta_modes*self.omega_n)


# ============================================================================
# PARTIE 3 — EFFORTS DE COUPE (Eqs. 3-4 de l'article)
#
# Pour chaque dent j, l'angle de coupe vaut
#     theta(t, j, z) = Omega t + 2 pi (j-1)/NT - z tan(eta)/RT
# Les fonctions trigonometriques sont integrees analytiquement sur la portion
# engagee [phi_st, phi_ex] et sur la profondeur axiale [za_low, za_high].
# ============================================================================
def cutting_coefficients(kn, mu_c, eta, gamma_n):
    """k1, k2 de l'Eq. (3) :  k1 = kn/cos(eta),
    k2 = 1 + mu_c tan(eta)(cos(gamma_n) - kn sin(gamma_n))."""
    return kn/np.cos(eta), 1.0 + mu_c*np.tan(eta)*(np.cos(gamma_n)
                                                   - kn*np.sin(gamma_n))


def milling_force_coeffs(t, Omega, NT, RT, eta, phi_st, phi_ex,
                         za_low, za_high, k1, k2, kt):
    """(alpha3, alpha4) a l'instant t. alpha3 porte l'effort d'avance,
    alpha4 le couplage regeneratif."""
    ss = sc = cc = 0.0
    te = np.tan(eta)
    for j in range(NT):
        theta0 = Omega*t + 2*np.pi*j/NT
        theta_max = theta0 - za_low*te/RT
        theta_min = theta0 - za_high*te/RT
        k_min = int(np.floor((theta_min - phi_ex)/(2*np.pi)))
        k_max = int(np.ceil((theta_max - phi_st)/(2*np.pi)))
        for kk in range(k_min, k_max + 1):
            th_lo = max(theta_min, phi_st + 2*np.pi*kk)
            th_hi = min(theta_max, phi_ex + 2*np.pi*kk)
            if th_hi <= th_lo:
                continue
            z2 = min((theta0 - th_lo)*RT/te, za_high)
            z1 = max((theta0 - th_hi)*RT/te, za_low)
            if z2 <= z1:
                continue
            theta1 = theta0 - z1*te/RT
            theta2 = theta0 - z2*te/RT
            ss += (z2 - z1)/2 + RT/(2*te)*np.sin(theta2-theta1)*np.cos(theta2+theta1)
            sc += RT/(2*te)*np.sin(theta1 - theta2)*np.sin(theta1 + theta2)
            cc += (z2 - z1)/2 - RT/(2*te)*np.sin(theta2-theta1)*np.cos(theta2+theta1)
    return k2*kt*ss - k1*kt*sc, k2*kt*sc - k1*kt*cc


def precompute_alpha_periodic(dt, n_per, n_total, Omega, NT, RT, eta,
                              phi_st, phi_ex, za_low, za_high, k1, k2, kt):
    """alpha3(t), alpha4(t) sur une periode tau, puis duplique.

    alpha3 et alpha4 sont periodiques de periode tau = 60/(NT rpm) : on ne
    calcule donc que n_per = 82 points, pas les ~40 000 pas de la simulation.
    """
    a3 = np.zeros(n_per)
    a4 = np.zeros(n_per)
    for k in range(n_per):
        a3[k], a4[k] = milling_force_coeffs(k*dt, Omega, NT, RT, eta,
                                            phi_st, phi_ex, za_low, za_high,
                                            k1, k2, kt)
    nrep = n_total//n_per + 2
    return np.tile(a3, nrep)[:n_total], np.tile(a4, nrep)[:n_total]


# ============================================================================
# PARTIE 4 — INTEGRATION TEMPORELLE (Newmark implicite, gamma=1/2, beta=1/4)
# ============================================================================
class NewmarkIntegrator:
    """Integre l'Eq. (13) avec retard regeneratif et outil mobile.

    v_max : borne de l'amplificateur en volts. La commande rendue par le
        correcteur y est ECRETEE avant d'etre appliquee A LA PLAQUE ET avant
        d'etre renvoyee au correcteur au pas suivant, de sorte qu'un
        observateur voie toujours la tension reellement appliquee. None
        desactive la saturation — a n'utiliser que pour un diagnostic, jamais
        pour comparer deux lois de commande.
    """

    def __init__(self, plate, dt, T_end, ft, tau, verbose=True, v_max=V_MAX):
        self.plate = plate
        self.dt = dt
        self.T_end = T_end
        self.ft = ft
        self.tau = tau
        self.v_max = v_max
        self.t_vec = np.arange(0, T_end + dt/2, dt)
        self.nstep = len(self.t_vec)
        self.n_tau = int(np.round(tau/dt))
        self.n_modes = plate.n_modes
        self.verbose = verbose
        if verbose:
            print(f"[Newmark] dt = {dt:.2e} s, T = {T_end:.3f} s, "
                  f"nstep = {self.nstep}")
            print(f"[Newmark] tau = {tau*1e3:.3f} ms = {self.n_tau} pas")

    def simulate(self, alpha3_t, alpha4_t, kp_idx, controller=None,
                 stop_threshold=5e-3, progress=False):
        """alpha3_t, alpha4_t : (nstep,) ; kp_idx : (nstep,) indices de position
        outil ; controller : voir PARTIE 6, ou None pour la boucle ouverte."""
        n = self.n_modes
        nstep = self.nstep
        dt = self.dt

        qm = np.zeros((n, nstep))
        qmd = np.zeros((n, nstep))
        qmdd = np.zeros((n, nstep))
        y_mill = np.zeros(nstep)
        y_meas = np.zeros(nstep)
        u_cmd = np.zeros(nstep)
        u_real = np.zeros(nstep)

        Mp, Kp_modal, Cp_modal = self.plate.Mp, self.plate.Kp, self.plate.Cp
        # H_Pe est TOUJOURS applique. La version precedente le forcait a zero
        # quand controller vaut None ; c'etait sans effet (u = 0 dans ce cas)
        # mais cela rendait impossible d'injecter une tension autrement que par
        # un objet correcteur, ce qui est exactement la liberte visee ici.
        H_Pe_modal = np.asarray(self.plate.H_Pe_modal, float)
        D_obs = self.plate.D_obs

        hook = _ControllerHook(controller, dt=dt, tau=self.tau, v_max=self.v_max)
        gNM, bNM = 0.5, 0.25
        u_prev = 0.0
        diverged_at = 0
        progress_step = max(1, nstep//10)

        for k in range(1, nstep):
            kp = int(kp_idx[k])
            Dp_now, DpT_Dp_now = self.plate.get_Dp_at(kp)
            a3, a4 = alpha3_t[k], alpha4_t[k]

            # La mesure est le deplacement au capteur a la fin du pas
            # PRECEDENT : un correcteur numerique ne peut pas utiliser une
            # grandeur qu'il n'a pas encore mesuree.
            y_now = float(D_obs @ qm[:, k-1])
            y_meas[k] = y_now

            u = hook(y=y_now, t=self.t_vec[k], k=k, u_prev=u_prev,
                     x_tool=self.plate.xp_array[kp])
            u_cmd[k] = u

            # Saturation amplificateur : la borne du banc s'applique a TOUT
            # correcteur, qu'il sature en interne ou non.
            if self.v_max is not None:
                u = float(np.clip(u, -self.v_max, self.v_max))
            u_real[k] = u

            q_delay = qm[:, k - self.n_tau] if k - self.n_tau > 0 else np.zeros(n)

            K_eff = Kp_modal + a4*DpT_Dp_now
            F_now = (self.ft*a3*Dp_now + a4*DpT_Dp_now @ q_delay
                     + H_Pe_modal*u)

            qd_pred = qmd[:, k-1] + (1-gNM)*dt*qmdd[:, k-1]
            q_pred = qm[:, k-1] + dt*qmd[:, k-1] + (0.5-bNM)*dt**2*qmdd[:, k-1]
            S_eff = Mp + gNM*dt*Cp_modal + bNM*dt**2*K_eff
            rhs = F_now - Cp_modal @ qd_pred - K_eff @ q_pred

            qmdd[:, k] = np.linalg.solve(S_eff, rhs)
            qmd[:, k] = qd_pred + gNM*dt*qmdd[:, k]
            qm[:, k] = q_pred + bNM*dt**2*qmdd[:, k]

            y_mill[k] = Dp_now @ qm[:, k]
            u_prev = u

            if progress and k % progress_step == 0:
                print(f"   {round(100*k/nstep):3d}%  t={self.t_vec[k]:.3f}s  "
                      f"|y|={abs(y_mill[k])*1e6:7.1f}um  "
                      f"|u|={abs(u_real[k]):6.1f}V")

            if abs(y_mill[k]) > stop_threshold:
                if self.verbose:
                    print(f"   ! divergence a t={self.t_vec[k]:.4f}s, "
                          f"|y|={abs(y_mill[k])*1e6:.0f}um")
                diverged_at = k
                break

        return dict(t=self.t_vec, y=y_mill, y_meas=y_meas, u=u_real,
                    u_cmd=u_cmd, qm=qm, qmd=qmd, qmdd=qmdd,
                    diverged_at=diverged_at,
                    stop_idx=(diverged_at if diverged_at > 0 else nstep - 1))


# ============================================================================
# PARTIE 5 — LE PROCEDE COMPLET
# ============================================================================
def growth_rate(y, t, nblock=12):
    """Taux de croissance exponentiel sigma [1/s] sur la fenetre fournie.

    On decoupe en blocs, on prend la RMS de chaque bloc, et on ajuste au moindre
    carre log(RMS) contre le temps : la pente EST sigma. Passer au log rend la
    mesure insensible a l'amplitude, donc au NIVEAU de la reponse forcee — c'est
    ce qui permet de voir un mode instable encore petit devant elle. Voir
    SIGMA_MAX et le paragraphe CRITERE DE STABILITE de l'en-tete.
    """
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    if len(y) < 8:
        return 0.0
    nb = max(3, min(nblock, len(y)//4))
    idx = np.array_split(np.arange(len(y)), nb)
    tb = np.array([t[k].mean() for k in idx])
    rb = np.array([np.sqrt(np.mean(y[k]**2)) for k in idx])
    m = rb > 0
    if int(m.sum()) < 3 or (tb[m].max() - tb[m].min()) <= 0:
        return 0.0
    A = np.vstack([tb[m], np.ones(int(m.sum()))]).T
    return float(np.linalg.lstsq(A, np.log(rb[m]), rcond=None)[0][0])


class MillingPlant:
    """Plaque + patch + capteur + coupe. AUCUN CORRECTEUR : il est fourni a
    l'appel de run() / stability_limit().

    patch, zeta : surchargent PATCH et ZETA_MODES pour cette instance
        seulement. Passer par ces arguments plutot que de reaffecter les
        constantes du module : une reaffectation resterait active pour toutes
        les instances construites ensuite dans le meme processus.
    gain_H : facteur multiplicatif sur H_Pe_modal. Le NIVEAU du couplage
        piezoelectrique n'est PAS valide experimentalement (voir l'en-tete) :
        ce facteur existe pour balayer cette incertitude. Construire la plaque
        nominale pour SYNTHETISER le correcteur, puis une plaque perturbee pour
        le SIMULER, donne un vrai essai de robustesse a l'erreur de gain
        actionneur.
    coupling : 'ident' (defaut) prend la repartition modale de H_Pe identifiee
        sur la Fig. 12(b) ; 'fem' garde celle des elements finis, qui ne
        reproduit aucun des creux mesures. Dans les deux cas la NORME de H_Pe
        est la meme, seule sa repartition change.
    force_sign : +1 pour l'Eq. (13) telle que publiee (defaut), -1 pour la
        convention obtenue en enchainant les Eqs. (1), (2), (5), (10), (A.4) du
        meme article. Voir FORCE_SIGN.
    """

    def __init__(self, verbose=False, patch=None, zeta=None, gain_H=1.0,
                 force_sign=None, coupling=None):
        patch = PATCH if patch is None else patch
        zeta = ZETA_MODES if zeta is None else zeta
        coupling = COUPLING if coupling is None else coupling
        if coupling not in ('ident', 'fem'):
            raise ValueError("coupling doit valoir 'ident' ou 'fem'")
        p = PlateModel(PLATE_L, PLATE_H, PLATE_T, RHO, YOUNG, POISSON,
                       N1=MESH_N1, N2=MESH_N2, n_modes=N_MODES,
                       zeta_modes=zeta, verbose=False)
        p.precompute_Dp(zp_pos=PLATE_H - 0.15e-3, n_pos=2001)
        p.set_observation(*SENSOR_XZ)
        p.add_piezo_patch(patch['x1'], patch['x2'], patch['z1'], patch['z2'],
                          patch['d31'], patch['thickness'], patch['E'],
                          patch['nu'], G_adh=patch.get('G_adh'),
                          t_adh=patch.get('t_adh'))
        if coupling == 'ident':
            if p.n_modes > len(H_IDENT):
                raise ValueError(
                    f"H_IDENT n'est identifie que sur {len(H_IDENT)} modes "
                    f"(les cinq de la Fig. 12) ; construire un modele a "
                    f"{p.n_modes} modes impose coupling='fem'")
            # Troncature : sur un modele reduit a n < 5 modes on garde les n
            # premiers residus identifies — meme troncature que celle deja
            # subie par les frequences et les amortissements.
            p.set_identified_coupling(H_IDENT[:p.n_modes])
        if gain_H != 1.0:
            p.H_Pe_modal = np.asarray(p.H_Pe_modal, float)*gain_H
        p.calibrate_frequencies(F_MEASURED)

        self.plate = p
        self.patch_cfg = patch
        self.coupling = coupling
        self.gain_H = gain_H
        self.force_sign = FORCE_SIGN if force_sign is None else float(force_sign)
        self.k1c, self.k2c = cutting_coefficients(KN, MU_C, HELIX, RAKE)
        self._cache = {}
        if verbose:
            self.describe()

    # -- ce dont un correcteur a besoin pour se synthetiser ------------------
    @property
    def freqs(self):
        """Frequences propres calibrees, Hz."""
        return np.asarray(self.plate.omega_n).ravel()/(2*np.pi)

    @property
    def modal(self):
        """Le modele reduit, sous la forme utile a une synthese :

            omega  (n,)   pulsations propres [rad/s]
            zeta   (n,)   amortissements modaux
            H      (n,)   entree patch  [N/V]      -> matrice B
            D_obs  (n,)   sortie capteur [1/sqrt(kg)] -> matrice C
            D_tool (n,)   deformee au contact outil, en x = 0

        L'etat naturel est [q ; q'] de dimension 2n, avec
            q" = -diag(w^2) q - diag(2 z w) q' + H u + D_tool^T f
            y  = D_obs q
        """
        return dict(H=np.asarray(self.plate.H_Pe_modal).ravel(),
                    D_obs=np.asarray(self.plate.D_obs).ravel(),
                    D_tool=np.asarray(self.plate.Dp_array[:, 0]).ravel(),
                    zeta=np.asarray(self.plate.zeta_modes).ravel(),
                    omega=np.asarray(self.plate.omega_n).ravel())

    def sampling(self, rpm):
        """(dt, tau) imposes par la vitesse. dt N'EST PAS un choix libre :
        c'est tau/82, pour que le retard tombe sur la grille."""
        tau = 60.0/(N_TEETH*rpm)
        return tau/STEPS_PER_TOOTH, tau

    def describe(self):
        m = self.modal
        print("frequences (Hz) :", np.round(self.freqs, 1))
        print("H (patch, N/V)  :", np.round(m['H'], 5))
        print("D_obs (capteur) :", np.round(m['D_obs'], 3))
        print("D_tool (outil)  :", np.round(m['D_tool'], 3))
        raw = np.abs(m['H']*m['D_obs'])
        print("|H D| brut      :", np.round(raw, 4))
        nrm = raw/m['omega']**2
        print("|H D|/w^2 norm. :", np.array2string(nrm/nrm.max(), precision=4))

    # -- mise en place d'une condition de coupe ------------------------------
    def _setup(self, rpm, ap, T, ae, fz, tool_pos):
        key = (rpm, round(ap, 9), round(T, 4), round(ae, 9), round(fz, 9),
               tool_pos)
        if key not in self._cache:
            dt, tau = self.sampling(rpm)
            s = NewmarkIntegrator(self.plate, dt=dt, T_end=T, ft=fz, tau=tau,
                                  verbose=False, v_max=V_MAX)
            phi = np.pi - np.arccos(1 - ae/(D_TOOL/2))
            a3, a4 = precompute_alpha_periodic(
                dt, STEPS_PER_TOOTH, s.nstep, 2*np.pi*rpm/60, N_TEETH,
                D_TOOL/2, HELIX, phi, np.pi, PLATE_H - ap, PLATE_H,
                self.k1c, self.k2c, KT)
            self._cache[key] = (s, self.force_sign*a3, self.force_sign*a4,
                                dt, tau)
        return self._cache[key]

    # -- une coupe -----------------------------------------------------------
    def run(self, controller=None, rpm=4900, ap=AP_TEST, T=T_RUN, ae=AE_NOM,
            fz=FZ_NOM, tool_pos=0, full_pass=False, keep_signals=False):
        """Simule une coupe. `controller` : voir PARTIE 6, ou None.

        tool_pos  : indice de station de l'outil (0 = debut d'arete, station la
                    moins stable). Ignore si full_pass=True.
        full_pass : l'outil parcourt les 100 mm ; T est alors recalcule.

        Retour : dict avec
            stable   verdict, sigma <= SIGMA_MAX et pas de divergence
            sigma    taux de croissance [1/s] sur la derniere moitie du releve
            rms_um   deplacement RMS a l'outil [um]
            peak_um  deplacement crete [um]
            rms_u    tension RMS [V]
            peak_u   tension crete [V] — comparer a V_MAX = 150
            diverged le seuil de securite a ete franchi
            t_end    instant du dernier point retenu
            dt, tau  pas de temps et retard
        """
        if full_pass:
            T = PLATE_L/(fz*N_TEETH*rpm/60.0)
        s, a3, a4, dt, tau = self._setup(rpm, ap, T, ae, fz, tool_pos)
        if full_pass:
            v = fz*N_TEETH*rpm/60.0
            kp = np.clip(np.round(np.minimum(v*s.t_vec, PLATE_L)/PLATE_L*2000
                                  ).astype(int), 0, 2000)
        else:
            kp = np.full(s.nstep, int(tool_pos), dtype=int)
        if controller is not None and hasattr(controller, 'reset'):
            controller.reset()
        r = s.simulate(a3, a4, kp, controller=controller, progress=False,
                       stop_threshold=5e-3 if full_pass else 1e-4)

        i = r['stop_idx']
        diverged = r['diverged_at'] > 0
        i0 = min(s.nstep//3, max(0, i - 1))
        y2 = r['y'][i0:i+1]
        if y2.size < 4:
            y2 = r['y'][:max(4, i+1)]
        # sigma se mesure sur la DERNIERE MOITIE du releve : la reponse forcee
        # d'une plaque a zeta = 0.17 % met ~0.3 s a s'etablir, et tant qu'elle
        # monte, une droite ajustee sur log(RMS) lit cette montee comme une
        # croissance. Sur la premiere moitie sigma vaut ~1 /s meme pour une
        # coupe parfaitement stable ; sur la seconde il tombe sous 0.02.
        yv, tv = r['y'][:i+1], r['t'][:i+1]
        h = len(yv)//2
        sigma = growth_rate(yv[h:], tv[h:])
        out = dict(stable=bool((not diverged) and sigma <= SIGMA_MAX),
                   diverged=bool(diverged), sigma=sigma,
                   rms_um=float(np.sqrt(np.mean(y2**2))*1e6),
                   peak_um=float(np.abs(r['y'][:i+1]).max()*1e6),
                   rms_u=float(np.sqrt(np.mean(r['u'][i0:i+1]**2))),
                   peak_u=float(np.abs(r['u'][:i+1]).max()),
                   t_end=float(r['t'][i]), dt=dt, tau=tau)
        if keep_signals:
            out['t'] = r['t'][:i+1]
            out['y'] = r['y'][:i+1]
            out['u'] = r['u'][:i+1]
        return out

    # -- limite de passe stable ---------------------------------------------
    def stability_limit(self, make_ctrl=None, rpm=4900, lo=0.02e-3, hi=1.5e-3,
                        T=T_LIMIT, tol=2e-5):
        """Plus grande profondeur de passe stable, en METRES.

        make_ctrl(dt, tau) -> correcteur neuf, ou None pour la boucle ouverte.
        La fabrique recoit dt et tau parce qu'un correcteur numerique en depend
        et que dt change avec la vitesse.

        Renvoie 0.0 si meme `lo` est instable, et `hi` si `hi` tient encore : ce
        sont de VRAIES bornes, pas des echecs silencieux.

        Les valeurs par defaut de lo, hi et tol sont celles de la base figee, de
        sorte que `stability_limit(None, rpm=4900)` rende sa valeur de reference
        (0.0489 mm). tol = 2e-5 m vaut 0.02 mm, ce qui est GROSSIER devant une
        limite libre de 0.05 mm : pour une comparaison entre correcteurs,
        prendre tol=2e-6 et hi=4.0e-3.

        ATTENTION : la stabilite N'EST PAS monotone en ap. Deux encadrements
        differents peuvent donc converger vers deux franchissements differents
        — c'est reel, pas un defaut de la bissection. D'ou l'importance de
        n'employer qu'un seul jeu (lo, hi, tol, T) dans une meme comparaison.
        """
        def ok(ap):
            dt, tau = self.sampling(rpm)
            c = None
            if make_ctrl is not None:
                try:
                    c = make_ctrl(dt, tau)
                except Exception:
                    return False
            return self.run(c, rpm=rpm, ap=ap, T=T)['stable']

        if not ok(lo):
            return 0.0
        if ok(hi):
            return hi
        while hi - lo > tol:
            mid = 0.5*(lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return 0.5*(lo + hi)

    # -- cout multi-vitesses, pour un reglage automatique --------------------
    def multi_speed_cost(self, make_ctrl, speeds=None, ap=AP_TEST, T=T_RUN,
                         penalty=12.0, w_worst=0.5):
        """Scalaire a minimiser. Une vitesse instable coute `penalty` ; le terme
        worst-case evite les reglages excellents a une vitesse et mediocres
        ailleurs."""
        speeds = speeds or SPEEDS_DEFAULT
        tot = worst = 0.0
        for rpm in speeds:
            dt, tau = self.sampling(rpm)
            try:
                c = make_ctrl(dt, tau)
            except Exception:
                tot += penalty
                continue
            r = self.run(c, rpm=rpm, ap=ap, T=T)
            if not r['stable']:
                tot += penalty
            else:
                tot += r['rms_um'] + 0.05*r['rms_u']
                worst = max(worst, r['rms_um'])
        return tot/len(speeds) + w_worst*worst

    # -- reponse frequentielle mesuree, sans hypothese de linearite ---------
    def receptance(self, make_ctrl=None, rpm=4900, f_lo=64.0, f_hi=1600.0,
                   amp=5e-3, n_period=3):
        """|Y/F| mesuree par injection d'un multisinus PERIODIQUE a la place de
        l'effort de coupe, couplage regeneratif desactive.

        Cette voie est la seule fiable pour un correcteur quelconque :
        l'extraction lineaire par reponse impulsionnelle echoue pour ceux dont
        la dynamique propre est instable — le cas de la plupart des schemas a
        observateur — leur reponse impulsionnelle divergeant.

        Retour : (frequences Hz, |Y/F| en um/N), ou (None, None) si divergence.
        """
        dt, tau = self.sampling(rpm)
        NP = 5022
        s = NewmarkIntegrator(self.plate, dt=dt, T_end=(NP*n_period + 2)*dt,
                              ft=1.0, tau=tau, verbose=False, v_max=V_MAX)
        n = s.nstep
        NP = n//n_period
        df = 1.0/(NP*dt)
        ks = np.arange(int(round(f_lo/df)), int(round(f_hi/df)) + 1)
        tp = np.arange(NP)*dt
        ph = np.pi*np.arange(len(ks))**2/len(ks)         # phases de Schroeder
        base = np.zeros(NP)
        for i, k in enumerate(ks):
            base += np.cos(2*np.pi*k*df*tp + ph[i])
        base /= np.abs(base).max()
        F = np.tile(base, n_period + 1)[:n]*amp
        c = None
        if make_ctrl is not None:
            try:
                c = make_ctrl(dt, tau)
            except Exception:
                return None, None
        r = s.simulate(F, np.zeros(n), np.zeros(n, dtype=int), controller=c,
                       progress=False, stop_threshold=1e-2)
        if r['diverged_at'] > 0:
            return None, None
        k0 = n - NP - 2
        Y = np.fft.rfft(r['y'][k0:k0+NP])
        Ff = np.fft.rfft(F[k0:k0+NP])
        return np.fft.rfftfreq(NP, dt)[ks], np.abs(Y[ks]/Ff[ks])*1e6

    # -- diagnostics du couplage --------------------------------------------
    def antiresonances(self):
        """Zeros exacts (systeme non amorti) du transfert tension -> capteur.

        G(w) = somme_i r_i/(w_i^2 - w^2) avec r_i = D_obs,i H_Pe,i. Les zeros
        sont les racines en s = w^2 du polynome somme_i r_i prod_{j!=i}(s_j - s).
        On les calcule EXACTEMENT plutot que de chercher des minima de |G| :
        avec amortissement, un intervalle SANS zero presente quand meme un
        minimum peu profond que l'on confondrait avec une antiresonance.

        Retour : frequences en Hz, croissantes.
        """
        r = (np.asarray(self.plate.D_obs).ravel()
             * np.asarray(self.plate.H_Pe_modal).ravel())
        s2 = np.asarray(self.plate.omega_n).ravel()**2
        n = len(r)
        P = np.zeros(n)
        for i in range(n):
            P += r[i]*np.poly(np.delete(s2, i))*((-1)**(n - 1))
        z = np.roots(P)
        z = z[np.abs(z.imag) <= 1e-6*max(np.abs(z.real).max(), 1e-30)].real
        return np.sort(np.sqrt(z[z > 0]))/(2*np.pi)

    def notch_occupancy(self):
        """Nombre de zeros dans chaque intervalle inter-mode.

        Signature de SIGNE, insensible a l'amortissement et a toute mise a
        l'echelle de H_Pe : l'intervalle k contient un nombre IMPAIR de zeros
        si sign(r_k) == sign(r_{k+1}), pair sinon. La mesure (Fig. 12b) donne
        (1, 1, 1, 1) ; les elements finis seuls donnent (0, 0, 1, 0).
        """
        f = self.freqs
        z = self.antiresonances()
        return tuple(int(((z > f[k]) & (z < f[k+1])).sum())
                     for k in range(len(f) - 1))


# ============================================================================
# PARTIE 6 — LE POINT DE BRANCHEMENT DU CORRECTEUR
#
# C'est le seul endroit ou une commande entre dans ce fichier. Aucune loi n'y
# est ecrite : seulement la convention d'appel.
#
# QUATRE FORMES SONT ACCEPTEES, de la plus simple a la plus complete.
#
#   1. une fonction d'une variable — la mesure, en METRES :
#
#          plant.run(lambda y: -1.0e5*y, rpm=4900, ap=0.25e-3)
#
#   2. une fonction qui nomme ce dont elle a besoin. Les noms reconnus sont
#      y, t, k, u_prev, x_tool, dt, tau ; seuls ceux que vous nommez vous sont
#      passes, dans n'importe quel ordre :
#
#          def ma_commande(y, t, u_prev):
#              return -1.0e5*y - 0.2*u_prev
#
#   3. un objet avec .step(...) suivant la meme regle de nommage, plus un
#      .reset() appele avant CHAQUE simulation. C'est la forme a utiliser des
#      qu'il y a un etat interne — observateur, filtre, integrateur :
#
#          class MonCorrecteur:
#              def __init__(self, dt, tau): ...
#              def reset(self): ...                 # remet les etats a zero
#              def step(self, y, t): return u       # en volts
#
#   4. la forme historique de ce depot, reconnue automatiquement :
#
#          def step(self, x_hat_prev, u_prev, y_meas) -> (x_hat, u)
#
#      Elle recoit en plus x_p_now et/ou k_step si elle les nomme.
#
# CONVENTIONS, VALABLES POUR LES QUATRE FORMES
#   * y est le deplacement au capteur EN METRES, mesure a la fin du pas
#     precedent — un correcteur numerique ne peut pas utiliser une grandeur
#     qu'il n'a pas encore mesuree.
#   * u est une tension EN VOLTS. Le simulateur l'ecrete lui-meme a +/- V_MAX
#     (150 V) avant de l'appliquer.
#   * u_prev est la tension SATUREE, donc ce qui a REELLEMENT ete applique. Un
#     correcteur qui sature en interne doit utiliser la MEME borne, sans quoi
#     il calculera sa commande suivante a partir d'une valeur qu'il croit
#     appliquee et qui ne l'est pas.
#   * la periode d'echantillonnage est dt = tau/82, donc elle CHANGE avec la
#     vitesse de broche. D'ou la fabrique make_ctrl(dt, tau) de
#     stability_limit : un correcteur discret doit etre reconstruit par vitesse.
#   * il n'y a PAS de bruit de mesure ni de retard capteur dans ce fichier. Les
#     ajouter est un choix d'etude, pas un defaut a corriger ici.
# ============================================================================
_HOOK_NAMES = ('y', 't', 'k', 'u_prev', 'x_tool', 'dt', 'tau')


class _ControllerHook:
    """Ramene les quatre formes acceptees a un seul appel u = hook(...).

    L'introspection se fait UNE FOIS a la construction, pas a chaque pas.
    """

    def __init__(self, ctrl, dt, tau, v_max):
        self.ctrl = ctrl
        self.const = dict(dt=dt, tau=tau)
        self.legacy = False
        self.x_hat = None
        self.v_max = v_max
        if ctrl is None:
            self.fn = None
            return
        fn = getattr(ctrl, 'step', None)
        if fn is None:
            if not callable(ctrl):
                raise TypeError(
                    "un correcteur doit etre appelable ou exposer .step() ; "
                    f"recu {type(ctrl).__name__}. Voir la PARTIE 6.")
            fn = ctrl
        self.fn = fn
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = {}
        names = [n for n, p in params.items()
                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                               p.KEYWORD_ONLY)]
        # Forme 4 : signature historique step(x_hat_prev, u_prev, y_meas).
        if (len(names) >= 3 and names[0] in ('x_hat_prev', 'x_hat', 'x_prev')
                and names[2] in ('y_meas', 'y')):
            self.legacy = True
            self.kw = tuple(n for n in ('x_p_now', 'k_step') if n in params)
            n_state = 2*getattr(getattr(ctrl, 'plate', None), 'n_modes', 0) or 2
            self.x_hat = np.zeros(n_state)
            return
        self.kw = tuple(n for n in _HOOK_NAMES if n in params)
        # Forme 1 : un seul parametre, quel que soit son nom.
        self.positional = not self.kw and len(names) >= 1

    def __call__(self, y, t, k, u_prev, x_tool):
        if self.fn is None:
            return 0.0
        if self.legacy:
            kw = {}
            if 'x_p_now' in self.kw:
                kw['x_p_now'] = x_tool
            if 'k_step' in self.kw:
                kw['k_step'] = k
            out = self.fn(self.x_hat, u_prev, y, **kw)
            self.x_hat = out[0]
            return float(out[1])
        if self.positional:
            return float(self.fn(y))
        avail = dict(y=y, t=t, k=k, u_prev=u_prev, x_tool=x_tool, **self.const)
        out = self.fn(**{n: avail[n] for n in self.kw})
        # Un correcteur peut renvoyer (u, diagnostic) ; on ne garde que u.
        return float(out[0] if isinstance(out, tuple) else out)


class ControllerTemplate:
    """SQUELETTE — ne contient AUCUNE loi de commande, et renvoie 0 V.

    Copiez-le, remplacez le corps de step(), et rien d'autre dans ce fichier
    n'a besoin de changer.

        plant = MillingPlant()
        lim = plant.stability_limit(lambda dt, tau: MonCorrecteur(dt, tau),
                                    rpm=4900)

    Le modele reduit dont vous avez besoin pour synthetiser est dans
    plant.modal : omega, zeta, H (entree), D_obs (sortie), D_tool.
    """

    def __init__(self, dt, tau, plant=None):
        self.dt = dt
        self.tau = tau            # utile a toute compensation de retard
        self.plant = plant
        self.reset()

    def reset(self):
        """Remet TOUS les etats internes a zero. Appele avant chaque
        simulation ; l'oublier fait fuir l'etat d'un essai vers le suivant et
        rend les limites de passe non reproductibles."""
        self.u = 0.0

    def step(self, y, t):
        """y : deplacement mesure [m]. Retour : tension [V], bornee a V_MAX
        par le simulateur."""
        return 0.0


# ============================================================================
# PARTIE 7 — VERIFICATION DE LA PHYSIQUE
# ============================================================================
def check_model(verbose=True, plant=None):
    """Refait les controles de validation et leve AssertionError si un ecart
    depasse la tolerance. A lancer une fois au debut d'une session.

    plant : modele a controler. Par defaut MillingPlant() ; passer
        MillingPlant(coupling='fem') montre ce que le controle 4 attrape.
    """
    if plant is None:
        plant = MillingPlant()
    ok = True

    f = plant.freqs
    err = np.abs(f/np.array(F_MEASURED) - 1)
    if verbose:
        print("1. frequences apres calibration")
        for i in range(N_MODES):
            print(f"   mode {i+1} : {f[i]:8.1f} Hz  (mesure "
                  f"{F_MEASURED[i]:8.1f}, ecart {100*err[i]:+.3f} %)")
    assert err.max() < 1e-3, "calibration des frequences incorrecte"

    # Les antiresonances de reference viennent de la Fig. 12(a), ou le marteau
    # ET le capteur sont au MEME point : c'est une reception AU POINT DE
    # FRAPPE. Les residus valent donc D_obs^2, tous positifs, et les zeros
    # s'intercalent strictement entre les poles — ce que la mesure confirme
    # (734 < 1072, 2161 < 2779, 3001 < 3359, 3805 < 4123). Utiliser
    # D_obs*D_tool comparerait une reception de TRANSFERT a une mesure au point
    # de frappe : residus de signes mixtes, motif de zeros sans rapport.
    ff = np.linspace(60, 4600, 60000)
    w = 2*np.pi*ff
    m = plant.modal
    den = ((m['omega'][:, None]**2 - w[None, :]**2)
           + 2j*m['zeta'][:, None]*m['omega'][:, None]*w[None, :])
    G = np.abs(np.sum((m['D_obs']**2)[:, None]/den, axis=0))
    lg = np.log(G)
    anti = [ff[i] for i in range(1, len(G)-1)
            if lg[i] < lg[i-1] and lg[i] < lg[i+1]]
    target = [734, 2161, 3001, 3805]
    if verbose:
        print("2. antiresonances (reception au point de frappe, Fig. 12a)")
        print(f"   mesurees : {target}")
        print(f"   modele   : {[round(a) for a in anti]}")
    assert len(anti) >= 4, "le modele ne produit pas quatre antiresonances"
    for t in target:
        d = min(abs(a - t) for a in anti)
        if verbose:
            print(f"   {t:5d} Hz -> ecart {d:5.0f} Hz ({100*d/t:+.1f} %)")
        if d/t > 0.05:
            ok = False
            print(f"   ATTENTION : antiresonance {t} Hz mal reproduite")

    # Niveau absolu. On ne peut PAS le confronter a la Fig. 12 (son echelle en
    # dB est inexploitable, cf. en-tete) ; on verifie donc la coherence INTERNE
    # du modele, ce qui attrape une erreur de normalisation modale ou de D_obs
    # sans rien emprunter a l'article.
    pl = plant.plate
    N_obs = shape_at_point(*SENSOR_XZ, MESH_N1, MESH_N2,
                           pl.lex, pl.ley, pl.n1)[pl.DOFf]
    C_exact = float(N_obs @ spsolve(pl.K_free.tocsc(), N_obs))
    C_modal = float(np.sum(np.asarray(pl.D_obs)**2
                           / np.asarray(pl.omega_n)**2))
    D_flex = YOUNG*PLATE_T**3/(12*(1 - POISSON**2))
    C_beam = PLATE_H**3/(3*D_flex*PLATE_L)
    if verbose:
        print("3. niveau absolu (coherence interne, sans recours a la Fig. 12)")
        print(f"   souplesse statique EF exacte      : {C_exact*1e6:7.3f} um/N")
        print(f"   modele modal a {N_MODES} modes           : "
              f"{C_modal*1e6:7.3f} um/N  ({100*C_modal/C_exact:.1f} % de l'exacte)")
        print(f"   borne inf. poutre pleine largeur  : {C_beam*1e6:7.3f} um/N"
              f"  (rapport {C_exact/C_beam:.2f})")
        print(f"   Fig. 12(a) lue a 1 um/N           :   43.83 um/N"
              f"  -> facteur {43.83e-6/C_exact:.2f}, ECHELLE INEXPLOITABLE")
        print("   => le NIVEAU de H_Pe n'est pas valide ; balayer gain_H.")
    assert 0.80 < C_modal/C_exact < 1.05, \
        "troncature modale incoherente avec la statique exacte"
    assert 1.0 < C_exact/C_beam < 3.0, \
        "souplesse statique hors des bornes physiques d'une plaque encastree"

    # Repartition modale du couplage. Les controles 2 et 3 ne portent que sur
    # D_obs (leurs residus valent D_i^2) : ils passeraient tels quels avec un
    # H_Pe entierement faux, ce qui a longtemps ete le cas ici. Ce test-ci
    # ferme cette porte. Les zeros du transfert tension -> capteur sont une
    # signature de SIGNE, insensible a toute mise a l'echelle de H_Pe — donc au
    # niveau non valide.
    zc = plant.antiresonances()
    NOTCH_MEAS = np.array([787.5, 1493.4, 2912.8, 3608.9])       # Fig. 12(b)
    if verbose:
        print("4. creux de la Fig. 12(b) : repartition modale de H_Pe")
        print(f"   mesures : {np.round(NOTCH_MEAS, 0)}")
        print(f"   modele  : {np.round(zc, 0)}")
    if len(zc) != len(NOTCH_MEAS):
        ok = False
        if verbose:
            print(f"   ATTENTION : {len(zc)} creux calcules contre "
                  f"{len(NOTCH_MEAS)} mesures — la repartition modale de H_Pe "
                  f"est fausse (coupling='fem' ?)")
    else:
        e = np.abs(zc - NOTCH_MEAS)/NOTCH_MEAS*100
        if verbose:
            print(f"   ecarts  : {np.round(e, 2)} %  (max {e.max():.2f} %)")
            print(f"   occupation des intervalles : {plant.notch_occupancy()} "
                  f"(mesure (1, 1, 1, 1))")
        if e.max() > 3.0:
            ok = False
            if verbose:
                print("   ATTENTION : creux mal reproduits")

    lim = plant.stability_limit(None, rpm=4900)
    if verbose:
        print(f"5. limite de passe en boucle ouverte a 4900 tr/min : "
              f"{lim*1e3:.4f} mm  (reference : 0.0489 mm)")
        print("   cette valeur depend de l'horizon T et de la fenetre de "
              "mesure ; c'est")
        print("   la reference de CE modele, a ne pas comparer a une valeur "
              "obtenue")
        print("   avec d'autres reglages d'evaluation.")
    assert abs(lim*1e3 - 0.0489) < 0.008, "limite libre hors tolerance"

    if verbose:
        print("\nmodele valide." if ok else "\nmodele utilisable, voir avertissements.")
    return plant


# ============================================================================
if __name__ == "__main__":
    import time

    t0 = time.time()
    print("=" * 74)
    print("VERIFICATION DU MODELE")
    print("=" * 74)
    plant = check_model()

    print()
    print("=" * 74)
    print("LE MODELE REDUIT (ce dont un correcteur a besoin)")
    print("=" * 74)
    plant.describe()
    dt, tau = plant.sampling(4900)
    print(f"\na 4900 tr/min : tau = {tau*1e3:.3f} ms, dt = tau/82 = "
          f"{dt*1e6:.2f} us  ({1/dt:.0f} Hz)")

    print()
    print("=" * 74)
    print("BOUCLE OUVERTE — limite de passe stable, en mm")
    print("=" * 74)
    print("  (aucune commande n'est appliquee : u = 0 a tous les pas)")
    lims = []
    for rpm in SPEEDS_DEFAULT:
        lim = plant.stability_limit(None, rpm=rpm, lo=0.02e-3, hi=4.0e-3,
                                    tol=2e-6)*1e3
        lims.append(lim)
        print(f"  {rpm:5d} tr/min : {lim:.4f} mm", flush=True)
    print(f"\n  pire cas : {min(lims):.4f} mm — c'est la valeur qu'une commande")
    print("  doit ameliorer, et la seule a laquelle la comparer.")
    print(f"\ntermine en {time.time() - t0:.0f} s")
