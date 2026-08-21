"""monodromy.py — spectre de la matrice de monodromie SANS l'assembler.

POURQUOI CE MODULE EXISTE : l'iteration de puissance qui vivait dans
`closed_loop.spectral_radius` et `lti_floquet.spectral_radius` rendait des
rayons spectraux FAUX, et son critere d'arret ne pouvait pas s'en apercevoir.

Mesure, correcteur FOPID stocke (results/pso_B.npz), 5 modes, m = 24,
a_p = 0.10 mm, position mediane. La monodromie assemblee explicitement
(dimension (m+1)*nx = 650) a pour rayon spectral EXACT 0.967392 ; l'iteration
de puissance adaptative rendait, selon la graine :

    graine        0          1          2
    rho       0.96690    0.79107    0.96756

et sur le FDOB au meme point, 0.9142 / 0.9138 / 0.9108 pour un exact de
0.959809 — 4.8 % d'erreur, TOUJOURS PAR DEFAUT, donc une coupe presentee comme
plus stable qu'elle ne l'est et une a_p,lim surestimee.

MECANISME. La monodromie de ce systeme est tres NON NORMALE : la matrice de
ses vecteurs propres a un conditionnement de 1.35e29 sur le cas FDOB. Le
facteur de croissance par periode traverse alors une LONGUE HUITIEME PLATE a
une valeur qui n'est pas la bonne avant de rejoindre le mode dominant. Trace
du cas a graine 1 ci-dessus, log rho par periode (exact : -0.033151) :

    periodes      20-29    30-39    40-49    50-59    60-69    70-79    75-84
    log rho      -0.2335  -0.2356  -0.2346  -0.1899  -0.0802  -0.0373  -0.0334

Entre les periodes 20 et 40 deux fenetres NON RECOUVRANTES de dix periodes
s'accordent a 1e-3 : le test d'arret se declenche, l'iteration rend
exp(-0.234) = 0.791, et la vraie valeur n'est atteinte qu'apres la periode 75.
AUCUN reglage du couple (tolerance, nombre de fenetres) ne distingue une
huitieme plate d'une convergence — c'est la methode qui est inadaptee, pas son
seuil.

CE QUE FAIT CE MODULE. Arnoldi redemarre implicitement (ARPACK) sur
l'application d'UNE periode vue comme operateur implicite : la monodromie
(m+1)*nx n'est jamais formee, seul le produit matrice-vecteur l'est, exactement
comme dans l'iteration de puissance. Sur douze cas de reference (quatre
structures x trois profondeurs) confrontes a la monodromie assemblee, l'accord
est de SIX chiffres sur douze cas sur douze, et le cout est 3 a 15 fois plus
FAIBLE que l'iteration de puissance qu'il remplace (m = 200 : 0.12 s contre
1.95 s). Arnoldi n'a pas de huitieme plate parce qu'il ne mesure pas un taux
de croissance : il construit un sous-espace de Krylov et lit les valeurs
propres de la projection.

BONUS : les valeurs propres sortent COMPLEXES, donc avec leur phase. La
frequence de broutement en boucle fermee se lit sur le meme calcul, sans le
detour par une iteration de sous-espace separee.

LE COUT MONTE QUAND ON APPROCHE DU BUT, et c'est inherent a la methode.
Arnoldi converge d'autant plus vite que le multiplicateur dominant est
SEPARE des suivants. Loin de la frontiere de stabilite il l'est largement ;
au voisinage de l'optimum plusieurs multiplicateurs se pressent autour de
|mu| = 1 et ARPACK redemarre davantage. Mesure sur le FDOB a cinq modes
vises (56 etats), cout d'Arnoldi par point : 0.035 s sur des tirages au
hasard, 0.123 s a l'optimum stocke — et l'evaluation complete passe de 1.7 s
a 6.7 s. Estimer la duree d'une campagne sur un echantillon aleatoire la
sous-estime donc d'un facteur quatre, parce que l'essaim passe justement sa
vie a se rapprocher de la region chere.

Le decalage de l'historique se fait par TAMPON CIRCULAIRE et non par np.roll :
`roll` recopie les (m+1)*nx elements a chaque sous-intervalle, soit un cout en
m^2 par periode (1.7 million de recopies par periode a m = 200, nx = 42) pour
un decalage qui est un simple changement d'indice.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, ArpackNoConvergence


def apply_period(maps, Z, m):
    """Une periode de dent appliquee au bloc d'historique Z.

    Z : (m+1, nx) ou (m+1, nx, q). Z[j] = x_{k-j}. Retourne le nouvel
    historique dans la meme convention (tampon circulaire ramene a l'origine).
    """
    Z = np.array(Z, dtype=float, copy=True)
    nh = m + 1
    head = 0
    for P0, C_lo, C_hi in maps:
        new = P0 @ Z[head] + C_lo @ Z[(head + m) % nh] + C_hi @ Z[(head + m - 1) % nh]
        head = (head - 1) % nh
        Z[head] = new
    return np.roll(Z, -head, axis=0)


def _explicit(maps, m, nx):
    """Monodromie assemblee — reference exacte, reservee aux petites tailles."""
    dim = (m + 1) * nx
    Phi = np.eye(dim)
    for P0, C_lo, C_hi in maps:
        Dk = np.zeros((dim, dim))
        Dk[:nx, :nx] = P0
        Dk[:nx, dim - nx:] = C_lo
        # a m = 1 le bloc "k-m+1" EST le bloc courant : le += est ce qui evite
        # d'ecraser P0 (meme piege que dans stability_fdm.floquet_matrix).
        Dk[:nx, dim - 2 * nx:dim - nx] += C_hi
        Dk[nx:, :dim - nx] = np.eye(dim - nx)
        Phi = Dk @ Phi
    return Phi


def dominant(maps, m, nx, k=4, seed=0, tol=1e-9):
    """Les k multiplicateurs de Floquet dominants, AVEC leur phase.

    Rend un tableau complexe, ou [inf] si l'application diverge
    numeriquement (profondeur trop grande : la matrice d'etat explose avant
    que le spectre ait un sens)."""
    dim = (m + 1) * nx
    for P0, C_lo, C_hi in maps:
        if not (np.all(np.isfinite(P0)) and np.all(np.isfinite(C_lo))
                and np.all(np.isfinite(C_hi))):
            return np.array([np.inf])
    # En dessous de cette taille l'assemblage explicite coute moins qu'ARPACK
    # et rend le spectre COMPLET ; c'est aussi le seul regime ou la contrainte
    # k < dim - 1 d'ARPACK devient genante.
    if dim <= 64:
        w = np.linalg.eigvals(_explicit(maps, m, nx))
        return w[np.argsort(-np.abs(w))][:k]
    k = int(min(k, dim - 2))
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(dim)
    op = LinearOperator(
        (dim, dim),
        matvec=lambda v: apply_period(maps, v.reshape(m + 1, nx), m).ravel(),
        dtype=float)
    try:
        w = eigs(op, k=k, ncv=min(dim - 1, max(2 * k + 2, 24)), which='LM',
                 tol=tol, v0=v0, return_eigenvectors=False, maxiter=10000)
    except ArpackNoConvergence as exc:
        w = exc.eigenvalues
        if w is None or len(w) == 0:
            return np.array([np.inf])
    w = np.asarray(w)
    if not np.all(np.isfinite(w)):
        return np.array([np.inf])
    return w[np.argsort(-np.abs(w))]


def spectral_radius(maps, m, nx, seed=0, k=4, tol=1e-9):
    """Rayon spectral de la monodromie."""
    w = dominant(maps, m, nx, k=k, seed=seed, tol=tol)
    if w.size == 0:
        return 0.0
    return float(np.max(np.abs(w)))
