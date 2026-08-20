"""
step_integrals.py — les deux integrales d'un pas, SANS jamais inverser A
=========================================================================
Les trois moteurs de Floquet du depot — `stability_fdm.floquet_matrix`,
`lti_floquet.period_maps` et `control/closed_loop.period_maps` — ont besoin
des memes deux integrales sur un sous-intervalle de longueur h :

    J1 = int_0^h e^{As} ds
    J2 = int_0^h e^{A(h-s)} s ds  =  h.J1 - int_0^h (h-s) e^{As} ds

Le chemin rapide les tire de `solve(A, e^{Ah} - I)`. Il suppose A inversible,
et cette hypothese TOMBE, de deux facons distinctes et toutes deux mesurees
ici :

  * mathematiquement, la raideur effective K0 + alpha4(t).D^T D est une
    correction de RANG UN de K0 ; a certaines profondeurs et a certains
    instants de la periode de dent elle passe exactement par zero, et A avec
    elle. Rien d'anormal : c'est le point ou la coupe annule la raideur
    apparente d'un mode.
  * numeriquement, la matrice d'etat en boucle fermee atteint un
    conditionnement de 1e21 avec les correcteurs les plus riches (banc modal
    a cinq modes, 66 etats), et la factorisation LU finit par rencontrer un
    pivot exactement nul.

Dans les deux cas les integrales elles-memes restent parfaitement definies :
seule la FORMULE choisie pour les calculer echoue. La forme augmentee les
donne sans aucune inversion,

    expm([[A, I, 0], [0, 0, I], [0, 0, 0]] h)
      = [[e^{Ah}, F1, F2], [0, I, hI], [0, 0, I]]

avec F1 = J1 et F2 = int_0^h (h-s) e^{As} ds, d'ou J2 = h.F1 - F2.

Le chemin lent n'est pris qu'en RATTRAPAGE : il coute un expm sur une matrice
trois fois plus grande. Les deux chemins ont ete compares la ou tous deux
aboutissent — ecart relatif de 2e-12 (FOPID) a 4e-7 (ADRC-FOPID) — donc ils
sont interchangeables, et `tests/test_invariants.py` le reverifie.
"""
import numpy as np
from scipy.linalg import expm


#: marge sur la BORNE de ||J1||. La borne est exacte ; la marge n'absorbe
#: que l'erreur d'arrondi de expm.
BOUND_SLACK = 1e-6

#: nombre de fois que le chemin augmente a du rattraper le chemin rapide,
#: depuis le chargement du module. Diagnostic, jamais une condition.
n_fallback = 0


def step_integrals(A, h, P0=None):
    """(P0, J1, J2) pour un pas de longueur h. P0 = e^{Ah} est recalcule si
    l'appelant ne le fournit pas.

    LA GARDE N'EST PAS `except LinAlgError`, ET C'EST LE POINT DELICAT.
    `np.linalg.solve` ne leve QUE sur un pivot exactement nul. Sur une
    matrice seulement TRES mal conditionnee elle reussit et rend un resultat
    FINI mais FAUX, sans le moindre signal. Mesure sur une matrice 8x8 de
    conditionnement 1.6e16 avec h = 1e-4 : le chemin rapide rend un J1 dont
    le plus grand element vaut 1.66e-2 quand la valeur exacte est 1.00e-4,
    soit un facteur 166. Un `try/except` seul laisse passer exactement
    ce cas-la, et une carte de Floquet fausse ne se voit pas : elle donne
    juste un rayon spectral, donc une limite de stabilite, qui n'est pas la
    bonne.

    LE RESIDU NE SUFFIT PAS NON PLUS, et c'est le piege suivant : dans une
    direction quasi-nulle de A, une solution fausse d'un facteur cent garde
    un residu ||A.J1 - (e^{Ah} - I)|| minuscule. C'est la definition meme du
    mauvais conditionnement.

    On se rabat donc sur une BORNE, qui elle ne se laisse pas tromper :

        ||J1|| = ||int_0^h e^{As} ds|| <= int_0^h e^{||A||s} ds
               = (e^{||A|| h} - 1) / ||A||

    en norme induite (ici la norme infinie). Elle est ATTEINTE pour
    A = ||A||.I, donc on ne peut pas la resserrer, et tout J1 qui la depasse
    est faux — sans avoir a savoir pourquoi.
    """
    global n_fallback
    A = np.asarray(A, float)
    nx = A.shape[0]
    if P0 is None:
        P0 = expm(A * h)
    R = P0 - np.eye(nx)
    na = float(np.max(np.sum(np.abs(A), axis=1)))          # ||A||_inf
    # La borne se compare EN LOGARITHME : ||A||.h depasse couramment 700 sur
    # ce modele (raideurs modales de 1e7), et un expm1 direct deborderait,
    # ce qui rendrait une borne infinie — donc une garde qui ne garde rien.
    lah = na * h
    log_bound = (lah - np.log(na) if lah > 30.0
                 else np.log(np.expm1(lah) / na)) if na > 0.0 else np.log(h)
    log_bound += BOUND_SLACK
    try:
        J1 = np.linalg.solve(A, R)
        J2 = h * J1 - np.linalg.solve(A, h * P0 - J1)
        n1 = float(np.max(np.sum(np.abs(J1), axis=1)))
        if (np.all(np.isfinite(J1)) and np.all(np.isfinite(J2))
                and (n1 <= 0.0 or np.log(n1) <= log_bound)):
            return P0, J1, J2
    except np.linalg.LinAlgError:
        pass
    n_fallback += 1
    I = np.eye(nx)
    Z = np.zeros((nx, nx))
    E = expm(np.block([[A, I, Z], [Z, Z, I], [Z, Z, Z]]) * h)
    J1 = E[:nx, nx:2 * nx]
    return P0, J1, h * J1 - E[:nx, 2 * nx:]
