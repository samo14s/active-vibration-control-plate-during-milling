"""
nmp_dob.py — l'observateur qui REGARDE le zero instable au lieu de l'eviter
============================================================================
Premiere des structures que le diagnostic PRESCRIT plutot qu'il ne constate.

CE QUE LE DIAGNOSTIC AVAIT ETABLI. Le procede complet porte un zero dans le
demi-plan droit a 2459 Hz (`DIAGNOSTIC_ADRC.md` §6), consequence geometrique
du fait que l'actionneur et le capteur ne sont pas au meme endroit. Il en
decoule que P^-1 est INSTABLE : aucun observateur de perturbation ne peut
inverser le procede complet. `fdob.py` contournait la difficulte en inversant
MODE PAR MODE — chaque mode pris seul est a minimum de phase (invariant
`test_reduced_model_is_minimum_phase`), donc chaque inverse modal existe.

Le contournement marche, mais il paie deux fois. Il n'inverse jamais le
procede que la boucle voit reellement, et il oblige a choisir a l'avance
QUELS modes viser — un choix dont `OBSERVATEUR_MODAL.md` §6.7 a mesure le
prix quand on vise les cinq.

CE QUE CETTE STRUCTURE FAIT A LA PLACE. La factorisation interieure-exterieure

    P(s) = B(s) . P_min(s),     B(s) = (z - s)/(z + s),   |B(jw)| = 1

separe le procede en une partie TOUT-PASSE, qui porte le zero instable et rien
d'autre, et une partie a MINIMUM DE PHASE qui porte tout le reste. B ne change
aucun module : il ne fait que de la phase. Donc P_min a EXACTEMENT le meme
module que P a toute frequence, et son inverse, lui, est stable.

L'observateur inverse alors `P_min`, c'est-a-dire tout ce qui est
inversible — ni plus (ce serait instable) ni moins (ce que fait l'inversion
modale). Ce n'est pas une astuce : c'est le meilleur inverse stable qui
existe, et le retard de phase de B est exactement le prix incompressible que
le theoreme de Poisson faisait deja payer.

CE QUE CELA NE FAIT PAS, ET IL FAUT LE DIRE. Cela ne supprime pas le plafond.
Le zero instable reste, la contrainte d'interpolation S(z) = 1 reste, et la
borne de Poisson reste la meme — elle ne depend que de la position du zero,
pas du correcteur. Ce qui change est l'usage qu'on fait du budget sous ce
plafond : l'inversion modale en gaspille une part en n'inversant qu'une
partie de ce qui etait inversible. La question mesurable est donc : cette
part vaut-elle quelque chose sur cette plaque ? Le protocole y repondra, et
la reponse peut tres bien etre non.
"""
import numpy as np
from scipy.signal import tf2ss, tf2zpk, zpk2tf


def plant_tf(w, zeta, res):
    """(num, den) du procede modal complet."""
    w = np.atleast_1d(np.asarray(w, float))
    zeta = np.atleast_1d(np.asarray(zeta, float))
    res = np.atleast_1d(np.asarray(res, float))
    den = np.array([1.0])
    for wk, zk in zip(w, zeta):
        den = np.convolve(den, [1.0, 2 * zk * wk, wk ** 2])
    num = np.zeros(1)
    for i, (wk, zk, rk) in enumerate(zip(w, zeta, res)):
        other = np.array([1.0])
        for j, (wj, zj) in enumerate(zip(w, zeta)):
            if j != i:
                other = np.convolve(other, [1.0, 2 * zj * wj, wj ** 2])
        num = np.polyadd(num, rk * other)
    return num, den


def inner_outer(num, den, tol=1e-9):
    """(P_min, B) : partie a minimum de phase et partie tout-passe.

    Les zeros du demi-plan droit sont REFLECHIS (z -> -z) et le gain est
    corrige pour que |P_min| = |P| exactement. Le facteur tout-passe rassemble
    les zeros d'origine et leurs miroirs en poles.

    On ne touche PAS aux poles : le procede est stable, sa partie exterieure
    l'est donc aussi.
    """
    z, p, k = tf2zpk(num, den)
    rhp = z[z.real > tol]
    if rhp.size == 0:
        return (np.asarray(num, float), np.asarray(den, float)), \
            (np.array([1.0]), np.array([1.0])), rhp
    z_min = np.where(z.real > tol, -np.conj(z), z)
    # Reflechir z -> -conj(z) multiplie le module du produit des zeros par 1 :
    # |jw - z| et |jw + conj(z)| sont egaux. Le gain est donc INCHANGE, mais
    # on le reverifie plus bas au lieu de le supposer.
    num_min, den_min = zpk2tf(z_min, p, k)
    # B(s) = prod (s - z_i)/(s + z_i).
    #
    # L'ORIENTATION COMPTE, et la convention repandue (z-s)/(z+s) est celle
    # qui NE reconstruit PAS. Avec elle, B.P_min = -P : le module est juste,
    # la phase est juste a pi pres, et un observateur bati dessus inverserait
    # le signe de la boucle sans que ni |P_min| = |P| ni |B| = 1 ne s'en
    # apercoivent. Les deux controles evidents passent donc, et seul le
    # controle de RECONSTRUCTION attrape la faute — c'est pour cela qu'il est
    # dans la liste.
    bn, bd = np.array([1.0]), np.array([1.0])
    for zi in rhp:
        bn = np.convolve(bn, [1.0, -zi.real])
        bd = np.convolve(bd, [1.0, zi.real])
    return ((np.real(num_min), np.real(den_min)),
            (np.real(bn), np.real(bd)), rhp)


def stable_inverse_ss(num, den, wc, order=None):
    """Inverse STABLE et propre de P_min, filtre passe-bas d'ordre suffisant.

    P_min^-1 = den/num est impropre : on le rend propre en le multipliant par
    Q(s) = (wc/(s+wc))^r avec r = deg(den) - deg(num). C'est le meme role que
    joue Q dans un observateur de perturbation classique, a ceci pres qu'ici
    l'inverse existe vraiment.
    """
    num = np.trim_zeros(np.asarray(num, float), 'f')
    den = np.asarray(den, float)
    r = len(den) - len(num) if order is None else int(order)
    r = max(r, 0)
    qn, qd = np.array([1.0]), np.array([1.0])
    for _ in range(r):
        qn = np.convolve(qn, [float(wc)])
        qd = np.convolve(qd, [1.0, float(wc)])
    inv_n = np.convolve(den, qn)
    inv_d = np.convolve(num, qd)
    return tf2ss(inv_n, inv_d)


def frf_tf(num, den, om):
    s = 1j * np.asarray(om, float)
    return np.polyval(num, s) / np.polyval(den, s)


def check_factorization(w, zeta, res, om=None):
    """Verifie ce que la factorisation PROMET, au lieu de le supposer.

    Rend un rapport : ecart de module entre P et P_min, nombre de zeros
    instables restants dans P_min, et module du tout-passe.
    """
    if om is None:
        om = 2 * np.pi * np.logspace(0, 4.5, 2000)
    num, den = plant_tf(w, zeta, res)
    (nm, dm), (bn, bd), rhp = inner_outer(num, den)
    P = frf_tf(num, den, om)
    Pm = frf_tf(nm, dm, om)
    B = frf_tf(bn, bd, om)
    zmin = tf2zpk(nm, dm)[0]
    return dict(
        n_rhp=int(rhp.size),
        f_rhp=np.sort(rhp.real) / (2 * np.pi),
        mag_err=float(np.max(np.abs(np.abs(Pm) - np.abs(P))
                             / np.maximum(np.abs(P), 1e-300))),
        allpass_err=float(np.max(np.abs(np.abs(B) - 1.0))),
        n_rhp_after=int(np.sum(zmin.real > 1e-9)),
        recon_err=float(np.max(np.abs(B * Pm - P) / np.maximum(np.abs(P),
                                                               1e-300))))
