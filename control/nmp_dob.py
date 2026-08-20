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

    P(s) = B(s) . P_min(s),     B(s) = (s - z)/(s + z),   |B(jw)| = 1

(l'orientation n'est pas indifferente : avec (z-s)/(z+s), pourtant la
convention la plus courante, on obtient B.P_min = -P — voir `inner_outer`)

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
from scipy.linalg import matrix_balance
from scipy.signal import tf2ss, tf2zpk, zpk2tf

from fopid import series


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


def _q_tf(wq, r):
    """Q(s) = (wq/(s+wq))^r."""
    qn, qd = np.array([1.0]), np.array([1.0])
    for _ in range(int(r)):
        qn = np.convolve(qn, [float(wq)])
        qd = np.convolve(qd, [1.0, float(wq)])
    return qn, qd


#: ordre du filtre Q. Il ne se choisit pas librement : W = Q.P_min^-1 doit
#: etre STRICTEMENT propre pour qu'il n'y ait pas de boucle algebrique dans le
#: montage (meme raison que Dv = Dw = 0 dans fdob.py). Avec P_min de degres
#: (2n-2)/(2n), cela impose r > 2, donc r = 3 — exactement l'ordre du filtre
#: implicite Q = (wo/(s+wo))^3 de l'observateur etendu de l'ADRC, ce qui rend
#: les deux structures comparables filtre pour filtre.
Q_ORDER = 3


# ---------------------------------------------------------------------------
# POURQUOI W = Q.P_min^-1 EST REALISE EN CASCADE, ET NON PAR tf2ss
# ----------------------------------------------------------------
# La version precedente formait W en CONVOLUANT les polynomes puis appelait
# `tf2ss` dessus. Le denominateur obtenu est de DEGRE 11 et ses coefficients
# s'etalent sur 6.5e45 (le numerateur sur 4.7e40) : la forme compagne les
# recopie tels quels dans une meme matrice, dont le conditionnement en norme 2
# vaut alors 2.5e70 — trente ordres de grandeur au-dela de 1/eps.
#
# Le piege est que cette matrice reste TRES BONNE en frequentiel : la reponse
# rendue par (A, B, C, D) collait encore a 1.3e-12 pres a la fraction
# rationnelle, parce que `ss_frf` resout (jw I - A) x = B et n'est donc pas
# sensible a l'echelle. Tous les controles frequentiels passaient, et la
# structure rendait a alpha = 0 exactement la fonction de transfert du FOPID
# stocke (accord a 1e-15, memes Ms = 1.97 et V = 153 V/N).
#
# Ce qui ne passait pas est le calcul de FLOQUET, qui travaille sur la
# MATRICE D'ETAT et non sur la reponse : a alpha = 0 il rendait J = +0.0000
# la ou le FOPID stocke, de fonction de transfert IDENTIQUE, vaut J = +0.2479.
# Les etats de W ne sont pourtant pas observes quand alpha = 0 — mais ils
# restent dans la matrice augmentee, et leur echelle y detruit tout.
#
# C'est la meme faute que celle deja corrigee dans `fopid.frac_ss` (forme
# modale au lieu de la forme compagne de `zpk2ss`) et dans `fdob._tf`
# (equilibrage), et elle se corrige ici de la meme facon : on ne forme JAMAIS
# le polynome de degre 11. On factorise W en sections d'ordre <= 2 — ou la
# forme compagne est inoffensive, ses coefficients etant w et w^2 — et on les
# met en SERIE. Le gain global, qui vaut wq^3/k et atteint 1e12, est reparti
# entre les sections au lieu d'etre porte par une seule.
# ---------------------------------------------------------------------------
def _real_factors(roots, tol=1e-9):
    """Racines -> facteurs moniques REELS de degre 1 ou 2.

    Chaque paire conjuguee donne son trinome s^2 - 2Re(r)s + |r|^2 ; les
    racines reelles sont appariees deux a deux par module croissant (une
    section d'ordre 2 par paire), la derniere restant seule si leur nombre est
    impair.
    """
    roots = np.asarray(roots, complex)
    scale = np.maximum(np.abs(roots), 1.0)
    is_real = np.abs(roots.imag) <= tol * scale
    real = np.sort(roots[is_real].real)
    cplx = roots[~is_real]
    cplx = cplx[cplx.imag > 0.0]
    fac = [np.array([1.0, -2.0 * r.real, float(abs(r) ** 2)]) for r in cplx]
    real = real[np.argsort(np.abs(real))]
    i = 0
    while i + 1 < real.size:
        fac.append(np.polymul([1.0, -real[i]], [1.0, -real[i + 1]]))
        i += 2
    if i < real.size:
        fac.append(np.array([1.0, -real[i]]))
    if sum(len(f) - 1 for f in fac) != roots.size:
        raise ValueError('racines complexes non conjuguees : factorisation '
                         'reelle impossible')
    return fac


def _wchar(fac):
    """Pulsation caracteristique d'un facteur monique : |prod racines|^(1/deg)."""
    d = len(fac) - 1
    return abs(fac[-1]) ** (1.0 / d)


def _sections(zeros, poles):
    """Appariement zeros/poles en sections PROPRES d'ordre <= 2.

    On apparie par pulsation caracteristique croissante et a degre egal quand
    c'est possible : chaque section garde ainsi un gain d'ordre 1 dans sa
    propre bande, ce qui est la condition pour que le gain global se reparte
    sans qu'aucune section ne porte un facteur d'echelle enorme.
    """
    zf = sorted(_real_factors(zeros), key=_wchar)
    pf = sorted(_real_factors(poles), key=_wchar)
    used = [False] * len(zf)
    sec = []
    for d in pf:
        pick = None
        for same in (True, False):
            for k, n in enumerate(zf):
                if used[k]:
                    continue
                if (len(n) == len(d)) if same else (len(n) < len(d)):
                    pick = k
                    break
            if pick is not None:
                break
        if pick is None:
            sec.append((np.array([1.0]), d))
        else:
            used[pick] = True
            sec.append((zf[pick], d))
    if not all(used):
        raise ValueError('appariement zeros/poles impossible : W n\'est pas '
                         'propre section par section')
    return sec


def _balanced_tf_ss(num, den):
    """(A, B, C, D) d'une section, EQUILIBREE (similitude diagonale).

    `permute=False` n'est pas un detail : avec la permutation, `matrix_balance`
    rend un T qui n'est plus diagonal et l'inverse par 1/diag(T) fabrique des
    inf puis des NaN.
    """
    A, B, C, D = tf2ss(np.asarray(num, float), np.asarray(den, float))
    A = np.atleast_2d(A)
    B = np.atleast_2d(B).reshape(-1, 1)
    C = np.atleast_2d(C).reshape(1, -1)
    Ab, T = matrix_balance(A, permute=False, separate=False)
    Ti = np.diag(1.0 / np.diag(T))
    return Ab, Ti @ B, C @ T, np.atleast_2d(D).reshape(1, 1)


def _cascade_ss(zeros, poles, gain):
    """Realisation en CASCADE de la fonction gain.prod(s-z)/prod(s-p).

    Le gain global est REPARTI : chaque section est normalisee a son propre
    module a SA pulsation caracteristique, puis toutes recoivent le meme
    facteur, de sorte qu'aucune ne porte seule wq^3/k (~1e12 ici).
    """
    sec = _sections(zeros, poles)
    m = len(sec)
    a = np.array([abs(np.polyval(n, 1j * _wchar(d)) / np.polyval(d, 1j * _wchar(d)))
                  for n, d in sec])
    g = (abs(gain) * np.prod(a)) ** (1.0 / m) / a
    g[0] *= np.sign(gain)
    out = None
    for (n, d), gi in zip(sec, g):
        blk = _balanced_tf_ss(gi * np.asarray(n, float), d)
        out = blk if out is None else series(out, blk)
    A, B, C, D = out
    # Equilibrage FINAL : la mise en serie recree du couplage B_i C_j entre
    # sections d'echelles differentes (conditionnement ~1e16 sans lui). Une
    # similitude diagonale ne change pas la fonction de transfert.
    Ab, T = matrix_balance(A, permute=False, separate=False)
    return Ab, np.diag(1.0 / np.diag(T)) @ B, C @ T, D


def _w_ss(nm, dm, wq, r=Q_ORDER):
    """W = Q . P_min^-1 en cascade : zeros = POLES de P_min, poles = ZEROS de
    P_min plus les r poles de Q en -wq."""
    nm = np.trim_zeros(np.asarray(nm, float), 'f')
    dm = np.asarray(dm, float)
    zeros = np.roots(dm)
    poles = np.concatenate([np.roots(nm), np.full(int(r), -float(wq))])
    gain = float(wq) ** int(r) * dm[0] / nm[0]
    return _cascade_ss(zeros, poles, gain)


def nmp_dob_fopid_ss(Kp, Ki, Kd, lam, mu, wq, alpha, w, zeta, res,
                     wb, wh, N, sign_loop=1.0):
    """FOPID + observateur inversant P_min (partie a minimum de phase).

    Meme montage que `fdob.fdob_fopid_ss`, a ceci pres que l'inversion porte
    sur le procede COMPLET debarrasse de son seul facteur non inversible, au
    lieu d'une somme d'inverses modaux.

    Etats : [FOPID (Oustaloup), W = Q P_min^-1, V = Q].
    """
    from fopid import fopid_ss

    Ac, Bc, Cc, Dc = [np.atleast_2d(np.asarray(m, float))
                      for m in fopid_ss(Kp, Ki, Kd, lam, mu, wb, wh, N,
                                        sign_loop)]
    num, den = plant_tf(w, zeta, res)
    (nm, dm), _, _ = inner_outer(num, den)
    qn, qd = _q_tf(wq, Q_ORDER)
    nm = np.trim_zeros(np.asarray(nm, float), 'f')
    # V = Q ; W = Q . P_min^-1 = (qn . dm) / (qd . nm)
    Av, Bv, Cv, Dv = tf2ss(qn, qd)
    # W n'est PAS forme comme un polynome : voir la note ci-dessus.
    Aw, Bw, Cw, Dw = _w_ss(nm, dm, wq, Q_ORDER)
    Av, Bv, Cv = np.atleast_2d(Av), np.atleast_2d(Bv), np.atleast_2d(Cv)
    Aw, Bw, Cw = np.atleast_2d(Aw), np.atleast_2d(Bw), np.atleast_2d(Cw)
    if abs(float(np.atleast_2d(Dw)[0, 0])) > 1e-12 or \
            abs(float(np.atleast_2d(Dv)[0, 0])) > 1e-12:
        raise ValueError('Q.P_min^-1 non strictement propre : boucle algebrique')
    nc, nw, nv = Ac.shape[0], Aw.shape[0], Av.shape[0]
    n = nc + nw + nv
    u_row = np.zeros(n)
    u_row[:nc] = Cc[0]
    u_row[nc:nc + nw] = -alpha * Cw[0]
    u_row[nc + nw:] = alpha * Cv[0]
    d_u = -float(Dc[0, 0])
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    A[:nc, :nc] = Ac
    B[:nc, 0] = -Bc[:, 0]
    A[nc:nc + nw, nc:nc + nw] = Aw
    B[nc:nc + nw, 0] = Bw[:, 0]
    A[nc + nw:, nc + nw:] = Av
    A[nc + nw:, :] += Bv @ u_row.reshape(1, -1)
    B[nc + nw:, 0] = Bv[:, 0] * d_u
    return A, B, u_row.reshape(1, -1), np.array([[d_u]])
