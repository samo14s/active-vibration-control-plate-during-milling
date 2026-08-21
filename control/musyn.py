"""
musyn.py — mu-synthese par iteration D-K, avec les ponderations DU PAPIER
=========================================================================
C'est la structure que Du, Liu, Dai & Long emploient eux-memes (`mu`-synthese
plus commande a retard). Jusqu'ici le depot portait un avertissement en toutes
lettres — « ceci n'est PAS une comparaison avec le correcteur du papier » —
parce qu'aucune des structures comparees n'etait la sienne. Ce module leve cet
avertissement.

CE QUI EST REPRIS DU PAPIER, TEL QUEL. L'incertitude additive des Eqs. (17)-(19)

    G_P(s) = G_Pr(s) + Delta_Pa(s) W_Pa(s),   ||Delta_Pa||_inf < 1
    W_Pa = [W_Paf  W_Pau],  W(s) = r (s^2+2z1w1 s+w1^2)/(s^2+2z2w2 s+w2^2)

avec r_Pau = 4e-7, z1 = 0.58, z2 = 0.22, w1 = 2pi 1100, w2 = 2pi 3500. Ces
nombres sont ceux du papier, deja transcrits dans `paper_model.milling_dynamics`
et deja audites par `verification/16`. On ne les rejoue pas : on les importe.

Seul `W_Pau` entre dans la BOUCLE. `W_Paf` pondere l'incertitude vue par la
force de coupe, qui est une entree exogene : elle ne referme rien, donc elle
n'intervient pas dans la stabilite robuste. La confondre avec `W_Pau` serait
compter deux fois la meme incertitude.

LA STRUCTURE DE mu, ET POURQUOI ELLE N'EST PAS UNE SIMPLE NORME H-INFINI. Deux
blocs :

    Delta = diag( delta_a ,  Delta_p )      delta_a  : 1x1, l'incertitude
                                            Delta_p  : 2x2, la performance

Avec UN SEUL bloc, mu se reduirait a la valeur singuliere maximale et la
mu-synthese ne serait qu'une synthese H-infini deguisee. Avec DEUX, la borne
superieure

    mu(M) <= min_{d>0}  sigma_max( diag(d,1,1) M diag(1/d,1,1) )

est strictement meilleure que sigma_max(M), et l'echelle `d` a un contenu
reel. C'est cette minimisation qui fait l'iteration D-K :

    K-etape : synthese H-infini sur le procede mis a l'echelle par D ;
    D-etape : a K fige, minimisation frequence par frequence de la borne,
              puis AJUSTEMENT RATIONNEL de d(w) pour la reinjecter.

L'AJUSTEMENT EST DU PREMIER ORDRE, ET C'EST UNE LIMITE ASSUMEE. `fit_d` cale
d(s) = k (s+a)/(s+b), stable et a minimum de phase. Un ajustement d'ordre
eleve serrerait davantage la borne, au prix d'un correcteur dont l'ordre
gonfle a chaque tour — et l'ordre est precisement l'une des grandeurs que la
comparaison rapporte. On prend donc le premier ordre, et on DIT que la borne
obtenue est superieure a celle d'une D-K a echelles riches.

CE QUI EST VERIFIE. `dk_iterate` refuse de rendre un resultat dont la borne mu
serait plus haute qu'au tour zero : une iteration D-K qui degrade est un bug,
pas un resultat. Et la borne elle-meme est controlee contre deux reperes
independants — mu <= sigma_max toujours, et mu >= rayon spectral reel — dans
`tests/test_invariants.py`.
"""
import os
import sys

import numpy as np


class MuBracket(RuntimeError):
    """La section doree s'est arretee sur un BORD du crochet : la valeur
    rendue serait le minimum du bord, pas de la droite, donc une borne mu
    fausse et silencieusement gonflee."""

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from hinf import (HinfFailure, lower_lft,                       # noqa: E402
                  synthesize)

# Les nombres du papier, importes et non recopies.
sys.path.insert(0, os.path.join(HERE, '..', 'paper_model'))
from milling_dynamics import W_PAU                              # noqa: E402


def weight_ss(p):
    """Etat-espace de W(s) = r (s^2+2z1w1 s+w1^2)/(s^2+2z2w2 s+w2^2).

    BIPROPRE : D = r. C'est ce D non nul qui oblige au decalage de boucle dans
    `hinf.central`.
    """
    a1, a0 = 2 * p['zeta1'] * p['w1'], p['w1'] ** 2
    b1, b0 = 2 * p['zeta2'] * p['w2'], p['w2'] ** 2
    r = p['r']
    A = np.array([[0.0, 1.0], [-b0, -b1]])
    B = np.array([[0.0], [1.0]])
    C = r * np.array([[a0 - b0, a1 - b1]])
    D = r * np.eye(1)
    return A, B, C, D


def augment_mu(plant, W1, w2, eps_n, Wu=None):
    """Procede generalise a DEUX blocs : incertitude + performance.

        w = [w_Delta ; d ; n]          z = [z_Delta ; W1 y ; w2 u]
        z_Delta = W_Pau u              y = P(u + d) + eps_n n + w_Delta

    `w_Delta` s'ajoute a la SORTIE et n'entre pas dans l'etat : sa colonne de
    B1 est nulle, ce qui preserve B1 D21' = 0 meme apres l'ajout du bloc.
    """
    Wu = weight_ss(W_PAU) if Wu is None else Wu
    Ap, Bp, Cp, _ = plant
    Aw, Bw, Cw, _ = W1
    Au, Bu, Cu, Du = Wu
    npp, nw, nu = Ap.shape[0], Aw.shape[0], Au.shape[0]
    n = npp + nw + nu
    A = np.zeros((n, n))
    A[:npp, :npp] = Ap
    A[npp:npp + nw, npp:npp + nw] = Aw
    A[npp:npp + nw, :npp] = Bw @ Cp                 # W1 voit y
    A[npp + nw:, npp + nw:] = Au                    # W_Pau voit u (via B2)
    B2 = np.zeros((n, 1))
    B2[:npp, 0:1] = Bp
    B2[npp + nw:, 0:1] = Bu
    # w = [w_Delta, d, n] : seule `d` entre dans l'etat, par la voie de u.
    B1 = np.zeros((n, 3))
    B1[:npp, 1:2] = Bp
    # z = [z_Delta, W1 y, w2 u]
    C1 = np.zeros((3, n))
    C1[0, npp + nw:] = Cu
    C1[1, npp:npp + nw] = Cw
    D12 = np.array([[float(Du[0, 0])], [0.0], [float(w2)]])
    C2 = np.zeros((1, n))
    C2[0, :npp] = Cp
    D21 = np.array([[1.0, 0.0, float(eps_n)]])
    return A, B1, B2, C1, C2, D12, D21


# ------------------------------------------------------------------ borne mu
def _frf(ss, om):
    A, B, C, D = [np.atleast_2d(np.asarray(m, float)) for m in ss]
    n = A.shape[0]
    return np.array([C @ np.linalg.solve(1j * w * np.eye(n) - A, B) + D
                     for w in om])


def mu_upper(M, n_a=1, n_gold=40):
    """Borne superieure de mu pour Delta = diag(delta_a I_na, Delta_p).

    Minimisation sur la SEULE echelle libre d, par section doree en log d.
    La fonction d -> sigma_max(D M D^-1) est unimodale en log d (c'est un
    maximum de fonctions convexes en log d), donc la section doree converge
    vers le minimum global.
    """
    M = np.atleast_2d(M)
    m = M.shape[0]
    sel = np.ones(m)
    sel[:n_a] = 0.0                       # 0 -> bloc mis a l'echelle par d

    def sig(ld):
        d = np.exp(ld)
        s = np.where(sel > 0, 1.0, d)
        return float(np.linalg.svd(np.diag(s) @ M @ np.diag(1.0 / s),
                                   compute_uv=False)[0])

    # CROCHET. Il valait [-12, +12] et il SATURAIT : sur les tours D-K de
    # l'optimum musyn stocke, 400 frequences sur 400 avaient leur argmin colle
    # a +12.00 alors que l'optimum reel vivait dans [13.4, 17.9] — hors du
    # crochet. La borne rendue etait alors le minimum SUR LE BORD et non sur
    # la droite : mu = 346.249 au lieu de 83.695, soit +314 %, et jusqu'a
    # +367 % point par point. Rien ne le signalait : la fonction rendait
    # simplement min(fc, fd), et les deux invariants de tests/ (mu <= sigma_max
    # et mu >= rayon spectral) sont satisfaits par une borne GONFLEE, donc
    # aveugles a ce mode de panne.
    #
    # Le crochet est donc elargi a [-40, 40] — la mise a l'echelle d est un
    # rapport de grandeurs physiques qui peut couvrir des decades — et la
    # SATURATION EST DETECTEE : si l'optimum retenu touche un bord, on le
    # signale au lieu de rendre une borne fausse en silence.
    lo, hi = -40.0, 40.0
    lo0, hi0 = lo, hi
    phi = 0.5 * (np.sqrt(5.0) - 1.0)
    c, dd = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = sig(c), sig(dd)
    for _ in range(n_gold):
        if fc < fd:
            hi, dd, fd = dd, c, fc
            c = hi - phi * (hi - lo)
            fc = sig(c)
        else:
            lo, c, fc = c, dd, fd
            dd = lo + phi * (hi - lo)
            fd = sig(dd)
    ld = 0.5 * (c + dd)
    if ld <= lo0 + 1e-6 or ld >= hi0 - 1e-6:
        # Au bord : la valeur rendue n'est pas un minimum interieur, donc pas
        # une borne superieure de mu utilisable pour comparer des tours.
        raise MuBracket(f'section doree saturee au bord log d = {ld:.3f} '
                        f'du crochet [{lo0:g}, {hi0:g}]')
    return min(fc, fd), float(np.exp(ld))


def mu_curve(P, K, om, n_a=1):
    """(mu(w), d_opt(w)) de la boucle fermee F_l(P, K)."""
    M = _frf(lower_lft(P, K), om)
    mus = np.empty(len(om))
    ds = np.empty(len(om))
    for i in range(len(om)):
        mus[i], ds[i] = mu_upper(M[i], n_a)
    return mus, ds


# --------------------------------------------------------- ajustement de d(w)
def fit_d(om, d, n_pt=200):
    """d(s) = k (s+a)/(s+b), stable et a minimum de phase, calee au sens des
    moindres carres sur log|d(w)|.

    On ne cale QUE le module : la phase d'une echelle D n'a aucun sens ici
    (D est reelle positive frequence par frequence), et lui imposer une phase
    ajouterait une contrainte que le probleme ne porte pas.
    """
    from scipy.optimize import least_squares
    om = np.asarray(om, float)
    d = np.asarray(d, float)
    good = np.isfinite(d) & (d > 0)
    if good.sum() < 8:
        return None
    i = np.linspace(0, good.sum() - 1, min(n_pt, good.sum())).astype(int)
    w = om[good][i]
    y = np.log(d[good][i])

    def resid(p):
        lk, la, lb = p
        mod = (lk + 0.5 * np.log(w ** 2 + np.exp(2 * la))
               - 0.5 * np.log(w ** 2 + np.exp(2 * lb)))
        return mod - y

    p0 = np.array([np.mean(y), np.log(np.median(w)), np.log(np.median(w))])
    try:
        r = least_squares(resid, p0, max_nfev=400)
    except Exception:                                     # noqa: BLE001
        return None
    k, a, b = np.exp(r.x[0]), np.exp(r.x[1]), np.exp(r.x[2])
    if not np.all(np.isfinite([k, a, b])) or k <= 0:
        return None
    # d(s) = k (s+a)/(s+b) : A = -b, B = 1, C = k(a-b), D = k
    return (np.array([[-b]]), np.array([[1.0]]),
            np.array([[k * (a - b)]]), np.array([[k]]))


def _scale_channel(P, Dss):
    """Reecrit le procede generalise avec le canal d'incertitude mis a
    l'echelle par D(s) : z_Delta <- D z_Delta et w_Delta <- D^-1 w_Delta.

    En pratique on insere D en SERIE sur la ligne z_Delta et D^-1 sur la
    colonne w_Delta. Comme D est du premier ordre, cela ajoute deux etats.
    """
    A, B1, B2, C1, C2, D12, D21 = [np.atleast_2d(np.asarray(m, float))
                                   for m in P]
    Ad, Bd, Cd, Dd = [np.atleast_2d(np.asarray(m, float)) for m in Dss]
    n, nd = A.shape[0], Ad.shape[0]
    dd = float(Dd[0, 0])
    if abs(dd) < 1e-300:
        return None
    # --- D en serie sur z_Delta (ligne 0 de C1/D12)
    n1 = n + nd
    A1 = np.zeros((n1, n1))
    A1[:n, :n] = A
    A1[n:, n:] = Ad
    A1[n:, :n] = Bd @ C1[0:1, :]
    B11 = np.vstack([B1, Bd @ np.zeros((1, B1.shape[1]))])
    B21 = np.vstack([B2, Bd @ D12[0:1, :]])
    C11 = np.zeros((C1.shape[0], n1))
    C11[:, :n] = C1
    C11[0, :] = 0.0
    C11[0, :n] = dd * C1[0, :]
    C11[0, n:] = Cd
    D121 = D12.copy()
    D121[0, :] = dd * D12[0, :]
    C21 = np.hstack([C2, np.zeros((1, nd))])
    # --- D^-1 en serie sur w_Delta (colonne 0 de B1/D21).
    # D^-1(s) = (1/dd)(s+b)/(s+a') est propre : realisation
    # A = -(Cd/dd + Ad ... ) ; on l'ecrit directement a partir de (Ad,Bd,Cd,Dd).
    Ai = Ad - Bd @ Cd / dd
    Bi = Bd / dd
    Ci = -Cd / dd
    Di = np.array([[1.0 / dd]])
    n2 = n1 + nd
    A2 = np.zeros((n2, n2))
    A2[:n1, :n1] = A1
    A2[n1:, n1:] = Ai
    A2[:n1, n1:] = B11[:, 0:1] @ Ci
    B12 = np.zeros((n2, B11.shape[1]))
    B12[:n1, :] = B11
    B12[:n1, 0:1] = B11[:, 0:1] * float(Di[0, 0])
    B12[n1:, 0:1] = Bi
    B22 = np.vstack([B21, np.zeros((nd, 1))])
    C12 = np.zeros((C11.shape[0], n2))
    C12[:, :n1] = C11
    # z ne depend pas directement de w (D11 = 0), donc rien a ajouter sur les
    # colonnes de l'etat de D^-1 pour les lignes penalisees.
    C22 = np.zeros((1, n2))
    C22[:, :n1] = C21
    # ... MAIS y, LUI, EN DEPEND. w_Delta n'entre pas dans l'etat du procede :
    # il s'AJOUTE A LA SORTIE (D21[0,0] = 1). Toute la voie w_Delta -> y passe
    # donc par le terme direct, et la sortie de D^-1 doit y etre reinjectee.
    # Sans cette ligne l'etat de D^-1 n'est relie a rien : la mise a l'echelle
    # est exacte pour un D CONSTANT — ou D^-1 n'a pas d'etat — et fausse de
    # 85 % des que D devient dynamique, c'est-a-dire dans tous les cas qui
    # servent a quelque chose.
    C22[:, n1:] = float(D21[0, 0]) * Ci
    D212 = D21.copy()
    D212[0, 0] = D21[0, 0] * float(Di[0, 0])
    return A2, B12, B22, C12, C22, D121, D212


def dk_iterate(P, om=None, n_dk=3, verbose=False):
    """Iteration D-K. Rend (K, mu, historique).

    Refuse de rendre un tour qui DEGRADE la borne : une iteration D-K qui
    remonte mu n'est pas un compromis, c'est un defaut.
    """
    if om is None:
        om = 2 * np.pi * np.logspace(0, 4.3, 400)
    K, g = synthesize(P, shifted=True)
    mus, ds = mu_curve(P, K, om)
    best = (K, float(np.max(mus)), P)
    hist = [dict(step=0, gamma=g, mu=best[1])]
    if verbose:
        print(f'    D-K 0 : gamma = {g:.4g}   mu = {best[1]:.4g}')
    Pc = P
    for it in range(1, n_dk + 1):
        Dss = fit_d(om, ds)
        if Dss is None:
            break
        Ps = _scale_channel(Pc, Dss)
        if Ps is None:
            break
        try:
            Kn, gn = synthesize(Ps, shifted=True)
        except HinfFailure:
            break
        # mu se mesure TOUJOURS sur le probleme d'origine : la mise a
        # l'echelle est un moyen, pas le critere.
        mus, ds = mu_curve(P, Kn, om)
        mu_n = float(np.max(mus))
        hist.append(dict(step=it, gamma=gn, mu=mu_n))
        if verbose:
            print(f'    D-K {it} : gamma = {gn:.4g}   mu = {mu_n:.4g}')
        if mu_n < best[1]:
            best = (Kn, mu_n, Ps)
        Pc = Ps
    return best[0], best[1], hist
