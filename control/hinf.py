"""
hinf.py — synthese H-infini par les DEUX equations de Riccati, sur scipy seul
=============================================================================
Le depot n'a ni `python-control` ni `slycot` ni `cvxpy` : rien qui sache faire
`hinfsyn`. Ce module l'ecrit, parce que la comparaison equitable ne vaut que si
la structure la plus classique du domaine y figure — et parce que la
mu-synthese, qui est le correcteur du PAPIER lui-meme, se construit par-dessus
(voir `musyn.py`).

LA SOLUTION CENTRALE (Doyle, Glover, Khargonekar, Francis 1989). Pour le
probleme standard

    [z]   [ A  | B1  B2 ] [w]
    [y] = [ C1 | 0   D12] [u]  ,   u = K y  ,   min ||F_l(P, K)||_inf
          [ C2 | D21 0  ]

sous les hypotheses SIMPLIFIEES D11 = 0, D12'C1 = 0, B1 D21' = 0,
D12'D12 = I, D21 D21' = I, il existe un correcteur atteignant gamma ssi

    X = Ric[ A            , g^-2 B1 B1' - B2 B2' ]  >= 0  existe
           [ -C1'C1       , -A'                  ]
    Y = Ric[ A'           , g^-2 C1'C1 - C2'C2   ]  >= 0  existe
           [ -B1 B1'      , -A                   ]
    rho(X Y) < gamma^2

et le correcteur central s'ecrit alors, avec F = -B2'X, L = -Y C2',
Z = (I - g^-2 Y X)^-1 :

    K : xdot = (A + g^-2 B1 B1' X + B2 F + Z L C2) x - Z L y
        u    = F x

POURQUOI ELLES TIENNENT ICI SANS DECALAGE DE BOUCLE — TANT QU'ON RESTE EN
SENSIBILITE MIXTE. Elles ne sont pas supposees : elles sont IMPOSEES par la
forme des ponderations, et ce choix de forme est celui que la physique
demandait de toute facon.

  * `D11 = 0` — le seul chemin direct w -> z passerait par le bruit de mesure
    vu par W1. On prend donc W1 STRICTEMENT PROPRE. Et la forme strictement
    propre qu'on veut ici est un PASSE-BANDE : le probleme est band-limite,
    les deux modes de broutement sont a 543 et 1068 Hz, et ponderer l'erreur
    ailleurs ne fait qu'acheter de la performance la ou il n'y a rien a
    gagner. La contrainte mathematique et le bon sens de commande tombent au
    meme endroit.
  * `D12'C1 = 0` — C1 empile [W1 y ; W2 u]. Si W2 est un SCALAIRE (pas de
    dynamique), la ligne de C1 qui lui correspond est nulle, et D12 ne
    selectionne que cette ligne-la. Le produit est nul par construction. On
    peut se le permettre parce que l'effort est deja borne DEUX FOIS
    ailleurs : par la contrainte dure du protocole (<= 450 V/N) et par le
    filtre de coupure que toutes les structures portent en serie.
  * `B1 D21' = 0` — le canal de bruit de mesure n'entre pas dans l'etat, il
    s'ajoute a la sortie. Sa colonne dans B1 est nulle.

Rien de tout cela n'est verifie sur parole : `assumptions()` les mesure et
`synthesize()` refuse de rendre un correcteur si l'une tombe.

MAIS LA MU-SYNTHESE LES CASSE, ET C'EST VOULU. Le canal d'incertitude du
papier, z_Delta = W_Pau u, fait intervenir une ponderation BIPROPRE
(D = r_Pau = 4e-7), donc D12'C1 cesse d'etre nul. Interdire les ponderations
biproprres reviendrait a interdire celles du papier — exactement celles qu'on
veut comparer. `central()` porte donc le DECALAGE DE BOUCLE general
(Ax = A - B2 D12'C1, Qx = C1'(I - D12 D12')C1, et F = -(D12'C1 + B2'X)), qui
se reduit a l'identite quand le produit est deja nul. Le cas simple reste donc
verifie a l'octet pres par les memes invariants qu'avant.

CE QUI EST VERIFIE, ET COMMENT. Un solveur H-infini faux est indetectable a
l'oeil : il rend une matrice d'etat plausible qui ne fait pas ce qu'elle
promet. Deux controles INDEPENDANTS du solveur, dans tests/ :

  1. la boucle fermee est stable — poles a partie reelle < 0 ;
  2. la norme H-infini ANNONCEE par l'iteration sur gamma est retrouvee par un
     balayage frequentiel direct de ||F_l(P,K)(jw)||, qui ne partage aucune
     ligne de code avec les Riccati.

Le second point est l'essentiel : c'est lui qui distingue "les equations ont
converge" de "le correcteur atteint vraiment gamma".
"""
import numpy as np
from scipy.linalg import schur


class HinfFailure(RuntimeError):
    """Aucun correcteur ne peut etre rendu pour ce gamma (ou ce probleme)."""


def care(A, S, Q):
    """X >= 0 solution de  A'X + XA + Q - X S X = 0, par le HAMILTONIEN.

    POURQUOI PAS `scipy.linalg.solve_continuous_are`. Cette routine attend le
    probleme sous la forme (A, B, Q, R) avec S = B R^-1 B' et R inversible.
    Or en H-infini S vaut  B2 B2' - B1 B1'/gamma^2  : elle est INDEFINIE — et
    c'est exactement ce qui fait le probleme H-infini plutot qu'un LQR. Elle
    est aussi singuliere des que n depasse le nombre d'entrees, ce qui est
    toujours le cas ici. Aucune ecriture (A, B, Q, R) ne la represente.

    On passe donc par le hamiltonien, ou l'indefinition ne pose aucun
    probleme :

        H = [  A   -S  ]        X = U2 U1^-1  sur le sous-espace invariant
            [ -Q   -A' ]                       STABLE  H [U1;U2] = [U1;U2] L

    L'existence d'une solution stabilisante equivaut a ce que H n'ait aucune
    valeur propre sur l'axe imaginaire et que U1 soit inversible — les deux
    sont testes, et leur echec EST la reponse "gamma trop petit".
    """
    A = np.asarray(A, float)
    n = A.shape[0]
    H = np.block([[A, -np.asarray(S, float)],
                  [-np.asarray(Q, float), -A.T]])
    if not np.all(np.isfinite(H)):
        raise HinfFailure('hamiltonien non fini')
    ev = np.linalg.eigvals(H)
    # Un pole sur l'axe imaginaire, c'est le gamma optimal lui-meme : la
    # frontiere de faisabilite. On la refuse plutot que de rendre une solution
    # au bord, numeriquement sans valeur.
    scale = max(1.0, float(np.max(np.abs(ev))))
    if np.min(np.abs(ev.real)) <= 1e-9 * scale:
        raise HinfFailure('valeur propre hamiltonienne sur l axe imaginaire')
    T, U, sdim = schur(H, output='real', sort='lhp')
    if sdim != n:
        # `sort='lhp'` compte les valeurs propres de partie reelle < 0. Le
        # spectre d'un hamiltonien est symetrique par rapport a l'axe, donc il
        # y en a exactement n — mais quand deux d'entre elles sont proches de
        # l'axe, l'arrondi peut en faire passer une du mauvais cote et le
        # compte tombe a n+-1 ou pire. On ne renonce pas pour autant : on
        # RANGE explicitement par partie reelle et on coupe entre la n-ieme et
        # la (n+1)-ieme, ce qui est la definition voulue et ne depend plus du
        # signe de quantites minuscules.
        order = np.sort(ev.real)
        if order[n] - order[n - 1] <= 0.0:
            raise HinfFailure('spectre hamiltonien non separable')
        thr = 0.5 * (order[n - 1] + order[n])
        T, U, sdim = schur(H, output='real',
                           sort=lambda re, im: re < thr)
        if sdim != n:
            raise HinfFailure(
                f'sous-espace stable de dimension {sdim}, attendu {n}')
    U1, U2 = U[:n, :n], U[n:, :n]
    if np.linalg.matrix_rank(U1, tol=1e-12 * max(1.0, np.max(np.abs(U1)))) < n:
        raise HinfFailure('U1 singuliere : pas de solution stabilisante')
    X = np.linalg.solve(U1.T, U2.T).T
    return 0.5 * (X + X.T)


# ----------------------------------------------------------------- ponderations
def bandpass_weight(k, f0, zeta):
    """W(s) = k . 2 zeta w0 s / (s^2 + 2 zeta w0 s + w0^2) — STRICTEMENT propre.

    Passe-bande centre sur f0. Gain k a la resonance ; `zeta` fixe la largeur
    relative. C'est la ponderation de PERFORMANCE : elle dit ou l'erreur coute
    cher, et ici elle coute cher dans la bande de broutement.
    """
    w0 = 2.0 * np.pi * float(f0)
    a = 2.0 * float(zeta) * w0
    A = np.array([[0.0, 1.0], [-w0 * w0, -a]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[0.0, float(k) * a]])
    D = np.array([[0.0]])
    return A, B, C, D


def _blkdiag(*mats):
    mats = [np.atleast_2d(m) for m in mats]
    r = sum(m.shape[0] for m in mats)
    c = sum(m.shape[1] for m in mats)
    out = np.zeros((r, c))
    i = j = 0
    for m in mats:
        out[i:i + m.shape[0], j:j + m.shape[1]] = m
        i += m.shape[0]
        j += m.shape[1]
    return out


# ------------------------------------------------------------- procede augmente
def plant_ss(w, zeta, res):
    """Etat-espace modal du procede : u [V] -> y [m], strictement propre.

    Chaque mode donne le bloc compagnon usuel ; les residus `res` portent le
    signe du couplage, donc le SIGNE DE BOUCLE est deja dedans. Aucun
    `sign_loop` n'est applique par-dessus : la synthese le decouvre elle-meme,
    ce que ne peut pas faire une structure a gains fixes.
    """
    w = np.atleast_1d(np.asarray(w, float))
    zeta = np.atleast_1d(np.asarray(zeta, float))
    res = np.atleast_1d(np.asarray(res, float))
    n = len(w)
    A = np.zeros((2 * n, 2 * n))
    B = np.zeros((2 * n, 1))
    C = np.zeros((1, 2 * n))
    for k in range(n):
        i = 2 * k
        A[i, i + 1] = 1.0
        A[i + 1, i] = -w[k] ** 2
        A[i + 1, i + 1] = -2.0 * zeta[k] * w[k]
        B[i + 1, 0] = 1.0
        C[0, i] = res[k]
    return A, B, C, np.zeros((1, 1))


def augment(plant, W1, w2, eps_n, extra_z=None):
    """Procede generalise du probleme de sensibilite mixte.

    Entrees exogenes  w = [d (force de coupe, via la meme voie que u) ; n]
    Commande          u
    Sorties penalisees z = [W1 y ; w2 u]  (+ canaux `extra_z` pour mu)
    Mesure            y = P(u + d) + eps_n n

    `d` est injectee PAR LA MEME VOIE que la commande. Ce n'est pas une
    commodite : la force de coupe agit sur la plaque, pas sur le capteur, et
    la rejeter la ou elle entre est exactement le probleme physique.
    """
    Ap, Bp, Cp, _ = plant
    Aw, Bw, Cw, Dw = W1
    np_, nw = Ap.shape[0], Aw.shape[0]
    n = np_ + nw
    A = np.zeros((n, n))
    A[:np_, :np_] = Ap
    A[np_:, np_:] = Aw
    A[np_:, :np_] = Bw @ Cp              # W1 voit y
    # w = [d, n] : d entre comme u ; n ne touche pas l'etat -> colonne nulle,
    # ce qui donne B1 D21' = 0 sans rien faire de plus.
    B1 = np.zeros((n, 2))
    B1[:np_, 0:1] = Bp
    B2 = np.zeros((n, 1))
    B2[:np_, 0:1] = Bp
    # z1 = W1 y   (Dw = 0 : strictement propre)   z2 = w2 u
    rows = [np.hstack([np.zeros((Cw.shape[0], np_)), Cw])]
    D12 = [np.zeros((Cw.shape[0], 1))]
    rows.append(np.zeros((1, n)))
    D12.append(np.array([[float(w2)]]))
    if extra_z is not None:
        Cz, Dz = extra_z
        rows.append(Cz)
        D12.append(Dz)
    C1 = np.vstack(rows)
    D12 = np.vstack(D12)
    C2 = np.zeros((1, n))
    C2[0, :np_] = Cp
    D21 = np.array([[0.0, float(eps_n)]])
    return A, B1, B2, C1, C2, D12, D21


def assumptions(A, B1, B2, C1, C2, D12, D21, tol=1e-9, shifted=False):
    """Mesure les hypotheses au lieu de les supposer. Rend (ok, rapport)."""
    rep = {}
    rep['D12_rank'] = int(np.linalg.matrix_rank(D12, tol=1e-10))
    rep['D21_rank'] = int(np.linalg.matrix_rank(D21, tol=1e-10))
    rep['D12tC1'] = float(np.max(np.abs(D12.T @ C1)))
    rep['B1D21t'] = float(np.max(np.abs(B1 @ D21.T)))
    ok = (rep['D12_rank'] == D12.shape[1]
          and rep['D21_rank'] == D21.shape[0])
    if not shifted:
        # Sans decalage de boucle, les deux produits croises doivent etre nuls.
        # Avec (`shifted=True`, cas des ponderations biproprres du papier),
        # `central` les absorbe et seuls les rangs comptent.
        ok = (ok
              and rep['D12tC1'] <= tol * max(1.0, float(np.max(np.abs(C1))))
              and rep['B1D21t'] <= tol * max(1.0, float(np.max(np.abs(B1)))))
    return ok, rep


# ------------------------------------------------------------------- synthese
def _normalize(B2, C2, D12, D21):
    """Rend D12'D12 = 1 et D21 D21' = 1 en remettant a l'echelle u ET y.

    ATTENTION, C'EST u ET y QU'IL FAUT ECHELONNER, PAS z ET w. Une premiere
    version divisait B1 et C1 — donc les canaux exogene et penalise — ce qui
    change le PROBLEME au lieu de le renormaliser, et rend un correcteur
    optimal pour une autre ponderation que celle demandee.

    Avec u~ = s12 u :  B2~ = B2/s12,  D12~ = D12/s12
    Avec y~ = y /s21 :  C2~ = C2/s21,  D21~ = D21/s21
    et le correcteur revient par  K = K~ / (s12 s21).

    D12 est une COLONNE et D21 une LIGNE (un actionneur, un capteur), donc la
    normalisation est une division par une norme : les rotations orthogonales
    du cas general ne sont pas necessaires.
    """
    s12 = float(np.linalg.norm(D12))
    s21 = float(np.linalg.norm(D21))
    if s12 <= 0.0 or s21 <= 0.0:
        raise HinfFailure('D12 ou D21 de rang deficient')
    return B2 / s12, C2 / s21, D12 / s12, D21 / s21, s12, s21


# L'equilibrage du SYSTEME vit desormais dans ss_balance.py : le meme besoin
# est apparu sur TOUS les correcteurs du depot, pas seulement sur le
# hamiltonien H-infini. Le corps est inchange.
from ss_balance import balance as _balance                       # noqa: E402


def scale_problem(P):
    """Ramene le probleme a des grandeurs O(1) sans le changer.

    Deux mises a l'echelle, toutes deux inoffensives pour le correcteur :

      * z <- z/alpha et w <- w/beta : uniformes, donc elles ne touchent NI aux
        poids relatifs (qui sont le reglage) NI a la voie u -> y. Seul gamma
        est affecte, et il revient par gamma = alpha.beta.gamma'.
      * similitude diagonale sur l'etat : la fonction de transfert du procede
        est invariante, donc le correcteur obtenu dans la base equilibree est
        DEJA le bon pour le procede d'origine — rien a defaire.
    """
    A, B1, B2, C1, C2, D12, D21 = [np.atleast_2d(np.asarray(m, float))
                                   for m in P]
    alpha = float(np.linalg.norm(np.hstack([C1, D12])))
    beta = float(np.linalg.norm(np.hstack([B1.T, D21.T])))
    if alpha <= 0.0 or beta <= 0.0:
        raise HinfFailure('canal penalise ou exogene identiquement nul')
    C1, D12 = C1 / alpha, D12 / alpha
    B1, D21 = B1 / beta, D21 / beta
    A, (B1, B2), (C1, C2), _ = _balance(A, [B1, B2], [C1, C2])
    return (A, B1, B2, C1, C2, D12, D21), alpha, beta


def central(A, B1, B2, C1, C2, D12, D21, gamma):
    """Correcteur central pour ce gamma, ou HinfFailure si gamma est trop bas.

    Le probleme est suppose DEJA mis a l'echelle (`scale_problem`).
    """
    B2n, C2n, D12n, D21n, s12, s21 = _normalize(B2, C2, D12, D21)
    g2 = float(gamma) ** 2
    n = A.shape[0]
    # --- DECALAGE DE BOUCLE (loop-shift).
    # Tant que la ponderation de commande est un scalaire, D12'C1 = 0 et ces
    # deux lignes ne font rien. Mais des qu'on ajoute le canal d'incertitude
    # du PAPIER, z_Delta = W_Pau u, le terme cesse d'etre nul : W_Pau est
    # BIPROPRE (D = r = 4e-7), donc D12 gagne une composante en regard d'une
    # ligne de C1 qui, elle, n'est pas nulle. Plutot que d'interdire les
    # ponderations biproprres — c'est-a-dire d'interdire celles du papier — on
    # ecrit le decalage general.
    DC = D12n.T @ C1                                    # (1, n)
    Ax = A - B2n @ DC
    Qx = C1.T @ C1 - DC.T @ DC                          # C1'(I - D12 D12')C1
    BD = B1 @ D21n.T                                    # (n, 1)
    Ay = A - BD @ C2n
    Qy = B1 @ B1.T - BD @ BD.T                          # B1(I - D21'D21)B1'
    # S indefinie dans les deux cas : c'est la signature du probleme H-infini.
    Sx = B2n @ B2n.T - (B1 @ B1.T) / g2
    Sy = C2n.T @ C2n - (C1.T @ C1) / g2
    try:
        X = care(Ax, Sx, Qx)
        Y = care(Ay.T, Sy, Qy)
    except (np.linalg.LinAlgError, ValueError) as e:
        raise HinfFailure(f'Riccati insoluble a gamma = {gamma:.4g} : {e}')
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise HinfFailure('Riccati non finie')
    for M, nm in ((X, 'X'), (Y, 'Y')):
        ev = np.linalg.eigvalsh(0.5 * (M + M.T))
        if ev.min() < -1e-6 * max(1.0, abs(ev).max()):
            raise HinfFailure(f'{nm} non semi-definie positive')
    rho = float(np.max(np.abs(np.linalg.eigvals(X @ Y))))
    if rho >= g2:
        raise HinfFailure(f'rho(XY) = {rho:.4g} >= gamma^2 = {g2:.4g}')
    F = -(DC + B2n.T @ X)
    L = -(BD + Y @ C2n.T)
    Z = np.linalg.inv(np.eye(n) - (Y @ X) / g2)
    Ak = A + (B1 @ B1.T @ X) / g2 + B2n @ F + Z @ L @ C2n
    Bk = -Z @ L
    Ck = F
    Dk = np.zeros((1, 1))
    # u~ = s12 u et y~ = y/s21  ->  K = K~ / (s12 s21)
    return Ak, Bk, Ck / (s12 * s21), Dk


def lower_lft(P, K):
    """F_l(P, K) en etat-espace, avec D22 = 0 et Dk = 0 (cas de ce module)."""
    A, B1, B2, C1, C2, D12, D21 = P
    Ak, Bk, Ck, _ = K
    n, nk = A.shape[0], Ak.shape[0]
    Acl = np.zeros((n + nk, n + nk))
    Acl[:n, :n] = A
    Acl[:n, n:] = B2 @ Ck
    Acl[n:, :n] = Bk @ C2
    Acl[n:, n:] = Ak
    Bcl = np.vstack([B1, Bk @ D21])
    Ccl = np.hstack([C1, D12 @ Ck])
    Dcl = np.zeros((C1.shape[0], B1.shape[1]))
    return Acl, Bcl, Ccl, Dcl


def hinf_norm(ss, w=None):
    """||G||_inf par BALAYAGE FREQUENTIEL — deliberement independant des
    Riccati, pour que le controle du solveur ne repose pas sur le solveur."""
    A, B, C, D = [np.atleast_2d(np.asarray(m, float)) for m in ss]
    if w is None:
        w = 2 * np.pi * np.logspace(-2, 5.2, 3000)
    ev = np.linalg.eigvals(A)
    if np.max(ev.real) >= 0.0:
        return np.inf
    n = A.shape[0]
    out = 0.0
    for om in w:
        G = C @ np.linalg.solve(1j * om * np.eye(n) - A, B) + D
        s = np.linalg.svd(G, compute_uv=False)
        out = max(out, float(s[0]))
    return out


def _accept(P, Ps, gamma):
    """Le correcteur central a ce gamma, s'il STABILISE reellement la boucle.

    Un gamma peut satisfaire les trois conditions de Riccati numeriquement et
    rendre malgre tout un correcteur qui ne stabilise pas : les gardes portent
    sur X, Y et rho(XY), pas sur le resultat. On teste donc directement ce qui
    compte, par les poles de F_l(P, K).

    Le test est fait A CHAQUE PALIER, et c'est le point : le laisser pour la
    fin faisait echouer toute la synthese des que la derniere descente allait
    trop loin, alors qu'un palier superieur parfaitement valide existait. Une
    valeur propre coute bien moins qu'une decomposition de Schur, donc verifier
    a chaque palier ne change pas l'ordre de grandeur du cout.
    """
    K = central(*Ps, gamma)
    ev = np.linalg.eigvals(lower_lft(P, K)[0])
    if not np.all(np.isfinite(ev)) or ev.real.max() >= 0.0:
        raise HinfFailure('correcteur non stabilisant a ce gamma')
    return K


def synthesize(P, g_hi=None, g_lo=None, tol=1e-3, n_iter=40, check=True,
               shifted=False, ratio=0.6, n_gap=8):
    """Iteration sur gamma (bissection) -> (K, gamma) au mieux atteignable.

    `check` mesure la norme obtenue par balayage et la compare a gamma : c'est
    la garde qui distingue "les Riccati ont converge" de "le correcteur fait
    ce qu'il promet".
    """
    ok, rep = assumptions(*P, shifted=shifted)
    if not ok:
        raise HinfFailure(f'hypotheses simplifiees violees : {rep}')
    Ps, alpha, beta = scale_problem(P)
    # ---------------------------------------------------------------- gamma
    # PAS DE BISSECTION. La theorie garantit que la faisabilite est MONOTONE
    # en gamma — si un gamma passe, tous les plus grands passent — et une
    # bissection ne vaut que sous cette hypothese. Mesure sur ce modele : elle
    # est FAUSSE numeriquement. En balayant gamma sur douze decades on obtient
    #
    #     ....FFF...FFFFFFFFFFFFFFFFFFFFFF        (F = faisable)
    #
    # c'est-a-dire des TROUS d'infaisabilite au milieu de la region faisable,
    # larges d'un facteur cinq, dus aux echecs numeriques des deux Riccati et
    # non a la theorie. Une bissection qui sonde dans un trou conclut
    # "infaisable", releve sa borne basse, et perd DEFINITIVEMENT toute la
    # region de gamma plus petits. Le correcteur rendu est alors bien plus
    # faible que celui qui existe, sans le moindre signe exterieur.
    #
    # On descend donc par paliers geometriques en TOLERANT les trous : il faut
    # `n_gap` echecs CONSECUTIFS pour declarer le fond atteint.
    lo_floor = 1e-9 if g_lo is None else float(g_lo) / (alpha * beta)
    hi = 1e6 if g_hi is None else float(g_hi) / (alpha * beta)
    # UN SEUL BALAYAGE DESCENDANT, sans exiger que le sommet marche. Une
    # version precedente cherchait d'abord un gamma faisable en MONTANT depuis
    # 1e6, et n'entamait la descente qu'apres. Elle echouait completement des
    # que le tout premier gamma ne stabilisait pas — ce qui arrive : a tres
    # grand gamma, Z = (I - YX/gamma^2)^-1 tend vers l'identite et les deux
    # Riccati deviennent numeriquement degenerees. Mesure : kw >= 6000 rendait
    # "infaisable jusqu a 1e16" alors qu'un correcteur parfaitement valide
    # existait a gamma = 0.022. On descend donc directement, et le premier
    # palier qui STABILISE ouvre la recherche.
    best = None
    g, misses = hi / ratio, 0
    while g > lo_floor and (best is None or misses < n_gap):
        g *= ratio
        try:
            K = _accept(P, Ps, g)
            best, misses = (K, g), 0
        except HinfFailure:
            misses += 1
            if best is None and g <= lo_floor * 10:
                break
    if best is None:
        raise HinfFailure(f"aucun gamma stabilisant entre {lo_floor:.2g} et"
                          f" {hi:.2g} (echelle interne)")
    # SECONDE PASSE, PLUS FINE. Le palier grossier peut ENJAMBER une fenetre
    # faisable plus etroite que son pas : mesure a kw = 5166, ou la fenetre de
    # bas gamma ne fait que 0.75 decade alors que le pas en fait 0.51. On
    # repasse donc au pas fin sur les deux decades sous le meilleur palier, ce
    # qui suffit a la resoudre pour une trentaine d'appels de plus.
    g, misses = best[1], 0
    floor2 = max(lo_floor, best[1] * 1e-2)
    while g > floor2 and misses < 3 * n_gap:
        g *= 0.9
        try:
            K = _accept(P, Ps, g)
            best, misses = (K, g), 0
        except HinfFailure:
            misses += 1
    # Raffinement local entre le meilleur palier et le palier juste en dessous.
    lo, hi = best[1] * 0.9, best[1]
    for _ in range(n_iter):
        if hi - lo <= tol * max(1e-12, hi):
            break
        mid = np.sqrt(lo * hi)
        try:
            K = _accept(P, Ps, mid)
            best = (K, mid)
            hi = mid
        except HinfFailure:
            lo = mid
    K, g = best
    g = g * alpha * beta
    if check:
        # Mesure sur le probleme D'ORIGINE, pas sur la version mise a
        # l'echelle : ce controle doit aussi attraper une erreur DANS la mise
        # a l'echelle, pas seulement dans les Riccati.
        got = hinf_norm(lower_lft(P, K))
        # 5 % de marge : le balayage est discret et gamma est atteint par le
        # haut. Un ecart plus grand signale une erreur de synthese, pas une
        # imprecision de grille.
        if not np.isfinite(got) or got > 1.05 * g:
            raise HinfFailure(
                f'norme mesuree {got:.4g} contre gamma annonce {g:.4g}')
    return K, g
