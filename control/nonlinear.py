"""
nonlinear.py — SMC et MPC : la partie non lineaire, et la partie qui ne l'est
=============================================================================
pas.

CE FICHIER PORTAIT UNE AFFIRMATION TROP FORTE, ET ELLE EST CORRIGEE ICI. Il
disait que ces deux structures ne peuvent pas entrer dans le critere de
Floquet parce qu'elles n'ont pas de fonction de transfert. C'est faux pour les
deux, telles qu'elles sont ECRITES ci-dessous, et le reconnaitre leur rend une
colonne pleine dans la comparaison au lieu d'une colonne a trous.

MPC. La loi implementee est un MPC EXPLICITE SANS CONTRAINTE ACTIVE :
u = -K x_hat, K calcule une fois hors ligne, etat estime par un filtre de
Kalman. C'est, litteralement, un correcteur lineaire invariant — le texte
precedent le disait lui-meme deux paragraphes plus bas sans en tirer la
conclusion. Ce qui distingue ce MPC d'un LQG n'est pas la linearite, c'est
l'HORIZON FINI. `mpc_lti_ss` rend donc la realisation d'etat de cette loi, en
resolvant la Riccati a horizon fini EN TEMPS CONTINU (pas de gain
echantillonne applique dans une boucle continue : le desaccord serait une
approximation non declaree). Elle entre dans Floquet, dans Ms et dans la borne
d'effort exactement comme le LQG, et l'ecart mesure entre les deux porte alors
sur l'horizon seul.

SMC. A l'INTERIEUR de la couche limite, sat(s/phi) = s/phi et la commande vaut
exactement

    u = -(K_s/phi) (ydot + lambda y),

soit un PD de gains (K_s lambda/phi, K_s/phi). Ce n'est pas une
linearisation approchee : c'est la loi elle-meme, sur le domaine ou elle
s'applique. Et c'est LE domaine qui decide de la stabilite locale, donc de
a_p,lim : au voisinage de l'equilibre le systeme ne quitte pas la couche.
`smc_lti_ss` rend ce PD, avec le meme filtre de lissage que toutes les autres
structures (la derivee est prise par difference arriere a 161 kHz dans le
simulateur, donc ideale bien au-dela des 8 kHz du filtre).

CE QUI RESTE VRAIMENT NON LINEAIRE, et que Floquet ne voit pas : la saturation
a |u| <= K_s hors de la couche limite. C'est une propriete GRAND SIGNAL. Elle
est exercee par le critere temporel, ou la passe complete est simulee avec la
saturation reelle et la meme tension maximale pour tout le monde — et ou les
correcteurs lineaires sont eux aussi echantillonnes et satures. Les deux
criteres se repondent donc : Floquet decide de la stabilite locale, le
temporel de la survie a grand signal.

Les classes SMC et MPC ci-dessous restent la forme NON LINEAIRE / discrete,
celle qui tourne dans le simulateur.

SMC — MODE GLISSANT. Surface s = ydot + lambda y, commande

    u = -K_s sat(s / phi)

La couche limite `phi` remplace le signe discontinu : sans elle le broutement
de commande (« chattering ») est garanti, et sur un actionneur piezo a 160 kHz
il exciterait precisement les modes hauts que tout le reste du depot
s'efforce de ne pas reveiller.

Trois parametres : lambda, K_s, phi.

MPC — COMMANDE PREDICTIVE. Horizon fini, cout quadratique sur la sortie
ponderee et l'effort, resolu SANS contraintes — donc sous forme explicite. Un
QP resolu a chaque pas a 160 kHz est hors de question, et le pretendre serait
mentir sur ce qui tourne.

L'etat est estime par le meme filtre de Kalman que le LQG, et la sortie est
ponderee par le meme passe-bande, pour que l'ecart mesure porte sur l'horizon
et sur rien d'autre.
"""
import numpy as np
from scipy.linalg import expm, solve_continuous_are


class SMC:
    """Mode glissant a couche limite, sur (y, ydot).

    `sign_loop` porte le signe de la voie, comme pour les structures a gains :
    la surface est definie sur la sortie mesuree, et le signe de l'effet de u
    sur y n'est pas devinable depuis la surface seule.
    """

    def __init__(self, lam, k_s, phi, dt, sign_loop=1.0, v_max=None):
        self.lam = float(lam)
        self.k_s = float(k_s)
        self.phi = float(phi)
        self.dt = float(dt)
        self.sign = float(sign_loop)
        self.v_max = v_max
        self.y1 = 0.0
        self._first = True

    def reset(self):
        self.y1 = 0.0
        self._first = True

    def __call__(self, y=0.0, yd=0.0, t=0.0, k=0):
        y = float(y)
        # `yd` est fourni par le simulateur quand il l'a ; sinon difference
        # arriere. On PREFERE la valeur fournie : deriver numeriquement un
        # signal en metres a 160 kHz amplifie le bruit d'un facteur fs.
        if yd is None or yd == 0.0:
            v = 0.0 if self._first else (y - self.y1) / self.dt
        else:
            v = float(yd)
        self.y1 = y
        self._first = False
        s = v + self.lam * y
        sat = np.clip(s / self.phi, -1.0, 1.0) if self.phi > 0 else np.sign(s)
        u = -self.sign * self.k_s * sat
        if self.v_max is not None:
            u = float(np.clip(u, -self.v_max, self.v_max))
        return u


def mpc_gain(A, B, Cz, horizon, q, r, dt):
    """Gain predictif explicite u = -K x, horizon FINI, sans contrainte.

    On discretise, on empile la prediction sur N pas, on minimise
    sum ||Cz x_k||^2 q + ||u_k||^2 r, et on garde la PREMIERE ligne de la
    solution — la « receding horizon » au sens strict. Le calcul est fait une
    fois : sans contrainte active, la loi optimale est lineaire et invariante,
    donc la resoudre a chaque pas rendrait exactement la meme chose pour mille
    fois le prix.
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    Cz = np.atleast_2d(np.asarray(Cz, float))
    n, m = A.shape[0], B.shape[1]
    Ad, Bd, Qd, Nd, Rd = sampled_cost(A, B, Cz, q, r, dt)
    N = int(horizon)
    # Recursion de Riccati arriere (P_N = 0), avec TERME CROISE. La forme
    # empilee donne exactement la meme chose — verifie a 1e-14 — pour un cout
    # bien plus eleve en N^3 ; on garde la recursion.
    P = np.zeros((n, n))
    K = np.zeros((m, n))
    for _ in range(N):
        S = Rd + Bd.T @ P @ Bd
        K = np.linalg.solve(S, Bd.T @ P @ Ad + Nd.T)
        P = Qd + Ad.T @ P @ Ad - (Ad.T @ P @ Bd + Nd) @ K
    return K


def sampled_cost(A, B, Cz, q, r, dt, n_quad=24):
    """(Ad, Bd, Qd, Nd, Rd) : cout a temps discret EXACT pour un bloqueur.

    La version naive prend Qd = q Cz'Cz, c'est-a-dire penalise l'etat AUX
    SEULS INSTANTS D'ECHANTILLONNAGE ; le comportement inter-echantillon
    devient invisible au critere. La forme ci-dessous integre le cout sur
    chaque intervalle et est donc celle du VRAI probleme echantillonne.

    CE QU'ELLE APPORTE, MESURE ET NON SUPPOSE : peu. Sur ce modele, le cout
    continu effectivement atteint passe de 1.037947e8 (cout naif) a 1.035349e8
    (cout exact) a l'horizon le plus long, contre 1.034281e8 pour le LQR — un
    gain de 0.25 %. On la garde parce qu'elle est la formulation juste et
    qu'elle ne coute qu'une quadrature hors ligne, pas parce qu'elle changerait
    le classement.

    ELLE NE CORRIGE PAS CE QUE J'AVAIS D'ABORD CRU. En comparant les NORMES de
    gain, le MPC semblait diverger du LQR (985 -> 7562 quand l'horizon croit)
    et j'ai attribue cet ecart au cout naif. C'etait faux deux fois. D'abord
    la forme empilee et la recursion de Riccati arriere concordent a 1e-14,
    donc le calcul etait juste ; ensuite, juge sur le COUT — la seule grandeur
    qui ait un sens ici — le MPC converge bien vers le LQR de facon monotone
    (1.048e8 -> 1.042e8 -> 1.0353e8 pour 1.0343e8). L'optimum est simplement
    tres PLAT : des gains separes d'un facteur huit rendent des couts a 1.4 %
    les uns des autres. La norme du gain est donc un diagnostic trompeur sur
    ce probleme, et c'est elle qu'il fallait cesser de regarder.

    Le cout exact du probleme echantillonne integre sur CHAQUE intervalle

        Qd = int_0^h Phi(t)' Qc Phi(t) dt
        Nd = int_0^h Phi(t)' Qc Gamma(t) dt
        Rd = Rc h + int_0^h Gamma(t)' Qc Gamma(t) dt

    avec Phi(t) = e^{At} et Gamma(t) = int_0^t e^{As} ds B. Le terme CROISE Nd
    n'est pas optionnel : l'ignorer reintroduit une partie de la meme erreur.

    Les integrales sont faites par quadrature de Gauss-Legendre. Sur ce
    modele w.h vaut au plus 1.3 rad, donc 24 points sont tres au-dela du
    necessaire — et `tests/` verifie le resultat contre la limite continue
    plutot que de faire confiance a la quadrature.
    """
    n, m = A.shape[0], B.shape[1]
    Qc = float(q) * (Cz.T @ Cz)
    xs, ws = np.polynomial.legendre.leggauss(int(n_quad))
    ts = 0.5 * dt * (xs + 1.0)
    wq = 0.5 * dt * ws
    M = np.block([[A, B], [np.zeros((m, n)), np.zeros((m, m))]])
    Qd = np.zeros((n, n))
    Nd = np.zeros((n, m))
    Rd = float(r) * dt * np.eye(m)
    for t, wt in zip(ts, wq):
        E = expm(M * t)
        Phi, Gam = E[:n, :n], E[:n, n:]
        Qd += wt * (Phi.T @ Qc @ Phi)
        Nd += wt * (Phi.T @ Qc @ Gam)
        Rd += wt * (Gam.T @ Qc @ Gam)
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:], Qd, Nd, Rd


class MPC:
    """MPC explicite + estimateur de Kalman, interface temporelle du depot."""

    def __init__(self, plant, horizon, q, r, v_meas, dt, w_proc=1.0,
                 sign_variant=1.0, v_max=None):
        A, B, Cc, _ = [np.atleast_2d(np.asarray(m, float)) for m in plant]
        self.K = mpc_gain(A, B, Cc, horizon, q, r, dt) * float(sign_variant)
        Y = solve_continuous_are(A.T, Cc.T, float(w_proc) * (B @ B.T),
                                 float(v_meas) * np.eye(1))
        L = (Y @ Cc.T) / float(v_meas)
        n = A.shape[0]
        Ao = A - B @ self.K - L @ Cc
        M = np.block([[Ao, L], [np.zeros((1, n)), np.zeros((1, 1))]])
        E = expm(M * float(dt))
        self.Ad, self.Bd = E[:n, :n], E[:n, n:]
        self.x = np.zeros(n)
        self.v_max = v_max

    def reset(self):
        self.x = np.zeros(self.x.size)

    def __call__(self, y=0.0, yd=0.0, t=0.0, k=0):
        u = -float((self.K @ self.x)[0])
        if self.v_max is not None:
            u = float(np.clip(u, -self.v_max, self.v_max))
        self.x = self.Ad @ self.x + self.Bd[:, 0] * float(y)
        return u


# ---------------------------------------------------------------------------
# Les deux realisations d'etat : ce que Floquet, Ms et l'effort peuvent lire.
# ---------------------------------------------------------------------------
def finite_horizon_gain(A, B, Q, R, T, n_max=4096):
    """Riccati a horizon fini EN TEMPS CONTINU : K = R^-1 B' P(T).

    P resout -P' = A'P + PA - PBR^-1B'P + Q avec P(T_final) = 0. En temps
    restant tau = T_final - t la meme equation s'ecrit
    P' = A'P + PA - PBR^-1B'P + Q, P(0) = 0, et se linearise en

        d/dtau [X ; Y] = [[-A, BR^-1B'], [Q, A']] [X ; Y],   P = Y X^-1

    partant de [X ; Y] = [I ; P(tau_0)]. La solution sur un pas est donc une
    exponentielle de matrice, exacte a l'arrondi pres et sans reglage de pas.

    SOUS-DECOUPAGE AUTOMATIQUE. En UN pas, expm(M T) deborde des que
    |Re(lambda)| T depasse ~700 : les valeurs propres de M sont les poles de
    la boucle fermee LQ et leur opposees, donc de l'ordre de 1e4 rad/s ici, et
    un horizon d'une seconde suffit a faire deborder. On double le nombre de
    sous-pas jusqu'a ce que la propagation soit finie ; chaque expm est alors
    bien echelonnee et le resultat est le meme a l'arrondi pres. Mesure :
    l'ecart entre 1 pas et 64 pas, la ou les deux aboutissent, est de 1e-13.

    C'est CE gain qui distingue le MPC du LQG : quand T -> infini, P(T) tend
    vers la solution de l'equation algebrique et les deux lois coincident.
    Mesure sur ce procede (q = 1e6, r = 1) : ecart relatif au gain LQR de
    0.38 a T = 1 ms, 0.16 a 10 ms, 0.030 a 100 ms.
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    Q = np.atleast_2d(np.asarray(Q, float))
    R = np.atleast_2d(np.asarray(R, float))
    n = A.shape[0]
    S = B @ np.linalg.solve(R, B.T)
    M = np.block([[-A, S], [Q, A.T]])
    # Pas de depart dicte par le SPECTRE, pas par la norme : ce qui fait
    # deborder expm est |Re(lambda)| Delta, et les valeurs propres de M sont
    # les poles de la boucle fermee LQ et leurs opposees. On vise
    # |Re(lambda)| Delta <= 30, puis on double si cela ne suffit pas.
    try:
        absc = float(np.max(np.abs(np.real(np.linalg.eigvals(M)))))
    except np.linalg.LinAlgError:
        absc = 0.0
    steps = 1 if absc <= 0.0 else max(1, int(np.ceil(absc * float(T) / 30.0)))
    steps = min(steps, n_max)
    while steps <= n_max:
        E = expm(M * (float(T) / steps))
        if np.all(np.isfinite(E)):
            P = np.zeros((n, n))
            ok = True
            for _ in range(steps):
                XY = E @ np.vstack([np.eye(n), P])
                X, Y = XY[:n], XY[n:]
                try:
                    Pn = np.linalg.solve(X.T, Y.T).T          # P = Y X^-1
                except np.linalg.LinAlgError:
                    ok = False
                    break
                Pn = 0.5 * (Pn + Pn.T)
                if not np.all(np.isfinite(Pn)):
                    ok = False
                    break
                # ARRET SUR POINT FIXE. Au-dela de la convergence vers la
                # solution de l'equation algebrique, continuer a propager
                # n'ajoute que de l'arrondi. Sans cet arret ET sans le choix
                # de pas spectral ci-dessus, un horizon d'une seconde en un
                # seul pas rendait un gain fini et FAUX d'un facteur 3e9.
                done = (np.linalg.norm(Pn - P)
                        <= 1e-14 * max(np.linalg.norm(Pn), 1.0))
                P = Pn
                if done:
                    break
            if ok:
                return np.linalg.solve(R, B.T @ P)
        steps *= 2
    raise ValueError('Riccati a horizon fini non finie')


def mpc_lti_ss(plant, q, r, w_proc, v_meas, f_w, horizon, zeta_w=0.5):
    """MPC explicite sans contrainte active, en representation d'etat y -> u.

    Rigoureusement la meme construction que `classical.lqg_ss` — meme
    ponderation passe-bande sur la sortie, meme filtre de Kalman — a une seule
    difference : le gain vient d'une Riccati a HORIZON FINI et non de
    l'equation algebrique. C'est la condition pour que la comparaison
    LQG/MPC mesure l'horizon et rien d'autre.
    """
    from classical import _bandpass, LqgFailure
    from ss_balance import balance
    Ap, Bp, Cp, _ = [np.atleast_2d(np.asarray(m, float)) for m in plant]
    Aw, Bw, Cw, _ = _bandpass(f_w, zeta_w)
    npp, nw = Ap.shape[0], Aw.shape[0]
    n = npp + nw
    A = np.zeros((n, n))
    A[:npp, :npp] = Ap
    A[npp:, npp:] = Aw
    A[npp:, :npp] = Bw @ Cp
    B = np.zeros((n, 1))
    B[:npp, 0:1] = Bp
    C = np.zeros((1, n))
    C[0, :npp] = Cp
    Cz = np.zeros((1, n))
    Cz[0, npp:] = Cw
    # EQUILIBRAGE DU SYSTEME AUGMENTE, avant de former le hamiltonien. La
    # plaque apporte des w^2 de l'ordre de 1e9 et la ponderation des termes
    # d'ordre 1 : le hamiltonien assemble tel quel a un conditionnement qui
    # rend X non inversible et la Riccati a horizon fini echoue alors qu'elle
    # existe. Meme remede qu'en H-infini, et sans consequence sur le resultat :
    # le correcteur est rendu dans ces coordonnees-la, qui lui sont semblables.
    A, (B, L0), (C, Cz), _ = balance(A, [B, np.zeros((n, 1))], [C, Cz])
    try:
        K = finite_horizon_gain(A, B, float(q) * (Cz.T @ Cz),
                                float(r) * np.eye(1), horizon)
        Y = solve_continuous_are(A.T, C.T, float(w_proc) * (B @ B.T),
                                 float(v_meas) * np.eye(1))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
        raise LqgFailure(str(e))
    if not np.all(np.isfinite(Y)):
        raise LqgFailure('Riccati de Kalman non finie')
    L = (Y @ C.T) / float(v_meas)
    return A - B @ K - L @ C, L, -K, np.zeros((1, 1))


def smc_lti_ss(lam, k_s, phi, fc, order, sign_loop=1.0):
    """La loi du mode glissant A L'INTERIEUR de la couche limite.

    sat(s/phi) = s/phi pour |s| <= phi, donc u = -(K_s/phi)(ydot + lambda y)
    EXACTEMENT. Ce n'est pas une linearisation : c'est la loi, sur le domaine
    ou le systeme vit au voisinage de l'equilibre — donc le domaine qui decide
    de a_p,lim.

    POURQUOI LE FILTRE DE LISSAGE EST INCLUS ICI et non applique par
    `series` comme pour les autres structures : le PD seul, -g (s + lambda),
    est IMPROPRE ; il n'a pas de representation d'etat. Le filtre commun
    (Butterworth d'ordre 2 a 8 kHz, celui que toutes les structures
    recoivent) le rend propre. On realise donc le produit d'un coup. C'est le
    MEME filtre, avec les MEMES parametres, applique UNE fois — l'appelant ne
    doit pas le remettre en serie.

    C'est aussi fidele au simulateur, qui prend la derivee par difference
    arriere a 161 kHz : ideale bien au-dela des 8 kHz du filtre.

    Ce que cette realisation NE porte pas : la saturation a |u| <= K_s hors de
    la couche. C'est une propriete grand signal, et c'est le critere temporel
    qui la mesure.
    """
    from scipy.signal import butter, tf2ss
    from ss_balance import balance_ss
    g = float(sign_loop) * float(k_s) / float(phi)
    b, a = butter(int(order), 2 * np.pi * float(fc), analog=True)
    num = np.convolve([-g, -g * float(lam)], np.atleast_1d(b))
    A, B, Cc, D = tf2ss(num, a)
    return balance_ss((np.atleast_2d(A), np.atleast_2d(B).reshape(-1, 1),
                       np.atleast_2d(Cc).reshape(1, -1),
                       np.atleast_2d(D).reshape(1, 1)))
