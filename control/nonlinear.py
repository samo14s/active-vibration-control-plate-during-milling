"""
nonlinear.py — SMC et MPC : les deux structures qui NE PEUVENT PAS entrer
=========================================================================
dans le critere de Floquet, et pourquoi ce n'est pas un choix.

LE POINT QUI DECIDE DE TOUT. Le mode glissant et la commande predictive ne
sont pas lineaires invariants. Or l'objectif du protocole est une marge de
FLOQUET, et Ms comme l'effort maximal sont des normes frequentielles : les
trois presupposent qu'il EXISTE une fonction de transfert du correcteur. Pour
ces deux structures il n'y en a pas. Les faire noter par cet objectif ne
serait pas severe ou genereux, ce serait DEPOURVU DE SENS — on appliquerait
un theoreme hors de ses hypotheses.

Elles entrent donc par le seul critere qui vaut pour TOUTES les structures,
lineaires ou non : la survie d'une passe complete en simulation temporelle,
avec la meme saturation, les memes conditions initiales et les memes
positions. Ce n'est pas une derogation inventee pour l'occasion :
`run_adaptive.py` a ete ecrit exactement pour cela quand l'observateur
SUPERVISE — lui non plus pas LTI — a du etre compare aux trois structures
lineaires. La regle existait avant ces deux-ci.

CE QUE CELA COUTE, ET IL FAUT LE DIRE. Ces deux structures ne recevront donc
NI marge de Floquet, NI Ms, NI borne d'effort frequentielle. Leur colonne dans
le tableau de comparaison sera partiellement vide, et cette absence est un
resultat en soi : une structure dont on ne sait pas certifier la stabilite par
les outils du domaine est, en atelier, une structure dont on ne sait pas
certifier la stabilite.

SMC — MODE GLISSANT. Surface s = ydot + lambda y, commande

    u = -K_s sat(s / phi)

La couche limite `phi` remplace le signe discontinu : sans elle le broutement
de commande (« chattering ») est garanti, et sur un actionneur piezo a 160 kHz
il exciterait precisement les modes hauts que tout le reste du depot s'efforce
de ne pas reveiller. C'est le meme compromis que partout ailleurs ici : la
robustesse theorique du signe pur contre un signal realisable.

Trois parametres : lambda, K_s, phi.

MPC — COMMANDE PREDICTIVE. Horizon fini sur le modele modal reduit, cout
quadratique sur la sortie predite et l'effort, resolu SANS contraintes — donc
sous forme explicite, u = -K_mpc x_hat, avec K_mpc calcule une fois hors
ligne. C'est le MPC honnete a ce niveau de simulation : un QP resolu a chaque
pas a 160 kHz est hors de question, et le pretendre serait mentir sur ce qui
tourne. La difference avec un LQR a horizon infini est alors l'horizon FINI,
qui est precisement ce que le MPC apporte ici.

L'etat est estime par le meme filtre de Kalman que le LQG, pour que l'ecart
mesure porte sur la loi de commande et non sur l'estimateur.

Quatre parametres : horizon, ponderation de sortie, ponderation d'effort,
intensite du bruit de mesure de l'estimateur.
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
