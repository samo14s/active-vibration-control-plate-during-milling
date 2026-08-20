"""
fopid.py — Operateurs d'ordre fractionnaire et correcteur FOPID
================================================================
    C(s) = K_p + K_i s^(-lambda) + K_d s^(mu),    lambda, mu dans ]0, 1]

Les operateurs s^alpha sont realises par le FILTRE RECURSIF D'OUSTALOUP
(Oustaloup et al., IEEE TCAS-I 47(1), 2000), qui approche s^alpha sur une
bande [w_b, w_h] par une fonction rationnelle d'ordre 2N+1 :

    s^alpha  ~  w_h^alpha * prod_{k=-N..N} (s + w'_k) / (s + w_k)
    w'_k = w_b (w_h/w_b)^((k + N + (1-alpha)/2)/(2N+1))
    w_k  = w_b (w_h/w_b)^((k + N + (1+alpha)/2)/(2N+1))

Toutes les racines sont reelles negatives : le filtre est stable et a phase
minimale. En dehors de la bande il sature (gain constant), ce qui est
souhaitable ici : l'integrateur fractionnaire n'a donc pas de pole exactement
en 0 et ne lutte pas indefiniment contre la deflexion statique de coupe.

MEME realisation (meme bande, meme ordre N) pour les deux correcteurs
compares : c'est une des conditions d'equite du protocole.
"""
import numpy as np

from ss_balance import balance_ss
from scipy.signal import zpk2ss


def oustaloup(alpha, wb, wh, N):
    """Approximation rationnelle de s^alpha : (zeros, poles, gain)."""
    alpha = float(np.clip(alpha, -0.999, 0.999))
    k = np.arange(-N, N + 1)
    n = 2 * N + 1
    wu = wh / wb
    zeros = -wb * wu ** ((k + N + 0.5 * (1.0 - alpha)) / n)
    poles = -wb * wu ** ((k + N + 0.5 * (1.0 + alpha)) / n)
    return zeros, poles, wh ** alpha


def frac_ss(alpha, wb, wh, N):
    """s^alpha en representation d'etat MODALE (A diagonale).

    Les poles d'Oustaloup sont reels, simples et negatifs : la decomposition
    en elements simples est exacte et donne

        s^alpha ~ k + sum_i c_i/(s + p_i),
        c_i = k * prod_j(p_i - z_j) / prod_{j != i}(p_i - p_j)

    d'ou A = diag(-p_i), B = 1, C = c_i, D = k.

    ATTENTION — la forme compagne de `zpk2ss` est INUTILISABLE ici. Les poles
    s'etalent sur cinq decades (2*pi a 6.3e5 rad/s) : les coefficients du
    polynome caracteristique couvrent alors ~1e30, le conditionnement de la
    matrice augmentee de Floquet atteint 1e51 et `expm(A h)` renvoie une norme
    de 1e44 au lieu de <= 1. Toutes les analyses de stabilite devenaient des
    NaN/inf. La forme modale ci-dessous est diagonale : conditionnement ~1.
    """
    z, p, k = oustaloup(alpha, wb, wh, N)
    n = len(p)
    c = np.empty(n)
    for i in range(n):
        num = np.prod(p[i] - z)
        den = np.prod([p[i] - p[j] for j in range(n) if j != i])
        c[i] = k * num / den
    A = np.diag(p)                      # p deja negatifs
    B = np.ones((n, 1))
    C = c.reshape(1, -1)
    D = np.array([[float(k)]])
    return A, B, C, D


def frac_frf(alpha, w, wb, wh, N):
    """Reponse frequentielle de l'approximation (pulsations w)."""
    z, p, k = oustaloup(alpha, wb, wh, N)
    s = 1j * np.asarray(w, float)
    out = np.full(s.shape, k, complex)
    for zz, pp in zip(z, p):
        out *= (s - zz) / (s - pp)
    return out


# ---------------------------------------------------------------------------
def fopid_ss(Kp, Ki, Kd, lam, mu, wb, wh, N, sign_loop=1.0):
    """FOPID complet en representation d'etat : y -> u.

    Retourne (A, B, C, D) tel que u = C x + D y, avec la convention
    u = sign_loop * (K_p + K_i s^-lambda + K_d s^mu) * y  (retour negatif).
    """
    Ai, Bi, Ci, Di = frac_ss(-abs(lam), wb, wh, N)      # s^(-lambda)
    Ad, Bd, Cd, Dd = frac_ss(abs(mu), wb, wh, N)        # s^(mu)
    ni, nd = Ai.shape[0], Ad.shape[0]
    A = np.zeros((ni + nd, ni + nd))
    A[:ni, :ni] = Ai
    A[ni:, ni:] = Ad
    B = np.vstack([Bi, Bd])
    C = sign_loop * np.hstack([Ki * Ci, Kd * Cd])
    D = sign_loop * np.array([[Kp + Ki * Di.item() + Kd * Dd.item()]])
    return A, B, C, D


def fopid_frf(Kp, Ki, Kd, lam, mu, w, wb, wh, N, sign_loop=1.0):
    """Reponse frequentielle du FOPID (utile pour les contraintes)."""
    return sign_loop * (Kp + Ki * frac_frf(-abs(lam), w, wb, wh, N)
                        + Kd * frac_frf(abs(mu), w, wb, wh, N))


# ---------------------------------------------------------------------------
def series(ss1, ss2):
    """Mise en cascade ss2 o ss1 (l'entree traverse ss1 puis ss2)."""
    A1, B1, C1, D1 = [np.atleast_2d(np.asarray(m, float)) for m in ss1]
    A2, B2, C2, D2 = [np.atleast_2d(np.asarray(m, float)) for m in ss2]
    n1, n2 = A1.shape[0], A2.shape[0]
    A = np.zeros((n1 + n2, n1 + n2))
    A[:n1, :n1] = A1
    A[n1:, n1:] = A2
    A[n1:, :n1] = B2 @ C1
    B = np.vstack([B1, B2 @ D1])
    C = np.hstack([D2 @ C1, C2])
    D = D2 @ D1
    # EQUILIBRAGE FINAL. Sans lui, la composition accumule les echelles des
    # deux blocs : le correcteur FDOB stocke atteignait cond(A) = 5.5e19, dont
    # 3.1e4 apres une simple similitude diagonale. Voir ss_balance.py — la
    # fonction de transfert est inchangee, la matrice d'etat ne l'est pas, et
    # c'est la matrice d'etat que Floquet exponentie.
    return balance_ss((A, B, C, D))


def rolloff_ss(fc, order=2):
    """Filtre de lissage passe-bas (Butterworth) applique aux DEUX
    correcteurs.

    Necessaire et equitable : le filtre d'Oustaloup place des poles jusqu'a
    w_h (100 kHz ici), tres au-dessus de la frequence de Nyquist de la
    commande echantillonnee (~20 kHz a n_sub = 164). Sans lissage, cette
    dynamique se replie et fait diverger la simulation alors que toutes les
    metriques frequentielles restent bonnes. Un correcteur reel comporte de
    toute facon un filtre d'anti-repliement ; on l'applique a l'identique aux
    deux structures comparees, il n'avantage donc ni l'une ni l'autre.
    """
    from scipy.signal import butter, tf2ss
    b, a = butter(order, 2 * np.pi * fc, analog=True)
    A, B, C, D = tf2ss(b, a)
    # La forme compagne de Butterworth porte C = [0, (2 pi fc)^2] = [0,
    # 2.53e9] a 8 kHz. C'est petit vu isolement — mais `series` greffe ce bloc
    # sur TOUS les correcteurs compares, et ce 2.53e9 se lisait verbatim dans
    # la ligne C de chaque optimum stocke. Il est la source commune du
    # conditionnement de toutes les realisations du depot.
    return balance_ss((np.atleast_2d(A), np.atleast_2d(B).reshape(-1, 1),
                       np.atleast_2d(C).reshape(1, -1),
                       np.atleast_2d(D).reshape(1, 1)))


# ---------------------------------------------------------------------------
def ss_frf(ss, w):
    """Reponse frequentielle d'un (A, B, C, D) SISO."""
    A, B, C, D = [np.atleast_2d(np.asarray(m, float)) for m in ss]
    n = A.shape[0]
    if n == 0:
        return np.full(len(w), float(D[0, 0]), complex)
    I = np.eye(n)
    out = np.empty(len(w), complex)
    for i, wk in enumerate(w):
        try:
            out[i] = float(D[0, 0]) + (C @ np.linalg.solve(1j * wk * I - A,
                                                           B))[0, 0]
        except np.linalg.LinAlgError:
            out[i] = np.inf
    return out
