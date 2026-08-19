"""
fdob.py — FOPID + observateur de perturbation CONSCIENT DE LA FREQUENCE
=========================================================================
Cette structure repond point par point au diagnostic de `DIAGNOSTIC_ADRC.md`.
Elle n'est pas une variante de plus : chaque choix ci-dessous corrige un
maillon nomme de la chaine de perte.

CE QUE LE DIAGNOSTIC A ETABLI
------------------------------
1. L'observateur etendu de l'ADRC n'est PAS en faute : z3 = Q(s)[s^2 y - b0 u]
   exactement (verifie a 7.9e-15). Le probleme n'est pas l'estimation.
2. b0 n'est pas un parametre de modele : K(s) = G(s)/b0, c'est un gain global.
   Il ne peut donc pas "suivre le signe de la voie", qui change 8 fois.
3. Le defaut reel est le residu non annule (1-Q)D avec D = s^2 P/b0 - 1, qui
   depasse 1 sur 35 % de la bande et culmine AUX RESONANCES (jusqu'a 11.1).
   Le seul reglage qui l'attaque est w_o, et il faudrait w_o/2pi = 115 kHz.
4. En forme fermee, l'ESO impose K ~ (w_o/3b0)[-C(s)/s - s] sous w_o : un
   integrateur ENTIER de plus, la derivee fractionnaire d'ordre mu dans (0,1)
   HORS d'atteinte, et six degres de liberte au lieu de sept.

Autrement dit : le mal est que l'ADRC prend le procede pour un double
integrateur `b0/s^2` et appelle "perturbation" tout le reste — donc le modele
modal lui-meme, qu'on connait pourtant et qu'on a valide.

CE QUE CETTE STRUCTURE CHANGE
------------------------------
* Le modele nominal n'est plus `b0/s^2` mais le MODELE MODAL, mode par mode :
  P_k(s) = r_k/(s^2 + 2 zeta_k w_k s + w_k^2). Le defaut de modele D n'est
  donc plus d'ordre 10 aux resonances : il y est d'ordre 1.
* L'inversion est MODALE, jamais globale. C'est une necessite mathematique,
  pas une commodite : le procede complet a un ZERO DANS LE DEMI-PLAN DROIT a
  2459 Hz (verifie robuste a la troncature : 2459-2793 Hz de 5 a 10 modes),
  donc P^-1 est instable et aucun observateur de perturbation classique ne
  peut l'employer. En revanche le modele reduit aux modes 1-2 (et 1-3) est
  A DEPHASAGE MINIMAL — ses zeros sont a 795 Hz, partie reelle -10.9 — donc
  chaque P_k^-1 modal existe et est stable.
* Le filtre de l'observateur n'est plus un passe-bas a UN parametre mais un
  banc de passe-BANDE centres sur les modes vises :

      Q_k(s) = [2 zeta_q w_k s / (s^2 + 2 zeta_q w_k s + w_k^2)] . [w_c/(s+w_c)]^2

  Placement (w_k) et profondeur (alpha) sont ainsi SEPARES, ce que w_o ne
  permettait pas. Le double pole en w_c rend Q_k P_k^-1 strictement propre,
  donc la structure n'ajoute AUCUN passage direct large bande de y vers u.
* Le signe est porte MODE PAR MODE par 1/r_k. C'est la reponse au maillon 3 :
  la ou b0 est un scalaire unique pour toute la bande, ici chaque mode vise
  apporte le signe de son propre residu. Avec le jeu complet
  r = [-0.65 -0.98 -0.88 +3.29 +2.62], viser les cinq modes exerce vraiment
  cette propriete ; viser les deux modes de broutement ne l'exerce pas (leurs
  residus sont de meme signe) mais suit la prescription du maillon 6 — ne
  rien depenser au-dela du zero instable. Les deux jeux sont donc essayes.
* Le FOPID reste l'EPINE DORSALE et l'observateur s'ajoute EN PARALLELE :

      u = u_FOPID + alpha . u_DO ,   u_DO = -sum_k Q_k (P_k^-1 y - u)

  A alpha = 0 on retrouve le FOPID EXACTEMENT. C'est la propriete que
  l'ADRC-FOPID n'avait pas : son ensemble de correcteurs realisables n'est
  pas plus grand que celui du FOPID, il est DECALE (ordres -1, -(1+lam),
  mu-1, +1 contre 0, -lam, +mu). Ici il est un vrai sur-ensemble, donc la
  structure ne peut pas etre battue par son propre cas particulier.

FORME FERMEE
-------------
En eliminant u_DO :

    u (1 - alpha sum_k Q_k) = u_FOPID - alpha sum_k Q_k P_k^-1 y

                  C(s) + alpha sum_k Q_k(s) P_k^-1(s)
    K(s) = u/y = -------------------------------------- . (-1)
                        1 - alpha sum_k Q_k(s)

avec u_FOPID = -C(s) y et C(s) = Kp + Ki s^-lam + Kd s^mu.

PARAMETRES : Kp, Ki, Kd, lam, mu (comme le FOPID) + zeta_q, alpha.
Soit SEPT — exactement autant que l'ADRC-FOPID, avec les memes graines et le
meme essaim. Les pulsations w_k ne sont pas ajustees : ce sont les frequences
MESUREES de la plaque (Tableau 4), au meme titre que le b0 nominal que la
fabrique ADRC recevait gratuitement. w_c est fixe a la coupure du filtre
d'anti-repliement (8 kHz), commune aux trois structures.
"""
import numpy as np
from scipy.signal import tf2ss

from fopid import fopid_ss


# ---------------------------------------------------------------------------
def _parallel(blocks):
    """Somme de plusieurs (A, B, C, D) SISO montes en parallele."""
    As, Bs, Cs, Ds = [], [], [], []
    for A, B, C, D in blocks:
        As.append(np.atleast_2d(A))
        Bs.append(np.atleast_2d(B).reshape(-1, 1))
        Cs.append(np.atleast_2d(C).reshape(1, -1))
        Ds.append(float(np.atleast_2d(D)[0, 0]))
    n = sum(a.shape[0] for a in As)
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    C = np.zeros((1, n))
    i = 0
    for Ak, Bk, Ck in zip(As, Bs, Cs):
        m = Ak.shape[0]
        A[i:i + m, i:i + m] = Ak
        B[i:i + m, 0] = Bk[:, 0]
        C[0, i:i + m] = Ck[0]
        i += m
    return A, B, C, np.array([[sum(Ds)]])


def _tf(num, den):
    """(A, B, C, D) d'une fraction rationnelle propre."""
    A, B, C, D = tf2ss(np.asarray(num, float), np.asarray(den, float))
    return (np.atleast_2d(A), np.atleast_2d(B).reshape(-1, 1),
            np.atleast_2d(C).reshape(1, -1), np.atleast_2d(D).reshape(1, 1))


def _modal_blocks(w, zeta, res, zeta_q, wc):
    """Les deux blocs du banc : V = sum Q_k  et  W = sum Q_k P_k^-1.

    Q_k(s)        = 2 zq wk s . wc^2 / [(s^2 + 2 zq wk s + wk^2)(s + wc)^2]
    Q_k P_k^-1(s) = Q_k(s) . (s^2 + 2 zk wk s + wk^2) / r_k

    Les deux sont STRICTEMENT PROPRES (degre 1 puis 3 au numerateur contre 4
    au denominateur), donc ni l'un ni l'autre n'ajoute de passage direct.
    """
    bv, bw = [], []
    lp = np.polymul([1.0, wc], [1.0, wc])            # (s + wc)^2
    for wk, zk, rk in zip(w, zeta, res):
        bp_num = [2.0 * zeta_q * wk, 0.0]            # 2 zq wk s
        bp_den = [1.0, 2.0 * zeta_q * wk, wk ** 2]
        den = np.polymul(bp_den, lp)
        num_q = np.polymul(bp_num, [wc ** 2])
        bv.append(_tf(num_q, den))
        inv = np.array([1.0, 2.0 * zk * wk, wk ** 2]) / rk
        bw.append(_tf(np.polymul(num_q, inv), den))
    return _parallel(bv), _parallel(bw)


def fdob_fopid_ss(Kp, Ki, Kd, lam, mu, zeta_q, alpha, w, zeta, res, wc,
                  wb, wh, N, sign_loop=1.0):
    """FOPID + observateur de perturbation modal : representation d'etat y -> u.

    `w`, `zeta`, `res` decrivent les modes VISES par l'observateur (pulsation,
    amortissement, residu D_obs(k) H_Pe(k)). Ils viennent du modele de la
    plaque, pas de l'optimiseur.

    Etats : [FOPID (Oustaloup), W = sum Q_k P_k^-1, V = sum Q_k].
    """
    Ac, Bc, Cc, Dc = [np.atleast_2d(np.asarray(m, float))
                      for m in fopid_ss(Kp, Ki, Kd, lam, mu, wb, wh, N,
                                        sign_loop)]
    (Av, Bv, Cv, Dv), (Aw, Bw, Cw, Dw) = _modal_blocks(w, zeta, res,
                                                       zeta_q, wc)
    nc, nw, nv = Ac.shape[0], Aw.shape[0], Av.shape[0]
    n = nc + nw + nv

    # u = -C(s) y - alpha W(s) y + alpha V(s) u      (Dv = Dw = 0)
    u_row = np.zeros(n)
    u_row[:nc] = Cc[0]
    u_row[nc:nc + nw] = -alpha * Cw[0]
    u_row[nc + nw:] = alpha * Cv[0]
    d_u = -float(Dc[0, 0])                    # passage direct : celui du FOPID

    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    # FOPID, entree e = -y
    A[:nc, :nc] = Ac
    B[:nc, 0] = -Bc[:, 0]
    # W, entree y
    A[nc:nc + nw, nc:nc + nw] = Aw
    B[nc:nc + nw, 0] = Bw[:, 0]
    # V, entree u  (u lui-meme fonction des etats et de y : pas de boucle
    # algebrique puisque Dv = 0)
    A[nc + nw:, nc + nw:] = Av
    A[nc + nw:, :] += Bv @ u_row.reshape(1, -1)
    B[nc + nw:, 0] = Bv[:, 0] * d_u

    C = u_row.reshape(1, -1)
    D = np.array([[d_u]])
    return A, B, C, D


def target_modes(plate, which):
    """(w, zeta, res) des modes vises. `which` est une liste d'indices 0-based.

    Les residus r_k = D_obs(k) H_Pe(k) portent le SIGNE de la voie a ce mode.
    C'est par eux que la structure devient consciente du signe, mode par mode,
    la ou b0 etait un scalaire unique pour toute la bande.
    """
    w = np.asarray(plate.omega_n, float)[list(which)]
    z = np.asarray(plate.zeta_modes, float)[list(which)]
    r = (plate.D_row(plate.lp, plate.hp)[list(which)]
         * np.asarray(plate.H_Pe_modal, float)[list(which)])
    return w, z, r
