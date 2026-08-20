"""
fdob_adaptive.py — l'observateur modal SUPERVISE
=================================================
La faiblesse mesuree de l'observateur modal a bandes fixes est nette
(`OBSERVATEUR_MODAL.md` §6.5) : a la derive modale que le papier constate
reellement, +17 % sur le mode 1 et +9 % sur le mode 2, sa limite tombe de
0.222 a 0.117 mm — sous celle de l'ADRC-FOPID. La cause est mecanique : un
passe-bande de largeur relative 0.7 % a le mode entierement hors de sa
fenetre des que celui-ci bouge de 17 %.

Ce module attaque exactement cela, et rien d'autre :

  * DEUX estimateurs de frequence, un par mode de broutement, chacun borne a
    sa propre sous-bande (400-800 Hz et 800-1400 Hz). Un seul estimateur ne
    suffirait pas : la derive constatee n'est PAS homothetique (+17 % contre
    +9 %), donc un facteur d'echelle unique replacerait correctement l'un des
    deux passe-bande et deplacerait l'autre a cote.
  * RECENTRAGE des passe-bande de l'observateur sur les frequences estimees.
  * SUPERVISION de alpha par le niveau de broutement : alpha monte de 0 a
    alpha_max quand le niveau passe de `level_on` a `level_hi`, avec une
    constante de temps qui evite les a-coups de commande.

CE QUE CELA COUTE, ET IL FAUT LE DIRE. Le correcteur n'est plus lineaire
invariant : ni Floquet ni les marges frequentielles ne s'y appliquent. Toute
comparaison avec les trois structures LTI doit donc se faire sur un critere
qui vaut pour LES QUATRE — ici la survie d'une passe complete en simulation
temporelle, avec la meme saturation et les memes conditions. C'est ce que
fait `control/run_adaptive.py`, qui reevalue AUSSI les structures LTI avec ce
critere plutot que de comparer des chiffres obtenus autrement.

Les reglages du superviseur (seuils, constantes de temps, facteur d'oubli)
sont FIXES et communs : ils ne passent pas par le PSO. Le correcteur garde
donc les sept parametres optimises de la version a bandes fixes, et le
superviseur s'ajoute par-dessus sans en reoptimiser aucun. C'est volontaire :
on mesure ce qu'apporte la supervision SEULE, sans la melanger a un gain
d'optimisation.
"""
import numpy as np
from scipy.signal import cont2discrete

from chatter_estimator import ChatterEstimator
from fdob import _modal_blocks
from fopid import fopid_ss, rolloff_ss


def _zoh(A, B, dt):
    n = np.atleast_2d(A).shape[0]
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 1))
    Ad, Bd, _, _, _ = cont2discrete(
        (A, B, np.zeros((1, n)), np.zeros((1, 1))), dt, method='zoh')
    return Ad, Bd


class AdaptiveFDOB:
    """Observateur modal a bandes RECENTREES + melange alpha supervise.

    `par` : dictionnaire des sept parametres optimises (Kp, Ki, Kd, lam, mu,
    zeta_q, alpha) — alpha y est lu comme alpha_MAX, la valeur atteinte en
    plein broutement.
    """

    def __init__(self, par, w, zeta, res, wc, dt, f_tooth, sign_loop,
                 wb, wh, N, rolloff_hz, rolloff_order,
                 adapt=True, supervise=True, tol=2e-3,
                 level_hi=0.60, tau_alpha=20e-3, mode='global',
                 alpha_floor=None):
        self.p = dict(par)
        self.w0 = np.asarray(w, float).copy()      # pulsations NOMINALES
        self.zeta = np.asarray(zeta, float)
        self.res = np.asarray(res, float)
        self.wc = float(wc)
        self.dt = float(dt)
        self.adapt = bool(adapt)
        self.supervise = bool(supervise)
        self.tol = float(tol)
        self.alpha_max = float(par['alpha'])
        self.level_hi = float(level_hi)
        self.b_alpha = float(np.exp(-dt / tau_alpha))
        # 'global' : un seul alpha, de 0 a alpha_max selon le niveau global.
        # 'bande'  : un alpha PAR MODE, partant d'un PLANCHER et montant vers
        #            alpha_max selon le niveau ET la confiance de SON
        #            estimateur.
        #
        # LE PLANCHER N'EST PAS UN DETAIL. La premiere version faisait
        # alpha_k = alpha_max . g_k . conf_k, donc alpha_k -> 0 des qu'une
        # bande n'a pas de verrou. Mesure : sur la plaque NOMINALE a 0.30 mm
        # la passe divergeait (3763 um) alors qu'elle tenait a 4.4 um avec un
        # alpha global. La raison est nette et vaut d'etre retenue : le mode 2
        # ne broute pas, mais l'autorite de l'observateur a ce mode est
        # precisement ce qui l'EMPECHE de brouter. Couper alpha la ou "il ne
        # se passe rien" supprime l'action preventive. Le plancher est donc la
        # valeur validee par le PSO sous Ms <= 2 : sans broutement et sans
        # verrou, on retombe exactement sur l'observateur a bandes fixes.
        assert mode in ('global', 'bande')
        self.mode = mode
        self.alpha_floor = (self.alpha_max if alpha_floor is None
                            else float(alpha_floor))

        # --- epine dorsale FOPID, invariante
        Ac, Bc, Cc, Dc = [np.atleast_2d(np.asarray(m, float)) for m in
                          fopid_ss(par['Kp'], par['Ki'], par['Kd'],
                                   par['lam'], par['mu'], wb, wh, N,
                                   sign_loop)]
        self.Afd, self.Bfd = _zoh(Ac, -Bc, dt)     # entree e = -y
        self.Cf, self.Df = Cc, -float(Dc[0, 0])
        self.xf = np.zeros(Ac.shape[0])

        # --- filtre de lissage, invariant, applique a la sortie
        Ar, Br, Cr, Dr = [np.atleast_2d(np.asarray(m, float))
                          for m in rolloff_ss(rolloff_hz, rolloff_order)]
        self.Ard, self.Brd = _zoh(Ar, Br, dt)
        self.Cr, self.Dr = Cr, float(Dr[0, 0])
        self.xr = np.zeros(Ar.shape[0])

        # --- estimateurs : un par mode vise, chacun dans sa sous-bande
        edges = self._subbands(self.w0 / (2 * np.pi))
        self.est = [ChatterEstimator(1.0 / dt, f_tooth, band=b,
                                     f_nom=float(fk), f_init=float(fk))
                    for fk, b in zip(self.w0 / (2 * np.pi), edges)]
        self.f_hat = (self.w0 / (2 * np.pi)).copy()
        # Sans supervision alpha vaut d'emblee sa valeur nominale : sinon la
        # rampe de 20 ms suffit a elle seule a rendre le cas "supervision
        # desactivee" different du correcteur LTI, et le test d'equivalence
        # ne prouverait plus rien.
        self.alpha = self._alpha_init()
        self.level = 0.0
        self.conf = np.zeros(len(self.w0))
        self.n_rebuild = 0
        self._build(self.w0)

    @staticmethod
    def _subbands(f):
        """Sous-bande de chaque mode : a mi-chemin geometrique des voisins,
        elargie de 25 % pour laisser la place a la derive."""
        f = np.asarray(f, float)
        out = []
        for i, fi in enumerate(f):
            lo = fi / 1.25 if i == 0 else np.sqrt(f[i - 1] * fi)
            hi = fi * 1.25 if i == len(f) - 1 else np.sqrt(fi * f[i + 1])
            out.append((float(lo), float(hi)))
        return out

    def _build(self, w):
        """(Re)construit les blocs de l'observateur centres sur `w`."""
        bv, bw = _modal_blocks(w, self.zeta, self.res, self.p['zeta_q'],
                               self.wc, separate=True)
        self.blk = []
        for (Av, Bv, Cv, _), (Aw, Bw, Cw, _) in zip(bv, bw):
            Avd, Bvd = _zoh(Av, Bv, self.dt)
            Awd, Bwd = _zoh(Aw, Bw, self.dt)
            self.blk.append((Avd, Bvd, Cv, Awd, Bwd, Cw))
        if not hasattr(self, 'xv'):
            self.xv = [np.zeros(b[0].shape[0]) for b in self.blk]
            self.xw = [np.zeros(b[3].shape[0]) for b in self.blk]
        self.w_built = np.asarray(w, float).copy()
        self.n_rebuild += 1

    def _alpha_init(self):
        """Valeur de depart de alpha. UN SEUL endroit, appele par __init__ et
        par reset() : les deux en avaient chacun sa version, et celle de
        reset() ignorait a la fois le mode 'bande' et le PLANCHER. Elle
        remettait donc le correcteur exactement dans l'etat que le
        commentaire de __init__ decrit comme divergent sur la plaque
        nominale — alpha nul par bande, donc plus d'action preventive au
        mode qui ne broute pas encore.
        """
        n = len(self.w0)
        if not self.supervise:
            return np.full(n, self.alpha_max)
        return np.full(n, self.alpha_floor) if self.mode == 'bande' \
            else np.zeros(n)

    def reset(self):
        self.xf[:] = 0.0
        self.xr[:] = 0.0
        for a in self.xv:
            a[:] = 0.0
        for a in self.xw:
            a[:] = 0.0
        for e in self.est:
            e.reset()
        self.f_hat = (self.w0 / (2 * np.pi)).copy()
        self.alpha = self._alpha_init()
        self._build(self.w0)

    def __call__(self, y=0.0, yd=0.0, t=0.0, k=0):
        y = float(y)
        # --- estimation
        lev = 0.0
        for i, e in enumerate(self.est):
            # Chaque estimateur retire les modes que les AUTRES suivent : sans
            # cela, un mode ayant derive vers le bas de la sous-bande voisine
            # y attire les moindres carres et le recentrage se fait a cote.
            if len(self.est) > 1:
                e.set_exclude([self.f_hat[j] for j in range(len(self.est))
                               if j != i])
            f, _ = e(y)
            # On ne recentre QUE sur un verrou etabli ; sinon on revient a la
            # frequence nominale du mode. Recentrer sur une estimee errante
            # est pire que ne pas recentrer du tout : cela deplace le
            # passe-bande la ou il n'y a aucun mode.
            self.f_hat[i] = f if e.locked else float(self.w0[i] / (2 * np.pi))
            lev = max(lev, e.level_slow)
        self.level = lev

        # --- superviseur
        on = self.est[0].level_on
        self.conf = np.array([e.conf for e in self.est])
        if not self.supervise:
            tgt = np.full(len(self.est), self.alpha_max)
        elif self.mode == 'bande':
            # Chaque bande part du plancher valide et ne monte au-dela que si
            # SON mode broute ET que SON estimee est sure : on ne pousse pas
            # une autorite qu'on ne sait pas viser.
            g = np.array([np.clip((e.level_slow - on)
                                  / max(self.level_hi - on, 1e-9), 0.0, 1.0)
                          for e in self.est])
            tgt = self.alpha_floor + (self.alpha_max - self.alpha_floor) \
                * g * self.conf
        else:
            g = float(np.clip((lev - on) / max(self.level_hi - on, 1e-9),
                              0.0, 1.0))
            tgt = np.full(len(self.est), self.alpha_max * g)
        self.alpha = self.b_alpha * self.alpha + (1 - self.b_alpha) * tgt

        # --- recentrage, avec hysteresis pour ne pas reconstruire sans cesse
        if self.adapt:
            wn = 2 * np.pi * self.f_hat
            if np.any(np.abs(wn / self.w_built - 1.0) > self.tol):
                self._build(wn)

        # --- loi de commande
        u_f = float((self.Cf @ self.xf)[0]) + self.Df * y
        acc = 0.0
        for i, (Avd, Bvd, Cv, Awd, Bwd, Cw) in enumerate(self.blk):
            acc += self.alpha[i] * (float((Cv @ self.xv[i])[0])
                                    - float((Cw @ self.xw[i])[0]))
        u_core = u_f + acc
        u = float((self.Cr @ self.xr)[0]) + self.Dr * u_core

        self.xf = self.Afd @ self.xf + self.Bfd[:, 0] * y
        for i, (Avd, Bvd, Cv, Awd, Bwd, Cw) in enumerate(self.blk):
            self.xw[i] = Awd @ self.xw[i] + Bwd[:, 0] * y
            self.xv[i] = Avd @ self.xv[i] + Bvd[:, 0] * u_core
        self.xr = self.Ard @ self.xr + self.Brd[:, 0] * u_core
        return u
