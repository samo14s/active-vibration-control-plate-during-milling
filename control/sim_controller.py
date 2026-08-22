"""
sim_controller.py — Correcteur LTI discretise pour la simulation temporelle
===========================================================================
Bloqueur d'ordre zero exact (cont2discrete, methode 'zoh') : y [m] -> u [V].
Le MEME code sert aux deux correcteurs compares.
"""
import numpy as np
from scipy.signal import cont2discrete


class LTIController:
    def __init__(self, ss, dt):
        A, B, C, D = [np.atleast_2d(np.asarray(m, float)) for m in ss]
        n = A.shape[0]
        if n:
            self.Ad, self.Bd, _, _, _ = cont2discrete((A, B, C, D), dt,
                                                      method='zoh')
        else:
            self.Ad, self.Bd = np.zeros((0, 0)), np.zeros((0, 1))
        self.C, self.D = C, D
        self.x = np.zeros(n)

    def reset(self):
        self.x = np.zeros(self.x.size)

    def __call__(self, y=0.0, yd=0.0, t=0.0, k=0):
        u = float(self.D[0, 0]) * y
        if self.x.size:
            u += float((self.C @ self.x)[0])
            self.x = self.Ad @ self.x + self.Bd[:, 0] * y
        return u


class DelayedPDController:
    """Correcteur robuste + controle a retard actif, Eq. (30) de Du et al.

        u(t) = K(s) y(t)  +  K_Pp y(t - tau)  +  K_Pd y'(t - tau)

    Le retard est CELUI DE LA COUPE — une periode de dent — et le simulateur
    pose dt = tau / n_sub exactement (voir `simulate.py`), donc le retard vaut
    un nombre ENTIER de pas et se lit dans l'historique sans interpolation.
    Le prendre en secondes puis arrondir introduirait ici une erreur de phase
    de l'ordre du pas, sur le terme meme dont tout l'interet est la phase.

    Avant que l'historique soit rempli (les n_sub premiers pas), le terme
    retarde vaut zero : c'est la meme convention que la coupe elle-meme, dont
    `simulate.py` prend q(t - tau) = 0 tant que k <= n_sub.

    `u_rob_last` et `u_pd_last` exposent la DECOMPOSITION du dernier appel,
    que `simulate.run(record_split=True)` enregistre : sans elle on ne pourrait
    pas dire lequel des deux termes fait le travail.
    """

    def __init__(self, ss, pd, lag, dt):
        self.inner = None if ss is None else LTIController(ss, dt)
        self.Kp, self.Kd = float(pd[0]), float(pd[1])
        self.lag = int(lag)
        if self.lag < 1:
            raise ValueError('retard nul : ce serait une autre loi')
        self.reset()

    def reset(self):
        if self.inner is not None:
            self.inner.reset()
        self.hy = [0.0] * (self.lag + 1)
        self.hyd = [0.0] * (self.lag + 1)
        self.i = 0
        self.u_rob_last = 0.0
        self.u_pd_last = 0.0

    def __call__(self, y=0.0, yd=0.0, t=0.0, k=0):
        n = self.lag + 1
        j = (self.i - self.lag) % n
        self.u_pd_last = self.Kp * self.hy[j] + self.Kd * self.hyd[j]
        self.hy[self.i % n] = float(y)
        self.hyd[self.i % n] = float(yd)
        self.i += 1
        self.u_rob_last = (0.0 if self.inner is None
                           else float(self.inner(y=y, yd=yd, t=t, k=k)))
        return self.u_rob_last + self.u_pd_last
