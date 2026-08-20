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
