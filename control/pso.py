"""
pso.py — Essaim particulaire et fabriques de correcteurs
=========================================================
PSO canonique a inertie (Shi & Eberhart 1998) :

    v <- w v + c1 r1 (p_best - x) + c2 r2 (g_best - x)
    x <- x + v

avec bornes reflechissantes, vitesse limitee a v_max fois l'etendue de chaque
intervalle, et initialisation par hypercube latin (meilleure couverture qu'un
tirage uniforme a effectif egal). Les MEMES reglages et les MEMES graines
servent aux deux correcteurs.

Le vecteur de decision est NORMALISE dans [0, 1]^n : chaque parametre est
ainsi explore a la meme echelle, quels que soient ses ordres de grandeur
(gains en log10, ordres fractionnaires en lineaire). C'est indispensable a
l'equite : sans cela, le correcteur ayant les bornes les plus larges serait
desavantage.
"""
import numpy as np

import config as C
from fopid import fopid_ss, rolloff_ss, series
from adrc import adrc_fopid_ss, b0_nominal


# ---------------------------------------------------------------------------
class Design:
    """Fabrique de correcteurs : vecteur normalise -> (A, B, C, D).

    CONVENTION DE SIGNE DE LA BOUCLE. Avec la geometrie de pastille imposee par
    la Fig. 12(b), les residus D_obs(i) H_Pe(i) ne sont PAS tous de meme signe
    ([-1 -1 -1 +1 +1]) : le gain du procede est NEGATIF en basse frequence et
    POSITIF en haute frequence. Aucune convention de signe unique n'est donc
    "la bonne" a priori — ni pour le FOPID (ou elle porte sur sign_loop), ni
    pour l'ADRC-FOPID (ou elle est portee par le signe de b0).

    On ne tranche donc pas a la main : `sign_variant` vaut +1 ou -1 et les DEUX
    valeurs sont explorees, pour LES DEUX structures, avec les memes graines et
    le meme budget. Ce n'est pas un parametre ajuste de plus (il reste 5 et 7
    parametres continus) : c'est une convention structurelle enumeree de facon
    exhaustive et identique des deux cotes.
    """

    def __init__(self, kind, plate, sign_loop, sign_variant=1.0):
        self.kind = kind
        self.plate = plate
        self.sign_loop = sign_loop
        self.sign_variant = float(sign_variant)
        self.b0_nom = b0_nominal(plate, C.N_MODES_DESIGN)
        bd = C.BOUNDS_FOPID if kind == 'fopid' else C.BOUNDS_ADRC
        self.names = list(bd.keys())
        self.lo = np.array([bd[k][0] for k in self.names], float)
        self.hi = np.array([bd[k][1] for k in self.names], float)
        self.n = len(self.names)

    def decode(self, u):
        """[0,1]^n -> dictionnaire de parametres physiques."""
        v = self.lo + np.clip(np.asarray(u, float), 0.0, 1.0) * (self.hi - self.lo)
        p = dict(zip(self.names, v))
        out = dict(Kp=10.0 ** p['log_Kp'], Ki=10.0 ** p['log_Ki'],
                   Kd=10.0 ** p['log_Kd'], lam=p['lam'], mu=p['mu'])
        if self.kind == 'adrc':
            out['wo'] = 10.0 ** p['log_wo']
            out['b0'] = p['b0_scale'] * self.b0_nom
        return out

    def build(self, u):
        p = self.decode(u)
        if self.kind == 'fopid':
            core = fopid_ss(p['Kp'], p['Ki'], p['Kd'], p['lam'], p['mu'],
                            C.OUST_WB, C.OUST_WH, C.OUST_N,
                            self.sign_loop * self.sign_variant)
        else:
            core = adrc_fopid_ss(p['Kp'], p['Ki'], p['Kd'], p['lam'], p['mu'],
                                 p['wo'], p['b0'] * self.sign_variant,
                                 C.OUST_WB, C.OUST_WH, C.OUST_N, 1.0)
        return series(core, rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))

    def order(self, u):
        return self.build(u)[0].shape[0]


# ---------------------------------------------------------------------------
def latin_hypercube(n_pts, n_dim, rng):
    u = np.empty((n_pts, n_dim))
    for j in range(n_dim):
        u[:, j] = (rng.permutation(n_pts) + rng.random(n_pts)) / n_pts
    return u


def pso(fitness, n_dim, seed=0, n_particles=None, n_iter=None, w=None,
        c1=None, c2=None, v_max=None, verbose=False, callback=None):
    """Maximise `fitness` sur [0, 1]^n_dim. Retourne (x*, f*, historique)."""
    cfg = dict(C.PSO)
    if n_particles is None:
        n_particles = cfg['n_particles']
    if n_particles is None:                     # taille liee a la dimension
        n_particles = int(cfg['n_particles_base']
                          + cfg['n_particles_per_dim'] * n_dim)
    n_iter = cfg['n_iter'] if n_iter is None else n_iter
    w = cfg['w'] if w is None else w
    c1 = cfg['c1'] if c1 is None else c1
    c2 = cfg['c2'] if c2 is None else c2
    v_max = cfg['v_max'] if v_max is None else v_max

    rng = np.random.default_rng(seed)
    x = latin_hypercube(n_particles, n_dim, rng)
    v = (rng.random((n_particles, n_dim)) - 0.5) * v_max
    f = np.array([fitness(xi) for xi in x])
    p_best, p_val = x.copy(), f.copy()
    g = int(np.argmax(p_val))
    g_best, g_val = p_best[g].copy(), float(p_val[g])
    hist = [g_val]
    n_eval = n_particles

    for it in range(n_iter):
        r1 = rng.random((n_particles, n_dim))
        r2 = rng.random((n_particles, n_dim))
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        v = np.clip(v, -v_max, v_max)
        x = x + v
        # bornes reflechissantes
        below, above = x < 0.0, x > 1.0
        x[below] = -x[below]
        v[below] *= -0.5
        x[above] = 2.0 - x[above]
        v[above] *= -0.5
        x = np.clip(x, 0.0, 1.0)
        f = np.array([fitness(xi) for xi in x])
        n_eval += n_particles
        imp = f > p_val
        p_best[imp], p_val[imp] = x[imp], f[imp]
        g = int(np.argmax(p_val))
        if p_val[g] > g_val:
            g_best, g_val = p_best[g].copy(), float(p_val[g])
        hist.append(g_val)
        if verbose:
            print(f"    [it {it + 1:2d}] meilleur J = {g_val:+.4f}", flush=True)
        if callback is not None:
            callback(it, g_val, g_best)
    return g_best, g_val, dict(history=np.array(hist), n_eval=n_eval)
