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
from fdob import fdob_fopid_ss, target_modes
from hinf import HinfFailure, augment, bandpass_weight, plant_ss, synthesize
from musyn import augment_mu, dk_iterate
from classical import LqgFailure, dvf_ss, lqg_ss, vpa_ss
from nmp_dob import nmp_dob_fopid_ss
from plate_model import plant_vectors


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

    def __init__(self, kind, plate, sign_loop, sign_variant=1.0,
                 targets=None):
        self.kind = kind
        self.targets = tuple(C.FDOB_TARGETS if targets is None else targets)
        self.plate = plate
        self.sign_loop = sign_loop
        self.sign_variant = float(sign_variant)
        self.b0_nom = b0_nominal(plate, C.N_MODES_DESIGN)
        if kind == 'fdob':
            self.tw, self.tz, self.tr = target_modes(plate, self.targets)
        if kind in ('hinf', 'musyn', 'lqg'):
            # Ces trois-la se synthetisent SUR le modele : elles ont besoin du
            # procede lui-meme, pas seulement de son signe de boucle.
            w, zt, Hv, D_obs, _ = plant_vectors(plate, C.N_MODES_DESIGN)
            self.plant = plant_ss(w, zt, D_obs * Hv)
        bd = dict(fopid=C.BOUNDS_FOPID, adrc=C.BOUNDS_ADRC,
                  fdob=C.BOUNDS_FDOB, hinf=C.BOUNDS_HINF,
                  musyn=C.BOUNDS_MU, dvf=C.BOUNDS_DVF, vpa=C.BOUNDS_VPA,
                  lqg=C.BOUNDS_LQG, nmpdob=C.BOUNDS_NMPDOB)[kind]
        self.names = list(bd.keys())
        self.lo = np.array([bd[k][0] for k in self.names], float)
        self.hi = np.array([bd[k][1] for k in self.names], float)
        self.n = len(self.names)

    def decode(self, u):
        """[0,1]^n -> dictionnaire de parametres physiques."""
        v = self.lo + np.clip(np.asarray(u, float), 0.0, 1.0) * (self.hi - self.lo)
        p = dict(zip(self.names, v))
        if self.kind in ('hinf', 'musyn'):
            # Boitier de PONDERATIONS, pas de gains : ces structures n'ont
            # aucune cle Kp/Ki/Kd/lam/mu, donc on sort AVANT de construire le
            # dictionnaire commun — le construire d'abord leverait KeyError.
            return dict(kw=10.0 ** p['log_kw'], f_w=p['f_w'],
                        zw=10.0 ** p['log_zw'], w2=10.0 ** p['log_w2'],
                        eps=10.0 ** p['log_eps'])
        if self.kind == 'dvf':
            return dict(g=10.0 ** p['log_g'], f_d=p['f_d'])
        if self.kind == 'vpa':
            # SCALAIRES, pas des listes. `run_pso` imprime et stocke chaque
            # parametre par f"{v:.4g}" et np.array([...]) : une liste y leve
            # TypeError APRES l'optimisation, donc apres avoir depense le
            # budget et AVANT d'ecrire le fichier — la pire place possible.
            # C'est `build` qui les regroupe, pas `decode`.
            return dict(g1=10.0 ** p['log_g1'], f1=p['f_a1'],
                        z1=10.0 ** p['log_z1'],
                        g2=10.0 ** p['log_g2'], f2=p['f_a2'],
                        z2=10.0 ** p['log_z2'])
        if self.kind == 'lqg':
            return dict(q=10.0 ** p['log_q'], r=10.0 ** p['log_r'],
                        w_proc=10.0 ** p['log_w'], v_meas=10.0 ** p['log_v'],
                        f_w=p['f_w'])
        out = dict(Kp=10.0 ** p['log_Kp'], Ki=10.0 ** p['log_Ki'],
                   Kd=10.0 ** p['log_Kd'], lam=p['lam'], mu=p['mu'])
        if self.kind == 'adrc':
            out['wo'] = 10.0 ** p['log_wo']
            out['b0'] = p['b0_scale'] * self.b0_nom
        elif self.kind == 'fdob':
            out['zeta_q'] = 10.0 ** p['log_zq']
            out['alpha'] = p['alpha']
        elif self.kind == 'nmpdob':
            out['wq'] = 10.0 ** p['log_wq']
            out['alpha'] = p['alpha']
        return out

    def build(self, u):
        """Correcteur, ou None si la SYNTHESE elle-meme echoue.

        Le None n'est pas un detail d'implementation. Une synthese H-infini
        n'aboutit pas pour toute ponderation : il existe des reglages ou aucun
        correcteur n'atteint le gamma demande. C'est une propriete REELLE de
        la structure, et la compter comme un echec plutot que la contourner
        fait partie de l'equite — au meme titre qu'un FOPID nominalement
        instable est compte comme un echec. Le taux d'echec est rapporte.
        """
        p = self.decode(u)
        if self.kind == 'dvf':
            core = dvf_ss(p['g'], p['f_d'], self.sign_loop * self.sign_variant)
            return series(core, rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))
        if self.kind == 'vpa':
            core = vpa_ss([p['g1'], p['g2']], [p['f1'], p['f2']],
                          [p['z1'], p['z2']],
                          self.sign_loop * self.sign_variant)
            return series(core, rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))
        if self.kind == 'lqg':
            try:
                K = lqg_ss(self.plant, **p)
            except (LqgFailure, np.linalg.LinAlgError, ValueError,
                    FloatingPointError):
                return None
            core = (K[0], K[1], self.sign_variant * K[2], K[3])
            return series(core, rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))
        if self.kind in ('hinf', 'musyn'):
            W1 = bandpass_weight(p['kw'], p['f_w'], p['zw'])
            try:
                if self.kind == 'hinf':
                    Pg = augment(self.plant, W1, p['w2'], p['eps'])
                    K, _ = synthesize(Pg)
                else:
                    Pg = augment_mu(self.plant, W1, p['w2'], p['eps'])
                    K, _, _ = dk_iterate(Pg, n_dk=C.N_DK)
            except (HinfFailure, np.linalg.LinAlgError, ValueError,
                    FloatingPointError):
                return None
            core = (K[0], K[1], self.sign_variant * K[2], K[3])
            return series(core, rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))
        if self.kind == 'nmpdob':
            # Le procede vu par l'observateur est le modele de SYNTHESE complet :
            # la factorisation a besoin de tous les modes, c'est meme son objet.
            w, zt, Hv, D_obs, _ = plant_vectors(self.plate, C.N_MODES_DESIGN)
            core = nmp_dob_fopid_ss(p['Kp'], p['Ki'], p['Kd'], p['lam'],
                                    p['mu'], p['wq'], p['alpha'], w, zt,
                                    D_obs * Hv, C.OUST_WB, C.OUST_WH,
                                    C.OUST_N,
                                    self.sign_loop * self.sign_variant)
            return series(core, rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))
        if self.kind == 'fopid':
            core = fopid_ss(p['Kp'], p['Ki'], p['Kd'], p['lam'], p['mu'],
                            C.OUST_WB, C.OUST_WH, C.OUST_N,
                            self.sign_loop * self.sign_variant)
        elif self.kind == 'adrc':
            core = adrc_fopid_ss(p['Kp'], p['Ki'], p['Kd'], p['lam'], p['mu'],
                                 p['wo'], p['b0'] * self.sign_variant,
                                 C.OUST_WB, C.OUST_WH, C.OUST_N, 1.0)
        else:
            # Le signe de l'observateur est porte mode par mode par 1/r_k et
            # vient donc du modele ; sign_variant ne porte que sur l'epine
            # dorsale FOPID, exactement comme pour le FOPID seul.
            core = fdob_fopid_ss(p['Kp'], p['Ki'], p['Kd'], p['lam'], p['mu'],
                                 p['zeta_q'], p['alpha'], self.tw, self.tz,
                                 self.tr, C.FDOB_WC, C.OUST_WB, C.OUST_WH,
                                 C.OUST_N, self.sign_loop * self.sign_variant)
        return series(core, rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))

    def order(self, u):
        ss = self.build(u)
        return -1 if ss is None else ss[0].shape[0]


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
