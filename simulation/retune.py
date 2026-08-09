"""
retune.py — re-optimise les correcteurs sur la base CORRIGEE.

Pourquoi. Les reglages X_LQG / X_ESO de run_demo.py avaient ete obtenus sur une
geometrie de patch fausse (horizontale, coin inferieur droit). Depuis, la
geometrie a ete corrigee (verticale, coin inferieur gauche, confirmee par la
Fig. 11 de l'article), la couche de colle et le couplage membrane-flexion ont
ete modelises, et l'autorite actionneur a baisse d'environ 21 % au total.

Puis, plus radicalement, la REPARTITION MODALE du couplage a ete identifiee sur
la Fig. 12(b) (constante H_IDENT) : elle etait structurellement fausse, le
modele elements finis ne produisant qu'un seul des quatre creux mesures. Un
correcteur regle sur l'ancienne repartition n'est plus regle du tout — le
reglage X_LQG d'origine SATURE meme l'amplificateur a toutes les vitesses sur la
base corrigee, donc il echoue d'emblee la contrainte ci-dessous.

Les comparer en l'etat mesure ce que CES reglages-la donnent sur un autre
modele, pas ce que chaque architecture peut atteindre : la comparaison
d'architectures n'a de sens qu'apres re-optimisation de chacune sur la meme
base.

Critere. On maximise la LIMITE DE PASSE DU PIRE CAS sur les cinq vitesses,
exactement la grandeur que run_demo publie :

    J(x) = min_rpm  SimBase.stability_limit(controleur(x), rpm)

SOUS CONTRAINTE de non-saturation au point de fonctionnement nominal.

Pourquoi la contrainte. Maximiser J seul est un critere MAL POSE, et on l'a
verifie : l'optimum sans contrainte est un correcteur en butee permanente
(peak_u = 150.00 V a toutes les vitesses, 120 V RMS contre 5 V pour le reglage
d'origine) qui gagne 34 % de limite tout en MULTIPLIANT PAR 3.5 l'amplitude
vibratoire reelle (2.07 um contre 0.59 um). Le critere de stabilite de
la base est un critere de CROISSANCE : une reponse grande mais bornee le
satisfait. Un correcteur constamment sur la butee n'est plus lineaire, et la
"limite de stabilite" qu'il affiche est un artefact.

On impose donc, au point nominal (ap = AP_TEST, horizon T_RUN) et a chaque
vitesse : coupe stable ET peak_u < V_MAX. La contrainte est physique -- le banc
ne delivre pas plus de +/- 150 V -- et elle est evaluee EN PREMIER, car elle
coute 5 simulations courtes la ou la bissection en coute 40 a 55.

--constraint none restitue le critere sans contrainte, pour reproduire le
resultat degenere ci-dessus.

Methode. PSO, meme famille que l'optimisation d'origine. Le reglage courant est
injecte comme particule 0, ce qui garantit de ne pas faire pire que l'existant.
Les particules sont evaluees en parallele (une par coeur).

Usage :
    python retune.py lqg            # ~7 min
    python retune.py eso            # ~15 min
    python retune.py both
"""
import argparse
import copy
import multiprocessing as mp
import sys
import time

import numpy as np

sys.path.insert(0, "sim_kit")
sys.path.insert(0, ".")
import simulation_base as SB                                    # noqa: E402
import run_demo as RD                                           # noqa: E402
from model_v2 import make_sim                                   # noqa: E402

SPEEDS = RD.SPEEDS
TOL_SEARCH = 5e-5      # 0.05 mm : suffisant pour classer des candidats
TOL_FINAL = 2e-6       # 0.002 mm : pour la valeur publiee

# (nom, bornes) — bornes centrees genereusement sur le reglage actuel
BOUNDS = {
    'lqg': np.array([(-12.0, -4.0),     # log10_rho
                     (-2.0, 5.0),       # log10_qw
                     (-12.0, -4.0)]),   # log10_R
    'eso': np.array([(-12.0, -4.0),     # log10_rho
                     (-4.0, 4.0),       # log10_qw
                     (-4.0, 6.0),       # log10_sigd
                     (-12.0, -4.0),     # log10_R
                     (0.0, 2.0),        # beta
                     (0.0, 1.0e6),      # g_reg
                     (1.0, 4.0)]),      # log10_fhp
}
SEED = {'lqg': np.array(RD.X_LQG, float), 'eso': np.array(RD.X_ESO, float)}
FACTORY = {'lqg': RD.factory_lqg, 'eso': RD.factory_eso}

_SIM = None
_PLATE = None


def _init_worker():
    global _SIM, _PLATE
    _SIM = make_sim()
    _PLATE = copy.deepcopy(_SIM.plate)


def _worst_blim(arch, x, tol, cutoff=0.0, gate=True):
    """Limite de passe du pire cas, en mm. 0.0 si le correcteur echoue.

    cutoff : meilleure valeur connue. Des qu'une vitesse fait tomber le minimum
    courant en dessous, le candidat ne peut plus gagner et on s'arrete ; la
    valeur rendue est alors une BORNE SUPERIEURE, ce qui suffit au classement
    et divise le cout par ~3, la plupart des particules perdant.
    """
    try:
        mk = FACTORY[arch](_PLATE, list(x))
    except Exception:
        return 0.0

    if gate:
        # contrainte physique, evaluee d'abord car bien moins chere
        for rpm in SPEEDS:
            tau = 60.0/(3*rpm)
            try:
                r = _SIM.run(mk(tau/82, tau), rpm=rpm, ap=SB.AP_TEST,
                             T=SB.T_RUN)
            except Exception:
                return 0.0
            if (not r['stable']) or r['peak_u'] >= 0.999*SB.V_MAX:
                return 0.0

    best = np.inf
    for rpm in SPEEDS:
        try:
            v = _SIM.stability_limit(mk, rpm=rpm, lo=0.02e-3, hi=1.5e-3,
                                     T=SB.T_LIMIT, tol=tol)*1e3
        except Exception:
            return 0.0
        if v < best:
            best = v
        if best <= max(cutoff, 0.021):
            return best
    return float(best)


def _eval(job):
    arch, x, tol, cutoff, gate = job
    return _worst_blim(arch, np.asarray(x), tol, cutoff, gate)


def pso(arch, n_part, n_iter, workers, rng, gate=True):
    lo, hi = BOUNDS[arch][:, 0], BOUNDS[arch][:, 1]
    d = len(lo)
    X = rng.uniform(lo, hi, size=(n_part, d))
    X[0] = np.clip(SEED[arch], lo, hi)          # reglage actuel comme graine
    V = rng.uniform(-1, 1, size=(n_part, d))*(hi - lo)*0.1
    with mp.Pool(workers, initializer=_init_worker) as pool:
        f = np.array(pool.map(
            _eval, [(arch, x, TOL_SEARCH, 0.0, gate) for x in X]))
        pbest, pbest_f = X.copy(), f.copy()
        g = int(np.argmax(f))
        gbest, gbest_f = X[g].copy(), f[g]
        print(f'  init   : meilleur = {gbest_f:.4f} mm '
              f'(graine = {f[0]:.4f} mm)', flush=True)
        w, c1, c2 = 0.72, 1.5, 1.5
        for it in range(n_iter):
            r1 = rng.random((n_part, d))
            r2 = rng.random((n_part, d))
            V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)
            V = np.clip(V, -(hi - lo)*0.25, (hi - lo)*0.25)
            X = np.clip(X + V, lo, hi)
            f = np.array(pool.map(
                _eval, [(arch, x, TOL_SEARCH, 0.97*gbest_f, gate)
                        for x in X]))
            imp = f > pbest_f
            pbest[imp], pbest_f[imp] = X[imp], f[imp]
            g = int(np.argmax(pbest_f))
            if pbest_f[g] > gbest_f:
                gbest, gbest_f = pbest[g].copy(), pbest_f[g]
            print(f'  iter {it+1:3d} : meilleur = {gbest_f:.4f} mm', flush=True)
    return gbest, gbest_f


def report(arch, x):
    """Evalue le reglage a la tolerance finale et rapporte la tension."""
    sim = make_sim()
    plate = copy.deepcopy(sim.plate)
    mk = FACTORY[arch](plate, list(x))
    lim = [sim.stability_limit(mk, rpm=r, lo=0.02e-3, hi=4.0e-3,
                               T=SB.T_LIMIT, tol=TOL_FINAL)*1e3 for r in SPEEDS]
    u, pk, y = [], [], []
    for r in SPEEDS:
        tau = 60.0/(3*r)
        res = sim.run(mk(tau/82, tau), rpm=r, ap=SB.AP_TEST, T=SB.T_RUN)
        u.append(res['rms_u']); pk.append(res['peak_u']); y.append(res['rms_um'])
    return lim, u, pk, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arch', choices=['lqg', 'eso', 'both'])
    ap.add_argument('--particles', type=int, default=0)
    ap.add_argument('--iters', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--constraint', choices=['sat', 'none'], default='sat')
    a = ap.parse_args()
    archs = ['lqg', 'eso'] if a.arch == 'both' else [a.arch]
    rng = np.random.default_rng(20240809)
    t0 = time.time()
    for arch in archs:
        npart = a.particles or (12 if arch == 'lqg' else 20)
        nit = a.iters or (15 if arch == 'lqg' else 20)
        print(f'=== {arch.upper()} : PSO {npart} particules x {nit} iterations '
              f'sur {a.workers} coeurs ===', flush=True)
        x, fx = pso(arch, npart, nit, a.workers, rng,
                    gate=(a.constraint == 'sat'))
        lim, u, pk, y = report(arch, x)
        print(f'\n  X_{arch.upper()} = [' + ', '.join(f'{v:.4f}' for v in x) + ']')
        print(f'  limites (mm)  : {[round(v, 4) for v in lim]}')
        print(f'  pire cas      : {min(lim):.4f} mm   (recherche : {fx:.4f})')
        print(f'  tension RMS V : {[round(v, 1) for v in u]}')
        print(f'  tension crete : {[round(v, 1) for v in pk]}   '
              f'(V_MAX = {SB.V_MAX})')
        print(f'  vibration um  : {[round(v, 3) for v in y]}')
        print(f'  ecoule        : {time.time()-t0:.0f} s\n', flush=True)


if __name__ == '__main__':
    main()
