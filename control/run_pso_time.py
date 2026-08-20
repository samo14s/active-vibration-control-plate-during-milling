"""
run_pso_time.py — optimiser DIRECTEMENT sur le critere qui sert a comparer
==========================================================================
Objection legitime : le FOPID et l'observateur ont tous deux ete optimises sur
l'objectif de FLOQUET (a_p,lim au pire poste, m = 24), puis compares sur un
critere TEMPOREL (survie d'une passe). Aucun des deux n'est donc optimal pour
la regle qui les departage. Et l'observateur supervise a recu en plus un
alpha_max choisi A LA MAIN apres avoir regarde les resultats — un privilege
que le FOPID n'a pas eu.

Ce script enleve la premiere de ces deux inegalites du bon cote : il optimise
le FOPID sur le critere TEMPOREL lui-meme. Le resultat est donc GENEREUX pour
le FOPID — c'est voulu. Si l'observateur supervise, qui n'a jamais ete
optimise sur ce critere, tient quand meme devant un FOPID qui l'a ete, la
conclusion est solide dans le sens qui compte. Et s'il ne tient pas, c'est que
son avantage venait de la regle de mesure, ce qu'il faut alors dire.

L'objectif de recherche est une version ECONOMIQUE du critere final :
3 positions au lieu de 6, T = 0.15 s au lieu de 0.5, dichotomie a 10 % au lieu
de 3 %. C'est 6.8 s par evaluation contre 16.4 s pour le critere complet. Le
gagnant est ensuite repasse au critere COMPLET, celui de run_adaptive.py, donc
la comparaison finale n'utilise jamais la version economique.

Les contraintes du protocole restent identiques et sont evaluees AVANT le
temporel, qui coute mille fois plus cher : stabilite nominale, Ms <= 2, effort
<= 450 V/N.

    PROTOCOL=B CALIB=measured python run_pso_time.py
"""
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'), HERE]

import config as C
from plate_model import build_plate, plant_vectors
from simulate import MillingSimulation
from sim_controller import LTIController
from objective import nominal_poles, frequency_metrics
from pso import Design, pso

OUT = os.path.join(HERE, '..', 'results')
POS = (0.0, 0.5, 1.0)          # positions de RECHERCHE (le final en a 6)
T_SIM = 0.15
RTOL = 0.10
AP_LO, AP_HI = 0.05e-3, 0.60e-3
N_ITER = 12
SEEDS = (1, 2)


def survives(plate, ss, ap):
    for fr in POS:
        sim = MillingSimulation(plate, C.RPM_DESIGN, ap, n_modes=C.N_MODES,
                                n_sub=C.N_SUB, sign=C.SIGN_SIM,
                                v_max=C.V_MAX)
        r = sim.run(controller=LTIController(ss, sim.dt), T=T_SIM,
                    moving=False, x0=fr * plate.lp)
        if bool(r['diverged']):
            return False
    return True


def ap_time(plate, ss):
    lo, hi = AP_LO, AP_HI
    if not survives(plate, ss, lo):
        return 0.0
    while hi - lo > RTOL * hi:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if survives(plate, ss, mid) else (lo, mid)
    return lo


def make_fitness(plate, D):
    def f(u):
        ss = D.build(u)
        ev = nominal_poles(plate, ss)
        mre = float(np.max(ev.real))
        if not np.isfinite(mre) or mre > 0.0:
            return -1e3 - (mre if np.isfinite(mre) else 1e3)
        Ms, V = frequency_metrics(plate, ss, poles=ev)
        pen = 0.0
        if Ms > C.MS_MAX:
            pen += 10.0 * (Ms / C.MS_MAX - 1.0)
        if V > C.V_PER_N:
            pen += 10.0 * (V / C.V_PER_N - 1.0)
        if pen > 0.0:
            return -100.0 - pen
        return ap_time(plate, ss) * 1e3
    return f


def main():
    t00 = time.time()
    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    _, _, _, _, sign_loop = plant_vectors(plate, C.N_MODES_DESIGN)
    print('=' * 78)
    print(' FOPID OPTIMISE SUR LE CRITERE TEMPOREL (celui qui sert a comparer)')
    print('=' * 78)
    print(f'  recherche : {len(POS)} positions {POS}, T = {T_SIM} s,'
          f' dichotomie a {RTOL:.0%}')
    print(f'  contraintes : identiques au protocole'
          f' (Ms <= {C.MS_MAX}, effort <= {C.V_PER_N:.0f} V/N)')
    print(f'  PSO : {10 + 4 * 5} particules x {N_ITER} iterations,'
          f' graines {SEEDS}, deux conventions de signe\n')

    best = dict(J=-np.inf)
    runs = []
    for variant in (+1.0, -1.0):
        D = Design('fopid', plate, sign_loop, sign_variant=variant)
        fit = make_fitness(plate, D)
        for seed in SEEDS:
            t0 = time.time()
            x, J, inf = pso(fit, D.n, seed=seed, n_iter=N_ITER)
            runs.append(dict(seed=seed, variant=variant, x=x, J=J,
                             n_eval=inf['n_eval']))
            print(f'    signe {variant:+.0f} graine {seed} : J = {J:+.4f} mm'
                  f'  ({inf["n_eval"]} evaluations,'
                  f' {time.time() - t0:.0f} s)', flush=True)
            if J > best['J']:
                best = dict(J=J, x=x.copy(), variant=variant, D=D)

    D = best['D']
    par = D.decode(best['x'])
    ss = D.build(best['x'])
    ev = nominal_poles(plate, ss)
    Ms, V = frequency_metrics(plate, ss, poles=ev)
    n_eval = int(sum(r['n_eval'] for r in runs))
    print(f'\n  meilleur : J = {best["J"]:+.4f} mm (critere de recherche),'
          f' signe {best["variant"]:+.0f}')
    print('  parametres : ' + '  '.join(f'{k}={v:.4g}'
                                        for k, v in par.items()))
    print(f'  Ms = {Ms:.3f} / {C.MS_MAX}   effort = {V:.0f} /'
          f' {C.V_PER_N:.0f} V/N   pole lent = {ev.real.max():.2f} 1/s')
    print(f'  budget : {n_eval} evaluations temporelles')

    np.savez_compressed(os.path.join(OUT, f'pso_time_{C.PROTOCOL}.npz'),
                        x=best['x'], J=best['J'],
                        sign_variant=best['variant'],
                        names=np.array(D.names),
                        keys=np.array(list(par.keys())),
                        values=np.array([par[k] for k in par]),
                        Ms=Ms, V=V, n_eval=n_eval,
                        A=ss[0], B=ss[1], C=ss[2], D=ss[3],
                        J_seeds=np.array([r['J'] for r in runs]),
                        variants=np.array([r['variant'] for r in runs]),
                        seeds=np.array([r['seed'] for r in runs]))
    print(f'  -> results/pso_time_{C.PROTOCOL}.npz'
          f'   ({time.time() - t00:.0f} s)')


if __name__ == '__main__':
    main()
