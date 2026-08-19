"""
run_adaptive.py — l'observateur SUPERVISE, mesure au meme etalon que les autres
===============================================================================
Le superviseur rend le correcteur non lineaire et variant : ni Floquet ni les
marges frequentielles ne s'appliquent. On ne peut donc PAS comparer son
resultat aux chiffres obtenus jusqu'ici. Ce script definit un etalon qui vaut
pour toutes les structures, et il le passe a TOUTES — y compris aux trois LTI,
dont les chiffres sont ainsi recalcules et non repris :

    a_p,lim TEMPOREL = plus grande profondeur pour laquelle la simulation ne
    diverge a AUCUNE des positions d'outil du protocole,
    avec la meme saturation +/-150 V, le meme pas, le meme modele a 5 modes.

C'est la transposition exacte du protocole de Floquet (min sur les positions)
dans le domaine temporel, ou tout le monde est mesurable.

Cinq configurations :
    fopid, adrc, fdob            les trois LTI, correcteurs stockes
    fdob_fixe                    le MEME observateur, mais realise par la
                                 classe adaptative avec adaptation et
                                 supervision DESACTIVEES
    fdob_supervise               adaptation + supervision actives

La quatrieme n'est pas une redondance : elle isole l'effet du superviseur de
l'effet de la realisation. La realisation bloc-par-bloc (celle d'un
calculateur reel) et la discretisation d'un seul tenant du correcteur LTI ne
coincident pas exactement — 1.5 % d'ecart mesure sur u, parce que la boucle
interne alpha.V(s).u est dans la matrice d'etat continue dans un cas et
fermee en discret dans l'autre. Comparer `fdob_supervise` a `fdob_fixe`
attribue donc la difference au SUPERVISEUR SEUL.

Deux plaques :
    nominale       les frequences mesurees du Tableau 4
    derivee        +17 % / +9 % sur les modes 1-2, la derive que la Section 5
                   du papier constate reellement — le cas ou l'observateur a
                   bandes fixes s'effondrait de 0.222 a 0.117 mm

    PROTOCOL=B CALIB=measured python run_adaptive.py
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
from fdob import target_modes
from fdob_adaptive import AdaptiveFDOB

OUT = os.path.join(HERE, '..', 'results')
T_SIM = 0.5              # duree de passe simulee a chaque position [s]
AP_HI = 0.60e-3          # borne haute de la dichotomie
RTOL = 0.03              # tolerance relative de la dichotomie


def survives(plate, make_ctrl, ap, positions):
    """La passe tient-elle a TOUTES les positions d'outil ?"""
    for fr in positions:
        sim = MillingSimulation(plate, C.RPM_DESIGN, ap, n_modes=C.N_MODES,
                                n_sub=C.N_SUB, sign=C.SIGN_SIM,
                                v_max=C.V_MAX)
        ctrl = make_ctrl(sim)
        r = sim.run(controller=ctrl, T=T_SIM, moving=False,
                    x0=fr * plate.lp)
        if bool(r['diverged']):
            return False
    return True


def ap_limit(plate, make_ctrl, positions, lo=0.0, hi=AP_HI):
    """Dichotomie relative sur la profondeur."""
    if not survives(plate, make_ctrl, max(lo, 1e-5), positions):
        return 0.0
    if survives(plate, make_ctrl, hi, positions):
        return hi
    while hi - lo > RTOL * hi:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if survives(plate, make_ctrl, mid,
                                       positions) else (lo, mid)
    return lo


def main():
    t00 = time.time()
    d = np.load(os.path.join(OUT, f'pso_{C.PROTOCOL}.npz'), allow_pickle=True)
    par = dict(zip([str(x) for x in d['fdob__keys']],
                   [float(v) for v in d['fdob__values']]))
    tg = [int(t) for t in d['fdob__targets']]
    var = float(d['fdob__sign_variant'])
    f_tooth = 3.0 * C.RPM_DESIGN / 60.0
    positions = C.POSITIONS_DESIGN

    print('=' * 78)
    print(' L OBSERVATEUR SUPERVISE — etalon temporel commun a toutes les'
          ' structures')
    print('=' * 78)
    print(f'  critere : passe de {T_SIM:.2f} s a chaque position'
          f' {positions}, saturation +/-{C.V_MAX:.0f} V,'
          f' dichotomie relative a {RTOL * 100:.0f} %')
    print(f'  superviseur : seuil de porte 0.25, niveau haut 0.60,'
          f' alpha_max = {par["alpha"]:.4f}, zeta_q = {par["zeta_q"]:.5f}')
    print(f'  f_dent = {f_tooth:.1f} Hz ; harmoniques rejetees par le peigne')

    plates = [('nominale', C.F_NOMINAL),
              ('derivee +17/+9 %', [C.F_NOMINAL[0] * 1.17,
                                    C.F_NOMINAL[1] * 1.09] +
               list(C.F_NOMINAL[2:]))]
    res = {}
    for pname, freqs in plates:
        plate = build_plate(C.PATCH_SIDE, freqs=freqs)
        _, _, _, _, sl = plant_vectors(plate, C.N_MODES)
        # L'observateur garde ses frequences NOMINALES : c'est tout l'enjeu.
        w0, z0, r0 = target_modes(build_plate(C.PATCH_SIDE,
                                              freqs=C.F_NOMINAL), tg)
        print(f'\n  --- plaque {pname} : f = {np.round(plate.freq_n, 1)} Hz')

        def mk_adapt(sim, adapt, supervise, alpha_max=None, mode='global',
                     floor=None):
            p = dict(par)
            if alpha_max is not None:
                p['alpha'] = alpha_max
            return AdaptiveFDOB(p, w0, z0, r0, C.FDOB_WC, sim.dt, f_tooth,
                                sl * var, C.OUST_WB, C.OUST_WH, C.OUST_N,
                                C.ROLLOFF_HZ, C.ROLLOFF_ORDER,
                                adapt=adapt, supervise=supervise, mode=mode,
                                alpha_floor=floor)

        cfgs = [('boucle ouverte', lambda s: None)]
        for k in ('fopid', 'adrc', 'fdob'):
            ss = tuple(d[f'{k}__{c}'] for c in 'ABCD')
            cfgs.append((k, (lambda ss_: lambda s: LTIController(ss_, s.dt))(ss)))
        cfgs.append(('fdob_fixe', lambda s: mk_adapt(s, False, False)))
        cfgs.append(('fdob_sup', lambda s: mk_adapt(s, True, True)))
        # alpha_max releve. Ce n'est PAS un parametre de plus : alpha etait
        # borne a [0, 0.9] des l'optimisation, et l'optimiseur a retenu 0.33
        # parce qu'a bandes FIXES alpha est permanent et que Ms <= 2 le
        # plafonne (Ms gele vaut 1.96 a alpha = 0.33 et 5.30 a 0.85). Sous
        # supervision alpha vaut ZERO tant qu'il n'y a pas de broutement :
        # la contrainte de marge au repos ne le plafonne donc plus, et on
        # mesure ce que vaut cette liberte. Le prix — Ms = 5.30 dans le
        # regime engage — est rapporte tel quel, pas dissimule.
        cfgs.append(('fdob_sup_haut',
                     lambda s: mk_adapt(s, True, True, alpha_max=0.85)))
        # alpha PAR BANDE, lie a la CONFIANCE de l'estimee de cette bande, en
        # partant du plancher valide par le PSO. Voir fdob_adaptive.py : sans
        # plancher cette loi coupe l'action preventive et fait diverger la
        # plaque nominale.
        cfgs.append(('fdob_sup_conf',
                     lambda s: mk_adapt(s, True, True, alpha_max=0.85,
                                        mode='bande',
                                        floor=float(par['alpha']))))

        for name, mk in cfgs:
            t0 = time.time()
            L = ap_limit(plate, mk, positions)
            res[(pname, name)] = L
            print(f'    {name:16s} : a_p,lim = {L * 1e3:.4f} mm'
                  f'   ({time.time() - t0:.0f} s)', flush=True)

    print('\n' + '=' * 78)
    print('  RECAPITULATIF (a_p,lim temporel, mm)')
    print('=' * 78)
    names = [c[0] for c in cfgs]
    print('  structure         ' + '  '.join(f'{p:>18s}' for p, _ in plates)
          + '     rapport')
    for name in names:
        a = res[(plates[0][0], name)] * 1e3
        b = res[(plates[1][0], name)] * 1e3
        print(f'  {name:16s} {a:18.4f} {b:18.4f} {b / max(a, 1e-9):11.2f}')

    np.savez_compressed(os.path.join(OUT, f'adaptive_{C.PROTOCOL}.npz'),
                        names=np.array(names),
                        plates=np.array([p for p, _ in plates]),
                        limits=np.array([[res[(p, n)] for n in names]
                                         for p, _ in plates]))
    print(f'\n  -> results/adaptive_{C.PROTOCOL}.npz'
          f'   ({time.time() - t00:.0f} s)')


if __name__ == '__main__':
    main()
