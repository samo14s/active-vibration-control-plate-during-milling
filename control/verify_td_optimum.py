"""
verify_td_optimum.py — l'optimum choisi par J tient-il sur la VRAIE profondeur ?
================================================================================
J n'est pas a_p,lim : c'est le premier croisement de max_x log rho(a_p) par
zero, interpole sur trois sondes. Mesure sur quatre points de la structure a
retard, le rapport J/vrai reste dans 1.02-1.16 et le CLASSEMENT est conserve —
mais l'ECART, lui, ne l'est pas : entre mu recule et mu a plein gain, J annonce
+0.047 mm la ou la bissection n'en trouve que +0.011. J exagere donc d'un
facteur quatre l'interet de reculer le correcteur robuste, et l'essaim suit J.

Le chiffre publie vient de la bissection, donc il reste juste quoi qu'il
arrive. Ce qui peut deriver, c'est le POINT DE CONCEPTION retenu : l'essaim
peut s'arreter un peu trop loin dans le recul. Ce script reprend les meilleurs
x de CHAQUE graine — ils sont stockes pour exactement ce genre de verification
— et les note sur la vraie profondeur, au pire poste, m = 200. Si un autre que
le champion de J gagne, c'est lui qu'il faut publier, et le dire.

    PROTOCOL=B CALIB=measured python control/verify_td_optimum.py
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'), HERE]

import config as C                                            # noqa: E402
from plate_model import build_plate                           # noqa: E402
from closed_loop import limit as cl_limit                     # noqa: E402
from pso import Design                                        # noqa: E402
from objective import evaluate                                # noqa: E402

OUT = os.path.join(HERE, '..', 'results')
KIND = os.environ.get('KIND', 'musyn_td')


def worst_true(plate, ss, pd):
    """a_p,lim au pire poste, bissection pleine resolution."""
    return min(cl_limit(plate, C.RPM_DESIGN, x * plate.lp, ctrl=ss, pd=pd,
                        n_modes=C.N_MODES, m=200, hi=6e-3)
               for x in C.POSITIONS_DESIGN)


def main():
    path = os.path.join(OUT, f'pso_{C.PROTOCOL}_{KIND}.npz')
    if not os.path.exists(path):
        path = os.path.join(OUT, f'pso_{C.PROTOCOL}.npz')
    d = np.load(path, allow_pickle=True)
    if f'{KIND}__xs' not in d.files:
        print(f'  {KIND} absent de {os.path.basename(path)}')
        return 1
    xs = np.atleast_2d(np.asarray(d[f'{KIND}__xs'], float))
    Js = np.asarray(d[f'{KIND}__J_seeds'], float).ravel()
    seeds = np.asarray(d[f'{KIND}__seeds']).ravel()
    var = float(d[f'{KIND}__sign_variant'])
    x_best = np.asarray(d[f'{KIND}__x'], float)
    J_best = float(d[f'{KIND}__J'])

    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    D = Design(KIND, plate, -1.0, sign_variant=var)

    print('=' * 78)
    print(f' {KIND} — le champion de J tient-il sur la VRAIE profondeur ?')
    print('=' * 78)
    print('  a_p,lim au pire poste, 5 modes, m = 200, bissection relative\n')
    print(f'{"graine":>8s} {"J [mm]":>9s} {"vrai [mm]":>10s} {"J/vrai":>8s}'
          f'  {"K_Pp":>11s} {"K_Pd":>8s}')

    lignes = list(zip([str(s) for s in seeds], xs, Js))
    lignes.append(('CHAMPION', x_best, J_best))
    res = []
    for tag, x, J in lignes:
        t0 = time.time()
        ss = D.build(x)
        pd = D.delay_gains(x)
        if ss is None:
            print(f'{tag:>8s}  synthese impossible')
            continue
        Jv, info = evaluate(plate, ss, C.RPM_DESIGN, detail=True, pd=pd)
        v = worst_true(plate, ss, pd)
        r = (J / (v * 1e3)) if v > 0 else float('nan')
        res.append((tag, J, v))
        print(f'{tag:>8s} {J:9.4f} {v * 1e3:10.4f} {r:8.3f}  '
              f'{pd[0]:11.4g} {pd[1]:8.3f}'
              f'   Ms={info["Ms"]:.3f}  ({time.time() - t0:.0f} s)',
              flush=True)

    if not res:
        return 1
    par_J = max(res, key=lambda t: t[1])
    par_vrai = max(res, key=lambda t: t[2])
    print()
    if par_J[0] == par_vrai[0] or abs(par_J[2] - par_vrai[2]) < 1e-6:
        print(f'  le champion de J ({par_J[0]}) est aussi le meilleur sur la'
              f' vraie profondeur : {par_vrai[2] * 1e3:.4f} mm')
    else:
        print(f'  DIVERGENCE : J designe {par_J[0]}'
              f' ({par_J[2] * 1e3:.4f} mm vrai), mais {par_vrai[0]} fait mieux'
              f' ({par_vrai[2] * 1e3:.4f} mm).')
        print('  -> publier le second, et le dire dans le rapport.')
    np.savez_compressed(
        os.path.join(OUT, f'verify_td_{C.PROTOCOL}_{KIND}.npz'),
        tags=np.array([t for t, _, _ in res]),
        J=np.array([j for _, j, _ in res]),
        vrai=np.array([v for _, _, v in res]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
