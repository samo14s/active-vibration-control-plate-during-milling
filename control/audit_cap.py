"""
audit_cap.py — ce que vaut un optimum qui SATURE le plafond de l'objectif
=========================================================================
`objective._ap_from_margins` refuse de discriminer au-dela de trois fois la
sonde la plus profonde (2.1 mm ici) : au-dela, il rend le plafond. C'est une
position honnete — l'extrapolation d'une droite sur trois points ne vaut rien
si loin — et le depot la documente. Mais elle a une consequence qu'il faut
mesurer plutot que supposer : parmi tous les jeux de parametres qui atteignent
le plafond, le PSO garde LE PREMIER TROUVE. Le representant d'une structure
saturee est donc choisi arbitrairement dans un ensemble d'ex aequo.

La question est de savoir si cet arbitraire se voit. Ce script y repond en
mesurant, pour chaque structure qui sature :

  1. combien de jeux distincts atteignent le plafond, sur un echantillonnage
     LHS du meme boitier de recherche ;
  2. de combien leur VRAIE limite (bissection de Floquet, cinq modes, m = 200)
     s'ecarte les uns des autres.

Si l'ecart est faible, la saturation est inoffensive et il faut le dire. S'il
est large, le classement de cette structure depend d'un tirage et il faut le
dire aussi — c'est exactement le genre de dependance qu'une comparaison
« equitable » doit exhiber au lieu de la laisser dormir dans un plafond.

    PROTOCOL=B python audit_cap.py [n_echantillons]
"""
import glob
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
from pso import Design, latin_hypercube                       # noqa: E402
from objective import evaluate, limits                        # noqa: E402

OUT = os.path.join(HERE, '..', 'results')


def stored():
    found = {}
    merged = os.path.join(OUT, f'pso_{C.PROTOCOL}.npz')
    paths = ([merged] if os.path.exists(merged) else []) + sorted(
        glob.glob(os.path.join(OUT, f'pso_{C.PROTOCOL}_*.npz')))
    for path in paths:
        d = np.load(path, allow_pickle=True)
        for k in d.files:
            kind, _, field = k.partition('__')
            found.setdefault(kind, {}).setdefault(field, d[k])
    return found


def main():
    n_lhs = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    cap = 3.0 * C.AP_PROBE[-1]
    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    st = stored()
    print('=' * 78)
    print(f' STRUCTURES QUI SATURENT LE PLAFOND DE L OBJECTIF'
          f' ({cap * 1e3:.3f} mm)')
    print('=' * 78)
    sat = [k for k, v in st.items()
           if 'J' in v and float(v['J']) >= cap - 1e-9]
    if not sat:
        print('  aucune : le plafond n est atteint par personne, '
              'la question ne se pose pas')
        return 0
    print(f'  saturees : {", ".join(sat)}\n')
    for kind in sat:
        rec = st[kind]
        var = float(rec.get('sign_variant', 1.0))
        tg = rec.get('targets')
        base = 'fdob' if kind.startswith('fdob') else kind
        D = Design(base, plate, -1.0, sign_variant=var,
                   targets=None if tg is None else tuple(int(t) for t in tg))
        rng = np.random.default_rng(0)
        U = latin_hypercube(n_lhs, D.n, rng)
        t0 = time.time()
        tied = []
        for u in U:
            ss = D.build(u)
            if ss is None:
                continue
            J, info = evaluate(plate, ss, C.RPM_DESIGN, detail=True)
            if info['feasible'] and J >= cap - 1e-9:
                tied.append((u, ss))
        print(f'  --- {kind} : {len(tied)}/{n_lhs} jeux ex aequo au plafond'
              f'   ({time.time() - t0:.0f} s)', flush=True)
        if not tied:
            print('      (le plafond n est atteint nulle part dans le LHS ;'
                  ' l optimum y arrive par recherche, pas par hasard)')
            continue
        # ... et ce qu'ils valent VRAIMENT, la ou l objectif a renonce
        take = tied[:12]
        true = []
        for u, ss in take:
            L = limits(plate, ss, C.RPM_DESIGN, positions=C.POSITIONS_DESIGN)
            true.append(float(np.min(L)))
        true = np.array(true)
        ref = (rec['A'], rec['B'], rec['C'], rec['D'])
        Lr = float(np.min(limits(plate, ref, C.RPM_DESIGN,
                                 positions=C.POSITIONS_DESIGN)))
        print(f'      vraie limite au pire poste, {len(true)} ex aequo :'
              f' min {true.min() * 1e3:.4f}  median'
              f' {np.median(true) * 1e3:.4f}  max {true.max() * 1e3:.4f} mm')
        print(f'      etendue = {(true.max() - true.min()) * 1e3:.4f} mm'
              f'  ({(true.max() / max(true.min(), 1e-12) - 1) * 100:.0f} %)')
        print(f'      optimum retenu par le PSO : {Lr * 1e3:.4f} mm'
              f'  -> rang {int(np.sum(true > Lr)) + 1}/{len(true) + 1}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
