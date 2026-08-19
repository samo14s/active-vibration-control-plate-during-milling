"""
run_pso.py — Etape 1 : optimisation PSO des deux correcteurs
=============================================================
Optimise, dans des conditions RIGOUREUSEMENT IDENTIQUES, le FOPID (5
parametres) et l'ADRC-FOPID (7 parametres) pour maximiser les limites de
stabilite du fraisage a la vitesse de synthese.

Protocole d'equite applique ici :
  * meme plaque, meme pastille, meme capteur, meme convention de signe ;
  * meme realisation d'ordre fractionnaire (Oustaloup, meme bande, meme N) ;
  * meme lissage anti-repliement ;
  * meme fonction objectif et memes contraintes (objective.evaluate) ;
  * meme PSO, memes graines, meme budget d'evaluations ;
  * les deux vecteurs de decision sont normalises dans [0, 1]^n.

    python run_pso.py
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
from pso import Design, pso
from objective import evaluate

OUT = os.path.join(HERE, '..', 'results')
os.makedirs(OUT, exist_ok=True)


def main():
    t00 = time.time()
    print("=" * 74)
    print(" ETAPE 1 — OPTIMISATION PSO : FOPID contre ADRC-FOPID")
    print("=" * 74)
    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    _, _, H, D_obs, sign_loop = plant_vectors(plate, C.N_MODES_DESIGN)
    print(f"  protocole {C.PROTOCOL} ; calage {C.CALIB} :"
          f" f = {np.round(plate.freq_n, 1)} Hz")
    print(f"  pastille {C.PATCH_SIDE} ; D_obs.H = {float(D_obs @ H):+.3f}"
          f" -> sign_loop = {sign_loop:+.0f}")
    print(f"  contraintes : Ms <= {C.MS_MAX}, effort <= {C.V_PER_N:.0f} V/N")
    print(f"  notation (objectif + contraintes) sur {C.N_MODES_OBJ} modes ;"
          f" evaluation finale sur {C.N_MODES} modes")
    print(f"  objectif : marge de Floquet aux profondeurs"
          f" {[f'{a * 1e3:.1f}' for a in C.AP_PROBE]} mm,"
          f" positions {C.POSITIONS_DESIGN}")
    print(f"  PSO : {C.PSO['n_particles']} particules x {C.PSO['n_iter']}"
          f" iterations, graines {C.PSO['seeds']}\n")

    store = {}
    for kind in ('fopid', 'adrc'):
        D0 = Design(kind, plate, sign_loop)
        n_states = D0.order(np.full(D0.n, 0.5))
        print(f"  --- {kind.upper()} : {D0.n} parametres,"
              f" {n_states} etats ---", flush=True)
        best_x, best_J, best_var, runs = None, -np.inf, None, []
        # les deux conventions de signe de boucle sont explorees pour LES DEUX
        # structures (le procede change de signe entre basse et haute
        # frequence : aucune n'est correcte a priori)
        for variant in (+1.0, -1.0):
            D = Design(kind, plate, sign_loop, sign_variant=variant)
            for seed in C.PSO['seeds']:
                t0 = time.time()
                fit = lambda u: evaluate(plate, D.build(u))
                x, J, inf = pso(fit, D.n, seed=seed)
                runs.append(dict(seed=seed, variant=variant, x=x, J=J,
                                 history=inf['history'],
                                 n_eval=inf['n_eval']))
                print(f"    signe {variant:+.0f}, graine {seed} :"
                      f" J = {J:+.4f}"
                      f"  ({inf['n_eval']} evaluations,"
                      f" {time.time() - t0:.0f} s)", flush=True)
                if J > best_J:
                    best_J, best_x, best_var = J, x.copy(), variant
        D = Design(kind, plate, sign_loop, sign_variant=best_var)
        print(f"    convention retenue : sign_variant = {best_var:+.0f}")
        par = D.decode(best_x)
        _, info = evaluate(plate, D.build(best_x), detail=True)
        print(f"    meilleur : J = {best_J:+.4f}   Ms = {info['Ms']:.2f}"
              f"   effort = {info['V']:.0f} V/N")
        print("    parametres : " + "  ".join(
            f"{k}={v:.4g}" for k, v in par.items()), flush=True)
        ss = D.build(best_x)
        store[kind] = dict(
            x=best_x, J=best_J, n_par=D.n, n_states=n_states,
            sign_variant=best_var,
            names=np.array(D.names), values=np.array([par[k] for k in par]),
            keys=np.array(list(par.keys())),
            Ms=info['Ms'], V=info['V'],
            A=ss[0], B=ss[1], C=ss[2], D=ss[3],
            hist=np.array([r['history'] for r in runs]),
            seeds=np.array([r['seed'] for r in runs]),
            variants=np.array([r['variant'] for r in runs]),
            J_seeds=np.array([r['J'] for r in runs]),
            n_eval=int(sum(r['n_eval'] for r in runs)))

    print(f"\n  budget identique : "
          f"{store['fopid']['n_eval']} evaluations chacun")
    np.savez_compressed(os.path.join(OUT, f'pso_{C.PROTOCOL}.npz'),
                        **{f'{k}__{kk}': vv for k, v in store.items()
                           for kk, vv in v.items()})
    print(f"  -> results/pso_{C.PROTOCOL}.npz   ({time.time() - t00:.0f} s)")


if __name__ == '__main__':
    main()
