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
import json
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


def _pack(D, best_x, best_J, best_var, runs, plate, n_states,
          hist=None, n_eval=None):
    """Le dictionnaire stocke pour une structure. Commun aux deux chemins
    (optimisation complete et ajout de graines), pour qu'ils ne puissent pas
    diverger."""
    par = D.decode(best_x)
    ss = D.build(best_x)
    _, info = evaluate(plate, ss, detail=True)
    hs = [r['history'] for r in runs if r.get('history') is not None] \
        if hist is None else hist
    return dict(
        targets=np.array(D.targets),
        x=best_x, J=best_J, n_par=D.n, n_states=n_states,
        sign_variant=best_var,
        names=np.array(D.names), values=np.array([par[k] for k in par]),
        keys=np.array(list(par.keys())),
        Ms=info['Ms'], V=info['V'],
        A=ss[0], B=ss[1], C=ss[2], D=ss[3],
        # x de CHAQUE graine : sans eux on ne peut pas mesurer la dispersion
        # sur la metrique FINALE (a_p,lim a m = 200), mais seulement sur J,
        # qui est l'estimation bruitee de l'optimiseur.
        xs=np.array([r['x'] for r in runs]),
        hist=np.array(hs),
        seeds=np.array([r['seed'] for r in runs]),
        variants=np.array([r['variant'] for r in runs]),
        J_seeds=np.array([r['J'] for r in runs]),
        n_eval=int(sum(r['n_eval'] for r in runs)) if n_eval is None
        else n_eval)


def _ckpt_path(kind, tag):
    """Ou vivent les tirages deja faits pour cette structure."""
    return os.path.join(OUT, f'ckpt_{C.PROTOCOL}{tag}_{kind}.json')


def _ckpt_load(kind, tag):
    """Tirages deja termines, releus depuis le disque.

    POURQUOI. `run_pso` n'ecrivait son .npz qu'a la toute fin, apres les cinq
    tirages. Sur cette machine les conteneurs redemarrent toutes les quelques
    heures : le FDOB a cinq modes a perdu DIX HEURES de calcul — depistage
    plus deux raffinements, tous mesures et affiches dans son journal — parce
    que le cinquieme tirage n'avait pas fini. Rien n'etait faux, tout etait
    perdu.

    Un tirage PSO est deterministe a (structure, convention, graine) donnes :
    il est donc licite de le relire au lieu de le refaire, et c'est exactement
    ce que fait la reprise par graines supplementaires depuis toujours. On
    ecrit maintenant apres CHAQUE tirage.
    """
    p = _ckpt_path(kind, tag)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            raw = json.load(f)
    except (ValueError, OSError):
        return {}
    return {(float(r['variant']), int(r['seed'])): r for r in raw}


def _ckpt_save(kind, tag, runs):
    p = _ckpt_path(kind, tag)
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump([dict(seed=int(r['seed']), variant=float(r['variant']),
                        x=list(map(float, np.atleast_1d(r['x']))),
                        J=float(r['J']), n_eval=int(r.get('n_eval', 0)))
                   for r in runs], f)
    os.replace(tmp, p)          # remplacement ATOMIQUE : jamais de fichier mi-ecrit


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
    print(f"  PSO : {C.PSO['n_particles_base']}"
          f" + {C.PSO['n_particles_per_dim']} x dim particules"
          f" x {C.PSO['n_iter']} iterations, graines {C.PSO['seeds']}"
          f" (1 depistage par convention de signe, puis raffinement)\n")

    # Structures a optimiser. Par defaut les deux d'origine ; KINDS permet
    # d'en ajouter une sans refaire les autres (les resultats deja calcules
    # sont relus et fusionnes, jamais ecrases).
    kinds = os.environ.get('KINDS', 'fopid,adrc').split(',')
    # EXTRA_SEEDS=5,6 : ajoute des graines a une structure DEJA optimisee, sur
    # la convention de signe qu'elle avait retenue, et fusionne les resultats.
    # Sert a trancher la separabilite : la regle du protocole compare l'ecart
    # entre structures a la dispersion INTRA-structure, et une dispersion
    # large ne se reduit qu'en tirant plus de graines — pour TOUTES les
    # structures, sans quoi ce serait une faveur.
    extra = [int(v) for v in os.environ['EXTRA_SEEDS'].split(',')] \
        if os.environ.get('EXTRA_SEEDS') else None
    # UN SEUL FIL BLAS PAR PROCESSUS, imperatif si l'on lance plusieurs
    # optimisations en parallele. Les matrices d'etat de ce depot font 16 a 66
    # lignes : a cette taille le multi-fil de BLAS coute plus qu'il ne
    # rapporte, et quatre processus ouvrant chacun sept fils sur quatre coeurs
    # passent leur temps a se disputer le processeur. MESURE, une evaluation
    # complete de l'objectif (3 profondeurs x 6 positions, m = 24, 5 modes) :
    #
    #     structure   par defaut   un fil    rapport
    #     fopid         8.155 s    0.433 s     19 x
    #     fdob          9.359 s    0.893 s     10 x
    #
    # A regler AVANT d'importer numpy — apres, la bibliotheque a deja fixe sa
    # reserve de fils :
    #
    #     export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    #
    # OUT_TAG : ecrire dans un fichier SEPARE. Sans cela, deux optimisations
    # lancees en parallele — une par coeur libre — reliraient le meme .npz et
    # la derniere a finir ecraserait le travail de l'autre. Chaque processus
    # ecrit donc son propre fichier, et `merge_pso.py` les reunit ensuite.
    tag = os.environ.get('OUT_TAG', '')
    store = {}
    path = os.path.join(OUT, f'pso_{C.PROTOCOL}.npz')
    if os.path.exists(path):
        old = np.load(path, allow_pickle=True)
        for k in old.files:
            kk, _, field = k.partition('__')
            store.setdefault(kk, {})[field] = old[k]
        print(f"  deja en memoire : {sorted(store)}\n")
    for kind in kinds:
        D0 = Design(kind, plate, sign_loop)
        n_states = D0.order(np.full(D0.n, 0.5))
        print(f"  --- {kind.upper()} : {D0.n} parametres,"
              f" {n_states} etats ---", flush=True)
        best_x, best_J, best_var, runs = None, -np.inf, None, []
        if extra is not None:
            key0 = kind if kind != 'fdob' or C.FDOB_MODES == '12' \
                else 'fdob' + C.FDOB_MODES
            prev = store[key0]
            best_var = float(prev['sign_variant'])
            best_x, best_J = np.array(prev['x']), float(prev['J'])
            D = Design(kind, plate, sign_loop, sign_variant=best_var,
                       targets=prev.get('targets'))
            # Les structures optimisees avant l'ajout du champ 'xs' n'ont
            # pas leurs x par graine ; on garde alors leurs J et on marque
            # les x manquants par NaN plutot que d'inventer une valeur.
            pxs = prev['xs'] if 'xs' in prev else np.full(
                (len(prev['seeds']), len(np.atleast_1d(prev['x']))), np.nan)
            # Les historiques de convergence DEJA stockes sont relus ici et
            # concatenes plus bas. La version precedente marquait toutes les
            # anciennes graines `history=None`, si bien que `hs` ne retenait
            # que les graines AJOUTEES : chaque reprise ECRASAIT les courbes
            # des graines initiales, definitivement, puisque le .npz est
            # reecrit en place. C'est ce qui a laisse `hist` a deux lignes
            # pour des structures qui comptent sept a neuf graines.
            #
            # On ne cherche PAS a les rattacher graine par graine : le champ
            # `hist` est un simple empilement de courbes, sans cle, et les
            # fichiers deja abimes en contiennent moins qu'ils n'ont de
            # graines. Les reapparier positionnellement les etiquetterait
            # faux. On les empile, et on le dit.
            phist = [np.asarray(r, float)
                     for r in np.atleast_2d(prev['hist'])] \
                if 'hist' in prev and np.size(prev['hist']) else []
            for sd, xx, jj, vv in zip(prev['seeds'], pxs,
                                      prev['J_seeds'], prev['variants']):
                runs.append(dict(seed=int(sd), variant=float(vv), x=xx,
                                 J=float(jj), history=None, n_eval=0))
            print(f"    reprise : {len(runs)} graines deja faites,"
                  f" convention {best_var:+.0f}, J = {best_J:+.4f}",
                  flush=True)
            for seed in extra:
                t0 = time.time()
                fit = lambda u: evaluate(plate, D.build(u))
                x, J, inf = pso(fit, D.n, seed=seed)
                runs.append(dict(seed=seed, variant=best_var, x=x, J=J,
                                 history=inf['history'],
                                 n_eval=inf['n_eval']))
                print(f"    graine supplementaire {seed} : J = {J:+.4f} mm"
                      f"  ({inf['n_eval']} evaluations,"
                      f" {time.time() - t0:.0f} s)", flush=True)
                if J > best_J:
                    best_J, best_x = J, x.copy()
            hs = phist + [r['history'] for r in runs
                          if r['history'] is not None]
            print(f"    historiques conserves : {len(phist)} anciens"
                  f" + {len(hs) - len(phist)} nouveaux", flush=True)
            n_ev = int(prev['n_eval']) + sum(r['n_eval'] for r in runs)
            store[key0] = _pack(D, best_x, best_J, best_var, runs,
                                plate, int(prev['n_states']), hs, n_ev)
            continue
        # ETAPE 1 — depistage : une graine par convention de signe.
        # "Meme budget" n'est pas "meme budget UTILE" : la moitie +1 du boitier
        # ADRC-FOPID ne contient AUCUN correcteur nominalement stable (0 sur
        # 2500 tirages, et 0 sur les 1800 evaluations qu'elle avait consommees
        # dans la version precedente), pendant que les deux moities du boitier
        # FOPID sont productives. On depiste donc, puis on REPORTE le budget de
        # la moitie vide sur la moitie qui vit — identiquement pour les deux
        # structures : 2 depistages + 3 raffinements chacune.
        screen = {}
        done = _ckpt_load(kind, tag)
        for variant in (+1.0, -1.0):
            D = Design(kind, plate, sign_loop, sign_variant=variant)
            t0 = time.time()
            prev = done.get((variant, int(C.PSO['seeds'][0])))
            if prev is not None:
                x, J, inf = (np.asarray(prev['x'], float), float(prev['J']),
                             dict(history=None, n_eval=int(prev['n_eval'])))
                repris = '  (repris du disque)'
            else:
                fit = lambda u: evaluate(plate, D.build(u))
                x, J, inf = pso(fit, D.n, seed=C.PSO['seeds'][0])
                repris = ''
            screen[variant] = J
            runs.append(dict(seed=C.PSO['seeds'][0], variant=variant, x=x, J=J,
                             history=inf['history'], n_eval=inf['n_eval']))
            _ckpt_save(kind, tag, runs)
            print(f"    depistage signe {variant:+.0f} : J = {J:+.4f} mm"
                  f"  ({inf['n_eval']} evaluations,"
                  f" {time.time() - t0:.0f} s){repris}", flush=True)
            if J > best_J:
                best_J, best_x, best_var = J, x.copy(), variant
        alive = [v for v, J in screen.items() if J > -900.0]
        if not alive:
            alive = list(screen)
        best_var = max(alive, key=lambda v: screen[v])
        print(f"    conventions productives : {sorted(alive)}"
              f" -> raffinement sur {best_var:+.0f}", flush=True)
        # ETAPE 2 — raffinement : les graines restantes sur la convention retenue
        D = Design(kind, plate, sign_loop, sign_variant=best_var)
        for seed in C.PSO['seeds'][1:]:
            t0 = time.time()
            prev = done.get((best_var, int(seed)))
            if prev is not None:
                x, J, inf = (np.asarray(prev['x'], float), float(prev['J']),
                             dict(history=None, n_eval=int(prev['n_eval'])))
                repris = '  (repris du disque)'
            else:
                fit = lambda u: evaluate(plate, D.build(u))
                x, J, inf = pso(fit, D.n, seed=seed)
                repris = ''
            runs.append(dict(seed=seed, variant=best_var, x=x, J=J,
                             history=inf['history'], n_eval=inf['n_eval']))
            _ckpt_save(kind, tag, runs)
            print(f"    raffinement graine {seed} : J = {J:+.4f} mm"
                  f"  ({inf['n_eval']} evaluations,"
                  f" {time.time() - t0:.0f} s){repris}", flush=True)
            if J > best_J:
                best_J, best_x, best_var = J, x.copy(), best_var
        D = Design(kind, plate, sign_loop, sign_variant=best_var)
        print(f"    convention retenue : sign_variant = {best_var:+.0f}")
        par = D.decode(best_x)
        _, info = evaluate(plate, D.build(best_x), detail=True)
        print(f"    meilleur : J = {best_J:+.4f}   Ms = {info['Ms']:.2f}"
              f"   effort = {info['V']:.0f} V/N")
        print("    parametres : " + "  ".join(
            f"{k}={v:.4g}" for k, v in par.items()), flush=True)
        ss = D.build(best_x)
        key = kind if kind != 'fdob' or C.FDOB_MODES == '12' \
            else 'fdob' + C.FDOB_MODES
        store[key] = _pack(D, best_x, best_J, best_var, runs, plate,
                           n_states)

    print("\n  budget : "
          + ", ".join(f"{int(v['n_eval'])} ({k})" for k, v in store.items())
          + " — l'essaim est proportionnel a la dimension, donc le nombre"
            " d'evaluations differe et est rapporte tel quel")
    out_path = os.path.join(OUT, f'pso_{C.PROTOCOL}{tag}.npz')
    np.savez_compressed(out_path,
                        **{f'{k}__{kk}': vv for k, v in store.items()
                           for kk, vv in v.items()})
    print(f"  -> {os.path.basename(out_path)}   ({time.time() - t00:.0f} s)")


if __name__ == '__main__':
    main()
