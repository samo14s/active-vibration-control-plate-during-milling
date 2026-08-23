"""
ordonnancer_vs_incertitude.py — LA JOUER ORDONNANCEE OU LA COUVRIR ?
=====================================================================
Question centrale du cadre propose : vaut-il mieux EXPLOITER la variation de
dynamique (un correcteur par vitesse, ordonnance sur tau) que la COUVRIR par
une conception unique robuste ? Personne ne l'a mesure sur ce probleme ; on
l'affirme des deux cotes.

PROTOCOLE. On ne compare PAS mu-synthese contre un correcteur ordonnance :
cela melangerait deux choses, la structure et le protocole de conception, et ne
prouverait rien. On prend UNE SEULE structure — dvf, deux parametres, donc une
grille exhaustive est possible — et DEUX protocoles :

    FIXE       : un seul vecteur de parametres, choisi pour maximiser le PIRE
                 des vitesses. C'est ce que fait une conception robuste : elle
                 couvre la plage avec un correcteur unique.
    ORDONNANCE : un vecteur PAR vitesse, chacun maximisant l'objectif a sa
                 vitesse. C'est l'enveloppe de ce que l'ordonnancement peut
                 atteindre — une borne SUPERIEURE, jamais atteinte en pratique
                 puisqu'un ordonnancement reel doit interpoler.

L'ecart entre les deux EST la valeur de l'ordonnancement, en millimetres de
profondeur atteignable. S'il est faible, le cadre propose ne vaut pas sa
complexite et il vaut mieux le savoir maintenant.

Les deux reponses sortent de la MEME table : on evalue l'objectif conjoint sur
une grille de parametres x une grille de vitesses, puis

    ORDONNANCE(v) = max_u  J(u, v)
    FIXE          = max_u  min_v J(u, v)

Aucun optimiseur, donc aucun doute sur la convergence : la grille est
exhaustive, et les deux protocoles voient EXACTEMENT les memes points.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'paper_model'), HERE]

import config as C                                                   # noqa
from plate_model import build_plate, plant_vectors                   # noqa
from pso import Design                                               # noqa
from objective_joint import evaluate_joint                           # noqa

SPEEDS = (3000, 3600, 4200, 4800, 5400, 6000, 6600, 7000)
NG = int(os.environ.get('NG', 15))          # grille NG x NG sur [0,1]^2


def main():
    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    _, _, _, _, sign_loop = plant_vectors(plate, C.N_MODES_DESIGN)
    grille = []
    for sv in (+1.0, -1.0):                 # les deux conventions de signe
        D = Design('dvf', plate, sign_loop, sign_variant=sv)
        for a in np.linspace(0.0, 1.0, NG):
            for b in np.linspace(0.0, 1.0, NG):
                grille.append((D, np.array([a, b])))
    print(f'  dvf, grille {NG}x{NG} x 2 conventions = {len(grille)} '
          f'correcteurs, {len(SPEEDS)} vitesses', flush=True)
    J = np.full((len(grille), len(SPEEDS)), -np.inf)
    t0 = time.time()
    for i, (D, u) in enumerate(grille):
        ss = D.build(u)
        for j, v in enumerate(SPEEDS):
            J[i, j] = evaluate_joint(plate, ss, pd=D.delay_gains(u),
                                     rpm=float(v))
        if (i + 1) % 50 == 0:
            print(f'    {i+1}/{len(grille)}  ({time.time()-t0:.0f} s)',
                  flush=True)
    np.savez_compressed(os.path.join(ROOT, 'results', 'ordonnancement.npz'),
                        J=J, speeds=np.array(SPEEDS), ng=NG)

    ok = J > 0.0                            # J <= 0 : crible non franchi
    ordo = np.where(ok.any(axis=0), J.max(axis=0), np.nan)
    pire_par_u = np.where(ok.all(axis=1), J.min(axis=1), -np.inf)
    i_fixe = int(np.argmax(pire_par_u))
    fixe_pire = float(pire_par_u[i_fixe])
    fixe = J[i_fixe]

    print(f'\n{"rpm":>5s} {"ORDONNANCE":>11s} {"FIXE":>8s} {"gain":>8s}')
    for j, v in enumerate(SPEEDS):
        g = (ordo[j] / fixe[j]) if fixe[j] > 0 else np.nan
        print(f'{v:5d} {ordo[j]:11.3f} {fixe[j]:8.3f} {g:7.2f}x')
    print(f'\n  FIXE   : pire des vitesses = {fixe_pire:.3f} mm '
          f'(moyenne {np.mean(fixe):.3f})')
    print(f'  ORDONN : pire des vitesses = {np.nanmin(ordo):.3f} mm '
          f'(moyenne {np.nanmean(ordo):.3f})')
    if fixe_pire > 0:
        print(f'\n  VALEUR DE L ORDONNANCEMENT : '
              f'{np.nanmin(ordo)/fixe_pire:.2f}x sur le pire cas, '
              f'{np.nanmean(ordo)/np.mean(fixe):.2f}x en moyenne')
    return 0


if __name__ == '__main__':
    sys.exit(main())
