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
un ensemble de parametres x une grille de vitesses, puis

    ORDONNANCE(v) = max_u  J(u, v)
    FIXE          = max_u  min_v J(u, v)

Aucun optimiseur, donc aucun doute sur la convergence, et les deux protocoles
voient EXACTEMENT les memes points.

ECHANTILLONNAGE. En dimension 2 (dvf) la grille est exhaustive. Au-dela elle ne
l'est plus — 15^5 fait 760 000 points pour le LQG — mais la vertu du montage
n'est pas l'exhaustivite, c'est que les DEUX protocoles partagent les memes
candidats. On passe donc a un hypercube latin a graine fixe, qui la conserve
entierement. Et la question posee n'est pas « ou est l'optimum global » mais
« l'optimum SE DEPLACE-T-IL avec la vitesse » : pour cela un echantillon
couvrant suffit, puisque c'est le meme echantillon a toutes les vitesses.

KIND=lqg NS=900 pour la version a cinq parametres.
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
KIND = os.environ.get('KIND', 'dvf')
# AXE=position : meme montage, mais l'axe balaye n'est plus la vitesse de
# broche — c'est la POSITION de l'outil le long de la plaque. C'est l'autre
# moitie de la variation invoquee par le cadre propose (« spatiale ET
# temporelle ») et elle n'avait jamais ete testee. La question est identique :
# un correcteur PAR POSITION bat-il un correcteur unique choisi pour le pire
# des positions ?
AXE = os.environ.get('AXE', 'vitesse')
POSITIONS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
NG = int(os.environ.get('NG', 15))          # grille NG x NG sur [0,1]^2
NS = int(os.environ.get('NS', 0))           # >0 : hypercube latin de NS points


def main():
    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    _, _, _, _, sign_loop = plant_vectors(plate, C.N_MODES_DESIGN)
    grille = []
    for sv in (+1.0, -1.0):                 # les deux conventions de signe
        D = Design(KIND, plate, sign_loop, sign_variant=sv)
        if NS > 0:
            from scipy.stats import qmc
            u = qmc.LatinHypercube(d=D.n, seed=12345).random(NS)
            grille += [(D, u[k]) for k in range(NS)]
        else:
            for a in np.linspace(0.0, 1.0, NG):
                for b in np.linspace(0.0, 1.0, NG):
                    grille.append((D, np.array([a, b])))
    quoi = (f'hypercube latin {NS}' if NS > 0 else f'grille {NG}x{NG}')
    print(f'  {KIND} ({grille[0][0].n} parametres), {quoi} x 2 conventions = '
          f'{len(grille)} correcteurs, axe = {AXE}', flush=True)
    axe = POSITIONS if AXE == 'position' else SPEEDS
    J = np.full((len(grille), len(axe)), -np.inf)
    t0 = time.time()
    for i, (D, u) in enumerate(grille):
        ss = D.build(u)
        for j, v in enumerate(axe):
            if AXE == 'position':
                J[i, j] = evaluate_joint(plate, ss, pd=D.delay_gains(u),
                                         positions=(float(v),))
            else:
                J[i, j] = evaluate_joint(plate, ss, pd=D.delay_gains(u),
                                         rpm=float(v))
        if (i + 1) % 50 == 0:
            print(f'    {i+1}/{len(grille)}  ({time.time()-t0:.0f} s)',
                  flush=True)
    np.savez_compressed(os.path.join(ROOT, 'results',
                                 f'ordonnancement_{KIND}_{AXE}.npz'),
                    J=J, speeds=np.array(SPEEDS), ng=NG, ns=NS)

    ok = J > 0.0                            # J <= 0 : crible non franchi
    ordo = np.where(ok.any(axis=0), J.max(axis=0), np.nan)
    pire_par_u = np.where(ok.all(axis=1), J.min(axis=1), -np.inf)
    i_fixe = int(np.argmax(pire_par_u))
    fixe_pire = float(pire_par_u[i_fixe])
    fixe = J[i_fixe]

    am = [int(np.argmax(J[:, j])) for j in range(len(axe))]
    print(f'\n  optimum par vitesse : indices {am}')
    print(f'  se deplace-t-il ? {"OUI" if len(set(am)) > 1 else "NON"} '
          f'({len(set(am))} candidat(s) distinct(s))')
    lbl = 'x/lp' if AXE == 'position' else 'rpm'
    print(f'\n{lbl:>6s} {"ORDONNANCE":>11s} {"FIXE":>8s} {"gain":>8s}')
    for j, v in enumerate(axe):
        g = (ordo[j] / fixe[j]) if fixe[j] > 0 else np.nan
        vv = f'{v:6.3f}' if AXE == 'position' else f'{int(v):6d}'
        print(f'{vv} {ordo[j]:11.3f} {fixe[j]:8.3f} {g:7.2f}x')
    print(f'\n  FIXE   : pire du balayage = {fixe_pire:.3f} mm '
          f'(moyenne {np.mean(fixe):.3f})')
    print(f'  ORDONN : pire du balayage = {np.nanmin(ordo):.3f} mm '
          f'(moyenne {np.nanmean(ordo):.3f})')
    if fixe_pire > 0:
        print(f'\n  VALEUR DE L ORDONNANCEMENT : '
              f'{np.nanmin(ordo)/fixe_pire:.2f}x sur le pire cas, '
              f'{np.nanmean(ordo)/np.mean(fixe):.2f}x en moyenne')
    return 0


if __name__ == '__main__':
    sys.exit(main())
