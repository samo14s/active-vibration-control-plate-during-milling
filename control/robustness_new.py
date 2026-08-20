"""
robustness_new.py — les sept cas de robustesse, pour TOUTES les structures
==========================================================================
Meme jeu de plaques perturbees que `run_compare.py`, meme metrique — la limite
axiale au PIRE poste, par bissection sur cinq modes a m = 200 — appliquee aux
structures ajoutees (H-infini, mu, LQG, VPA, DVF, NMP-DOB) comme aux quatre
d'origine.

POURQUOI CE SCRIPT PLUTOT QUE `run_compare.py`. Ce dernier recalcule aussi les
lobes sur toute la plage de vitesse et le temporel, ce qui coute des heures. Ici
on veut UNE chose : le tableau de robustesse. Les correcteurs sont relus tels
qu'ils ont ete optimises, jamais reconstruits — un correcteur rebati depuis ses
parametres arrondis n'est pas le meme objet, comme la campagne H-infini
retiree l'a montre.

Les cas viennent du papier, pas d'un choix commode : la derive +17/+9 % est
celle que sa Section 5 constate, l'amortissement a 80 % et les +/-10 % de
masse/raideur viennent de sa Section 4.2, et le calage theorique est le jeu de
frequences de son Tableau 1.

    PROTOCOL=B CALIB=measured python robustness_new.py
"""
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'), HERE]

import config as C                                            # noqa: E402
from plate_model import build_plate                           # noqa: E402
from closed_loop import limit as cl_limit                     # noqa: E402

OUT = os.path.join(HERE, '..', 'results')

#: ORDRE D'AFFICHAGE, pas liste de structures : rien n'est code en dur ici.
#: `discover` lit ce que les fichiers contiennent REELLEMENT — le fichier
#: fusionne d'abord, puis les fichiers paralleles pour les structures que la
#: fusion n'a pas encore reunies. Une structure nouvelle apparait donc dans le
#: tableau sans qu'on touche a ce fichier, ce qui est la condition meme de
#: l'equite : le meme code d'evaluation pour toutes.
ORDER = ['fopid', 'adrc', 'fdob', 'fdob12345', 'dvf', 'vpa', 'hinf', 'musyn',
         'lqg', 'mpc', 'smc', 'nmpdob']


def discover():
    """[(kind, (A, B, C, D))], boucle ouverte en tete."""
    found = {}
    merged = os.path.join(OUT, f'pso_{C.PROTOCOL}.npz')
    paths = ([merged] if os.path.exists(merged) else []) + sorted(
        glob.glob(os.path.join(OUT, f'pso_{C.PROTOCOL}_*.npz')))
    for path in paths:
        d = np.load(path, allow_pickle=True)
        for k in d.files:
            kind, _, field = k.partition('__')
            if field == 'A' and kind not in found:
                found[kind] = (d[f'{kind}__A'], d[f'{kind}__B'],
                               d[f'{kind}__C'], d[f'{kind}__D'])
    rank = {k: i for i, k in enumerate(ORDER)}
    keys = sorted(found, key=lambda k: (rank.get(k, len(ORDER)), k))
    return [('boucle ouverte', None)] + [(k, found[k]) for k in keys]


def worst_limit(plate, ss, n_modes):
    """Limite au pire poste. `POSITIONS_DESIGN` sont des FRACTIONS de l_P ;
    `limit` attend une coordonnee physique — l'oublier rend zero partout."""
    return min(cl_limit(plate, C.RPM_DESIGN, x * plate.lp, ctrl=ss,
                        n_modes=n_modes, m=200, hi=6e-3)
               for x in C.POSITIONS_DESIGN)


def main():
    drift = [632.0, 1162.0] + list(C.F_NOMINAL[2:])
    cases = [('modele de synthese', dict(), C.N_MODES_OBJ),
             ('modele complet (verite)', dict(), C.N_MODES),
             ('derive +17/+9 %', dict(freqs=drift), C.N_MODES),
             ('amortissement x0.8', dict(zeta_scale=0.8), C.N_MODES),
             ('raideur/masse +10 %', dict(w_scale=np.sqrt(1.1)), C.N_MODES),
             ('raideur/masse -10 %', dict(w_scale=np.sqrt(0.9)), C.N_MODES),
             ('calage theorique', dict(freqs=C.F_THEORETICAL), C.N_MODES)]

    # Le MEME `perturbed` que run_compare.py, recopie a l'identique plutot
    # qu'importe : run_compare le definit a l'interieur de main(). Une
    # divergence entre les deux rendrait les deux tableaux incomparables, donc
    # `tests/` compare les deux constructions plaque par plaque.
    def perturbed(freqs=None, zeta_scale=1.0, w_scale=1.0):
        pl = build_plate(C.PATCH_SIDE,
                         freqs=C.F_NOMINAL if freqs is None else freqs)
        if zeta_scale != 1.0:
            pl.zeta_modes = np.asarray(pl.zeta_modes, float) * zeta_scale
        if w_scale != 1.0:
            pl.calibrate_frequencies(list(np.asarray(
                pl.freq_n, float) * w_scale))
        return pl

    got = discover()
    print('  structures trouvees : '
          + ', '.join(k for k, _ in got[1:]))
    print('=' * 78)
    print(' ROBUSTESSE — limite axiale au pire poste [mm], 5 modes, m = 200')
    print('=' * 78)
    print(f'  structures : {", ".join(k for k, _ in got)}\n')

    table = {}
    for tag, kw, nm in cases:
        plate = perturbed(**kw)
        t0 = time.time()
        row = {}
        for kind, ss in got:
            row[kind] = worst_limit(plate, ss, nm)
        table[tag] = row
        print(f'  {tag:26s} ' + '  '.join(f'{k}={v * 1e3:.3f}'
                                          for k, v in row.items())
              + f'   ({time.time() - t0:.0f} s)', flush=True)

    np.savez_compressed(os.path.join(OUT, f'robust_new_{C.PROTOCOL}.npz'),
                        labels=np.array([t for t, _, _ in cases]),
                        kinds=np.array([k for k, _ in got]),
                        limits=np.array([[table[t][k] for k, _ in got]
                                         for t, _, _ in cases]))
    print(f'\n  -> results/robust_new_{C.PROTOCOL}.npz')


if __name__ == '__main__':
    main()
