"""
robust_poles.py — POURQUOI une limite vaut zero, cas par cas
=============================================================
`closed_loop.limit` rend 0.0 dans DEUX situations que rien ne distinguait
ensuite :

  (a) la boucle fermee est INSTABLE A PROFONDEUR NULLE — le correcteur ne
      tient pas la plaque, coupe ou pas ;
  (b) la boucle tient, mais la limite de coupe est SOUS LA BORNE BASSE de la
      bissection (5 um), donc trop petite pour etre mesuree par ce protocole.

Les deux rendent « 0.000 mm » et ne veulent pas du tout dire la meme chose.
Mesure sur la campagne : a -10 % de raideur/masse, LQG et MPC sont dans le
cas (a) — max Re(pole) = +7.68 et +40.06 — tandis que NMP-DOB est dans le cas
(b), avec max Re(pole) = -0.1398, donc STABLE, de justesse. Presenter les
trois comme « instables » aurait ete faux pour l'un des trois.

Ce script calcule, pour chaque cas de robustesse et chaque structure, le
maximum de la partie reelle des poles de la boucle fermee A PROFONDEUR NULLE.
C'est un simple probleme de valeurs propres sur la matrice augmentee plaque +
correcteur : aucun rapport avec la bissection, donc c'est une verification
INDEPENDANTE et non une reformulation.

    PROTOCOL=B python robust_poles.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'), HERE]

import config as C                                            # noqa: E402
from plate_model import build_plate                           # noqa: E402
from objective import nominal_max_re                          # noqa: E402
from stored_ctrl import discover as _discover                 # noqa: E402

OUT = os.path.join(HERE, '..', 'results')


def perturbed(freqs=None, zeta_scale=1.0, w_scale=1.0):
    pl = build_plate(C.PATCH_SIDE,
                     freqs=C.F_NOMINAL if freqs is None else freqs)
    if zeta_scale != 1.0:
        pl.zeta_modes = np.asarray(pl.zeta_modes, float) * zeta_scale
    if w_scale != 1.0:
        pl.calibrate_frequencies(list(np.asarray(pl.freq_n, float) * w_scale))
    return pl


def max_re(plate, ss, n_modes, pd=None):
    """max Re(pole) de la boucle fermee a a_p = 0, position mediane.

    Le calcul lui-meme vit dans `objective.nominal_max_re`, partage avec
    `run_compare` : les deux tableaux affichent la meme colonne, autant
    qu'elle vienne du meme code. A profondeur nulle et sans terme retarde le
    terme de coupe disparait, la matrice ne depend plus ni de la position ni
    du retard, et le probleme est purement propre ; avec le terme retarde de
    l'Eq. (30) c'est FAUX, et silencieusement — d'ou le detour par la
    monodromie decrit la-bas."""
    return nominal_max_re(plate, ss, pd, n_modes)


def stored():
    """[(kind, ss, pd)], boucle ouverte en tete."""
    return [('boucle ouverte', None, None)] + [
        (k, ss, pd) for k, (ss, pd) in _discover().items()]


def main():
    drift = [632.0, 1162.0] + list(C.F_NOMINAL[2:])
    cases = [('modele de synthese', dict(), C.N_MODES_OBJ),
             ('modele complet (verite)', dict(), C.N_MODES),
             ('derive +17/+9 %', dict(freqs=drift), C.N_MODES),
             ('amortissement x0.8', dict(zeta_scale=0.8), C.N_MODES),
             ('raideur/masse +10 %', dict(w_scale=np.sqrt(1.1)), C.N_MODES),
             ('raideur/masse -10 %', dict(w_scale=np.sqrt(0.9)), C.N_MODES),
             ('calage theorique', dict(freqs=C.F_THEORETICAL), C.N_MODES)]
    got = stored()
    print('=' * 78)
    print(' POLES DE LA BOUCLE FERMEE A PROFONDEUR NULLE — max Re [1/s]')
    print('=' * 78)
    print('  un maximum POSITIF signifie que le correcteur ne tient pas la'
          ' plaque,')
    print('  coupe ou pas : la limite nulle correspondante est une'
          ' INSTABILITE,')
    print('  non une limite trop petite pour la bissection.\n')
    ks = [t[0] for t in got]
    print(f'{"cas":26s}' + ''.join(f'{k[:10]:>11s}' for k in ks))
    table = np.zeros((len(cases), len(got)))
    for i, (tag, kw, nm) in enumerate(cases):
        plate = perturbed(**kw)
        row = []
        for j, (kind, ss, pd) in enumerate(got):
            table[i, j] = max_re(plate, ss, nm, pd)
            row.append(table[i, j])
        print(f'{tag[:25]:26s}' + ''.join(f'{v:11.3f}' for v in row),
              flush=True)
    np.savez_compressed(os.path.join(OUT, f'robust_poles_{C.PROTOCOL}.npz'),
                        labels=np.array([t for t, _, _ in cases]),
                        kinds=np.array(ks), max_re=table)
    bad = [(cases[i][0], ks[j]) for i in range(len(cases))
           for j in range(len(ks)) if table[i, j] > 0.0]
    print()
    if bad:
        print('  INSTABLES a coupe nulle :')
        for tag, k in bad:
            print(f'    {k:12s} sous « {tag} »')
    else:
        print('  aucune structure instable a coupe nulle')
    print(f'\n  -> robust_poles_{C.PROTOCOL}.npz')
    return 0


if __name__ == '__main__':
    sys.exit(main())
