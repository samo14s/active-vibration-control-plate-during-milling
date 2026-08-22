"""
delay_speed_scan.py — le gain du terme retarde est-il propre a UNE vitesse ?
============================================================================
Le controle a retard actif de l'Eq. (30) exploite le retard regeneratif : son
retard EST la periode de dent, donc il suit la vitesse de broche — ce n'est pas
un parametre libre. Mais ses deux GAINS, eux, sont optimises a une seule
vitesse (C.RPM_DESIGN). Rien ne garantit qu'ils restent bons ailleurs : a
retard different, la meme paire de gains ne rend pas la meme phase.

Le tableau des fossoles le disait deja sans le dire : `musyn_td` affiche une
moyenne de 0.179 mm contre 0.320 pour `musyn`, alors qu'il fait MIEUX a la
vitesse de conception. Une moyenne plus basse avec un meilleur maximum, c'est
qu'il y a des zeros ailleurs — onze sur vingt et une vitesses.

Ce script separe les deux lectures possibles de ces zeros, comme
`robust_poles` le fait pour les cas de robustesse : limite trop petite pour la
bissection, ou boucle qui ne tient pas la plaque du tout. On calcule donc le
rayon spectral A PROFONDEUR NULLE, ou le terme de coupe disparait et ou il ne
reste que la boucle et son terme retarde. rho > 1 la : le correcteur est
instable sans meme couper.

    PROTOCOL=B CALIB=measured python control/delay_speed_scan.py
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
from closed_loop import period_maps, spectral_radius          # noqa: E402
from stored_ctrl import discover                              # noqa: E402

OUT = os.path.join(HERE, '..', 'results')
SPEEDS = np.arange(3000, 7001, 200)


def rho_a_vide(plate, ss, pd, rpm):
    """(rho, log(rho)/tau, tau) de la boucle a a_p = 0."""
    m = C.M_FLOQUET
    maps, tau = period_maps(plate, rpm, 0.0, 0.5 * plate.lp, ctrl=ss, pd=pd,
                            n_modes=C.N_MODES, m=m, coeff_mode='time',
                            coeff_scale=C.SIGN_SIM, ae=C.AE)
    rho = spectral_radius(maps, m, maps[0][0].shape[0])
    return rho, float(np.log(max(rho, 1e-300)) / tau), tau


def main():
    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    st = discover()
    avec = [(k, ss, pd) for k, (ss, pd) in st.items() if pd is not None]
    if not avec:
        print('  aucune structure a terme retarde dans le fichier')
        return 1
    sans = [(k, ss, None) for k, (ss, pd) in st.items()
            if pd is None and k in ('musyn', 'hinf')]

    print('=' * 78)
    print(' STABILITE A PROFONDEUR NULLE EN FONCTION DE LA VITESSE')
    print('=' * 78)
    print(f'  gains optimises a {C.RPM_DESIGN} tr/min ; le retard, lui, suit'
          ' la vitesse\n')
    cols = avec + sans
    print(f'{"tr/min":>7s} {"tau [ms]":>9s}'
          + ''.join(f'{k[:12]:>14s}' for k, _, _ in cols))
    table = np.zeros((len(SPEEDS), len(cols)))
    for i, rpm in enumerate(SPEEDS):
        ligne, tau = [], None
        for j, (k, ss, pd) in enumerate(cols):
            r, _, tau = rho_a_vide(plate, ss, pd, rpm)
            table[i, j] = r
            ligne.append(f'{r:14.6f}' if r <= 1.0 else f'{r:13.6f}*')
        print(f'{rpm:7.0f} {tau * 1e3:9.4f}' + ''.join(ligne), flush=True)
    print('\n  * rho > 1 : la boucle ne tient pas la plaque A COUPE NULLE.')

    for j, (k, _, pd) in enumerate(cols):
        mauvais = SPEEDS[table[:, j] > 1.0]
        if pd is None:
            continue
        print(f'\n  {k} : instable sans coupe a '
              f'{len(mauvais)}/{len(SPEEDS)} vitesses')
        if mauvais.size:
            print('    ' + ', '.join(f'{int(r)}' for r in mauvais))
        bonnes = SPEEDS[table[:, j] <= 1.0]
        if bonnes.size:
            print(f'    fenetre utile : {int(bonnes.min())}'
                  f'-{int(bonnes.max())} tr/min, avec des trous'
                  if np.any(np.diff(bonnes) > 200) else
                  f'    fenetre utile : {int(bonnes.min())}'
                  f'-{int(bonnes.max())} tr/min')
    np.savez_compressed(os.path.join(OUT, f'delay_speed_{C.PROTOCOL}.npz'),
                        rpm=SPEEDS, kinds=np.array([k for k, _, _ in cols]),
                        rho=table)
    print(f'\n  -> delay_speed_{C.PROTOCOL}.npz')
    return 0


if __name__ == '__main__':
    sys.exit(main())
