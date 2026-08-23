"""
balayage_qualite.py — le plafond de qualite depend-il de la vitesse ?
======================================================================
Toute l'analyse de qualite a ete faite a 5200 tr/min, choisie parce que c'est
la vitesse des phases theoriques — sans verifier ce qu'elle vaut ici. Or les
modes de la plaque sont a 540 et 1068 Hz, et a 5200 tr/min la frequence de dent
vaut 260 Hz, donc :

    harmonique 2 = 520 Hz  -> 3.7 % du mode 1 (540 Hz)
    harmonique 4 = 1040 Hz -> 2.6 % du mode 2 (1068 Hz)

LES DEUX modes sont quasi resonants avec des harmoniques de passage de dent.
5200 tr/min est donc un point de RESONANCE DOUBLE, et les chiffres de qualite
qui en sortent sont proches du pire cas — ce qui n'avait pas ete dit.

(L'explication donnee auparavant — « les correcteurs amortissent la bande de
broutement vers 1400 Hz et amplifient celle de passage de dent » — etait
fausse : 1400 Hz est la frequence de coin d'une ponderation d'incertitude, pas
un mode. Le mecanisme reel est la resonance harmonique.)

Ce script mesure donc les deux plafonds de qualite SUR LA PLAGE, et repond a
deux questions que la table a une seule vitesse ne pouvait pas trancher :

  1. existe-t-il des vitesses ou la qualite ne lie PAS, et la stabilite
     redevient le critere pertinent ?
  2. le classement des structures survit-il au changement de vitesse, ou
     n'etait-il qu'un artefact de 5200 ?
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'paper_model'), HERE]

import config as C                                                   # noqa
from plate_model import build_plate                                  # noqa
import stored_ctrl                                                   # noqa
from objective_joint import plafonds_qualite                         # noqa

SPEEDS = list(range(3000, 7001, 200))


def main():
    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    st = stored_ctrl.discover()
    noms = ['AUCUN', 'dvf', 'lqg', 'musyn_td', 'vpa']
    f1, f2 = plate.freq_n[0], plate.freq_n[1]
    print(f'  modes : {f1:.0f} et {f2:.0f} Hz ; plafonds au PIRE des '
          f'{len(C.POSITIONS_DESIGN)} positions ; qual = |SLE| <= 10 um, '
          f'val = crete-a-crete <= f_z\n')
    print(f'{"rpm":>5s} {"f_dent":>7s} {"h2":>7s} {"h4":>7s}' +
          ''.join(f'{n[:8]:>18s}' for n in noms))
    print(f'{"":>5s} {"[Hz]":>7s} {"[Hz]":>7s} {"[Hz]":>7s}' +
          ''.join(f'{"qual / val [mm]":>18s}' for _ in noms))
    for rpm in SPEEDS:
        ft = 3.0 * rpm / 60.0
        row = f'{rpm:5d} {ft:7.1f} {2*ft:7.1f} {4*ft:7.1f}'
        for n in noms:
            ss, pd = (None, None) if n == 'AUCUN' else st[n]
            q, v = plafonds_qualite(plate, ss, pd=pd, rpm=rpm)
            fq = '  inf' if not np.isfinite(q) else f'{q*1e3:5.3f}'
            row += f'{fq:>9s} /{v*1e3:8.3f}'
        print(row, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
