"""Pourquoi la fonctionnelle echoue : elle est INDEPENDANTE DU RETARD.

Hypothese : la LMI
    [A B C]' P' [A B C] < diag(P - Q - R, Q, R)
avec Q, R CONSTANTES impose en substance  P > A'P'A + B'P'B + C'P'C, c est-a-dire
la stabilite quelle que soit la PHASE du terme retarde. Or les fossoles du
fraisage sont precisement un phenomene DEPENDANT du retard : a profondeur
egale, la meme plaque est stable a une vitesse et instable a une autre, avec
des blocs retardes de meme TAILLE et de phase differente.

Si c est vrai, aucun critere independant du retard ne peut reussir dans la
region lobee — l infaisabilite est structurelle, pas numerique.

Test : comparer ||B||, ||C|| a 5200 (stable, rho=0.940) et a 5600 (instable,
rho=1.079). Si les tailles sont comparables, un critere qui ne voit que les
tailles ne peut pas separer les deux cas, donc doit echouer aux deux.
"""
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import matrix_balance
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(R, 'paper_model'), os.path.join(R, 'control')]
import config as C
from plate_model import build_plate
from closed_loop import period_maps, spectral_radius
import stored_ctrl
NM, AP, M = 2, 0.60e-3, 40
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()['vpa']
print(f'{"tr/min":>7s} {"rho (m=200)":>12s} {"max||A||":>11s} {"max||B||":>11s} '
      f'{"max||C||":>11s} {"(B+C)/A":>9s}')
for r in (4800, 5000, 5200, 5400, 5600, 5800, 6000):
    mp2 = period_maps(plate, r, AP, 0.5*plate.lp, ctrl=ss, pd=pd, n_modes=NM,
                      m=200, coeff_mode='time', coeff_scale=C.SIGN_SIM, ae=C.AE)[0]
    rho = spectral_radius(mp2, 200, mp2[0][0].shape[0])
    mp = period_maps(plate, r, AP, 0.5*plate.lp, ctrl=ss, pd=pd, n_modes=NM,
                     m=M, coeff_mode='time', coeff_scale=C.SIGN_SIM, ae=C.AE)[0]
    moy = np.mean([np.abs(a)+np.abs(b)+np.abs(c) for a, b, c in mp], axis=0)
    _, T = matrix_balance(moy, permute=False); Ti = np.linalg.inv(T)
    BL = [(Ti@a@T, Ti@b@T, Ti@c@T) for a, b, c in mp]
    na = max(np.linalg.norm(a) for a, _, _ in BL)
    nb = max(np.linalg.norm(b) for _, b, _ in BL)
    nc = max(np.linalg.norm(c) for _, _, c in BL)
    print(f'{r:7d} {rho:12.6f} {na:11.4f} {nb:11.4f} {nc:11.4f} '
          f'{(nb+nc)/na:9.4f}' + ('' if rho < 1 else '   INSTABLE'), flush=True)
print('\n  Si les tailles se ressemblent entre lignes stables et instables,')
print('  un critere qui ne voit que les tailles ne peut separer les deux.')
