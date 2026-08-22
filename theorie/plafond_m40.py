"""Le plafond, recalcule A LA MEME DISCRETISATION que le certificat.

La SDP par pas tourne a m = 40, le plafond de la phase 4 avait ete mesure a
m = 200. rho au centre coincide a six chiffres (0.943863 / 0.943864), ce qui
m avait rassure a tort : la BANDE est bien plus sensible que rho, puisque pres
du bord un ecart au sixieme chiffre deplace le croisement de dizaines de
tours/min. Comparer 319 (m=40) a 312.9 (m=200) comparait deux systemes.

On refait donc le plafond a m = 40, avec un pas de balayage FIN (1 tr/min) au
lieu de 21 points : un echantillonnage grossier peut enjamber une poche etroite
et SURESTIMER la bande vraie, ce qui fausserait le ratio dans l autre sens.
"""
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import expm, matrix_balance
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(R, 'paper_model'), os.path.join(R, 'control')]
import config as C
from plate_model import build_plate
from milling_dynamics import alpha4_series, N_TEETH
from closed_loop import build_matrices
import stored_ctrl
NM, AP = 2, 0.60e-3
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()['vpa']
D = plate.D_row(0.5*plate.lp, plate.hp)[:NM]; DtD = np.outer(D, D)
Do = plate.D_row(plate.lp, plate.hp)[:NM]
H = np.asarray(plate.H_Pe_modal, float)[:NM]
t = lambda r: 60.0/(N_TEETH*r)

def rho_m(r, m):
    _, a4 = alpha4_series(4900, AP, plate.hp, m)
    A = [build_matrices(plate, DtD, Do, H, C.SIGN_SIM*a, ss, pd, NM)[0]
         for a in a4]
    moy = np.mean([np.abs(x) for x in A], axis=0)
    _, T = matrix_balance(moy, permute=False); Ti = np.linalg.inv(T)
    P = np.eye(2*NM + np.shape(ss[0])[0])
    for x in A: P = expm((Ti@x@T)*t(r)/m) @ P
    return float(np.max(np.abs(np.linalg.eigvals(P))))

print(f'{"centre":>7s} {"bande m=40":>12s} {"bande m=200":>13s} '
      f'{"ecart":>9s}   (balayage a 1 tr/min)')
for r0 in (5600, 5200, 4400, 4900):
    out = []
    for m in (40, 200):
        dw = 0
        while dw < 900:
            nxt = dw + 1
            if rho_m(r0-nxt, m) >= 1.0 or rho_m(r0+nxt, m) >= 1.0: break
            dw = nxt
        out.append(dw)
    print(f'{r0:7d} {out[0]:10d} tr {out[1]:11d} tr '
          f'{out[0]-out[1]:+7d} tr', flush=True)
