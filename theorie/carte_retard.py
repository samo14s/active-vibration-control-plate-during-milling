"""Ou le systeme AVEC retard est-il stable, et de combien le retard change-t-il
la donne ? Les phases 3-6 ont travaille a A1 = 0 ; il faut savoir ce que cette
mise a zero a rendu plus facile."""
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
from milling_dynamics import N_TEETH, alpha4_series
from closed_loop import period_maps, spectral_radius, build_matrices
import stored_ctrl
NM, AP, M = 2, 0.60e-3, 200
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()['vpa']
t = lambda r: 60.0/(N_TEETH*r)
def rho_ret(r):
    mp = period_maps(plate, r, AP, 0.5*plate.lp, ctrl=ss, pd=pd, n_modes=NM,
                     m=M, coeff_mode='time', coeff_scale=C.SIGN_SIM, ae=C.AE)[0]
    return spectral_radius(mp, M, mp[0][0].shape[0])
D = plate.D_row(0.5*plate.lp, plate.hp)[:NM]; DtD = np.outer(D, D)
Do = plate.D_row(plate.lp, plate.hp)[:NM]
H = np.asarray(plate.H_Pe_modal, float)[:NM]
_, a4 = alpha4_series(4900, AP, plate.hp, M)
A0r = [build_matrices(plate, DtD, Do, H, C.SIGN_SIM*a, ss, pd, NM)[0] for a in a4]
moy = np.mean([np.abs(x) for x in A0r], axis=0)
_, T = matrix_balance(moy, permute=False); Ti = np.linalg.inv(T)
A0 = [Ti@x@T for x in A0r]; N = A0[0].shape[0]
def rho_sans(r):
    P = np.eye(N)
    for x in A0: P = expm(x*t(r)/M) @ P
    return float(np.max(np.abs(np.linalg.eigvals(P))))
print(f'  vpa, {NM} modes, a_p = {AP*1e3:.2f} mm\n')
print(f'{"tr/min":>7s} {"rho SANS retard":>16s} {"rho AVEC retard":>16s} '
      f'{"ecart":>9s}')
stables = []
for r in range(3000, 7001, 200):
    a, b = rho_sans(r), rho_ret(r)
    if b < 1.0: stables.append(r)
    print(f'{r:7d} {a:16.6f} {b:16.6f} {b-a:+9.4f}'
          + ('' if b < 1.0 else '   INSTABLE avec retard'), flush=True)
print(f'\n  stables AVEC retard : {stables}')
