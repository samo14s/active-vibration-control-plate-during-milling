import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(R, 'paper_model'), os.path.join(R, 'control')]
import config as C
from plate_model import build_plate
from milling_dynamics import alpha4_series
from closed_loop import build_matrices
from stored_ctrl import discover
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
n = C.N_MODES
D = plate.D_row(0.5*plate.lp, plate.hp)[:n]; DtD = np.outer(D, D)
D_obs = plate.D_row(plate.lp, plate.hp)[:n]
H = np.asarray(plate.H_Pe_modal, float)[:n]
def pire(ss, pd, ap, m=128):
    _, a4 = alpha4_series(4900, ap, plate.hp, m)
    return max(float(np.max(np.real(np.linalg.eigvals(
        build_matrices(plate, DtD, D_obs, H, C.SIGN_SIM*a, ss, pd, n)[0]))))
        for a in a4)
print(f'{"structure":12s} {"a_p ou A0(s) cesse d etre Hurwitz":>36s}'
      f'  {"a_p,lim mesuree":>16s}')
for kind, lim in (('musyn_td', 0.3969), ('musyn', 0.3757), ('vpa', 0.560),
                  ('lqg', 0.759), ('fopid', 0.201)):
    ss, pd = discover()[kind]
    lo, hi = 0.0, 1.2e-3
    if pire(ss, pd, hi) <= 0: print(f'{kind:12s} {"> 1.20 mm":>36s}"); continue'); continue
    for _ in range(22):
        mid = 0.5*(lo+hi)
        lo, hi = (mid, hi) if pire(ss, pd, mid) <= 0 else (lo, mid)
    print(f'{kind:12s} {0.5*(lo+hi)*1e3:33.4f} mm  {lim:14.4f} mm')
