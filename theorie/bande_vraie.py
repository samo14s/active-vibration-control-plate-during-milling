"""La bande CERTIFIEE contre la bande VRAIE : que laisse le certificat ?

Le certificat construit avec W_k = I n est pas optimal — c est un choix, pas
un optimum. Sa bande est donc une BORNE INFERIEURE de ce qui est certifiable.
Le plafond est la bande ou rho(tau) < 1 reellement. L ecart entre les deux
mesure ce que le choix W_k = I laisse sur la table.
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
NM, AP, M = 2, 0.60e-3, 200
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()['vpa']
D = plate.D_row(0.5*plate.lp, plate.hp)[:NM]; DtD = np.outer(D, D)
Do = plate.D_row(plate.lp, plate.hp)[:NM]
H = np.asarray(plate.H_Pe_modal, float)[:NM]
_, a4 = alpha4_series(4900, AP, plate.hp, M)
A0r = [build_matrices(plate, DtD, Do, H, C.SIGN_SIM*a, ss, pd, NM)[0] for a in a4]
moy = np.mean([np.abs(A) for A in A0r], axis=0)
_, T = matrix_balance(moy, permute=False); Ti = np.linalg.inv(T)
A0 = [Ti@A@T for A in A0r]; N = A0[0].shape[0]
t = lambda r: 60.0/(N_TEETH*r)
def rho(r):
    tau = t(r); P = np.eye(N)
    for A in A0: P = expm(A*tau/M) @ P
    return float(np.max(np.abs(np.linalg.eigvals(P))))
print(f'{"centre":>8s} {"rho":>10s} {"bande VRAIE":>14s} {"bande CERTIFIEE":>17s}'
      f' {"ratio":>8s}')
CERT = {4900: 2.6, 4400: 21.9, 5200: 68.1, 5600: 84.0}
for r0, c in CERT.items():
    lo, hi = 0.0, 1200.0
    for _ in range(30):
        mid = 0.5*(lo+hi)
        ok = all(rho(r) < 1.0 for r in np.linspace(r0-mid, r0+mid, 21))
        lo, hi = (mid, hi) if ok else (lo, mid)
    vr = 0.5*(lo+hi)
    print(f'{r0:8d} {rho(r0):10.6f} {vr:12.1f} tr {c:15.1f} tr '
          f'{c/max(vr,1e-9):8.3f}')
