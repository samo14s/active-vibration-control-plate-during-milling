"""La fonctionnelle de Krasovskii discrete est-elle faisable a UN SEUL tau ?

La bande rendue est nulle et la marge vaut +1.68e-7 a TOUTES les largeurs,
largeur nulle comprise — or a largeur nulle le systeme est stable (rho=0.9397).
Un rejet a largeur nulle ne parle donc pas de l intervalle : il dit que la
CLASSE de fonctionnelle ne suffit pas.

On teste a un seul tau, en balayant m : si cela passe a m court et echoue a m
long, l obstruction est le conservatisme classique des fonctionnelles de
Krasovskii vis-a-vis des retards longs, et non un defaut de mise a l echelle.
On decompose aussi lambda_max par bloc pour voir QUI le pilote.
"""
import os, sys, time
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np, cvxpy as cp
from scipy.linalg import matrix_balance
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(R, 'paper_model'), os.path.join(R, 'control')]
import config as C
from plate_model import build_plate
from milling_dynamics import N_TEETH
from closed_loop import period_maps, spectral_radius
import stored_ctrl
NM, AP, R0 = 2, 0.60e-3, 5200.0
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()['vpa']
def mp_de(m):
    return period_maps(plate, R0, AP, 0.5*plate.lp, ctrl=ss, pd=pd,
                       n_modes=NM, m=m, coeff_mode='time',
                       coeff_scale=C.SIGN_SIM, ae=C.AE)[0]
print(f'{"m":>4s} {"rho":>10s} {"||B||/||A||":>12s} {"||C||/||A||":>12s} '
      f'{"faisable":>9s} {"marge":>11s} {"t [s]":>6s}')
for m in (4, 8, 16, 24, 40):
    mp = mp_de(m)
    N = mp[0][0].shape[0]
    rho = spectral_radius(mp, m, N)
    moy = np.mean([np.abs(a)+np.abs(b)+np.abs(c) for a, b, c in mp], axis=0)
    _, T = matrix_balance(moy, permute=False); Ti = np.linalg.inv(T)
    BL = [(Ti@a@T, Ti@b@T, Ti@c@T) for a, b, c in mp]
    rB = max(np.linalg.norm(b) for _, b, _ in BL)/max(
        np.linalg.norm(a) for a, _, _ in BL)
    rC = max(np.linalg.norm(c) for _, _, c in BL)/max(
        np.linalg.norm(a) for a, _, _ in BL)
    P = [cp.Variable((N, N), symmetric=True) for _ in range(m)]
    Q = cp.Variable((N, N), symmetric=True)
    Rv = cp.Variable((N, N), symmetric=True)
    g = cp.Variable(nonneg=True)
    cons = [p >> np.eye(N) for p in P] + [Q >> 1e-6*np.eye(N),
                                          Rv >> 1e-6*np.eye(N)]
    Z = np.zeros((N, N))
    for k, (A, B, Cm) in enumerate(BL):
        Pn = P[(k+1) % m]; ABC = np.hstack([A, B, Cm])
        D1 = cp.bmat([[P[k]-Q-Rv, Z, Z], [Z, Q, Z], [Z, Z, Rv]])
        cons.append(cp.bmat([[D1, ABC.T@Pn], [Pn@ABC, Pn]]) >> g*np.eye(4*N))
    pb = cp.Problem(cp.Maximize(g), cons)
    t0 = time.time()
    for slv in (cp.CLARABEL, cp.SCS):
        try:
            pb.solve(solver=slv, verbose=False)
            if P[0].value is not None: break
        except Exception: continue
    dt = time.time()-t0
    if P[0].value is None:
        print(f'{m:4d} {rho:10.6f} {rB:12.3e} {rC:12.3e} {"non":>9s} '
              f'{"-":>11s} {dt:6.1f}'); continue
    Pv = [np.asarray(p.value) for p in P]
    Qv, Rr = np.asarray(Q.value), np.asarray(Rv.value)
    pire = -np.inf
    for k, (A, B, Cm) in enumerate(BL):
        Pn = Pv[(k+1) % m]; ABC = np.hstack([A, B, Cm])
        M = ABC.T@Pn@ABC - np.block([[Pv[k]-Qv-Rr, Z, Z], [Z, Qv, Z],
                                     [Z, Z, Rr]])
        pire = max(pire, float(np.max(np.linalg.eigvalsh(0.5*(M+M.T)))))
    print(f'{m:4d} {rho:10.6f} {rB:12.3e} {rC:12.3e} '
          f'{str(pire < 0):>9s} {pire:+11.2e} {dt:6.1f}', flush=True)
