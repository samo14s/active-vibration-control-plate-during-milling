"""Krasovskii DEPENDANTE DU RETARD — discrete, avec inegalite de Jensen.

La phase 7 a montre que la fonctionnelle a Q, R constantes ne voit que les
TAILLES de B et C, et que la stabilite du fraisage depend de leur PHASE. Le
remede classique est le terme de differences double :

    W = sum_{theta=-m}^{-1} sum_{j=k+theta}^{k-1} eta_j' Z eta_j ,
        eta_j = x_{j+1} - x_j

dont la difference vaut exactement

    dW = m eta_k' Z eta_k - sum_{j=k-m}^{k-1} eta_j' Z eta_j

et dont la somme se replie par JENSEN discrete :

    sum_{j=k-m}^{k-1} eta_j' Z eta_j  >=  (1/m) (x_k - x_{k-m})' Z (x_k - x_{k-m})

C est CE terme qui rend le critere dependant du retard : il relie l etat
retarde a l etat courant par les increments accumules, et non par leur seule
norme.

Avec xi = [x_k ; x_{k-m} ; x_{k-m+1}],  ABC = [A B C],  E1 = [I 0 0],
F = [I -I 0] :

    M = ABC' P' ABC + m (ABC-E1)' Z (ABC-E1)
        - diag(P_k - Q - R, Q, R) - (1/m) F' Z F  < 0

Deux formes quadratiques en ABC, donc un complement de Schur sur
diag(P', m Z) : la LMI reste AFFINE en (A, B, C), et la convexite sur
l intervalle de tau tient comme aux phases 4-6.

Test decisif : faisabilite a UN SEUL tau, la ou la version independante du
retard echouait alors que rho = 0.9397 < 1.
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
from closed_loop import period_maps, spectral_radius
import stored_ctrl

NM, AP = 2, 0.60e-3
R0 = float(os.environ.get('R0', 5200))
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()['vpa']

def maps(r, m):
    return period_maps(plate, r, AP, 0.5*plate.lp, ctrl=ss, pd=pd, n_modes=NM,
                       m=m, coeff_mode='time', coeff_scale=C.SIGN_SIM,
                       ae=C.AE)[0]

def essai(m, r, dd=True):
    mp = maps(r, m); N = mp[0][0].shape[0]
    moy = np.mean([np.abs(a)+np.abs(b)+np.abs(c) for a, b, c in mp], axis=0)
    _, T = matrix_balance(moy, permute=False); Ti = np.linalg.inv(T)
    BL = [(Ti@a@T, Ti@b@T, Ti@c@T) for a, b, c in mp]
    I, Zz = np.eye(N), np.zeros((N, N))
    E1 = np.hstack([I, Zz, Zz]); F = np.hstack([I, -I, Zz])
    P = [cp.Variable((N, N), symmetric=True) for _ in range(m)]
    Q = cp.Variable((N, N), symmetric=True)
    Rv = cp.Variable((N, N), symmetric=True)
    Zv = cp.Variable((N, N), symmetric=True)
    e = 1e-5
    cons = [p >> I for p in P] + [Q >> e*I, Rv >> e*I, Zv >> e*I]
    for k, (A, B, Cm) in enumerate(BL):
        Pn = P[(k+1) % m]
        ABC = np.hstack([A, B, Cm]); ETA = ABC - E1
        D1 = cp.bmat([[P[k]-Q-Rv, Zz, Zz], [Zz, Q, Zz], [Zz, Zz, Rv]])
        if dd:
            haut = D1 + (1.0/m)*(F.T@Zv@F)
            cons.append(cp.bmat([
                [haut,        ABC.T@Pn,   ETA.T@(m*Zv)],
                [Pn@ABC,      Pn,         np.zeros((N, N))],
                [(m*Zv)@ETA,  np.zeros((N, N)), m*Zv]]) >> e*np.eye(5*N))
        else:
            cons.append(cp.bmat([[D1, ABC.T@Pn], [Pn@ABC, Pn]]) >> e*np.eye(4*N))
    pb = cp.Problem(cp.Minimize(0), cons)
    t0 = time.time()
    for slv in (cp.CLARABEL, cp.SCS):
        try:
            pb.solve(solver=slv, verbose=False)
            if P[0].value is not None: break
        except Exception: continue
    dt = time.time()-t0
    if P[0].value is None: return None, dt, N
    Pv = [np.asarray(p.value) for p in P]
    Qv, Rr, Zr = (np.asarray(Q.value), np.asarray(Rv.value),
                  np.asarray(Zv.value))
    pire = -np.inf
    for k, (A, B, Cm) in enumerate(BL):
        Pn = Pv[(k+1) % m]
        ABC = np.hstack([A, B, Cm]); ETA = ABC - E1
        M = ABC.T@Pn@ABC - np.block([[Pv[k]-Qv-Rr, Zz, Zz], [Zz, Qv, Zz],
                                     [Zz, Zz, Rr]])
        if dd:
            M = M + m*(ETA.T@Zr@ETA) - (1.0/m)*(F.T@Zr@F)
        pire = max(pire, float(np.max(np.linalg.eigvalsh(0.5*(M+M.T)))))
    return pire, dt, N

mp2 = maps(R0, 200)
print(f'  vpa, {NM} modes, a_p = {AP*1e3:.2f} mm, {R0:.0f} tr/min')
print(f'  rho (m=200) = {spectral_radius(mp2, 200, mp2[0][0].shape[0]):.6f}\n')
print(f'{"m":>4s} {"independante du retard":>24s} {"DEPENDANTE du retard":>22s} '
      f'{"t [s]":>7s}')
for m in (8, 16, 24, 40):
    pi, _, N = essai(m, R0, dd=False)
    pd_, dt, _ = essai(m, R0, dd=True)
    f = lambda x: 'infaisable' if x is None else (
        f'FAISABLE ({x:+.1e})' if x < 0 else f'non ({x:+.1e})')
    print(f'{m:4d} {f(pi):>24s} {f(pd_):>22s} {dt:7.1f}', flush=True)
