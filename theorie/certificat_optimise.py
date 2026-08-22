"""Elargir la bande certifiee : marge RELATIVE au lieu d absolue.

Le certificat de la phase 4 vient de  D' P_{k+1} D - P_k = -I : une marge
ABSOLUE de 1 face a ||P|| ~ 5e10, soit une marge relative de 2e-11. D ou une
bande a 8-22 % de la vraie.

On demande plutot un taux de decroissance RELATIF :

    D_k' P_{k+1} D_k  <=  (1 - alpha) P_k

Poser D~ = D/sqrt(1-alpha) ramene cela a la stabilite du systeme mis a
l echelle, donc

    alpha admissible  <=>  rho(Phi)/(1-alpha)^{m/2} < 1  <=>  alpha < 1 - rho^{2/m}

et alpha* = 1 - rho^{2/m} est EXACT, pas estime. Pour alpha < alpha*, la meme
recurrence exacte sur D~ donne P_k verifiant

    D' P_{k+1} D - P_k = -alpha P_k - (1-alpha) I  <=  -alpha P_k

soit une marge PROPORTIONNELLE a P. Reste a choisir alpha : trop petit, la
marge est faible ; trop grand, P explose (systeme quasi marginal). L optimum
se balaie — et tout reste en algebre lineaire, sans solveur.
"""
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov, matrix_balance
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
Dk = lambda tau: [expm(A*tau/M) for A in A0]

def rho_de(Ds):
    P = np.eye(N)
    for Dm in Ds: P = Dm @ P
    return float(np.max(np.abs(np.linalg.eigvals(P))))

def cert(tau, alpha):
    """P_k tel que D' P_{k+1} D - P_k <= -alpha P_k, par recurrence exacte."""
    Ds = Dk(tau); f = 1.0/np.sqrt(1.0 - alpha)
    Dt = [f*Dm for Dm in Ds]
    Phi = np.eye(N)
    for Dm in Dt: Phi = Dm @ Phi
    if np.max(np.abs(np.linalg.eigvals(Phi))) >= 1.0: return None, Ds
    Psi = np.eye(N); S = np.eye(N)
    for k in range(1, M):
        Psi = Dt[k-1] @ Psi; S = S + Psi.T @ Psi
    P0 = solve_discrete_lyapunov(Phi.T, S)
    P = [None]*M; P[0] = P0
    for k in range(M-1, 0, -1):
        P[k] = Dt[k].T @ (P[k+1] if k+1 < M else P0) @ Dt[k] + np.eye(N)
    return P, Ds

def pire(P, tau):
    Ds = Dk(tau); w = -np.inf
    for k in range(M):
        W = Ds[k].T @ P[(k+1) % M] @ Ds[k] - P[k]
        w = max(w, float(np.max(np.linalg.eigvalsh(0.5*(W+W.T)))))
    return w

def bande(P, r0):
    lo, hi = 0.0, 1500.0
    for _ in range(32):
        mid = 0.5*(lo+hi)
        ok = all(pire(P, t(r)) < 0 for r in (r0-mid, r0+mid))
        lo, hi = (mid, hi) if ok else (lo, mid)
    return 0.5*(lo+hi)

VRAI = {4900: 12.9, 4400: 280.6, 5200: 312.9, 5600: 631.8}
BASE = {4900: 2.6, 4400: 21.9, 5200: 68.1, 5600: 84.0}
print(f'  vpa, {N} etats, a_p = {AP*1e3:.2f} mm, m = {M}\n')
for r0 in (5600, 5200, 4400, 4900):
    rho = rho_de(Dk(t(r0)))
    astar = 1.0 - rho**(2.0/M)
    print(f'  centre {r0} tr/min : rho = {rho:.6f}   alpha* = {astar:.4e}')
    best = (0.0, None)
    for frac in (0.0, 0.25, 0.5, 0.75, 0.90, 0.97, 0.995):
        al = frac*astar
        P, _ = cert(t(r0), al)
        if P is None: print(f'      frac={frac:5.3f} : infaisable'); continue
        b = bande(P, r0)
        cond = max(np.linalg.cond(p) for p in P)
        print(f'      alpha = {frac:5.3f} alpha*  ->  bande +/- {b:7.1f} tr/min'
              f'   cond(P) = {cond:.2e}')
        if b > best[0]: best = (b, frac)
    print(f'    -> meilleure bande {best[0]:.1f} tr/min a frac={best[1]}   '
          f'(phase 4 : {BASE[r0]}, vraie : {VRAI[r0]}, '
          f'ratio {best[0]/VRAI[r0]:.3f})\n', flush=True)
