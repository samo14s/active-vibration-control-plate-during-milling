"""Enfermer D_k(tau) dans un polytope : le tester, pas le supposer.

D_k(tau) = exp(tau A0(s_k) h) est un ARC dans l espace des matrices, pas un
segment. Deux questions, dans l ordre :

  1. l arc tient-il dans la corde conv{D(tau_min), D(tau_max)} ? Si oui, deux
     sommets suffisent et la forme de Schur — [[P, D'P'],[P'D, P']] > 0, qui
     est AFFINE en D, contrairement a D'P'D - P < 0 qui est quadratique —
     transporte la faisabilite des sommets a tout l intervalle.
  2. sinon, de combien deborde-t-il ? C est ce debordement qu un polytope doit
     couvrir, et son cout se mesure.

On mesure aussi le RAYON DE ROBUSTESSE du certificat exact construit au centre :
avec W_k = I la marge vaut exactement 1, et

    (D+Delta)' P' (D+Delta) <= D'P'D + 2||P'^{1/2}D|| ||P'^{1/2}|| ||Delta||
                                     + ||P'^{1/2}||^2 ||Delta||^2

donne une borne sur ||Delta|| encore admissible. La demi-largeur d intervalle
certifiee s en deduit — c est l intervalle, obtenu sans SDP geant.
"""
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov, matrix_balance, sqrtm
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
A0r = [build_matrices(plate, DtD, Do, H, C.SIGN_SIM*a, ss, pd, NM)[0]
       for a in a4]
moy = np.mean([np.abs(A) for A in A0r], axis=0)
_, T = matrix_balance(moy, permute=False); Ti = np.linalg.inv(T)
A0 = [Ti@A@T for A in A0r]
N = A0[0].shape[0]
t = lambda r: 60.0/(N_TEETH*r)

def Dk(tau):
    return [expm(A*tau/M) for A in A0]

def phi(Ds):
    P = np.eye(N)
    for Dm in Ds: P = Dm @ P
    return P

print(f'  vpa, {NM} modes -> {N} etats, a_p = {AP*1e3:.2f} mm, m = {M}')
print(f'  ||tau A0/m|| au centre : '
      f'{max(np.linalg.norm(A*t(4900)/M) for A in A0):.4f}\n')

# ---- 1. l arc tient-il dans la corde ?
print('1. l arc D(tau) tient-il dans la corde des deux sommets ?')
for demi in (100.0, 300.0, 600.0):
    r0 = 4900.0
    ta, tb = t(r0+demi), t(r0-demi)          # tau decroit avec rpm
    Da, Db = Dk(ta), Dk(tb)
    pire = 0.0
    for lam in np.linspace(0.05, 0.95, 19):
        tau = (1-lam)*ta + lam*tb
        Dt = Dk(tau)
        for k in range(M):
            corde = (1-lam)*Da[k] + lam*Db[k]
            e = np.linalg.norm(Dt[k]-corde)/max(np.linalg.norm(Dt[k]), 1e-300)
            pire = max(pire, e)
    print(f'   +/-{demi:5.0f} tr/min : ecart relatif max arc-corde = {pire:.3e}')

# ---- 2. certificat exact au centre, puis rayon de robustesse
print('\n2. rayon de robustesse du certificat exact construit au centre')
def certificat(tau):
    Ds = Dk(tau); Phi = phi(Ds)
    Psi = np.eye(N); Ssum = np.eye(N)
    for k in range(1, M):
        Psi = Ds[k-1] @ Psi
        Ssum = Ssum + Psi.T @ Psi
    P0 = solve_discrete_lyapunov(Phi.T, Ssum)
    P = [None]*M; P[0] = P0
    for k in range(M-1, 0, -1):
        P[k] = Ds[k].T @ (P[k+1] if k+1 < M else P0) @ Ds[k] + np.eye(N)
    return P, Ds, float(np.max(np.abs(np.linalg.eigvals(Phi))))

for r0 in (4900.0, 5200.0):
    tau0 = t(r0)
    P, Ds, rho = certificat(tau0)
    # borne admissible sur ||Delta|| : 2 a b d + b^2 d^2 < 1
    dmin = np.inf
    for k in range(M):
        Pn = P[(k+1) % M]
        s = sqrtm(0.5*(Pn+Pn.T)).real
        a = np.linalg.norm(s @ Ds[k]); b = np.linalg.norm(s)
        # b^2 d^2 + 2 a b d - 1 = 0
        d = (-2*a*b + np.sqrt(4*a*a*b*b + 4*b*b))/(2*b*b)
        dmin = min(dmin, d)
    # quelle demi-largeur en tr/min garde ||Delta_k|| <= dmin ?
    lo, hi = 0.0, 800.0
    for _ in range(40):
        mid = 0.5*(lo+hi)
        dev = 0.0
        for rr in (r0-mid, r0+mid):
            Dv = Dk(t(rr))
            dev = max(dev, max(np.linalg.norm(Dv[k]-Ds[k]) for k in range(M)))
        lo, hi = (mid, hi) if dev <= dmin else (lo, mid)
    print(f'   {r0:.0f} tr/min : rho = {rho:.6f}   ||Delta|| admissible = '
          f'{dmin:.3e}   -> bande certifiee +/- {0.5*(lo+hi):.2f} tr/min')
