"""Quelle LARGEUR d intervalle un seul certificat de Floquet couvre-t-il ?

Le rayon par inegalite triangulaire donnait +/- 0.08 tr/min : c est la borne
qui est mauvaise, pas le certificat. cond(P) ~ 5e10 fait exploser ||P^{1/2}||
et la majoration par les normes perd des ordres de grandeur.

Route directe et bien moins conservatrice :
  * la forme de Schur  [[P_k, D'P_{k+1}],[P_{k+1}D, P_{k+1}]] > 0  est AFFINE
    en D. Donc si elle tient AUX DEUX SOMMETS avec le MEME P, elle tient sur
    toute l enveloppe convexe conv{D(tau_min), D(tau_max)} ;
  * l arc D(tau) n est pas exactement dans cette enveloppe, mais on a mesure
    l ecart : 7.8e-5 a +/-100 tr/min, 2.8e-3 a +/-600. Il reste a verifier que
    la MARGE aux sommets couvre cet ecart.

On evalue donc, pour un P construit exactement au centre, la quantite
    max_k lambda_max(D_k(tau)' P_{k+1} D_k(tau) - P_k)
aux deux sommets, et on bissecte la plus grande demi-largeur certifiee.
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
A0r = [build_matrices(plate, DtD, Do, H, C.SIGN_SIM*a, ss, pd, NM)[0]
       for a in a4]
moy = np.mean([np.abs(A) for A in A0r], axis=0)
_, T = matrix_balance(moy, permute=False); Ti = np.linalg.inv(T)
A0 = [Ti@A@T for A in A0r]
N = A0[0].shape[0]
t = lambda r: 60.0/(N_TEETH*r)
Dk = lambda tau: [expm(A*tau/M) for A in A0]

def certificat(tau):
    Ds = Dk(tau)
    Phi = np.eye(N)
    for Dm in Ds: Phi = Dm @ Phi
    Psi = np.eye(N); Ssum = np.eye(N)
    for k in range(1, M):
        Psi = Ds[k-1] @ Psi; Ssum = Ssum + Psi.T @ Psi
    P0 = solve_discrete_lyapunov(Phi.T, Ssum)
    P = [None]*M; P[0] = P0
    for k in range(M-1, 0, -1):
        P[k] = Ds[k].T @ (P[k+1] if k+1 < M else P0) @ Ds[k] + np.eye(N)
    return P, float(np.max(np.abs(np.linalg.eigvals(Phi))))

def pire(P, tau):
    """max_k lambda_max(D' P_{k+1} D - P_k), relatif a ||P||."""
    Ds = Dk(tau); w = -np.inf
    for k in range(M):
        W = Ds[k].T @ P[(k+1) % M] @ Ds[k] - P[k]
        w = max(w, float(np.max(np.linalg.eigvalsh(0.5*(W+W.T)))))
    return w

print(f'  vpa, {N} etats, a_p = {AP*1e3:.2f} mm, m = {M}\n')
print(f'{"centre":>8s} {"rho":>10s} {"demi-largeur certifiee":>24s} '
      f'{"marge au sommet":>17s}')
for r0 in (4900.0, 5200.0, 5600.0, 4400.0):
    P, rho = certificat(t(r0))
    if pire(P, t(r0)) >= 0:
        print(f'{r0:8.0f} {rho:10.6f}  certificat central invalide'); continue
    lo, hi = 0.0, 1500.0
    for _ in range(34):
        mid = 0.5*(lo+hi)
        ok = all(pire(P, t(r)) < 0 for r in (r0-mid, r0+mid))
        lo, hi = (mid, hi) if ok else (lo, mid)
    dw = 0.5*(lo+hi)
    mg = max(pire(P, t(r0-dw)), pire(P, t(r0+dw)))
    print(f'{r0:8.0f} {rho:10.6f} {dw:22.1f} tr/min {mg:17.3e}')
