"""Deux leviers de plus, tous deux en algebre exacte.

(A) RECENTRAGE. Le certificat est bati au centre, mais la contrainte qui mord
    est au tau le PIRE de l intervalle. On itere : bande -> pire tau -> nouveau
    certificat -> bande. Point fixe en quelques tours.

(B) MOYENNE DES SOMMETS. P^a construite a tau_min, P^b a tau_max ; leur moyenne
    peut satisfaire les deux mieux que chacune. La forme de Schur etant affine
    en D et la contrainte convexe en P, c est un candidat naturel.
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
def cert(tau, frac):
    Ds = Dk(tau); rho = rho_de(Ds)
    al = frac*(1.0 - rho**(2.0/M)); f = 1.0/np.sqrt(1.0-al)
    Dt = [f*Dm for Dm in Ds]
    Phi = np.eye(N)
    for Dm in Dt: Phi = Dm @ Phi
    if np.max(np.abs(np.linalg.eigvals(Phi))) >= 1.0: return None
    Psi = np.eye(N); S = np.eye(N)
    for k in range(1, M):
        Psi = Dt[k-1] @ Psi; S = S + Psi.T @ Psi
    P0 = solve_discrete_lyapunov(Phi.T, S)
    P = [None]*M; P[0] = P0
    for k in range(M-1, 0, -1):
        P[k] = Dt[k].T @ (P[k+1] if k+1 < M else P0) @ Dt[k] + np.eye(N)
    return P
def pire(P, tau):
    Ds = Dk(tau); w = -np.inf
    for k in range(M):
        W = Ds[k].T @ P[(k+1) % M] @ Ds[k] - P[k]
        w = max(w, float(np.max(np.linalg.eigvalsh(0.5*(W+W.T)))))
    return w
def bande(P, r0):
    if P is None or pire(P, t(r0)) >= 0: return 0.0
    lo, hi = 0.0, 1500.0
    for _ in range(30):
        mid = 0.5*(lo+hi)
        lo, hi = ((mid, hi) if all(pire(P, t(r)) < 0 for r in (r0-mid, r0+mid))
                  else (lo, mid))
    return 0.5*(lo+hi)

VRAI = {4900: 12.9, 4400: 280.6, 5200: 312.9, 5600: 631.8}
OPT  = {4900: 3.5, 4400: 81.6, 5200: 117.0, 5600: 164.3}
FRAC = {4900: 0.5, 4400: 0.995, 5200: 0.995, 5600: 0.995}
print(f'{"centre":>7s} {"phase5":>8s} {"recentre":>9s} {"moyenne":>8s} '
      f'{"vraie":>8s} {"meilleur/vraie":>15s}')
for r0 in (5600, 5200, 4400, 4900):
    fr = FRAC[r0]
    # (A) recentrage iteratif sur le pire tau de la bande courante
    b, centre = OPT[r0], float(r0)
    for _ in range(4):
        cand = np.linspace(r0-b, r0+b, 9)
        pirer = max(cand, key=lambda r: rho_de(Dk(t(r))))
        P = cert(t(pirer), fr)
        nb = bande(P, r0)
        if nb <= b: break
        b, centre = nb, pirer
    brec = b
    # (B) moyenne des deux certificats de sommet
    Pa, Pb = cert(t(r0-OPT[r0]), fr), cert(t(r0+OPT[r0]), fr)
    bmoy = 0.0
    if Pa is not None and Pb is not None:
        na = max(np.linalg.norm(p) for p in Pa)
        nb_ = max(np.linalg.norm(p) for p in Pb)
        Pm = [0.5*(pa/na + pb/nb_) for pa, pb in zip(Pa, Pb)]
        bmoy = bande(Pm, r0)
    best = max(OPT[r0], brec, bmoy)
    print(f'{r0:7d} {OPT[r0]:8.1f} {brec:9.1f} {bmoy:8.1f} {VRAI[r0]:8.1f} '
          f'{best/VRAI[r0]:15.3f}', flush=True)
