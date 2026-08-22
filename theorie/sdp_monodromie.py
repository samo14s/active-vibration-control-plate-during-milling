"""SDP conjointe — mais au niveau de la MONODROMIE, pas des 200 pas.

La sequence periodique P_0..P_{m-1} n est pas necessaire : sur le systeme
echantillonne periode par periode, la stabilite de Floquet equivaut a

    exists P0 > 0 :  Phi(tau)' P0 Phi(tau) < P0

soit 55 inconnues et deux contraintes au lieu de 200 matrices et 400 contraintes.
La forme de Schur  [[P0, Phi'P0],[P0 Phi, P0]] > 0  est AFFINE en Phi, donc si
elle tient aux deux sommets avec le MEME P0, elle tient sur conv{Phi_a, Phi_b}.

Reste la question qui decide de tout : l arc Phi(tau) est-il pres de sa corde ?
Pour D_k l ecart valait 2.8e-3 a +/-600 tr/min, mais Phi est un produit de 200
exponentielles — rien ne garantit qu il herite de cette douceur. On le MESURE
avant de s en servir.

Puis on bissecte la plus grande demi-largeur pour laquelle la SDP conjointe est
faisable, et l on VERIFIE le P0 rendu a l interieur, independamment du solveur.
"""
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np, cvxpy as cp
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
def Phi(tau):
    P = np.eye(N)
    for A in A0: P = expm(A*tau/M) @ P
    return P
rho = lambda tau: float(np.max(np.abs(np.linalg.eigvals(Phi(tau)))))

print('1. l arc Phi(tau) est-il pres de sa corde ?')
for r0, dw in ((5600, 300), (5600, 600), (5200, 150), (4400, 100)):
    Pa, Pb = Phi(t(r0-dw)), Phi(t(r0+dw))
    pire = 0.0
    for lam in np.linspace(0.05, 0.95, 19):
        tau = (1-lam)*t(r0-dw) + lam*t(r0+dw)
        Pt = Phi(tau); corde = (1-lam)*Pa + lam*Pb
        pire = max(pire, np.linalg.norm(Pt-corde)/np.linalg.norm(Pt))
    print(f'   {r0} +/- {dw:3d} tr/min : ecart relatif max = {pire:.3e}')

def faisable(r0, dw, verif=True):
    """SDP : un meme P0 pour les deux sommets. Rend (ok, marge interieure)."""
    Pa, Pb = Phi(t(r0-dw)), Phi(t(r0+dw))
    P0 = cp.Variable((N, N), symmetric=True)
    cons = [P0 >> np.eye(N)]
    for F in (Pa, Pb):
        cons.append(cp.bmat([[P0, F.T@P0], [P0@F, P0]]) >> 1e-9*np.eye(2*N))
    pb = cp.Problem(cp.Minimize(cp.trace(P0)), cons)
    for slv in (cp.CLARABEL, cp.SCS):
        try:
            pb.solve(solver=slv, verbose=False)
            if P0.value is not None: break
        except Exception: continue
    if P0.value is None: return False, None
    Pv = np.asarray(P0.value); Pv = 0.5*(Pv+Pv.T)
    if np.min(np.linalg.eigvalsh(Pv)) <= 0: return False, None
    if not verif: return True, None
    pire = -np.inf
    for r in np.linspace(r0-dw, r0+dw, 41):
        F = Phi(t(r)); W = F.T@Pv@F - Pv
        pire = max(pire, float(np.max(np.linalg.eigvalsh(0.5*(W+W.T)))))
    return pire < 0, pire

print('\n2. plus grande demi-largeur admise par la SDP conjointe')
print(f'{"centre":>7s} {"phase5":>8s} {"SDP conjointe":>14s} {"vraie":>8s} '
      f'{"ratio":>7s} {"marge int.":>12s}')
OPT = {5600: 164.3, 5200: 117.0, 4400: 81.6, 4900: 3.5}
VRAI = {5600: 631.8, 5200: 312.9, 4400: 280.6, 4900: 12.9}
for r0 in (5600, 5200, 4400, 4900):
    lo, hi = 0.0, VRAI[r0]*1.05
    for _ in range(14):
        mid = 0.5*(lo+hi)
        ok, _ = faisable(r0, mid)
        lo, hi = (mid, hi) if ok else (lo, mid)
    dw = lo
    _, mg = faisable(r0, dw)
    print(f'{r0:7d} {OPT[r0]:8.1f} {dw:12.1f} tr {VRAI[r0]:8.1f} '
          f'{dw/VRAI[r0]:7.3f} {("%+.2e" % mg) if mg is not None else "-":>12s}',
          flush=True)
