"""SDP conjointe PAR PAS — la seule ou la convexite est valide.

Mesure de vsdp.py : l ecart arc-corde vaut 2.8e-3 pour D_k et 0.35 a 1.06 pour
Phi. La convexite survit au niveau du PAS et meurt au niveau du TOUR. C est
donc sur les m pas qu il faut poser la SDP conjointe :

    trouver P_0..P_{m-1} > 0 tels que, AUX DEUX SOMMETS tau_a, tau_b,
        [[P_k , D_k(tau)' P_{k+1}],[P_{k+1} D_k(tau), P_{k+1}]] > 0
    (forme de Schur, AFFINE en D_k) ,  P_m = P_0

et la faisabilite s etend alors a conv{D_k(tau_a), D_k(tau_b)}, qui contient
l arc a 2.8e-3 pres.

Cout : m matrices symetriques N x N et 2m contraintes de taille 2N. A m = 200
cela fait 11000 inconnues et 400 contraintes 20x20. On sonde d abord a m reduit
pour savoir si la classe de probleme est seulement soluble ici — et l on DIT
que m reduit change la discretisation, donc le systeme certifie.
"""
import os, sys, time
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
NM, AP = 2, 0.60e-3
MM = int(os.environ.get('MM', 40))
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()['vpa']
D = plate.D_row(0.5*plate.lp, plate.hp)[:NM]; DtD = np.outer(D, D)
Do = plate.D_row(plate.lp, plate.hp)[:NM]
H = np.asarray(plate.H_Pe_modal, float)[:NM]
_, a4 = alpha4_series(4900, AP, plate.hp, MM)
A0r = [build_matrices(plate, DtD, Do, H, C.SIGN_SIM*a, ss, pd, NM)[0] for a in a4]
moy = np.mean([np.abs(A) for A in A0r], axis=0)
_, T = matrix_balance(moy, permute=False); Ti = np.linalg.inv(T)
A0 = [Ti@A@T for A in A0r]; N = A0[0].shape[0]
t = lambda r: 60.0/(N_TEETH*r)
Dk = lambda tau: [expm(A*tau/MM) for A in A0]
def rho(tau):
    P = np.eye(N)
    for Dm in Dk(tau): P = Dm @ P
    return float(np.max(np.abs(np.linalg.eigvals(P))))
print(f'  vpa, {N} etats, m = {MM} (reduit) ; {MM} inconnues N x N, '
      f'{2*MM} contraintes {2*N}x{2*N}')
print(f'  rho a 5600 tr/min avec m = {MM} : {rho(t(5600)):.6f}'
      f'   (m = 200 : 0.943864)\n')

def faisable(r0, dw):
    Da, Db = Dk(t(r0-dw)), Dk(t(r0+dw))
    P = [cp.Variable((N, N), symmetric=True) for _ in range(MM)]
    cons = [p >> np.eye(N) for p in P]
    for Ds in (Da, Db):
        for k in range(MM):
            Pn = P[(k+1) % MM]
            cons.append(cp.bmat([[P[k], Ds[k].T@Pn], [Pn@Ds[k], Pn]])
                        >> 1e-7*np.eye(2*N))
    pb = cp.Problem(cp.Minimize(0), cons)
    t0 = time.time()
    for slv in (cp.CLARABEL, cp.SCS):
        try:
            pb.solve(solver=slv, verbose=False)
            if P[0].value is not None: break
        except Exception: continue
    dt = time.time()-t0
    if P[0].value is None: return False, None, dt
    Pv = [0.5*(np.asarray(p.value)+np.asarray(p.value).T) for p in P]
    if min(np.min(np.linalg.eigvalsh(p)) for p in Pv) <= 0:
        return False, None, dt
    pire = -np.inf
    for r in np.linspace(r0-dw, r0+dw, 21):
        Ds = Dk(t(r))
        for k in range(MM):
            W = Ds[k].T @ Pv[(k+1) % MM] @ Ds[k] - Pv[k]
            pire = max(pire, float(np.max(np.linalg.eigvalsh(0.5*(W+W.T)))))
    return pire < 0, pire, dt

VRAI = {5600: 631.8, 5200: 312.9, 4400: 280.6, 4900: 12.9}
PH5  = {5600: 164.3, 5200: 117.0, 4400: 81.6, 4900: 3.5}
print(f'{"centre":>7s} {"phase5":>8s} {"SDP par pas":>12s} {"vraie":>8s} '
      f'{"ratio":>7s} {"marge int.":>12s} {"t [s]":>7s}')
for r0 in (5600, 5200, 4400, 4900):
    t0 = time.time()
    lo, hi = 0.0, VRAI[r0]*1.02
    for _ in range(11):
        mid = 0.5*(lo+hi)
        ok, _, _ = faisable(r0, mid)
        lo, hi = (mid, hi) if ok else (lo, mid)
    ok, mg, _ = faisable(r0, lo)
    print(f'{r0:7d} {PH5[r0]:8.1f} {lo:10.1f} tr {VRAI[r0]:8.1f} '
          f'{lo/VRAI[r0]:7.3f} {("%+.2e" % mg) if mg is not None else "-":>12s} '
          f'{time.time()-t0:7.0f}', flush=True)
