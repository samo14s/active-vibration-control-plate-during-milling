"""PHASE 10-E — le meme rapport de conservatisme, sur le systeme PERIODIQUE.

Sur le systeme MOYENNE (phase 10-D) : beta_cert = 0.7955, beta_vrai = 2.3362,
rapport 2.9. On refait la mesure a l identique sur le systeme PERIODIQUE, au
meme point, avec le meme m. La difference entre les deux rapports EST la
reponse quantitative a la question de la phase 10 — bien meilleure que le
oui/non des essais precedents, qui ne disait rien quand les deux cotes
echouaient.

Deux precautions que les phases 7-8 n avaient pas :

  * P_k PERIODIQUE, soit la classe la plus GENERALE de la famille. Un echec
    ne peut donc pas etre impute a une fonctionnelle trop pauvre.
  * TEMOIN NEGATIF a beta = 0 : sans retard, le systeme periodique est celui
    que les phases 3-6 ont certifie a 99.8 % de son plafond. Si la LMI
    echoue LA, le montage est en cause et le script refuse de conclure au
    lieu de bissecter du bruit.
"""
import os, sys, time
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
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
M = int(os.environ.get('M', 24))
MODE = os.environ.get('MODE', 'time')
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()['vpa']

mp = period_maps(plate, R0, AP, 0.5 * plate.lp, ctrl=ss, pd=pd, n_modes=NM,
                 m=M, coeff_mode=MODE, coeff_scale=C.SIGN_SIM, ae=C.AE)[0]
moy = np.mean([np.abs(a) + np.abs(b) + np.abs(c) for a, b, c in mp], axis=0)
_, T = matrix_balance(moy, permute=False)
Ti = np.linalg.inv(T)
BL = [(Ti @ a @ T, Ti @ b @ T, Ti @ c @ T) for a, b, c in mp]
N = BL[0][0].shape[0]
I, Z0 = np.eye(N), np.zeros((N, N))
E1 = np.hstack([I, Z0, Z0]); F = np.hstack([I, -I, Z0])


def rho(beta):
    return spectral_radius([(a, beta*b, beta*c) for a, b, c in BL], M, N)


def certifie(beta):
    P = [cp.Variable((N, N), symmetric=True) for _ in range(M)]
    Q = cp.Variable((N, N), symmetric=True)
    Rv = cp.Variable((N, N), symmetric=True)
    Zv = cp.Variable((N, N), symmetric=True)
    e = 1e-5
    cons = [p >> I for p in P] + [Q >> e*I, Rv >> e*I, Zv >> e*I]
    for k in range(M):
        a, b, c = BL[k]
        ABC = np.hstack([a, beta*b, beta*c]); ETA = ABC - E1
        Pn = P[(k + 1) % M]
        haut = (cp.bmat([[P[k] - Q - Rv, Z0, Z0], [Z0, Q, Z0], [Z0, Z0, Rv]])
                + (1.0/M) * (F.T @ Zv @ F))
        cons.append(cp.bmat([
            [haut,           ABC.T @ Pn,       ETA.T @ (M * Zv)],
            [Pn @ ABC,       Pn,               np.zeros((N, N))],
            [(M * Zv) @ ETA, np.zeros((N, N)), M * Zv]])
            >> e * np.eye(5 * N))
    pb = cp.Problem(cp.Minimize(0), cons)
    for slv in (cp.CLARABEL, cp.SCS):
        try:
            pb.solve(solver=slv, verbose=False)
            if P[0].value is not None:
                break
        except Exception:
            continue
    if P[0].value is None:
        return False
    Pv = [np.asarray(p.value) for p in P]
    Qv, Rr, Zr = (np.asarray(Q.value), np.asarray(Rv.value),
                  np.asarray(Zv.value))
    pire = -np.inf
    for k in range(M):
        a, b, c = BL[k]
        ABC = np.hstack([a, beta*b, beta*c]); ETA = ABC - E1
        Pn = Pv[(k + 1) % M]
        Mm = (ABC.T @ Pn @ ABC
              - np.block([[Pv[k]-Qv-Rr, Z0, Z0], [Z0, Qv, Z0], [Z0, Z0, Rr]])
              + M * (ETA.T @ Zr @ ETA) - (1.0/M) * (F.T @ Zr @ F))
        pire = max(pire, float(np.max(np.linalg.eigvalsh(0.5*(Mm + Mm.T)))))
    return pire < 0


def bissection(pred, lo, hi, n, nom):
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        t0 = time.time()
        ok = pred(mid)
        (lo, hi) = (mid, hi) if ok else (lo, mid)
        print(f'    {nom} [{lo:.5f}, {hi:.5f}]  ({time.time()-t0:.0f} s)',
              flush=True)
    return 0.5 * (lo + hi)


print(f'  vpa, {NM} modes, a_p = {AP*1e3:.2f} mm, {R0:.0f} tr/min, '
      f'systeme {MODE}, m = {M}, N = {N}', flush=True)
print(f'  rho(0) = {rho(0.0):.6f}   rho(1) = {rho(1.0):.6f}\n', flush=True)

print('  0) TEMOIN NEGATIF : beta = 0, aucun retard', flush=True)
t0 = time.time()
if not certifie(0.0):
    print(f'     ECHEC a beta = 0 ({time.time()-t0:.0f} s). Le montage est en '
          f'cause, pas le systeme. Aucune conclusion.', flush=True)
    sys.exit(0)
print(f'     certifie ({time.time()-t0:.0f} s) — le montage tient.\n',
      flush=True)

print('  1) beta_vrai : beta ou rho = 1', flush=True)
hi = 1.0
while rho(hi) < 1.0 and hi < 4096.0:
    hi *= 2.0
    print(f'    rho({hi:.0f}) = {rho(hi):.6f}', flush=True)
b_vrai = (float('inf') if rho(hi) < 1.0
          else bissection(lambda b: rho(b) < 1.0, hi/2.0, hi, 14, 'rho=1 :'))
print(f'    beta_vrai = {b_vrai}\n', flush=True)

print('  2) beta_cert : dernier beta certifie', flush=True)
hi_c = 1.0 if not certifie(1.0) else 4.0
b_cert = bissection(certifie, 0.0, hi_c, 10, 'cert  :')

print('\n  ================ RESULTAT ================')
print(f'  systeme   = {MODE}')
print(f'  beta_cert = {b_cert:.4f}')
print(f'  beta_vrai = {b_vrai}')
if np.isfinite(b_vrai):
    print(f'  rapport   = {b_vrai / b_cert:.2f}')
print('  (moyenne, phase 10-D : 0.7955 / 2.3362 = 2.94)')
