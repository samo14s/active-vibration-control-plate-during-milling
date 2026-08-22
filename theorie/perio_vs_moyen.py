"""PHASE 10-A — version economique de la QUESTION A.

Le script bessel_legendre.py repond a la meme question mais paie m
contraintes LMI des deux cotes ; a m = 8 il a deja coute ~20 min par
colonne. Or du cote MOYENNE le systeme est INVARIANT : une seule matrice
P constante suffit, donc UNE seule LMI au lieu de m.

Et ce desequilibre joue CONTRE la these, ce qui est exactement ce qu il
faut :

  * cote moyenne  : classe RESTREINTE  (P constante)
  * cote periodique : classe GENERALE  (P_k periodique, qui contient la
                      precedente)

Si la classe generale echoue la ou la classe restreinte reussit, la cause
ne peut pas etre la richesse de la fonctionnelle. Elle est dans le systeme.

Meme fonctionnelle dependante du retard qu en phase 8 (Jensen), meme point
de fonctionnement, meme retard, et rho identique a 2e-5 pres.
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
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()['vpa']


def maps(r, m, mode):
    return period_maps(plate, r, AP, 0.5 * plate.lp, ctrl=ss, pd=pd,
                       n_modes=NM, m=m, coeff_mode=mode,
                       coeff_scale=C.SIGN_SIM, ae=C.AE)[0]


def equilibre(mp):
    moy = np.mean([np.abs(a) + np.abs(b) + np.abs(c) for a, b, c in mp], axis=0)
    _, T = matrix_balance(moy, permute=False)
    Ti = np.linalg.inv(T)
    return [(Ti @ a @ T, Ti @ b @ T, Ti @ c @ T) for a, b, c in mp]


def essai(m, r, mode):
    BL = equilibre(maps(r, m, mode))
    N = BL[0][0].shape[0]
    I, Z0 = np.eye(N), np.zeros((N, N))
    E1 = np.hstack([I, Z0, Z0])
    F = np.hstack([I, -I, Z0])
    invariant = (mode != 'time')
    # Cote moyenne : les m pas sont identiques -> une seule LMI, P constante.
    pas = [0] if invariant else list(range(m))
    nP = 1 if invariant else m
    P = [cp.Variable((N, N), symmetric=True) for _ in range(nP)]
    Q = cp.Variable((N, N), symmetric=True)
    Rv = cp.Variable((N, N), symmetric=True)
    Zv = cp.Variable((N, N), symmetric=True)
    e = 1e-5
    cons = [p >> I for p in P] + [Q >> e*I, Rv >> e*I, Zv >> e*I]
    for k in pas:
        A, B, Cm = BL[k]
        Pk = P[0] if invariant else P[k]
        Pn = P[0] if invariant else P[(k + 1) % m]
        ABC = np.hstack([A, B, Cm]); ETA = ABC - E1
        haut = (cp.bmat([[Pk - Q - Rv, Z0, Z0], [Z0, Q, Z0], [Z0, Z0, Rv]])
                + (1.0/m) * (F.T @ Zv @ F))
        cons.append(cp.bmat([
            [haut,           ABC.T @ Pn,        ETA.T @ (m * Zv)],
            [Pn @ ABC,       Pn,                np.zeros((N, N))],
            [(m * Zv) @ ETA, np.zeros((N, N)),  m * Zv]]) >> e * np.eye(5*N))
    pb = cp.Problem(cp.Minimize(0), cons)
    t0 = time.time()
    for slv in (cp.CLARABEL, cp.SCS):
        try:
            pb.solve(solver=slv, verbose=False)
            if P[0].value is not None:
                break
        except Exception:
            continue
    dt = time.time() - t0
    if P[0].value is None:
        return None, dt, N
    Pv = [np.asarray(p.value) for p in P]
    Qv, Rr, Zr = (np.asarray(Q.value), np.asarray(Rv.value),
                  np.asarray(Zv.value))
    # Verification hors solveur, sur TOUS les pas meme du cote invariant.
    pire = -np.inf
    for k in range(m):
        A, B, Cm = BL[k]
        Pk = Pv[0] if invariant else Pv[k]
        Pn = Pv[0] if invariant else Pv[(k + 1) % m]
        ABC = np.hstack([A, B, Cm]); ETA = ABC - E1
        M = (ABC.T @ Pn @ ABC
             - np.block([[Pk - Qv - Rr, Z0, Z0], [Z0, Qv, Z0], [Z0, Z0, Rr]])
             + m * (ETA.T @ Zr @ ETA) - (1.0/m) * (F.T @ Zr @ F))
        pire = max(pire, float(np.max(np.linalg.eigvalsh(0.5 * (M + M.T)))))
    return pire, dt, N


def dis(x):
    if x is None:
        return 'infaisable'
    return f'FAISABLE ({x:+.1e})' if x < 0 else f'non ({x:+.1e})'


mp = maps(R0, 8, 'time')
print(f'  vpa, {NM} modes, a_p = {AP*1e3:.2f} mm, {R0:.0f} tr/min, '
      f'N = {mp[0][0].shape[0]}', flush=True)
for mode, nom in (('time', 'periodique'), ('moyen', 'moyennee')):
    mm = maps(R0, 200, mode)
    print(f'  rho ({nom}, m=200) = '
          f'{spectral_radius(mm, 200, mm[0][0].shape[0]):.6f}', flush=True)
print(f'\n{"m":>4s} {"periodique (P_k, general)":>26s} '
      f'{"MOYENNEE (P const, restreint)":>30s} {"t_per":>7s} {"t_moy":>7s}')
for m in (16, 24, 40):
    pm_, tm, N = essai(m, R0, 'moyen')
    pp_, tp, _ = essai(m, R0, 'time')
    print(f'{m:4d} {dis(pp_):>26s} {dis(pm_):>30s} {tp:7.1f} {tm:7.1f}',
          flush=True)
