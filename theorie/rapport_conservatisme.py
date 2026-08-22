"""PHASE 10-D — LE RAPPORT DE CONSERVATISME, EN UN SEUL NOMBRE.

Le temoin positif (phase 10-C) a etabli deux choses :

  * la fonctionnelle (P, Q, R, Z) FONCTIONNE — elle certifie a beta = 0 et
    jusqu a beta = 0.70, et meurt entre 0.70 et 0.85 ;
  * rho ne bouge PAS sur tout l intervalle : 0.939672 a beta = 0 contre
    0.939671 a beta = 1, soit 1e-6.

Donc le certificat ne meurt pas parce que le systeme s approche de
l instabilite. Il meurt alors que rho reste a 0.94.

On transforme cela en UNE mesure. Deux seuils sur le meme parametre :

    beta_cert : dernier beta ou la LMI passe encore
    beta_vrai : beta ou rho(beta) atteint 1  (instabilite reelle)

et le rapport beta_vrai / beta_cert dit, en un nombre, quelle part de la
regeneration reelle cette classe de fonctionnelles laisse sur la table sur
ce probleme. C est la version QUANTIFIEE de « the application of a Lyapunov
approach leads to conservative results » (van Dijk et al. 2014), qui n est
chez eux qu une justification en une phrase.

On mesure les deux par bissection, sur le systeme MOYENNE (invariant) —
celui-la meme sur lequel Shadkami et al. 2025 travaillent. Le cas periodique
suit dans un second temps s il est calculable.
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
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()['vpa']

mp = period_maps(plate, R0, AP, 0.5 * plate.lp, ctrl=ss, pd=pd, n_modes=NM,
                 m=M, coeff_mode='moyen', coeff_scale=C.SIGN_SIM, ae=C.AE)[0]
moy = np.mean([np.abs(a) + np.abs(b) + np.abs(c) for a, b, c in mp], axis=0)
_, T = matrix_balance(moy, permute=False)
Ti = np.linalg.inv(T)
A, B, Cm = [Ti @ x @ T for x in mp[0]]
N = A.shape[0]
I, Z0 = np.eye(N), np.zeros((N, N))
E1 = np.hstack([I, Z0, Z0]); F = np.hstack([I, -I, Z0])


def rho(beta):
    return spectral_radius([(A, beta*B, beta*Cm)] * M, M, N)


def certifie(beta):
    Bb, Cb = beta * B, beta * Cm
    ABC = np.hstack([A, Bb, Cb]); ETA = ABC - E1
    P = cp.Variable((N, N), symmetric=True)
    Q = cp.Variable((N, N), symmetric=True)
    Rv = cp.Variable((N, N), symmetric=True)
    Zv = cp.Variable((N, N), symmetric=True)
    e = 1e-5
    haut = (cp.bmat([[P - Q - Rv, Z0, Z0], [Z0, Q, Z0], [Z0, Z0, Rv]])
            + (1.0/M) * (F.T @ Zv @ F))
    cons = [P >> I, Q >> e*I, Rv >> e*I, Zv >> e*I,
            cp.bmat([[haut,           ABC.T @ P,        ETA.T @ (M * Zv)],
                     [P @ ABC,        P,                np.zeros((N, N))],
                     [(M * Zv) @ ETA, np.zeros((N, N)), M * Zv]])
            >> e * np.eye(5 * N)]
    pb = cp.Problem(cp.Minimize(0), cons)
    for slv in (cp.CLARABEL, cp.SCS):
        try:
            pb.solve(solver=slv, verbose=False)
            if P.value is not None:
                break
        except Exception:
            continue
    if P.value is None:
        return False
    Pv, Qv = np.asarray(P.value), np.asarray(Q.value)
    Rr, Zr = np.asarray(Rv.value), np.asarray(Zv.value)
    Mm = (ABC.T @ Pv @ ABC
          - np.block([[Pv - Qv - Rr, Z0, Z0], [Z0, Qv, Z0], [Z0, Z0, Rr]])
          + M * (ETA.T @ Zr @ ETA) - (1.0/M) * (F.T @ Zr @ F))
    return float(np.max(np.linalg.eigvalsh(0.5 * (Mm + Mm.T)))) < 0


def bissection(pred, lo, hi, n=14, nom=''):
    """pred(lo) vrai, pred(hi) faux."""
    for i in range(n):
        mid = 0.5 * (lo + hi)
        if pred(mid):
            lo = mid
        else:
            hi = mid
        print(f'    {nom} [{lo:.5f}, {hi:.5f}]', flush=True)
    return 0.5 * (lo + hi)


print(f'  vpa, {NM} modes, a_p = {AP*1e3:.2f} mm, {R0:.0f} tr/min, '
      f'systeme MOYENNE, m = {M}, N = {N}', flush=True)
print(f'  rho(0) = {rho(0.0):.6f}   rho(1) = {rho(1.0):.6f}', flush=True)

# 1) ou la VRAIE instabilite arrive-t-elle ?
print('\n  1) beta_vrai : recherche du beta ou rho = 1', flush=True)
hi = 1.0
while rho(hi) < 1.0 and hi < 4096.0:
    hi *= 2.0
    print(f'    rho({hi:.0f}) = {rho(hi):.6f}', flush=True)
if rho(hi) < 1.0:
    print(f'    rho reste < 1 jusqu a beta = {hi:.0f} : la regeneration ne '
          f'destabilise PAS ce point, meme amplifiee {hi:.0f} fois.',
          flush=True)
    b_vrai = float('inf')
else:
    b_vrai = bissection(lambda b: rho(b) < 1.0, hi/2.0, hi, 14, 'rho=1  :')
    print(f'    beta_vrai = {b_vrai:.4f}', flush=True)

# 2) ou le CERTIFICAT meurt-il ?
print('\n  2) beta_cert : dernier beta certifie', flush=True)
b_cert = bissection(certifie, 0.70, 0.85, 12, 'cert   :')
print(f'    beta_cert = {b_cert:.4f}', flush=True)

print('\n  ================ RESULTAT ================')
print(f'  beta_cert = {b_cert:.4f}')
print(f'  beta_vrai = {b_vrai}')
if np.isfinite(b_vrai):
    print(f'  rapport   = {b_vrai / b_cert:.1f}')
else:
    print(f'  rapport   = infini (rho < 1 pour tout beta teste)')
