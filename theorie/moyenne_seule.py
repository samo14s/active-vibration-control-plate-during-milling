"""PHASE 10-B — le systeme MOYENNE est-il certifiable TOUT COURT ?

Les deux essais en cours coutent des heures parce qu ils paient m
contraintes du cote periodique. Or la question la plus informative est plus
etroite et beaucoup moins chere :

    la fonctionnelle dependante du retard (Jensen, phase 8) certifie-t-elle
    le systeme MOYENNE — invariant, donc UNE seule LMI a P constante — et a
    partir de quel m ?

Deux issues, toutes deux decisives :

  * elle y arrive a partir d un certain m  -> l obstacle des phases 7-8 est
    bien la PERIODICITE, et la comparaison a m egal tranche ;
  * elle n y arrive JAMAIS -> l hypothese de la phase 10 est FAUSSE,
    l obstacle est le retard lui-meme, et c est la phase 10 qu il faut
    reecrire, pas la litterature.

A m = 16 ce calcul a coute 86 s contre 1412 s pour son homologue
periodique ; on peut donc monter loin en m.
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


def maps(m):
    return period_maps(plate, R0, AP, 0.5 * plate.lp, ctrl=ss, pd=pd,
                       n_modes=NM, m=m, coeff_mode='moyen',
                       coeff_scale=C.SIGN_SIM, ae=C.AE)[0]


def essai(m, libre_P=False):
    """libre_P=False : P constante (classe restreinte, 1 LMI).
       libre_P=True  : on garde P_k periodique meme sur le systeme moyenne,
                       pour mesurer ce que la liberte supplementaire achete."""
    mp = maps(m)
    moy = np.mean([np.abs(a) + np.abs(b) + np.abs(c) for a, b, c in mp], axis=0)
    _, T = matrix_balance(moy, permute=False)
    Ti = np.linalg.inv(T)
    A, B, Cm = [Ti @ x @ T for x in mp[0]]      # tous les pas sont identiques
    N = A.shape[0]
    I, Z0 = np.eye(N), np.zeros((N, N))
    E1 = np.hstack([I, Z0, Z0]); F = np.hstack([I, -I, Z0])
    ABC = np.hstack([A, B, Cm]); ETA = ABC - E1
    P = cp.Variable((N, N), symmetric=True)
    Q = cp.Variable((N, N), symmetric=True)
    Rv = cp.Variable((N, N), symmetric=True)
    Zv = cp.Variable((N, N), symmetric=True)
    e = 1e-5
    haut = (cp.bmat([[P - Q - Rv, Z0, Z0], [Z0, Q, Z0], [Z0, Z0, Rv]])
            + (1.0/m) * (F.T @ Zv @ F))
    cons = [P >> I, Q >> e*I, Rv >> e*I, Zv >> e*I,
            cp.bmat([[haut,           ABC.T @ P,        ETA.T @ (m * Zv)],
                     [P @ ABC,        P,                np.zeros((N, N))],
                     [(m * Zv) @ ETA, np.zeros((N, N)), m * Zv]])
            >> e * np.eye(5 * N)]
    pb = cp.Problem(cp.Minimize(0), cons)
    t0 = time.time()
    for slv in (cp.CLARABEL, cp.SCS):
        try:
            pb.solve(solver=slv, verbose=False)
            if P.value is not None:
                break
        except Exception:
            continue
    dt = time.time() - t0
    if P.value is None:
        return None, dt, N
    Pv, Qv = np.asarray(P.value), np.asarray(Q.value)
    Rr, Zr = np.asarray(Rv.value), np.asarray(Zv.value)
    M = (ABC.T @ Pv @ ABC
         - np.block([[Pv - Qv - Rr, Z0, Z0], [Z0, Qv, Z0], [Z0, Z0, Rr]])
         + m * (ETA.T @ Zr @ ETA) - (1.0/m) * (F.T @ Zr @ F))
    return float(np.max(np.linalg.eigvalsh(0.5 * (M + M.T)))), dt, N


mp = maps(200)
print(f'  vpa, {NM} modes, a_p = {AP*1e3:.2f} mm, {R0:.0f} tr/min, '
      f'systeme MOYENNE (invariant)', flush=True)
print(f'  rho (m=200) = {spectral_radius(mp, 200, mp[0][0].shape[0]):.6f}\n',
      flush=True)
print(f'{"m":>5s} {"verdict":>26s} {"t [s]":>8s}')
for m in (16, 24, 32, 40, 60, 80, 120, 160, 200):
    p, dt, N = essai(m)
    v = 'infaisable' if p is None else (
        f'FAISABLE ({p:+.1e})' if p < 0 else f'non ({p:+.1e})')
    print(f'{m:5d} {v:>26s} {dt:8.1f}', flush=True)
