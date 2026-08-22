"""PHASE 10-C — LE TEMOIN POSITIF QUI MANQUAIT.

Depuis la phase 7 ce depot accumule des verdicts NEGATIFS avec la meme
fonctionnelle (P, Q, R, Z) sans avoir jamais verifie qu elle REUSSIT sur
quoi que ce soit. C est une faute de methode : un test qui n a jamais dit
oui ne prouve rien quand il dit non. Il peut etre casse.

On la repare par homotopie. On prend le systeme MOYENNE (invariant) et on
attenue les matrices RETARDEES par un facteur beta :

    x_{k+1} = A x_k + beta ( B x_{k-m} + C x_{k-m+1} )

  * beta = 0 : plus de retard du tout. Le systeme est LTI ordinaire, stable
    (rho(A) < 1), et la fonctionnelle DOIT le certifier — il suffit de
    prendre Q = R = Z = eps I et P solution de l equation de Lyapunov.
    Si le code dit « infaisable » ICI, le code est faux, et tout ce qui a
    ete conclu depuis la phase 7 tombe.
  * beta = 1 : le systeme reel.

En balayant beta on obtient, au lieu d un oui/non, une MESURE : quelle
fraction de la regeneration cette classe de fonctionnelles sait absorber.
C est un resultat quantitatif dans les deux cas.

On imprime aussi rho(beta), pour distinguer « la classe echoue » de
« le systeme est instable ».
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


def base(m, mode='moyen'):
    mp = period_maps(plate, R0, AP, 0.5 * plate.lp, ctrl=ss, pd=pd,
                     n_modes=NM, m=m, coeff_mode=mode,
                     coeff_scale=C.SIGN_SIM, ae=C.AE)[0]
    moy = np.mean([np.abs(a) + np.abs(b) + np.abs(c) for a, b, c in mp], axis=0)
    _, T = matrix_balance(moy, permute=False)
    Ti = np.linalg.inv(T)
    return [(Ti @ a @ T, Ti @ b @ T, Ti @ c @ T) for a, b, c in mp]


def essai(BL, m, beta):
    A, B, Cm = BL[0]
    B, Cm = beta * B, beta * Cm
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
        return None, dt
    Pv, Qv = np.asarray(P.value), np.asarray(Q.value)
    Rr, Zr = np.asarray(Rv.value), np.asarray(Zv.value)
    Mm = (ABC.T @ Pv @ ABC
          - np.block([[Pv - Qv - Rr, Z0, Z0], [Z0, Qv, Z0], [Z0, Z0, Rr]])
          + m * (ETA.T @ Zr @ ETA) - (1.0/m) * (F.T @ Zr @ F))
    return float(np.max(np.linalg.eigvalsh(0.5 * (Mm + Mm.T)))), dt


BL = base(M)
A0 = BL[0][0]
print(f'  vpa, {NM} modes, a_p = {AP*1e3:.2f} mm, {R0:.0f} tr/min, '
      f'systeme MOYENNE, m = {M}', flush=True)
print(f'  rho(A) sans retard (beta=0) = '
      f'{max(abs(np.linalg.eigvals(A0))):.6f}', flush=True)
print(f'\n{"beta":>6s} {"rho(beta)":>10s} {"verdict":>26s} {"t [s]":>7s}')
for beta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0):
    BLb = [(a, beta*b, beta*c) for a, b, c in BL]
    rho = spectral_radius(BLb, M, A0.shape[0])
    p, dt = essai(BL, M, beta)
    v = 'infaisable' if p is None else (
        f'FAISABLE ({p:+.1e})' if p < 0 else f'non ({p:+.1e})')
    print(f'{beta:6.2f} {rho:10.6f} {v:>26s} {dt:7.1f}', flush=True)
