"""Le terme retarde, enfin — sans faire exploser la dimension.

L etat souleve vaut (m+1)N = 410 composantes : une P_k pleine y ferait 84 000
inconnues. Inutile. Le systeme souleve est une IMPULSION + DECALAGE :

    x_{k+1} = A_k x_k + B_k x_{k-m} + C_k x_{k-m+1}

(les trois blocs que `closed_loop.period_maps` fabrique deja). Une fonctionnelle
de Krasovskii DISCRETE a deux sommes telescopiques suffit :

    V = x_k' P_k x_k + sum_{j=1..m} x_{k-j}' Q x_{k-j}
                     + sum_{j=1..m-1} x_{k-j}' R x_{k-j}

    dV = [x_k ; x_{k-m} ; x_{k-m+1}]' M [ . ]
    M  = [A B C]' P_{k+1} [A B C] + diag(Q + R - P_k, -Q, -R)

Les deux sommes sont necessaires : avec une seule, la direction x_{k-m+1}
n aurait aucun terme negatif et M ne pourrait jamais etre definie negative.

Inconnues : m matrices N x N plus Q et R — meme ordre qu a la phase 6. Et la
forme de Schur

    [[ diag(P_k - Q - R, Q, R) , [A B C]' P_{k+1} ],
     [ P_{k+1} [A B C]         , P_{k+1}          ]] > 0

reste AFFINE en (A, B, C), donc la convexite sur l intervalle de tau tient
exactement comme a la phase 6.

Le plafond de comparaison est cette fois le VRAI rho du fraisage avec
regeneration — celui que le depot calcule depuis le debut.
"""
import os, sys, time
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np, cvxpy as cp
from scipy.linalg import matrix_balance
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(R, 'paper_model'), os.path.join(R, 'control')]
import config as C
from plate_model import build_plate
from milling_dynamics import N_TEETH
from closed_loop import period_maps, spectral_radius
import stored_ctrl

KIND = os.environ.get('KIND', 'vpa')
NM = int(os.environ.get('NM', 2))
AP = float(os.environ.get('AP', 0.60))*1e-3
MM = int(os.environ.get('MM', 40))
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()[KIND]
t = lambda r: 60.0/(N_TEETH*r)

def maps_de(r, m=MM):
    return period_maps(plate, r, AP, 0.5*plate.lp, ctrl=ss, pd=pd,
                       n_modes=NM, m=m, coeff_mode='time',
                       coeff_scale=C.SIGN_SIM, ae=C.AE)[0]

def rho(r, m=200):
    mp = maps_de(r, m)
    return spectral_radius(mp, m, mp[0][0].shape[0])

mp0 = maps_de(4900)
N = mp0[0][0].shape[0]
# equilibrage commun sur la moyenne des trois blocs
moy = np.mean([np.abs(a)+np.abs(b)+np.abs(c) for a, b, c in mp0], axis=0)
_, T = matrix_balance(moy, permute=False); Ti = np.linalg.inv(T)
sc = lambda X: Ti @ X @ T
print(f'  {KIND}, {NM} modes -> {N} etats, a_p = {AP*1e3:.2f} mm, m = {MM}')
print(f'  AVEC terme retarde : etat souleve = {(MM+1)*N} '
      f'(evite), inconnues = {MM} x {N}x{N} + 2\n')

def blocs(r):
    return [(sc(a), sc(b), sc(c)) for a, b, c in maps_de(r)]

def faisable(r0, dw, verif=True):
    V = [blocs(r0-dw), blocs(r0+dw)]
    P = [cp.Variable((N, N), symmetric=True) for _ in range(MM)]
    Q = cp.Variable((N, N), symmetric=True)
    Rv = cp.Variable((N, N), symmetric=True)
    # NORMALISATION : la LMI est homogene en (P, Q, R), donc on fixe l echelle
    # avec P >= I. La marge peut alors etre 1e-4 — trois ordres au-dessus du
    # bruit du solveur, qui valait 1e-7 et se faisait passer pour une
    # infaisabilite : le rejet etait CONSTANT a toutes les largeurs, y compris
    # a largeur nulle ou le systeme est pourtant stable.
    e = 1e-4
    cons = [p >> np.eye(N) for p in P] + [Q >> e*np.eye(N), Rv >> e*np.eye(N)]
    for Bl in V:
        for k, (A, B, Cm) in enumerate(Bl):
            Pn = P[(k+1) % MM]
            ABC = np.hstack([A, B, Cm])
            D1 = cp.bmat([[P[k]-Q-Rv, np.zeros((N, N)), np.zeros((N, N))],
                          [np.zeros((N, N)), Q, np.zeros((N, N))],
                          [np.zeros((N, N)), np.zeros((N, N)), Rv]])
            cons.append(cp.bmat([[D1, ABC.T@Pn], [Pn@ABC, Pn]])
                        >> e*np.eye(4*N))
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
    Qv = 0.5*(np.asarray(Q.value)+np.asarray(Q.value).T)
    Rr = 0.5*(np.asarray(Rv.value)+np.asarray(Rv.value).T)
    if min(np.min(np.linalg.eigvalsh(x)) for x in Pv+[Qv, Rr]) <= 0:
        return False, None, dt
    if not verif: return True, None, dt
    pire = -np.inf
    for r in np.linspace(r0-dw, r0+dw, 15):
        for k, (A, B, Cm) in enumerate(blocs(r)):
            Pn = Pv[(k+1) % MM]; ABC = np.hstack([A, B, Cm])
            M = ABC.T @ Pn @ ABC - np.block([
                [Pv[k]-Qv-Rr, np.zeros((N,N)), np.zeros((N,N))],
                [np.zeros((N,N)), Qv, np.zeros((N,N))],
                [np.zeros((N,N)), np.zeros((N,N)), Rr]])
            pire = max(pire, float(np.max(np.linalg.eigvalsh(0.5*(M+M.T)))))
    return pire < -0.5*e, pire, dt

# plafond VRAI avec retard, balaye finement
def plafond(r0, pas=2.0):
    dw = 0.0
    while dw < 800:
        n = dw + pas
        if rho(r0-n) >= 1.0 or rho(r0+n) >= 1.0: break
        dw = n
    return dw

r0 = float(os.environ.get('R0', 5200))
print(f'  centre {r0:.0f} : rho (avec retard, m=200) = {rho(r0):.6f}')
plaf = plafond(r0)
print(f'  plafond VRAI avec retard : +/- {plaf:.0f} tr/min\n')
lo, hi = 0.0, plaf*1.05
for _ in range(9):
    mid = 0.5*(lo+hi)
    ok, mg, dt = faisable(r0, mid)
    print(f'    +/- {mid:7.2f} tr/min : {str(ok):>5s}  '
          f'marge {("%+.2e" % mg) if mg is not None else "-":>10s}  {dt:5.1f} s',
          flush=True)
    lo, hi = (mid, hi) if ok else (lo, mid)
print(f'\n  bande certifiee AVEC retard : +/- {lo:.2f} tr/min'
      f'   (plafond {plaf:.0f})   ratio {lo/max(plaf,1e-9):.3f}')
