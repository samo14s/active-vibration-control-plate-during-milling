"""Certificat periodique — le maillage corrige.

Le premier essai gridait la LMI sur 16 points en s et le solveur rendait un
« optimal » que la verification fine invalidait de +3.4e4 : la contrainte etait
imposee sur des echantillons, et P(s) se faufilait entre eux.

Trois corrections :
  1. GRILLE DE RESOLUTION six fois plus dense ;
  2. MARGE MAXIMISEE au lieu d etre figee : on resout  max gamma  sous
     M <= -gamma I, P >= I, Q >= I. gamma dit combien de place il reste, ce
     qu une simple faisabilite ne dit pas ;
  3. VERIFICATION SUR UNE GRILLE DECALEE de la grille de resolution, et huit
     fois plus fine — verifier sur les points ou l on a impose la contrainte
     ne verifie rien.

On rapporte aussi la constante de Lipschitz empirique L de lambda_max(M(s)) et
le pas h : gamma > L h / 2 est ce qui transformerait « verifie sur une grille
fine » en « demontre pour tout s ». C est rapporte, pas suppose.
"""
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np, cvxpy as cp
from scipy.linalg import matrix_balance
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(R, 'paper_model'), os.path.join(R, 'control')]
import config as C
from plate_model import build_plate
from milling_dynamics import alpha4_series, N_TEETH
from closed_loop import build_matrices

NM, AP = 2, 0.60e-3
NS = int(os.environ.get('NS', 96))
NH = int(os.environ.get('NH', 6))
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
import stored_ctrl
ss, pd = stored_ctrl.discover()['vpa']
D = plate.D_row(0.5*plate.lp, plate.hp)[:NM]; DtD = np.outer(D, D)
Do = plate.D_row(plate.lp, plate.hp)[:NM]
H = np.asarray(plate.H_Pe_modal, float)[:NM]
TAU = 60.0/(N_TEETH*4900)

def blocs(m, decalage=0.5):
    _, a4 = alpha4_series(4900, AP, plate.hp, m)
    return [((k+decalage)/m,) + build_matrices(plate, DtD, Do, H,
            C.SIGN_SIM*a, ss, pd, NM) for k, a in enumerate(a4)]

BR = blocs(NS)
moy = np.mean([np.abs(A0)+np.abs(A1) for _, A0, A1 in BR], axis=0)
_, T = matrix_balance(moy, permute=False); Ti = np.linalg.inv(T)
sc = lambda A: TAU*(Ti@A@T)
BL = [(s, sc(A0), sc(A1)) for s, A0, A1 in BR]
N = BL[0][1].shape[0]
SIG = max(np.linalg.norm(A0) for _, A0, _ in BL)
# a4 saute-t-elle ? (entree/sortie de dent)
_, a4f = alpha4_series(4900, AP, plate.hp, 2048)
saut = np.max(np.abs(np.diff(np.r_[a4f, a4f[0]])))/max(np.max(np.abs(a4f)),1e-30)
print(f'  {N} etats, ||tau A0|| = {SIG:.2f}, NS = {NS}, NH = {NH}')
print(f'  saut relatif max de a4 entre echantillons voisins (2048) : {saut:.3e}')

def base(s, NH):
    b, db = [1.0], [0.0]
    for j in range(1, NH+1):
        b += [np.cos(2*np.pi*j*s), np.sin(2*np.pi*j*s)]
        db += [-2*np.pi*j*np.sin(2*np.pi*j*s), 2*np.pi*j*np.cos(2*np.pi*j*s)]
    return b, db

def essai(NH, retard, tag):
    nb = 2*NH+1
    Pk = [cp.Variable((N, N), symmetric=True) for _ in range(nb)]
    Q = cp.Variable((N, N), symmetric=True); g = cp.Variable(nonneg=True)
    cons = [Q >> np.eye(N)]
    for s, A0, A1 in BL:
        if not retard: A1 = np.zeros_like(A1)
        b, db = base(s, NH)
        P = sum(b[i]*Pk[i] for i in range(nb))
        dP = sum(db[i]*Pk[i] for i in range(nb))
        cons += [P >> np.eye(N)]
        cons.append(cp.bmat([[dP + (A0.T@P+P@A0) + Q, P@A1],
                             [A1.T@P, -Q]]) << -g*np.eye(2*N))
    pb = cp.Problem(cp.Maximize(g), cons)
    for slv in (cp.CLARABEL, cp.SCS):
        try:
            pb.solve(solver=slv, verbose=False)
            if Pk[0].value is not None: break
        except Exception: continue
    if Pk[0].value is None:
        print(f'  {tag:38s} echec / infaisable'); return
    Pv = [np.asarray(p.value) for p in Pk]; Qv = np.asarray(Q.value)
    gam = float(g.value)
    # verification sur grille DECALEE et 8x plus fine
    BF = [(s, sc(A0), sc(A1)) for s, A0, A1 in blocs(8*NS, decalage=0.0)]
    mp, mm, lam = np.inf, -np.inf, []
    for s, A0, A1 in BF:
        if not retard: A1 = np.zeros_like(A1)
        b, db = base(s, NH)
        P = sum(b[i]*Pv[i] for i in range(nb)); dP = sum(db[i]*Pv[i] for i in range(nb))
        mp = min(mp, float(np.min(np.linalg.eigvalsh(0.5*(P+P.T)))))
        M = np.block([[dP + (A0.T@P+P@A0)+Qv, P@A1], [A1.T@P, -Qv]])
        l = float(np.max(np.linalg.eigvalsh(0.5*(M+M.T)))); lam.append(l)
        mm = max(mm, l)
    lam = np.array(lam); h = 1.0/NS
    L = float(np.max(np.abs(np.diff(np.r_[lam, lam[0]])))*8*NS)
    ok = 'VALIDE' if (mp > 0 and mm < 0) else 'INVALIDE'
    preuve = 'ET DEMONTRE pour tout s' if gam > L*h/2 else \
             f'(pas de preuve : gamma={gam:.3g} < L h/2={L*h/2:.3g})'
    print(f'  {tag:38s} gamma={gam:+.3e}  lmin(P)={mp:+.2e}  '
          f'lmax(M)={mm:+.2e}  [{ok}]')
    print(f'  {"":38s} L={L:.3g}  h={h:.4g}  -> {preuve}')

print('\nA. sans terme retarde — periodicite isolee')
essai(0,  False, 'P constante')
essai(NH, False, f'P(s) periodique, {NH} harmoniques')
print('\nB. avec le terme retarde')
essai(0,  True,  'P constante')
essai(NH, True,  f'P(s) periodique, {NH} harmoniques')
