"""PHASE 10-F — beta_cert des DEUX systemes, meme grille, meme protocole.

rapport_perio.py a tourne 31 minutes sur un seul appel — certifie(1.0) —
uniquement pour fixer une borne superieure que trois essais anterieurs
avaient deja etablie. Faute de conception : les appels INFAISABLES sont les
plus chers (le solveur s epuise avant de conclure), et une bissection en
paie la moitie.

Correction : un balayage ASCENDANT qui s arrete au PREMIER echec. Les appels
faisables sont rapides (15 s a beta = 0 sur le periodique) ; on ne paie donc
qu UN SEUL appel cher par systeme, celui qui franchit le seuil.

Meme grille et memes reglages de solveur des deux cotes : c est la condition
pour que la comparaison des deux seuils veuille dire quelque chose. La
resolution 0.05 suffit largement — on cherche un ecart entre deux systemes,
pas une decimale.

Reference deja mesuree (phase 10-D, bissection complete, systeme moyenne) :
beta_cert = 0.7955. La grille doit la retrouver entre 0.75 et 0.80, ce qui
sert de controle croise entre les deux protocoles.
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


def charge(mode):
    mp = period_maps(plate, R0, AP, 0.5 * plate.lp, ctrl=ss, pd=pd,
                     n_modes=NM, m=M, coeff_mode=mode,
                     coeff_scale=C.SIGN_SIM, ae=C.AE)[0]
    moy = np.mean([np.abs(a)+np.abs(b)+np.abs(c) for a, b, c in mp], axis=0)
    _, T = matrix_balance(moy, permute=False)
    Ti = np.linalg.inv(T)
    return [(Ti @ a @ T, Ti @ b @ T, Ti @ c @ T) for a, b, c in mp]


def certifie(BL, beta):
    """P_k periodique des DEUX cotes : classe identique, comparaison honnete.
    (Sur le systeme moyenne les m pas sont egaux, donc P_k constante est
    incluse dans la classe — lui laisser P_k ne peut que l avantager, ce qui
    va contre la these et non dans son sens.)"""
    N = BL[0][0].shape[0]
    I, Z0 = np.eye(N), np.zeros((N, N))
    E1 = np.hstack([I, Z0, Z0]); F = np.hstack([I, -I, Z0])
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
        haut = (cp.bmat([[P[k]-Q-Rv, Z0, Z0], [Z0, Q, Z0], [Z0, Z0, Rv]])
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
        return False, None
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
        pire = max(pire, float(np.max(np.linalg.eigvalsh(0.5*(Mm+Mm.T)))))
    return pire < 0, pire


GRILLE = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70,
          0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
B_VRAI = {'moyen': 2.3362, 'time': 1.8969}
res = {}
for mode, nom in (('moyen', 'MOYENNE'), ('time', 'PERIODIQUE')):
    BL = charge(mode)
    N = BL[0][0].shape[0]
    print(f'\n===== systeme {nom} — m = {M}, N = {N} =====', flush=True)
    print(f'{"beta":>6s} {"verdict":>24s} {"t [s]":>8s}', flush=True)
    dernier = None
    for beta in GRILLE:
        t0 = time.time()
        ok, pire = certifie(BL, beta)
        dt = time.time() - t0
        v = (f'FAISABLE ({pire:+.1e})' if ok else
             ('infaisable' if pire is None else f'non ({pire:+.1e})'))
        print(f'{beta:6.2f} {v:>24s} {dt:8.1f}', flush=True)
        if ok:
            dernier = beta
        else:
            print(f'   -> premier echec a beta = {beta:.2f} ; arret.',
                  flush=True)
            break
    res[mode] = dernier
    if dernier is None:
        print('   AUCUN beta certifie, pas meme 0 : montage en cause.',
              flush=True)

print('\n================ RESULTAT ================')
for mode, nom in (('moyen', 'MOYENNE'), ('time', 'PERIODIQUE')):
    d = res.get(mode)
    if d is None:
        print(f'  {nom:11s} beta_cert = aucun')
    else:
        print(f'  {nom:11s} beta_cert >= {d:.2f}   beta_vrai = '
              f'{B_VRAI[mode]}   rapport <= {B_VRAI[mode]/d:.2f}')
print('  (phase 10-D, bissection, moyenne : beta_cert = 0.7955, '
      'rapport 2.94 — controle croise)')
