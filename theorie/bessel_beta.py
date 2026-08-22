"""PHASE 10-G — Bessel-Legendre N=1 contre Jensen N=0, MESURE EN beta.

L obligation venue de Ramirez et al. (2018, IET CTA) est toujours ouverte :
la phase 8 n a utilise que JENSEN, soit le cas N = 0, alors que l etat de
l art sur CE probleme emploie la methode descripteur et Bessel-Legendre.

bessel_legendre.py y repond, mais en binaire et apres la question A : plus de
deux heures deja, et les lignes N=1 sont encore loin. Or depuis la phase 10-D
on dispose d une mesure CONTINUE bien meilleure que le oui/non — beta — et la
question devient nette et bon marche :

    beta_cert(N=1)  >  beta_cert(N=0) ?

  * si oui, Bessel-Legendre achete quelque chose et la conclusion de la
    phase 8 doit etre reecrite ;
  * si les deux seuils coincident, la phase 8 se renforce beaucoup : ce n est
    plus « Jensen echoue » mais « la classe echoue, y compris son membre le
    plus fin publie sur ce probleme ».

Inegalite de sommation de Wirtinger (Seuret, Gouaisbaut & Fridman), N = 1 :

    sum eta_j' Z eta_j >= (1/m) Om0' Z Om0 + (3 kappa/m) Om1' Z Om1
    Om0 = x_k - x_{k-m}
    Om1 = x_k + x_{k-m} - (2/(m+1)) sum_{j=k-m}^{k} x_j
    kappa = (m+1)/(m-1)

Le terme en Om1 ne vaut RIEN si V ignore la somme accumulee : S serait libre
dans la LMI et le pire cas annulerait Om1, ramenant exactement a Jensen.
D ou l etat de Lyapunov AUGMENTE zeta_k = [x_k ; S_{k-1}], P_k de taille 2N.

Les deux ordres partagent une seule fonction : on compare deux inegalites,
pas deux programmes.
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
from closed_loop import period_maps
import stored_ctrl

NM, AP = 2, 0.60e-3
R0 = float(os.environ.get('R0', 5200))
M = int(os.environ.get('M', 24))
MODE = os.environ.get('MODE', 'moyen')
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()['vpa']

mp = period_maps(plate, R0, AP, 0.5 * plate.lp, ctrl=ss, pd=pd, n_modes=NM,
                 m=M, coeff_mode=MODE, coeff_scale=C.SIGN_SIM, ae=C.AE)[0]
moy = np.mean([np.abs(a)+np.abs(b)+np.abs(c) for a, b, c in mp], axis=0)
_, T = matrix_balance(moy, permute=False)
Ti = np.linalg.inv(T)
BL = [(Ti @ a @ T, Ti @ b @ T, Ti @ c @ T) for a, b, c in mp]
N = BL[0][0].shape[0]
I, Z0 = np.eye(N), np.zeros((N, N))
KAP = (M + 1.0) / (M - 1.0)
C1 = 2.0 / (M + 1.0)


def blocs(ordre):
    n = 4 if ordre else 3
    col = lambda *c: np.hstack([{'I': I, '-I': -I, '0': Z0}[x] for x in c])
    if ordre:
        E1 = col('I', '0', '0', '0'); Exm = col('0', 'I', '0', '0')
        Ex1 = col('0', '0', 'I', '0'); Es = col('0', '0', '0', 'I')
        return (n, E1, Exm, Ex1, np.vstack([E1, Es]),
                col('I', '-I', '0', '0'), (1.0-C1)*E1 + Exm - C1*Es)
    E1 = col('I', '0', '0'); Exm = col('0', 'I', '0'); Ex1 = col('0', '0', 'I')
    return n, E1, Exm, Ex1, E1, col('I', '-I', '0'), None


def certifie(beta, ordre):
    n, E1, Exm, Ex1, Ez, F0, F1 = blocs(ordre)
    nP = 2 * N if ordre else N
    P = [cp.Variable((nP, nP), symmetric=True) for _ in range(M)]
    Q = cp.Variable((N, N), symmetric=True)
    Rv = cp.Variable((N, N), symmetric=True)
    Zv = cp.Variable((N, N), symmetric=True)
    e = 1e-5
    cons = [p >> np.eye(nP) for p in P] + [Q >> e*I, Rv >> e*I, Zv >> e*I]

    def geom(k, beta):
        a, b, c = BL[k]
        b, c = beta * b, beta * c
        if ordre:
            G = np.vstack([np.hstack([a, b, c, Z0]),
                           np.hstack([I, -I, Z0, I])])
            ETA = np.hstack([a - I, b, c, Z0])
        else:
            G = np.hstack([a, b, c])
            ETA = np.hstack([a - I, b, c])
        return G, ETA

    for k in range(M):
        G, ETA = geom(k, beta)
        Pn = P[(k + 1) % M]
        M0 = (Ez.T @ P[k] @ Ez - E1.T @ (Q + Rv) @ E1
              + Exm.T @ Q @ Exm + Ex1.T @ Rv @ Ex1
              + (1.0/M) * (F0.T @ Zv @ F0))
        if ordre:
            M0 = M0 + (3.0*KAP/M) * (F1.T @ Zv @ F1)
        cons.append(cp.bmat([
            [M0,             G.T @ Pn,          ETA.T @ (M * Zv)],
            [Pn @ G,         Pn,                np.zeros((nP, N))],
            [(M*Zv) @ ETA,   np.zeros((N, nP)), M * Zv]])
            >> e * np.eye(n*N + nP + N))
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
        G, ETA = geom(k, beta)
        Pn = Pv[(k + 1) % M]
        Mm = (G.T @ Pn @ G - Ez.T @ Pv[k] @ Ez
              + E1.T @ (Qv + Rr) @ E1 - Exm.T @ Qv @ Exm
              - Ex1.T @ Rr @ Ex1
              + M * (ETA.T @ Zr @ ETA) - (1.0/M) * (F0.T @ Zr @ F0))
        if ordre:
            Mm = Mm - (3.0*KAP/M) * (F1.T @ Zr @ F1)
        pire = max(pire, float(np.max(np.linalg.eigvalsh(0.5*(Mm+Mm.T)))))
    return pire < 0, pire


GRILLE = [0.0, 0.40, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
          1.10, 1.25, 1.50]
print(f'  vpa, {NM} modes, a_p = {AP*1e3:.2f} mm, {R0:.0f} tr/min, '
      f'systeme {MODE}, m = {M}, N = {N}', flush=True)
print(f'  reference phase 10-D (Jensen, bissection) : beta_cert = 0.7955\n',
      flush=True)
seuil = {}
for ordre, nom in ((0, 'JENSEN  N=0'), (1, 'WIRTINGER N=1')):
    print(f'===== {nom} =====', flush=True)
    print(f'{"beta":>6s} {"verdict":>24s} {"t [s]":>8s}', flush=True)
    dernier = None
    for beta in GRILLE:
        t0 = time.time()
        ok, pire = certifie(beta, ordre)
        v = (f'FAISABLE ({pire:+.1e})' if ok else
             ('infaisable' if pire is None else f'non ({pire:+.1e})'))
        print(f'{beta:6.2f} {v:>24s} {time.time()-t0:8.1f}', flush=True)
        if ok:
            dernier = beta
        else:
            print(f'   -> premier echec a beta = {beta:.2f} ; arret.\n',
                  flush=True)
            break
    seuil[ordre] = dernier

print('================ RESULTAT ================')
print(f'  Jensen    N=0 : beta_cert >= {seuil.get(0)}')
print(f'  Wirtinger N=1 : beta_cert >= {seuil.get(1)}')
a, b = seuil.get(0), seuil.get(1)
if a is not None and b is not None:
    if b > a:
        print(f'  -> Bessel-Legendre ACHETE {b - a:.2f} en beta. '
              f'La conclusion de la phase 8 doit etre reecrite.')
    elif b == a:
        print('  -> AUCUN gain a la resolution de la grille. La phase 8 se '
              'renforce : ce n est pas Jensen qui echoue, c est la classe.')
    else:
        print('  -> N=1 fait MOINS BIEN que N=0 : impossible en theorie '
              '(N=0 est un cas particulier). Bogue ou bruit de solveur.')
