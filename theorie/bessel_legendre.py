"""PHASE 10 — deux questions que la litterature vient de poser.

QUESTION A (Shadkami Ahvazi et al. 2025, ISA Trans.). Ils annoncent une
synthese optimale a memoire, gains par LMI depuis une fonctionnelle de
Lyapunov generale, qui « substantially expand[s] the stability region ...
at each spindle speed ». Or les phases 7-8 ont conclu que cette voie est
bloquee. L'article est INACCESSIBLE (ScienceDirect refuse par la politique
de sortie reseau de cette session ; OpenAlex aussi, 403 au CONNECT). On ne
peut donc pas le lire — mais on peut trancher la question structurelle qu'il
pose, par la mesure.

Toute la machinerie qu'ils citent (Riccati algebro-differentiel, fonctionnelle
« generale » = type complet au sens de Kharitonov) est definie pour un systeme
a retard INVARIANT. Le fraisage, lui, est PERIODIQUE : a4(t) varie sur la
periode de dent. Hypothese testable : l'obstacle des phases 7-8 vient de la
PERIODICITE, pas du retard. Si c'est vrai, la meme fonctionnelle est faisable
sur la version MOYENNEE (approximation d'ordre zero, coefficient constant,
donc systeme invariant) et infaisable sur la version periodique — AU MEME
point de fonctionnement, avec le meme retard et le meme rho.

QUESTION B (Ramirez et al. 2018, IET CTA ; IEEE Access special section 2020).
La phase 8 n'a utilise que JENSEN, c.-a-d. le cas N = 0 de Bessel-Legendre.
L'etat de l'art sur CE probleme utilise la methode descripteur et
Bessel-Legendre. On implemente ici le cas N = 1 — l'inegalite de sommation
de Wirtinger (Seuret, Gouaisbaut & Fridman) — avec la fonctionnelle AUGMENTEE
qu'elle exige :

    sum_{j=k-m}^{k-1} eta_j' Z eta_j
        >=  (1/m) Om0' Z Om0  +  (3 kappa / m) Om1' Z Om1
    Om0 = x_k - x_{k-m}
    Om1 = x_k + x_{k-m} - (2/(m+1)) sum_{j=k-m}^{k} x_j
    kappa = (m+1)/(m-1)

Le terme en Om1 ne sert a RIEN si V ne depend pas de la somme accumulee : S
serait libre dans la LMI et le pire cas annulerait Om1, ramenant a Jensen.
C'est le point classique. On augmente donc l'etat de Lyapunov :

    zeta_k = [x_k ; S_{k-1}],   S_{k-1} = sum_{j=k-m}^{k-1} x_j
    S_k = S_{k-1} + x_k - x_{k-m}

et V1 = zeta_k' P_k zeta_k avec P_k periodique de taille 2N.

Avec xi = [x_k ; x_{k-m} ; x_{k-m+1} ; S_{k-1}] :

    G  = [[A, B, C, 0], [I, -I, 0, I]]      (zeta_{k+1})
    Ez = [[I, 0, 0, 0], [0, 0, 0, I]]       (zeta_k)
    E1 = [I, 0, 0, 0]     Exm = [0, I, 0, 0]     Exm1 = [0, 0, I, 0]
    ETA = [A-I, B, C, 0]  F0 = [I, -I, 0, 0]
    F1 = (1 - 2/(m+1)) E1 + Exm - (2/(m+1)) Es

    M0 = Ez' P_k Ez - E1'(Q+R)E1 + Exm' Q Exm + Exm1' R Exm1
         + (1/m) F0' Z F0 + (3 kappa/m) F1' Z F1

et la LMI, apres complement de Schur sur les DEUX formes quadratiques en
(A,B,C) — G' P_{k+1} G et ETA'(mZ)ETA — reste AFFINE en (A,B,C) :

    [[ M0,          G' P_{k+1},   ETA'(mZ) ],
     [ P_{k+1} G,   P_{k+1},      0        ],
     [ (mZ) ETA,    0,            mZ       ]]  >  0

N = 0 se retrouve exactement en supprimant S de xi et le terme en F1.
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
    """mode='time' : coefficient periodique. mode='moyen' : ordre zero (LTI)."""
    return period_maps(plate, r, AP, 0.5 * plate.lp, ctrl=ss, pd=pd,
                       n_modes=NM, m=m, coeff_mode=mode,
                       coeff_scale=C.SIGN_SIM, ae=C.AE)[0]


def equilibre(mp):
    """Meme mise a l'echelle qu'en phase 8 — sinon la norme domine tout."""
    moy = np.mean([np.abs(a) + np.abs(b) + np.abs(c) for a, b, c in mp], axis=0)
    _, T = matrix_balance(moy, permute=False)
    Ti = np.linalg.inv(T)
    return [(Ti @ a @ T, Ti @ b @ T, Ti @ c @ T) for a, b, c in mp]


def essai(m, r, mode='time', ordre=0):
    """ordre=0 : Jensen (phase 8).  ordre=1 : Wirtinger + LKF augmentee."""
    BL = equilibre(maps(r, m, mode))
    N = BL[0][0].shape[0]
    I, Z0 = np.eye(N), np.zeros((N, N))
    nb = 4 if ordre else 3
    def bloc(*cols):
        return np.hstack([{'I': I, '-I': -I, '0': Z0}[c] if isinstance(c, str)
                          else c for c in cols])
    if ordre:
        E1 = bloc('I', '0', '0', '0'); Exm = bloc('0', 'I', '0', '0')
        Exm1 = bloc('0', '0', 'I', '0'); Es = bloc('0', '0', '0', 'I')
        Ez = np.vstack([E1, Es])
        F0 = bloc('I', '-I', '0', '0')
        c1 = 2.0 / (m + 1.0)
        F1 = (1.0 - c1) * E1 + Exm - c1 * Es
        kap = (m + 1.0) / (m - 1.0)
        nP = 2 * N
    else:
        E1 = bloc('I', '0', '0'); Exm = bloc('0', 'I', '0')
        Exm1 = bloc('0', '0', 'I')
        F0 = bloc('I', '-I', '0')
        nP = N
    P = [cp.Variable((nP, nP), symmetric=True) for _ in range(m)]
    Q = cp.Variable((N, N), symmetric=True)
    Rv = cp.Variable((N, N), symmetric=True)
    Zv = cp.Variable((N, N), symmetric=True)
    e = 1e-5
    cons = [p >> np.eye(nP) for p in P] + [Q >> e*I, Rv >> e*I, Zv >> e*I]
    for k, (A, B, Cm) in enumerate(BL):
        Pn = P[(k + 1) % m]
        if ordre:
            G = np.vstack([bloc(A, B, Cm, '0'), bloc('I', '-I', '0', 'I')])
            ETA = bloc(A - I, B, Cm, '0')
            M0 = (Ez.T @ P[k] @ Ez - E1.T @ (Q + Rv) @ E1
                  + Exm.T @ Q @ Exm + Exm1.T @ Rv @ Exm1
                  + (1.0/m) * (F0.T @ Zv @ F0)
                  + (3.0*kap/m) * (F1.T @ Zv @ F1))
        else:
            G = bloc(A, B, Cm)
            ETA = bloc(A - I, B, Cm)
            M0 = (E1.T @ P[k] @ E1 - E1.T @ (Q + Rv) @ E1
                  + Exm.T @ Q @ Exm + Exm1.T @ Rv @ Exm1
                  + (1.0/m) * (F0.T @ Zv @ F0))
        cons.append(cp.bmat([
            [M0,            G.T @ Pn,        ETA.T @ (m * Zv)],
            [Pn @ G,        Pn,              np.zeros((nP, N))],
            [(m * Zv) @ ETA, np.zeros((N, nP)), m * Zv]])
            >> e * np.eye(nb * N + nP + N))
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
    # Verification INDEPENDANTE du solveur : on reconstruit M et on regarde
    # sa plus grande valeur propre. Une phase precedente a produit quatre
    # faux certificats etiquetes « optimal » ; on ne fait plus confiance au
    # statut rendu.
    Pv = [np.asarray(p.value) for p in P]
    Qv, Rr, Zr = (np.asarray(Q.value), np.asarray(Rv.value),
                  np.asarray(Zv.value))
    pire = -np.inf
    for k, (A, B, Cm) in enumerate(BL):
        Pn = Pv[(k + 1) % m]
        if ordre:
            G = np.vstack([bloc(A, B, Cm, '0'), bloc('I', '-I', '0', 'I')])
            ETA = bloc(A - I, B, Cm, '0')
            M = (G.T @ Pn @ G - Ez.T @ Pv[k] @ Ez
                 + E1.T @ (Qv + Rr) @ E1 - Exm.T @ Qv @ Exm
                 - Exm1.T @ Rr @ Exm1
                 + m * (ETA.T @ Zr @ ETA)
                 - (1.0/m) * (F0.T @ Zr @ F0)
                 - (3.0*kap/m) * (F1.T @ Zr @ F1))
        else:
            G = bloc(A, B, Cm)
            ETA = bloc(A - I, B, Cm)
            M = (G.T @ Pn @ G - E1.T @ Pv[k] @ E1
                 + E1.T @ (Qv + Rr) @ E1 - Exm.T @ Qv @ Exm
                 - Exm1.T @ Rr @ Exm1
                 + m * (ETA.T @ Zr @ ETA)
                 - (1.0/m) * (F0.T @ Zr @ F0))
        pire = max(pire, float(np.max(np.linalg.eigvalsh(0.5 * (M + M.T)))))
    return pire, dt, N


def dis(x):
    if x is None:
        return 'infaisable'
    return f'FAISABLE ({x:+.1e})' if x < 0 else f'non ({x:+.1e})'


print(f'  vpa, {NM} modes, a_p = {AP*1e3:.2f} mm, {R0:.0f} tr/min\n')
for mode, nom in (('time', 'periodique'), ('moyen', 'MOYENNE (ordre zero)')):
    mp = maps(R0, 200, mode)
    print(f'  rho ({nom}, m=200) = '
          f'{spectral_radius(mp, 200, mp[0][0].shape[0]):.6f}')
print()
print('QUESTION A — periodique contre moyennee, meme fonctionnelle (Jensen)')
print(f'{"m":>4s} {"periodique":>22s} {"MOYENNEE (LTI)":>22s} {"t [s]":>7s}')
for m in (8, 16, 24, 40):
    pp, _, N = essai(m, R0, 'time', 0)
    pm, dt, _ = essai(m, R0, 'moyen', 0)
    print(f'{m:4d} {dis(pp):>22s} {dis(pm):>22s} {dt:7.1f}', flush=True)

print()
print('QUESTION B — Jensen (N=0) contre Wirtinger/Bessel-Legendre (N=1),')
print('             systeme PERIODIQUE, fonctionnelle augmentee')
print(f'{"m":>4s} {"Jensen N=0":>22s} {"Wirtinger N=1":>22s} {"t [s]":>7s}')
for m in (8, 16, 24, 40):
    p0, _, N = essai(m, R0, 'time', 0)
    p1, dt, _ = essai(m, R0, 'time', 1)
    print(f'{m:4d} {dis(p0):>22s} {dis(p1):>22s} {dt:7.1f}', flush=True)
