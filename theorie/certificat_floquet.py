"""Certificat de Floquet-Lyapunov EXACT — sans grille, sans solveur.

Pour un systeme periodique DISCRET  z_{k+1} = D_k z_k,  D_{k+m} = D_k,
la suite P_0..P_{m-1} > 0 verifiant

    D_k' P_{k+1} D_k - P_k = -W_k < 0,      P_m = P_0

existe SI ET SEULEMENT SI rho(Phi) < 1, ou Phi = D_{m-1}...D_0. Il n y a
donc AUCUN conservatisme, et P_k se calcule par algebre lineaire :

    P_0 = solution de  Phi' P_0 Phi - P_0 = -S,   S = somme des W transportes
    puis P_k a rebours.

Aucun gridding : la contrainte porte sur les m pas eux-memes, pas sur des
echantillons d une fonction continue. Aucun solveur : pas de statut a ne pas
croire.

On l applique la ou P CONSTANTE est PROUVEE impossible (A0(s) gelee non
Hurwitz) et ou la coupe est pourtant stable. Si les P_k sortent definies
positives avec decroissance stricte, la periodicite est demontree isolement.
"""
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, expm
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(R, 'paper_model'), os.path.join(R, 'control')]
import config as C
from plate_model import build_plate
from milling_dynamics import alpha4_series, N_TEETH
from closed_loop import build_matrices
import stored_ctrl

NM, AP, M = 2, 0.60e-3, 200
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = stored_ctrl.discover()['vpa']
D = plate.D_row(0.5*plate.lp, plate.hp)[:NM]; DtD = np.outer(D, D)
Do = plate.D_row(plate.lp, plate.hp)[:NM]
H = np.asarray(plate.H_Pe_modal, float)[:NM]
TAU = 60.0/(N_TEETH*4900)
_, a4 = alpha4_series(4900, AP, plate.hp, M)
A0s = [build_matrices(plate, DtD, Do, H, C.SIGN_SIM*a, ss, pd, NM)[0]
       for a in a4]
N = A0s[0].shape[0]
gele = max(float(np.max(np.real(np.linalg.eigvals(A)))) for A in A0s)
print(f'  vpa, {NM} modes -> {N} etats, a_p = {AP*1e3:.2f} mm, m = {M}')
print(f'  max_s max Re eig A0(s) = {gele:+.2f}  -> P CONSTANTE IMPOSSIBLE\n')

h = TAU/M
Dk = [expm(A*h) for A in A0s]                 # systeme SANS retard, exact
Phi = np.eye(N)
for Dm in Dk: Phi = Dm @ Phi
rho = float(np.max(np.abs(np.linalg.eigvals(Phi))))
print(f'  rho(Phi) = {rho:.8f}  -> systeme periodique '
      f'{"STABLE" if rho < 1 else "INSTABLE"}')

# P_0 par Lyapunov discret AVEC LE BON SECOND MEMBRE.
# Derouler P_k = D_k' P_{k+1} D_k + I autour du tour donne
#     P_0 = Phi' P_0 Phi + sum_k Psi_k' Psi_k,   Psi_0 = I, Psi_k = D_{k-1}..D_0
# Utiliser I au lieu de cette somme ne referme pas la boucle : la violation
# apparaissait alors exactement au dernier pas, la ou P_M doit valoir P_0.
Psi = np.eye(N); Ssum = np.eye(N)
for k in range(1, M):
    Psi = Dk[k-1] @ Psi
    Ssum = Ssum + Psi.T @ Psi
P0 = solve_discrete_lyapunov(Phi.T, Ssum)      # P0 = Phi' P0 Phi + Ssum
P = [None]*M; P[0] = P0
for k in range(M-1, 0, -1):
    Pn = P[k+1] if k+1 < M else P0
    P[k] = Dk[k].T @ Pn @ Dk[k] + np.eye(N)
lmin = min(float(np.min(np.linalg.eigvalsh(0.5*(p+p.T)))) for p in P)
worst, kbad = -np.inf, -1
for k in range(M):
    W = P[k] - Dk[k].T @ P[(k+1) % M] @ Dk[k]
    lw = float(np.min(np.linalg.eigvalsh(0.5*(W+W.T))))
    if -lw > worst: worst, kbad = -lw, k
print(f'  min_k lambda_min(P_k)            = {lmin:+.6e}')
print(f'  max_k [-lambda_min(P_k - Dk\' P_{{k+1}} Dk)] = {worst:+.6e}'
      f'   (pire pas k = {kbad})')
ok = (lmin > 0) and (worst < 0)
print(f'\n  CERTIFICAT DE FLOQUET-LYAPUNOV : {"VALIDE" if ok else "INVALIDE"}')
print(f'  conditionnement max_k cond(P_k)  = '
      f'{max(np.linalg.cond(p) for p in P):.3e}')
