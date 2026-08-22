"""La classe (P, Q) CONSTANTES est-elle faisable en principe, ou pas du tout ?

Le bloc superieur gauche de M exige  tau (A0(s)' P + P A0(s)) + Q < 0  pour
TOUT s. Avec Q > 0, cela impose  A0(s)' P + P A0(s) < 0 : une SEULE forme
quadratique doit etre de Lyapunov pour A0(s) a chaque instant du tour.

Condition necessaire : A0(s) doit etre Hurwitz pour tout s. Si elle ne l'est
pas — ne serait-ce qu'a un seul s — aucun P n'existe, et l'infaisabilite est
STRUCTURELLE, pas numerique.

C'est physiquement plausible : le terme de coupe -a4(s) D'D RETRANCHE de la
raideur. Quand a4 est grand, la plaque gelee peut avoir des poles a droite.
C'est d'ailleurs pourquoi la stabilite du fraisage est un phenomene de
FLOQUET et non de temps gele.
"""
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(R, 'paper_model'), os.path.join(R, 'control')]
import config as C
from plate_model import build_plate
from milling_dynamics import alpha4_series
from closed_loop import build_matrices
from stored_ctrl import discover

plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
n = C.N_MODES
D = plate.D_row(0.5*plate.lp, plate.hp)[:n]
DtD = np.outer(D, D); D_obs = plate.D_row(plate.lp, plate.hp)[:n]
H = np.asarray(plate.H_Pe_modal, float)[:n]
M = 64
for tag, kind in (('mu + retard', 'musyn_td'), ('mu seul', 'musyn')):
    ss, pd = discover()[kind]
    print(f'\n=== {tag} ===')
    print(f'{"a_p [mm]":>9s} {"max_s max Re eig A0(s)":>24s} '
          f'{"# de s ou A0 non Hurwitz":>26s}  {"max a4":>10s}')
    for ap in (0.0, 0.10e-3, 0.30e-3, 0.70e-3):
        _, a4 = alpha4_series(4900, ap, plate.hp, M)
        pires, nbad = -np.inf, 0
        for a in a4:
            A0, _ = build_matrices(plate, DtD, D_obs, H, C.SIGN_SIM*a,
                                   ss, pd, n)
            r = float(np.max(np.real(np.linalg.eigvals(A0))))
            pires = max(pires, r)
            nbad += (r > 0.0)
        print(f'{ap*1e3:9.2f} {pires:24.3f} {nbad:20d} / {M}  '
              f'{np.max(np.abs(a4)):10.3e}')
