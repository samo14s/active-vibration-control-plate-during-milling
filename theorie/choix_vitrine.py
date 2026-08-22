"""Quel est le PLUS PETIT systeme qui exhibe l ecart « gele non Hurwitz mais
fraisage stable » ? C est la seule vitrine honnete du certificat periodique :
il faut une profondeur OU P constante est PROUVEE impossible et ou la coupe
est pourtant stable."""
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
from closed_loop import build_matrices, limit as cl_limit
from stored_ctrl import discover
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
st = discover()

def geometrie(ss, pd, nm):
    D = plate.D_row(0.5*plate.lp, plate.hp)[:nm]
    return (np.outer(D, D), plate.D_row(plate.lp, plate.hp)[:nm],
            np.asarray(plate.H_Pe_modal, float)[:nm])

def pire_gele(ss, pd, nm, ap, m=96):
    DtD, Do, H = geometrie(ss, pd, nm)
    _, a4 = alpha4_series(4900, ap, plate.hp, m)
    return max(float(np.max(np.real(np.linalg.eigvals(
        build_matrices(plate, DtD, Do, H, C.SIGN_SIM*a, ss, pd, nm)[0]))))
        for a in a4)

def seuil(ss, pd, nm):
    lo, hi = 0.0, 1.5e-3
    if pire_gele(ss, pd, nm, hi) <= 0: return np.inf
    for _ in range(20):
        mid = 0.5*(lo+hi)
        lo, hi = (mid, hi) if pire_gele(ss, pd, nm, mid) <= 0 else (lo, mid)
    return 0.5*(lo+hi)

print(f'{"structure":10s} {"nm":>3s} {"etats":>6s} {"seuil gele":>11s} '
      f'{"a_p,lim vrai":>13s} {"ecart":>9s}')
for kind in ('dvf', 'vpa', 'fopid', 'adrc', 'lqg', 'mpc', 'hinf', 'musyn',
             'musyn_td'):
    ss, pd = st[kind]
    for nm in (2, 5):
        nx = 2*nm + (0 if ss is None else np.shape(ss[0])[0])
        sg = seuil(ss, pd, nm)
        vrai = cl_limit(plate, C.RPM_DESIGN, 0.5*plate.lp, ctrl=ss, pd=pd,
                        n_modes=nm, m=200, hi=4e-3, coeff_mode='time',
                        coeff_scale=C.SIGN_SIM, ae=C.AE)
        ec = (vrai - sg)*1e3 if np.isfinite(sg) else -np.inf
        mark = '  <== ECART' if ec > 0.02 else ''
        print(f'{kind:10s} {nm:3d} {nx:6d} '
              f'{(sg*1e3 if np.isfinite(sg) else 999):11.4f} {vrai*1e3:13.4f} '
              f'{ec:9.4f}{mark}')
