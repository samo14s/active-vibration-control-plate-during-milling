"""
verif_reduction.py — la reduction en temps normalise, testee et non supposee
============================================================================
Affirmation (theorie/01_REDUCTION_TEMPS_NORMALISE.md) : en posant s = t/tau,
le fraisage en boucle fermee devient

    x'(s) = tau A0(s) x(s) + tau A1(s) x(s-1),   retard 1, periode 1

ou A0 et A1 sont 1-periodiques et INDEPENDANTES de la vitesse de broche.

Ce script rebatit la monodromie a la main sur cette forme et la compare a ce
que rend le moteur du depot sur le systeme d'origine. Deux assemblages qui ne
partagent que step_integrals : si la reduction est fausse, ils divergent.

On teste a des vitesses STABLES ET INSTABLES. Une reduction qui ne se verifie
que du cote stable ne verifie rien : c'est le franchissement de rho = 1 qui
porte toute la decision.

    PROTOCOL=B CALIB=measured python theorie/verif_reduction.py
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
from milling_dynamics import alpha4_series, N_TEETH
from closed_loop import build_matrices, period_maps, spectral_radius
from step_integrals import step_integrals
from stored_ctrl import discover
import monodromy as MD

AP, M = 0.30e-3, 200
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = discover()['musyn_td']
n = C.N_MODES
D = plate.D_row(0.5*plate.lp, plate.hp)[:n]
DtD = np.outer(D, D)
D_obs = plate.D_row(plate.lp, plate.hp)[:n]
H = np.asarray(plate.H_Pe_modal, float)[:n]

def rho_normalise(tau, ap=AP, m=M):
    """monodromie du systeme NORMALISE : pas h = 1/m, matrices tau*A."""
    _, a4 = alpha4_series(4900, ap, plate.hp, m)     # vitesse arbitraire :
    h = 1.0 / m                                       # la serie n'en depend pas
    maps = []
    for k in range(m):
        A0, A1 = build_matrices(plate, DtD, D_obs, H, C.SIGN_SIM*a4[k],
                                ss, pd, n)
        P0, J1, J2 = step_integrals(tau*A0, h)
        maps.append((P0, (J1 - J2/h) @ (tau*A1), (J2/h) @ (tau*A1)))
    return MD.spectral_radius(maps, m, maps[0][0].shape[0])

print(f'{"tr/min":>7s} {"tau [ms]":>9s} {"rho (depot)":>14s} '
      f'{"rho (normalise)":>16s} {"ecart relatif":>14s}')
for rpm in (3000, 3600, 4200, 4900, 5600, 6400, 7000):
    tau = 60.0/(N_TEETH*rpm)
    maps, _ = period_maps(plate, rpm, AP, 0.5*plate.lp, ctrl=ss, pd=pd,
                          n_modes=n, m=M, coeff_mode='time',
                          coeff_scale=C.SIGN_SIM, ae=C.AE)
    r_dep = spectral_radius(maps, M, maps[0][0].shape[0])
    r_nor = rho_normalise(tau)
    e = abs(r_nor - r_dep)/max(r_dep, 1e-300)
    print(f'{rpm:7d} {tau*1e3:9.4f} {r_dep:14.10f} {r_nor:16.10f} {e:14.2e}')
