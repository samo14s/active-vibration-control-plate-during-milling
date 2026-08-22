"""Le test aux SOMMETS de l'intervalle suffit-il ?

Deux enonces se cachent derriere la conjecture :

  (a) « la STABILITE aux deux bornes tau_min, tau_max entraine la stabilite
      sur tout l'intervalle » ;
  (b) « la FAISABILITE d'une meme fonctionnelle LK aux deux bornes entraine sa
      faisabilite sur tout l'intervalle » — vraie par convexite si la LMI est
      affine en tau, ce qu'elle est ici.

(a) est ce qu'un praticien testerait, et c'est celui-la qu'on attaque : il
suffit d'UN contre-exemple. On balaie tau finement sur la forme normalisee et
l'on cherche un intervalle dont les deux bouts sont stables et l'interieur non.
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
from closed_loop import build_matrices
from step_integrals import step_integrals
from stored_ctrl import discover
import monodromy as MD

AP, M = 0.30e-3, 200
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
ss, pd = discover()['musyn_td']
n = C.N_MODES
D = plate.D_row(0.5*plate.lp, plate.hp)[:n]
DtD = np.outer(D, D); D_obs = plate.D_row(plate.lp, plate.hp)[:n]
H = np.asarray(plate.H_Pe_modal, float)[:n]
_, A4 = alpha4_series(4900, AP, plate.hp, M)      # independante de la vitesse
BLOCS = [build_matrices(plate, DtD, D_obs, H, C.SIGN_SIM*a, ss, pd, n)
         for a in A4]

def rho(tau, m=M):
    h = 1.0/m
    maps = []
    for A0, A1 in BLOCS:
        P0, J1, J2 = step_integrals(tau*A0, h)
        maps.append((P0, (J1 - J2/h) @ (tau*A1), (J2/h) @ (tau*A1)))
    return MD.spectral_radius(maps, m, maps[0][0].shape[0])

rpm = np.arange(3500, 4601, 25.0)
print('  balayage fin autour de la poche suspectee (pas de 25 tr/min)')
print(f'{"tr/min":>7s} {"tau [ms]":>9s} {"rho":>12s}   ')
vals = []
for r in rpm:
    t = 60.0/(N_TEETH*r)
    v = rho(t)
    vals.append(v)
    flag = '  INSTABLE' if v > 1.0 else ''
    if r % 100 == 0 or (v > 1.0) != (vals[-2] > 1.0 if len(vals) > 1 else False):
        print(f'{r:7.0f} {t*1e3:9.4f} {v:12.6f}{flag}')
vals = np.array(vals)
np.savez(os.path.join(HERE, '..', 'results', 'sommets_B.npz'), rpm=rpm, rho=vals)
inst = rpm[vals > 1.0]
print(f'\n  instables : {inst.min():.0f}-{inst.max():.0f} tr/min'
      if inst.size else '\n  aucune instable')
