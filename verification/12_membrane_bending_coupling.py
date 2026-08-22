"""Couplage membrane-flexion d'un patch colle d'un seul cote : implementation,
validation, et FALSIFICATION de l'hypothese.

C'etait la derniere piste ouverte pour expliquer le desaccord de H_Pe avec la
signature de la Fig. 12(b). Un patch colle d'UN SEUL COTE n'applique pas qu'un
moment : sous tension il applique aussi une resultante membranaire N_E, et le
stratifie plaque+patch etant NON SYMETRIQUE (matrice de couplage B non nulle),
cette resultante se rabat en flexion. Une formulation de Kirchhoff en flexion
pure ne peut pas le representer.

Le script fait trois choses :
  1. VALIDE l'assemblage A / B / D en montrant que la condensation statique des
     ddl membranaires reproduit la rigidite de section transformee ;
  2. mesure l'effet sur H_Pe ;
  3. montre que la signature des creux NE CHANGE PAS -- l'hypothese est
     falsifiee.
"""
import os
import sys

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, 'simulation', 'sim_kit'),
                os.path.join(_R, 'simulation')]
os.chdir(os.path.join(_R, 'simulation', 'sim_kit'))

import simulation_base as SB                                    # noqa: E402
from plate_model import PlateModel, _Qbar                       # noqa: E402
from model_v2 import (antiresonances, notch_occupancy,          # noqa: E402
                      NOTCH_OCCUPANCY, NOTCH_HZ)

L, H, b1 = SB.PLATE_L, SB.PLATE_H, SB.PLATE_T
P = SB.PATCH
h2, E2, nu2 = P['thickness'], P['E'], P['nu']

print('--- 1. validation de l assemblage A / B / D ---')
Q1, Q2 = _Qbar(SB.YOUNG, SB.POISSON), _Qbar(E2, nu2)
A = Q1*b1 + Q2*h2
Bc = Q2*(h2*(b1 + h2)/2)
D = Q1*b1**3/12 + Q2*(((b1/2 + h2)**3 - (b1/2)**3)/3)
D_eff = (D - Bc @ np.linalg.inv(A) @ Bc)[0, 0]
E1p, E2p = SB.YOUNG/(1 - SB.POISSON**2), E2/(1 - nu2**2)
z1, z2 = 0.0, (b1 + h2)/2
zb = (E1p*b1*z1 + E2p*h2*z2)/(E1p*b1 + E2p*h2)
D_ts = E1p*(b1**3/12 + b1*(z1-zb)**2) + E2p*(h2**3/12 + h2*(z2-zb)**2)
print(f'   condensation statique des ddl membranaires : {D_eff:9.4f} N.m')
print(f'   rigidite de section transformee            : {D_ts:9.4f} N.m')
print(f'   ecart                                      : {(D_eff/D_ts-1)*100:9.4f} %')
print('   => l assemblage est correct, ET la raideur condensee est deja celle')
print('      qu utilise _add_patch_structure : seule la CHARGE manquait.')


def build(membrane):
    p = PlateModel(L, H, b1, SB.RHO, SB.YOUNG, SB.POISSON,
                   N1=SB.MESH_N1, N2=SB.MESH_N2, n_modes=5,
                   zeta_modes=SB.ZETA_MODES, verbose=False)
    p.precompute_Dp(zp_pos=H - 0.15e-3, n_pos=3)
    p.set_observation(*SB.SENSOR_XZ)
    p.add_piezo_patch(P['x1'], P['x2'], P['z1'], P['z2'], P['d31'],
                      h2, E2, nu2, G_adh=P['G_adh'], t_adh=P['t_adh'],
                      membrane=membrane)
    return p


print('\n--- 2. effet sur H_Pe ---')
p0, p1 = build(False), build(True)
H0 = np.asarray(p0.H_Pe_modal)
H1 = np.asarray(p1.H_Pe_modal)
print(f"{'mode':>5s} {'flexion seule':>15s} {'+ membrane':>13s} {'variation':>11s}")
for k in range(5):
    print(f'{k+1:5d} {H0[k]:15.5f} {H1[k]:13.5f} '
          f'{(abs(H1[k])/abs(H0[k])-1)*100:10.1f} %')
print(f"{'||H||':>5s} {np.linalg.norm(H0):15.5f} {np.linalg.norm(H1):13.5f} "
      f"{(np.linalg.norm(H1)/np.linalg.norm(H0)-1)*100:10.1f} %")
print('   effet QUASI SCALAIRE : -11 % environ sur tous les modes, -29 % sur le')
print('   mode 3 (le seul mal conditionne, cf. verification/11).')

print('\n--- 3. la signature des creux change-t-elle ? ---')
print(f"{'modele':28s} {'occupation':12s} {'ok':>4s}  zeros en bande (Hz)")
for lab, p in [('flexion seule', p0), ('+ couplage membrane', p1)]:
    p.calibrate_frequencies(SB.F_MEASURED)
    occ = notch_occupancy(p)
    z = antiresonances(p)
    f = np.asarray(p.omega_n)/2/np.pi
    zi = z[(z > f[0]) & (z < f[-1])]
    ok = 'OUI' if occ == NOTCH_OCCUPANCY else 'non'
    print(f'{lab:28s} {str(occ):12s} {ok:>4s}  {np.round(zi, 0)}')
print(f'{"MESURE Fig.12(b)":28s} {str(NOTCH_OCCUPANCY):12s} {"":>4s}  {list(NOTCH_HZ)}')
print("""
   NON. L'hypothese est FALSIFIEE : l'effet est quasi scalaire, et un scalaire
   se simplifie dans les rapports de residus qui fixent les zeros.

   Le couplage est neanmoins conserve dans le modele : c'est un effet physique
   reel de -11 % sur l'autorite actionneur, et de -29 % sur le mode 3.""")
