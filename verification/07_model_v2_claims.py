"""F4 — quelle position de patch les mesures soutiennent-elles ?

Ce script remplace une premiere version fautive qui cherchait les
antiresonances comme minima locaux de |G|. Avec amortissement, un intervalle
inter-mode SANS zero presente quand meme un minimum peu profond : le detecteur
les comptait comme des antiresonances et concluait a tort que model_v2 ne
reproduisait pas la Fig. 12(b).

Le critere correct est exact et insensible a l'amortissement : pour
G = somme r_i/(w_i^2 - w^2) avec r_i = D_obs,i * H_Pe,i, l'intervalle k
contient un nombre IMPAIR de zeros si sign(r_k) == sign(r_{k+1}), PAIR sinon.
On calcule donc les zeros exacts comme racines d'un polynome, et on compare
leur repartition a celle des creux profonds mesures.
"""
import os
import sys

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K = os.path.join(_R, 'simulation', 'sim_kit')
sys.path[:0] = [_K, os.path.join(_R, 'simulation')]
os.chdir(_K)

import simulation_base as SB                                    # noqa: E402
from plate_model import PlateModel                              # noqa: E402
from model_v2 import antiresonances, notch_occupancy            # noqa: E402

L, Hh = SB.PLATE_L, SB.PLATE_H
MEAS_F = np.array([540., 1068., 2787., 3351., 4122.])           # Table 4
Z_VOLT = np.array([788., 1493., 3609.])                         # Fig. 12(b)
OCC = (1, 1, 0, 1)
Z_DRIVE = np.array([734., 2161., 3001., 3805.])                 # Fig. 12(a)

CFG = {
    'bas-gauche 20x60 (v1)': dict(x1=0.,       x2=.020, z1=0., z2=.060),
    'bas-droit   20x60    ': dict(x1=L - .020, x2=L,    z1=0., z2=.060),
    'bas-gauche  60x20    ': dict(x1=0.,       x2=.060, z1=0., z2=.020),
    'bas-droit   60x20 (v2)': dict(x1=L - .060, x2=L,   z1=0., z2=.020),
}

# residus deduits des hauteurs de pic de la Fig. 12(b) : |r_i| = pic * 2 zeta w^2
PK = np.array([55.8, 48.7, 29.8, 31.8, 35.1])
ZT = np.array([.0031, .0017, .0027, .0056, .0035])
R_MEAS = 10 ** (PK / 20) * 0.01e-6 * 2 * ZT * (2 * np.pi * MEAS_F) ** 2


def build(P, structural=True):
    p = PlateModel(L, Hh, SB.PLATE_T, SB.RHO, SB.YOUNG, SB.POISSON,
                   N1=SB.MESH_N1, N2=SB.MESH_N2, n_modes=5,
                   zeta_modes=SB.ZETA_MODES, verbose=False)
    p.precompute_Dp(zp_pos=Hh - .15e-3, n_pos=3)
    p.set_observation(L, Hh)
    p.add_piezo_patch(P['x1'], P['x2'], P['z1'], P['z2'], SB.PATCH['d31'],
                      SB.PATCH['thickness'], SB.PATCH['E'], SB.PATCH['nu'],
                      structural=structural)
    return p


print('--- 1. signature de la Fig. 12(b) : occupation des intervalles ---')
print(f"{'configuration':24s} {'f1 brut':>8s} {'occupation':12s} {'accord':7s} "
      f"{'residus /r1':32s} zeros en bande (Hz)")
print('-' * 126)
for nm, P in CFG.items():
    p = build(P)
    f1 = p.freq_n[0]
    p.calibrate_frequencies(SB.F_MEASURED)
    occ = notch_occupancy(p)
    z = antiresonances(p)
    fn = p.freq_n
    zi = z[(z > fn[0]) & (z < fn[-1])]
    r = np.abs(p.D_obs * p.H_Pe_modal)
    print(f'{nm:24s} {f1:8.1f} {str(occ):12s} {"OUI" if occ == OCC else "non":7s} '
          f'{str(np.round(r / r[0], 2)):32s} {np.round(zi, 0)}')
print('-' * 126)
print(f'{"MESURE":24s} {MEAS_F[0]:8.1f} {str(OCC):12s} {"":7s} '
      f'{str(np.round(R_MEAS / R_MEAS[0], 2)):32s} {Z_VOLT}')
print('\n=> une seule configuration reproduit (1, 1, 0, 1) : bas-droit 60x20,')
print('   celle de model_v2. Les FREQUENCES, elles, ne suivent qu aux extremites.')

print('\n--- 2. le premier mode ne discrimine pas l orientation ---')
print('    _add_patch_structure suppose un collage PARFAIT ; la colle reelle ne')
print('    transmet le cisaillement que partiellement. Bornes physiques :')
p0 = PlateModel(L, Hh, SB.PLATE_T, SB.RHO, SB.YOUNG, SB.POISSON,
                N1=SB.MESH_N1, N2=SB.MESH_N2, n_modes=5,
                zeta_modes=SB.ZETA_MODES, verbose=False)
print(f'      sans raidissement           : {p0.freq_n[0]:6.1f} Hz')
print(f'      vertical,   collage parfait : {build(CFG["bas-gauche 20x60 (v1)"]).freq_n[0]:6.1f} Hz')
print(f'      horizontal, collage parfait : {build(CFG["bas-droit   60x20 (v2)"]).freq_n[0]:6.1f} Hz')
print(f'      MESURE (Table 4)            : {MEAS_F[0]:6.1f} Hz  -> dans les deux encadrements')

print('\n--- 3. check_model : quelle FRF comparer aux antiresonances de Fig. 12(a) ? ---')
print('    Fig. 12(a) : marteau ET capteur au coin superieur droit -> residus D_obs^2.')
s1 = SB.SimBase()
q = s1.plate
Do1 = np.asarray(q.D_obs).ravel()
Dt1 = np.asarray(q.Dp_array[:, 0]).ravel()
ff = np.linspace(60, 4900, 200000)
w = 2 * np.pi * ff


def anti2(res):
    G = np.abs(np.sum(res[:, None] / ((q.omega_n[:, None] ** 2 - w[None, :] ** 2)
               + 2j * q.zeta_modes[:, None] * q.omega_n[:, None] * w[None, :]),
               axis=0))
    lg = np.log(G)
    return np.array([ff[i] for i in range(1, len(G) - 1)
                     if lg[i] < lg[i - 1] and lg[i] < lg[i + 1]])


for lab, res in [('D_obs*D_tool (avant correction F2)', Do1 * Dt1),
                 ('D_obs*D_obs  (apres correction F2)', Do1 * Do1)]:
    a = anti2(res)
    e = [min(abs(a - t)) / t * 100 for t in Z_DRIVE]
    print(f'    {lab:36s} {np.round(a).astype(int)}  ecarts % {np.round(e, 1)}'
          f'  RMS {np.sqrt(np.mean(np.square(e))):.2f} %')
