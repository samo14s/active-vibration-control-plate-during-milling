"""Ou exactement le modele de couplage H_Pe echoue-t-il ?

Le desaccord avec la Fig. 12(b) est le seul ecart structurel qui subsiste. Ce
script le localise.

H_Pe,j est proportionnel au flux de courbure au bord du patch, qui se decompose
en une contribution selon x et une selon z :

    H_j  ∝  A_x,j + rho * A_z,j ,   A_x = ∫∫ ∂²phi_j/∂x² dA , A_z = ∫∫ ∂²phi_j/∂z² dA

rho = 1 est le laplacien physique (choix du code). L'Eq. (14) de l'article,
ecrite en coordonnees non dimensionnelles sans les jacobiens, se lirait
rho = (hP/lP)^2 = 0.64 ; l'assignation reciproque donnerait (lP/hP)^2 = 1.5625.

Deux resultats :

  1. le mode 3 est en QUASI-ANNULATION -- A_x et A_z s'opposent et H_3 n'est que
     ~25 % du plus grand des deux, contre 70-81 % pour les autres modes. Son
     signe, donc l'occupation des intervalles, y est fragile ;

  2. mais balayer rho ne suffit pas : les modes 2, 4 et 5 sont BIEN conditionnes
     et leurs signes restent en desaccord avec la mesure. Le defaut n'est donc
     ni un probleme de conditionnement seul, ni un probleme de ponderation.

Conclusion : l'ecart est structurel dans le modele de moment equivalent
lui-meme. L'hypothese suivante a tester serait le couplage membrane-flexion
d'un patch colle d'UN SEUL COTE, qu'une formulation de Kirchhoff en flexion pure
ne peut pas representer.
"""
import os
import sys

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, 'simulation', 'sim_kit'),
                os.path.join(_R, 'simulation')]
os.chdir(os.path.join(_R, 'simulation', 'sim_kit'))

import simulation_base as SB                                    # noqa: E402
from plate_model import PlateModel                              # noqa: E402
from kirchhoff_q4 import matrix_der_K                           # noqa: E402
from model_v2 import NOTCH_OCCUPANCY, NOTCH_HZ                  # noqa: E402

L, H = SB.PLATE_L, SB.PLATE_H
P = SB.PATCH
R_MEAS = np.array([1.00, 0.95, 1.16, 4.39, 6.07])   # residus deduits de Fig.12(b)


def curvature_fluxes(xP1, xP2, zP1, zP2, N1, N2, lex, ley, n1, ndof):
    """Separe ∫∫ ∂²N/∂x² dA et ∫∫ ∂²N/∂z² dA sur le rectangle du patch."""
    gx = np.zeros(ndof)
    gz = np.zeros(ndof)
    xg = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    for J in range(max(1, int(np.floor(zP1/ley)) + 1),
                   min(N2, int(np.ceil(zP2/ley))) + 1):
        for I in range(max(1, int(np.floor(xP1/lex)) + 1),
                       min(N1, int(np.ceil(xP2/lex))) + 1):
            xl, xh = (I-1)*lex, I*lex
            yl, yh = (J-1)*ley, J*ley
            a, b = max(xl, xP1), min(xh, xP2)
            c, d = max(yl, zP1), min(yh, zP2)
            if b <= a or d <= c:
                continue
            xm, xr = (a+b)/2, (b-a)/2
            ym, yr = (c+d)/2, (d-c)/2
            Dof = np.r_[3*(J-1)*n1 + 3*(I-1) + np.arange(6),
                        3*J*n1 + 3*(I-1) + np.arange(3, 6),
                        3*J*n1 + 3*(I-1) + np.arange(3)]
            for ig in range(2):
                for jg in range(2):
                    xi = 2*((xm + xr*xg[ig]) - xl)/lex - 1
                    et = 2*((ym + yr*xg[jg]) - yl)/ley - 1
                    B = matrix_der_K(xi, et, lex, ley)
                    gx[Dof] += -B[0, :]*(xr*yr)
                    gz[Dof] += -B[1, :]*(xr*yr)
    return gx, gz


def occupancy(r, om):
    n = len(r)
    s2 = om**2
    Pp = np.zeros(n)
    for i in range(n):
        Pp += r[i]*np.poly(np.delete(s2, i))*((-1)**(n-1))
    z = np.roots(Pp)
    z = z[np.abs(z.imag) <= 1e-6*max(np.abs(z.real).max(), 1e-30)].real
    z = np.sort(np.sqrt(z[z > 0]))/(2*np.pi)
    f = om/2/np.pi
    return (tuple(int(((z > f[k]) & (z < f[k+1])).sum()) for k in range(4)),
            z[(z > f[0]) & (z < f[-1])])


p = PlateModel(L, H, SB.PLATE_T, SB.RHO, SB.YOUNG, SB.POISSON,
               N1=SB.MESH_N1, N2=SB.MESH_N2, n_modes=5,
               zeta_modes=SB.ZETA_MODES, verbose=False)
p.precompute_Dp(zp_pos=H - 0.15e-3, n_pos=3)
p.set_observation(L, H)
p.add_piezo_patch(P['x1'], P['x2'], P['z1'], P['z2'], P['d31'], P['thickness'],
                  P['E'], P['nu'], G_adh=P['G_adh'], t_adh=P['t_adh'])
gx, gz = curvature_fluxes(P['x1'], P['x2'], P['z1'], P['z2'],
                          SB.MESH_N1, SB.MESH_N2, p.lex, p.ley, p.n1, p.ndof)
Ax = p.V.T @ gx[p.DOFf]
Az = p.V.T @ gz[p.DOFf]

print('--- 1. conditionnement de H_Pe, mode par mode (rho = 1) ---')
print(f"{'mode':>5s} {'A_x':>11s} {'A_z':>11s} {'A_x+A_z':>11s} "
      f"{'|somme|/max|part|':>18s}")
S = Ax + Az
for k in range(5):
    ratio = abs(S[k])/max(abs(Ax[k]), abs(Az[k]))
    flag = '   <-- QUASI-ANNULATION' if ratio < 0.15 else ''
    print(f'{k+1:5d} {Ax[k]:11.4f} {Az[k]:11.4f} {S[k]:11.4f} '
          f'{ratio:18.3f}{flag}')
print('   le mode 3 est un residu de deux termes opposes : son SIGNE y est')
print('   fragile, et c est ce signe qui fixe l occupation des intervalles.')

p.calibrate_frequencies(SB.F_MEASURED)
Do = p.D_obs
print('\n--- 2. mais aucune ponderation rho ne reconstitue la mesure ---')
print(f"{'rho':>8s} {'occupation':12s} {'ok':>4s}  {'residus /r1':30s} zeros (Hz)")
for rho in (0.64, 1.0, 1.25, 1.5625, 1.8, 2.2, 3.0):
    r = Do*(Ax + rho*Az)
    occ, z = occupancy(r, p.omega_n)
    a = np.abs(r)
    ok = 'OUI' if occ == NOTCH_OCCUPANCY else 'non'
    print(f'{rho:8.4f} {str(occ):12s} {ok:>4s}  '
          f'{str(np.round(a/a[0], 2)):30s} {np.round(z, 0)}')
print(f'{"MESURE":>8s} {str(NOTCH_OCCUPANCY):12s} {"":>4s}  '
      f'{str(R_MEAS):30s} {list(NOTCH_HZ)}')
print("""
   rho = 0.6400 : Eq.(14) telle qu ecrite (x par lP, z par hP)
   rho = 1.0000 : laplacien physique -- choix du code
   rho = 1.5625 : assignation reciproque des echelles

   rho >= 1.5625 fait bien basculer le signe du mode 3, mais les modes 2, 4
   et 5 -- BIEN conditionnes -- restent en desaccord. Le defaut n est donc pas
   une affaire de ponderation ni de conditionnement seul : il est structurel
   dans le modele de moment equivalent. Piste suivante : le couplage
   membrane-flexion d un patch colle d un seul cote.""")
