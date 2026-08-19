"""15_piezo_coupling_eq14_15.py — audit du couplage piezoelectrique,
Eqs. (14)-(15) du papier.

Du, Liu, Dai, Long, "Robust combined time delay control for milling chatter
suppression of flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257.

Ce que dit le papier (texte brut, Eqs. 14-15) :

  H_Pe = [piezo_1 ... piezo_n]^T
  piezo_j = -(C_P0 d31 / h_Pa) * { [ int_{z~1}^{z~2} D_Px(x~2, z~) dz~
                                   - int_{z~1}^{z~2} D_Px(x~1, z~) dz~ ]
                                 + [ int_{x~1}^{x~2} D_Pz(x~, z~2) dx~
                                   - int_{x~1}^{x~2} D_Pz(x~, z~1) dx~ ] } U_Pj
  avec D_Px = dD_P/dx~, D_Pz = dD_P/dz~ (coordonnees NON DIMENSIONNELLES).

  C_P0 = (1/6) * (1+nu_Pe)/(1-nu_P) * E_P b_P^2 * P_M / (1 + nu_P - (1+nu_Pe) P_M)
  P_M  = (E_Pe/E_P) * (1-nu_P^2)/(1-nu_Pe^2)
         * 3 h_Pa b_P (b_P + h_Pa) / (0.5 b_P^3 + 4 h_Pa^3 + 3 b_P h_Pa^2)

Contenu de ce script :
  1. evaluation LITTERALE de l'Eq. (15) avec Tableau 1 + Tableau 2, valeurs
     intermediaires (P_M, denominateur) et finale, plus le moment par volt
     -C_P0 d31/h_Pa [N/V] ; sensibilite du resultat (le denominateur est une
     quasi-annulation) ;
  2. comparaison au modele classique du moment equivalent utilise par le depot
     m_piezo = -eta E_Pe d31 (b_P + h_Pa) / (2 (1-nu_Pe)), avec eta = 1 et avec
     le rendement de collage par cisaillement decale (Crawley & de Luis) ;
  3. verification NUMERIQUE de l'identite de la divergence : la forme
     "integrales de bord" de l'Eq. (14) et l'integrale d'aire du laplacien sur
     le rectangle du patch sont evaluees par DEUX quadratures independantes
     (Gauss 1D sur les 4 aretes / Gauss 2D pointwise sur le rectangle) ;
  4. probleme dimensionnel (VERIFICATION.md sec. 3.4) : forme litterale non
     dimensionnelle contre laplacien PHYSIQUE, rapport par mode et effet sur
     l'autorite de commande totale sum_i |D_obs(i) H_Pe(i)| ;
  5. H_Pe et D_obs(i) H_Pe(i) par mode pour les DEUX positions du patch
     (coin bas gauche et coin bas droit), motif de signes, signe du produit
     colocalise b0 = D_obs . H_Pe.

Toutes les valeurs "papier" citees sont soit imprimees dans l'article
(Tableaux 1-2), soit recalculees ici a partir de ces tableaux ; aucune valeur
de C_P0 n'est imprimee par le papier, il n'y a donc pas de reference directe.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'),
                os.path.join(HERE, '..', 'control')]

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chebyshev_plate import ChebyshevPlate, cheb_matrix
from plate_model import build_plate, PATCH

OUT = os.path.join(HERE, '..', 'figures', 'verification',
                   '15_piezo_coupling_eq14_15.png')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# --------------------------------------------------------------------------
# Donnees Tableaux 1 et 2
# --------------------------------------------------------------------------
L_P, H_P, B_P = 0.100, 0.080, 0.004        # m
RHO, E_P, NU_P = 2830.0, 69.0e9, 0.33      # Tableau 1 (AL6061)
D31, H_PA = 175e-12, 0.7e-3                # Tableau 2 (QDA60-20-0.7)
E_PE, NU_PE = 63.0e9, 0.35
G_ADH, T_ADH = 1.0e9, 30e-6                # colle (defaut du depot)


def PM_paper(EPe=E_PE, EP=E_P, nuP=NU_P, nuPe=NU_PE, bP=B_P, hPa=H_PA):
    """P_M de l'Eq. (15), litteralement comme imprime."""
    return ((EPe / EP) * ((1 - nuP**2) / (1 - nuPe**2))
            * (3 * hPa * bP * (bP + hPa))
            / (0.5 * bP**3 + 4 * hPa**3 + 3 * bP * hPa**2))


def CP0_from_PM(PM, EP=E_P, nuP=NU_P, nuPe=NU_PE, bP=B_P):
    """C_P0 de l'Eq. (15) pour un P_M donne (signe de P_M laisse libre)."""
    return ((1.0 / 6.0) * ((1 + nuPe) / (1 - nuP)) * EP * bP**2
            * PM / (1 + nuP - (1 + nuPe) * PM))


def sep(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


print('AUDIT Eqs. (14)-(15) — couplage piezoelectrique')
print('plaque : l_P=%.0f mm  h_P=%.0f mm  b_P=%.1f mm  E_P=%.0f GPa  nu_P=%.2f'
      % (1e3 * L_P, 1e3 * H_P, 1e3 * B_P, 1e-9 * E_P, NU_P))
print('patch  : 60 x 20 mm  h_Pa=%.1f mm  E_Pe=%.0f GPa  nu_Pe=%.2f  '
      'd31=%.0f pm/V' % (1e3 * H_PA, 1e-9 * E_PE, NU_PE, 1e12 * D31))

# --------------------------------------------------------------------------
# 1. Eq. (15) litterale
# --------------------------------------------------------------------------
sep('1. Eq. (15) evaluee LITTERALEMENT (Tableau 1 + Tableau 2)')

fac_E = E_PE / E_P
fac_nu = (1 - NU_P**2) / (1 - NU_PE**2)
num_geo = 3 * H_PA * B_P * (B_P + H_PA)
den_geo = 0.5 * B_P**3 + 4 * H_PA**3 + 3 * B_P * H_PA**2
fac_geo = num_geo / den_geo
PM_lit = PM_paper()
den_lit = 1 + NU_P - (1 + NU_PE) * PM_lit
CP0_lit = CP0_from_PM(PM_lit)
mom_lit = -CP0_lit * D31 / H_PA

print('  E_Pe/E_P                                  = %.6f  [-]' % fac_E)
print('  (1-nu_P^2)/(1-nu_Pe^2)                    = %.6f  [-]' % fac_nu)
print('  3 h_Pa b_P (b_P+h_Pa)                     = %.6e  [m^3]' % num_geo)
print('  0.5 b_P^3 + 4 h_Pa^3 + 3 b_P h_Pa^2       = %.6e  [m^3]' % den_geo)
print('  rapport geometrique                       = %.6f  [-]' % fac_geo)
print('  P_M (Eq. 15, litteral)                    = %.6f  [-]' % PM_lit)
print('  1 + nu_P                                  = %.6f' % (1 + NU_P))
print('  (1 + nu_Pe) P_M                           = %.6f' % ((1 + NU_PE) * PM_lit))
print('  denominateur 1+nu_P-(1+nu_Pe)P_M          = %.6f  <-- QUASI-ANNULATION'
      % den_lit)
print('  C_P0 (Eq. 15, litteral)                   = %.6e  [N]  (= Pa*m^2)' % CP0_lit)
print('  moment par volt  -C_P0 d31 / h_Pa         = %+.6f  [N/V]' % mom_lit)

# sensibilite : d ln C_P0 / d ln P_M
sens = 1.0 + (1 + NU_PE) * PM_lit / den_lit
PM_pole = (1 + NU_P) / (1 + NU_PE)
print('  d ln C_P0 / d ln P_M                      = %+.2f   '
      '(1 %% sur P_M -> %.0f %% sur C_P0)' % (sens, abs(sens)))
print('  P_M annulant le denominateur (pole)       = %.6f  '
      '(soit +%.2f %% au-dessus du P_M calcule)'
      % (PM_pole, 100 * (PM_pole / PM_lit - 1)))

sep('1b. Variante implantee dans le depot (chebyshev_plate.py:193-195)')
PM_repo = -PM_paper()          # le depot met un signe - devant P_M
den_repo = 1 + NU_P - (1 + NU_PE) * PM_repo
CP0_repo = -CP0_from_PM(PM_repo)   # et un signe - global
mom_repo = -CP0_repo * D31 / H_PA
print('  P_M (depot, signe inverse)                = %+.6f' % PM_repo)
print('  denominateur                              = %+.6f  (pas de '
      'quasi-annulation)' % den_repo)
print('  C_P0 (depot)                              = %+.6e  [N]' % CP0_repo)
print('  moment par volt  -C_P0 d31 / h_Pa         = %+.6f  [N/V]' % mom_repo)
print('  reference VERIFICATION.md sec. 3.4        = -0.033387 N/V   '
      'ecart = %+.3f %%' % (100 * (mom_repo / -0.033387 - 1)))
print('  rapport litteral / depot                  = %.2f x' % (CP0_lit / CP0_repo))

# --------------------------------------------------------------------------
# 2. Moment equivalent classique
# --------------------------------------------------------------------------
sep('2. Modele du moment equivalent utilise par le depot')
m_eta1 = -E_PE * D31 * (B_P + H_PA) / (2 * (1 - NU_PE))
eta, Gam = ChebyshevPlate.shear_lag_efficiency(
    0.5 * 0.060, 0.5 * 0.020, E_P, NU_P, B_P, E_PE, NU_PE, H_PA, G_ADH, T_ADH)
m_bond = eta * m_eta1
print('  m_piezo = -eta E_Pe d31 (b_P + h_Pa) / (2 (1 - nu_Pe))')
print('  eta = 1 (collage parfait)                 = %+.6f  [N/V]' % m_eta1)
print('  Gamma (shear lag, G=%.0e Pa, t=%.0f um)   = %.1f  1/m'
      % (G_ADH, 1e6 * T_ADH, Gam))
print('  eta (collage reel, defaut du depot)       = %.4f  [-]' % eta)
print('  m_piezo avec eta                          = %+.6f  [N/V]' % m_bond)
print('')
print('  rapport m_piezo(eta=1)   / (-C_P0 d31/h_Pa) litteral = %.4f'
      % (m_eta1 / mom_lit))
print('  rapport m_piezo(eta)     / (-C_P0 d31/h_Pa) litteral = %.4f'
      % (m_bond / mom_lit))
print('  rapport m_piezo(eta=1)   / (-C_P0 d31/h_Pa) depot    = %.4f'
      % (m_eta1 / mom_repo))
print('  rapport m_piezo(eta)     / (-C_P0 d31/h_Pa) depot    = %.4f'
      % (m_bond / mom_repo))

# --------------------------------------------------------------------------
# 3. Identite de la divergence, verifiee numeriquement sur la base reelle
# --------------------------------------------------------------------------
sep('3. Eq. (14) : integrales de bord  ==  integrale d aire du laplacien')

plate_r = build_plate(patch='right')
plate_l = build_plate(patch='left')
PX, PZ = plate_r.PX, plate_r.PZ
cx, cz = 2.0 / L_P, 2.0 / H_P
Aj = (L_P / 2.0) * (H_P / 2.0)


def _map(u1, u2, n):
    ug, wg = np.polynomial.legendre.leggauss(n)
    return 0.5 * (u1 + u2) + 0.5 * (u2 - u1) * ug, 0.5 * (u2 - u1) * wg


def area_row_physical(xt1, xt2, zt1, zt2, nq=32):
    """Integrale d aire du laplacien PHYSIQUE, quadrature 2D point par point.
    Aucune factorisation separable : chaque point de Gauss est evalue."""
    xm, wx = _map(xt1, xt2, nq)
    zm, wz = _map(zt1, zt2, nq)
    Bx0, Bx2 = cheb_matrix(PX, xm, 0), cheb_matrix(PX, xm, 2)
    Bz0, Bz2 = cheb_matrix(PZ, zm, 0), cheb_matrix(PZ, zm, 2)
    row = np.zeros(PX * PZ)
    for i in range(nq):
        for j in range(nq):
            lap_pt = (cx**2 * np.kron(Bx2[:, i], Bz0[:, j])
                      + cz**2 * np.kron(Bx0[:, i], Bz2[:, j]))
            row += wx[i] * wz[j] * Aj * lap_pt
    return row


def edge_row_physical(xt1, xt2, zt1, zt2, nq=40):
    """Somme des 4 integrales de bord de l Eq. (14), en derivees PHYSIQUES
    et longueurs PHYSIQUES ; quadrature 1D point par point sur chaque arete."""
    xm, wx = _map(xt1, xt2, nq)
    zm, wz = _map(zt1, zt2, nq)
    Bx0, Bx1 = cheb_matrix(PX, xm, 0), cheb_matrix(PX, xm, 1)
    Bz0, Bz1 = cheb_matrix(PZ, zm, 0), cheb_matrix(PZ, zm, 1)
    tx1 = cheb_matrix(PX, xt1, 1)[:, 0]
    tx2 = cheb_matrix(PX, xt2, 1)[:, 0]
    tz1 = cheb_matrix(PZ, zt1, 1)[:, 0]
    tz2 = cheb_matrix(PZ, zt2, 1)[:, 0]
    row = np.zeros(PX * PZ)
    for j in range(nq):                      # aretes x = x2 et x = x1, dz
        row += wz[j] * (H_P / 2.0) * cx * (np.kron(tx2, Bz0[:, j])
                                           - np.kron(tx1, Bz0[:, j]))
    for i in range(nq):                      # aretes z = z2 et z = z1, dx
        row += wx[i] * (L_P / 2.0) * cz * (np.kron(Bx0[:, i], tz2)
                                           - np.kron(Bx0[:, i], tz1))
    return row


def area_row_nondim(xt1, xt2, zt1, zt2, nq=32):
    """Forme LITTERALE de l Eq. (14) : derivees et longueurs en coordonnees
    non dimensionnelles (aucun facteur cx, cz, jacobien)."""
    xm, wx = _map(xt1, xt2, nq)
    zm, wz = _map(zt1, zt2, nq)
    Bx0, Bx2 = cheb_matrix(PX, xm, 0), cheb_matrix(PX, xm, 2)
    Bz0, Bz2 = cheb_matrix(PZ, zm, 0), cheb_matrix(PZ, zm, 2)
    row = np.zeros(PX * PZ)
    for i in range(nq):
        for j in range(nq):
            row += wx[i] * wz[j] * (np.kron(Bx2[:, i], Bz0[:, j])
                                    + np.kron(Bx0[:, i], Bz2[:, j]))
    return row


def edge_row_nondim(xt1, xt2, zt1, zt2, nq=40):
    xm, wx = _map(xt1, xt2, nq)
    zm, wz = _map(zt1, zt2, nq)
    Bx0 = cheb_matrix(PX, xm, 0)
    Bz0 = cheb_matrix(PZ, zm, 0)
    tx1 = cheb_matrix(PX, xt1, 1)[:, 0]
    tx2 = cheb_matrix(PX, xt2, 1)[:, 0]
    tz1 = cheb_matrix(PZ, zt1, 1)[:, 0]
    tz2 = cheb_matrix(PZ, zt2, 1)[:, 0]
    row = np.zeros(PX * PZ)
    for j in range(nq):
        row += wz[j] * (np.kron(tx2, Bz0[:, j]) - np.kron(tx1, Bz0[:, j]))
    for i in range(nq):
        row += wx[i] * (np.kron(Bx0[:, i], tz2) - np.kron(Bx0[:, i], tz1))
    return row


def patch_nd(side):
    p = PATCH[side]
    return (2 * p['x1'] / L_P - 1, 2 * p['x2'] / L_P - 1,
            2 * p['z1'] / H_P - 1, 2 * p['z2'] / H_P - 1)


nd_r = patch_nd('right')
V_r = plate_r.V
area_ph = area_row_physical(*nd_r) @ V_r
edge_ph = edge_row_physical(*nd_r) @ V_r
area_nd = area_row_nondim(*nd_r) @ V_r
edge_nd = edge_row_nondim(*nd_r) @ V_r

print('  patch coin bas DROIT, x~ in [%.3f, %.3f], z~ in [%.3f, %.3f]' % nd_r)
print('  (Gauss 2D 32x32 point par point pour l aire, Gauss 1D 40 par arete)')
print('')
print('  mode | aire lap. phys.  | bord Eq.(14) phys. | ecart relatif')
print('  -----+------------------+--------------------+---------------')
for k in range(len(area_ph)):
    rel = abs(edge_ph[k] - area_ph[k]) / max(abs(area_ph[k]), 1e-30)
    print('   %2d  | %+16.9e | %+18.9e | %10.2e' % (k + 1, area_ph[k],
                                                    edge_ph[k], rel))
rel_nd = np.abs(edge_nd - area_nd) / np.maximum(np.abs(area_nd), 1e-30)
print('  idem pour la forme non dimensionnelle : ecart max = %.2e'
      % rel_nd.max())
print('  --> le theoreme de la divergence est verifie a la precision machine :')
print('      les deux ecritures de l Eq. (14) sont bien la meme quantite.')

# --------------------------------------------------------------------------
# 4. Probleme dimensionnel : laplacien physique vs forme litterale non-dim
# --------------------------------------------------------------------------
sep('4. Probleme dimensionnel de l Eq. (14) telle qu imprimee')
print('  I_xx = int_patch d2Y/dx~2 dx~dz~ ,  I_zz = int_patch d2Y/dz~2 dx~dz~')
print('  forme litterale (non dim.)  :  I_xx + I_zz            [poids 1 : 1]')
print('  forme physique (x Aj)       :  (h_P/l_P) I_xx + (l_P/h_P) I_zz')
print('                              =  %.2f I_xx + %.2f I_zz'
      % (H_P / L_P, L_P / H_P))
print('  -> la forme litterale sur-pondere la courbure selon x d un facteur')
print('     l_P/h_P = %.2f et sous-pondere celle selon z de h_P/l_P = %.2f.'
      % (L_P / H_P, H_P / L_P))
print('')

results = {}
for side, plate in (('left', plate_l), ('right', plate_r)):
    nd = patch_nd(side)
    V = plate.V
    a_ph = area_row_physical(*nd) @ V
    a_ndm = area_row_nondim(*nd) @ V
    m_pz = plate.m_piezo                       # -eta E_Pe d31 (b+h)/(2(1-nu))
    H_ph = m_pz * a_ph
    H_nd = m_pz * a_ndm
    D_obs = plate.D_row(plate.lp, plate.hp)
    results[side] = dict(H_ph=H_ph, H_nd=H_nd, D_obs=D_obs, m_pz=m_pz,
                         a_ph=a_ph, a_nd=a_ndm)

print('  patch DROIT — H_Pe par mode [N/V / sqrt(kg)] (m_piezo = %+.6f N/V)'
      % results['right']['m_pz'])
print('  mode | H_Pe physique   | H_Pe litteral nd | rapport nd/phys')
print('  -----+-----------------+------------------+----------------')
for k in range(5):
    hp_, hn_ = results['right']['H_ph'][k], results['right']['H_nd'][k]
    print('   %2d  | %+15.6e | %+16.6e | %+13.4f'
          % (k + 1, hp_, hn_, hn_ / hp_))
for side in ('left', 'right'):
    r = results[side]
    a_ph_tot = float(np.sum(np.abs(r['D_obs'] * r['H_ph'])))
    a_nd_tot = float(np.sum(np.abs(r['D_obs'] * r['H_nd'])))
    print('  autorite totale sum_i |D_obs H_Pe|  patch %-5s : '
          'phys = %.4f   nd = %.4f   (nd/phys = %.3f)'
          % (side, a_ph_tot, a_nd_tot, a_nd_tot / a_ph_tot))
print('  --> la forme PHYSIQUE est la bonne : le patch a deformation imposee')
print('      applique un moment de ligne isotrope M par unite de longueur sur')
print('      son contour, dont le travail virtuel est M * int (w,xx + w,zz) dA')
print('      en coordonnees PHYSIQUES. La forme litterale melange 1/l_P^2 et')
print('      1/h_P^2 dans une meme somme : elle n est homogene que si l_P=h_P.')

# --------------------------------------------------------------------------
# 5. H_Pe et residus colocalises pour les deux positions du patch
# --------------------------------------------------------------------------
sep('5. H_Pe et residus D_obs(i) H_Pe(i) — patch GAUCHE vs DROIT')
print('  capteur : coin superieur droit (x = l_P, z = h_P)')
print('  frequences du modele (calage theorique Tab. 4) : %s Hz'
      % np.round(plate_r.freq_n, 1))
print('')
hdr = ('  mode |    H_Pe (gauche) |  D_obs*H (gauche) | s |'
       '    H_Pe (droit)  |   D_obs*H (droit) | s')
print(hdr)
print('  ' + '-' * (len(hdr) - 2))
for k in range(5):
    hl = results['left']['H_ph'][k]
    hr = results['right']['H_ph'][k]
    pl = results['left']['D_obs'][k] * hl
    pr = results['right']['D_obs'][k] * hr
    print('   %2d  | %+16.6e | %+17.6e | %+d | %+16.6e | %+17.6e | %+d'
          % (k + 1, hl, pl, np.sign(pl), hr, pr, np.sign(pr)))
for side in ('left', 'right'):
    r = results[side]
    prod = r['D_obs'] * r['H_ph']
    b0 = float(np.sum(prod))
    sg = [int(v) for v in np.sign(prod)]
    print('  patch %-5s : motif de signes = %s   b0 = sum(D_obs.H) = %+.4f  '
          '(signe %+d)' % (side, sg, b0, int(np.sign(b0))))
sg_r = [int(v) for v in np.sign(results['right']['D_obs']
                                * results['right']['H_ph'])]
print('  NB : le brief annonce "tous les D_obs(i) H_Pe(i) de meme signe" pour')
print('       la configuration DROITE ; le modele donne %s : les signes NE' % sg_r)
print('       sont PAS tous identiques (idem entete de plate_model.py, qui')
print('       annonce [-1 -1 -1 +1 +1] et b0 = +3.40 — confirme ici).')

# controle de coherence avec le vecteur livre par le depot
d_r = np.max(np.abs(results['right']['H_ph'] - plate_r.H_Pe_modal)
             / np.abs(plate_r.H_Pe_modal))
d_l = np.max(np.abs(results['left']['H_ph'] - plate_l.H_Pe_modal)
             / np.abs(plate_l.H_Pe_modal))
print('')
print('  controle : H_Pe recalcule ici vs plate.H_Pe_modal du depot —')
print('  ecart relatif max = %.2e (droit), %.2e (gauche)' % (d_r, d_l))
print('  effet du choix de coefficient sur H_Pe (facteur multiplicatif) :')
print('    Eq.(15) litterale / m_piezo(eta collage) = %.2f x'
      % (mom_lit / m_bond))
print('    Eq.(15) variante depot / m_piezo(eta)    = %.4f x'
      % (mom_repo / m_bond))

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
modes = np.arange(1, 6)
w = 0.38
fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.6))

a = ax[0, 0]
a.bar(modes - w / 2, results['left']['H_ph'], w, label='left lower corner',
      color='#4878a8')
a.bar(modes + w / 2, results['right']['H_ph'], w, label='right lower corner',
      color='#c8622a')
a.axhline(0, color='k', lw=0.8)
a.set_xticks(modes)
a.set_xlabel('mode')
a.set_ylabel(r'$H_{Pe}$  [N/V/$\sqrt{\mathrm{kg}}$]')
a.set_title('(a) Modal piezo coupling $H_{Pe}$, Eq. (14)')
a.legend(fontsize=8)
a.grid(alpha=0.3, axis='y')

a = ax[0, 1]
pl = results['left']['D_obs'] * results['left']['H_ph']
pr = results['right']['D_obs'] * results['right']['H_ph']
a.bar(modes - w / 2, pl, w, label='left  (b0=%+.2f)' % pl.sum(),
      color='#4878a8')
a.bar(modes + w / 2, pr, w, label='right (b0=%+.2f)' % pr.sum(),
      color='#c8622a')
a.axhline(0, color='k', lw=0.8)
a.set_xticks(modes)
a.set_xlabel('mode')
a.set_ylabel(r'$D_{obs}(i)\,H_{Pe}(i)$  [1/V]')
a.set_title('(b) Collocated residues (sign pattern)')
a.legend(fontsize=8)
a.grid(alpha=0.3, axis='y')

a = ax[1, 0]
labels = ['Eq.(15)\nliteral', 'Eq.(15)\nrepo variant',
          'equiv. moment\n$\\eta=1$', 'equiv. moment\nbonded $\\eta$=%.3f' % eta]
vals = [abs(mom_lit), abs(mom_repo), abs(m_eta1), abs(m_bond)]
cols = ['#a03030', '#c89040', '#4878a8', '#3a7a4a']
b = a.bar(range(4), vals, 0.62, color=cols)
a.set_yscale('log')
a.set_xticks(range(4))
a.set_xticklabels(labels, fontsize=8)
a.set_ylabel('|per-volt moment|  [N/V]')
a.set_title('(c) Piezo coupling coefficient (log scale)')
for r_, v in zip(b, vals):
    a.text(r_.get_x() + r_.get_width() / 2, v * 1.25, '%.4g' % v,
           ha='center', fontsize=8)
a.set_ylim(min(vals) / 3, max(vals) * 5)
a.grid(alpha=0.3, axis='y', which='both')

a = ax[1, 1]
rat_l = results['left']['H_nd'] / results['left']['H_ph']
rat_r = results['right']['H_nd'] / results['right']['H_ph']
a.bar(modes - w / 2, rat_l, w, label='left', color='#4878a8')
a.bar(modes + w / 2, rat_r, w, label='right', color='#c8622a')
a.axhline(1.0, color='k', lw=0.9, ls='--', label='ratio = 1')
a.axhline(L_P / H_P, color='#a03030', lw=0.9, ls=':',
          label='$l_P/h_P$ = %.2f' % (L_P / H_P))
a.axhline(H_P / L_P, color='#3a7a4a', lw=0.9, ls=':',
          label='$h_P/l_P$ = %.2f' % (H_P / L_P))
a.set_xticks(modes)
a.set_xlabel('mode')
a.set_ylabel('$H_{Pe}$ literal non-dim / physical Laplacian')
a.set_title('(d) Dimensional defect of Eq. (14) as printed')
a.legend(fontsize=7.5)
a.grid(alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig(OUT, dpi=140)
print('\nfigure : %s' % os.path.abspath(OUT))
