"""
09_ritz_model_identification.py
===============================
"Est-ce vraiment le modele du papier ?"  Verification d'identification de la
discretisation Chebyshev-Ritz de paper_model/chebyshev_plate.py contre
Du, Liu, Dai & Long, Int. J. Mech. Sci. 274 (2024) 109257.

Cinq etapes :
  1. convergence de la plaque NUE en taille de base PX = PZ = 6..18 et en
     penalisation d'encastrement (kw, kr) = (1e10,1e6)..(1e13,1e9) ;
  2. contre-verification par un modele EF Kirchhoff-Q4 INDEPENDANT
     (simulation/sim_kit/plate_model.py), meme continuum, meme Tableau 1 ;
  3. identification de l'ecart residuel a la colonne "theoretical" du
     Tableau 4 : le rapport theo/modele est CONSTANT sur les cinq modes,
     donc scalaire dans sqrt(D/(rho b)), donc materiau et non discretisation ;
  4. effet de la pastille piezo collee (frequences avec / sans, rendement
     de collage eta) et rappel des frequences MESUREES du Tableau 4 ;
  5. consequence pour le modele de commande : le recalage de frequences du
     depot n'est PAS strictement equivalent au changement de materiau.

Sortie : tableau numerique imprime + figure 4 panneaux.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'),
                os.path.join(HERE, '..', 'simulation', 'sim_kit')]

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chebyshev_plate import ChebyshevPlate
from plate_model import PlateModel          # EF Kirchhoff-Q4 de sim_kit

# ---------------------------------------------------------------------------
# Donnees du papier
# ---------------------------------------------------------------------------
LP, HP, BP = 0.100, 0.080, 0.004            # Tableau 1
RHO_T1, E_T1, NU_T1 = 2830.0, 69.0e9, 0.33  # Tableau 1
RHO_ID, E_ID = 2700.0, 70.0e9               # jeu AL6061 "manuel" identifie
ZETA = (0.0031, 0.0017, 0.0027, 0.0056, 0.0035)          # Tableau 4
F_THEO = np.array([537.0, 1101.0, 2805.0, 3423.0, 4254.0])   # Tableau 4
F_MEAS = np.array([540.0, 1068.0, 2787.0, 3351.0, 4122.0])   # Tableau 4
PATCH = dict(x1=0.040, x2=0.100, z1=0.0, z2=0.020)   # coin bas droit, 60 x 20
#   (geometrie retenue par verification/19_patch_orientation.py : la seule des
#    quatre admissibles qui reproduit la signature de zeros de la Fig. 12b)
KW0, KR0 = 1e12, 1e8                        # penalisation par defaut du depot

P_LIST = [6, 8, 10, 12, 14, 16, 18]
PEN_LIST = [(1e10, 1e6), (1e11, 1e7), (1e12, 1e8), (1e13, 1e9), (1e14, 1e10)]
MESH_LIST = [(20, 16), (30, 24), (40, 32), (50, 40), (60, 48)]

fmt5 = lambda v: " ".join(f"{x:9.2f}" for x in v)
fmtp = lambda v: " ".join(f"{x:+9.3f}" for x in v)


def ritz(P=18, kw=KW0, kr=KR0, E=E_T1, rho=RHO_T1, nu=NU_T1):
    """Plaque NUE Chebyshev-Ritz (pas de pastille)."""
    return ChebyshevPlate(lp=LP, hp=HP, bp=BP, rho=rho, E=E, nu=nu,
                          PX=P, PZ=P, n_modes=5, kw=kw, kr=kr,
                          zeta_modes=ZETA, rotary=True)


print("=" * 92)
print(" 09 - IDENTIFICATION DU MODELE DE PLAQUE CHEBYSHEV-RITZ")
print(" Du et al., Int. J. Mech. Sci. 274 (2024) 109257 - Tableaux 1, 2 et 4")
print("=" * 92)
print(" Plaque nue AL6061 : lP=100 mm  hP=80 mm  bP=4 mm ; inertie de rotation I2 activee")
print(" Reference Tableau 4 'theoretical' :", fmt5(F_THEO))
print(" Reference Tableau 4 'measured'    :", fmt5(F_MEAS))

# ===========================================================================
# 1. CONVERGENCE
# ===========================================================================
print("\n" + "-" * 92)
print(" 1a. CONVERGENCE EN TAILLE DE BASE (kw=%.0e, kr=%.0e, plaque nue)" % (KW0, KR0))
print("-" * 92)
print(f" {'PX=PZ':>6} {'ndof':>6} | {'f1':>9} {'f2':>9} {'f3':>9} {'f4':>9} {'f5':>9} |"
      f" {'max|df| vs P=18 [%]':>20}")
freqs_P = {}
for P in P_LIST:
    freqs_P[P] = ritz(P=P).freq_n.copy()
fref = freqs_P[P_LIST[-1]]
for P in P_LIST:
    d = 100.0 * np.abs(freqs_P[P] / fref - 1.0)
    print(f" {P:>6d} {P*P:>6d} | {fmt5(freqs_P[P])} | {d.max():>20.4f}")
print(" -> convergence monotone ; la base P=14 (defaut du depot) est deja a "
      f"{100*np.abs(freqs_P[14]/fref-1).max():.3f} % de P=18.")

print("\n" + "-" * 92)
print(" 1b. CONVERGENCE EN PENALISATION D'ENCASTREMENT (PX=PZ=18, plaque nue)")
print("-" * 92)
print(f" {'kw [N/m2]':>10} {'kr [N]':>8} | {'f1':>9} {'f2':>9} {'f3':>9} {'f4':>9} {'f5':>9} |"
      f" {'max|df| vs kw=1e14 [%]':>23}")
freqs_pen = {}
for kw, kr in PEN_LIST:
    freqs_pen[(kw, kr)] = ritz(P=18, kw=kw, kr=kr).freq_n.copy()
fpen_ref = freqs_pen[PEN_LIST[-1]]
for kw, kr in PEN_LIST:
    d = 100.0 * np.abs(freqs_pen[(kw, kr)] / fpen_ref - 1.0)
    print(f" {kw:>10.0e} {kr:>8.0e} | {fmt5(freqs_pen[(kw, kr)])} | {d.max():>23.4f}")
F_RITZ = freqs_P[18]                          # reference convergee du depot
print(" -> l'encastrement penalise converge par le BAS (frequences trop basses) ; a (1e12,1e8) il reste "
      f"{100*np.abs(F_RITZ/fpen_ref-1).max():.3f} % de souplesse residuelle.")
print(" REFERENCE RITZ CONVERGEE (P=18, kw=1e12, kr=1e8) :", fmt5(F_RITZ))

# ===========================================================================
# 2. CONTRE-VERIFICATION EF KIRCHHOFF-Q4 INDEPENDANTE
# ===========================================================================
print("\n" + "-" * 92)
print(" 2. CONTRE-VERIFICATION : EF KIRCHHOFF-Q4 (simulation/sim_kit) - plaque NUE")
print("-" * 92)
print(f" {'N1 x N2':>9} {'ndof':>7} | {'f1':>9} {'f2':>9} {'f3':>9} {'f4':>9} {'f5':>9}")
freqs_fe = {}
for N1, N2 in MESH_LIST:
    fe = PlateModel(LP, HP, BP, RHO_T1, E_T1, NU_T1, N1=N1, N2=N2,
                    n_modes=5, zeta_modes=list(ZETA), verbose=False)
    freqs_fe[(N1, N2)] = fe.freq_n.copy()
    print(f" {N1:>4d} x{N2:>3d} {3*(N1+1)*(N2+1):>7d} | {fmt5(fe.freq_n)}")
F_FE = freqs_fe[MESH_LIST[-1]]
print(" -> maillage Q4 converge par le HAUT (il n'y a pas d'encastrement penalise :")
print("    les ddl du bord z=0 sont bloques exactement).")

d_ritz_fe = 100.0 * (F_RITZ / F_FE - 1.0)
d_ritz_fe_stiff = 100.0 * (fpen_ref / F_FE - 1.0)
print("\n  comparaison des DEUX discretisations independantes (memes donnees Tableau 1) :")
print(f"   {'':<34}{'f1':>9} {'f2':>9} {'f3':>9} {'f4':>9} {'f5':>9}")
print(f"   {'Ritz P=18, kw=1e12 kr=1e8':<34}{fmt5(F_RITZ)}")
print(f"   {'Ritz P=18, kw=1e14 kr=1e10':<34}{fmt5(fpen_ref)}")
print(f"   {'EF Q4 60x48 (encast. exact)':<34}{fmt5(F_FE)}")
print(f"   {'ecart Ritz(1e12) / Q4  [%]':<34}{fmtp(d_ritz_fe)}")
print(f"   {'ecart Ritz(1e14) / Q4  [%]':<34}{fmtp(d_ritz_fe_stiff)}")
print(f" -> a penalisation raide l'accord est de {np.abs(d_ritz_fe_stiff).max():.4f} % au pire ;")
print(f"    a la penalisation par defaut du depot {np.abs(d_ritz_fe).max():.3f} %. Les deux")
print("    discretisations decrivent donc bien LE MEME continuum : le desaccord avec le")
print("    Tableau 4 ne peut pas venir du schema numerique.")

# ===========================================================================
# 3. IDENTIFICATION DE L'ECART RESIDUEL
# ===========================================================================
print("\n" + "-" * 92)
print(" 3. LE RAPPORT (Tableau 4 theorique) / (modele) EST-IL CONSTANT ?")
print("-" * 92)
r_conv = F_THEO / F_RITZ
r_P6 = F_THEO / freqs_P[6]
r_fe = F_THEO / F_FE
spread = lambda r: 100.0 * (r.max() / r.min() - 1.0)
print(f" {'':<30}{'r1':>9} {'r2':>9} {'r3':>9} {'r4':>9} {'r5':>9} | {'spread [%]':>11}")
for lab, r in [("Ritz P=18 (convergee)", r_conv),
               ("Ritz P=6  (tronquee)", r_P6),
               ("EF Q4 60x48", r_fe)]:
    print(f" {lab:<30}" + " ".join(f"{x:9.5f}" for x in r) + f" | {spread(r):>11.3f}")
print(" -> base convergee : rapport plat a %.2f %% pres (%.4f .. %.4f)."
      % (spread(r_conv), r_conv.min(), r_conv.max()))
print("    base tronquee P=6 : rapport disperse sur %.2f %% (%.4f .. %.4f)."
      % (spread(r_P6), r_P6.min(), r_P6.max()))
print("    Une erreur de troncature Ritz n'est JAMAIS independante du mode (elle raidit")
print("    d'autant plus que le mode est haut) : ici P=6 disperse ~10x plus que la base")
print("    convergee. Un rapport plat implique donc un SCALAIRE unique dans")
print("    omega ~ sqrt(D / (rho bP)) = (bP/2) sqrt(E / (3 rho (1-nu^2))).")

s_geo = float(np.exp(np.mean(np.log(r_conv))))    # moyenne geometrique
print("\n  facteur scalaire identifie (moyenne geometrique des 5 rapports) :")
print(f"    s = {s_geo:.5f}   ->  facteur sur E / (rho (1-nu^2)) : s^2 = {s_geo**2:.5f}")
print("    Le meme s^2 s'obtient par n'importe quelle combinaison equivalente :")
print(f"      E/rho              x {s_geo**2:.4f}  (ex. 70 GPa / 2700 vs 69 GPa / 2830 : "
      f"{(E_ID/RHO_ID)/(E_T1/RHO_T1):.4f})")
print(f"      bP                 x {s_geo:.4f}  (4.000 mm -> {1e3*BP*s_geo:.3f} mm)")
print(f"      lP et hP ensemble  x {1/np.sqrt(s_geo):.4f}  (100.0/80.0 mm -> "
      f"{1e3*LP/np.sqrt(s_geo):.2f}/{1e3*HP/np.sqrt(s_geo):.2f} mm)")
print("    -> c'est une IDENTIFICATION, pas une lecture des intentions des auteurs :")
print("       le jeu de frequences seul ne separe pas ces trois hypotheses.")

print("\n  Verification du jeu 'AL6061 manuel' E = 70 GPa, rho = 2700 kg/m3 (nu = 0.33) :")
f_id = ritz(P=18, E=E_ID, rho=RHO_ID).freq_n
f_t1 = F_RITZ
e_t1 = 100.0 * (f_t1 / F_THEO - 1.0)
e_id = 100.0 * (f_id / F_THEO - 1.0)
e_meas = 100.0 * (F_MEAS / F_THEO - 1.0)
rms = lambda e: float(np.sqrt(np.mean(e**2)))
print(f"   {'':<38}{'f1':>9} {'f2':>9} {'f3':>9} {'f4':>9} {'f5':>9} | {'RMS err [%]':>11}")
print(f"   {'Tableau 4 theorique (reference)':<38}{fmt5(F_THEO)} | {0.0:>11.3f}")
print(f"   {'Ritz, Tableau 1 (69 GPa, 2830)':<38}{fmt5(f_t1)} |")
print(f"   {'   erreur relative [%]':<38}{fmtp(e_t1)} | {rms(e_t1):>11.3f}")
print(f"   {'Ritz, identifie (70 GPa, 2700)':<38}{fmt5(f_id)} |")
print(f"   {'   erreur relative [%]':<38}{fmtp(e_id)} | {rms(e_id):>11.3f}")
print(f"   {'Tableau 4 mesure':<38}{fmt5(F_MEAS)} |")
print(f"   {'   erreur relative [%]':<38}{fmtp(e_meas)} | {rms(e_meas):>11.3f}")
print(f" -> le jeu identifie reproduit la colonne theorique du Tableau 4 a "
      f"{np.abs(e_id).max():.3f} % au pire sur les 5 modes,")
print(f"    contre {np.abs(e_t1).max():.2f} % avec les donnees litterales du Tableau 1.")

# ===========================================================================
# 4. PASTILLE PIEZO ET FREQUENCES MESUREES
# ===========================================================================
print("\n" + "-" * 92)
print(" 4. EFFET DE LA PASTILLE PIEZO COLLEE (QDA60-20-0.7, Tableau 2)")
print("-" * 92)
p_patch = ritz(P=18)
f_bare = p_patch.freq_n.copy()
p_patch.add_piezo_patch(**PATCH)
f_with = p_patch.freq_n.copy()
p_perf = ritz(P=18)
p_perf.add_piezo_patch(G_adh=None, **PATCH)     # collage parfait, eta = 1
f_perf = p_perf.freq_n.copy()
d_patch = 100.0 * (f_with / f_bare - 1.0)
print(f"   patch {1e3*(PATCH['x2']-PATCH['x1']):.0f} x {1e3*(PATCH['z2']-PATCH['z1']):.0f} mm "
      f"au coin bas droit ; rendement de collage eta = {p_patch.eta_bond:.4f}")
print(f"   rK (surcroit de raideur local) = {p_patch.rK_patch:.4f} ; "
      f"rM (surcroit de masse local) = {p_patch.rM_patch:.4f}")
print(f"   {'':<38}{'f1':>9} {'f2':>9} {'f3':>9} {'f4':>9} {'f5':>9}")
print(f"   {'plaque nue':<38}{fmt5(f_bare)}")
print(f"   {'avec pastille (eta = %.3f)' % p_patch.eta_bond:<38}{fmt5(f_with)}")
print(f"   {'avec pastille, collage parfait':<38}{fmt5(f_perf)}")
print(f"   {'decalage du a la pastille [%]':<38}{fmtp(d_patch)}")
print(f" -> mode 1 : {f_bare[0]:.1f} Hz -> {f_with[0]:.1f} Hz ({d_patch[0]:+.2f} %).")
print("    IMPORTANT : ce decalage est fortement DEPENDANT DU MODE (%.2f %% .. %.2f %%),"
      % (d_patch.min(), d_patch.max()))
print("    il ne peut donc pas produire le rapport plat de l'etape 3 : la pastille")
print("    n'est pas l'explication de l'ecart au Tableau 4.")
e_patch = 100.0 * (f_with / F_THEO - 1.0)
print(f"   {'erreur plaque+pastille vs Tab.4 theo [%]':<40}{fmtp(e_patch)}"
      f"  RMS {rms(e_patch):.3f} %")
print(f"   {'erreur plaque+pastille vs Tab.4 mesure [%]':<40}"
      f"{fmtp(100*(f_with/F_MEAS-1))}  RMS {rms(100*(f_with/F_MEAS-1)):.3f} %")

# ===========================================================================
# 5. CONSEQUENCE POUR LE MODELE DE COMMANDE
# ===========================================================================
print("\n" + "-" * 92)
print(" 5. CONSEQUENCE : 'calibrate_frequencies' EQUIVAUT-IL AU CHANGEMENT DE MATERIAU ?")
print("-" * 92)
# (i) Plaque NUE : un scalaire sur E/rho ne change pas la DIRECTION des vecteurs
#     propres (K -> aK, M -> bM). Seule la normalisation en masse bouge
#     (V ~ 1/sqrt(rho)). Le recalage du depot ne touche ni V ni H_Pe.
pa0 = ritz(P=18)
pb0 = ritz(P=18, E=E_ID, rho=RHO_ID)
Da0, Db0 = pa0.D_row(LP, HP), pb0.D_row(LP, HP)
cos0 = float(np.abs(Da0 @ Db0) / (np.linalg.norm(Da0) * np.linalg.norm(Db0)))
print("  (i) plaque NUE, Tableau 1 contre materiau identifie :")
print(f"      colinearite des deformees D_obs : cos = {cos0:.10f}  (invariance exacte)")
print(f"      rapport des D_obs mode a mode   : "
      + " ".join(f"{x:8.5f}" for x in Db0 / Da0)
      + f"   attendu sqrt(rho_T1/rho_ID) = {np.sqrt(RHO_T1/RHO_ID):.5f}")
print("      -> sur la plaque nue, 'Tableau 1 + recalage sur F_THEO' et 'materiau")
print(f"         identifie' donnent les MEMES frequences a {np.abs(e_id).max():.2f} % pres, mais des")
print(f"         residus modaux D*H differents d'un facteur {RHO_T1/RHO_ID:.4f} "
      f"({100*(RHO_T1/RHO_ID-1):+.1f} % de gain), que")
print("         le recalage de frequences ne peut pas capturer (il ne touche pas V).")

# (ii) Avec la pastille collee, l'invariance est rompue : rK depend de E, et le
#      Tableau 4 theorique est deja atteint par la plaque NUE identifiee.
pa = ritz(P=18); pa.add_piezo_patch(**PATCH); pa.calibrate_frequencies(F_THEO)
pb = ritz(P=18, E=E_ID, rho=RHO_ID); pb.add_piezo_patch(**PATCH)
Da, Db = pa.D_row(LP, HP), pb.D_row(LP, HP)
ra, rb = Da * pa.H_Pe_modal, Db * pb.H_Pe_modal
print("\n  (ii) plaque AVEC pastille (configuration reellement utilisee par le depot) :")
print(f"      depot : Tableau 1 + pastille + recalage F_THEO : {fmt5(pa.freq_n)}")
print(f"      materiau identifie + pastille (sans recalage) : {fmt5(pb.freq_n)}")
print(f"      ecart relatif [%]                             : "
      f"{fmtp(100*(pb.freq_n/pa.freq_n-1))}")
print(f"      residus D_obs*H_Pe, depot     : " + " ".join(f"{x:9.4f}" for x in ra))
print(f"      residus D_obs*H_Pe, identifie : " + " ".join(f"{x:9.4f}" for x in rb))
print(f"      rapport identifie / depot     : " + " ".join(f"{x:9.4f}" for x in rb / ra))
print("      -> les deux voies NE coincident PAS une fois la pastille collee : le")
print("         materiau identifie reproduit deja le Tableau 4 theorique SANS pastille,")
print("         donc y ajouter le raidissement du patch fait depasser les cibles de")
print(f"         {np.abs(100*(pb.freq_n/pa.freq_n-1)).max():.1f} % au pire. Autrement dit, la colonne 'theoretical' du")
print("         Tableau 4 est coherente avec une plaque NUE, pas avec la plaque")
print("         instrumentee decrite au paragraphe 4.1 du papier.")
print("      -> mode 3 fait exception (rapport de residus %.3f) : son residu au capteur"
      % (rb[2] / ra[2]))
print("         est quasi nul (%.4f), donc tres sensible ; l'ajout du patch rompt aussi"
      % ra[2])
print("         l'invariance exacte des deformees (rK depend de E).")

# ===========================================================================
# FIGURE
# ===========================================================================
fig, ax = plt.subplots(2, 2, figsize=(13.0, 9.0))

# (a) convergence en taille de base
a = ax[0, 0]
for k in range(5):
    err = np.array([100 * abs(freqs_P[P][k] / fref[k] - 1) for P in P_LIST])
    err = np.maximum(err, 1e-5)
    a.semilogy(P_LIST, err, 'o-', ms=4, label=f"mode {k+1} ({fref[k]:.0f} Hz)")
a.axhline(0.01, color='0.5', ls=':', lw=1)
a.text(6.2, 0.0115, "0.01 %", color='0.4', fontsize=8)
a.set_xlabel("Chebyshev basis size  PX = PZ")
a.set_ylabel("| f(P) / f(P=18) - 1 |   [%]")
a.set_title("(a) Ritz convergence, bare plate (kw=1e12, kr=1e8)")
a.grid(True, which='both', alpha=0.3)
a.legend(fontsize=8)

# (b) erreur relative des jeux de parametres contre la colonne theorique
b = ax[0, 1]
modes = np.arange(1, 6)
w = 0.2
b.bar(modes - 1.5 * w, e_t1, w, label="Ritz, Table 1 (69 GPa, 2830)", color='#c0392b')
b.bar(modes - 0.5 * w, e_id, w, label="Ritz, identified (70 GPa, 2700)", color='#27ae60')
b.bar(modes + 0.5 * w, e_patch, w, label="Ritz, Table 1 + bonded patch", color='#8e44ad')
b.bar(modes + 1.5 * w, e_meas, w, label="Table 4 measured", color='#2980b9')
b.axhline(0, color='k', lw=1)
b.set_xticks(modes)
b.set_xlabel("mode")
b.set_ylabel("relative error vs Table 4 theoretical  [%]")
b.set_title("(b) Which parameter set reproduces the paper's theory?")
b.grid(True, axis='y', alpha=0.3)
_lo = min(e_t1.min(), e_patch.min(), e_meas.min())
_hi = max(e_meas.max(), e_id.max(), e_patch.max()) + 0.55
b.set_ylim(_lo - 1.35, _hi)
for x, v in zip(modes - 0.5 * w, e_id):          # etiquettes du jeu identifie
    b.text(x, _hi - 0.13, f"{v:+.2f}", ha='center', va='top',
           fontsize=7.5, color='#1e7e40')
b.legend(fontsize=8, loc='lower center', ncol=2, framealpha=0.95)

# (c) rapport theo/modele : plat contre disperse
c = ax[1, 0]
c.plot(modes, r_conv, 'o-', color='#27ae60', ms=7,
       label=f"converged basis P=18  (spread {spread(r_conv):.2f} %)")
c.plot(modes, r_P6, 's--', color='#c0392b', ms=7,
       label=f"truncated basis P=6   (spread {spread(r_P6):.2f} %)")
c.plot(modes, f_with / f_bare, '^:', color='#8e44ad', ms=7,
       label=f"bonded-patch shift     (spread {spread(f_with/f_bare):.2f} %)")
c.axhline(s_geo, color='#27ae60', ls=':', lw=1)
c.text(4.6, s_geo + 0.0012, f"s = {s_geo:.4f}", color='#1e7e40', fontsize=8)
c.axhspan(r_conv.min(), r_conv.max(), color='#27ae60', alpha=0.12)
c.set_xticks(modes)
c.set_xlabel("mode")
c.set_ylabel("ratio  f(Table 4 theory) / f(model)")
c.set_title("(c) A flat ratio = one scalar in sqrt(D/(rho b)), not truncation")
c.grid(True, alpha=0.3)
c.legend(fontsize=8)

# (d) Ritz contre EF Q4
d = ax[1, 1]
xb = np.arange(5)
d.bar(xb - 0.19, F_RITZ, 0.38, label="Chebyshev-Ritz P=18", color='#2c7fb8')
d.bar(xb + 0.19, F_FE, 0.38, label="Kirchhoff-Q4 FE 60x48", color='#f39c12')
for i in range(5):
    d.annotate(f"{d_ritz_fe[i]:+.3f} %", (xb[i], max(F_RITZ[i], F_FE[i])),
               textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8)
d.set_xticks(xb)
d.set_xticklabels([f"mode {i+1}" for i in range(5)])
d.set_ylabel("bare-plate natural frequency  [Hz]")
d.set_ylim(0, 1.16 * max(F_FE.max(), F_RITZ.max()))
d.set_title("(d) Two independent discretizations of the same continuum")
d.grid(True, axis='y', alpha=0.3)
d.legend(fontsize=8, loc='upper left')

fig.suptitle("Chebyshev-Ritz plate model of Du et al. (2024): convergence, "
             "cross-check and identification of the Table-4 gap", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
outdir = os.path.join(HERE, '..', 'figures', 'verification')
os.makedirs(outdir, exist_ok=True)
outpng = os.path.abspath(os.path.join(outdir, '09_ritz_model_identification.png'))
fig.savefig(outpng, dpi=140)
print("\nfigure -> " + outpng)
print("=" * 92)
