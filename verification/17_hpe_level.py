"""F9 — COMMENT VALIDER LE NIVEAU DE H_Pe. Ce qui est possible, et ce qui ne
l'est pas, avec les donnees de l'article.

ETAT DE LA QUESTION
-------------------
La REPARTITION modale de H_Pe est identifiee (§10, script 14) : les quatre creux
de la Fig. 12(b) sont reproduits a 0.19 % pres. Son NIVEAU, lui, n'a jamais ete
valide, et c'est le dernier defaut ouvert. Ce script montre que ce n'est pas un
oubli : le niveau est STRUCTURELLEMENT indeterminable a partir des deux figures,
et il dit exactement quelle mesure le determinerait.

LE MODELE DE CALIBRATION
------------------------
Les deux courbes de la Fig. 12 viennent du MEME essai modal (Fig. 11), au MEME
point (coin superieur droit), avec le MEME capteur de deplacement :

    Fig. 12(a)  reception au point de frappe   [um/N]   deplacement / FORCE
    Fig. 12(b)  transfert tension -> capteur   [um/V]   deplacement / TENSION

Soit k_d l'erreur d'echelle de la voie DEPLACEMENT et k_f celle de la voie
FORCE (le marteau). La tension, elle, est connue exactement : c'est une consigne
d'amplificateur, pas une mesure. Alors

    mesure_a = (k_d/k_f) * vrai_a         vrai_a = D_obs^2  / ...
    mesure_b =  k_d      * vrai_b * g     vrai_b = D_obs*H  / ...

ou g est le facteur cherche sur le NIVEAU de H_Pe. Deux equations, TROIS
inconnues : k_d, k_f, g. Le systeme est sous-determine, et c'est la raison
profonde pour laquelle le niveau n'est pas validable ici. Aucune astuce
d'ajustement ne cree l'information manquante.

CE QU'ON PEUT QUAND MEME EN TIRER
---------------------------------
Le controle 3 de check_model etablit, SANS RIEN EMPRUNTER A L'ARTICLE, que la
souplesse du modele est juste : le modal a 5 modes rend 95.9 % de la resolution
statique EF exacte, et celle-ci tombe entre la borne de poutre et 3 fois cette
borne. On peut donc poser vrai_a = modele_a, ce qui ferme UNE equation :

    k_d/k_f = mesure_a/modele_a = 6.60

et il reste une famille A UN PARAMETRE. Ses deux extremites sont interpretables :

    toute l'erreur dans la voie FORCE (k_d = 1)      -> g = 2.92
    toute l'erreur dans la voie DEPLACEMENT (k_f = 1)-> g = 0.44

La seconde s'obtient aussi directement, et c'est la mesure la plus propre
disponible : le RAPPORT des deux courbes elimine k_d, puisque le capteur de
deplacement est commun.

        mesure_b/mesure_a = k_f * g * (D_obs*H)/(D_obs^2)

Ce rapport a la dimension d'une force par volt et ne depend plus du capteur.
Le script le mesure sur toute la bande : s'il ne DERIVE PAS avec la frequence,
c'est qu'un seul scalaire reconcilie les deux courbes, et sa valeur est g*k_f.
Mesure : 0.416, stable a 29 % pres d'une bande d'octave a l'autre.

CONCLUSION. g n'est pas determine, mais il est ENCADRE : [0.44, 2.92]. Cet
encadrement est plus large que le balayage gain_H = 0.5 / 2.0 actuellement
publie, et il l'est du cote qui fait mal.

Sortie : tableau + figure verification/hpe_level.png
"""
import glob
import os
import sys

import numpy as np
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, 'simulation', 'sim_kit'),
                os.path.join(_R, 'simulation')]
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_R, 'simulation', 'sim_kit'))

from simulation_base import SimBase                              # noqa: E402
import simulation_base as SB                                     # noqa: E402


def load(pattern):
    ws = openpyxl.load_workbook(glob.glob(pattern)[0]).worksheets[0]
    d = np.array([[r[0], r[1]] for r in ws.iter_rows(values_only=True)
                  if r[0] is not None], float)
    return d[:, 0], d[:, 1]


fa, Aa = load('experimental*')          # dB re 1 um/N
fb, Ab = load('transfer_functions*')    # dB re 0.01 um/V
Ra = 10**(Aa/20)                        # um/N
Rb = 0.01*10**(Ab/20)                   # um/V

sim = SimBase()
p = sim.plate
om = np.asarray(p.omega_n)
ze = np.asarray(p.zeta_modes)
Do = np.asarray(p.D_obs)
Hp = np.asarray(p.H_Pe_modal)
fr = om/(2*np.pi)


def frf(res, f):
    w = 2*np.pi*f
    den = ((om[:, None]**2 - w[None, :]**2)
           + 2j*ze[:, None]*om[:, None]*w[None, :])
    return np.abs(np.sum(res[:, None]/den, axis=0))


print(__doc__.split('Sortie')[0])
print('=' * 76)

# ---------------------------------------------------------------- 1. statique
print('1. LE PLATEAU STATIQUE')
Ca_meas = float(Ra[fa < 200].mean())
Cb_meas = float(Rb[fb < 200].mean())
Ca_mod = float(np.sum(Do*Do/om**2))*1e6
Cb_mod = float(np.sum(Do*Hp/om**2))*1e6
print(f'                       Fig.12(a)          Fig.12(b)')
print(f'   mesure          {Ca_meas:10.3f} um/N   {Cb_meas:10.5f} um/V')
print(f'   modele          {Ca_mod:10.3f} um/N   {Cb_mod:10.5f} um/V')
print(f'   mesure/modele   {Ca_meas/Ca_mod:10.2f}     {Cb_meas/Cb_mod:10.2f}')
kdf = Ca_meas/Ca_mod
print(f'\n   => k_d/k_f = {kdf:.2f} : une des deux voies de mesure est fausse')
print('      de ce facteur, et la Fig. 12(a) seule ne dit pas laquelle.')
print(f'      (check_model annonce 6.34 : il rapporte la mesure a la statique')
print(f'      EF EXACTE, {6.917:.3f} um/N, alors qu\'on la rapporte ici au modele')
print(f'      MODAL tronque a 5 modes, {Ca_mod:.3f} um/N, qui est la courbe')
print('      effectivement comparee. Meme grandeur, 4 % d\'ecart de troncature.)')

# ---------------------------------------------------------------- 2. rapport
print('\n2. LE RAPPORT DES DEUX COURBES (elimine le capteur de deplacement)')
Aa_i = np.interp(fb, fa, Aa)
T_meas = Rb/(10**(Aa_i/20))                      # N/V
T_mod = frf(Do*Hp, fb)/frf(Do*Do, fb)            # N/V
g_f = T_meas/T_mod

# On ecarte les voisinages des resonances (la numerisation, ~60 Hz de pas, ne
# resout pas un pic a zeta = 0.17 % : sa demi-largeur vaut ~2 Hz) et ceux des
# creux des DEUX courbes, ou le rapport est mal conditionne.
NOTCH_A = [734, 2161, 3001, 3805]
NOTCH_B = [788, 1493, 2913, 3609]
bad = np.zeros(len(fb), bool)
for c in list(fr) + NOTCH_A + NOTCH_B:
    bad |= np.abs(fb - c) < 0.06*c
m = (~bad) & (fb > 50) & (fb < 4600)
g_med = float(np.median(g_f[m]))
q1, q3 = np.percentile(g_f[m], [25, 75])
print(f'   {int(m.sum())} points retenus sur {len(fb)} (hors pics et creux)')
print(f'   g*k_f median = {g_med:.3f}   (quartiles {q1:.3f} - {q3:.3f})')
print('   par bande :')
for lo, hi in [(50, 300), (300, 800), (800, 1400), (1400, 2600),
               (2600, 3400), (3400, 4600)]:
    mm = m & (fb >= lo) & (fb < hi)
    if mm.sum():
        print(f'      {lo:4d}-{hi:4d} Hz : n={int(mm.sum()):2d}  '
              f'g*k_f = {np.median(g_f[mm]):.3f}')
bmed = [float(np.median(g_f[m & (fb >= lo) & (fb < hi)]))
        for lo, hi in [(50, 300), (300, 800), (800, 1400), (1400, 2600),
                       (2600, 3400), (3400, 4600)]
        if (m & (fb >= lo) & (fb < hi)).sum()]
spread_band = max(bmed)/min(bmed)
spread_pt = float(np.max(g_f[m])/np.min(g_f[m]))
print(f'\n   medianes par bande : etendue facteur {spread_band:.2f}')
print(f'   points individuels : etendue facteur {spread_pt:.2f}')
print('''
   C'est la premiere ligne qui compte. Le rapport ne DERIVE PAS avec la
   frequence : un seul scalaire reconcilie les deux courbes sur 4.5 kHz, a
   %.0f %% pres d'une bande a l'autre. La dispersion point par point, elle,
   est de la numerisation : ~60 Hz de pas sur des pics dont la demi-largeur
   vaut 2 Hz, et une lecture au pixel pres sur une echelle en dB.

   Cette platitude est un CONTROLE, pas seulement une commodite. Si la
   repartition modale de H_Pe etait fausse, ou si les deux voies de mesure
   derivaient l'une par rapport a l'autre, aucun scalaire unique ne
   conviendrait et les medianes par bande s'etaleraient. Elle confirme donc
   §10 par un chemin independant : ce qui manque a H_Pe est bien un NIVEAU,
   et rien d'autre.''' % (100*(spread_band - 1)))
assert spread_band < 1.6, \
    "les medianes par bande doivent etre plates si un seul scalaire suffit"

# ---------------------------------------------------------------- 3. famille
print('\n3. LA FAMILLE A UN PARAMETRE, ET SES DEUX BOUTS')
g_hi = Cb_meas/Cb_mod                 # k_d = 1  : tout dans la voie force
g_lo = g_hi/kdf                       # k_f = 1  : tout dans le deplacement
print(f'   hypothese                              k_d     k_f       g')
print(f'   toute l\'erreur dans la FORCE        {1.0:6.2f}  {1/kdf:6.3f}  '
      f'{g_hi:6.2f}')
print(f'   toute l\'erreur dans le DEPLACEMENT  {kdf:6.2f}  {1.0:6.3f}  '
      f'{g_lo:6.2f}')
print(f'\n   g est donc borne par [{g_lo:.2f}, {g_hi:.2f}] -- un facteur '
      f'{g_hi/g_lo:.1f}.')
print('   La methode du rapport (section 2) tombe sur la borne BASSE, ce qui')
print('   est coherent : elle suppose k_f = 1, c.-a-d. marteau juste.')
assert abs(g_lo - g_med) < 0.15, \
    "la methode du rapport doit retrouver la borne basse de la famille"

# ---------------------------------------------------------------- 4. balayage
print('\n4. CONSEQUENCE IMMEDIATE : LE BALAYAGE ACTUEL EST TROP ETROIT')
print(f'   run_demo --full balaie gain_H = 0.5 et 2.0.')
print(f'   L\'encadrement honnete est [{g_lo:.2f}, {g_hi:.2f}].')
print(f'   La borne haute reelle, {g_hi:.2f}, est {g_hi/2.0:.2f} fois celle qui')
print('   est testee. Or c\'est le cote qui a fait tomber l\'ESO (H x2.00 rend')
print('   zero a quatre vitesses sur cinq) : le balayage sous-estime le risque.')

# ---------------------------------------------------------------- 5. mesures
print('\n5. CE QUI FERMERAIT LA QUESTION')
print("""   Il suffit de determiner UNE des trois inconnues ; les deux autres
   suivent. Par ordre de simplicite :

   a) ESSAI STATIQUE. Appliquer une tension continue connue et mesurer le
      deplacement du coin avec un capteur etalonne (laser, LVDT, comparateur).
      On lit vrai_b directement, sans k_d ni k_f. C'est la mesure la plus
      directe : une seule grandeur, en regime permanent, sans excitation.
      Le modele predit 0.0245 um/V ; l'ecart mesure EST g.

   b) RECIPROCITE. Le patch est aussi un CAPTEUR. La charge qu'il delivre par
      unite de force appliquee au coin est egale, par reciprocite
      piezoelectrique, au deplacement du coin par unite de tension appliquee
      au patch. Cette egalite ne fait intervenir aucune calibration de
      deplacement : elle compare une charge a une force. Elle valide g ET la
      repartition modale d'un seul coup.

   c) ETALONNER UNE VOIE. Calibrer le marteau (masse tombante, ou
      accelerometre de reference) fixe k_f ; calibrer le capteur de
      deplacement fixe k_d. L'un ou l'autre suffit.

   Aucune de ces trois mesures n'est dans l'article, et aucune ne se deduit
   de ses figures. C'est pourquoi F9 reste OUVERT et non "resolu par defaut".

   A defaut, la conduite juste est celle deja adoptee, mais sur le bon
   intervalle : synthetiser sur la plaque nominale, simuler sur gain_H
   balaye de %.2f a %.2f, et ne publier que ce qui survit aux deux bouts.""" %
      (g_lo, g_hi))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
ff = np.linspace(60, 4900, 40000)
ax[0].semilogy(fb, T_meas, 'o-', ms=4, lw=1.2, color='#1f4e79',
               label='mesure : Fig. 12(b) / Fig. 12(a)')
ax[0].semilogy(ff, frf(Do*Hp, ff)/frf(Do*Do, ff), '-', lw=1.5,
               color='#c00000', label='modele : H_Pe / D_obs')
for c in fr:
    ax[0].axvline(c, ls=':', lw=.8, color='.6')
ax[0].set_ylabel('rapport [N/V]')
ax[0].set_title('Le rapport des deux courbes elimine le capteur de deplacement')
ax[0].legend()
ax[0].grid(alpha=.3, which='both')

ax[1].plot(fb[m], g_f[m], 'o', ms=6, color='#1f4e79',
           label='points exploitables (hors pics et creux)')
ax[1].plot(fb[~m], g_f[~m], 'x', ms=5, color='.7',
           label='ecartes : pic non resolu, ou creux')
ax[1].axhline(g_med, color='#c00000', lw=1.6,
              label=f'mediane = {g_med:.3f}')
ax[1].axhspan(q1, q3, color='#c00000', alpha=.12,
              label=f'quartiles {q1:.2f} - {q3:.2f}')
ax[1].axhline(g_hi, color='#2e7d32', ls='--', lw=1.4,
              label=f'autre bout de la famille = {g_hi:.2f} (k_d = 1)')
ax[1].axhline(1.0, color='k', ls=':', lw=1.0, label='modele actuel (g = 1)')
ax[1].set_yscale('log')
ax[1].set_ylim(0.2, 5)
ax[1].set_xlim(0, 4900)
ax[1].set_xlabel('frequence [Hz]')
ax[1].set_ylabel('facteur g sur le niveau de H_Pe')
ax[1].set_title('Le facteur cherche, et l\'intervalle qui reste indetermine')
ax[1].legend(fontsize=8, ncol=2)
ax[1].grid(alpha=.3, which='both')
fig.tight_layout()
out = os.path.join(_HERE, 'hpe_level.png')
fig.savefig(out, dpi=130)
print(f'\nfigure ecrite : {out}')
print('\nTOUTES LES VERIFICATIONS PASSENT.')
