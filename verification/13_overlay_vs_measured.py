"""Superposition COMPLETE modele / mesure — la comparaison courbe contre courbe.

Les autres scripts comparent des CARACTERISTIQUES isolees (pics, creux). Celui-ci
superpose les courbes entieres aux deux mesures numerisees de la Fig. 12 et
chiffre l'accord sur toute la bande, ce qui est la seule facon honnete de dire
si "la nmodelisation concorde".

Sortie : figure PNG + tableau d'ecarts.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import openpyxl

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, 'simulation', 'sim_kit'),
                os.path.join(_R, 'simulation')]
os.chdir(os.path.join(_R, 'simulation', 'sim_kit'))

import simulation_base as SB                                    # noqa: E402
from simulation_base import SimBase                             # noqa: E402
import glob                                                     # noqa: E402


def load(pattern):
    ws = openpyxl.load_workbook(glob.glob(pattern)[0]).worksheets[0]
    d = np.array([[r[0], r[1]] for r in ws.iter_rows(values_only=True)
                  if r[0] is not None], float)
    return d[:, 0], d[:, 1]


fa, Aa = load('experimental*')          # Fig.12(a), dB re 1 um/N
fb, Ab = load('transfer_functions*')    # Fig.12(b), dB re 0.01 um/V

sim = SimBase()
p = sim.plate
om = np.asarray(p.omega_n)
z = np.asarray(p.zeta_modes)
Do = np.asarray(p.D_obs)
Hp = np.asarray(p.H_Pe_modal)


def frf(res, f):
    w = 2*np.pi*f
    den = ((om[:, None]**2 - w[None, :]**2)
           + 2j*z[:, None]*om[:, None]*w[None, :])
    return np.abs(np.sum(res[:, None]/den, axis=0))


Ga = frf(Do*Do, fa)*1e6                 # um/N   (reception au point de frappe)
Gb = frf(Do*Hp, fb)*1e6/0.01            # en unites de 0.01 um/V
Ga_dB = 20*np.log10(Ga)
Gb_dB = 20*np.log10(Gb)

# Le niveau absolu de la Fig. 12 est inexploitable (cf. F9) : on compare donc
# les FORMES, en recalant chaque courbe sur sa mediane.
oa = np.median(Aa) - np.median(Ga_dB)
ob = np.median(Ab) - np.median(Gb_dB)
print('--- accord sur toute la bande, apres recalage du niveau median ---')
print(f'  Fig.12(a)  decalage de niveau : {oa:+6.1f} dB  '
      f'(= facteur {10**(oa/20):.2f} en amplitude)')
print(f'  Fig.12(b)  decalage de niveau : {ob:+6.1f} dB')
for lab, fm, Am, Gm, off in [('Fig.12(a) choc   ', fa, Aa, Ga_dB, oa),
                             ('Fig.12(b) tension', fb, Ab, Gb_dB, ob)]:
    m = (fm > 100) & (fm < 4600)
    e = Am[m] - (Gm[m] + off)
    print(f'  {lab} : ecart RMS = {np.sqrt(np.mean(e**2)):5.1f} dB, '
          f'median |ecart| = {np.median(np.abs(e)):5.1f} dB, '
          f'max = {np.max(np.abs(e)):5.1f} dB   (n = {m.sum()} points)')

ff = np.linspace(60, 4900, 40000)
fig, ax = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
ax[0].plot(fa, Aa, 'o-', ms=3, lw=1.2, color='#1f4e79', label='mesure Du et al., Fig. 12(a)')
ax[0].plot(ff, 20*np.log10(frf(Do*Do, ff)*1e6) + oa, '-', lw=1.6,
           color='#c00000', label=f'modele (recale de {oa:+.0f} dB)')
ax[0].set_ylabel('reception au point de frappe [dB]')
ax[0].set_title('Superposition modele / mesure — coin superieur droit')
ax[1].plot(fb, Ab, 'o-', ms=3, lw=1.2, color='#1f4e79', label='mesure Du et al., Fig. 12(b)')
ax[1].plot(ff, 20*np.log10(frf(Do*Hp, ff)*1e6/0.01) + ob, '-', lw=1.6,
           color='#c00000', label=f'modele (recale de {ob:+.0f} dB)')
ax[1].set_ylabel('transfert tension -> deplacement [dB]')
ax[1].set_xlabel('frequence [Hz]')
for a in ax:
    a.grid(alpha=.3); a.legend(loc='upper right', fontsize=9); a.set_xlim(0, 4900)
for f0 in SB.F_MEASURED:
    for a in ax:
        a.axvline(f0, color='0.6', lw=.7, ls=':')
out = os.path.join(_R, 'verification', 'overlay_model_vs_measured.png')
fig.tight_layout(); fig.savefig(out, dpi=130)
print(f'\nfigure : {out}')
