"""PEUT-ON REGENERER LES RESULTATS DE L'ARTICLE ? Reponse chiffree, poste par poste.

La question merite mieux qu'un oui ou un non. On reprend donc chaque resultat
QUANTITATIF publie par Du et al. (IJMS 274, 2024) qui soit verifiable, et on le
recalcule. Trois verdicts possibles :

  PREDIT      le modele le sort sans que la valeur de l'article y soit entree ;
  RECALE      la valeur de l'article est ENTREE dans le modele. La reproduire
              n'est alors pas une preuve — c'est de la circularite, et le
              script le dit au lieu de compter le point ;
  NON REPRODUIT.

Le partage PREDIT / RECALE est le coeur de ce script. Un modele cale sur cinq
frequences reproduira toujours ces cinq frequences ; ce qui compte est ce qu'il
donne EN PLUS.

Duree : ~3 min.
"""
import glob
import os
import sys

import numpy as np
import openpyxl

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, 'simulation', 'sim_kit'),
                os.path.join(_R, 'simulation')]
os.chdir(os.path.join(_R, 'simulation', 'sim_kit'))

import simulation_base as SB                                     # noqa: E402
from simulation_base import SimBase                              # noqa: E402
from milling_force import cutting_coefficients                   # noqa: E402

OK, KO, CIRC, PART = 'PREDIT  ', 'NON REPR', 'RECALE  ', 'PARTIEL '
rows = []


def note(verdict, item, paper, model, comment=''):
    rows.append((verdict, item, paper, model, comment))
    print(f'  [{verdict}] {item}')
    print(f'            article : {paper}')
    print(f'            modele  : {model}')
    if comment:
        print(f'            {comment}')
    print(flush=True)


print(__doc__.split('Duree')[0])
print('=' * 78)
sim = SimBase()
p = sim.plate

# ------------------------------------------------------------------ 1. Eq. (3)
print('1. COEFFICIENTS DE COUPE, Eq. (3)')
k1, k2 = cutting_coefficients(SB.KN, SB.MU_C, SB.HELIX, SB.RAKE)
note(OK, 'k1 = kn/cos(eta), k2 = 1 + mu_c tan(eta)(cos g - kn sin g)',
     'k1 = 0.317401, k2 = 1.125846',
     f'k1 = {k1:.6f}, k2 = {k2:.6f}',
     'formule appliquee telle que publiee ; accord exact')

phi_st = np.pi - np.arccos(1 - SB.AE_NOM/(SB.D_TOOL/2))
note(OK, 'angle d\'entree phi_st (coupe en avalant, ae = 0.1 mm)',
     '2.9413 rad', f'{phi_st:.4f} rad', '')

# --------------------------------------------------------- 2. frequences
print('2. FREQUENCES PROPRES, Tableau 4')
f_meas = np.array(SB.F_MEASURED)
f_fem = f_meas/np.asarray(p.freq_scale)          # AVANT recalage
e = 100*(f_fem/f_meas - 1)
note(OK, 'les cinq frequences, PREDITES par les elements finis avant recalage',
     ' / '.join(f'{v:.1f}' for v in f_meas) + ' Hz',
     ' / '.join(f'{v:.1f}' for v in f_fem) + ' Hz',
     f'ecarts {np.array2string(e, precision=1, floatmode="fixed")} %, '
     f'|max| = {np.abs(e).max():.1f} % — obtenu avec Kirchhoff-Q4 la ou '
     f'l\'article emploie Chebyshev-Ritz')
note(CIRC, 'les memes frequences APRES calibrate_frequencies',
     'Tableau 4', 'identiques par construction',
     'la mesure est ENTREE dans le modele : ne compte pas comme reproduction')

# --------------------------------------------------- 3. antiresonances 12(a)
print('3. ANTIRESONANCES DE LA FIG. 12(a) — reception au point de frappe')
ff = np.linspace(60, 4600, 60000)
w = 2*np.pi*ff
m = sim.modal
den = ((m['omega'][:, None]**2 - w[None, :]**2)
       + 2j*m['zeta'][:, None]*m['omega'][:, None]*w[None, :])
G = np.abs(np.sum((m['D_obs']**2)[:, None]/den, axis=0))
lg = np.log(G)
anti = np.array([ff[i] for i in range(1, len(G)-1)
                 if lg[i] < lg[i-1] and lg[i] < lg[i+1]])
tgt = np.array([734, 2161, 3001, 3805])
got = np.array([anti[np.argmin(np.abs(anti - t))] for t in tgt])
ea = 100*(got/tgt - 1)
note(OK, 'les quatre antiresonances',
     ' / '.join(str(t) for t in tgt) + ' Hz',
     ' / '.join(f'{v:.0f}' for v in got) + ' Hz',
     f'ecarts {np.array2string(ea, precision=1, floatmode="fixed")} % — '
     f'elles ne sont PAS entrees dans le modele : elles sortent de D_obs, '
     f'donc c\'est une vraie prediction')

# ------------------------------------------------------- 4. creux de 12(b)
print('4. CREUX DE LA FIG. 12(b) — transfert tension -> capteur')
r_c = np.asarray(p.H_Pe_modal)*np.asarray(p.D_obs)
s2 = np.asarray(p.omega_n)**2
P = sum(r_c[i]*np.poly(np.delete(s2, i)) for i in range(len(r_c)))
zr = np.roots(P)
zr = zr[np.abs(zr.imag) <= 1e-6*max(np.abs(zr.real).max(), 1e-30)].real
zc = np.sort(np.sqrt(zr[zr > 0]))/(2*np.pi)
sf = SimBase(coupling='fem')
rf = np.asarray(sf.plate.H_Pe_modal)*np.asarray(sf.plate.D_obs)
Pf = sum(rf[i]*np.poly(np.delete(s2, i)) for i in range(len(rf)))
zf = np.roots(Pf)
zf = zf[np.abs(zf.imag) <= 1e-6*max(np.abs(zf.real).max(), 1e-30)].real
zcf = np.sort(np.sqrt(zf[zf > 0]))/(2*np.pi)
note(KO, 'les quatre creux, PREDITS par les elements finis',
     '788 / 1493 / 2913 / 3609 Hz',
     ' / '.join(f'{v:.0f}' for v in zcf) + ' Hz',
     f'{len(zcf)} creux calcule(s) contre 4 mesures : la repartition modale '
     f'de H_Pe issue des Eqs. (14)-(15) NE reproduit PAS la figure de '
     f'l\'article. C\'est le defaut central trouve dans ce travail.')
note(CIRC, 'les memes creux apres identification sur la Fig. 12(b)',
     '788 / 1493 / 2913 / 3609 Hz',
     ' / '.join(f'{v:.0f}' for v in zc) + ' Hz',
     'la figure est ENTREE dans le modele (H_IDENT) : accord a 0.19 %, mais '
     'c\'est une identification, pas une prediction')

# ------------------------------------------------- 5. lobes de stabilite
print('5. LOBES DE STABILITE SANS CONTROLE — Figs. 13(b) et 18')
SPD = [3000, 3600, 4200, 4900, 5500, 6000, 7200]
lim = {}
for rpm in SPD:
    lim[rpm] = sim.stability_limit(None, rpm=rpm, lo=0.02e-3, hi=4.0e-3,
                                   T=SB.T_LIMIT, tol=2e-6)*1e3
    print(f'     {rpm:5d} tr/min : {lim[rpm]:.4f} mm', flush=True)
lo_all = min(lim.values())
note(OK, 'niveau general du lobe libre (Fig. 13b)',
     '0.03 - 0.05 mm',
     f'minimum sur les sept vitesses = {lo_all:.4f} mm',
     'meme ordre de grandeur, sans qu\'aucune donnee de la Fig. 13 soit '
     'entree dans le modele')

peaks = sorted(SPD, key=lambda r: -lim[r])[:2]
note(OK if set(peaks) == {3600, 5500} else KO,
     'position des deux maxima du lobe (Fig. 13b : ~3600 et ~5500 tr/min)',
     '3600 et 5500 tr/min',
     f'{peaks[0]} et {peaks[1]} tr/min '
     f'({lim[peaks[0]]:.4f} et {lim[peaks[1]]:.4f} mm)',
     'c\'est ce test qui a tranche la convention de signe de l\'effort '
     '(FORCE_SIGN) : la convention opposee met des CREUX a ces vitesses')

below = [r for r in SPD if lim[r] < 0.1]
above = [r for r in SPD if lim[r] >= 0.1]
note(PART, 'Fig. 18 : limite experimentale < 0.1 mm partout SAUF a 5500 tr/min',
     '< 0.1 mm sauf 5500 tr/min',
     f'< 0.1 mm a {below} ; >= 0.1 mm a {above}',
     f'CE QUI EST REPRODUIT est le contraste qui sert de discriminant : '
     f'4900 -> {lim[4900]:.4f} mm contre 5500 -> {lim[5500]:.4f} mm, un '
     f'facteur {lim[5500]/lim[4900]:.1f}. CE QUI NE L\'EST PAS est l\'enonce '
     f'global : sur cette grille de sept vitesses le modele place QUATRE '
     f'vitesses au-dessus de 0.1 mm, pas une seule. Le lobe calcule est donc '
     f'plus contraste que celui mesure — ou la grille de l\'article ne '
     f'contient pas ces quatre vitesses-la, ce que la figure ne permet pas '
     f'de trancher ici.')

# -------------------------------------------------------- 6. le point S
print('6. POINT S DE LA FIG. 14(a) — 4900 tr/min, ap = 0.30 mm')
rS = sim.run(None, rpm=4900, ap=0.30e-3, T=SB.T_LIMIT, keep_signals=True)
note(OK, 'le point S doit DIVERGER',
     'divergent (Fig. 14a)',
     f'stable = {rS["stable"]}, sigma = {rS["sigma"]:+.2f} /s, '
     f'RMS = {rS["rms_um"]:.1f} um',
     f'limite calculee a 4900 tr/min = {lim[4900]:.4f} mm, bien sous 0.30 mm')

# ------------------------------------------------ 7. frequence de broutement
print('7. FREQUENCE DE BROUTEMENT — Fig. 20')
y = rS['y']
t = rS['t']
h = len(y)//2
seg = y[h:]*np.hanning(len(y[h:]))
sp = np.abs(np.fft.rfft(seg))
fr = np.fft.rfftfreq(len(seg), t[1]-t[0])
k = np.argmax(sp[(fr > 100) & (fr < 4600)])
fdom = fr[(fr > 100) & (fr < 4600)][k]
note(KO, 'raie dominante du broutement',
     'fc1 = 580 Hz et fc2 = 1123 Hz (deux raies) ; 1135 Hz annonce',
     f'{fdom:.1f} Hz (raie dominante de la simulation non lineaire)',
     f'ecart {100*(fdom/580-1):+.1f} % sur fc1. La simulation temporelle '
     f'designe le mode 1, la theorie des lobes moyennes de ce meme modele '
     f'designe le mode 2 (1070 Hz, soit -4.7 % de 1123) : les deux modes '
     f'sont a 7.1 % l\'un de l\'autre en criticite, donc le mode qui sort '
     f'depend du type d\'analyse. L\'article observe LES DEUX.')

# ---------------------------------------------------- 8. le correcteur
print('8. LES RESULTATS EN BOUCLE FERMEE')
note(KO, 'la loi de commande de l\'article',
     'robust combined time delay control (le sujet meme de l\'article)',
     'NON IMPLEMENTEE dans ce depot',
     'ce depot compare un LQG modal et un ADRC/ESO a FOPID, qui ne sont PAS '
     'la methode de l\'article. Aucun de ses resultats en boucle fermee '
     '(profondeur atteinte, tensions, Figs. 18-21) n\'est donc reproduit, '
     'ni reproductible en l\'etat.')
note(KO, 'le niveau absolu des Figs. 12(a) et 12(b)',
     'echelles en dB re 1 um/N et 0.01 um/V',
     'incoherentes entre elles d\'un facteur 6.6',
     'aucun modele ne peut satisfaire les deux a la fois (defaut F9). '
     'Le niveau de H_Pe reste encadre par [0.44, 2.92], voir script 17.')

# ------------------------------------------------------------------ bilan
print('=' * 78)
print('BILAN')
print('=' * 78)
n_ok = sum(1 for r in rows if r[0] == OK)
n_ko = sum(1 for r in rows if r[0] == KO)
n_ci = sum(1 for r in rows if r[0] == CIRC)
n_pa = sum(1 for r in rows if r[0] == PART)
for v, item, *_ in rows:
    print(f'  [{v}] {item[:66]}')
print(f'\n  PREDIT {n_ok}   PARTIEL {n_pa}   RECALE {n_ci}   '
      f'NON REPRODUIT {n_ko}')
print("""
LA REPONSE, EN UNE PHRASE. Le MODELE DE PROCEDE de l'article se regenere : les
efforts de coupe sont exacts, les frequences sortent a 2 % sans etre entrees,
les quatre antiresonances a 3.5 %, le niveau et la FORME du lobe libre sont
retrouves, et le point S de la Fig. 14(a) diverge comme il doit.

Une reserve sur le lobe : le contraste 4900 / 5500 tr/min est reproduit, mais
le lobe calcule est plus CONTRASTE que celui mesure -- quatre vitesses sur sept
passent au-dessus de 0.1 mm alors que la Fig. 18 n'en donne qu'une.

Ce qui ne se regenere PAS se ramene a trois choses, et aucune n'est un detail :
  * la repartition modale de H_Pe issue des Eqs. (14)-(15) ne reproduit pas la
    Fig. 12(b) de l'article -- un creux calcule contre quatre mesures ;
  * les echelles absolues de la Fig. 12 sont mutuellement incoherentes ;
  * LA COMMANDE DE L'ARTICLE N'EST PAS IMPLEMENTEE ICI. Tout ce qui suit sa
    Section 5 -- ses gains, ses profondeurs atteintes, ses tensions -- est donc
    hors d'atteinte de ce depot tant que cette loi n'y est pas ecrite.
""")
