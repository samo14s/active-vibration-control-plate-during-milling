"""11_frf_and_antiresonances.py — audit de la Fig. 12 du papier
(reponses impulsionnelle et balayee au coin superieur droit).

Du, Liu, Dai, Long, "Robust combined time delay control for milling chatter
suppression of flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257.

Ce que dit le papier (Sec. 4.1, legende Fig. 12) : le marteau PCB 086C03 ET le
capteur a courants de Foucault ZA11 sont TOUS DEUX au coin superieur droit ;
Fig. 12(a) est donc une receptance AU POINT DE FRAPPE (Ref = 1 um/N) et
Fig. 12(b) la fonction de transfert tension -> deplacement (Ref = 0.01 um/V).

Contenu :
  1. plaque du Tableau 1 + pastille du Tableau 2, pastille au coin inferieur
     DROIT puis GAUCHE, calee une fois sur les frequences MESUREES et une fois
     sur les frequences THEORIQUES du Tableau 4, amortissements mesures ;
  2. Fig. 12(a) : H(f) = sum_i D_obs(i)^2 / (w_i^2 - w^2 + 2 j z_i w_i w).
     Les antiresonances sont les RACINES EXACTES du numerateur de
     sum_i r_i/(w_i^2 - s) en s = w^2 (polynome de degre n-1), et non des
     minima locaux de |H| : avec amortissement, un intervalle SANS zero
     presente quand meme un creux mou et serait compte a tort ;
  3. Fig. 12(b) : G(f) = sum_i D_obs(i) H_Pe(i) / den. Zeros exacts, motif de
     signes sign(D_obs(i) H_Pe(i)), et OCCUPATION (nombre de zeros par
     intervalle entre poles consecutifs) — signature de signes non ajustable ;
  4. souplesse statique au coin (f -> 0), confrontee au niveau absolu de la
     Fig. 12(a) : le niveau du papier est inatteignable pour toute plaque
     coherente avec le Tableau 1 (cf. VERIFICATION.md F9), donc seules les
     FORMES (frequences, antiresonances) sont verifiables ici.

Valeurs de reference "digitisees" : elles proviennent d'une NUMERISATION des
courbes de la Fig. 12 consignee dans VERIFICATION.md — ce ne sont pas des
nombres imprimes par le papier, leur incertitude est celle du releve
graphique (quelques % en frequence, ~1 dB en niveau).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'),
                os.path.join(HERE, '..', 'control')]

import numpy as np
from numpy.polynomial import polynomial as PP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plate_model import build_plate, PATCH, F_MEASURED, F_THEORETICAL

FIGDIR = os.path.join(HERE, '..', 'figures', 'verification')
os.makedirs(FIGDIR, exist_ok=True)
FIGPATH = os.path.abspath(os.path.join(FIGDIR, '11_frf_and_antiresonances.png'))

# ---------------------------------------------------------------------------
# Valeurs de reference (papier + numerisation de la Fig. 12)
# ---------------------------------------------------------------------------
ZETA = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]     # Tableau 4, mesures
ANTIRES_A = [734.0, 2161.0, 3001.0, 3805.0]         # Fig. 12(a), DIGITISE
NOTCH_B = [788.0, 1493.0, 3609.0]                   # Fig. 12(b), DIGITISE
OCC_B = (1, 1, 0, 1)                                # occupation deduite
LEVEL_A_DB = 32.83                                  # dB re 1 um/N, DIGITISE
LEVEL_A_UM = 10.0 ** (LEVEL_A_DB / 20.0)            # ~43.8 um/N
FE_STATIC = 6.896                                   # um/N, VERIFICATION.md F9


# ---------------------------------------------------------------------------
# Outils : FRF modale, zeros exacts, occupation
# ---------------------------------------------------------------------------
def frf(f_hz, r, f_pole, zeta):
    """sum_i r_i / (w_i^2 - w^2 + 2 j z_i w_i w)  (unites de r par kg.s^-2)."""
    w = 2 * np.pi * np.asarray(f_hz, float)
    wn = 2 * np.pi * np.asarray(f_pole, float)
    z = np.asarray(zeta, float)
    den = (wn[:, None]**2 - w[None, :]**2
           + 2j * z[:, None] * wn[:, None] * w[None, :])
    return np.sum(np.asarray(r, float)[:, None] / den, axis=0)


def exact_zeros(f_pole, r):
    """Racines EXACTES du numerateur de sum_i r_i/(w_i^2 - s), s = w^2.

    N(s) = sum_i r_i prod_{j!=i} (w_j^2 - s), degre n-1. On travaille en
    s~ = (f/1000)^2 pour le conditionnement. Retourne (racines brutes,
    zeros physiques en Hz)."""
    a = (np.asarray(f_pole, float) / 1000.0) ** 2
    n = a.size
    num = np.zeros(1)
    for i in range(n):
        num = PP.polyadd(num, r[i] * PP.polyfromroots(np.delete(a, i)))
    rts = PP.polyroots(num)
    scale = max(1.0, float(np.max(np.abs(rts.real))))
    real = rts[np.abs(rts.imag) < 1e-8 * scale].real
    phys = np.sort(1000.0 * np.sqrt(real[real > 0.0]))
    return rts, phys


def occupancy(f_pole, zeros_hz):
    """Nombre de zeros dans chaque intervalle entre poles consecutifs."""
    f = np.asarray(f_pole, float)
    return tuple(int(np.sum((zeros_hz > f[k]) & (zeros_hz < f[k + 1])))
                 for k in range(f.size - 1))


def parity_rule(r):
    """Parite attendue par le theoreme des signes : un intervalle contient un
    nombre IMPAIR de zeros ssi sign(r_k) == sign(r_{k+1})."""
    s = np.sign(np.asarray(r, float))
    return tuple('odd' if s[k] == s[k + 1] else 'even'
                 for k in range(s.size - 1))


def static_compliance(f_pole, r):
    """H(0) = sum_i r_i / w_i^2."""
    wn = 2 * np.pi * np.asarray(f_pole, float)
    return float(np.sum(np.asarray(r, float) / wn**2))


# ===========================================================================
SEP = '=' * 78
print(SEP)
print(' SCRIPT 11 — FIG. 12 : RECEPTANCE AU POINT DE FRAPPE ET ANTIRESONANCES')
print('   Du, Liu, Dai, Long, Int. J. Mech. Sci. 274 (2024) 109257')
print(SEP)
print(' Toutes les valeurs "Fig. 12" ci-dessous sont DIGITISEES (relevees sur')
print(' les courbes du papier, cf. VERIFICATION.md) et non imprimees par les')
print(' auteurs : incertitude de lecture graphique, quelques % en frequence.')

# ---------------------------------------------------------------------------
# 1. Construction des modeles
# ---------------------------------------------------------------------------
print('\n[1] MODELES — plaque Tableau 1 + pastille Tableau 2 (Chebyshev-Ritz)')
plates = {}
for side in ('right', 'left'):
    raw = build_plate(patch=side, n_modes=5, calibrate=False)
    for tag, fr in (('meas', F_MEASURED), ('theo', F_THEORETICAL)):
        plates[(side, tag)] = build_plate(patch=side, n_modes=5,
                                          calibrate=True, freqs=fr)
    plates[(side, 'raw')] = raw

raw = plates[('right', 'raw')]
print('    frequences Ritz brutes (avant calage)  : '
      + '  '.join('%7.1f' % v for v in raw.freq_n))
print('    Tableau 4 "theoretical"                : '
      + '  '.join('%7.1f' % v for v in F_THEORETICAL))
print('    ecart relatif                          : '
      + '  '.join('%6.2f%%' % v for v in
                  100 * (raw.freq_n / np.array(F_THEORETICAL) - 1)))
print('    Tableau 4 "measured"                   : '
      + '  '.join('%7.1f' % v for v in F_MEASURED))
print('    ecart relatif                          : '
      + '  '.join('%6.2f%%' % v for v in
                  100 * (raw.freq_n / np.array(F_MEASURED) - 1)))
print('    amortissements imposes (Tableau 4)     : '
      + '  '.join('%6.2f%%' % (100 * z) for z in ZETA))
print('    pastille "%s" = [%.0f,%.0f]x[%.0f,%.0f] mm ; "%s" = [%.0f,%.0f]x[%.0f,%.0f] mm'
      % (('right',) + tuple(1e3 * PATCH['right'][k] for k in ('x1', 'x2', 'z1', 'z2'))
         + ('left',) + tuple(1e3 * PATCH['left'][k] for k in ('x1', 'x2', 'z1', 'z2'))))
print('    capteur / marteau : coin superieur droit (x=100, z=80 mm)')

res = {}
for side in ('right', 'left'):
    p = plates[(side, 'meas')]
    D = p.D_row(p.lp, p.hp)
    H = np.asarray(p.H_Pe_modal, float)
    res[side] = dict(D=D, H=H, ra=D * D, rb=D * H)
    print('\n    -- pastille %-5s --' % side)
    print('       D_obs (coin sup. droit) : '
          + '  '.join('%8.4f' % v for v in D))
    print('       H_Pe [N/V]              : '
          + '  '.join('%8.5f' % v for v in H))
    print('       r_a = D_obs^2           : '
          + '  '.join('%8.3f' % v for v in D * D))
    print('       r_b = D_obs*H_Pe        : '
          + '  '.join('%8.5f' % v for v in D * H))
    print('       sign(D_obs*H_Pe)        : '
          + '  '.join('%8d' % v for v in np.sign(D * H).astype(int)))

# ---------------------------------------------------------------------------
# 2. Fig. 12(a) — antiresonances du point de frappe
# ---------------------------------------------------------------------------
print('\n' + SEP)
print('[2] FIG. 12(a) — RECEPTANCE AU POINT DE FRAPPE, ZEROS EXACTS')
print(SEP)
print('    methode : racines du polynome numerateur en s = w^2 (PAS de')
print('    detection de minima locaux de |H|, qui donne des faux positifs).')

zeros_a = {}
for side in ('right', 'left'):
    for tag in ('meas', 'theo'):
        p = plates[(side, tag)]
        rts, z = exact_zeros(p.freq_n, res[side]['ra'])
        zeros_a[(side, tag)] = z
        occ = occupancy(p.freq_n, z)
        print('\n    pastille %-5s, calage %s  (poles %s)'
              % (side, tag, ' '.join('%.0f' % v for v in p.freq_n)))
        print('      zeros modele [Hz] : '
              + '  '.join('%8.1f' % v for v in z))
        print('      occupation        : %s   (attendu %s : residus tous > 0'
              ' => entrelacement strict)' % (str(occ), str((1, 1, 1, 1))))
        if len(z) == len(ANTIRES_A):
            e = 100 * (z / np.array(ANTIRES_A) - 1)
            print('      Fig.12a DIGITISE  : '
                  + '  '.join('%8.1f' % v for v in ANTIRES_A))
            print('      ecart relatif     : '
                  + '  '.join('%7.2f%%' % v for v in e))
            print('      ecart absolu [Hz] : '
                  + '  '.join('%8.1f' % v for v in z - np.array(ANTIRES_A)))
            print('      RMS relatif       : %6.2f %%   |  max %6.2f %%'
                  % (np.sqrt(np.mean(e**2)), np.max(np.abs(e))))

# ---------------------------------------------------------------------------
# 3. Fig. 12(b) — zeros tension -> deplacement, occupation
# ---------------------------------------------------------------------------
print('\n' + SEP)
print('[3] FIG. 12(b) — TENSION -> DEPLACEMENT, ZEROS ET OCCUPATION')
print(SEP)
pm = plates[('right', 'meas')].freq_n
print('    intervalles entre poles mesures : '
      + '  '.join('[%.0f,%.0f]' % (pm[k], pm[k + 1]) for k in range(4)))
print('    creux profonds Fig.12b DIGITISES: '
      + '  '.join('%.0f' % v for v in NOTCH_B)
      + '   => occupation mesuree %s' % str(OCC_B))

zeros_b = {}
occ_b = {}
for side in ('right', 'left'):
    p = plates[(side, 'meas')]
    rts, z = exact_zeros(p.freq_n, res[side]['rb'])
    zeros_b[side] = z
    occ = occupancy(p.freq_n, z)
    occ_b[side] = occ
    n_cplx = int(np.sum(np.abs(rts.imag) > 1e-8 * max(1.0, np.max(np.abs(rts.real)))))
    n_neg = int(np.sum((np.abs(rts.imag) <= 1e-8 * max(1.0, np.max(np.abs(rts.real))))
                       & (rts.real <= 0)))
    print('\n    pastille %-5s' % side)
    print('      signes r_b        : %s' % np.sign(res[side]['rb']).astype(int))
    print('      parite attendue   : %s' % str(parity_rule(res[side]['rb'])))
    print('      racines brutes s~ : %s'
          % np.array2string(rts, precision=3, suppress_small=True))
    print('      -> %d zeros physiques, %d racines complexes, %d racines s<=0'
          % (z.size, n_cplx, n_neg))
    print('      zeros [Hz]        : '
          + ('  '.join('%8.1f' % v for v in z) if z.size else '  (aucun)'))
    print('      occupation modele : %s     mesuree %s     -> %s'
          % (str(occ), str(OCC_B),
             'MATCH' if occ == OCC_B else 'MISMATCH (%d/4 intervalles corrects)'
             % sum(int(a == b) for a, b in zip(occ, OCC_B))))

print('\n    verdict position : %s'
      % ('right' if occ_b['right'] == OCC_B else
         ('left' if occ_b['left'] == OCC_B else
          'AUCUNE des deux ne reproduit exactement (1,1,0,1)')))
n_ok_r = sum(int(a == b) for a, b in zip(occ_b['right'], OCC_B))
n_ok_l = sum(int(a == b) for a, b in zip(occ_b['left'], OCC_B))
print('      intervalles corrects : right %d/4   left %d/4' % (n_ok_r, n_ok_l))
print('      -> la pastille %s est la plus proche de la signature mesuree.'
      % ('right' if n_ok_r >= n_ok_l else 'left'))

# comparaison frequentielle des creux communs (right)
print('\n    frequences des creux, pastille right vs Fig.12b DIGITISE :')
zr = zeros_b['right']
for fmeas in NOTCH_B:
    if zr.size:
        j = int(np.argmin(np.abs(zr - fmeas)))
        print('      mesure %7.0f Hz  ->  modele %8.1f Hz   ecart %+7.2f %%'
              % (fmeas, zr[j], 100 * (zr[j] / fmeas - 1)))

# ---------------------------------------------------------------------------
# 4. Niveau absolu : souplesse statique au coin
# ---------------------------------------------------------------------------
print('\n' + SEP)
print('[4] NIVEAU ABSOLU — SOUPLESSE STATIQUE AU COIN SUPERIEUR DROIT')
print(SEP)
print('    grandeur                                       souplesse    facteur')
print('                                                    [um/N]    vs Fig.12a')
rows = []
for side in ('right', 'left'):
    for tag in ('meas', 'theo'):
        p = plates[(side, tag)]
        c = static_compliance(p.freq_n, res[side]['ra']) * 1e6
        rows.append(('troncature 5 modes, patch %-5s, calage %s' % (side, tag), c))
for side in ('right', 'left'):
    p = plates[(side, 'raw')]
    Y = p.Y_row(p.lp, p.hp)
    c = float(Y @ np.linalg.solve(p.K, Y)) * 1e6
    rows.append(('tous DDL Ritz (K^-1), patch %-5s        ' % side, c))
rows.append(('EF Kirchhoff-Q4 complet (VERIFICATION.md F9)', FE_STATIC))
for name, c in rows:
    print('    %-46s %9.3f    x%5.2f' % (name, c, LEVEL_A_UM / c))
print('    %-46s %9.3f    x%5.2f' % ('Fig. 12(a) DIGITISE (%.2f dB re 1 um/N)'
                                     % LEVEL_A_DB, LEVEL_A_UM, 1.0))
c_ref = static_compliance(plates[('right', 'meas')].freq_n,
                          res['right']['ra']) * 1e6
print('\n    niveau BF du modele (patch right, calage mesure) : %.2f dB re 1 um/N'
      % (20 * np.log10(c_ref)))
print('    niveau BF DIGITISE de la Fig. 12(a)              : %.2f dB re 1 um/N'
      % LEVEL_A_DB)
print('    ecart : %+.1f dB  (facteur %.2f en souplesse)'
      % (LEVEL_A_DB - 20 * np.log10(c_ref), LEVEL_A_UM / c_ref))
print('    => cf. VERIFICATION.md F9 : une plaque %.1fx plus souple donnerait'
      % (LEVEL_A_UM / c_ref))
print('       f1 = 540/sqrt(%.2f) = %.0f Hz, pas 540 Hz. L echelle en dB de la'
      % (LEVEL_A_UM / c_ref, 540 / np.sqrt(LEVEL_A_UM / c_ref)))
print('       Fig. 12 est inexploitable : seules les FORMES sont verifiables.')

# ---------------------------------------------------------------------------
# 5. Figure
# ---------------------------------------------------------------------------
ff = np.linspace(60.0, 4600.0, 40001)
fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.0))

# (a) receptance au coin, deux calages
axa = ax[0, 0]
sty = dict(meas=('-', 'tab:blue', 'measured-freq calibration'),
           theo=('--', 'tab:red', 'theoretical-freq calibration'))
for tag in ('meas', 'theo'):
    p = plates[('right', tag)]
    Ha = frf(ff, res['right']['ra'], p.freq_n, ZETA)
    ls, col, lab = sty[tag]
    axa.plot(ff, 20 * np.log10(np.abs(Ha) * 1e6), ls, color=col, lw=1.2,
             label=lab)
for k, fm in enumerate(F_MEASURED):
    axa.axvline(fm, color='0.75', lw=0.7, zorder=0,
                label='measured natural freq. (Table 4)' if k == 0 else None)
for k, fa in enumerate(ANTIRES_A):
    axa.axvline(fa, color='tab:green', ls=':', lw=1.1, zorder=0,
                label='digitized antiresonance, Fig. 12(a)' if k == 0 else None)
zm = zeros_a[('right', 'meas')]
axa.plot(zm, np.interp(zm, ff, 20 * np.log10(np.abs(
    frf(ff, res['right']['ra'], plates[('right', 'meas')].freq_n, ZETA)) * 1e6)),
    'v', color='k', ms=6, label='exact model zeros')
axa.axhline(LEVEL_A_DB, color='tab:orange', lw=1.0, ls='-.',
            label='digitized level Fig. 12(a): %.1f dB' % LEVEL_A_DB)
axa.set_title('(a) Drive-point receptance, upper right corner\n'
              'patch right; markers = exact zeros, not local minima')
axa.set_xlabel('frequency [Hz]')
axa.set_ylabel('|H| [dB re 1 ' + r'$\mu$' + 'm/N]')
axa.grid(alpha=0.3)
axa.set_xlim(0, 4600)
y0, y1 = axa.get_ylim()                 # place pour la legende sous les creux
axa.set_ylim(y0 - 0.45 * (y1 - y0), y1)
axa.legend(fontsize=7, loc='lower center', ncol=2, framealpha=0.95)

# (b) et (c) tension -> deplacement
for col_i, side in enumerate(('right', 'left')):
    a = ax[0, 1] if side == 'right' else ax[1, 0]
    p = plates[(side, 'meas')]
    G = frf(ff, res[side]['rb'], p.freq_n, ZETA)
    gdb = 20 * np.log10(np.abs(G) * 1e6 / 0.01)
    a.plot(ff, gdb, '-', color='tab:blue', lw=1.2, label='model |G|')
    for k, fm in enumerate(F_MEASURED):
        a.axvline(fm, color='0.75', lw=0.7, zorder=0,
                  label='natural freq. (Table 4)' if k == 0 else None)
    for k, fn in enumerate(NOTCH_B):
        a.axvline(fn, color='tab:green', ls=':', lw=1.1, zorder=0,
                  label='digitized deep notch, Fig. 12(b)' if k == 0 else None)
    zb = zeros_b[side]
    if zb.size:
        a.plot(zb, np.interp(zb, ff, gdb), 'v', color='k', ms=6,
               label='exact model zeros')
    a.set_title('(%s) Voltage -> displacement, patch %s\n'
                'model occupancy %s   vs measured %s'
                % ('b' if side == 'right' else 'c', side,
                   str(occ_b[side]), str(OCC_B)))
    a.set_xlabel('frequency [Hz]')
    a.set_ylabel('|G| [dB re 0.01 ' + r'$\mu$' + 'm/V]')
    a.grid(alpha=0.3)
    a.set_xlim(0, 4600)
    y0, y1 = a.get_ylim()
    a.set_ylim(y0 - 0.42 * (y1 - y0), y1)
    a.legend(fontsize=7, loc='lower center', ncol=2, framealpha=0.95)

# (d) tableau d'occupation
axd = ax[1, 1]
axd.axis('off')
gaps = ['%.0f-%.0f' % (pm[k], pm[k + 1]) for k in range(4)]
cell = []
for k in range(4):
    zr_in = [z for z in zeros_b['right'] if pm[k] < z < pm[k + 1]]
    zl_in = [z for z in zeros_b['left'] if pm[k] < z < pm[k + 1]]
    nm_in = [n for n in NOTCH_B if pm[k] < n < pm[k + 1]]
    cell.append([gaps[k],
                 '%d  (%s)' % (OCC_B[k],
                               ', '.join('%.0f' % v for v in nm_in) or '-'),
                 '%d  (%s)' % (occ_b['right'][k],
                               ', '.join('%.0f' % v for v in zr_in) or '-'),
                 '%d  (%s)' % (occ_b['left'][k],
                               ', '.join('%.0f' % v for v in zl_in) or '-')])
cell.append(['MATCH', '-', '%d/4' % n_ok_r, '%d/4' % n_ok_l])
tb = axd.table(cellText=cell,
               colLabels=['pole gap [Hz]', 'measured (digitized)',
                          'model, patch right', 'model, patch left'],
               loc='center', cellLoc='center')
tb.auto_set_font_size(False)
tb.set_fontsize(8.5)
tb.scale(1.0, 1.9)
for j in range(4):
    tb[(0, j)].set_facecolor('#dddddd')
for k in range(4):
    ok_r = occ_b['right'][k] == OCC_B[k]
    ok_l = occ_b['left'][k] == OCC_B[k]
    tb[(k + 1, 2)].set_facecolor('#cdeccd' if ok_r else '#f2c9c9')
    tb[(k + 1, 3)].set_facecolor('#cdeccd' if ok_l else '#f2c9c9')
axd.set_title('(d) Zero occupancy per pole gap, Fig. 12(b)\n'
              'sign(D_obs*H_Pe): right %s   left %s'
              % (np.sign(res['right']['rb']).astype(int),
                 np.sign(res['left']['rb']).astype(int)), fontsize=10)
axd.text(0.5, 0.06,
         'Static compliance at corner: %.2f ' % c_ref + r'$\mu$' + 'm/N '
         '(5 modes, right, measured cal.)\n'
         'Digitized Fig. 12(a) level %.2f dB = %.1f ' % (LEVEL_A_DB, LEVEL_A_UM)
         + r'$\mu$' + 'm/N  ->  x%.1f softer: levels NOT verifiable'
         % (LEVEL_A_UM / c_ref),
         ha='center', va='center', transform=axd.transAxes, fontsize=8.5)

fig.tight_layout()
fig.savefig(FIGPATH, dpi=140)
print('\n    figure : %s' % FIGPATH)
print(SEP)
