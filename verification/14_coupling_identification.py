"""Identification de la repartition modale du couplage piezo sur la Fig. 12(b).

CE QUE CE SCRIPT ETABLIT
------------------------
1. H_Pe etait la SEULE grandeur modale jamais confrontee a une mesure. Les
   frequences sont recalees (F_MEASURED), les amortissements sont mesures
   (Tableau 4), et D_obs est valide par la Fig. 12(a) — une FRF au point
   d'impact, dont les residus valent D_i^2. H_Pe, lui, sortait des elements
   finis sans controle, et ne reproduisait aucun des creux mesures.

2. La Fig. 12(b) contient exactement ce qu'il faut pour l'identifier. La
   fonction de transfert tension -> capteur vaut

       G_b(w) = somme_i r_i / (w_i^2 - w^2 + 2 j z_i w_i w),  r_i = D_i * H_i

   Ses POLES sont les frequences mesurees, ses ZEROS sont les creux mesures, et
   poles + zeros determinent les r_i A UNE CONSTANTE PRES : ils sont les racines
   du polynome N(s) = somme_i r_i prod_{j!=i}(s_j - s), donc r_k s'obtient en
   evaluant N au pole k. Aucun optimiseur n'intervient.

3. Il fallait d'abord trancher une ambiguite de LECTURE de la figure. La
   numerisation fournie ne compte que 76 points sur 0-5000 Hz (~48 Hz de pas) :
   elle montre trois creux francs (788, 1493, 3609 Hz) et un simple epaulement
   a +11.9 dB vers 2913 Hz. Compte-t-on l'epaulement comme un 4e zero ? Un
   ajustement LIBRE des cinq residus (16 motifs de signe x 60 graines
   aleatoires) repond : oui.

4. Le NIVEAU n'est pas identifiable — c'est le defaut F9. On ne corrige donc
   que la repartition, a norme de H_Pe constante.

Sortie : tableau + figure coupling_identification.png
"""
import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import openpyxl
from scipy.optimize import least_squares

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_R, 'simulation', 'sim_kit'),
                os.path.join(_R, 'simulation')]
os.chdir(os.path.join(_R, 'simulation', 'sim_kit'))

import simulation_base as SB                                     # noqa: E402
from model_v2 import antiresonances, notch_occupancy             # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=170)


def load(pattern):
    ws = openpyxl.load_workbook(glob.glob(pattern)[0]).worksheets[0]
    d = np.array([[r[0], r[1]] for r in ws.iter_rows(values_only=True)
                  if r[0] is not None], float)
    return d[:, 0], d[:, 1]


fb, Ab = load('transfer_functions*')          # Fig. 12(b), dB re 0.01 um/V

# Poles et amortissements MESURES (pics de la Fig. 12 et Tableau 4). Ce sont des
# donnees d'entree de l'identification, pas des inconnues.
FM = np.array([540.0, 1068.0, 2787.0, 3351.0, 4122.0])
OM = 2*np.pi*FM
ZM = np.array(SB.ZETA_MODES)

# Creux mesures : le minimum de la courbe numerisee dans CHAQUE intervalle
# inter-mode. On ne les code pas en dur — ils se lisent sur les donnees.
NOTCH = np.array([fb[(fb > FM[k]) & (fb < FM[k+1])][
                      np.argmin(Ab[(fb > FM[k]) & (fb < FM[k+1])])]
                  for k in range(4)])
NOTCH_DB = np.array([Ab[(fb > FM[k]) & (fb < FM[k+1])].min() for k in range(4)])


def G(res, f):
    w = 2*np.pi*f
    den = (OM[:, None]**2 - w[None, :]**2
           + 2j*ZM[:, None]*OM[:, None]*w[None, :])
    return (res[:, None]/den).sum(0)


def dB(x):
    return 20*np.log10(np.maximum(np.abs(x), 1e-300))


def rms_vs_measured(res):
    """Ecart courbe a courbe, apres recalage du niveau median (F9)."""
    m = dB(G(res, fb))
    return float(np.sqrt(np.mean((Ab - m - np.median(Ab - m))**2)))


def residues_from_poles_and_zeros(zeros):
    """r_k = prod_l (s_k - z_l) / prod_{j!=k} (s_j - s_k), s = f^2. Exact."""
    s = FM**2
    zz = np.asarray(zeros, float)**2
    r = np.array([np.prod(s[k] - zz)/np.prod(np.delete(s, k) - s[k])
                  for k in range(len(s))])
    return r/r[0]


def occupancy(r):
    """Nb de zeros par intervalle inter-mode, par la regle des signes."""
    return tuple(1 if np.sign(r[k]) == np.sign(r[k+1]) else 0
                 for k in range(len(r)-1))


print(__doc__.split('Sortie')[0])
print('=' * 78)
print('1. CE QUE MONTRE LA NUMERISATION DE LA FIG. 12(b)')
print('=' * 78)
print('   minimum de la courbe dans chaque intervalle inter-mode :')
for k in range(4):
    print(f'      ({FM[k]:6.0f}, {FM[k+1]:6.0f}) Hz : '
          f'{NOTCH[k]:7.1f} Hz a {NOTCH_DB[k]:6.1f} dB')
print('   -> trois creux francs et, dans le 3e intervalle, un simple')
print('      epaulement 38 dB plus haut.')
print(f'   pas d\'echantillonnage median : '
      f'{np.median(np.diff(fb)):.1f} Hz — le fond d\'un creux etroit passe '
      f'entre deux points.')

print()
print('=' * 78)
print('2. AJUSTEMENT LIBRE DES CINQ RESIDUS -> REGION DE CONFIANCE')
print('=' * 78)
rng = np.random.default_rng(7)


def free_fit(signs, seed):
    """r_1 = +1 fige ; le niveau absolu est porte par un decalage en dB."""
    def resid(q):
        return dB(G(signs*np.r_[1.0, np.exp(q[:4])], fb)) + q[4] - Ab
    s = least_squares(resid, np.r_[np.log(np.abs(seed[1:])), 0.0],
                      method='lm', max_nfev=40000)
    return signs*np.r_[1.0, np.exp(s.x[:4])], float(np.sqrt(np.mean(s.fun**2)))


sols = []
with np.errstate(over='ignore', invalid='ignore'):
    for k in range(16):
        sg = np.array([1.0] + [1.0 if (k >> j) & 1 else -1.0 for j in range(4)])
        for _ in range(60):
            try:
                r, e = free_fit(sg, np.r_[1.0, np.exp(rng.uniform(-3, 3, 4))])
            except Exception:
                continue
            if np.isfinite(e):
                sols.append((e, r))
sols.sort(key=lambda t: t[0])
e_best = sols[0][0]
good = np.array([s[1] for s in sols if s[0] < e_best + 0.5])
lo, hi = good.min(0), good.max(0)
print(f'   {len(sols)} ajustements convergents ; {len(good)} a moins de '
      f'0.5 dB du meilleur ({e_best:.2f} dB)')
for i in range(5):
    print(f'      r_{i+1}/r_1 dans [{lo[i]:8.3f}, {hi[i]:8.3f}]')

print()
print('=' * 78)
print('3. LES DEUX LECTURES DE LA FIGURE, JUGEES SUR CETTE REGION')
print('=' * 78)
LECTURES = [
    ('4 creux (l\'epaulement compte)', list(NOTCH)),
    ('3 creux, 4e zero hors bande',
     [NOTCH[0], NOTCH[1], NOTCH[3], 5000.0]),
]
verdict = {}
for lbl, z in LECTURES:
    r = residues_from_poles_and_zeros(z)
    inside = (r >= lo - 1e-9) & (r <= hi + 1e-9)
    verdict[lbl] = (r, inside.all(), rms_vs_measured(r))
    print(f'   {lbl}')
    print(f'      zeros   {np.round(z, 1)}')
    print(f'      r_i/r_1 {np.round(r, 3)}   occupation {occupancy(r)}')
    print(f'      RMS {rms_vs_measured(r):5.2f} dB   '
          f'dans la region de confiance : '
          f'{"OUI" if inside.all() else "NON (%d/5)" % inside.sum()}')

r4 = verdict[LECTURES[0][0]]
r3 = verdict[LECTURES[1][0]]
assert r4[1] and not r3[1], 'la lecture a 4 creux doit etre la seule retenue'
assert r4[2] < r3[2] - 2.0, 'elle doit aussi etre nettement meilleure en RMS'
print('   -> la lecture a QUATRE creux est retenue. C\'est aussi celle de la')
print('      courbe THEORIQUE de l\'article, tracee sur la meme figure.')

print()
print('=' * 78)
print('4. EFFET SUR LA BASE')
print('=' * 78)
print(f'   H_IDENT publie dans simulation_base : {np.array(SB.H_IDENT)}')
assert np.allclose(r4[0], SB.H_IDENT, rtol=2e-4), \
    'H_IDENT doit etre exactement le resultat ci-dessus'

rows = []
for cpl in ('fem', 'ident'):
    p = SB.SimBase(coupling=cpl).plate
    H = np.asarray(p.H_Pe_modal).ravel()
    D = np.asarray(p.D_obs).ravel()
    r = H*D
    z = antiresonances(p)
    rows.append((cpl, np.linalg.norm(H), notch_occupancy(p), z,
                 rms_vs_measured(r)))
    print(f'   coupling = {cpl:5s} : ||H|| = {np.linalg.norm(H):.5f} N/V   '
          f'occupation {notch_occupancy(p)}')
    print(f'{"":22s}creux {np.round(z, 0)} Hz   '
          f'ecart a la mesure {rms_vs_measured(r):5.2f} dB RMS')
print(f'   mesure               : occupation (1, 1, 1, 1)   '
      f'creux {np.round(NOTCH, 0)} Hz')

fem, ident = rows[0], rows[1]
# (tolerance relative : eigsh part d'un vecteur aleatoire, donc deux
#  constructions successives de la meme plaque different de ~1e-9)
assert abs(fem[1]/ident[1] - 1.0) < 1e-6, \
    'la norme de H_Pe doit etre inchangee : seule la repartition est corrigee'
assert ident[2] == (1, 1, 1, 1) and fem[2] != (1, 1, 1, 1)
assert ident[4] < fem[4] - 5.0, \
    'la repartition identifiee doit ameliorer nettement l\'accord'
zi = ident[3][(ident[3] > FM[0]) & (ident[3] < FM[-1])]
zt = NOTCH
err = np.abs(zi - zt)/zt*100
print(f'   erreur sur les quatre creux : {np.round(err, 2)} %  '
      f'(max {err.max():.2f} %)')
assert err.max() < 1.0

print()
print('=' * 78)
print('5. CE QUI N\'EST PAS IDENTIFIE : LE NIVEAU')
print('=' * 78)
p = SB.SimBase().plate
r = np.asarray(p.H_Pe_modal).ravel()*np.asarray(p.D_obs).ravel()
g0_model = abs(G(r, np.array([1.0]))[0])*1e6
g0_meas = 0.01*10**(np.median(Ab[fb < 200])/20)
print(f'   gain statique du modele : {g0_model:.4f} um/V')
print(f'   gain statique mesure    : {g0_meas:.4f} um/V  '
      f'(facteur {g0_meas/g0_model:.2f})')
print('   Le niveau absolu des figures de l\'article n\'est pas exploitable')
print('   (F9 : sur toute la bande, la Fig. 12(a) demande +12.4 dB de recalage')
print('   median et la Fig. 12(b) +6.8 dB pour se superposer au meme modele —')
print('   aucun modele ne peut satisfaire les deux). On garde donc la norme des')
print('   elements finis ; run_demo --full balaie le niveau (gain_H = 0.5 et 2), et')
print(f'   le facteur {g0_meas/g0_model:.2f} ci-dessus tombe dans ce balayage.')

# ---------------------------------------------------------------- figure
f = np.linspace(20, 5000, 4000)
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(fb, Ab, 'o-', color='#1f4e9c', ms=3, lw=1.2,
        label='mesure Fig. 12(b) (numerisee)')
for cpl, style, col in [('fem', '--', '#c02020'), ('ident', '-', '#1a8a3a')]:
    pl = SB.SimBase(coupling=cpl).plate
    rr = np.asarray(pl.H_Pe_modal).ravel()*np.asarray(pl.D_obs).ravel()
    y = dB(G(rr, f))
    y = y + np.median(Ab - dB(G(rr, fb)))
    ax.plot(f, y, style, color=col, lw=1.4,
            label=f'modele, coupling={cpl} ({rms_vs_measured(rr):.1f} dB RMS)')
for z in zt:
    ax.axvline(z, color='0.7', lw=0.8, ls=':', zorder=0)
ax.set_xlabel('frequence (Hz)')
ax.set_ylabel('amplitude (dB re 0.01 um/V)')
ax.set_title('Fig. 12(b) : repartition modale du couplage, EF vs identifiee')
ax.set_xlim(0, 5000)
ax.set_ylim(-60, 70)
ax.grid(alpha=0.3)
ax.legend(loc='upper right', fontsize=9)
fig.tight_layout()
out = os.path.join(_OUT, 'coupling_identification.png')
fig.savefig(out, dpi=130)
print(f'\nfigure ecrite : {os.path.relpath(out, _R)}')
print('\nTOUTES LES VERIFICATIONS PASSENT.')
