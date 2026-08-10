"""FIG. 7 DE L'ARTICLE, RECALCULEE — les elements de D_r^T D_r le long de
l'arete superieure.

Legende de l'article : "The actual, mean, maximum, and minimum values of
elements in D_pr(x_p, z_p) varied with milling positions on the upper edge of
the thin-walled cantilever plate, (a) D_pr^T D_pr(1,1), (b) (1,2), (c) (2,1),
(d) (2,2)."

POURQUOI CETTE FIGURE COMPTE. C'est elle qui fournit les nombres de l'Eq. (24) :
la MOYENNE (max+min)/2 de chaque element sert de valeur nominale, la
DEMI-AMPLITUDE (max-min)/2 sert de perturbation. Avec l'Eq. (23) sur alpha4,
elles forment l'Eq. (25), c.-a-d. le point de fonctionnement sur lequel tout le
correcteur robuste est synthetise. Se tromper ici deplace la synthese entiere.

L'article n'en publie aucune valeur numerique -- seulement des courbes. Ce
script les recalcule et verifie l'affirmation QUALITATIVE qui les accompagne :
"Except for the first element D_pr^T D_pr(1,1), other elements all vary with
position largely."

UNE PRECAUTION QUI N'EST PAS COSMETIQUE. Les elements DIAGONAUX sont des carres,
donc invariants ; les elements CROISES (1,2) et (2,1) valent D_1 D_2 et changent
de signe si l'un des deux modes change de signe. Or ARPACK part d'un vecteur
aleatoire : dans simulation_base, le signe de chaque mode change d'une
construction a l'autre. Cette figure serait donc a moitie non reproductible.
On la trace depuis milling_plant, qui FIXE la convention de signe (composante
dominante positive) -- c'est exactement le cas d'usage pour lequel elle a ete
ajoutee.

Sortie : verification/figure7.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, 'simulation')]
_HERE = os.path.dirname(os.path.abspath(__file__))

from milling_plant import MillingPlant, PLATE_L                  # noqa: E402

print(__doc__.split('Sortie')[0])
print('=' * 76)

plant = MillingPlant()
p = plant.plate
x = np.asarray(p.xp_array)*1e3                    # mm
D = np.asarray(p.Dp_array)[:2, :]                 # deux premiers modes
E = np.einsum('ik,jk->ijk', D, D)                 # (2, 2, n_pos)

LAB = [('(a)', 0, 0), ('(b)', 0, 1), ('(c)', 1, 0), ('(d)', 1, 1)]
print('%6s %12s %12s %12s %12s %10s'
      % ('elem', 'min', 'max', 'nominal', 'perturb.', 'variation'))
stats = {}
for tag, i, j in LAB:
    v = E[i, j]
    nom = 0.5*(v.max() + v.min())                 # Eq. (24), valeur nominale
    per = 0.5*(v.max() - v.min())                 # Eq. (24), perturbation
    rel = per/max(abs(nom), 1e-30)
    stats[(i, j)] = (v, nom, per, rel)
    print('%6s %12.3f %12.3f %12.3f %12.3f %9.1f %%'
          % (f'({i+1},{j+1})', v.min(), v.max(), nom, per, 100*rel))

r11 = stats[(0, 0)][3]
others = [stats[k][3] for k in stats if k != (0, 0)]
print()
print("VERIFICATION DE L'AFFIRMATION DE L'ARTICLE")
print('  "Except for the first element, other elements all vary largely"')
print(f'  variation relative de (1,1)      : {100*r11:.1f} %')
print(f'  variation relative des autres    : '
      f'{", ".join(f"{100*o:.0f} %" for o in others)}')
ok = all(o > 5*r11 for o in others)
print(f'  -> {"CONFORME" if ok else "NON CONFORME"} : les trois autres varient '
      f'{min(others)/r11:.0f} a {max(others)/r11:.0f} fois plus que (1,1)')
assert ok, "l'affirmation qualitative de l'article n'est pas reproduite"

print()
print("CE QUE CELA DONNE POUR L'Eq. (25)")
print('  a4_0 D_r0^T D_r0 = 1.6 a4_moyen x [nominal], et la perturbation')
print('  associee vaut L_alpha x [perturbation] avec L_alpha = 1.3 a4_moyen.')
print('  (1,1) est donc le SEUL element dont le nominal a un sens etroit ; les')
print('  trois autres sont des moyennes de grandeurs qui changent de signe le')
print('  long de la passe, ce que la synthese traite comme incertitude.')

fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
for k, (tag, i, j) in enumerate(LAB):
    a = ax[k//2][k % 2]
    v, nom, per, rel = stats[(i, j)]
    a.plot(x, v, '-', lw=1.4, color='#1f4e79', label='valeur reelle')
    a.axhline(nom, color='#c00000', lw=1.5,
              label=f'moyenne (max+min)/2 = {nom:.2f}')
    a.axhline(v.max(), color='.45', ls='--', lw=1.1,
              label=f'max = {v.max():.2f}')
    a.axhline(v.min(), color='.45', ls=':', lw=1.1,
              label=f'min = {v.min():.2f}')
    a.fill_between(x, nom - per, nom + per, color='#c00000', alpha=.08)
    a.set_title(f'{tag}  $D_{{pr}}^T D_{{pr}}({i+1},{j+1})$'
                f'   — variation {100*rel:.0f} %')
    a.grid(alpha=.3)
    a.legend(fontsize=7.5, loc='best')
    if k >= 2:
        a.set_xlabel('position de fraisage le long de l\'arete [mm]')
ax[0][0].set_ylabel('element de $D_{pr}^T D_{pr}$')
ax[1][0].set_ylabel('element de $D_{pr}^T D_{pr}$')
fig.suptitle('Fig. 7 de Du et al. (IJMS 2024), recalculee — '
             'ce sont les nombres des Eqs. (24)-(25)', fontsize=11)
fig.tight_layout()
out = os.path.join(_HERE, 'figure7.png')
fig.savefig(out, dpi=130)
print(f'\nfigure : {out}')
print('\nTOUTES LES VERIFICATIONS PASSENT.')
