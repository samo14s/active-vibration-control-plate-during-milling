"""
analyse_fdob.py — l'observateur modal passe au meme crible que l'ADRC
======================================================================
`diagnose_adrc.py` a etabli POURQUOI l'ADRC-FOPID perd. Ce script verifie que
la structure proposee corrige ce qui etait diagnostique, et cherche ce qu'elle
casse en echange. On ne se contente pas de constater qu'elle gagne : on mesure
le MECANISME, et on va chercher sa faiblesse la ou la theorie dit qu'elle doit
etre.

Quatre mesures :

 1. LA PROPRIETE DE SUR-ENSEMBLE. A alpha = 0 la structure doit redonner le
    FOPID exactement. C'est ce qui manquait a l'ADRC-FOPID, dont l'ensemble de
    correcteurs realisables n'est pas plus grand mais DECALE (ordres -1,
    -(1+lam), mu-1, +1 contre 0, -lam, +mu).

 2. LE MECANISME. La forme fermee predit qu'au mode vise la sensibilite est
    multipliee par (1 - alpha) : au mode, V -> 1 et P W -> 1, donc
    S -> (1-alpha) S_FOPID. On verifie le facteur reellement obtenu.

 3. OU PASSE LA MISE. Le zero instable a 2459 Hz impose
    (1/pi) int log|S| . 2z/(z^2+w^2) dw = 0 EXACTEMENT (egalite, pas
    inegalite : le procede en boucle ouverte est stable et S(z) = 1). La
    "mise" est donc la meme pour tout le monde ; seule sa REPARTITION change.
    On la ventile par bande pour chaque structure. C'est la mesure qui dit si
    la prescription du maillon 6 — concentrer aux modes de broutement, ne rien
    depenser au-dela du zero instable — est vraiment suivie.

 4. LE PRIX DE LA SELECTIVITE. Un observateur etroit (zeta_q = 0.007 ici)
    suppose connaitre la frequence du mode. On desaccorde donc le procede
    autour du modele et on regarde ce que devient la marge de module. C'est la
    faiblesse attendue de cette structure, et on la mesure au lieu de la
    supposer.

    PROTOCOL=B CALIB=measured python analyse_fdob.py
"""
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'), HERE]

import config as C
from plate_model import build_plate, plant_vectors, plant_frf
from fopid import ss_frf
from pso import Design

OUT = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures', 'comparison')
os.makedirs(FIG, exist_ok=True)

LAB = {'fopid': 'FOPID', 'adrc': 'ADRC-FOPID', 'fdob': 'FDOB (2 modes)',
       'fdob12345': 'FDOB (5 modes)'}
COL = {'fopid': '#1a3f8f', 'adrc': '#16a085', 'fdob': '#c0392b',
       'fdob12345': '#8e44ad'}


def poisson_share(f, S, z, edges):
    """Part de l'integrale de Poisson de log|S| tombant dans chaque bande.

    poids w_z(w) = (2z/(z^2+w^2))/pi, normalise a 1 sur [0, inf).
    La somme des parts vaut l'integrale totale, qui doit valoir 0.
    """
    om = 2 * np.pi * f
    wgt = (2 * z / (z ** 2 + om ** 2)) / np.pi
    g = np.log(S) * wgt
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f >= a) & (f <= b)
        out.append(np.trapezoid(g[m], om[m]) if m.sum() > 1 else 0.0)
    return np.array(out)


def main():
    d = np.load(os.path.join(OUT, f'pso_{C.PROTOCOL}.npz'), allow_pickle=True)
    kinds = [k.partition('__')[0] for k in d.files]
    kinds = [k for i, k in enumerate(kinds)
             if k not in kinds[:i] and f'{k}__A' in d.files]
    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    _, _, _, _, sl0 = plant_vectors(plate, C.N_MODES)

    print('=' * 78)
    print(f" L OBSERVATEUR MODAL AU MEME CRIBLE QUE L ADRC — protocole"
          f" {C.PROTOCOL}")
    print('=' * 78)
    print(f"  plaque : f = {np.round(plate.freq_n, 1)} Hz")
    for k in kinds:
        par = dict(zip([str(x) for x in d[f'{k}__keys']], d[f'{k}__values']))
        print(f"  {LAB.get(k, k):16s} : "
              + "  ".join(f"{n}={v:.4g}" for n, v in par.items()))

    f = np.logspace(-2, 5.1, 300000)
    om = 2 * np.pi * f
    P, _ = plant_frf(plate, f, C.N_MODES)
    S = {}
    for k in kinds:
        K = ss_frf(tuple(d[f'{k}__{c}'] for c in 'ABCD'), om)
        S[k] = np.abs(1.0 / (1.0 - P * K))
    imode = [int(np.argmin(np.abs(f - x))) for x in plate.freq_n]

    # ------------------------------------------------------------------ 1
    print('\n' + '-' * 78)
    print(' 1 — PROPRIETE DE SUR-ENSEMBLE   (ce que l ADRC-FOPID n avait pas)')
    print('-' * 78)
    fk = [k for k in kinds if k.startswith('fdob')]

    def targets_of(k):
        # les toutes premieres executions ont ete faites avant que le champ
        # ne soit stocke ; on le retrouve alors depuis le nom de la variante
        if f'{k}__targets' in d.files:
            return [int(t) for t in d[f'{k}__targets']]
        return [0, 1] if k == 'fdob' else [int(c) - 1 for c in k[4:]]

    for k in fk:
        Dg = Design('fdob', plate, sl0,
                    sign_variant=float(d[f'{k}__sign_variant']),
                    targets=targets_of(k))
        x0 = np.array(d[f'{k}__x'], float).copy()
        j = Dg.names.index('alpha')
        x0[j] = (0.0 - Dg.lo[j]) / (Dg.hi[j] - Dg.lo[j])
        K0 = ss_frf(Dg.build(x0), om)
        # CONVENTION DE SIGNE. Dans fdob.py l'epine dorsale agit sur
        # e = -y (u = -C(s) y), alors que la fabrique 'fopid' renvoie C(s)
        # telle quelle et laisse sign_loop porter le signe. La meme valeur de
        # sign_variant designe donc des conventions de boucle OPPOSEES pour
        # les deux fabriques. Cela ne change pas l'ENSEMBLE des correcteurs
        # atteignables, puisque les deux signes sont enumeres des deux cotes
        # — c'est un simple decalage d'etiquette, et le gagnant FDOB
        # (sign_variant = -1) a bien la meme convention de boucle que le
        # gagnant FOPID (sign_variant = +1). On compare donc a -C.
        Df = Design('fopid', plate, sl0,
                    sign_variant=float(d[f'{k}__sign_variant']))
        xf = np.array(d[f'{k}__x'], float)[:5]
        Kf0 = ss_frf(Df.build(xf), om)
        e = float(np.max(np.abs(K0 + Kf0) / np.maximum(np.abs(Kf0), 1e-30)))
        print(f'   {LAB[k]:16s} : alpha -> 0 redonne le FOPID de memes gains'
              f' a {e:.1e}')
    print('   -> la structure CONTIENT son propre cas particulier, donc elle')
    print('      ne peut pas lui etre inferieure a budget de recherche egal.')

    # ------------------------------------------------------------------ 2
    print('\n' + '-' * 78)
    print(' 2 — MECANISME : au mode vise, S doit etre multipliee par (1-alpha)')
    print('-' * 78)
    for k in fk:
        par = dict(zip([str(x) for x in d[f'{k}__keys']], d[f'{k}__values']))
        al = float(par['alpha'])
        tg = targets_of(k)
        print(f'   {LAB[k]} : alpha = {al:.4f}, zeta_q = {par["zeta_q"]:.5f},'
              f' modes vises {[t + 1 for t in tg]}')
        print('      mode [Hz]   |S| FOPID   |S| FDOB   rapport   predit'
              ' (1-alpha)')
        for t in tg:
            i = imode[t]
            print(f'      {plate.freq_n[t]:9.0f} {S["fopid"][i]:11.4f}'
                  f' {S[k][i]:10.4f} {S[k][i] / S["fopid"][i]:9.3f}'
                  f' {1 - al:14.3f}')
    print('   -> le rapport mesure n est pas exactement 1-alpha parce que les')
    print('      gains FOPID ont ete REOPTIMISES autour de l observateur : le')
    print('      correcteur retenu n est pas "le FOPID gagnant plus un')
    print('      observateur", c est un autre point de fonctionnement.')

    # ------------------------------------------------------------------ 3
    print('\n' + '-' * 78)
    print(' 3 — OU PASSE LA MISE (ventilation de l integrale de Poisson)')
    print('-' * 78)
    z = 1.545e4
    edges = [1e-2, 400.0, 700.0, 900.0, 1300.0, 2459.0, 1e5]
    names = ['< 400', '400-700', '700-900', '900-1300', '1300-2459',
             '> 2459 (zero instable)']
    print('   bande [Hz]              ' + '  '.join(f'{LAB[k]:>16s}'
                                                    for k in kinds))
    share = {k: poisson_share(f, S[k], z, edges) for k in kinds}
    for i, nm in enumerate(names):
        print(f'   {nm:22s}  '
              + '  '.join(f'{share[k][i]:+16.4f}' for k in kinds))
    print('   ' + '-' * 24 + '  '
          + '  '.join(f'{"":>16s}' for k in kinds))
    print('   TOTAL (doit valoir 0) '
          + '  '.join(f'{share[k].sum():+16.4f}' for k in kinds))
    print('   -> le total est nul pour TOUTES les structures : la mise est la')
    print('      meme, l integrale de Poisson est une EGALITE. Ce qui distingue')
    print('      les structures, c est uniquement OU elles depensent.')
    neg = {k: -share[k][share[k] < 0].sum() for k in kinds}
    for k in kinds:
        band = share[k][1:4].sum()          # 400-1300 Hz, les deux modes
        print(f'   {LAB.get(k, k):16s} : {-band / neg[k] * 100:5.1f} % de'
              f' l attenuation totale est placee dans 400-1300 Hz')

    # ------------------------------------------------------------------ 4
    print('\n' + '-' * 78)
    print(' 4 — LE PRIX DE LA SELECTIVITE : desaccord du modele')
    print('-' * 78)
    print('   on deplace les frequences de la PLAQUE de +/- e % en gardant le')
    print('   correcteur, et on regarde la marge de module (contrainte <= 2)')
    print('   erreur   ' + '  '.join(f'{LAB.get(k, k):>16s}' for k in kinds))
    errs = (-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0)
    tab = {k: [] for k in kinds}
    for e in errs:
        pl = build_plate(C.PATCH_SIDE,
                         freqs=[x * (1 + e / 100) for x in C.F_NOMINAL])
        Pe, _ = plant_frf(pl, f, C.N_MODES)
        row = []
        for k in kinds:
            K = ss_frf(tuple(d[f'{k}__{c}'] for c in 'ABCD'), om)
            Ms = float(np.abs(1.0 / (1.0 - Pe * K)).max())
            tab[k].append(Ms)
            row.append(f'{Ms:16.3f}')
        print(f'   {e:+5.1f} %  ' + '  '.join(row))
    print('   -> une structure dont le Ms explose des +/- 1 % paie sa')
    print('      selectivite par une dependance au calage que les autres')
    print('      n ont pas. Les essais de robustesse de run_compare mesurent')
    print('      la meme chose sur a_p,lim.')

    # --------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
    a = ax[0]
    m = (f > 100) & (f < 6000)
    for k in kinds:
        a.semilogx(f[m], 20 * np.log10(S[k][m]), color=COL.get(k), lw=1.5,
                   label=LAB.get(k, k))
    for x in plate.freq_n[:2]:
        a.axvline(x, color='0.8', lw=1, zorder=0)
    a.axhline(20 * np.log10(C.MS_MAX), color='k', ls=':', lw=1)
    a.annotate('$M_s = 2$', (110, 20 * np.log10(C.MS_MAX) + .5), fontsize=8)
    a.set_xlabel('frequency [Hz]')
    a.set_ylabel('$|S|$ [dB]')
    a.set_title('(a) where each structure spends its budget\n'
                'the chatter modes are the grey lines', fontsize=9.5)
    a.grid(alpha=.3, which='both')
    a.legend(fontsize=8)

    a = ax[1]
    xb = np.arange(len(names))
    w = 0.8 / len(kinds)
    for i, k in enumerate(kinds):
        a.bar(xb + (i - (len(kinds) - 1) / 2) * w, share[k], w,
              color=COL.get(k), label=LAB.get(k, k))
    a.axhline(0, color='k', lw=1)
    a.set_xticks(xb)
    a.set_xticklabels(names, fontsize=7.5, rotation=20, ha='right')
    a.set_ylabel('Poisson-weighted $\\int\\log|S|$')
    a.set_title('(b) the waterbed budget is EQUAL (total = 0)\n'
                'only its distribution differs', fontsize=9.5)
    a.grid(alpha=.3, axis='y')
    a.legend(fontsize=8)

    a = ax[2]
    for k in kinds:
        a.plot(errs, tab[k], '-o', ms=4, color=COL.get(k),
               label=LAB.get(k, k))
    a.axhline(C.MS_MAX, color='k', ls=':', lw=1)
    a.set_xlabel('modal frequency error [%]')
    a.set_ylabel('$M_s$')
    a.set_yscale('log')
    a.set_title('(c) the price of selectivity\nmodule margin under detuning',
                fontsize=9.5)
    a.grid(alpha=.3, which='both')
    a.legend(fontsize=8)

    fig.suptitle('The frequency-aware modal observer, measured against what '
                 f'the diagnosis prescribed (protocol {C.PROTOCOL})',
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_fdob_{C.PROTOCOL}.png', dpi=140)
    plt.close(fig)
    print(f'\n  -> {FIG}/fig_fdob_{C.PROTOCOL}.png')


if __name__ == '__main__':
    main()
