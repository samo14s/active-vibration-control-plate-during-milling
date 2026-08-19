"""
figures.py — Figures de la comparaison FOPID / ADRC-FOPID
==========================================================
Les textes des figures sont en ANGLAIS (matplotlib ne sait pas mettre en forme
l'arabe) ; les commentaires du code restent en francais comme le reste du
depot.

    PROTOCOL=A|B python figures.py           # figures d'un protocole
    python figures.py --cross                # figure de synthese A contre B
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

OUT = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures', 'comparison')
os.makedirs(FIG, exist_ok=True)

COL = {'boucle ouverte': '#c8963e', 'fopid': '#1a3f8f', 'adrc': '#16a085'}
LAB = {'boucle ouverte': 'no control', 'fopid': 'FOPID',
       'adrc': 'ADRC-FOPID'}
KEYS = ('boucle ouverte', 'fopid', 'adrc')
# les etiquettes des cas de robustesse sont stockees en francais dans le .npz ;
# les figures, elles, sont en anglais
ROB_EN = {'modele de synthese': 'design model',
          'modele complet (verite)': 'full 5-mode model (truth)',
          'derive +17/+9 %': 'modal drift +17/+9 %',
          'amortissement x0.8': 'damping x0.8',
          'raideur/masse +10 %': 'stiffness/mass +10 %',
          'raideur/masse -10 %': 'stiffness/mass -10 %',
          'calage theorique': 'theoretical calibration'}
PROT_TITLE = {'A': 'Protocol A - design on the 2-mode reduced model,'
                   ' evaluation on 5 modes',
              'B': 'Protocol B - design and evaluation on the full 5-mode'
                   ' model'}


def load(name):
    d = np.load(os.path.join(OUT, name), allow_pickle=True)
    out = {}
    for k in d.files:
        if '__' in k:
            g, f = k.split('__', 1)
            out.setdefault(g, {})[f] = d[k]
        else:
            out[k] = d[k]
    return out


def _suptitle(fig, txt, prot):
    fig.suptitle(f'{txt}\n{PROT_TITLE[prot]}', fontsize=10.5)


# ---------------------------------------------------------------------------
def fig_lobes(cp, prot):
    d = cp['lobes']
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for k in KEYS:
        ax.plot(d['rpm'], d[k] * 1e3, '-o', ms=4, color=COL[k], lw=1.6,
                label=LAB[k])
    ax.axvline(C.RPM_DESIGN, color='k', ls=':', lw=1)
    ax.annotate('design speed', (C.RPM_DESIGN, ax.get_ylim()[1]), fontsize=8,
                rotation=90, va='top', ha='right')
    g_f = np.mean(d['fopid']) / np.mean(d['boucle ouverte'])
    g_a = np.mean(d['adrc']) / np.mean(d['boucle ouverte'])
    ax.text(0.02, 0.97, f'mean limit gain vs open loop:\n'
                        f'FOPID  x{g_f:.1f}\nADRC-FOPID  x{g_a:.1f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(fc='w', ec='0.7', alpha=.9))
    ax.set_xlabel('Spindle speed (rpm)')
    ax.set_ylabel('Limiting axial depth $a_{p,lim}$ (mm)')
    ax.grid(alpha=.3)
    ax.legend(fontsize=9, loc='upper right')
    _suptitle(fig, 'Stability lobes - worst position along the top edge '
                   '(5-mode model, Floquet m = 200)', prot)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_lobes_{prot}.png', dpi=140)
    plt.close(fig)


def fig_positions(cp, prot):
    d = cp['positions']
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    w = 0.26
    xs = np.arange(len(d['x']))
    for i, k in enumerate(KEYS):
        ax.bar(xs + (i - 1) * w, d[k] * 1e3, w, color=COL[k], label=LAB[k])
        for j, v in enumerate(d[k]):
            ax.text(xs[j] + (i - 1) * w, v * 1e3, f'{v * 1e3:.2f}',
                    ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{v * 100:.0f} %' for v in d['x']])
    ax.set_xlabel('Tool position along the top edge ($x/l_P$)')
    ax.set_ylabel('Limiting axial depth (mm)')
    ax.grid(alpha=.3, axis='y')
    ax.legend(fontsize=9)
    _suptitle(fig, f'Position-wise stability limit at {C.RPM_DESIGN} rpm',
              prot)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_positions_{prot}.png', dpi=140)
    plt.close(fig)


def fig_time(cp, prot, tag='', fname='fig_time'):
    meta = cp[f'time{tag}_meta']
    ap = float(meta['ap']) * 1e3
    fig, ax = plt.subplots(3, 2, figsize=(11.5, 9))
    for r, k in enumerate(KEYS):
        s, sp = cp[f'time{tag}_{k}'], cp[f'spec{tag}_{k}']
        ax[r, 0].fill_between(s['t'], s['y_mill_min'] * 1e6,
                              s['y_mill_max'] * 1e6, color=COL[k], alpha=.85,
                              lw=0, label=LAB[k])
        txt = (f"max |y| = {float(s['max_y']) * 1e6:.2f} $\\mu$m")
        if bool(s['diverged']):
            txt += f"\nDIVERGES at {float(s['t_div']):.3f} s"
        ax[r, 0].text(0.02, 0.95, txt, transform=ax[r, 0].transAxes,
                      va='top', fontsize=8,
                      bbox=dict(fc='w', ec='0.7', alpha=.85))
        ax[r, 0].set_ylabel('Displacement ($\\mu$m)')
        ax[r, 0].legend(fontsize=8, loc='upper right')
        ax[r, 1].plot(sp['f'], sp['A'], color=COL[k], lw=1.0)
        ax[r, 1].set_ylabel('Amplitude ($\\mu$m)')
        ax[r, 1].set_xlim(0, 1600)
        for a in ax[r]:
            a.grid(alpha=.3)
    for a in ax[:, 0]:
        a.set_xlabel('Time (s)')
    for a in ax[:, 1]:
        a.set_xlabel('Frequency (Hz)')
    _suptitle(fig, f'Milling pass at {C.RPM_DESIGN} rpm, '
                   f'$a_e$ = 0.1 mm, $a_p$ = {ap:.2f} mm '
                   f'(left: displacement at the cut, right: spectrum)', prot)
    fig.tight_layout()
    fig.savefig(f'{FIG}/{fname}_{prot}.png', dpi=140)
    plt.close(fig)


def fig_voltage(cp, prot, tag=''):
    meta = cp[f'time{tag}_meta']
    vmax = float(meta['v_max'])
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for k in ('fopid', 'adrc'):
        s = cp[f'time{tag}_{k}']
        ax[0].fill_between(s['t'], s['u_min'], s['u_max'], color=COL[k],
                           alpha=.6, lw=0,
                           label=f"{LAB[k]}  (peak {float(s['max_u']):.1f} V,"
                                 f" mean {float(s['mean_u']):.2f} V)")
        ax[1].plot(cp[f'spec{tag}_{k}']['fu'], cp[f'spec{tag}_{k}']['Au'],
                   color=COL[k], lw=1.1, label=LAB[k])
    for s in (+1, -1):
        ax[0].axhline(s * vmax, color='r', ls='--', lw=1,
                      label='amplifier saturation' if s > 0 else None)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Control voltage (V)')
    ax[0].set_title('(a) control voltage', fontsize=10)
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Amplitude (V)')
    ax[1].set_xlim(0, 1600)
    ax[1].set_title('(b) voltage spectrum', fontsize=10)
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    _suptitle(fig, 'Control effort under identical constraints and identical '
                   f'+/-{vmax:.0f} V saturation', prot)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_voltage_{prot}.png', dpi=140)
    plt.close(fig)


def fig_freq(cp, prot):
    d = cp['freq']
    f = d['f']
    fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.1))
    for k in ('fopid', 'adrc'):
        ax[0].loglog(f, d[f'K_{k}'], color=COL[k], lw=1.4, label=LAB[k])
        ax[1].semilogx(f, d[f'S_{k}'], color=COL[k], lw=1.4, label=LAB[k])
        ax[2].loglog(f, d[f'U_{k}'], color=COL[k], lw=1.4, label=LAB[k])
    ax[0].set_ylabel('$|K(j\\omega)|$  (V/m)')
    ax[0].set_title('(a) optimised controllers', fontsize=10)
    ax[1].axhline(C.MS_MAX, color='r', ls='--', lw=1,
                  label=f'constraint $M_s \\leq$ {C.MS_MAX}')
    ax[1].set_ylabel('$|S(j\\omega)|$')
    ax[1].set_title('(b) sensitivity', fontsize=10)
    ax[2].axhline(C.V_PER_N, color='r', ls='--', lw=1,
                  label=f'constraint {C.V_PER_N:.0f} V/N')
    ax[2].set_ylabel('$|K S P_f|$  (V/N)')
    ax[2].set_title('(c) actuator effort', fontsize=10)
    for a in ax:
        a.set_xlabel('Frequency (Hz)')
        a.grid(alpha=.3, which='both')
        a.legend(fontsize=8)
    _suptitle(fig, 'Frequency-domain signatures under identical constraints',
              prot)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_frequency_{prot}.png', dpi=140)
    plt.close(fig)


def fig_pso(ps, prot):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for k in ('fopid', 'adrc'):
        h = ps[k]['hist']
        it = np.arange(h.shape[1])
        for i in range(h.shape[0]):
            ax[0].plot(it, h[i], color=COL[k], lw=1.0, alpha=.55,
                       label=LAB[k] if i == 0 else None)
        ax[0].plot(it, h.max(axis=0), color=COL[k], lw=2.4)
    ax[0].set_xlabel('PSO iteration')
    ax[0].set_ylabel('$J$  (Floquet margin)')
    ax[0].set_title('(a) convergence, 3 seeds per structure', fontsize=10)
    ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=9)
    fin = np.concatenate([ps[k]['J_seeds'] for k in ('fopid', 'adrc')])
    lo = np.min(fin) - 0.4 * (np.ptp(fin) + 1e-3)
    ax[0].set_ylim(lo, np.max(fin) + 0.15 * (np.ptp(fin) + 1e-3))
    ks = ('fopid', 'adrc')
    xs = np.arange(len(ks))
    for i, k in enumerate(ks):
        js = ps[k]['J_seeds']
        ax[1].bar(xs[i], js.max(), 0.5, color=COL[k])
        ax[1].plot(np.full(len(js), xs[i]), js, 'ko', ms=6, zorder=5)
        ax[1].text(xs[i], js.max(), f'{js.max():+.3f}', ha='center',
                   va='bottom', fontsize=9)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"{LAB[k]}\n({int(ps[k]['n_par'])} parameters, "
                           f"{int(ps[k]['n_states'])} states)" for k in ks],
                          fontsize=9)
    ax[1].set_ylabel('final $J$')
    ax[1].set_title('(b) best value and seed spread', fontsize=10)
    ax[1].grid(alpha=.3, axis='y')
    ax[1].axhline(0, color='k', lw=.8)
    _suptitle(fig, 'PSO optimisation - identical budget '
                   f"({int(ps['fopid']['n_eval'])} evaluations each), "
                   'identical seeds, identical objective', prot)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_pso_{prot}.png', dpi=140)
    plt.close(fig)


def fig_robust(cp, prot):
    tags = [str(x) for x in cp['robust_labels']]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    w = 0.26
    xs = np.arange(len(tags))
    for i, name in enumerate(KEYS):
        v = np.array([cp['robust'][t][i] for t in tags]) * 1e3
        ax.bar(xs + (i - 1) * w, v, w, color=COL[name], label=LAB[name])
        for j, val in enumerate(v):
            ax.text(xs[j] + (i - 1) * w, val, f'{val:.2f}', ha='center',
                    va='bottom', fontsize=7, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels([ROB_EN.get(t, t).replace(' ', '\n', 1) for t in tags],
                       fontsize=8)
    ax.set_ylabel('Worst-position limit at design speed (mm)')
    ax.grid(alpha=.3, axis='y')
    ax.legend(fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    _suptitle(fig, 'Robustness - same controllers, perturbed plants', prot)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_robust_{prot}.png', dpi=140)
    plt.close(fig)


def fig_summary(cp, ps, prot):
    """Tableau recapitulatif dessine comme une figure."""
    lob = cp['lobes']
    pos = cp['positions']
    rows = [
        ('Design parameters', '-', f"{int(ps['fopid']['n_par'])}",
         f"{int(ps['adrc']['n_par'])}"),
        ('Controller states', '-', f"{int(ps['fopid']['n_states'])}",
         f"{int(ps['adrc']['n_states'])}"),
        ('PSO evaluations', '-', f"{int(ps['fopid']['n_eval'])}",
         f"{int(ps['adrc']['n_eval'])}"),
        ('Objective J', '-', f"{float(ps['fopid']['J']):+.3f}",
         f"{float(ps['adrc']['J']):+.3f}"),
    ]
    rows.append(('Lobes, mean 3000-7000 rpm (mm)',
                 *[f"{np.mean(lob[k]) * 1e3:.3f}" for k in KEYS]))
    rows.append(('Lobes, minimum (mm)',
                 *[f"{np.min(lob[k]) * 1e3:.3f}" for k in KEYS]))
    rows.append((f'Limit at {C.RPM_DESIGN} rpm, worst position (mm)',
                 *[f"{np.min(pos[k]) * 1e3:.3f}" for k in KEYS]))
    for tag, lab in ((''.join('_S'), f"$a_p$ = "
                      f"{float(cp['time_S_meta']['ap']) * 1e3:.2f} mm"),
                     ('', f"$a_p$ = "
                      f"{float(cp['time_meta']['ap']) * 1e3:.2f} mm")):
        rows.append((f'Max displacement, {lab} ($\\mu$m)',
                     *[('diverges' if bool(cp[f'time{tag}_{k}']['diverged'])
                        else f"{float(cp[f'time{tag}_{k}']['max_y']) * 1e6:.2f}")
                       for k in KEYS]))
        rows.append((f'Peak voltage, {lab} (V)', '-',
                     *[f"{float(cp[f'time{tag}_{k}']['max_u']):.1f}"
                       for k in ('fopid', 'adrc')]))
        rows.append((f'Mean |voltage|, {lab} (V)', '-',
                     *[f"{float(cp[f'time{tag}_{k}']['mean_u']):.2f}"
                       for k in ('fopid', 'adrc')]))
        rows.append((f'Saturated steps, {lab}', '-',
                     *[f"{int(cp[f'time{tag}_{k}']['n_saturated'])}"
                       for k in ('fopid', 'adrc')]))
    rows.append(('Modulus margin $M_s$ (constraint 2.0)', '-',
                 *[f"{cp[f'metrics_{k}'][0]:.3f}" for k in ('fopid', 'adrc')]))
    rows.append(('Actuator effort (V/N, constraint 450)', '-',
                 *[f"{cp[f'metrics_{k}'][1]:.0f}" for k in ('fopid', 'adrc')]))
    rows.append(('Slowest nominal pole (1/s)', '-',
                 *[f"{cp[f'metrics_{k}'][2]:.1f}" for k in ('fopid', 'adrc')]))

    fig, ax = plt.subplots(figsize=(11.5, 0.32 * len(rows) + 1.4))
    ax.axis('off')
    tab = ax.table(cellText=[list(r) for r in rows],
                   colLabels=['quantity', 'no control', 'FOPID',
                              'ADRC-FOPID'],
                   colWidths=[0.46, 0.18, 0.18, 0.18],
                   cellLoc='center', colLoc='center', loc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.35)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('0.8')
        if r == 0:
            cell.set_facecolor('#e8e8e8')
            cell.set_text_props(weight='bold')
        elif c == 0:
            cell.set_text_props(ha='left')
            cell._loc = 'left'
        elif c == 2:
            cell.set_facecolor('#eaeef8')
        elif c == 3:
            cell.set_facecolor('#e6f5f1')
    _suptitle(fig, 'Fair comparison summary - identical plant, objective, '
                   'constraints, seeds and evaluation;\nswarm size scales with '
                   'dimension, so the evaluation count differs and is reported',
              prot)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_summary_{prot}.png', dpi=140)
    plt.close(fig)


def fig_cross():
    """A contre B : ce que change le modele de synthese."""
    have = [p for p in ('A', 'B')
            if os.path.exists(os.path.join(OUT, f'compare_{p}.npz'))]
    if len(have) < 2:
        print('  (figure croisee ignoree : il manque un protocole)')
        return
    cps = {p: load(f'compare_{p}.npz') for p in have}
    fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.3))
    for i, p in enumerate(('A', 'B')):
        d = cps[p]['lobes']
        for k in KEYS:
            ax[i].plot(d['rpm'], d[k] * 1e3, '-o', ms=3, color=COL[k], lw=1.5,
                       label=LAB[k])
        ax[i].set_title(f'({chr(97 + i)}) protocol {p}', fontsize=10)
        ax[i].set_xlabel('Spindle speed (rpm)')
        ax[i].set_ylabel('$a_{p,lim}$ (mm)')
        ax[i].grid(alpha=.3)
        ax[i].legend(fontsize=8)
    w = 0.35
    xs = np.arange(2)
    for j, k in enumerate(('fopid', 'adrc')):
        v = [np.mean(cps[p]['lobes'][k]) * 1e3 for p in ('A', 'B')]
        ax[2].bar(xs + (j - 0.5) * w, v, w, color=COL[k], label=LAB[k])
        for x, val in zip(xs + (j - 0.5) * w, v):
            ax[2].text(x, val, f'{val:.2f}', ha='center', va='bottom',
                       fontsize=8)
    ol = [np.mean(cps[p]['lobes']['boucle ouverte']) * 1e3 for p in ('A', 'B')]
    ax[2].plot(xs, ol, 'o--', color=COL['boucle ouverte'], label='no control')
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(['A: design on 2 modes', 'B: design on 5 modes'],
                          fontsize=9)
    ax[2].set_ylabel('Mean $a_{p,lim}$ over 3000-7000 rpm (mm)')
    ax[2].set_title('(c) effect of the design model', fontsize=10)
    ax[2].grid(alpha=.3, axis='y')
    ax[2].legend(fontsize=8)
    fig.suptitle('Design model vs evaluation model - both structures always '
                 'scored on the full 5-mode plant', fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_protocols.png', dpi=140)
    plt.close(fig)


def main():
    if '--cross' in sys.argv:
        fig_cross()
        print(f'  -> {FIG}')
        return
    prot = C.PROTOCOL
    ps = load(f'pso_{prot}.npz')
    fig_pso(ps, prot)
    cp_path = os.path.join(OUT, f'compare_{prot}.npz')
    if os.path.exists(cp_path):
        cp = load(f'compare_{prot}.npz')
        fig_lobes(cp, prot)
        fig_positions(cp, prot)
        fig_time(cp, prot, tag='_S', fname='fig_time_S')
        fig_time(cp, prot, tag='', fname='fig_time')
        fig_voltage(cp, prot, tag='')
        fig_freq(cp, prot)
        fig_robust(cp, prot)
        fig_summary(cp, ps, prot)
    print(f'  -> {FIG}')


if __name__ == '__main__':
    main()
