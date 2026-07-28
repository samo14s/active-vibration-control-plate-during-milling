"""
Manuscript figures from the _newlaw_*.npz artefacts.

  figs/paper_sld.png     SLDs: uncontrolled / PD inner / new law (fresh) and
                         the machined panel (stage-matched vs frozen law)
  figs/paper_frontier.png  price-of-robustness sweep + per-stage floors
                           against the authority frontier
  figs/paper_time.png    time-domain showcase at the benchmark's S point
  figs/paper_authority.png  authority map at the worst position
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import milling_model as mm
import avc_frontier as af
from avc_plant import Plant

plt.style.use('seaborn-v0_8-whitegrid')
C = {'unc': '#8C8C8C', 'pd': '#1D9E75', 'new': '#D85A30', 'sched': '#378ADD',
     'frontier': '#222222'}


def fig_sld():
    d = np.load('_newlaw_sld.npz')
    d2 = np.load('_newlaw_sld2.npz')
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    rpm = d['rpm']
    ax[0].plot(rpm, d['unc'] * 1e3, color=C['unc'], lw=1.6, label='uncontrolled')
    ax[0].plot(rpm, d['pd'] * 1e3, color=C['pd'], lw=1.6, label='passive PD inner')
    ax[0].plot(rpm, d['new'] * 1e3, color=C['new'], lw=2.0,
               label='passive + convex (new law)')
    try:
        sc = np.load('_newlaw_sched.npz')
        for k in sc.files:
            r = float(k.split('_')[1])
            ax[0].plot([r], [sc[k][-1] * 1e3], 'v', color=C['sched'], ms=7)
        ax[0].plot([], [], 'v', color=C['sched'], label='speed-scheduled LP (ZOA)')
    except Exception:
        pass
    ax[0].scatter([4900], [0.3], marker='*', s=110, color='k', zorder=5)
    ax[0].annotate('S (benchmark)', (4900, 0.3), textcoords='offset points',
                   xytext=(6, 6), fontsize=8)
    ax[0].set_title('Fresh plate, worst position (x = 100 mm)', fontsize=11)
    ax[1].plot(rpm, d2['machined'] * 1e3, color=C['new'], lw=2.0,
               label='stage-matched law (adaptive)')
    ax[1].plot(rpm, d2['cross'] * 1e3, color=C['new'], lw=1.4, ls='--',
               label='frozen fresh law (no re-solve)')
    ax[1].set_title('Machined plate (f1 +17 %, f2 +9 %)', fontsize=11)
    for a in ax:
        a.set_xlabel('spindle speed [rpm]')
        a.legend(fontsize=8.5)
        a.set_ylim(0, None)
    ax[0].set_ylabel('FDM stability limit  a$_p$ [mm]')
    fig.suptitle('Full-discretisation stability lobes, IJMS-274 milling model '
                 'on the MITC4 plant', fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figs/paper_sld.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig_frontier():
    dd = np.load('_newlaw_drift.npz')
    dr = dd['drift']
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    ax[0].semilogx(dr[:, 0] * 100, dr[:, 1] * 1e3, 'o-', color=C['new'], lw=1.8)
    ax[0].axhline(float(dd['f_pd']) * 1e3, color=C['pd'], lw=1.4, ls='--')
    ax[0].text(0.55, float(dd['f_pd']) * 1e3 * 1.03, 'passive PD alone',
               fontsize=8.5, color=C['pd'])
    ax[0].axhline(float(dd['f_unc']) * 1e3, color=C['unc'], lw=1.2, ls=':')
    ax[0].text(0.55, float(dd['f_unc']) * 1e3 * 1.2, 'uncontrolled',
               fontsize=8.5, color=C['unc'])
    ax[0].set_xlabel('frequency drift ridden out between re-solves [%]')
    ax[0].set_ylabel('guaranteed ZOA floor [mm]')
    ax[0].set_title('The price of robustness', fontsize=11, fontweight='bold')

    tags = ['fresh', 'mid', 'machined']
    floors, fronts, reals = [], [], []
    for t in tags:
        s = np.load('_newlaw_stage_%s.npz' % t)
        floors.append(float(s['floor']) * 1e3)
        fronts.append(float(s['frontier']) * 1e3)
        reals.append(float(s['floor_real']) * 1e3)
    xx = np.arange(3)
    ax[1].bar(xx - 0.18, floors, 0.34, color=C['new'], label='guaranteed floor (SOCP)')
    ax[1].bar(xx + 0.18, reals, 0.34, color=C['new'], alpha=0.45,
              label='realised & reduced law')
    for i, f in enumerate(fronts):
        ax[1].plot([i - 0.4, i + 0.4], [f, f], color=C['frontier'], lw=2.0)
    ax[1].plot([], [], color=C['frontier'], lw=2.0,
               label='authority frontier (any LTI law)')
    ax[1].set_xticks(xx); ax[1].set_xticklabels(tags)
    ax[1].set_ylabel('a$_p$ floor [mm]')
    ax[1].set_title('Per-stage floors vs the single-patch ceiling',
                    fontsize=11, fontweight='bold')
    ax[1].legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig('figs/paper_frontier.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig_time():
    d = np.load('_newlaw_sim.npz')
    fig, ax = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
    for tag, col, lab in [('uncontrolled', C['unc'], 'uncontrolled'),
                          ('pd', C['pd'], 'passive PD'),
                          ('new', C['new'], 'passive + convex')]:
        ax[0].plot(d[tag + '_t'] * 1e3, d[tag + '_y'] * 1e6, lw=0.7,
                   color=col, label=lab)
        if tag != 'uncontrolled':
            ax[1].plot(d[tag + '_t'] * 1e3, d[tag + '_u'], lw=0.7, color=col)
    ax[0].set_ylabel('w at cutting point [µm]')
    ax[0].legend(fontsize=8.5)
    ax[0].set_title("Milling at the benchmark's S point (4900 rpm, "
                    'a$_p$ = 0.3 mm, a$_e$ = 0.1 mm)', fontsize=12,
                    fontweight='bold')
    ax[1].axhline(200, color='#993C1D', lw=1, ls=':')
    ax[1].text(2, 205, 'patch limit 200 V', fontsize=8, color='#993C1D')
    ax[1].set_ylabel('patch voltage [V]')
    ax[1].set_xlabel('time [ms]')
    plt.tight_layout()
    plt.savefig('figs/paper_time.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig_authority():
    om = 2 * np.pi * np.geomspace(200, 5200, 900)
    P = Plant(zeta=mm.ZETA_PAPER)
    x = 100.0
    D = P.edge_D(x)
    GDD = P.frf(D, D, om)
    GDu = P.frf(D, P.bmod, om)
    need = (np.abs(GDD) - GDD.real) * 1e6
    have = 2 * np.abs(GDu) * 150.0 * 1e6
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    fr = om / 2 / np.pi
    ax.semilogy(fr, need, lw=1.8, color=C['unc'],
                label='required:  |G$_{DD}$| $-$ Re G$_{DD}$')
    ax.semilogy(fr, have, lw=1.8, color=C['new'],
                label='available:  2 |G$_{Du}$| $\\cdot$ 150 V/N')
    ax.fill_between(fr, need, have, where=need > have, color='#D85A30',
                    alpha=0.25, label='authority deficit -> frontier')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('[µm/N]')
    ax.set_title('Authority map at the worst position: where the patch runs '
                 'out', fontsize=11.5, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('figs/paper_authority.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    fig_authority()
    for fn in (fig_sld, fig_frontier, fig_time):
        try:
            fn()
        except FileNotFoundError as e:
            print('skip %s: %s' % (fn.__name__, e))
    print('paper figures written')
