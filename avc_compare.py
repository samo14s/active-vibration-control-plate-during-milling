"""
Four-way comparison: PPF (baseline) vs PD vs mu-synthesis vs PD+mu combined.

Every law is closed on the SAME plant, scored by the SAME metrics, and then
replayed with FROZEN gains on the machined plate (f1 +17 %, f2 +11 %) to see
what the robustness actually bought.
"""
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from avc_plant import (Plant, Ctrl, score, settling, settle_time, close,
                       NM, ZETA, VLIM, ZMIN, REMOVAL)
from avc_pd import pd_ctrl

plt.style.use('seaborn-v0_8-whitegrid')
OUT = 'figs/'
COL = {'PPF': '#B4B2A9', 'PD': '#1D9E75', 'mu': '#378ADD', 'combined': '#D85A30'}


def ppf_ctrl(g, P, NF=2, zf=0.40):
    wf = P.wn[:NF].copy()
    scale = P.wn[:NF] ** 2 / (P.bmod[:NF] * P.cS[:NF])
    kap = np.asarray(g) * scale
    Ak = np.zeros((2 * NF, 2 * NF)); Ak[:NF, NF:] = np.eye(NF)
    Ak[NF:, :NF] = -np.diag(wf ** 2); Ak[NF:, NF:] = -np.diag(2 * zf * wf)
    Bk = np.zeros(2 * NF); Bk[NF:] = wf ** 2
    Ck = np.zeros(2 * NF); Ck[:NF] = kap
    return Ctrl(Ak, Bk, Ck, 0.0, 'PPF')


def load_best(pattern, label):
    """Selection rule, in order of preference:
       1. two-plant feasible (stable + <= VLIM + >= ZMIN on the fresh AND
          the machined plate, stable at mid-excursion) AND certified
          (full-structure peak MIXED mu < 1 at the design's own weights):
          largest fresh a_lim;
       2. two-plant feasible only (its mu peak lands >= 1 in the table).
    The old workflow admitted the combined law at 151.3 V/N with a 2 %
    'tolerance' -- that concession is gone; VLIM is VLIM."""
    cands = []
    for fn in sorted(glob.glob(pattern)):
        d = np.load(fn)
        feas = (bool(d['stable']) and bool(d['m_stable'])
                and bool(d['mid_stable'])
                and float(d['volt']) <= VLIM and float(d['m_volt']) <= VLIM
                and float(d['zeta_min']) >= ZMIN
                and float(d['m_zeta_min']) >= ZMIN)
        if not feas:
            continue
        pk = float(d['peak_mixed']) if 'peak_mixed' in d.files else float(d['peak'])
        cands.append((float(d['a_lim']), pk, fn, d))
    if not cands:
        raise RuntimeError('no two-plant-feasible design for ' + label)
    certified = [c for c in cands if np.isfinite(c[1]) and c[1] < 1.0]
    pool = certified if certified else cands
    best = max(pool, key=lambda c: c[0])
    d = best[3]
    K = Ctrl(d['Ak'], d['Bk'].ravel(), d['Ck'].ravel(), float(d['Dk']), label)
    return K, best[1], d['mu'], d['om'], best[2]


def main():
    P = Plant()
    par = np.load('_pd.npz')

    laws = {}
    laws['PPF'] = ppf_ctrl([0.092, 0.055], P)
    laws['PD'] = pd_ctrl(float(par['kp']), float(par['kd']), float(par['wc']), 'PD')
    K_mu, mu_peak, mu_c, mu_om, fmu = load_best('_res_mu_*.npz', 'mu-synthesis')
    laws['mu'] = K_mu
    K_cb, cb_peak, cb_c, cb_om, fcb = load_best('_res_cb_*.npz',
                                                'PD + mu (combined)')
    laws['combined'] = K_cb
    peaks_mu = {'PPF': None, 'PD': None, 'mu': mu_peak, 'combined': cb_peak}
    print('selected  mu : %s   combined : %s\n' % (fmu, fcb))

    # canonical archive copies of the selected designs
    import shutil
    shutil.copy(fmu, 'ctrl_mu.npz')
    shutil.copy(fcb, 'ctrl_combined.npz')
    shutil.copy('_pd.npz', 'pd_gains.npz')

    # one COMMON certificate for all four laws: mixed-mu (real modal deltas
    # + real actuator gain via G-scales) on the full six-mode structure, at
    # the physical budget V = 150 V/N and a 45 um/N performance bound, flat
    # control weight.  Certifying every law on the same structure at the
    # same weights is what makes the mu row of the table a comparison
    # rather than four unrelated numbers.
    import control as ct
    import avc_mu
    CERT = dict(C=45e-6, V=150.0, FWU=2000.0, NOISE=1e-3, FLAT=True)
    avc_mu.C_TARGET, avc_mu.V_TARGET = CERT['C'], CERT['V']
    avc_mu.F_WU, avc_mu.NOISE = CERT['FWU'], CERT['NOISE']
    avc_mu.WU_FLAT = CERT['FLAT']
    P_full, sc = avc_mu.gen_plant(P, P.tools['corner (100, 80)'], unc=None)
    mu_mx = {}
    for k in laws:
        Ks = avc_mu._scale_ctrl(laws[k], sc)
        mu_mx[k] = avc_mu.mu_curve_mixed(ct.ss(Ks), ct.ss(P_full),
                                         mu_om / sc['w_ref'])
        peaks_mu[k] = float(mu_mx[k].max())
        print('  certified %-24s mixed mu %.3f' % (laws[k].name, peaks_mu[k]),
              flush=True)

    # ---------------------------------------------------- nominal
    Pm = Plant(freq_scale=REMOVAL)          # machined plate, same mode shapes
    res_n, res_m = {}, {}
    for k, K in laws.items():
        res_n[k] = score(P, K, full=True)
        res_m[k] = score(Pm, K)

    cT = P.tools['corner (100, 80)']
    t, y_ol, _ = settling(P, Ctrl(np.zeros((0, 0)), [], [], 0.0, 'open'), cT)
    env = np.abs(y_ol).max()
    st = {'open loop': settle_time(t, y_ol, env)}
    tv = {}
    for k, K in laws.items():
        tt, yy, vv = settling(P, K, cT)
        st[k] = settle_time(tt, yy, env)
        tv[k] = (tt, yy, vv)

    # ---------------------------------------------------- table
    hdr = ('%-26s %10s %10s %10s %10s' % ('', 'PPF', 'PD', 'mu-synth', 'combined'))
    print(hdr); print('-' * len(hdr))
    rows = [
        ('a_lim lift (worst tool)', lambda r: 'x%.2f' % r['a_lim']),
        ('peak receptance [um/N]', lambda r: '%.1f' % (r['peak_cl'] * 1e6)),
        ('peak reduction', lambda r: 'x%.1f' % r['peak_ratio']),
        ('zeta mode 1 [%]', lambda r: '%.2f' % r['zeta1']),
        ('zeta mode 2 [%]', lambda r: '%.2f' % r['zeta2']),
        ('min zeta all modes [%]', lambda r: '%.2f' % (r['zeta_min'] * 100)),
        ('static stiffness ratio', lambda r: '%.3f' % r['static_ratio']),
        ('peak effort [V/N]', lambda r: '%.1f' % r['volt']),
        ('controller order', lambda r: '%d' % r['order']),
    ]
    for name, f in rows:
        print('%-26s %10s %10s %10s %10s'
              % (name, f(res_n['PPF']), f(res_n['PD']), f(res_n['mu']), f(res_n['combined'])))
    print('%-26s %10.1f %10.1f %10.1f %10.1f'
          % ('5% settling [ms]', st['PPF'], st['PD'], st['mu'], st['combined']))
    print('%-26s %10.3f %10.3f %10.3f %10.3f'
          % ('peak mu upper bound', peaks_mu['PPF'], peaks_mu['PD'],
             peaks_mu['mu'], peaks_mu['combined']))
    print('\n(open loop: peak %.1f um/N, static %.3f um/N, settling %.0f ms)'
          % (res_n['PPF']['peak_ol'] * 1e6, res_n['PPF']['static_ol'] * 1e6, st['open loop']))

    print('\n--- gains FROZEN, replayed on the machined plate (f1 +17 %, f2 +11 %) ---')
    print('%-26s %10s %10s %10s %10s' % ('', 'PPF', 'PD', 'mu-synth', 'combined'))
    for name, f in [('a_lim lift', lambda r: ('x%.2f' % r['a_lim']) if r['stable'] else 'UNSTABLE'),
                    ('peak receptance [um/N]', lambda r: '%.1f' % (r['peak_cl'] * 1e6)),
                    ('peak effort [V/N]', lambda r: '%.1f' % r['volt']),
                    ('min zeta [%]', lambda r: '%.2f' % (r['zeta_min'] * 100))]:
        print('%-26s %10s %10s %10s %10s'
              % (name, f(res_m['PPF']), f(res_m['PD']), f(res_m['mu']), f(res_m['combined'])))
    print('%-26s %10s %10s %10s %10s'
          % ('a_lim retained', *['%.0f %%' % (100 * res_m[k]['a_lim'] / res_n[k]['a_lim'])
                                 for k in ['PPF', 'PD', 'mu', 'combined']]))

    # ---------------------------------------------------- fig 1 : FRF
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8), sharex='col')
    for j, tag in enumerate(P.tools):
        d0 = res_n['PPF']['per_tool'][tag]
        ax = axs[0, j]
        ax.semilogy(d0['fr'], np.abs(d0['Gol']) * 1e6, lw=1.6, color='0.55',
                    ls='--', label='open loop')
        for k in laws:
            d = res_n[k]['per_tool'][tag]
            ax.semilogy(d['fr'], np.abs(d['Gcl']) * 1e6, lw=1.5, color=COL[k],
                        label=laws[k].name)
        ax.set_ylabel('|receptance| [µm/N]')
        ax.set_title('Tool at ' + tag, fontsize=11)
        ax.set_xlim(50, 4500)
        if j == 0:
            ax.legend(fontsize=8.5)
        ax = axs[1, j]
        m = d0['fr'] < 1500
        ax.plot(d0['fr'][m], d0['Gol'].real[m] * 1e6, lw=1.6, color='0.55', ls='--')
        for k in laws:
            d = res_n[k]['per_tool'][tag]
            ax.plot(d['fr'][m], d['Gcl'].real[m] * 1e6, lw=1.5, color=COL[k])
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('frequency [Hz]'); ax.set_ylabel('Re(FRF) [µm/N]')
    fig.suptitle('Four control laws on the same plate: amplitude (top) and the '
                 'chatter-relevant real part (bottom)', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(OUT + 'cmp_frf.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------- fig 2 : time + effort
    fig, ax = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True)
    ax[0].plot(t * 1000, y_ol * 1e6, lw=0.6, color='0.6', label='open loop')
    for k in laws:
        tt, yy, vv = tv[k]
        ax[0].plot(tt * 1000, yy * 1e6, lw=1.0, color=COL[k], label=laws[k].name)
        ax[1].plot(tt * 1000, vv, lw=1.0, color=COL[k])
    ax[0].set_ylabel('w at the free corner [µm]\nfor a 1 N·s impulse')
    ax[0].legend(fontsize=8.5, ncol=2)
    ax[0].set_title('Impulse response and control effort', fontsize=12, fontweight='bold')
    ax[1].axhline(200, color='#993C1D', lw=1, ls=':')
    ax[1].text(52, 205, 'patch limit 200 V', fontsize=8, color='#993C1D')
    ax[1].set_ylabel('patch voltage [V]'); ax[1].set_xlabel('time [ms]')
    ax[0].set_xlim(0, 60)
    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT + 'cmp_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------- fig 3 : mu + robustness
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for k in ['PPF', 'PD', 'mu', 'combined']:
        ax[0].semilogx(mu_om / 2 / np.pi, mu_mx[k], lw=1.7, color=COL[k],
                       label='%s (peak %.2f)' % (laws[k].name, peaks_mu[k]))
    ax[0].axhline(1.0, color='k', lw=1, ls='--')
    ax[0].text(12, 1.03, 'robust performance boundary', fontsize=8)
    ax[0].set_xlabel('frequency [Hz]')
    ax[0].set_ylabel('mixed-mu upper bound')
    ax[0].set_title('Mixed-mu certificate, full structure (six real modal\n'
                    'deltas + actuator gain), at C = 45 µm/N, V = 150 V/N',
                    fontsize=10.5, fontweight='bold')
    ax[0].legend(fontsize=8.5)

    ks = ['PPF', 'PD', 'mu', 'combined']
    xx = np.arange(len(ks)); w = 0.38
    ax[1].bar(xx - w / 2, [res_n[k]['a_lim'] for k in ks], w,
              color=[COL[k] for k in ks], label='fresh plate')
    ax[1].bar(xx + w / 2, [res_m[k]['a_lim'] for k in ks], w,
              color=[COL[k] for k in ks], alpha=0.45, hatch='//',
              label='machined (f1 +17 %, f2 +11 %)')
    ax[1].set_xticks(xx); ax[1].set_xticklabels([laws[k].name for k in ks], fontsize=8.5)
    ax[1].set_ylabel('stability-limit lift  a$_{lim}$ / a$_{lim,0}$')
    ax[1].set_title('Productivity, before and after material removal',
                    fontsize=11, fontweight='bold')
    ax[1].legend(fontsize=8.5)
    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT + 'cmp_robust.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\nfigures written to', OUT)


if __name__ == '__main__':
    main()
