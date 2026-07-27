"""
Tuning campaign for the mu and PD+mu laws, with feasibility enforced on the
MACHINED plate as well as the fresh one.

What went wrong last time, and what this driver changes:

  1. The old workflow tuned against fresh-plate feasibility only and then
     DISCOVERED in the replay that the robust designs fall below the 0.45 %
     spillover floor on the machined plate (mu 0.39 %, combined 0.43 %) --
     the robust laws violating their own constraint under the very
     perturbation they were synthesised against.  Here the machined-plate
     replay (and a mid-excursion stability check) is part of the acceptance
     test of every candidate, not a post-mortem.
  2. The old certificate did not even cover the replay: its uncertainty
     structure stopped at modes 1-2, while material removal also shifts
     modes 3-6 by +5 %.  Every feasible candidate is now certified on the
     FULL six-mode + actuator structure (avc_mu.mu_cert); a peak below 1
     now means robust performance over the whole modelled excursion, replay
     included.
  3. The D-K iteration is non-monotone, and the best-a_lim controller is
     frequently NOT the smallest-mu iterate.  The old code kept only the
     smallest-mu controller per (C, V) cell; this driver scores EVERY
     iterate of every cell on both plants and lets the selector choose.
  4. The weight grid is finer, and the combined law adds an inner-gain axis
     at de-rated PD levels, where voltage headroom for the outer loop
     actually exists.

Each cell's D-K runs in a forked subprocess with a hard timeout: SB10AD's
gamma iteration can hang outright on a badly D-scaled plant (observed at
100 % CPU indefinitely), and one wild cell must not take the campaign down.
Certificates and scoring run in the parent (pure numpy, iteration-bounded).

Every candidate is saved to _res_mu_*.npz / _res_cb_*.npz with its fresh
AND machined metrics; avc_compare.py applies the final selection rule.
Run single-threaded (OMP_NUM_THREADS=1) for reproducibility: the D-K
amplifies BLAS reduction-order noise into different iterates.
"""
import multiprocessing as mp
import numpy as np
import avc_mu
import avc_combined
from avc_plant import Plant, Ctrl, score, VLIM, ZMIN, REMOVAL
from avc_pd import design as pd_design, pd_ctrl

MID = 1.0 + 0.5 * (REMOVAL - 1.0)       # mid-excursion frequency scale
CELL_TIMEOUT = 150.0                    # s per D-K cell; productive cells
                                        # finish in well under 60 s, a hung
                                        # SB10AD never recovers

_P = None                               # fork-inherited by cell workers


def plant_metrics(K, P, Pm, Pmid):
    """Score a candidate on fresh, machined and mid-excursion plants."""
    r = score(P, K)
    rm = score(Pm, K)
    rmid_stable = score(Pmid, K)['stable']
    return dict(
        Ak=K.Ak, Bk=K.Bk, Ck=K.Ck, Dk=K.Dk, order=K.order,
        stable=r['stable'], a_lim=r['a_lim'], volt=r['volt'],
        zeta_min=r['zeta_min'], peak_cl=r['peak_cl'],
        static_ratio=r['static_ratio'],
        m_stable=rm['stable'], m_a_lim=rm['a_lim'], m_volt=rm['volt'],
        m_zeta_min=rm['zeta_min'], m_peak_cl=rm['peak_cl'],
        mid_stable=rmid_stable,
    )


def feasible(d):
    """Two-plant feasibility: the machined plate is part of the constraint
    set, not a post-mortem."""
    return (bool(d['stable']) and bool(d['m_stable']) and bool(d['mid_stable'])
            and float(d['volt']) <= VLIM and float(d['m_volt']) <= VLIM
            and float(d['zeta_min']) >= ZMIN and float(d['m_zeta_min']) >= ZMIN)


def _cell_worker(q, kind, params):
    try:
        avc_mu.C_TARGET, avc_mu.V_TARGET = params['C'], params['V']
        if 'FWU' in params:
            avc_mu.F_WU = params['FWU']
        if kind == 'mu':
            recs, om, P_full, sc = avc_mu.mu_ctrl_all(
                _P, n_iter=params['n_iter'], verbose=False)
        else:
            K_pd = pd_ctrl(params['kp'], params['kd'], params['wc'], 'PD')
            recs, om, P_full, sc = avc_combined.design_all(
                _P, K_pd, params['C'], params['V'],
                inner_gain=params['g'], n_iter=params['n_iter'])
        out = [dict(it=r['it'], gamma=r['gamma'], peak_syn=r['peak_syn'],
                    K=(r['K'].Ak, r['K'].Bk, r['K'].Ck, r['K'].Dk),
                    K_ss=r['K_ss']) for r in recs]
        q.put(('ok', (out, om, P_full, sc)))
    except Exception as e:
        q.put(('err', repr(e)))


def run_cell(kind, params, timeout=CELL_TIMEOUT, retries=1):
    """One D-K cell in a forked subprocess with a hard timeout.

    SB10AD's hangs are not a property of the cell: run-to-run the same cell
    can hang or sail through (LAPACK-level nondeterminism the single-thread
    pin does not remove), so a timed-out cell gets a fresh fork before it
    is given up on."""
    for attempt in range(retries + 1):
        ctx = mp.get_context('fork')
        q = ctx.Queue()
        p = ctx.Process(target=_cell_worker, args=(q, kind, params))
        p.start()
        try:
            status, payload = q.get(timeout=timeout)
        except Exception:
            status, payload = 'timeout', None
        p.join(5)
        if p.is_alive():
            p.terminate()
            p.join(5)
        if p.is_alive():                # SIGTERM is enough for a process
            p.kill()                    # spinning in Fortran, but belt and
            p.join()                    # braces costs nothing
        if status == 'ok':
            return payload, status
    return None, status


def process(payload, P, Pm, Pmid, prefix, tag, extra=None):
    recs, om, P_full, sc = payload
    for rec in recs:
        Ak, Bk, Ck, Dk = rec['K']
        K = Ctrl(Ak, np.asarray(Bk).ravel(), np.asarray(Ck).ravel(),
                 float(np.asarray(Dk).ravel()[0]))
        d = plant_metrics(K, P, Pm, Pmid)
        d['peak_syn'] = rec['peak_syn']
        d['gamma'] = rec['gamma']
        d['om'] = om
        if extra:
            d.update(extra)
        ok = feasible(d)
        if ok:
            d['peak'], d['mu'] = avc_mu.mu_cert(rec['K_ss'], P_full, om, sc)
        else:
            d['peak'], d['mu'] = np.nan, np.full(len(om), np.nan)
        np.savez('%s_%s_i%d.npz' % (prefix, tag, rec['it']), **d)
        print('  %s i%d : mu_syn %.3f  mu_full %s  a_lim x%.2f  volt %.0f  zmin %.2f%% | machined x%.2f  volt %.0f  zmin %.2f%%  %s'
              % (tag, rec['it'], rec['peak_syn'],
                 ('%.3f' % d['peak']) if np.isfinite(d['peak']) else '  -  ',
                 d['a_lim'], d['volt'], d['zeta_min'] * 100, d['m_a_lim'],
                 d['m_volt'], d['m_zeta_min'] * 100,
                 'FEAS' if ok else 'infeas'), flush=True)


def main():
    global _P
    _P = Plant()
    P = _P
    Pm = Plant(freq_scale=REMOVAL)
    Pmid = Plant(freq_scale=MID)

    # ------------------------------------------------------------------ PD
    print('=== PD design ===', flush=True)
    import os
    if os.path.exists('_pd.npz'):
        par = dict(np.load('_pd.npz'))
        K_pd = pd_ctrl(float(par['kp']), float(par['kd']), float(par['wc']),
                       'PD')
        print('loaded _pd.npz: kp %.3e  kd %.3e' % (par['kp'], par['kd']),
              flush=True)
    else:
        K_pd, par = pd_design(P)
        np.savez('_pd.npz', **par)

    # ------------------------------------------------------------------ mu
    print('\n=== mu-synthesis weight grid ===', flush=True)
    for C in [20e-6, 25e-6, 30e-6, 35e-6, 45e-6]:
        for V in [110.0, 130.0, 150.0, 170.0, 200.0]:
            tag = 'C%.0f_V%.0f' % (C * 1e6, V)
            payload, status = run_cell('mu', dict(C=C, V=V, n_iter=4))
            if status != 'ok':
                print('  %s : %s' % (tag, status), flush=True)
                continue
            process(payload, P, Pm, Pmid, '_res_mu', tag)

    # ------------------------------------------------------------ combined
    print('\n=== PD + mu combined grid ===', flush=True)
    for g in [0.3, 0.5, 0.7, 1.0]:
        for C in [25e-6, 30e-6, 35e-6, 45e-6]:
            for V in [130.0, 150.0]:
                tag = 'g%02.0f_C%.0f_V%.0f' % (g * 10, C * 1e6, V)
                payload, status = run_cell('cb', dict(
                    C=C, V=V, n_iter=4, g=g,
                    kp=par['kp'], kd=par['kd'], wc=par['wc']))
                if status != 'ok':
                    print('  %s : %s' % (tag, status), flush=True)
                    continue
                process(payload, P, Pm, Pmid, '_res_cb', tag,
                        extra=dict(inner_gain=g))
    print('\ntuning campaign done', flush=True)


if __name__ == '__main__':
    main()
