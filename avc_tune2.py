"""
Supplementary tuning grid: the control-weight corner F_WU joins the search.

Diagnosis from the main campaign: H-infinity iterates invert the design
model's high modes; the +5 % machined shift moves those modes off the
inverted notches and the replay goes unstable or under the zeta floor.
Rolling the control weight off EARLIER (F_WU 1000 / 1400 Hz instead of
2000 Hz) forbids controller authority near modes 3-6 in the first place --
spillover safety by construction rather than by luck.  The collocated PD
inner loop keeps its authority there (it is passive on that pair), so the
combined law is the natural beneficiary.
"""
import numpy as np
import avc_mu
from avc_plant import Plant, REMOVAL
import avc_tune
from avc_tune import run_cell, process, MID


def main():
    avc_tune._P = Plant()
    P = avc_tune._P
    Pm = Plant(freq_scale=REMOVAL)
    Pmid = Plant(freq_scale=MID)
    par = np.load('_pd.npz')

    print('=== mu grid, F_WU axis ===', flush=True)
    for FWU in [1000.0, 1400.0]:
        for C in [25e-6, 30e-6, 35e-6, 45e-6]:
            for V in [130.0, 150.0, 170.0]:
                tag = 'F%.0f_C%.0f_V%.0f' % (FWU, C * 1e6, V)
                payload, status = run_cell('mu', dict(C=C, V=V, n_iter=4,
                                                      FWU=FWU))
                if status != 'ok':
                    print('  %s : %s' % (tag, status), flush=True)
                    continue
                process(payload, P, Pm, Pmid, '_res_mu', tag)

    print('\n=== combined grid, F_WU axis ===', flush=True)
    for FWU in [1000.0, 1400.0]:
        for g in [0.3, 0.5, 0.7]:
            for C in [30e-6, 45e-6]:
                for V in [150.0]:
                    tag = 'F%.0f_g%02.0f_C%.0f_V%.0f' % (FWU, g * 10,
                                                         C * 1e6, V)
                    payload, status = run_cell('cb', dict(
                        C=C, V=V, n_iter=4, g=g, FWU=FWU,
                        kp=par['kp'], kd=par['kd'], wc=par['wc']))
                    if status != 'ok':
                        print('  %s : %s' % (tag, status), flush=True)
                        continue
                    process(payload, P, Pm, Pmid, '_res_cb', tag,
                            extra=dict(inner_gain=g))
    print('\nsupplementary grid done', flush=True)


if __name__ == '__main__':
    main()
