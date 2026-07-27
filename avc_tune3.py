"""
Third tuning grid: the flat control weight (WU_FLAT) joins the search.

Root cause found while the F_WU / NOISE grid was still failing: the legacy
Wu shape (s + wz)/(s + 10 wz) / V_TARGET evaluates to 1/(10 V_TARGET) at low
frequency, and the resonant voltage peaks of modes 1-2 -- which is where
score() finds the 300+ V/N -- sit exactly in that 10x-weak zone.  gamma < 1
therefore never implied the budget was respected.  The flat shape holds the
full 1/V_TARGET through the resonances and still rises to 10/V_TARGET above
the corner, so V_TARGET finally means what it says.

Searched together with the two anti-inversion levers from the second grid
(early F_WU corner, heavier sensor-noise weight).
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

    print('=== mu grid, flat Wu ===', flush=True)
    for FWU in [1400.0, 2000.0]:
        for NZ in [1e-3, 1e-2]:
            for C in [30e-6, 45e-6]:
                for V in [110.0, 150.0]:
                    tag = 'W_F%.0f_N%.0e_C%.0f_V%.0f' % (FWU, NZ, C * 1e6, V)
                    payload, status = run_cell('mu', dict(
                        C=C, V=V, n_iter=4, FWU=FWU, NOISE=NZ, WUFLAT=True))
                    if status != 'ok':
                        print('  %s : %s' % (tag, status), flush=True)
                        continue
                    process(payload, P, Pm, Pmid, '_res_mu', tag)

    print('\n=== combined grid, flat Wu ===', flush=True)
    for FWU in [1400.0, 2000.0]:
        for NZ in [1e-3, 1e-2]:
            for g in [0.3, 0.5]:
                for C in [30e-6, 45e-6]:
                    tag = 'W_F%.0f_N%.0e_g%02.0f_C%.0f_V150' % (FWU, NZ,
                                                                g * 10,
                                                                C * 1e6)
                    payload, status = run_cell('cb', dict(
                        C=C, V=150.0, n_iter=4, g=g, FWU=FWU, NOISE=NZ,
                        WUFLAT=True,
                        kp=par['kp'], kd=par['kd'], wc=par['wc']))
                    if status != 'ok':
                        print('  %s : %s' % (tag, status), flush=True)
                        continue
                    process(payload, P, Pm, Pmid, '_res_cb', tag,
                            extra=dict(inner_gain=g))
    print('\nflat-Wu grid done', flush=True)


if __name__ == '__main__':
    main()
