"""
Fourth (focused) grid: the immediate neighbourhood of the two winners.

Both two-plant-feasible winners came from the flat-Wu, low-noise corner
(mu: W_F2000_N1e-03_C45_V110; combined: W_F1400_N1e-03_g05_C45_V150), so
this grid samples around them -- softer C, both V levels, both F_WU
corners, both useful inner gains -- with six D-K draws per cell instead of
four (the good iterate is a draw, more draws per cell is the cheapest way
to harvest).
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

    print('=== mu focused grid ===', flush=True)
    for C in [45e-6, 60e-6]:
        for V in [110.0, 130.0]:
            tag = 'W4_F2000_C%.0f_V%.0f' % (C * 1e6, V)
            payload, status = run_cell('mu', dict(
                C=C, V=V, n_iter=6, FWU=2000.0, NOISE=1e-3, WUFLAT=True))
            if status != 'ok':
                print('  %s : %s' % (tag, status), flush=True)
                continue
            process(payload, P, Pm, Pmid, '_res_mu', tag)

    print('\n=== combined focused grid ===', flush=True)
    for g in [0.5, 0.7]:
        for FWU in [1400.0, 2000.0]:
            for C in [45e-6, 60e-6]:
                tag = 'W4_F%.0f_g%02.0f_C%.0f_V150' % (FWU, g * 10, C * 1e6)
                payload, status = run_cell('cb', dict(
                    C=C, V=150.0, n_iter=6, g=g, FWU=FWU, NOISE=1e-3,
                    WUFLAT=True,
                    kp=par['kp'], kd=par['kd'], wc=par['wc']))
                if status != 'ok':
                    print('  %s : %s' % (tag, status), flush=True)
                    continue
                process(payload, P, Pm, Pmid, '_res_cb', tag,
                        extra=dict(inner_gain=g))
    print('\nfocused grid done', flush=True)


if __name__ == '__main__':
    main()
