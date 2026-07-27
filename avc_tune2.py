"""
Supplementary tuning grid: control-weight corner F_WU and sensor-noise
weight NOISE join the search.

Diagnosis from the main campaign (103 candidates, zero two-plant feasible):
every H-infinity iterate inverts the design model -- huge narrowband gain
parked on the high modes -- and the +5 % machined shift moves those modes
off the inverted notches, so the replay blows the voltage up by orders of
magnitude and goes unstable.  Plant inversion has two classical antidotes,
both searched here:

  * F_WU 1000 / 1400 Hz (was 2000): the effort penalty starts rising below
    the first high mode, forbidding authority near modes 3-6 by
    construction rather than by luck;
  * NOISE 1e-3 / 1e-2: the sensor-noise weight is the standard
    regulariser -- inversion stops paying when the loop must also reject
    noise through the same channel.

The collocated PD inner loop keeps its broadband authority regardless (it
is passive on that pair), so the combined law is the natural beneficiary.
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

    print('=== mu grid, F_WU x NOISE axes ===', flush=True)
    for FWU in [1000.0, 1400.0]:
        for NZ in [1e-3, 1e-2]:
            for C in [30e-6, 45e-6]:
                for V in [150.0, 170.0]:
                    tag = 'F%.0f_N%.0e_C%.0f_V%.0f' % (FWU, NZ, C * 1e6, V)
                    payload, status = run_cell('mu', dict(
                        C=C, V=V, n_iter=4, FWU=FWU, NOISE=NZ))
                    if status != 'ok':
                        print('  %s : %s' % (tag, status), flush=True)
                        continue
                    process(payload, P, Pm, Pmid, '_res_mu', tag)

    print('\n=== combined grid, F_WU x NOISE axes ===', flush=True)
    for FWU in [1000.0, 1400.0]:
        for NZ in [1e-3, 1e-2]:
            for g in [0.3, 0.5, 0.7]:
                for C in [30e-6, 45e-6]:
                    tag = 'F%.0f_N%.0e_g%02.0f_C%.0f_V150' % (FWU, NZ,
                                                              g * 10, C * 1e6)
                    payload, status = run_cell('cb', dict(
                        C=C, V=150.0, n_iter=4, g=g, FWU=FWU, NOISE=NZ,
                        kp=par['kp'], kd=par['kd'], wc=par['wc']))
                    if status != 'ok':
                        print('  %s : %s' % (tag, status), flush=True)
                        continue
                    process(payload, P, Pm, Pmid, '_res_cb', tag,
                            extra=dict(inner_gain=g))
    print('\nsupplementary grid done', flush=True)


if __name__ == '__main__':
    main()
