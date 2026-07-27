"""
Re-certify every two-plant-feasible candidate with the MIXED-mu bound.

The complex-D certificate is structurally vacuous on this plant: delta_3..6
are REAL frequency shifts of 0.5 %-damped modes, and a complex delta of
p = 4.9 % on a mode with 2*zeta ~ 1 % is a phantom damping perturbation no
voltage-limited controller can cover -- the complex-D peak lands at 10-30
for designs whose physical replay is clean.  The old report flagged exactly
this conservatism in a footnote; the G-scale (Fan-Tits-Doyle) bound closes
it while remaining a true upper bound (it reduces to the complex-D bound at
G = 0 and is seeded there, so it can only tighten).

For each feasible candidate the certificate is computed on the FULL
structure (six real modal deltas + real actuator gain + complex 2x2
performance block) with the same weights the cell was synthesised with,
parsed back from the file tag.  A combined candidate is certified as the
TOTAL parallel law on the unfolded plant, which closes the same loop as
outer-on-folded-plant and needs no inner bookkeeping.  The stored complex-D
curve is kept and the reported mixed curve is the pointwise minimum of the
two (both are valid upper bounds).

Updates each npz in place: adds peak_mixed, mu_mixed.
"""
import glob
import re
import numpy as np
import control as ct
import avc_mu
from avc_plant import Plant, Ctrl, VLIM, ZMIN


def parse_tag(fn):
    """Weights out of the file name; defaults are the legacy values."""
    w = dict(C=None, V=None, FWU=2000.0, NOISE=1e-3, WUFLAT=False)
    if re.search(r'_W\d*_', fn):        # tune3 tags are W_..., tune4 W4_...
        w['WUFLAT'] = True
    m = re.search(r'_F(\d+)_', fn)
    if m:
        w['FWU'] = float(m.group(1))
    m = re.search(r'_N(\de-\d+)_', fn)
    if m:
        w['NOISE'] = float(m.group(1))
    m = re.search(r'_C(\d+)_', fn)
    w['C'] = float(m.group(1)) * 1e-6
    m = re.search(r'_V(\d+)', fn)
    w['V'] = float(m.group(1))
    return w


def feasible(d):
    return (bool(d['stable']) and bool(d['m_stable']) and bool(d['mid_stable'])
            and float(d['volt']) <= VLIM and float(d['m_volt']) <= VLIM
            and float(d['zeta_min']) >= ZMIN and float(d['m_zeta_min']) >= ZMIN)


def main():
    P = Plant()
    cT = P.tools['corner (100, 80)']
    cache = {}
    for fn in sorted(glob.glob('_res_mu_*.npz') + glob.glob('_res_cb_*.npz')):
        d = dict(np.load(fn))
        if not feasible(d):
            continue
        w = parse_tag(fn)
        key = (w['C'], w['V'], w['FWU'], w['NOISE'], w['WUFLAT'])
        if key not in cache:
            avc_mu.C_TARGET, avc_mu.V_TARGET = w['C'], w['V']
            avc_mu.F_WU, avc_mu.NOISE = w['FWU'], w['NOISE']
            avc_mu.WU_FLAT = w['WUFLAT']
            cache[key] = avc_mu.gen_plant(P, cT, unc=None)
        P_full, sc = cache[key]
        K = Ctrl(d['Ak'], d['Bk'].ravel(), d['Ck'].ravel(), float(d['Dk']))
        K_ss = avc_mu._scale_ctrl(K, sc)
        om = d['om']
        mu_mx = avc_mu.mu_curve_mixed(ct.ss(K_ss), ct.ss(P_full),
                                      om / sc['w_ref'])
        both = np.minimum(mu_mx, d['mu']) if np.all(np.isfinite(d['mu'])) \
            else mu_mx
        d['mu_mixed'] = both
        d['peak_mixed'] = float(both.max())
        np.savez(fn, **d)
        print('%-42s complexD %7.3f   mixed %6.3f   a_lim x%.2f  volt %.0f'
              % (fn, d['peak'], d['peak_mixed'], d['a_lim'], d['volt']),
              flush=True)
    print('recertification done', flush=True)


if __name__ == '__main__':
    main()
