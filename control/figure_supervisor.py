"""
figure_supervisor.py — le superviseur pris sur le fait
======================================================
Un tableau de limites dit CE QUE fait le superviseur ; il ne montre pas
COMMENT. Cette figure enregistre, sur la plaque derivee et a une profondeur
ou l'observateur a bandes fixes perd et ou le superviseur tient, les
grandeurs internes qui decident : les frequences estimees contre les vraies,
l'etat des verrous, le niveau de broutement et la valeur de alpha.

    PROTOCOL=B CALIB=measured python figure_supervisor.py
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
from plate_model import build_plate, plant_vectors
from simulate import MillingSimulation
from fdob import target_modes
from fdob_adaptive import AdaptiveFDOB

OUT = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures', 'comparison')
os.makedirs(FIG, exist_ok=True)


class Probe:
    """Enveloppe le correcteur et garde la trace de ses decisions."""

    def __init__(self, c):
        self.c = c
        self.f = []
        self.a = []
        self.l = []
        self.k = []

    def reset(self):
        self.c.reset()

    def __call__(self, **kw):
        u = self.c(**kw)
        self.f.append(self.c.f_hat.copy())
        self.a.append(self.c.alpha)
        self.l.append(self.c.level)
        self.k.append([e.locked for e in self.c.est])
        return u


def main():
    d = np.load(os.path.join(OUT, f'pso_{C.PROTOCOL}.npz'), allow_pickle=True)
    par = dict(zip([str(x) for x in d['fdob__keys']],
                   [float(v) for v in d['fdob__values']]))
    tg = [int(t) for t in d['fdob__targets']]
    var = float(d['fdob__sign_variant'])
    w0, z0, r0 = target_modes(build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL), tg)
    fr = [C.F_NOMINAL[0] * 1.17, C.F_NOMINAL[1] * 1.09] + list(C.F_NOMINAL[2:])
    plate = build_plate(C.PATCH_SIDE, freqs=fr)
    _, _, _, _, sl = plant_vectors(plate, C.N_MODES)
    ft = 3.0 * C.RPM_DESIGN / 60.0
    ap, x0 = 0.30e-3, 0.125

    print(f'  plaque derivee : modes a {fr[0]:.1f} et {fr[1]:.1f} Hz')
    print(f'  a_p = {ap * 1e3:.2f} mm, position {x0:.3f}'
          f'  (bandes fixes : limite 0.186 mm ; supervise : 0.394 mm)')

    runs = {}
    for tag, kw in (('bandes fixes', dict(adapt=False, supervise=False)),
                    ('supervise', dict(adapt=True, supervise=True))):
        sim = MillingSimulation(plate, C.RPM_DESIGN, ap, n_modes=C.N_MODES,
                                n_sub=C.N_SUB, sign=C.SIGN_SIM,
                                v_max=C.V_MAX)
        p = Probe(AdaptiveFDOB(par, w0, z0, r0, C.FDOB_WC, sim.dt, ft,
                               sl * var, C.OUST_WB, C.OUST_WH, C.OUST_N,
                               C.ROLLOFF_HZ, C.ROLLOFF_ORDER, **kw))
        r = sim.run(controller=p, T=0.5, moving=False, x0=x0 * plate.lp)
        runs[tag] = (r, p)
        print(f'    {tag:14s} : diverge = {bool(r["diverged"])},'
              f' |y|max = {np.abs(r["y_obs"]).max() * 1e6:.1f} um,'
              f' |u|max = {np.abs(r["u"]).max():.1f} V')

    fig, ax = plt.subplots(2, 2, figsize=(13, 7.4))

    a = ax[0, 0]
    for tag, col in (('bandes fixes', '#c0392b'), ('supervise', '#16a085')):
        r, _ = runs[tag]
        lab = ('fixed bands' if tag == 'bandes fixes' else 'supervised')
        if bool(r['diverged']):
            lab += f" — diverges at {float(r['t_div']):.3f} s"
        a.plot(r['t'], r['y_obs'] * 1e6, color=col, lw=.8, label=lab)
    a.set_yscale('symlog', linthresh=1.0)
    a.set_xlabel('Time (s)')
    a.set_ylabel('Displacement ($\\mu$m)')
    a.set_title(f'(a) drifted plate, $a_p$ = {ap * 1e3:.2f} mm, '
                f'tool at {x0 * 100:.1f} % of the pass', fontsize=10)
    a.grid(alpha=.3)
    a.legend(fontsize=8)

    a = ax[0, 1]
    r, p = runs['supervise']
    f = np.array(p.f)
    n = min(len(r['t']), len(f))
    for i, (c, tr) in enumerate(zip(('#1a3f8f', '#8e44ad'), fr[:2])):
        a.plot(r['t'][:n], f[:n, i], color=c, lw=1.3,
               label=f'$\\hat f_{i + 1}$')
        a.axhline(tr, color=c, ls='--', lw=1,
                  label=f'true drifted mode {i + 1} = {tr:.0f} Hz')
        a.axhline(w0[i] / 2 / np.pi, color=c, ls=':', lw=1,
                  label=f'nominal {w0[i] / 2 / np.pi:.0f} Hz')
    a.set_xlabel('Time (s)')
    a.set_ylabel('Frequency (Hz)')
    a.set_title('(b) what the estimators do — mode 2 never locks,\n'
                'because mode 2 is not the one chattering', fontsize=10)
    a.grid(alpha=.3)
    a.legend(fontsize=7, ncol=2)

    a = ax[1, 0]
    a.plot(r['t'][:n], np.array(p.l)[:n], color='#c8963e', lw=1.2,
           label='chatter level')
    a.axhline(0.25, color='0.4', ls=':', lw=1, label='gate threshold')
    a.set_xlabel('Time (s)')
    a.set_ylabel('level')
    a2 = a.twinx()
    a2.plot(r['t'][:n], np.array(p.a)[:n], color='#16a085', lw=1.6,
            label='$\\alpha$')
    a2.set_ylabel('$\\alpha$', color='#16a085')
    a.set_title('(c) the supervisor engages only when there is\n'
                'something to engage against', fontsize=10)
    a.grid(alpha=.3)
    a.legend(fontsize=8, loc='center right')
    a2.legend(fontsize=8, loc='lower right')

    a = ax[1, 1]
    pos = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
    fixe = [0.1477, 0.1863, 0.2391, 0.4430, 0.4430, 0.1688]
    sup = [0.1512, 0.3937, 0.4430, 0.4430, 0.4430, 0.1793]
    xb = np.arange(len(pos))
    a.bar(xb - 0.2, fixe, 0.4, color='#c0392b', label='fixed bands')
    a.bar(xb + 0.2, sup, 0.4, color='#16a085', label='supervised')
    for i, (u, v) in enumerate(zip(fixe, sup)):
        if v / u > 1.15:
            a.annotate(f'x{v / u:.2f}', (i, v), (0, 3), 'data',
                       'offset points', ha='center', fontsize=8)
    a.set_xticks(xb)
    a.set_xticklabels([f'{v * 100:.0f} %' for v in pos])
    a.set_xlabel('Tool position along the edge')
    a.set_ylabel('$a_{p,lim}$ (mm)')
    a.set_title('(d) the gain is in the middle of the pass;\n'
                'the worst-position metric hides it', fontsize=10)
    a.grid(alpha=.3, axis='y')
    a.legend(fontsize=8)

    fig.suptitle('The supervisor caught in the act — drifted plate '
                 '(+17 % / +9 % on modes 1-2)', fontsize=11.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_supervisor_{C.PROTOCOL}.png', dpi=140)
    plt.close(fig)
    print(f'  -> {FIG}/fig_supervisor_{C.PROTOCOL}.png')


if __name__ == '__main__':
    main()
