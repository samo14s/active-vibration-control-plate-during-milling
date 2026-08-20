"""
18_sign_convention.py — Quel signe de alpha3/alpha4 le papier utilise-t-il ?
============================================================================
Le papier se contredit : en partant de ses Eqs. (1), (2), (5) et (10) — avec
f_y = -F_y et y_r = -y_P(t) + y_P(t - tau) — on obtient

    M q" + C q' + (K - a4 D^T D) q(t) + a4 D^T D q(t - tau) = -f_tau a3 D^T

alors que ses Eqs. (12)-(13) donnent les signes INVERSES (a3 -> -a3,
a4 -> -a4). Les deux couches de ce depot avaient herite de conventions
OPPOSEES :

  * simulation/  (elements finis)  : FORCE_SIGN = +1 -> Eq. (13) telle que
    publiee (choix documente dans VERIFICATION.md, defaut F10) ;
  * control/     (Chebyshev-Ritz)  : SIGN_SIM = -1 -> convention derivee.

Le choix n'est pas cosmetique : il echange les creux et les bosses des lobes,
donc il change d'un facteur 10 la limite de coupe A UNE VITESSE DONNEE, et
c'est cette limite que l'optimiseur PSO maximise. Ce script tranche sur le
modele de Chebyshev en confrontant les DEUX signes aux trois preuves
independantes que le papier fournit :

  P1  Fig. 13(b) : la limite sans commande est "relativement plus grande"
      autour de 3600 tr/min et 5400 tr/min ;
  P2  Fig. 14(a) : la condition S (4900 tr/min, a_p = 0.30 mm) DIVERGE sans
      commande ;
  P3  Fig. 18 (experience) : la limite sans commande est sous 0.1 mm a toutes
      les vitesses testees SAUF 5500 tr/min.

Sortie : figures/verification/18_sign_convention.png + tableau chiffre.
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'),
                os.path.join(HERE, '..', 'control')]

from plate_model import build_plate                       # noqa: E402
from lti_floquet import limit                             # noqa: E402
from simulate import MillingSimulation                    # noqa: E402

FIG = os.path.join(HERE, '..', 'figures', 'verification')
os.makedirs(FIG, exist_ok=True)

F_MEASURED = [540.0, 1068.0, 2787.0, 3351.0, 4122.0]
F_THEORETICAL = [537.0, 1101.0, 2805.0, 3423.0, 4254.0]
POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
SPEEDS = np.arange(3000, 7001, 200)
RPM_EXP = (4300, 4900, 5500, 6100, 6700)      # vitesses de la Fig. 18


def lobes(plate, sign, speeds=SPEEDS, n_modes=5, m=120, hi=4.0e-3):
    """Limite minimale sur le bord superieur, pour un signe donne."""
    out = np.empty(len(speeds))
    for i, rpm in enumerate(speeds):
        out[i] = min(limit(plate, rpm, fr * plate.lp, ctrl=None,
                           n_modes=n_modes, m=m, coeff_mode='time',
                           coeff_scale=sign, hi=hi, tol=2e-5)
                     for fr in POSITIONS)
    return out


def local_maxima(speeds, vals):
    return [(int(speeds[i]), float(vals[i]))
            for i in range(1, len(vals) - 1)
            if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]]


def main():
    t0 = time.time()
    print("=" * 78)
    print(" CONVENTION DE SIGNE DE alpha3/alpha4 — Eq. (13) publiee"
          " contre signe derive")
    print("=" * 78)
    res = {}
    for cal, freqs in (('measured', F_MEASURED), ('theoretical',
                                                  F_THEORETICAL)):
        plate = build_plate('right', freqs=freqs)
        for tag, sign in (('Eq. (13)', +1.0), ('derived', -1.0)):
            v = lobes(plate, sign)
            res[(cal, tag)] = v
            print(f"  [{cal:11s}] {tag:9s} : moyenne"
                  f" {v.mean() * 1e3:.4f} mm, min {v.min() * 1e3:.4f} mm"
                  f"   ({time.time() - t0:.0f} s)", flush=True)

    print("\n  P1 — maxima locaux des lobes (le papier annonce ~3600 et"
          " ~5400 tr/min)")
    for k, v in res.items():
        mx = sorted(local_maxima(SPEEDS, v), key=lambda p: -p[1])[:3]
        print(f"    {k[0]:11s} {k[1]:9s} : " + ", ".join(
            f"{s} tr/min ({a * 1e3:.3f} mm)" for s, a in mx))

    print("\n  P2 — condition S : 4900 tr/min, a_p = 0.30 mm sans commande")
    p2 = {}
    for cal, freqs in (('measured', F_MEASURED), ('theoretical',
                                                  F_THEORETICAL)):
        plate = build_plate('right', freqs=freqs)
        for tag, sign in (('Eq. (13)', +1.0), ('derived', -1.0)):
            lim = min(limit(plate, 4900, fr * plate.lp, ctrl=None, n_modes=5,
                            m=200, coeff_mode='time', coeff_scale=sign,
                            hi=4.0e-3, tol=1e-5) for fr in POSITIONS)
            sim = MillingSimulation(plate, 4900, 0.3e-3, n_modes=5, n_sub=656,
                                    sign=sign, v_max=None)
            r = sim.run(controller=None, T=None)
            p2[(cal, tag)] = (lim, r['diverged'], r['t_div'],
                              float(np.abs(r['y_mill']).max()))
            print(f"    {cal:11s} {tag:9s} : a_p,lim ="
                  f" {lim * 1e3:.4f} mm ->"
                  f" {'DIVERGE' if r['diverged'] else 'stable'}"
                  f"   (max |y| = {np.abs(r['y_mill']).max() * 1e6:.1f} um"
                  f"{', t = %.3f s' % r['t_div'] if r['diverged'] else ''})",
                  flush=True)

    print("\n  P3 — vitesses de la Fig. 18 : limite < 0.1 mm partout SAUF"
          " 5500 tr/min")
    p3 = {}
    for cal, freqs in (('measured', F_MEASURED), ('theoretical',
                                                  F_THEORETICAL)):
        plate = build_plate('right', freqs=freqs)
        for tag, sign in (('Eq. (13)', +1.0), ('derived', -1.0)):
            vals = np.array([min(limit(plate, rpm, fr * plate.lp, ctrl=None,
                                       n_modes=5, m=200, coeff_mode='time',
                                       coeff_scale=sign, hi=4.0e-3, tol=1e-5)
                                 for fr in POSITIONS) for rpm in RPM_EXP])
            p3[(cal, tag)] = vals
            ok = (vals[[0, 1, 3, 4]] < 0.1e-3).all() and vals[2] > 0.1e-3
            print(f"    {cal:11s} {tag:9s} : " + "  ".join(
                f"{r}:{v * 1e3:.3f}" for r, v in zip(RPM_EXP, vals))
                + f"   -> motif du papier : {'OUI' if ok else 'non'}")

    # ----------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for i, cal in enumerate(('measured', 'theoretical')):
        for tag, c in (('Eq. (13)', '#1a3f8f'), ('derived', '#c0392b')):
            ax[i].semilogy(SPEEDS, res[(cal, tag)] * 1e3, '-o', ms=3, lw=1.5,
                           color=c, label=f'{tag} sign')
        ax[i].axhline(0.1, color='k', ls='--', lw=1, label='0.1 mm (Fig. 18)')
        ax[i].plot([4900], [0.3], 'k*', ms=13, zorder=6,
                   label='condition S (diverges in Fig. 14a)')
        for s in (3600, 5400):
            ax[i].axvline(s, color='0.6', ls=':', lw=1)
        ax[i].set_title(f'({chr(97 + i)}) {cal} calibration', fontsize=10)
        ax[i].set_xlabel('Spindle speed (rpm)')
        ax[i].set_ylabel('$a_{p,lim}$ (mm), worst position')
        ax[i].grid(alpha=.3, which='both')
        ax[i].legend(fontsize=7.5)
    lab = [f'{r}' for r in RPM_EXP]
    xs = np.arange(len(RPM_EXP))
    w = 0.38
    for j, (tag, c) in enumerate((('Eq. (13)', '#1a3f8f'),
                                  ('derived', '#c0392b'))):
        ax[2].bar(xs + (j - .5) * w, p3[('measured', tag)] * 1e3, w, color=c,
                  label=f'{tag} sign')
    ax[2].axhline(0.1, color='k', ls='--', lw=1, label='0.1 mm')
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(lab)
    ax[2].set_yscale('log')
    ax[2].set_xlabel('Spindle speed of Fig. 18 (rpm)')
    ax[2].set_ylabel('$a_{p,lim}$ (mm)')
    ax[2].set_title('(c) test P3: experiment reports > 0.1 mm only at 5500 rpm',
                    fontsize=10)
    ax[2].grid(alpha=.3, axis='y', which='both')
    ax[2].legend(fontsize=8)
    fig.suptitle('Which sign of the cutting coefficients does the paper use? '
                 'Three independent tests from the paper itself', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, '18_sign_convention.png'), dpi=140)
    plt.close(fig)
    print(f"\n  -> {FIG}/18_sign_convention.png   ({time.time() - t0:.0f} s)")


if __name__ == '__main__':
    main()
