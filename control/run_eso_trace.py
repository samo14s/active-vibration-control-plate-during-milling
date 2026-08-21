"""
run_eso_trace.py — Le mecanisme de l'ADRC, mesure et non postule
=================================================================
La these de l'ADRC est que l'observateur d'etat etendu estime en ligne la
"perturbation totale" f = y'' - b0 u — ici : la dynamique modale non modelisee,
le couplage regeneratif retarde alpha4 D^T D y(t - tau) et la variation de D(x)
le long de la passe — et que la loi u = (u0 - z3)/b0 l'annule.

Ce script la VERIFIE sur la simulation de coupe : on enregistre les etats de
l'observateur pendant une passe complete, on reconstruit la perturbation vraie
par derivation numerique de la mesure, et on compare.

    PROTOCOL=A|B python run_eso_trace.py
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
from plate_model import build_plate
from simulate import MillingSimulation
from sim_controller import LTIController

OUT = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures', 'comparison')
os.makedirs(FIG, exist_ok=True)


class LoggingController(LTIController):
    """Meme correcteur, mais qui garde la trace de ses etats."""

    def __init__(self, ss, dt, nstep):
        super().__init__(ss, dt)
        self.log = np.zeros((nstep, self.x.size))
        self.i = 0

    def __call__(self, y=0.0, yd=0.0, t=0.0, k=0):
        if self.i < self.log.shape[0]:
            self.log[self.i] = self.x
            self.i += 1
        return super().__call__(y=y, yd=yd, t=t, k=k)


def main():
    d = np.load(os.path.join(OUT, f'pso_{C.PROTOCOL}.npz'), allow_pickle=True)
    par = dict(zip([str(k) for k in d['adrc__keys']], d['adrc__values']))
    # LE CORRECTEUR STOCKE EST EQUILIBRE, DONC ILLISIBLE PAR INDICE. Ce script
    # lit x[0] et x[2] comme z1 et z3 : c'est licite dans les coordonnees
    # NATURELLES de l'ESO, ou B = [3wo, 3wo^2, wo^3], et faux dans celles de
    # results/pso_*.npz, que `series` equilibre. Sur l'optimum stocke la
    # similitude vaut t = (2^-1, 2^15, 2^25), si bien que x[2] valait z3/2^25 :
    # ce trace annoncait 100.000 % d'erreur d'estimation (contre 75.144 %),
    # un suivi de mesure a 103.338 % (contre 12.437 %) et |z3|/b0 a 3.7e-7 V
    # de moyenne (contre 12.48 V). La correlation, invariante d'echelle, ne
    # bougeait pas : rien dans la sortie ne pouvait signaler la panne.
    #
    # On RECONSTRUIT donc le meme correcteur sans l'equilibrage final. La
    # fonction de transfert est identique — c'est la meme loi — seules les
    # coordonnees changent, et ce sont elles que ce script inspecte.
    from pso import Design
    from plate_model import build_plate as _bp
    _plate = _bp(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    from plate_model import plant_vectors
    _sl = plant_vectors(_plate, C.N_MODES_DESIGN)[4]
    _D = Design('adrc', _plate, _sl,
                sign_variant=float(d['adrc__sign_variant']))
    ss = _D.build(np.atleast_1d(d['adrc__x']), balance=False)
    _ss_bal = _D.build(np.atleast_1d(d['adrc__x']))
    from fopid import ss_frf
    _w = 2 * np.pi * np.logspace(1, 4.3, 200)
    _e = np.max(np.abs(ss_frf(ss, _w) - ss_frf(_ss_bal, _w))) / \
        np.max(np.abs(ss_frf(_ss_bal, _w)))
    print(f"  realisation NON equilibree pour la lecture des etats ;"
          f" ecart de reponse frequentielle au correcteur stocke : {_e:.2e}")
    # b0 SIGNE. Le correcteur construit par pso.Design applique
    #     b0_effectif = par['b0'] * sign_variant
    # (control/pso.py). Comparer z3 a y'' - par['b0'] u au lieu de
    # y'' - b0_effectif u inverse la reference et rend la mesure d'erreur
    # d'estimation sans valeur : c'est le defaut qui avait produit le chiffre
    # de 147 % annonce dans la premiere version de ce trace.
    var = float(d[f'adrc__sign_variant'])
    b0 = float(par['b0']) * var
    wo = float(par['wo'])
    print(f"  ADRC-FOPID (protocole {C.PROTOCOL}) : b0 = {b0:.4g}"
          f" (= {float(par['b0']):.4g} x sign_variant {var:+.0f}),"
          f" w_o = {wo:.4g} rad/s ({wo / 2 / np.pi:.0f} Hz)")

    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    ap = 0.6e-3
    sim = MillingSimulation(plate, C.RPM_DESIGN, ap, n_modes=C.N_MODES,
                            n_sub=C.N_SUB, sign=C.SIGN_SIM, v_max=C.V_MAX)
    nstep = int(round(sim.dt and (plate.lp / (C.FZ * 3 * C.RPM_DESIGN / 60.0))
                      / sim.dt)) + 4
    ctrl = LoggingController(ss, sim.dt, nstep)
    r = sim.run(controller=ctrl, T=None)

    n = len(r['t'])
    z = ctrl.log[:n]                       # [z1, z2, z3, fractionnaires...]
    y = r['y_obs']
    u = r['u']
    dt = float(r['dt'])
    # acceleration mesuree (difference centree) et perturbation totale vraie
    ydd = np.zeros(n)
    ydd[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dt**2
    ydd[0], ydd[-1] = ydd[1], ydd[-2]
    f_true = ydd - b0 * u
    f_hat = z[:, 2]

    # fenetre d'analyse : on ecarte le transitoire initial
    i0 = int(0.15 * n)
    err = f_hat[i0:] - f_true[i0:]
    rel = np.linalg.norm(err) / np.linalg.norm(f_true[i0:])
    corr = float(np.corrcoef(f_hat[i0:], f_true[i0:])[0, 1])
    obs_err = z[i0:, 0] - y[i0:]
    rel_y = np.linalg.norm(obs_err) / np.linalg.norm(y[i0:])
    print(f"  passe a a_p = {ap * 1e3:.2f} mm, {n} pas, dt = {dt * 1e6:.2f} us")
    print(f"  estimation de la perturbation totale :"
          f" erreur relative {rel * 100:.2f} %, correlation {corr:.4f}")
    print(f"  suivi de la mesure par z1 : erreur relative"
          f" {rel_y * 100:.3f} %")
    print(f"  |z3|/b0 (tension equivalente de compensation) :"
          f" moyenne {np.mean(np.abs(f_hat[i0:] / b0)):.2f} V,"
          f" max {np.max(np.abs(f_hat[i0:] / b0)):.2f} V")
    print(f"  tension totale : moyenne {np.mean(np.abs(u[i0:])):.2f} V,"
          f" max {np.max(np.abs(u)):.2f} V")

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    sl = slice(i0, min(n, i0 + int(0.02 / dt)))          # 20 ms de detail
    ax[0, 0].plot(r['t'], f_true, color='0.45', lw=.8, label='true $y$" $-$ '
                  '$b_0 u$')
    ax[0, 0].plot(r['t'], f_hat, color='#16a085', lw=.8, alpha=.85,
                  label='ESO estimate $z_3$')
    ax[0, 0].set_title('(a) total disturbance over the whole pass',
                       fontsize=10)
    ax[0, 0].set_xlabel('Time (s)')
    ax[0, 0].set_ylabel('$m/s^2$')
    ax[0, 1].plot(r['t'][sl], f_true[sl], color='0.45', lw=1.1,
                  label='true')
    ax[0, 1].plot(r['t'][sl], f_hat[sl], color='#16a085', lw=1.1,
                  label='$z_3$')
    ax[0, 1].set_title(f'(b) 20 ms detail - relative error {rel * 100:.1f} %,'
                       f' correlation {corr:.3f}', fontsize=10)
    ax[0, 1].set_xlabel('Time (s)')
    ax[0, 1].set_ylabel('$m/s^2$')
    ax[1, 0].plot(r['t'][sl], y[sl] * 1e6, color='0.45', lw=1.1,
                  label='measured $y$')
    ax[1, 0].plot(r['t'][sl], z[sl, 0] * 1e6, color='#16a085', lw=1.1,
                  ls='--', label='$z_1$')
    ax[1, 0].set_title(f'(c) observer tracking - relative error'
                       f' {rel_y * 100:.2f} %', fontsize=10)
    ax[1, 0].set_xlabel('Time (s)')
    ax[1, 0].set_ylabel('Displacement ($\\mu$m)')
    ax[1, 1].plot(r['t'], u, color='#1a3f8f', lw=.7, label='total $u$')
    ax[1, 1].plot(r['t'], -f_hat / b0, color='#16a085', lw=.7, alpha=.8,
                  label='cancellation term $-z_3/b_0$')
    ax[1, 1].set_title('(d) how much of the voltage is disturbance '
                       'cancellation', fontsize=10)
    ax[1, 1].set_xlabel('Time (s)')
    ax[1, 1].set_ylabel('Voltage (V)')
    for a in ax.ravel():
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.suptitle('ADRC-FOPID mechanism measured in the milling simulation '
                 f'({C.RPM_DESIGN} rpm, $a_p$ = {ap * 1e3:.2f} mm, '
                 f'protocol {C.PROTOCOL})', fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_eso_{C.PROTOCOL}.png', dpi=140)
    plt.close(fig)
    print(f'  -> {FIG}/fig_eso_{C.PROTOCOL}.png')


if __name__ == '__main__':
    main()
