"""
cross_check_delay.py — Floquet et le simulateur decrivent-ils la MEME loi ?
==========================================================================
Le controle a retard actif de l'Eq. (30) existe DEUX fois dans le depot, en
deux implementations qui n'ont aucun code commun :

  - Floquet met les gains dans A_tau et calcule la monodromie sur une periode
    de dent (closed_loop.build_matrices) ;
  - le simulateur les lit dans un historique de n_sub pas
    (sim_controller.DelayedPDController).

Le tableau publie les DEUX profondeurs limites. Un retard decale d'un pas, ou
un signe inverse sur l'un des gains, donnerait deux colonnes coherentes
chacune avec elle-meme et fausses l'une par rapport a l'autre — le genre
d'ecart qu'aucune des deux ne peut signaler seule.

Le point de mesure compte autant que la mesure. La premiere version de ce
script comparait cinq profondeurs choisies a l'avance, au poste MEDIAN : les
cinq etaient stables des deux cotes, ce qui ne compare rien — les deux moteurs
sont d'accord la ou tout est d'accord. Le seul endroit qui discrimine est le
voisinage immediat de la limite. On la bissecte donc par Floquet, AU PIRE
POSTE, puis on demande au simulateur ce qu'il en pense a -20 %, -5 %, +5 % et
+20 % : ce qui est teste alors, c'est le RENVERSEMENT du verdict, pas sa
stabilite.

    PROTOCOL=B CALIB=measured python control/cross_check_delay.py
"""
import os, sys, time
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(R, 'paper_model'), HERE]
import config as C
from plate_model import build_plate
from closed_loop import limit as cl_limit, is_stable
from simulate import MillingSimulation
from sim_controller import LTIController, DelayedPDController
plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
d = np.load(os.path.join(R, 'results', f'pso_{C.PROTOCOL}.npz'),
            allow_pickle=True)
ss = (d['musyn__A'], d['musyn__B'], d['musyn__C'], d['musyn__D'])
PD = (-7.5e4, -40.0)
T = 3.0
kw = dict(n_modes=C.N_MODES, m=200, coeff_mode='time',
          coeff_scale=C.SIGN_SIM, ae=C.AE)

for tag, pd in (('sans retard', None), ('avec retard', PD)):
    lim = {fr: cl_limit(plate, C.RPM_DESIGN, fr*plate.lp, ctrl=ss, pd=pd,
                        hi=6e-3, **kw) for fr in C.POSITIONS_DESIGN}
    fr = min(lim, key=lambda f: lim[f])
    L = lim[fr]
    print(f'\n  {tag} : pire poste {fr} l_P, limite Floquet '
          f'{L*1e3:.4f} mm   (T = {T} s)')
    print(f'{"a_p [mm]":>9s} {"rapport":>8s} {"Floquet":>10s} {"rho":>8s}'
          f' {"simulateur":>12s}')
    for r in (0.80, 0.95, 1.05, 1.20):
        ap = r * L
        st, rho = is_stable(plate, C.RPM_DESIGN, ap, fr*plate.lp, ctrl=ss,
                            pd=pd, **kw)
        sim = MillingSimulation(plate, C.RPM_DESIGN, ap, n_modes=C.N_MODES,
                                n_sub=C.N_SUB, sign=C.SIGN_SIM, v_max=C.V_MAX)
        c = (LTIController(ss, sim.dt) if pd is None
             else DelayedPDController(ss, pd, sim.n_sub, sim.dt))
        out = sim.run(controller=c, T=T, moving=False, x0=fr*plate.lp)
        s = ('diverge a %.2f s' % out['t_div'] if out['diverged'] else 'tient')
        print(f'{ap*1e3:9.4f} {r:8.2f} {"stable" if st else "INSTAB":>10s}'
              f' {rho:8.4f} {s:>12s}', flush=True)
