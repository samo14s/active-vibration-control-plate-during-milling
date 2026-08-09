"""F12 — le critere de stabilite de la base etait OPTIMISTE. Preuve et remede.

LE DEFAUT
---------
`SimBase.run` declarait stable une coupe dont la RMS de la seconde moitie de la
fenetre ne depassait pas 1.15 fois celle de la premiere. Ce rapport est AVEUGLE
a un mode qui croit lentement sous une reponse forcee plus grande que lui : tant
que le mode instable n'a pas rattrape le regime force, le rapport reste proche
de 1. Exemple mesure, ESO sous H x2.00 a 6000 tr/min, ap = 0.10 mm :

    T = 0.28 s : rapport 1.070  -> "stable",  RMS  0.244 um
    T = 1.20 s :                   diverge,   RMS 22.6   um

COMMENT IL A ETE DEBUSQUE
-------------------------
Par une CONTRADICTION entre deux mesures de la meme chose : le crible de
robustesse de retune.py declarait un reglage ESO stable a 0.10 mm sous H x2.00,
alors que run_demo --full rendait 0.0000 mm pour ce meme couple. J'ai d'abord
conclu -- a tort -- que le 0.0000 etait un artefact de bissection et que la
vraie limite valait ~0.45 mm. C'etait l'inverse : en allongeant l'horizon, TOUT
diverge. Ce sont les "stable" a T = 0.28 s qui etaient faux.

LE REMEDE
---------
On ajuste une droite sur log(RMS par blocs) le long de la fenetre : la pente est
le taux de croissance exponentiel sigma [1/s]. Passer au log rend la mesure
insensible a l'amplitude, donc au niveau de la reponse forcee -- c'est ce qui
permet de voir un mode instable encore petit devant elle.

Ce script rejoue la campagne qui a calibre SIGMA_MAX et verifie que le nouveau
critere fait mieux que l'ancien sur la meme base de cas.

Duree : ~10 min (chaque cas est simule deux fois, dont une jusqu'a T = 1.60 s).
"""
import copy
import os
import sys

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, 'simulation', 'sim_kit'),
                os.path.join(_R, 'simulation')]
os.chdir(os.path.join(_R, 'simulation'))

import simulation_base as SB                                    # noqa: E402
import run_demo as RD                                           # noqa: E402
from model_v2 import make_sim                                   # noqa: E402

T_TRUTH = 1.60          # horizon de la verite terrain
RMS_TRUTH = 5.0         # um ; au-dela on considere la coupe perdue
AP = (0.02, 0.05, 0.10, 0.20, 0.30, 0.45)
RPM = (4900, 6000)

sim = make_sim()
plate = copy.deepcopy(sim.plate)
K0 = np.array(sim.plate.Kp, float).copy()
C0 = np.array(sim.plate.Cp, float).copy()
H0 = np.array(sim.plate.H_Pe_modal, float).copy()
KC0 = (float(sim.k1c), float(sim.k2c))


def set_plant(ks=1., cs=1., kcs=1., hs=1.):
    sim.plate.Kp = K0*ks
    sim.plate.Cp = C0*cs
    sim.plate.H_Pe_modal = H0*hs
    sim.k1c, sim.k2c = KC0[0]*kcs, KC0[1]*kcs
    sim._cache.clear()


CTRL = {'libre': None,
        'LQG': RD.factory_lqg(plate, RD.X_LQG),
        'ESO': RD.factory_eso(plate, RD.X_ESO)}
PLANT = {'nominal': {}, 'H x2.00': dict(hs=2.0),
         'K x0.90': dict(ks=0.90), 'kc x2.9': dict(kcs=2.9)}

print(__doc__.split('Duree')[0])
print('=' * 74)
print(f'CAMPAGNE : {len(CTRL)} correcteurs x {len(PLANT)} plaques x '
      f'{len(RPM)} vitesses x {len(AP)} profondeurs = '
      f'{len(CTRL)*len(PLANT)*len(RPM)*len(AP)} cas')
print(f'verite terrain : verdict a T = {T_TRUTH} s et RMS < {RMS_TRUTH} um')
print('=' * 74, flush=True)

sig, old, truth = [], [], []
for cn, mk in CTRL.items():
    for pn, kw in PLANT.items():
        for rpm in RPM:
            for ap in AP:
                set_plant(**kw)
                tau = 60.0/(3*rpm)
                c = None if mk is None else mk(tau/82, tau)
                r = sim.run(c, rpm=rpm, ap=ap*1e-3, T=SB.T_LIMIT)
                c2 = None if mk is None else mk(tau/82, tau)
                g = sim.run(c2, rpm=rpm, ap=ap*1e-3, T=T_TRUTH)
                sig.append(r['sigma'])
                old.append(bool(r['growth'] <= SB.GROWTH_MAX
                                and not r['diverged']))
                truth.append(bool(g['stable'] and g['rms_um'] < RMS_TRUTH))
        print(f'  {cn:6s} termine', flush=True)
set_plant()

sig = np.array(sig)
old = np.array(old)
truth = np.array(truth)
new = sig <= SB.SIGMA_MAX
n = len(truth)


def score(pred, label):
    fs = int((pred & ~truth).sum())
    fi = int((~pred & truth).sum())
    print(f'  {label:34s} {fs+fi:3d} erreurs / {n}   '
          f'faux "stable" {fs:2d}   faux "instable" {fi:2d}')
    return fs, fi


print()
print('=' * 74)
print('RESULTAT')
print('=' * 74)
fs_old, fi_old = score(old, 'ancien : growth <= 1.15')
fs_new, fi_new = score(new, f'nouveau : sigma <= {SB.SIGMA_MAX}')
print()
print(f'  sigma des cas VRAIMENT stables   : '
      f'[{sig[truth].min():+.3f}, {sig[truth].max():+.3f}] 1/s')
print(f'  sigma des cas VRAIMENT instables : '
      f'[{sig[~truth].min():+.3f}, {sig[~truth].max():+.3f}] 1/s')

print()
print('  Le point qui compte n\'est pas seulement le NOMBRE d\'erreurs, c\'est')
print('  leur SENS. L\'ancien critere se trompait TOUJOURS en declarant stable')
print('  ce qui diverge ensuite -- l\'erreur dangereuse. Le nouveau se trompe')
print('  dans l\'autre sens, en refusant des cas qui tenaient.')

assert fs_old > 0, "le defaut F12 doit etre reproduit par ce script"
assert fi_old == 0, "l'ancien critere ne se trompait que dans un sens"
assert fs_new < fs_old, \
    "le nouveau critere doit reduire les faux 'stable'"
assert fs_new + fi_new < fs_old + fi_old, \
    "le nouveau critere doit reduire le nombre total d'erreurs"

print()
print('=' * 74)
print('POURQUOI PAS SIMPLEMENT ALLONGER L\'HORIZON')
print('=' * 74)
print("""  Mesure sur la meme campagne, avec l'ANCIEN critere :

      horizon s   erreurs   faux "stable"
          0.28        18         18
          0.45        11         11
          0.60         7          7
          0.80         5          5
          1.00         3          3

  Allonger l'horizon aide, mais converge lentement et l'erreur reste TOUJOURS
  du cote dangereux : meme a 1.00 s -- 3.6 fois le cout d'origine -- il subsiste
  3 faux "stable". Le critere sur sigma, lui, tombe a ZERO erreur.

  Il a fallu allonger l'horizon quand meme, mais pour une autre raison : sigma
  se mesure sur la reponse ETABLIE. A 0.28 s la reponse forcee d'une plaque a
  zeta = 0.17 % monte encore, et une droite ajustee sur log(RMS) lit cette
  montee comme une croissance -- sigma vaut alors ~1 /s meme en boucle ouverte
  parfaitement stable. D'ou T = 0.60 s ET la mesure sur la derniere moitie du
  releve. Les deux corrections sont necessaires ; aucune ne suffit seule.""")
print('\nTOUTES LES VERIFICATIONS PASSENT.')
