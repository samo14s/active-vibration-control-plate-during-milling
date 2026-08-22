"""
pareto_qualite.py — la profondeur REELLEMENT atteignable : stabilite ET qualite
===============================================================================
Ce depot a toujours classe les correcteurs sur UN critere, a_p,lim — la
profondeur limite de broutement. C'est aussi le critere de van Dijk et al.
(2014), de Dong et al. (2023) et de Du et al. (2022, 2024). Or une piece dont
la surface est hors tolerance est rebutee, qu'elle ait broute ou non.

TROIS plafonds, pas un :

  1. a_p,lim   — stabilite (Floquet, deja au depot, lu dans results/)
  2. a_p,qual  — |SLE| <= tolerance de finition
  3. a_p,val   — validite du modele : crete-a-crete de la reponse nominale
                 <= charge par dent. Au-dela, la dent decroche une partie de
                 la periode, le retard devient dependant de l'etat et alpha4(t)
                 n'est plus le bon coefficient (Niu et al., IJMS 2021).

et la profondeur atteignable est le MINIMUM des trois.

Le plafond 3 depend du correcteur, ce qui n'est pas evident : un correcteur qui
calme la resonance entre les dents REPOUSSE le domaine de validite du modele
avec lequel on le juge.

POURQUOI PAR BISSECTION ET NON PAR LA PENTE. Une premiere version lisait |SLE|
en un point et extrapolait lineairement. La structure s'y pretait — sur la
solution nominale la raideur regenerative s'annule, donc A + A_tau ne depend
plus de alpha4 : le systeme est INVARIANT et seul le corsage b = alpha4 f_z D
est periodique, d'ou une SLE quasi lineaire en a_p. Mais l'extrapolation casse
exactement la ou le resultat est le plus interessant : quand un correcteur
ANNULE presque la SLE, la pente residuelle est du bruit et l'extrapolation
rendait a_p,qual = 11.9 mm pour le LQG. On bissecte donc, et on plafonne
franchement la recherche au lieu d'extrapoler hors du domaine.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'paper_model'), HERE]

import config as C                                                  # noqa
from plate_model import build_plate                                 # noqa
import stored_ctrl                                                  # noqa
from surface_error import periodic_response                         # noqa

TOL_SLE = 10e-6          # tolerance de finition
AP_MAX = 1.0e-3          # plafond franc de la recherche
M_GRID = 192


def _bissecte(f, lo, hi, n=18):
    """Plus grand a_p ou f(a_p) est vrai. f(lo) suppose vrai."""
    if f(hi):
        return hi, True                       # jamais atteint sous le plafond
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) else (lo, mid)
    return 0.5 * (lo + hi), False


def plafonds(plate, rpm, x_pos, ctrl=None, pd=None, tol=TOL_SLE,
             ap_max=AP_MAX, m=M_GRID):
    """(a_p,qual, a_p,val) et les drapeaux « jamais atteint »."""
    def rep(ap):
        return periodic_response(plate, rpm, ap, x_pos, ctrl=ctrl, pd=pd,
                                 m=m, coeff_scale=C.SIGN_SIM)
    fz = rep(0.05e-3)['fz']
    q, q_libre = _bissecte(lambda a: abs(rep(a)['sle']) <= tol, 1e-6, ap_max)
    v, v_libre = _bissecte(lambda a: rep(a)['pv'] <= fz, 1e-6, ap_max)
    return dict(qual=q, qual_libre=q_libre, val=v, val_libre=v_libre)


def main():
    rpm = float(os.environ.get('RPM', 5200))
    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    x = 0.5 * plate.lp
    st = stored_ctrl.discover()
    d = np.load(os.path.join(ROOT, 'results', 'compare_B.npz'),
                allow_pickle=True)
    i = int(np.where(d['lobes__rpm'] == rpm)[0][0])
    lim = {k.split('__', 1)[1]: float(d[k][i]) for k in d
           if k.startswith('lobes__') and not k.endswith('rpm')}

    print(f'  {rpm:.0f} tr/min, x = lp/2, |SLE| <= {TOL_SLE*1e6:.0f} um, '
          f'plafond de recherche {AP_MAX*1e3:.1f} mm')
    print(f'\n{"structure":>10s} {"ap,lim":>8s} {"ap,qual":>9s} {"ap,val":>8s} '
          f'{"atteint":>8s} {"liee par":>10s} {"x ouvert":>9s}')
    lignes = [('AUCUN', None, None, lim['boucle ouverte'])]
    lignes += [(k, st[k][0], st[k][1], lim.get(k, np.nan))
               for k in sorted(st)]
    base = None
    out = []
    for nom, ss, pdg, L in lignes:
        p = plafonds(plate, rpm, x, ctrl=ss, pd=pdg)
        cand = [(L, 'stabilite'),
                (p['qual'], 'qualite' + ('*' if p['qual_libre'] else '')),
                (p['val'], 'validite' + ('*' if p['val_libre'] else ''))]
        a, lie = min(cand, key=lambda z: z[0])
        if base is None:
            base = a
        out.append((nom, L, p, a, lie))
        print(f'{nom:>10s} {L*1e3:8.3f} {p["qual"]*1e3:9.3f}'
              f'{"*" if p["qual_libre"] else " "}{p["val"]*1e3:8.3f}'
              f'{"*" if p["val_libre"] else " "}{a*1e3:8.3f} {lie:>10s} '
              f'{a/base:9.2f}')
    print('\n  * = plafond jamais atteint sous 1.0 mm ; la valeur est le '
          'plafond de recherche, pas une mesure.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
