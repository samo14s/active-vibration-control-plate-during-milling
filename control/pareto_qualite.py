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

# ---------------------------------------------------------------------------
# POURQUOI DEUX MODES, ET NON LES CINQ DE N_MODES
# ---------------------------------------------------------------------------
# config fixe N_MODES = 5 comme modele d'evaluation. La question s'est posee de
# refaire tout le calcul de qualite sur cinq modes. Elle a ete TRANCHEE PAR LA
# MESURE, pas par l'argument : profondeur atteignable a 5200 tr/min, x = lp/2,
# treize configurations, cinq modes contre deux :
#
#   AUCUN -0.1%   adrc +0.3%   dvf +0.5%   fdob -0.3%   fopid -0.6%
#   hinf  -0.1%   lqg  -0.8%   mpc +/-0.5% musyn -0.3%  musyn_td -0.6%
#   nmpdob 0.0%   smc  -2.2%   vpa  -0.9%
#
# Douze cas sur treize a 1 % pres, le pire a 2.2 %. Les modes superieurs
# n'apportent rien ici — l'excitation de fraisage ne les atteint pas — et le
# modele a deux modes suffit. Ce n'est donc pas une approximation subie mais un
# resultat de robustesse : les conclusions de qualite ne dependent pas du
# nombre de modes, et l'objection « deux modes ne suffisent pas » est reglee
# d'avance, chiffres a l'appui.

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


# ---------------------------------------------------------------------------
# L'OBJECTIF CONJOINT, sous une forme utilisable par une optimisation
# ---------------------------------------------------------------------------
def profondeur_atteignable(plate, rpm, x_pos, ctrl=None, pd=None,
                           tol=TOL_SLE, ap_max=AP_MAX, m_sle=96, m_flo=40,
                           n_modes=2, n_bis=14):
    """Plus grande profondeur qui satisfait les TROIS conditions a la fois.

    On ne calcule PAS trois plafonds separement pour en prendre le minimum :
    on bissecte une seule fois sur un predicat conjoint

        stable(a_p)  ET  |SLE(a_p)| <= tol  ET  crete-a-crete(a_p) <= f_z

    C'est exact (aucune extrapolation) et cela coute une bissection, pas trois.
    L'ordre des tests est choisi pour sortir tot : les deux tests de qualite
    partagent UNE reponse nominale, devenue tres bon marche depuis que la
    structure invariante de A + A_tau est exploitee, et le test de stabilite —
    de loin le plus cher — n'est fait qu'en dernier.
    """
    from closed_loop import is_stable

    def ok(ap):
        r = periodic_response(plate, rpm, ap, x_pos, ctrl=ctrl, pd=pd,
                              n_modes=n_modes, m=m_sle,
                              coeff_scale=C.SIGN_SIM)
        if abs(r['sle']) > tol or r['pv'] > r['fz']:
            return False
        # is_stable rend un COUPLE (stable, rho), pas un booleen. Ecrire
        # « return is_stable(...) » rendait un tuple non vide, donc TOUJOURS
        # vrai : le test de stabilite n'etait jamais applique. Le defaut etait
        # masque par les correcteurs, chez qui la validite lie de toute facon ;
        # seule la boucle ouverte — le seul cas ou la stabilite lie — l'a
        # revele, en rendant 0.128 mm (le plafond de validite) alors que rho
        # y vaut 1.018.
        stable, _ = is_stable(plate, rpm, ap, x_pos, ctrl=ctrl, pd=pd,
                              n_modes=n_modes, m=m_flo,
                              coeff_scale=C.SIGN_SIM)
        return bool(stable)

    lo, hi = 1e-6, ap_max
    if ok(hi):
        return hi, True
    if not ok(lo):
        return 0.0, False
    for _ in range(n_bis):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    a = 0.5 * (lo + hi)
    # Garde-fou : on ne rend jamais une profondeur ou rho depasse 1. Le defaut
    # ci-dessus aurait pu passer inapercu dans une optimisation entiere.
    st, rho = is_stable(plate, rpm, lo, x_pos, ctrl=ctrl, pd=pd,
                        n_modes=n_modes, m=m_flo, coeff_scale=C.SIGN_SIM)
    if not st:
        raise AssertionError(f'profondeur rendue instable : rho = {rho:.6f}')
    return a, False
