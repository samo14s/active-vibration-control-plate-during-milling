"""
objective_joint.py — l'objectif CONJOINT : profondeur reellement atteignable
=============================================================================
`objective.evaluate` note un correcteur par a_p,lim seul. La mesure de
`pareto_qualite` a montre que ce critere a une correlation de rang NULLE avec
la profondeur reellement atteignable (Spearman -0.021, Kendall tau 0.000 sur
douze correcteurs). On remplace donc l'objectif, en gardant les cribles.

CE QUI CHANGE, ET CE QUI NE CHANGE PAS.

  * Cribles INCHANGES : stabilite nominale, marge de module Ms, effort V/N.
    L'incertitude et la contrainte d'actionneur restent des CONTRAINTES, pas
    des objectifs — c'est le perimetre convenu.
  * Objectif REMPLACE : J n'est plus a_p,lim mais min(a_p,lim, a_p,qual,
    a_p,val), toujours en mm, donc directement comparable aux notes deja
    publiees.

LES DEUX PLAFONDS DE QUALITE, ET POURQUOI ILS SONT BON MARCHE. Sur la solution
nominale, alpha4 disparait de A + A_tau : la matrice est INVARIANTE et seul le
corsage b(t) = alpha4(t) f_z D est periodique. Une reponse nominale coute donc
un seul expm — environ 0.08 s. Et comme alpha4 est proportionnel a a_p a 3 %
pres dans ce domaine (verifie : SLE = 1.2702, 3.2652, 6.5488 um a 0.02, 0.05 et
0.10 mm, contre 1.25, 3.26 et 6.51 attendus d'une stricte proportionnalite),
SLE et crete-a-crete se mettent a l'echelle depuis UNE reponse de reference au
lieu d'etre bissectees.

    a_p,qual = a_ref . tol / |SLE(a_ref)|
    a_p,val  = a_ref . f_z / pv(a_ref)

Quand un correcteur ANNULE presque la SLE, a_p,qual part a l'infini : c'est le
resultat CORRECT — la qualite ne lie plus, et c'est la validite qui prend le
relais. La crete-a-crete, elle, ne s'annule jamais, donc a_p,val reste toujours
defini. C'est pourquoi la mise a l'echelle est sure ici alors qu'elle etait
trompeuse dans le TABLEAU de `pareto_qualite` : la-bas on RAPPORTAIT le
plafond, ici on ne fait que CLASSER, et l'optimum final est de toute facon
reverifie par bissection exacte.

SUR TOUTES LES POSITIONS. Comme a_p,lim, les plafonds de qualite sont pris au
PIRE des positions de C.POSITIONS_DESIGN : un correcteur qui ne tient qu'au
milieu de la plaque ne vaut rien sur un passage complet.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'paper_model'), HERE]

import config as C                                                   # noqa
from objective import evaluate                                       # noqa
from surface_error import periodic_response                          # noqa

AP_REF = 0.10e-3         # profondeur de reference des plafonds de qualite
TOL_SLE = 10e-6          # tolerance de finition


def plafonds_qualite(plate, ss, pd=None, rpm=None, positions=None,
                     n_modes=2, m=96, ap_ref=AP_REF, tol=TOL_SLE):
    """(a_p,qual, a_p,val) au PIRE des positions, par mise a l'echelle."""
    rpm = C.RPM_DESIGN if rpm is None else rpm
    pos = C.POSITIONS_DESIGN if positions is None else positions
    q = np.inf
    v = np.inf
    for fr in pos:
        r = periodic_response(plate, rpm, ap_ref, fr * plate.lp, ctrl=ss,
                              pd=pd, n_modes=n_modes, m=m,
                              coeff_scale=C.SIGN_SIM)
        s = abs(r['sle'])
        q = min(q, np.inf if s <= 0.0 else ap_ref * tol / s)
        v = min(v, np.inf if r['pv'] <= 0.0 else ap_ref * r['fz'] / r['pv'])
    return q, v


def evaluate_joint(plate, ss, pd=None, rpm=None, probes=None, positions=None,
                   m=None, detail=False, n_modes=2, tol=TOL_SLE):
    """Comme objective.evaluate, mais J = profondeur ATTEIGNABLE en mm."""
    J, info = evaluate(plate, ss, rpm=rpm, probes=probes, positions=positions,
                       m=m, detail=True, pd=pd)
    if not info.get('feasible', False):
        return (J, info) if detail else J        # crible echoue : note inchangee
    q, v = plafonds_qualite(plate, ss, pd=pd, rpm=rpm, positions=positions,
                            n_modes=n_modes, tol=tol)
    ap_lim = J * 1e-3
    a = min(ap_lim, q, v)
    lie = ('stabilite' if a == ap_lim else
           'qualite' if a == q else 'validite')
    info.update(J=a * 1e3, ap_lim=ap_lim, ap_qual=q, ap_val=v, liee_par=lie)
    return (info['J'], info) if detail else info['J']
