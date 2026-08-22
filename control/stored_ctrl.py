"""
stored_ctrl.py — recharger un correcteur optimise, EN ENTIER
============================================================
Quatre scripts d'aval (lobes, robustesse, poles, comparaison) rechargeaient le
correcteur depuis `pso_{PROTOCOL}.npz` avec la meme boucle recopiee quatre
fois, qui lisait quatre champs : A, B, C, D. Tant que toutes les structures
etaient de simples correcteurs LTI, c'etait exact.

Le controle a retard actif de l'Eq. (30) casse cette hypothese. Ses deux gains
n'ont PAS de place dans un (A, B, C, D) : la loi
u_d = K_Pp y(t - tau) + K_Pd y'(t - tau) agit sur l'etat RETARDE, donc sur la
matrice A_tau, qu'aucune realisation non retardee ne porte. Une boucle qui ne
lit que les quatre champs recharge donc mu TOUT SEUL — sans rien lever, sans
rien afficher d'anormal, et en attribuant a la reference du papier des chiffres
qui ne sont pas les siens.

D'ou ce module : UN seul chargeur, qui rend le couple (ss, pd). Une structure
future portant elle aussi un terme hors-(A,B,C,D) n'aura qu'un endroit a
mettre a jour, au lieu de quatre endroits a ne pas oublier.
"""
import glob
import os

import numpy as np

import config as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

#: ordre d'affichage commun a tout l'aval.
ORDER = ['fopid', 'adrc', 'fdob', 'fdob12345', 'dvf', 'vpa', 'hinf', 'musyn',
         'musyn_td', 'lqg', 'mpc', 'smc', 'nmpdob']


def _paths():
    merged = os.path.join(OUT, f'pso_{C.PROTOCOL}.npz')
    return ([merged] if os.path.exists(merged) else []) + sorted(
        glob.glob(os.path.join(OUT, f'pso_{C.PROTOCOL}_*.npz')))


def delay_gains(d, kind):
    """(K_Pp, K_Pd) stockes pour cette structure, ou None.

    Un fichier ANTERIEUR au champ `pd` n'en a pas : on rend None, ce qui est
    la bonne reponse pour les douze structures sans retard. La seule qui en a
    un est `musyn_td`, et pour elle un fichier sans le champ est un fichier
    incomplet — mieux vaut le dire que le deviner."""
    key = f'{kind}__pd'
    if key not in d.files:
        if kind == 'musyn_td':
            raise KeyError(
                f'{kind} stocke sans ses gains de retard : le fichier date '
                f"d'avant le champ `pd`, le recharger evaluerait mu seul")
        return None
    v = np.asarray(d[key], float).ravel()
    return None if v.size == 0 else (float(v[0]), float(v[1]))


def discover():
    """{kind: (ss, pd)}, dans l'ordre d'affichage.

    Le fichier fusionne prime sur les partiels, et la premiere occurrence
    gagne — meme regle que les quatre boucles qu'il remplace."""
    found = {}
    for path in _paths():
        d = np.load(path, allow_pickle=True)
        for k in d.files:
            kind, _, field = k.partition('__')
            if field == 'A' and kind not in found:
                found[kind] = ((d[f'{kind}__A'], d[f'{kind}__B'],
                                d[f'{kind}__C'], d[f'{kind}__D']),
                               delay_gains(d, kind))
    rank = {k: i for i, k in enumerate(ORDER)}
    return {k: found[k]
            for k in sorted(found, key=lambda k: (rank.get(k, len(ORDER)), k))}
