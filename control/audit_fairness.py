"""
audit_fairness.py — Verification du protocole d'equite lui-meme
================================================================
Une comparaison n'est equitable que si on peut le VERIFIER. Ce script controle,
apres coup et sur les fichiers de resultats :

  1. budget identique  : nombre d'evaluations de l'objectif par structure ;
  2. graines identiques ;
  3. bornes non actives : si un optimum se colle a une borne de recherche, ce
     n'est plus la structure qui limite mais la boite — on le signale ;
  4. contraintes egalement actives : Ms et effort atteints par chacun ;
  5. dispersion sur les graines : la difference entre structures est-elle plus
     grande que la dispersion de l'optimiseur ?
  6. rappel des couts : nombre de parametres et nombre d'etats.

    PROTOCOL=A|B python audit_fairness.py
"""
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'), HERE]

import config as C
from plate_model import build_plate, plant_vectors
from pso import Design
from objective import evaluate

OUT = os.path.join(HERE, '..', 'results')


def main():
    d = np.load(os.path.join(OUT, f'pso_{C.PROTOCOL}.npz'), allow_pickle=True)
    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    _, _, H, D_obs, sign_loop = plant_vectors(plate, C.N_MODES_DESIGN)

    print("=" * 74)
    print(f" AUDIT DU PROTOCOLE D'EQUITE  [protocole {C.PROTOCOL},"
          f" calage {C.CALIB}]")
    print("=" * 74)
    print(f"  plaque       : f = {np.round(plate.freq_n, 1)} Hz")
    print(f"  pastille     : {C.PATCH_SIDE} ; D_obs.H = {float(D_obs @ H):+.4f}"
          f" ; sign_loop = {sign_loop:+.0f}")
    print(f"  objectif     : {C.N_MODES_OBJ} modes, Floquet m ="
          f" {C.M_FLOQUET_PSO} ; evaluation : {C.N_MODES} modes, m ="
          f" {C.M_FLOQUET}")
    print(f"  contraintes  : Ms <= {C.MS_MAX} ; effort <= {C.V_PER_N:.0f} V/N"
          f" ; saturation +/-{C.V_MAX:.0f} V (temporel)")
    print(f"  Oustaloup    : [{C.OUST_WB / 2 / np.pi:.0f},"
          f" {C.OUST_WH / 2 / np.pi:.0f}] Hz, N = {C.OUST_N} ; lissage"
          f" {C.ROLLOFF_HZ:.0f} Hz ordre {C.ROLLOFF_ORDER}")

    rows = {}
    for kind in ('fopid', 'adrc'):
        Dg = Design(kind, plate, sign_loop)
        x = d[f'{kind}__x']
        par = Dg.decode(x)
        J, info = evaluate(plate, Dg.build(x), detail=True)
        rows[kind] = dict(x=x, par=par, J=J, info=info, D=Dg,
                          n_eval=int(d[f'{kind}__n_eval']),
                          seeds=d[f'{kind}__seeds'],
                          J_seeds=d[f'{kind}__J_seeds'],
                          n_par=int(d[f'{kind}__n_par']),
                          n_states=int(d[f'{kind}__n_states']))

    print("\n  1-2. budget et graines")
    for k, r in rows.items():
        print(f"    {k:5s} : {r['n_eval']} evaluations, graines"
              f" {list(r['seeds'])}, {r['n_par']} parametres,"
              f" {r['n_states']} etats")
    same = (rows['fopid']['n_eval'] == rows['adrc']['n_eval']
            and list(rows['fopid']['seeds']) == list(rows['adrc']['seeds']))
    print(f"    -> budget et graines identiques : {'OUI' if same else 'NON'}")

    print("\n  3. position de l'optimum dans les bornes"
          " (0 = borne basse, 1 = borne haute)")
    flagged = []
    for k, r in rows.items():
        Dg = r['D']
        for nm, u, lo, hi in zip(Dg.names, r['x'], Dg.lo, Dg.hi):
            val = lo + u * (hi - lo)
            edge = ' <-- BORNE ACTIVE' if (u < 0.02 or u > 0.98) else ''
            if edge:
                flagged.append((k, nm))
            print(f"    {k:5s} {nm:10s} : u = {u:5.3f}   valeur = {val:10.4g}"
                  f"   [{lo:g}, {hi:g}]{edge}")
    print(f"    -> bornes actives : "
          f"{'aucune' if not flagged else flagged}")

    print("\n  4. contraintes atteintes (evaluees sur le modele de l'objectif)")
    for k, r in rows.items():
        i = r['info']
        print(f"    {k:5s} : J = {r['J']:+.4f}   Ms = {i['Ms']:.3f} /"
              f" {C.MS_MAX}   effort = {i['V']:.0f} / {C.V_PER_N:.0f} V/N"
              f"   max Re(pole) = {i['max_re']:.2f} 1/s"
              f"   {'(faisable)' if i['feasible'] else '(NON FAISABLE)'}")

    print("\n  5. dispersion sur les graines")
    for k, r in rows.items():
        js = np.asarray(r['J_seeds'], float)
        print(f"    {k:5s} : J par graine = {np.round(js, 4)}"
              f"   etendue = {js.max() - js.min():.4f}")
    gap = rows['adrc']['J'] - rows['fopid']['J']
    spread = max(np.ptp(rows['fopid']['J_seeds']),
                 np.ptp(rows['adrc']['J_seeds']))
    print(f"    ecart entre structures = {gap:+.4f} ; plus grande dispersion"
          f" intra-structure = {spread:.4f}")
    print(f"    -> ecart {'SUPERIEUR' if abs(gap) > spread else 'INFERIEUR'}"
          f" a la dispersion de l'optimiseur")

    print("\n  6. controle croise : chaque structure notee par le MEME code")
    for k, r in rows.items():
        print(f"    {k:5s} : J recalcule = {r['J']:+.4f}"
              f"   (stocke {float(d[f'{k}__J']):+.4f})")


if __name__ == '__main__':
    main()
