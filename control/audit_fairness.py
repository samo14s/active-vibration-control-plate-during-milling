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

    kinds = []
    for k in d.files:
        kk = k.partition('__')[0]
        if kk not in kinds and f'{kk}__A' in d.files:
            kinds.append(kk)
    rows = {}
    for kind in kinds:
        var = float(d[f'{kind}__sign_variant'])
        # 'fdob12345' est la variante a cinq modes de la meme fabrique
        tg = d[f'{kind}__targets'] if f'{kind}__targets' in d.files else None
        Dg = Design(kind[:4] if kind.startswith('fdob') else kind,
                    plate, sign_loop, sign_variant=var, targets=tg)
        x = d[f'{kind}__x']
        par = Dg.decode(x)
        J, info = evaluate(plate, Dg.build(x), detail=True)
        rows[kind] = dict(x=x, par=par, J=J, info=info, D=Dg,
                          n_eval=int(d[f'{kind}__n_eval']),
                          seeds=d[f'{kind}__seeds'],
                          J_seeds=d[f'{kind}__J_seeds'],
                          n_par=int(d[f'{kind}__n_par']),
                          n_states=int(d[f'{kind}__n_states']),
                          variant=var,
                          variants=d[f'{kind}__variants'])

    print("\n  1-2. budget et graines")
    for k, r in rows.items():
        print(f"    {k:5s} : {r['n_eval']} evaluations, graines"
              f" {list(r['seeds'])}, conventions de signe"
              f" {sorted(set(r['variants'].tolist()))}, {r['n_par']}"
              f" parametres, {r['n_states']} etats"
              f", convention retenue {r['variant']:+.0f}")
    ref = list(rows[kinds[0]]['seeds'])
    same_seeds = all(list(r['seeds']) == ref for r in rows.values())
    print(f"    -> graines identiques : {'OUI' if same_seeds else 'NON'}")
    print("    -> nombres d'evaluations DIFFERENTS par construction :"
          " l'essaim est proportionnel a la dimension (10 + 4 n_dim), parce"
          " qu'a budget d'evaluations egal la qualite de recherche ne l'est"
          " pas (ecart a l'optimum ~2x plus grand en 7-D qu'en 5-D sur des"
          " paysages de reference). C'est la qualite de recherche qu'on"
          " egalise, et le cout est rapporte tel quel.")

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

    print("\n  5. dispersion sur les graines (DANS la convention retenue :"
          " melanger les deux conventions comparerait des choses differentes)")
    spreads = {}
    for k, r in rows.items():
        js = np.asarray(r['J_seeds'], float)
        vs = np.asarray(r['variants'], float)
        keep = js[vs == r['variant']]
        spreads[k] = float(np.ptp(keep)) if keep.size else np.nan
        print(f"    {k:5s} : J par graine = {np.round(keep, 4)}"
              f"   etendue = {spreads[k]:.4f}"
              f"   (autre convention : {np.round(js[vs != r['variant']], 4)})")
    best = max(kinds, key=lambda k: rows[k]['J'])
    spread = max(v for v in spreads.values() if np.isfinite(v))
    print(f"    meilleure structure : {best} (J = {rows[best]['J']:+.4f})")
    for k in kinds:
        if k == best:
            continue
        gap = rows[best]['J'] - rows[k]['J']
        print(f"    ecart {best} - {k:9s} = {gap:+.4f}"
              f"  -> {'SUPERIEUR' if abs(gap) > spread else 'INFERIEUR'}"
              f" a la plus grande dispersion intra-structure ({spread:.4f})")

    print("\n  6. controle croise : chaque structure notee par le MEME code")
    for k, r in rows.items():
        print(f"    {k:5s} : J recalcule = {r['J']:+.4f}"
              f"   (stocke {float(d[f'{k}__J']):+.4f})")


if __name__ == '__main__':
    main()
