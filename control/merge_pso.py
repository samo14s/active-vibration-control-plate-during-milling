"""
merge_pso.py — reunir les fichiers d'optimisation produits en parallele
=======================================================================
`run_pso.py` accepte OUT_TAG pour ecrire dans un fichier separe, ce qui permet
d'optimiser plusieurs structures en meme temps sur les coeurs libres au lieu
de les enchainer. Ce script recolle les morceaux.

REGLE DE FUSION, ET ELLE N'EST PAS ANODINE. Chaque fichier partiel contient
AUSSI les structures deja optimisees, relues au demarrage. Les recopier
aveuglement ferait dependre le resultat de l'ordre des fichiers. On prend donc,
pour chaque structure, la version dont le `J` est le plus eleve, puis a `J`
egal celle qui compte le plus de graines — ce qui est
sans effet quand une structure n'apparait qu'une fois, et resout le conflit de
la seule facon defendable quand elle apparait plusieurs fois.

    python merge_pso.py resultat.npz partiel1.npz partiel2.npz ...
"""
import os
import sys

import numpy as np


def load(path):
    d = np.load(path, allow_pickle=True)
    out = {}
    for k in d.files:
        kind, _, field = k.partition('__')
        out.setdefault(kind, {})[field] = d[k]
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    dest, srcs = sys.argv[1], sys.argv[2:]
    merged = {}
    for p in srcs:
        if not os.path.exists(p):
            print(f'  absent, ignore : {p}')
            continue
        for kind, v in load(p).items():
            J = float(v.get('J', -np.inf))
            # LE CRITERE N'EST PAS `J` SEUL. Une reprise avec des graines
            # supplementaires conserve TOUTES les graines precedentes et son
            # `J` est le maximum sur l'ensemble : elle est donc un SURENSEMBLE
            # de l'ancienne, meme quand les graines ajoutees n'ameliorent
            # rien et que `J` ne bouge pas. Trier sur `J` seul gardait alors
            # l'ANCIEN enregistrement et jetait les graines ajoutees.
            #
            # Mesure : apres les graines 5-6, seules les cinq structures dont
            # `J` avait progresse gardaient six graines ; les six autres
            # retombaient a quatre. Le test de rang tournait donc avec des
            # effectifs INEGAUX — exactement l'iniquite que ces graines
            # devaient supprimer, et au detriment des structures que les
            # graines n'avaient pas flattees.
            n_new = int(np.size(v.get('J_seeds', [])))
            n_old = (-1 if kind not in merged
                     else int(np.size(merged[kind].get('J_seeds', []))))
            # L'ORDRE EST (J, nombre de graines) — ET PAS L'INVERSE.
            # J d'abord : une REOPTIMISATION sur du code corrige doit gagner
            # meme avec moins de graines. Le nombre de graines ensuite : a J
            # EGAL, une reprise avec graines supplementaires est un
            # surensemble de l'ancienne et doit gagner.
            #
            # J'avais d'abord ecrit (n_graines, J) : cela reglait le second
            # cas et cassait le premier EN SILENCE. Apres la correction de la
            # Riccati, musyn (J 0.2725 -> 0.4021) et hinf (0.3144 -> 0.4610)
            # ont ete reoptimises a cinq graines, pendant que des fichiers
            # partiels portaient encore leurs anciens enregistrements a sept
            # graines. La fusion a garde les ANCIENS, et tout l'aval a tourne
            # vingt-cinq minutes sur les correcteurs d'AVANT la correction,
            # sans que rien ne le signale.
            if kind not in merged or (J, n_new) > (
                    float(merged[kind].get('J', -np.inf)), n_old):
                merged[kind] = v
            print(f'  {kind:12s} J = {J:+.4f}   ({os.path.basename(p)})')
    np.savez_compressed(dest, **{f'{k}__{kk}': vv
                                 for k, v in merged.items()
                                 for kk, vv in v.items()})
    print(f'\n  {len(merged)} structures -> {dest}')
    print('  ' + ', '.join(sorted(merged)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
