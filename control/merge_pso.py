"""
merge_pso.py — reunir les fichiers d'optimisation produits en parallele
=======================================================================
`run_pso.py` accepte OUT_TAG pour ecrire dans un fichier separe, ce qui permet
d'optimiser plusieurs structures en meme temps sur les coeurs libres au lieu
de les enchainer. Ce script recolle les morceaux.

REGLE DE FUSION, ET ELLE N'EST PAS ANODINE. Chaque fichier partiel contient
AUSSI les structures deja optimisees, relues au demarrage. Les recopier
aveuglement ferait dependre le resultat de l'ordre des fichiers. On prend donc,
pour chaque structure, la version dont le `J` est le plus eleve — ce qui est
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
            if kind not in merged or J > float(merged[kind].get('J', -np.inf)):
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
