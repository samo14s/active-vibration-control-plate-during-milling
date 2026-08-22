"""
run_lobes.py — les fossoles de stabilite, calculees A PART et EN PARALLELE
==========================================================================
Dans `run_compare.py` les deux premieres etapes — fossoles sur la grille de
vitesses, puis limites par position a la vitesse de synthese — sont a elles
seules la majeure partie du temps : vingt et une vitesses fois cinq positions
fois une bissection a m = 200 et cinq modes, par structure. Mesure sur le
FOPID (26 etats) : un appel `limits()` coute 19.5 s, donc sept minutes de
fossoles ; a douze structures, dont plusieurs bien plus lourdes, on depasse
l'heure et demie en un seul processus.

Or ces calculs sont INDEPENDANTS d'une structure a l'autre. Ce script les
fait pour le sous-ensemble demande et les met en cache ; `run_compare.py`
relit le cache au lieu de recalculer. Rien d'autre ne change : le cache
contient exactement ce que run_compare aurait produit, par le meme appel a
`objective.limits`.

    PROTOCOL=B KINDS=fopid,adrc python run_lobes.py
    PROTOCOL=B python run_lobes.py            (toutes celles qu'on trouve)

Le fichier de cache porte le nom de la structure. Cela NE SUFFIT PAS a
garantir qu'un seul processus l'ecrit : rien n'empeche de confier des listes
qui se recouvrent a deux processus, et deux `np.savez_compressed` simultanes
sur le meme chemin le laisseraient tronque. `compute` prend donc un VERROU
`mkdir`, atomique sur un systeme de fichiers POSIX. (Cette phrase affirmait
auparavant que le nommage suffisait — il ne suffisait pas.)
"""
import glob
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'), HERE]

import config as C                                            # noqa: E402
from plate_model import build_plate                           # noqa: E402
from objective import limits                                  # noqa: E402
from stored_ctrl import discover as stored                    # noqa: E402

OUT = os.path.join(HERE, '..', 'results')
SPEEDS = np.arange(3000, 7001, 200)


def cache_path(kind):
    return os.path.join(OUT, f'lobes_{C.PROTOCOL}_{kind.replace(" ", "_")}.npz')


def load_cache(kind):
    """(lobes, positions) si le cache existe ET porte la meme grille."""
    p = cache_path(kind)
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    if not np.array_equal(np.asarray(d['rpm']), SPEEDS):
        return None
    if not np.allclose(np.asarray(d['x']), np.asarray(C.POSITIONS)):
        return None
    return np.asarray(d['lobes'], float), np.asarray(d['positions'], float)


def compute(kind, ss, pd):
    """Calcule et met en cache. VERROU ATOMIQUE : plusieurs processus peuvent
    se voir confier des listes qui se recouvrent, et deux d'entre eux ecrivant
    le meme .npz en meme temps le laisseraient tronque. `mkdir` est atomique
    sur un systeme de fichiers POSIX ; celui qui echoue passe son chemin."""
    lock = cache_path(kind) + '.lock'
    try:
        os.mkdir(lock)
    except FileExistsError:
        print(f'  {kind:16s} deja en cours dans un autre processus')
        return
    try:
        _compute(kind, ss, pd)
    finally:
        os.rmdir(lock)


def _compute(kind, ss, pd):
    t0 = time.time()
    lob = np.array([limits(plate_g, ss, rpm, hi=4.0e-3, pd=pd).min()
                    for rpm in SPEEDS])
    pos = limits(plate_g, ss, C.RPM_DESIGN, hi=4.0e-3, pd=pd)
    np.savez_compressed(cache_path(kind), rpm=SPEEDS, lobes=lob,
                        x=np.asarray(C.POSITIONS), positions=pos)
    print(f'  {kind:16s} moyenne {np.mean(lob) * 1e3:.3f} mm, '
          f'min {np.min(lob) * 1e3:.3f} mm, '
          f'pire poste a {C.RPM_DESIGN} tr/min {pos.min() * 1e3:.4f} mm'
          f'   ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    plate_g = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    st = stored()
    want = os.environ.get('KINDS')
    todo = ([('boucle ouverte', (None, None))] if not want
            or 'boucle ouverte' in want else [])
    if want:
        keep = {w.strip() for w in want.split(',')}
        todo += [(k, st[k]) for k in st if k in keep]
    else:
        todo += [(k, st[k]) for k in st]
    if not todo:
        print(f'  rien a calculer (demande : {want})')
        sys.exit(1)
    print(f'  fossoles {SPEEDS[0]}-{SPEEDS[-1]} tr/min, '
          f'{len(C.POSITIONS)} positions, m = {C.M_FLOQUET}, '
          f'{C.N_MODES} modes')
    for kind, (ss, pd) in todo:
        if load_cache(kind) is not None:
            print(f'  {kind:16s} deja en cache')
            continue
        compute(kind, ss, pd)
