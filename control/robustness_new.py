"""
robustness_new.py — les sept cas de robustesse, pour TOUTES les structures
==========================================================================
Meme jeu de plaques perturbees que `run_compare.py`, meme metrique — la limite
axiale au PIRE poste, par bissection sur cinq modes a m = 200 — appliquee aux
structures ajoutees (H-infini, mu, LQG, VPA, DVF, NMP-DOB) comme aux quatre
d'origine.

POURQUOI CE SCRIPT PLUTOT QUE `run_compare.py`. Ce dernier recalcule aussi les
lobes sur toute la plage de vitesse et le temporel, ce qui coute des heures. Ici
on veut UNE chose : le tableau de robustesse. Les correcteurs sont relus tels
qu'ils ont ete optimises, jamais reconstruits — un correcteur rebati depuis ses
parametres arrondis n'est pas le meme objet, comme la campagne H-infini
retiree l'a montre.

Les cas viennent du papier, pas d'un choix commode : la derive +17/+9 % est
celle que sa Section 5 constate, l'amortissement a 80 % et les +/-10 % de
masse/raideur viennent de sa Section 4.2, et le calage theorique est le jeu de
frequences de son Tableau 1.

    PROTOCOL=B CALIB=measured python robustness_new.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'), HERE]

import config as C                                            # noqa: E402
from plate_model import build_plate                           # noqa: E402
from closed_loop import limit as cl_limit                     # noqa: E402
from stored_ctrl import ORDER, discover as _discover           # noqa: E402

OUT = os.path.join(HERE, '..', 'results')

#: ORDRE D'AFFICHAGE, pas liste de structures : rien n'est code en dur ici.
#: `discover` lit ce que les fichiers contiennent REELLEMENT — le fichier
#: fusionne d'abord, puis les fichiers paralleles pour les structures que la
#: fusion n'a pas encore reunies. Une structure nouvelle apparait donc dans le
#: tableau sans qu'on touche a ce fichier, ce qui est la condition meme de
#: l'equite : le meme code d'evaluation pour toutes. Le chargement lui-meme
#: vit dans `stored_ctrl`, partage avec les trois autres scripts d'aval.
def discover():
    """[(kind, ss, pd)], boucle ouverte en tete.

    Le troisieme membre porte les gains de retard de l'Eq. (30) : sans lui,
    `musyn_td` serait recharge comme un correcteur mu ordinaire et la
    reference du papier serait mesuree amputee de sa moitie."""
    return [('boucle ouverte', None, None)] + [
        (k, ss, pd) for k, (ss, pd) in _discover().items()]


def worst_limit(plate, ss, pd, n_modes):
    """Limite au pire poste. `POSITIONS_DESIGN` sont des FRACTIONS de l_P ;
    `limit` attend une coordonnee physique — l'oublier rend zero partout."""
    return min(cl_limit(plate, C.RPM_DESIGN, x * plate.lp, ctrl=ss, pd=pd,
                        n_modes=n_modes, m=200, hi=6e-3)
               for x in C.POSITIONS_DESIGN)


def main():
    drift = [632.0, 1162.0] + list(C.F_NOMINAL[2:])
    cases = [('modele de synthese', dict(), C.N_MODES_OBJ),
             ('modele complet (verite)', dict(), C.N_MODES),
             ('derive +17/+9 %', dict(freqs=drift), C.N_MODES),
             ('amortissement x0.8', dict(zeta_scale=0.8), C.N_MODES),
             ('raideur/masse +10 %', dict(w_scale=np.sqrt(1.1)), C.N_MODES),
             ('raideur/masse -10 %', dict(w_scale=np.sqrt(0.9)), C.N_MODES),
             ('calage theorique', dict(freqs=C.F_THEORETICAL), C.N_MODES)]

    # Le MEME `perturbed` que run_compare.py, recopie a l'identique plutot
    # qu'importe : run_compare le definit a l'interieur de main(). Une
    # divergence entre les deux rendrait les deux tableaux incomparables, donc
    # `tests/` compare les deux constructions plaque par plaque.
    def perturbed(freqs=None, zeta_scale=1.0, w_scale=1.0):
        pl = build_plate(C.PATCH_SIDE,
                         freqs=C.F_NOMINAL if freqs is None else freqs)
        if zeta_scale != 1.0:
            pl.zeta_modes = np.asarray(pl.zeta_modes, float) * zeta_scale
        if w_scale != 1.0:
            pl.calibrate_frequencies(list(np.asarray(
                pl.freq_n, float) * w_scale))
        return pl

    got = discover()
    # KINDS : n'evaluer qu'une partie des structures, pour repartir le calcul
    # sur les coeurs. Sept cas x douze structures x six positions x une
    # bissection a m = 200 font environ six mille resolutions de Floquet ; en
    # un seul processus c'est l'etape la plus longue de toute la chaine, et
    # elle se decoupe sans la moindre interaction entre structures.
    want = os.environ.get('KINDS')
    if want:
        keep = {w.strip() for w in want.split(',')}
        got = [t for t in got if t[0] in keep]
        if not got:
            print(f'  aucune des structures demandees ({want}) n est presente')
            return 1
    print('  structures trouvees : '
          + ', '.join(t[0] for t in got))
    print('=' * 78)
    print(' ROBUSTESSE — limite axiale au pire poste [mm], 5 modes, m = 200')
    print('=' * 78)
    print(f'  structures : {", ".join(t[0] for t in got)}\n')

    table = {}
    for tag, kw, nm in cases:
        plate = perturbed(**kw)
        t0 = time.time()
        row = {}
        for kind, ss, pd in got:
            row[kind] = worst_limit(plate, ss, pd, nm)
        table[tag] = row
        print(f'  {tag:26s} ' + '  '.join(f'{k}={v * 1e3:.3f}'
                                          for k, v in row.items())
              + f'   ({time.time() - t0:.0f} s)', flush=True)

    tag = os.environ.get('OUT_TAG', '')
    dest = os.path.join(OUT, f'robust_new_{C.PROTOCOL}{tag}.npz')
    np.savez_compressed(dest,
                        labels=np.array([t for t, _, _ in cases]),
                        kinds=np.array([t[0] for t in got]),
                        limits=np.array([[table[t][k] for k, _, _ in got]
                                         for t, _, _ in cases]))
    print(f'\n  -> {os.path.basename(dest)}')
    return 0


def merge(dest, srcs):
    """Recolle les fichiers partiels produits avec OUT_TAG.

    Les ETIQUETTES DE CAS doivent coincider exactement : deux morceaux calcules
    avec des listes de cas differentes ne forment pas un tableau, ils forment
    deux tableaux. On refuse plutot que d'aligner au hasard.

    LE FICHIER DE DESTINATION EST AUSSI UNE SOURCE, et il doit l'etre. Il ne
    l'etait pas, et la fusion n'etait donc pas idempotente : elle reconstruisait
    le tableau a partir des seuls morceaux presents sur le disque, si bien que
    toute structure dont le morceau avait ete efface — parce qu'une campagne
    precedente l'avait deja repliee dans le fichier fusionne — disparaissait
    sans un mot. C'est arrive : en ajoutant `musyn_td`, la fusion a rendu un
    tableau de onze colonnes ou `hinf` et `musyn` n'etaient plus, et rien dans
    sa sortie ne le signalait — elle annonce le nombre de colonnes ecrites, pas
    celles qu'elle a perdues.

    Les morceaux frais gagnent : la destination n'est lue qu'a la fin, et ne
    sert qu'a combler. Une colonne ainsi reprise est SIGNALEE, parce qu'elle a
    ete calculee par une version anterieure du code et que ce depot a deja vu
    une table rester bit a bit identique alors qu'elle aurait du changer."""
    labels, cols = None, {}
    for p in srcs:
        if not os.path.exists(p):
            print(f'  absent, ignore : {p}')
            continue
        d = np.load(p, allow_pickle=True)
        lab = [str(x) for x in d['labels']]
        if labels is None:
            labels = lab
        elif lab != labels:
            raise SystemExit(f'  cas incompatibles dans {p} :\n'
                             f'    {lab}\n  contre\n    {labels}')
        M = np.asarray(d['limits'], float)
        for j, k in enumerate([str(x) for x in d['kinds']]):
            cols.setdefault(k, M[:, j])
        print(f'  {os.path.basename(p)} : '
              + ', '.join(str(x) for x in d['kinds']))
    repris = []
    if os.path.exists(dest):
        d = np.load(dest, allow_pickle=True)
        lab = [str(x) for x in d['labels']]
        if labels is not None and lab != labels:
            raise SystemExit(
                f'  {dest} existe avec d autres cas :\n    {lab}\n'
                f'  contre\n    {labels}\n'
                '  l ecraser perdrait ses colonnes — le deplacer d abord')
        labels = lab if labels is None else labels
        M = np.asarray(d['limits'], float)
        for j, k in enumerate([str(x) for x in d['kinds']]):
            if k not in cols:
                cols[k] = M[:, j]
                repris.append(k)
    if not cols:
        raise SystemExit('  rien a fusionner')
    if repris:
        print(f'\n  REPRISES du fichier fusionne precedent (aucun morceau'
              f' frais sur le disque) : {", ".join(repris)}')
        print('  ces colonnes datent d une campagne anterieure ; les'
              ' recalculer si le code d evaluation a change depuis')
    order = ['boucle ouverte'] + [k for k in ORDER if k in cols]
    order += [k for k in cols if k not in order]
    order = [k for k in order if k in cols]
    np.savez_compressed(dest, labels=np.array(labels),
                        kinds=np.array(order),
                        limits=np.column_stack([cols[k] for k in order]))
    print(f'\n  {len(order)} structures -> {dest}')


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--merge':
        merge(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(main() or 0)
