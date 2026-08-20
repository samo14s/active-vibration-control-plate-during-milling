"""
report_extended.py — le tableau de la comparaison ETENDUE, en lignes
=====================================================================
`report_tables.py` met les structures en COLONNES. C'etait lisible a quatre ;
a douze, la table markdown fait treize colonnes et devient illisible, surtout
en arabe ou la table se lit de droite a gauche.

Ici les structures sont des LIGNES et les criteres des colonnes, et le
classement est celui du critere demande (`--par`), pas un ordre d'ajout.

Sources lues, toutes facultatives — ce qui manque laisse une case vide plutot
que d'arreter le script :
    results/pso_B.npz            J, Ms, effort, parametres, etats, budget
    results/compare_B.npz        fossoles, limites par position, temporel
    results/robust_new_B.npz     sept cas d'aggravation
    results/time_compare_B.npz   etalon temporel des douze

    PROTOCOL=B python report_extended.py [--par J|lim|pire|temps]
"""
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'), HERE]

import config as C                                            # noqa: E402
from report_tables import LAB                                 # noqa: E402

OUT = os.path.join(HERE, '..', 'results')


def _absorb(out, path):
    d = np.load(path, allow_pickle=True)
    for k in d.files:
        g, _, f = k.partition('__')
        if f:
            out.setdefault(g, {}).setdefault(f, d[k])
        else:
            out.setdefault(k, d[k])
    return out


def load(name, partials=None):
    """Lit `name`, et, si `partials` est un motif, complete avec les fichiers
    par structure que la fusion n'a pas encore reunis. Une campagne en cours
    est ainsi lisible sans attendre sa fin."""
    out, found = {}, False
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        _absorb(out, p)
        found = True
    if partials:
        for q in sorted(glob.glob(os.path.join(OUT, partials))):
            _absorb(out, q)
            found = True
    return out if found else None


def cell(v, fmt='{:.3f}'):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '—'
    return fmt.format(v) if isinstance(v, float) else str(v)


def main():
    par = 'J'
    if '--par' in sys.argv:
        par = sys.argv[sys.argv.index('--par') + 1]
    ps = load(f'pso_{C.PROTOCOL}.npz', f'pso_{C.PROTOCOL}_*.npz')
    cp = load(f'compare_{C.PROTOCOL}.npz')
    rb = load(f'robust_new_{C.PROTOCOL}.npz')
    tc = load(f'time_compare_{C.PROTOCOL}.npz')
    if ps is None:
        print('  results/pso_*.npz absent — rien a rapporter')
        return 1
    kinds = [k for k in ps if isinstance(ps[k], dict) and 'J' in ps[k]]

    # --- robustesse : pire cas et retention, par structure -----------------
    worst, nom = {}, {}
    if rb is not None and 'kinds' in rb:
        names = [str(x) for x in np.asarray(rb['kinds']).ravel()]
        M = np.asarray(rb['limits'], float)      # (cas, structures)
        for j, nme in enumerate(names):
            col = M[:, j]
            worst[nme] = float(np.min(col))
            nom[nme] = float(col[0])       # ligne 0 = modele de synthese

    # --- etalon temporel ---------------------------------------------------
    tlim = {}
    if tc is not None and 'names' in tc:
        for nme, v in zip([str(x) for x in np.asarray(tc['names']).ravel()],
                          np.asarray(tc['limits'], float).ravel()):
            tlim[nme] = float(v)

    # --- fossoles et position ---------------------------------------------
    lob, pos = {}, {}
    if cp is not None:
        for k in cp.get('lobes', {}):
            if k != 'rpm':
                lob[k] = float(np.mean(np.asarray(cp['lobes'][k], float)))
        for k in cp.get('positions', {}):
            if k != 'x':
                pos[k] = float(np.min(np.asarray(cp['positions'][k], float)))

    key = {'J': lambda k: -float(ps[k]['J']),
           'lim': lambda k: -pos.get(k, -np.inf),
           'pire': lambda k: -worst.get(k, -np.inf),
           'temps': lambda k: -tlim.get(k, -np.inf)}.get(par)
    if key is None:
        print(f'  critere inconnu : {par}')
        return 1
    kinds.sort(key=key)

    hdr = ['البنية', 'بارامترات', 'حالات', 'تقييمات', 'J (mm)', 'Ms',
           'الجهد V/N', 'حدّ التصميم (mm)', 'متوسّط الفصوص (mm)',
           'أسوأ متانة (mm)', 'الاحتفاظ', 'زمنيّ (mm)']
    print(f'\n### البروتوكول {C.PROTOCOL} — مرتّبة حسب `{par}`\n')
    print('| ' + ' | '.join(hdr) + ' |')
    print('|' + '---|' * len(hdr))
    for k in kinds:
        r = ps[k]
        ret = ('—' if k not in worst or nom.get(k, 0.0) <= 0.0
               else f'{100 * worst[k] / nom[k]:.0f} %')
        print('| ' + ' | '.join([
            LAB[k],
            cell(int(r['n_par'])) if 'n_par' in r else '—',
            cell(int(r['n_states'])) if 'n_states' in r else '—',
            cell(int(r['n_eval'])) if 'n_eval' in r else '—',
            cell(float(r['J']), '{:+.4f}'),
            cell(float(r['Ms']), '{:.2f}') if 'Ms' in r else '—',
            cell(float(r['V']), '{:.0f}') if 'V' in r else '—',
            cell(pos.get(k, None) and pos[k] * 1e3),
            cell(lob.get(k, None) and lob[k] * 1e3),
            cell(worst.get(k, None) and worst[k] * 1e3),
            ret,
            cell(tlim.get(k, None) and tlim[k] * 1e3),
        ]) + ' |')

    # --- ce que le tableau ne dit pas tout seul ----------------------------
    # J est stocke en MILLIMETRES (voir le format d'impression de
    # run_pso), AP_PROBE en metres : sans le facteur 1e3 le test de
    # saturation etait vrai pour tout le monde.
    cap = 3.0 * C.AP_PROBE[-1] * 1e3
    sat = [k for k in kinds if float(ps[k]['J']) >= cap - 1e-9]
    if sat:
        print(f'\n> **{", ".join(LAB[k] for k in sat)}** يبلغ سقف الهدف'
              f' ({cap:.3f} mm) : الهدف يعلن أنه لا يميّز فوقه،'
              ' فالترتيب داخل هذه المجموعة تقرّره المناصفة النهائية وحدها'
              ' (`audit_cap.py` يقيس كم يتفرّق المتعادلون).')
    if worst:
        flip = [k for k in kinds
                if k in worst and k in pos and pos[k] > 0
                and worst[k] / pos[k] < 0.5]
        if flip:
            print(f'\n> **{", ".join(LAB[k] for k in flip)}** يفقد أكثر من نصف'
                  ' حدّه الاسمي في أسوأ حالة اضطراب : الترتيب الاسمي ليس'
                  ' ترتيب الأسوأ.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
