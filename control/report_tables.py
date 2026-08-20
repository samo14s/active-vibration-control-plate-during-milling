"""
report_tables.py — Tableaux Markdown des resultats de comparaison
==================================================================
Lit results/compare_{A,B}.npz et results/pso_{A,B}.npz et imprime les tableaux
prets a coller dans COMPARAISON_ADRC_FOPID.md. Aucun calcul : uniquement de la
mise en forme, pour qu'aucun chiffre du rapport ne soit recopie a la main.

    PROTOCOL=A|B python report_tables.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'), HERE]

import config as C

OUT = os.path.join(HERE, '..', 'results')
_LAB = {'boucle ouverte': 'بلا تحكّم', 'fopid': 'FOPID',
        'adrc': 'ADRC-FOPID', 'fdob': 'FDOB (نمطان)',
        'fdob12345': 'FDOB (5 أنماط)', 'dvf': 'DVF', 'vpa': 'VPA',
        'hinf': 'H∞', 'musyn': 'μ-synthesis', 'lqg': 'LQG', 'mpc': 'MPC',
        'smc': 'SMC', 'nmpdob': 'NMP-DOB'}


class _Fallback(dict):
    def __missing__(self, k):
        return str(k).upper()


LAB = _Fallback(_LAB)


def load(name):
    d = np.load(os.path.join(OUT, name), allow_pickle=True)
    out = {}
    for k in d.files:
        if '__' in k:
            g, f = k.split('__', 1)
            out.setdefault(g, {})[f] = d[k]
        else:
            out[k] = d[k]
    return out


def row(label, vals, fmt='{:.3f}'):
    cells = ' | '.join(fmt.format(v) if isinstance(v, float) else str(v)
                       for v in vals)
    return f'| {label} | {cells} |'


def main(prot=None):
    prot = C.PROTOCOL if prot is None else prot
    cp = load(f'compare_{prot}.npz')
    ps = load(f'pso_{prot}.npz')
    lob, pos = cp['lobes'], cp['positions']
    KEYS = [str(x) for x in cp['config_labels']]
    K = KEYS[1:]                                  # sans la boucle ouverte

    print(f'\n### البروتوكول {prot}\n')
    print('| الكمّية | ' + ' | '.join(LAB.get(k, k) for k in KEYS) + ' |')
    print('|---' * (len(KEYS) + 1) + '|')
    print(row('عدد البارامترات', ['—'] + [int(ps[k]['n_par']) for k in K]))
    print(row('عدد الحالات', ['—'] + [int(ps[k]['n_states']) for k in K]))
    print(row('تقييمات PSO', ['—'] + [int(ps[k]['n_eval']) for k in K]))
    print(row('اتّفاقية الإشارة المُختارة',
              ['—'] + [f"{float(ps[k]['sign_variant']):+.0f}"
                       for k in K]))
    print(row('الهدف J', ['—'] + [float(ps[k]['J'])
                                  for k in K], '{:+.4f}'))
    print(row('متوسّط الفصوص 3000–7000 (mm)',
              [float(np.mean(lob[k]) * 1e3) for k in KEYS]))
    print(row('أدنى الفصوص (mm)',
              [float(np.min(lob[k]) * 1e3) for k in KEYS]))
    print(row(f'الحدّ عند {C.RPM_DESIGN} tr/min، أسوأ موضع (mm)',
              [float(np.min(pos[k]) * 1e3) for k in KEYS]))
    for tag in ('_S', ''):
        meta = cp[f'time{tag}_meta']
        ap = float(meta['ap']) * 1e3
        vals = []
        for k in KEYS:
            s = cp[f'time{tag}_{k}']
            vals.append('يتباعد' if bool(s['diverged'])
                        else f"{float(s['max_y']) * 1e6:.2f}")
        print(row(f'أقصى إزاحة عند ap = {ap:.2f} mm (µm)', vals))
        print(row(f'ذروة التوتر عند ap = {ap:.2f} mm (V)',
                  ['—'] + [float(cp[f'time{tag}_{k}']['max_u'])
                           for k in K], '{:.1f}'))
        print(row(f'متوسّط |التوتر| عند ap = {ap:.2f} mm (V)',
                  ['—'] + [float(cp[f'time{tag}_{k}']['mean_u'])
                           for k in K], '{:.2f}'))
        print(row(f'خطوات التشبّع عند ap = {ap:.2f} mm',
                  ['—'] + [int(cp[f'time{tag}_{k}']['n_saturated'])
                           for k in K]))
    print(row('هامش الوحدة Ms (قيد ≤ 2)',
              ['—'] + [float(cp[f'metrics_{k}'][0])
                       for k in K]))
    print(row('الجهد لكل نيوتن (V/N، قيد ≤ 450)',
              ['—'] + [float(cp[f'metrics_{k}'][1])
                       for k in K], '{:.0f}'))
    print(row('أبطأ قطب اسمي (1/s)',
              ['—'] + [float(cp[f'metrics_{k}'][2])
                       for k in K], '{:.1f}'))

    print('\n**المتانة** (أدنى حدّ عند سرعة التصميم، mm)\n')
    labs = [str(x) for x in cp['robust_labels']]
    print('| الحالة | ' + ' | '.join(LAB.get(k, k) for k in KEYS) + ' |')
    print('|---' * (len(KEYS) + 1) + '|')
    for t in labs:
        print(row(t, [float(v * 1e3) for v in cp['robust'][t]]))

    print('\n**البارامترات المُختارة**\n')
    for k in K:
        pr = cp[f'par_{k}']
        print(f"* {LAB.get(k, k)} : " + ', '.join(
            f"`{n}` = {v:.4g}" for n, v in zip(pr['keys'], pr['values'])))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
