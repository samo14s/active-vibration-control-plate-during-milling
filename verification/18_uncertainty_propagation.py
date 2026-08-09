"""L'INCERTITUDE QUE L'ARTICLE PORTE, PROPAGEE JUSQU'AUX CHIFFRES PUBLIES.

CE QUE CE SCRIPT AJOUTE
-----------------------
`run_demo --full` teste six perturbations UNE A UNE. C'est un depistage, pas une
propagation d'incertitude, et il est optimiste par construction : les erreurs de
modele ne se presentent pas une a une. Le pire cas sur une BOITE est en general
pire que le pire cas sur chacune de ses ARETES.

Ce script traite les memes grandeurs comme ce qu'elles sont — des quantites
incertaines — et propage cette incertitude jusqu'a la limite de passe, qui est
le chiffre publie.

D'OU VIENT CHAQUE INTERVALLE
----------------------------
UN SEUL est mesure. Les autres sont des CONVENTIONS, et le script les etiquette
comme telles ; le classement de sensibilite (etape A) dit lesquelles portent
reellement le resultat, donc lesquelles meriteraient d'etre mesurees.

  gain_H   [0.44, 2.92]   MESURE. Les deux bouts de l'encadrement de F9
                          (verification/17). Il ABSORBE d31, le rendement de
                          collage eta, et le module du patch : tous agissent
                          comme un facteur d'echelle pur sur H_Pe, donc ils
                          sont deja dedans et ne comptent pas deux fois.
  zeta_i   +/- 30 %       CONVENTION, par mode et independants. Le Tableau 4
                          donne des valeurs sans incertitude ; un amortissement
                          modal identifie sur des modes a zeta ~ 0.2 % se
                          disperse couramment de 20 a 40 %.
  K        +/- 10 %       CONVENTION. Erreur RESIDUELLE apres recalage : les
                          frequences sont deja calees sur la mesure, donc ceci
                          represente l'ecart plaque simulee / plaque de synthese
                          (souplesse d'encastrement, usinage), pas l'erreur EF.
  Kt       +/- 15 %       CONVENTION. Pression de coupe specifique identifiee.
                          Elle multiplie a3 ET a4, donc elle agit comme une
                          erreur sur ap : c'est la voie la plus directe vers la
                          limite de stabilite.
  kc       +/- 30 %       CONVENTION. k1, k2 de l'Eq. (3) : ils fixent le
                          MELANGE des deux composantes d'effort, pas son
                          amplitude.

Tirage LOG-uniforme sur chaque intervalle : ce sont des facteurs d'echelle, et
sur un facteur d'echelle c'est le log qui est la grandeur naturellement plate.

CE QUI N'EST PAS INCLUS, ET POURQUOI
------------------------------------
  * les frequences : calees EXACTEMENT sur les cinq valeurs mesurees, donc leur
    incertitude est celle de la mesure de l'article, que l'article ne donne pas ;
  * la geometrie et les materiaux (Tableaux 1-2) : ils agissent par les modes et
    par H_Pe, et leur effet sur H_Pe est deja dans gain_H. Les faire varier
    exigerait de reassembler les elements finis a chaque tirage (14.8 s), pour
    un effet residuel petit devant les cinq ci-dessus ;
  * kc x2.9 : ce n'est pas une incertitude, c'est un SCENARIO extreme. Il reste
    dans run_demo --full a ce titre.

LE CORRECTEUR EST TOUJOURS SYNTHETISE SUR LA PLAQUE NOMINALE et simule sur la
plaque tiree — c'est la convention de run_demo --full, et c'est ce qui fait de
l'essai un vrai essai d'erreur de modele.

Duree : ~40 min sur 4 coeurs.
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, 'simulation', 'sim_kit'),
                os.path.join(_R, 'simulation')]
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_R, 'simulation'))

import simulation_base as SB                                     # noqa: E402
import run_demo as RD                                            # noqa: E402
from model_v2 import make_sim                                    # noqa: E402

SPEEDS = RD.SPEEDS
AP_GATE = 0.10e-3            # profondeur de reference, celle du crible
N_MC_SIGMA = 400             # tirages pour P(stable a AP_GATE)
N_MC_BLIM = 80               # tirages pour la loi de la limite publiee

# (nom, borne basse, borne haute, statut)
UNCERT = [('gain_H', 0.44, 2.92, 'MESURE (F9, script 17)'),
          ('zeta', 0.70, 1.43, 'convention +/- 30 %, par mode'),
          ('K', 0.90, 1.10, 'convention +/- 10 %'),
          ('Kt', 0.85, 1.18, 'convention +/- 15 %'),
          ('kc', 0.70, 1.43, 'convention +/- 30 %')]

_SIM = _PLATE = None
_K0 = _C0 = _H0 = _KC0 = _KT0 = None


def _init_worker():
    global _SIM, _PLATE, _K0, _C0, _H0, _KC0, _KT0
    _SIM = make_sim()
    _PLATE = __import__('copy').deepcopy(_SIM.plate)   # plaque de SYNTHESE
    _K0 = np.array(_SIM.plate.Kp, float).copy()
    _C0 = np.array(_SIM.plate.Cp, float).copy()
    _H0 = np.array(_SIM.plate.H_Pe_modal, float).copy()
    _KC0 = (float(_SIM.k1c), float(_SIM.k2c))
    _KT0 = float(SB.KT)


def _apply(s):
    """s = dict des facteurs. Perturbe la plaque SIMULEE uniquement."""
    zs = np.asarray(s.get('zeta', 1.0), float)
    _SIM.plate.Kp = _K0*s.get('K', 1.0)
    # Cp = diag(2 zeta_i omega_i) : un facteur PAR MODE se pose directement sur
    # la diagonale. On reconstruit explicitement plutot que de multiplier deux
    # matrices diagonales terme a terme, qui ne marcherait que par accident.
    if zs.ndim == 0:
        _SIM.plate.Cp = _C0*float(zs)
    else:
        _SIM.plate.Cp = np.diag(np.diag(_C0)*zs)
    _SIM.plate.H_Pe_modal = _H0*s.get('gain_H', 1.0)
    _SIM.k1c = _KC0[0]*s.get('kc', 1.0)
    _SIM.k2c = _KC0[1]*s.get('kc', 1.0)
    SB.KT = _KT0*s.get('Kt', 1.0)
    _SIM._cache.clear()


def _factory(arch):
    if arch == 'libre':
        return None
    if arch == 'LQG':
        return RD.factory_lqg(_PLATE, RD.X_LQG)
    return RD.factory_eso(_PLATE, RD.X_ESO)


def _eval_sigma(args):
    """Retourne, pour un tirage : stable a AP_GATE aux CINQ vitesses ?"""
    s, arch = args
    _apply(s)
    mk = _factory(arch)
    worst = -1e9
    ok = True
    for rpm in SPEEDS:
        tau = 60.0/(SB.N_TEETH*rpm)
        c = None if mk is None else mk(tau/SB.STEPS_PER_TOOTH, tau)
        r = _SIM.run(c, rpm=rpm, ap=AP_GATE, T=SB.T_LIMIT)
        worst = max(worst, r['sigma'])
        ok = ok and r['stable']
    return bool(ok), float(worst)


def _eval_blim(args):
    """Retourne la limite du PIRE CAS sur les cinq vitesses, en mm."""
    s, arch = args
    _apply(s)
    mk = _factory(arch)
    return min(_SIM.stability_limit(mk, rpm=rpm, lo=0.02e-3, hi=4.0e-3,
                                    T=SB.T_LIMIT, tol=2e-6)
               for rpm in SPEEDS)*1e3


def sample(rng, n):
    """n tirages log-uniformes de tous les facteurs simultanement."""
    out = []
    for _ in range(n):
        s = {}
        for name, lo, hi, _st in UNCERT:
            if name == 'zeta':
                s[name] = np.exp(rng.uniform(np.log(lo), np.log(hi), size=5))
            else:
                s[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        out.append(s)
    return out


def figure(npz_path):
    """Trace depuis le .npz : tornade de sensibilite + lois de la limite."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=False)
    ARCH = [str(x) for x in d['arch']]
    base = {a: float(v) for a, v in zip(ARCH, d['base'])}
    names = [str(x) for x in d['oat_names']]
    lo, hi, bnd = d['oat_lo'], d['oat_hi'], d['oat_bounds']
    iL = ARCH.index('LQG')

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # --- tornade : effet de chaque parametre, un a la fois, sur le LQG
    eff = [max(abs(lo[i, iL] - base['LQG']), abs(hi[i, iL] - base['LQG']))
           for i in range(len(names))]
    o = np.argsort(eff)
    for k, i in enumerate(o):
        a, b = lo[i, iL], hi[i, iL]
        ax[0].barh(k, abs(b - a), left=min(a, b), height=.6,
                   color='#1f4e79', alpha=.85)
        ax[0].text(max(a, b) + .012, k, f'x{bnd[i,0]:.2f} - x{bnd[i,1]:.2f}',
                   va='center', fontsize=8, color='.35')
    ax[0].axvline(base['LQG'], color='#c00000', lw=1.6,
                  label=f"publie = {base['LQG']:.4f} mm")
    ax[0].set_yticks(np.arange(len(o)))
    ax[0].set_yticklabels([names[i] for i in o])
    ax[0].set_xlim(0, None)
    ax[0].set_xlabel('limite du pire cas, LQG [mm]')
    ax[0].set_title('Un parametre a la fois : ce qui porte le resultat')
    ax[0].legend(loc='lower right', fontsize=9)
    ax[0].grid(alpha=.3, axis='x')

    # --- lois simultanees
    bins = np.linspace(0, max(d['blim_LQG'].max(), d['blim_ESO'].max())*1.02, 26)
    ax[1].hist(d['blim_LQG'], bins=bins, alpha=.55, color='#1f4e79',
               label=f"LQG  (median {np.median(d['blim_LQG']):.4f})")
    ax[1].hist(d['blim_ESO'], bins=bins, alpha=.55, color='#2e7d32',
               label=f"ESO  (median {np.median(d['blim_ESO']):.4f})")
    ax[1].axvline(base['LQG'], color='#1f4e79', ls='--', lw=1.8,
                  label=f"LQG publie {base['LQG']:.4f}")
    ax[1].axvline(base['ESO'], color='#2e7d32', ls='--', lw=1.8,
                  label=f"ESO publie {base['ESO']:.4f}")
    ax[1].axvline(base['libre'], color='k', ls=':', lw=1.4,
                  label=f"libre nominal {base['libre']:.4f}")
    ax[1].set_xlabel('limite du pire cas sur les cinq vitesses [mm]')
    ax[1].set_ylabel(f"tirages (n = {len(d['blim_LQG'])})")
    ax[1].set_title('Tous les parametres varies ENSEMBLE')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    fig.tight_layout()
    out = os.path.join(_HERE, 'uncertainty.png')
    fig.savefig(out, dpi=130)
    print(f'figure  : {out}')


if __name__ == '__main__':
    if '--plot' in sys.argv:
        figure(os.path.join(_HERE, 'uncertainty_mc.npz'))
        sys.exit(0)
    t0 = time.time()
    print(__doc__.split('Duree')[0])
    print('=' * 76)
    nw = min(4, max(1, (os.cpu_count() or 2)))
    ARCH = ['libre', 'LQG', 'ESO']

    # ------------------------------------------------------- A. sensibilite
    print('A. SENSIBILITE, UN PARAMETRE A LA FOIS')
    print('   limite du pire cas sur les cinq vitesses, en mm\n')
    jobs, tags = [], []
    for name, lo, hi, _st in UNCERT:
        for lab, v in (('bas', lo), ('haut', hi)):
            s = {name: (np.full(5, v) if name == 'zeta' else v)}
            for a in ARCH:
                jobs.append((s, a))
                tags.append((name, lab, v, a))
    with mp.Pool(nw, initializer=_init_worker) as pool:
        res = pool.map(_eval_blim, jobs)
    base = {}
    with mp.Pool(nw, initializer=_init_worker) as pool:
        base_res = pool.map(_eval_blim, [({}, a) for a in ARCH])
    for a, v in zip(ARCH, base_res):
        base[a] = v
    print(f'   {"nominal":22s} ' + ''.join(f'{base[a]:>10.4f}' for a in ARCH)
          + '     (reference)')
    R = {}
    for (name, lab, v, a), val in zip(tags, res):
        R[(name, lab, a)] = val
    rank = []
    for name, lo, hi, st in UNCERT:
        for lab, v in (('bas', lo), ('haut', hi)):
            row = [R[(name, lab, a)] for a in ARCH]
            print(f'   {name+" x"+f"{v:.2f}":22s} ' + ''.join(f'{x:>10.4f}'
                                                              for x in row))
        swing = max(abs(R[(name, l, a)]/max(base[a], 1e-9) - 1)
                    for l in ('bas', 'haut') for a in ARCH if a != 'libre')
        rank.append((swing, name, st))
    print('\n   classement par amplitude d\'effet sur les correcteurs :')
    for swing, name, st in sorted(rank, reverse=True):
        print(f'      {name:8s} {100*swing:6.1f} %   {st}')
    print(f'\n   [{time.time()-t0:.0f} s]', flush=True)

    # ------------------------------------------- B. Monte-Carlo sur sigma
    print(f'\nB. MONTE-CARLO SIMULTANE — tenue a ap = {AP_GATE*1e3:.2f} mm')
    print(f'   {N_MC_SIGMA} tirages, les cinq facteurs varies ENSEMBLE,')
    print('   correcteur synthetise sur la plaque nominale\n')
    rng = np.random.default_rng(20240809)
    S = sample(rng, N_MC_SIGMA)
    for a in ARCH:
        with mp.Pool(nw, initializer=_init_worker) as pool:
            out = pool.map(_eval_sigma, [(s, a) for s in S])
        ok = np.array([o[0] for o in out])
        sg = np.array([o[1] for o in out])
        p = 100*ok.mean()
        print(f'   {a:6s} : stable aux 5 vitesses dans {p:5.1f} % des tirages'
              f'   (sigma pire cas median {np.median(sg):+7.3f} /s)', flush=True)
    print(f'\n   [{time.time()-t0:.0f} s]', flush=True)

    # ------------------------------------------- C. Monte-Carlo sur b_lim
    print(f'\nC. MONTE-CARLO SIMULTANE — loi de la LIMITE PUBLIEE')
    print(f'   {N_MC_BLIM} tirages ; la valeur publiee est le "nominal"\n')
    S2 = sample(rng, N_MC_BLIM)
    summary = {}
    for a in ARCH:
        with mp.Pool(nw, initializer=_init_worker) as pool:
            v = np.array(pool.map(_eval_blim, [(s, a) for s in S2]))
        summary[a] = v
        q = np.percentile(v, [5, 25, 50, 75, 95])
        print(f'   {a:6s} : publie {base[a]:.4f} | median {q[2]:.4f} | '
              f'5-95 % [{q[0]:.4f}, {q[4]:.4f}] | min {v.min():.4f} mm',
              flush=True)

    print('\n' + '=' * 76)
    print('LECTURE')
    print('=' * 76)
    vl, ve = summary['LQG'], summary['ESO']
    win = 100*float((vl > ve).mean())
    print(f'   LQG > ESO dans {win:.0f} % des tirages '
          f'(le classement publie tient-il sous incertitude ?)')
    for a in ('LQG', 'ESO'):
        v = summary[a]
        print(f'   {a:4s} : la valeur publiee ({base[a]:.4f}) est le '
              f'{100*float((v < base[a]).mean()):.0f}e centile du tirage')
    print(f'   marge libre nominale : {base["libre"]:.4f} mm ; '
          f'P(LQG sous cette valeur) = '
          f'{100*float((vl < base["libre"]).mean()):.1f} %')
    npz = os.path.join(_HERE, 'uncertainty_mc.npz')
    np.savez(npz,
             **{f'blim_{a}': summary[a] for a in ARCH},
             base=np.array([base[a] for a in ARCH]),
             arch=np.array(ARCH),
             oat_names=np.array([u[0] for u in UNCERT]),
             oat_lo=np.array([[R[(u[0], 'bas', a)] for a in ARCH]
                              for u in UNCERT]),
             oat_hi=np.array([[R[(u[0], 'haut', a)] for a in ARCH]
                              for u in UNCERT]),
             oat_bounds=np.array([[u[1], u[2]] for u in UNCERT]))
    print(f'\ndonnees : {npz}')
    figure(npz)
    print(f'termine en {time.time()-t0:.0f} s')
