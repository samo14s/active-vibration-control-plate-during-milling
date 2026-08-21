"""
run_compare.py — Etape 2 : evaluation equitable des deux correcteurs optimises
==============================================================================
Tout est calcule a PLEINE resolution (Floquet m = 200) sur le modele COMPLET a
cinq modes, sur les MEMES grilles, pour la boucle ouverte, le FOPID et
l'ADRC-FOPID :

  1. lobes de stabilite : a_p,lim (minimum sur tout le bord superieur) en
     fonction de la vitesse de broche ;
  2. limites par position a la vitesse de synthese ;
  3. reponses temporelles a une profondeur ou la boucle ouverte broute :
     deplacement, tension (SATUREE a +/- V_MAX pour les deux), spectres ;
  4. robustesse : modele de synthese contre modele complet, derive modale
     constatee en Section 5 du papier (+17 %, +9 %), amortissement a 80 %,
     perturbation +/-10 % de masse/raideur modales (Section 4.2), et calage
     theorique du Tableau 4 ;
  5. metriques frequentielles : marge de module, effort, Bode des correcteurs.

    PROTOCOL=A|B  CALIB=measured|theoretical  python run_compare.py
"""
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'), HERE]

import config as C
from plate_model import build_plate, plant_vectors, plant_frf
from fopid import ss_frf
from objective import limits, frequency_metrics, nominal_poles
from simulate import MillingSimulation, amplitude_spectrum, mean_abs_amplitude
from sim_controller import LTIController

OUT = os.path.join(HERE, '..', 'results')
# Les structures comparees sont celles que le fichier PSO contient, dans
# l'ordre ou elles ont ete ajoutees. Rien n'est code en dur : une structure
# nouvelle passe par EXACTEMENT le meme code d'evaluation que les anciennes,
# ce qui est la condition meme de l'equite.
def stored_kinds(d):
    seen = []
    for k in d.files:
        kk = k.partition('__')[0]
        if kk not in seen and f'{kk}__A' in d.files:
            seen.append(kk)
    return seen


def load():
    d = np.load(os.path.join(OUT, f'pso_{C.PROTOCOL}.npz'), allow_pickle=True)
    out = {}
    for k in stored_kinds(d):
        out[k] = dict(ss=(d[f'{k}__A'], d[f'{k}__B'], d[f'{k}__C'],
                          d[f'{k}__D']),
                      keys=d[f'{k}__keys'], values=d[f'{k}__values'],
                      J=float(d[f'{k}__J']), n_par=int(d[f'{k}__n_par']),
                      n_states=int(d[f'{k}__n_states']))
    return out


def envelope(r, n=1200):
    """Enveloppe min/max par bloc + metriques a pleine resolution."""
    N = len(r['t'])
    k = max(1, N // n)
    nb = N // k
    o = dict(t=r['t'][:nb * k:k], diverged=r['diverged'],
             t_div=np.nan if r['t_div'] is None else r['t_div'],
             mean_u=mean_abs_amplitude(r['u']),
             mean_y=mean_abs_amplitude(r['y_mill']),
             max_u=float(np.abs(r['u']).max()),
             max_u_cmd=float(np.abs(r['u_cmd']).max()),
             n_saturated=float(r['n_saturated']),
             max_y=float(np.abs(r['y_mill']).max()))
    for key in ('y_mill', 'u'):
        v = r[key][:nb * k].reshape(nb, k)
        o[key + '_min'], o[key + '_max'] = v.min(axis=1), v.max(axis=1)
    return o


def time_case(plate, ap, cfgs, store, tag):
    """Une passe complete par configuration, a la profondeur ap."""
    sim = MillingSimulation(plate, C.RPM_DESIGN, ap, n_modes=C.N_MODES,
                            n_sub=C.N_SUB, sign=C.SIGN_SIM, v_max=C.V_MAX)
    for name, ss in cfgs:
        c = None if ss is None else LTIController(ss, sim.dt)
        r = sim.run(controller=c, T=None)
        f, A = amplitude_spectrum(r['t'], r['y_mill'])
        fu, Au = amplitude_spectrum(r['t'], r['u'], scale=1.0)
        store[f'time{tag}_{name}'] = envelope(r)
        store[f'spec{tag}_{name}'] = dict(f=f, A=A, fu=fu, Au=Au)
        print(f"  temporel{tag} {name:14s} (a_p = {ap * 1e3:.2f} mm) : "
              f"{'DIVERGE a %.3f s' % r['t_div'] if r['diverged'] else 'stable'}"
              f"   |y|max = {np.abs(r['y_mill']).max() * 1e6:8.2f} um"
              f"   |u|max = {np.abs(r['u']).max():6.1f} V"
              f"   moy|u| = {mean_abs_amplitude(r['u']):5.2f} V"
              f"   sat : {r['n_saturated']} pas", flush=True)
    store[f'time{tag}_meta'] = dict(ap=ap, rpm=C.RPM_DESIGN, v_max=C.V_MAX)


def main():
    t00 = time.time()
    print("=" * 74)
    print(f" ETAPE 2 — COMPARAISON EQUITABLE DES STRUCTURES"
          f"   [protocole {C.PROTOCOL}, calage {C.CALIB}]")
    print("=" * 74)
    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    print(f"  plaque : f = {np.round(plate.freq_n, 1)} Hz,"
          f" zeta = {np.round(np.asarray(plate.zeta_modes) * 100, 2)} %")
    ctl = load()
    kinds = list(ctl)
    for k in kinds:
        print(f"  {k:5s} : {ctl[k]['n_par']} parametres,"
              f" {ctl[k]['n_states']} etats, J = {ctl[k]['J']:+.4f}")
        print("          " + "  ".join(
            f"{n}={v:.4g}" for n, v in zip(ctl[k]['keys'], ctl[k]['values'])))
    cfgs = [('boucle ouverte', None)] + [(k, ctl[k]['ss']) for k in kinds]
    store = {}

    # ---------------- 1-2. lobes de stabilite et limites par position -----
    # Ces deux etapes sont l'essentiel du temps de ce script : vingt et une
    # vitesses fois cinq positions fois une bissection a m = 200, par
    # structure — 19.5 s par appel a `limits` sur le FOPID, donc sept minutes
    # de fossoles, et bien plus sur les structures a cinquante-six etats. A
    # douze structures cela depasse l'heure et demie en un seul processus.
    # `run_lobes.py` les calcule EN PARALLELE et les met en cache ; on relit
    # le cache quand il est la. Le contenu est identique : meme appel a
    # `objective.limits`, meme grille, et le cache est invalide si la grille
    # de vitesses ou de positions a change.
    from run_lobes import SPEEDS as speeds, load_cache
    lob, pos = {}, {}
    for name, ss in cfgs:
        hit = load_cache(name)
        if hit is not None:
            lob[name], pos[name] = hit
            print(f"  lobes {name:14s} : (cache) moyenne"
                  f" {np.mean(lob[name]) * 1e3:.3f} mm,"
                  f" min {np.min(lob[name]) * 1e3:.3f} mm", flush=True)
            continue
        lob[name] = np.array([limits(plate, ss, rpm, hi=4.0e-3).min()
                              for rpm in speeds])
        pos[name] = limits(plate, ss, C.RPM_DESIGN, hi=4.0e-3)
        print(f"  lobes {name:14s} : moyenne {np.mean(lob[name]) * 1e3:.3f} mm,"
              f" min {np.min(lob[name]) * 1e3:.3f} mm"
              f"  ({time.time() - t00:.0f} s)", flush=True)
    store['lobes'] = dict(rpm=speeds, **lob)
    for name, _ in cfgs:
        print(f"  positions {name:14s} : {np.round(pos[name] * 1e3, 3)} mm"
              f"   min = {pos[name].min() * 1e3:.3f}", flush=True)
    store['positions'] = dict(x=np.array(C.POSITIONS), **pos)

    # ---------------- 3. reponses temporelles -----------------------------
    # (a) la condition S du papier : 4900 tr/min, a_p = 0.3 mm
    time_case(plate, 0.3e-3, cfgs, store, '_S')
    # (b) une profondeur ou meme la boucle fermee est sollicitee
    time_case(plate, 0.6e-3, cfgs, store, '')

    # ---------------- 4. robustesse ---------------------------------------
    def perturbed(freqs=None, zeta_scale=1.0, w_scale=1.0):
        pl = build_plate(C.PATCH_SIDE,
                         freqs=C.F_NOMINAL if freqs is None else freqs)
        if zeta_scale != 1.0:
            pl.zeta_modes = np.asarray(pl.zeta_modes, float) * zeta_scale
        if w_scale != 1.0:
            pl.calibrate_frequencies(list(np.asarray(
                pl.freq_n, float) * w_scale))
        return pl

    drift = [632.0, 1162.0] + list(C.F_NOMINAL[2:])
    cases = [('modele de synthese', dict(), C.N_MODES_OBJ),
             ('modele complet (verite)', dict(), C.N_MODES),
             ('derive +17/+9 %', dict(freqs=drift), C.N_MODES),
             ('amortissement x0.8', dict(zeta_scale=0.8), C.N_MODES),
             ('raideur/masse +10 %', dict(w_scale=np.sqrt(1.1)), C.N_MODES),
             ('raideur/masse -10 %', dict(w_scale=np.sqrt(0.9)), C.N_MODES),
             ('calage theorique', dict(freqs=C.F_THEORETICAL), C.N_MODES)]
    rob, labels = {}, []
    for tag, kw, nm in cases:
        pl = perturbed(**kw)
        row = [limits(pl, ss, C.RPM_DESIGN, n_modes=nm,
                      hi=4.0e-3).min() for _, ss in cfgs]
        rob[tag] = np.array(row)
        labels.append(tag)
        print(f"  robustesse [{tag:24s}] (n_modes={nm}) : " + "  ".join(
            f"{n} = {v * 1e3:.3f} mm" for (n, _), v in zip(cfgs, row)),
            flush=True)
    store['robust'] = rob
    store['robust_labels'] = np.array(labels)
    store['config_labels'] = np.array([n for n, _ in cfgs])

    # ---------------- 5. metriques frequentielles -------------------------
    f = np.logspace(0.5, 4.1, 400)
    fr = dict(f=f)
    Pu, Pf = plant_frf(plate, f, C.N_MODES, x_force=0.0)
    fr['Pu'] = np.abs(Pu)
    for name, ss in cfgs:
        if ss is None:
            continue
        K = ss_frf(ss, 2 * np.pi * f)
        S = 1.0 / (1.0 - Pu * K)
        fr[f'K_{name}'] = np.abs(K)
        fr[f'Kph_{name}'] = np.angle(K, deg=True)
        fr[f'S_{name}'] = np.abs(S)
        fr[f'U_{name}'] = np.abs(K * S * Pf)
        Ms, V = frequency_metrics(plate, ss, positions=C.POSITIONS,
                                  n_modes=C.N_MODES)
        ev = nominal_poles(plate, ss, n_modes=C.N_MODES)
        print(f"  frequentiel {name:8s} : Ms = {Ms:.3f}"
              f"   effort = {V:.0f} V/N   max Re(pole) = {ev.real.max():.1f}")
        store[f'metrics_{name}'] = np.array([Ms, V, ev.real.max()])
    store['freq'] = fr

    store['meta'] = dict(protocol=np.array(C.PROTOCOL),
                         calib=np.array(C.CALIB),
                         n_modes_obj=np.array(C.N_MODES_OBJ),
                         n_modes=np.array(C.N_MODES),
                         v_max=np.array(C.V_MAX),
                         kinds=np.array(kinds),
                         n_par=np.array([ctl[k]['n_par'] for k in kinds]),
                         n_states=np.array([ctl[k]['n_states']
                                            for k in kinds]))
    for k in kinds:
        store[f'par_{k}'] = dict(keys=ctl[k]['keys'], values=ctl[k]['values'])

    np.savez_compressed(os.path.join(OUT, f'compare_{C.PROTOCOL}.npz'),
                        **{f'{k}__{kk}': vv for k, v in store.items()
                           if isinstance(v, dict) for kk, vv in v.items()},
                        **{k: v for k, v in store.items()
                           if not isinstance(v, dict)})
    print(f"\n  -> results/compare_{C.PROTOCOL}.npz"
          f"   ({time.time() - t00:.0f} s)")


if __name__ == '__main__':
    main()
