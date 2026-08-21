"""
run_time_compare.py — l'etalon TEMPOREL, passe a TOUTES les structures
======================================================================
Floquet decide de la stabilite LOCALE : il linearise autour de l'equilibre et
ignore tout ce qui est grand signal. Ce script mesure l'autre moitie, avec le
meme critere pour tout le monde :

    a_p,lim TEMPOREL = plus grande profondeur pour laquelle la simulation ne
    diverge a AUCUNE des positions d'outil du protocole, avec la meme
    saturation +/-V_MAX, le meme pas, le meme modele a cinq modes.

C'est la transposition exacte du protocole de Floquet (minimum sur les
positions) dans le domaine temporel. `run_adaptive.py` l'avait defini pour
comparer l'observateur SUPERVISE, qui n'est ni lineaire ni invariant ; il
vaut ici pour les douze structures.

POURQUOI CE SCRIPT EXISTE, ET CE QU'IL AJOUTE. Deux structures ont une forme
NON LINEAIRE que Floquet ne peut pas voir :

  SMC. Sa realisation d'etat (`nonlinear.smc_lti_ss`) est EXACTE a l'interieur
  de la couche limite, donc pour la stabilite locale. Hors de la couche, la
  commande sature a |u| <= K_s : c'est du grand signal. Ici la classe `SMC`
  tourne avec sa vraie saturation, et la colonne « ecart » dit ce que la
  non-linearite coute ou rapporte.

  MPC. Sa loi explicite sans contrainte active EST lineaire, et
  `nonlinear.mpc_lti_ss` la rend exactement. La classe `MPC` resout le MEME
  probleme — meme procede augmente, meme ponderation passe-bande, meme filtre
  de Kalman — mais avec le cout du probleme ECHANTILLONNE (sampled-data
  exact) au lieu de son equivalent continu. L'ecart entre les deux colonnes
  mesure donc ce passage-la.

  Cette phrase disait auparavant que l'ecart « mesure le bloqueur, pas la
  structure ». C'ETAIT FAUX : la classe penalisait la sortie BRUTE, sans
  ponderation, la ou la forme continue penalise la sortie filtree. Les deux
  n'etaient pas la meme loi a la discretisation pres, elles etaient deux lois
  differentes, et leur ecart n'aurait rien mesure d'interpretable. Le procede
  pondere vit desormais dans `nonlinear._weighted_plant`, appele par les
  deux.

Toutes les autres passent par `LTIController`, donc echantillonnees et
saturees comme les deux precedentes : personne n'est mesure dans un regime
que les autres n'ont pas.

    PROTOCOL=B CALIB=measured python run_time_compare.py
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
from plate_model import build_plate, plant_vectors            # noqa: E402
from simulate import MillingSimulation                        # noqa: E402
from sim_controller import LTIController                      # noqa: E402
from nonlinear import SMC, MPC                                # noqa: E402
from hinf import plant_ss                                     # noqa: E402

OUT = os.path.join(HERE, '..', 'results')
# HORIZON. Il valait 0.5 s, et a cet horizon le critere ne mesurait PAS la
# stabilite : il mesurait si une croissance exponentielle atteint 5 mm avant
# la fin de la fenetre. Mesure a la limite alors PUBLIEE de la boucle ouverte
# (0.095215 mm) : log rho = +0.076 aux SIX positions, rho = 1.079, croissance
# x1.1e4 sur 0.5 s — et |y|max = 4897.3 um contre un seuil de 5000, donc
# « survivante ». A T = 1 s la meme coupe diverge. Le seuil etait franchi a
# t = 0.5014 s, 1.4 ms apres la fin de la fenetre.
#
# LE BIAIS DEPEND DE LA STRUCTURE, donc il faussait les COMPARAISONS et pas
# seulement l'echelle. Rapport (limite temporelle / limite de Floquet) a
# T = 0.5 s : boucle ouverte 2.59, smc 1.63, nmpdob 1.48, fopid 1.46,
# adrc 1.19, fdob 1.16, musyn 1.16, dvf 1.13, vpa 1.05, hinf 1.05,
# lqg 0.93, mpc 0.93. Le gain du FOPID sur la boucle ouverte passait de
# x3.08 a x3.71 en changeant ce seul nombre de 0.5 a 1 s.
#
# Le surplus au-dessus de la limite de Floquet suit 1/T (mesure : 0.05851,
# 0.03044, 0.01579 mm a T = 0.5, 1, 2 s ; rapports 1.922 et 1.928). D'ou
# T = 4.1 s pour 20 % de surplus, 8.3 s pour 10 %, 16.6 s pour 5 %. On prend
# 4 s : le surplus residuel d'environ 20 % est ANNONCE, il n'est plus
# silencieux, et le cout mesure reste tenable (la boucle ouverte passe de
# 110 s a environ 700 s).
#
# CE QUE CE CRITERE MESURE VRAIMENT, une fois l'horizon assaini : la survie
# GRAND SIGNAL sous saturation reelle. Il ne remplace pas Floquet et ne doit
# pas etre lu comme une limite de stabilite.
T_SIM = 4.0              # duree simulee a chaque position [s]
AP_HI = 2.5e-3           # borne haute de la dichotomie
RTOL = 0.03              # tolerance relative de la dichotomie

ORDER = ['fopid', 'adrc', 'fdob', 'fdob12345', 'dvf', 'vpa', 'hinf', 'musyn',
         'lqg', 'mpc', 'smc', 'nmpdob']


def discover():
    """{kind: {champ: valeur}} depuis le fichier fusionne puis les partiels."""
    found = {}
    merged = os.path.join(OUT, f'pso_{C.PROTOCOL}.npz')
    paths = ([merged] if os.path.exists(merged) else []) + sorted(
        glob.glob(os.path.join(OUT, f'pso_{C.PROTOCOL}_*.npz')))
    for path in paths:
        d = np.load(path, allow_pickle=True)
        for k in d.files:
            kind, _, field = k.partition('__')
            found.setdefault(kind, {}).setdefault(field, d[k])
    rank = {k: i for i, k in enumerate(ORDER)}
    return {k: found[k] for k in sorted(
        found, key=lambda k: (rank.get(k, len(ORDER)), k))}


def survives(plate, make_ctrl, ap, positions):
    """La passe tient-elle a TOUTES les positions d'outil ?"""
    for fr in positions:
        sim = MillingSimulation(plate, C.RPM_DESIGN, ap, n_modes=C.N_MODES,
                                n_sub=C.N_SUB, sign=C.SIGN_SIM,
                                v_max=C.V_MAX)
        if bool(sim.run(controller=make_ctrl(sim), T=T_SIM, moving=False,
                        x0=fr * plate.lp)['diverged']):
            return False
    return True


def ap_limit(plate, make_ctrl, positions, lo=0.0, hi=AP_HI):
    if not survives(plate, make_ctrl, max(lo, 1e-5), positions):
        return 0.0
    if survives(plate, make_ctrl, hi, positions):
        return hi
    while hi - lo > RTOL * hi:
        mid = 0.5 * (lo + hi)
        lo, hi = ((mid, hi) if survives(plate, make_ctrl, mid, positions)
                  else (lo, mid))
    return lo


def _params(rec):
    if 'keys' not in rec:
        return None
    return dict(zip([str(x) for x in rec['keys']],
                    [float(v) for v in rec['values']]))


def main():
    t00 = time.time()
    store = discover()
    if not store:
        print('  aucun resultat d optimisation : lancer run_pso.py d abord')
        return 1
    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    sl = plant_vectors(plate, C.N_MODES_DESIGN)[4]
    positions = C.POSITIONS_DESIGN

    print('=' * 78)
    print(' ETALON TEMPOREL — le meme critere pour les douze structures')
    print('=' * 78)
    print(f'  critere : passe de {T_SIM:.2f} s a chaque position {positions},'
          f' saturation +/-{C.V_MAX:.0f} V,')
    print(f'            dichotomie relative a {RTOL * 100:.0f} %,'
          f' {C.N_MODES} modes, n_sub = {C.N_SUB}')
    print(f'  structures : {", ".join(store)}\n')

    res = {}
    cfgs = [('boucle ouverte', lambda s: None, 'LTI')]
    for kind, rec in store.items():
        if 'A' not in rec:
            continue
        ss = (rec['A'], rec['B'], rec['C'], rec['D'])
        cfgs.append((kind, (lambda z: lambda s: LTIController(z, s.dt))(ss),
                     'LTI'))
    # ... et les DEUX formes non lineaires, avec les memes parametres optimises
    if 'smc' in store and _params(store['smc']):
        p = _params(store['smc'])
        var = float(store['smc'].get('sign_variant', 1.0))

        def mk_smc(s, p=p, var=var):
            return SMC(p['lam'], p['k_s'], p['phi'], s.dt, sl * var,
                       v_max=C.V_MAX)
        cfgs.append(('smc_nonlin', mk_smc, 'non lineaire'))
    if 'mpc' in store and _params(store['mpc']):
        p = _params(store['mpc'])
        var = float(store['mpc'].get('sign_variant', 1.0))
        w, zt, Hv, D_obs, _ = plant_vectors(plate, C.N_MODES_DESIGN)
        P = plant_ss(w, zt, D_obs * Hv)

        def mk_mpc(s, p=p, var=var, P=P):
            # L'horizon optimise est en SECONDES ; la classe MPC compte en
            # PAS d'echantillonnage. La conversion se fait ici, une fois, et
            # au pas reel du simulateur — sinon les deux formes ne
            # decriraient pas le meme horizon physique.
            n = max(1, int(round(p['horizon'] / s.dt)))
            return MPC(P, n, p['q'], p['r'], p['v_meas'], s.dt,
                       w_proc=p['w_proc'], f_w=p['f_w'], sign_variant=var,
                       v_max=C.V_MAX)
        cfgs.append(('mpc_discret', mk_mpc, 'discret'))

    for name, mk, form in cfgs:
        t0 = time.time()
        L = ap_limit(plate, mk, positions)
        res[name] = L
        print(f'    {name:14s} {form:12s} a_p,lim = {L * 1e3:7.4f} mm'
              f'   ({time.time() - t0:.0f} s)', flush=True)

    print('\n' + '=' * 78)
    print('  RECAPITULATIF (a_p,lim temporel, mm) — classement decroissant')
    print('=' * 78)
    for name in sorted(res, key=lambda k: -res[k]):
        base = res.get('boucle ouverte', 0.0)
        gain = (f'x{res[name] / base:.2f}' if base > 0 else '-')
        print(f'    {name:14s} {res[name] * 1e3:8.4f}   {gain}')
    np.savez_compressed(os.path.join(OUT, f'time_compare_{C.PROTOCOL}.npz'),
                        names=np.array(list(res)),
                        limits=np.array([res[k] for k in res]))
    print(f'\n  -> time_compare_{C.PROTOCOL}.npz   ({time.time() - t00:.0f} s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
