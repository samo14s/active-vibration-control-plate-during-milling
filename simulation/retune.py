"""
retune.py — re-optimise les correcteurs sur la base CORRIGEE.

Pourquoi. Les reglages X_LQG / X_ESO de run_demo.py avaient ete obtenus sur une
geometrie de patch fausse (horizontale, coin inferieur droit). Depuis, la
geometrie a ete corrigee (verticale, coin inferieur gauche, confirmee par la
Fig. 11 de l'article), la couche de colle et le couplage membrane-flexion ont
ete modelises, et l'autorite actionneur a baisse d'environ 21 % au total.

Puis, plus radicalement, la REPARTITION MODALE du couplage a ete identifiee sur
la Fig. 12(b) (constante H_IDENT) : elle etait structurellement fausse, le
modele elements finis ne produisant qu'un seul des quatre creux mesures. Un
correcteur regle sur l'ancienne repartition n'est plus regle du tout — le
reglage X_LQG d'origine SATURE meme l'amplificateur a toutes les vitesses sur la
base corrigee, donc il echoue d'emblee la contrainte ci-dessous.

Les comparer en l'etat mesure ce que CES reglages-la donnent sur un autre
modele, pas ce que chaque architecture peut atteindre : la comparaison
d'architectures n'a de sens qu'apres re-optimisation de chacune sur la meme
base.

Critere. On maximise la LIMITE DE PASSE DU PIRE CAS sur les cinq vitesses,
exactement la grandeur que run_demo publie :

    J(x) = min_rpm  SimBase.stability_limit(controleur(x), rpm)

SOUS CONTRAINTE de non-saturation au point de fonctionnement nominal.

Pourquoi la contrainte. Maximiser J seul est un critere MAL POSE, et on l'a
verifie : l'optimum sans contrainte est un correcteur en butee permanente
(peak_u = 150.00 V a toutes les vitesses, 120 V RMS contre 5 V pour le reglage
d'origine) qui gagne 34 % de limite tout en MULTIPLIANT PAR 3.5 l'amplitude
vibratoire reelle (2.07 um contre 0.59 um). Le critere de stabilite de
la base est un critere de CROISSANCE : une reponse grande mais bornee le
satisfait. Un correcteur constamment sur la butee n'est plus lineaire, et la
"limite de stabilite" qu'il affiche est un artefact.

On impose donc, au point nominal (ap = AP_TEST, horizon T_RUN) et a chaque
vitesse : coupe stable ET peak_u < V_MAX. La contrainte est physique -- le banc
ne delivre pas plus de +/- 150 V -- et elle est evaluee EN PREMIER, car elle
coute 5 simulations courtes la ou la bissection en coute 40 a 55.

Deuxieme facon dont le critere est mal pose, et elle aussi verifiee. Avec la
seule contrainte de saturation, l'optimum obtenu sur la base a couplage
identifie donne 0.4038 mm (LQG) et 0.3902 mm (ESO) au nominal, mais s'EFFONDRE
sous une erreur de raideur de -10 % : limite nulle a 6000 et 7200 tr/min, alors
que les reglages d'origine y tenaient encore 0.279 et 0.482 mm. Maximiser la
limite nominale achete donc de la performance en depensant de la marge de
robustesse -- exactement le travers que run_demo --full est cense debusquer.

On ajoute donc une contrainte de ROBUSTESSE : le correcteur doit rester stable
a AP_GATE = 0.10 mm sous les SIX perturbations reservees de run_demo --full
(K x0.90, K x1.10, C x0.80, kc x2.9, H x0.44, H x2.92), le correcteur restant
synthetise sur la plaque NOMINALE. C'est un vrai essai d'erreur de modele.

Les deux bornes de gain actionneur ne sont pas rondes parce qu'elles ne sont
pas choisies : ce sont les deux bouts de l'encadrement MESURE de F9, voir
verification/17_hpe_level.py. Elles valaient 0.50 et 2.00, ce qui etait plus
etroit que l'incertitude reelle du cote qui fait mal.

Deux details qui ont chacun coute un resultat faux :

  * l'horizon du crible doit etre T_LIMIT, pas T_RUN. Le critere de stabilite
    de la base DEPEND de l'horizon -- une instabilite lente a besoin de temps
    pour se voir -- donc cribler a T_RUN est plus indulgent que la grandeur
    publiee. Un reglage ESO a ainsi passe le crible puis rendu 0.0000 mm sous
    H x2.00 dans run_demo --full ;

  * un candidat INFAISABLE ne rend pas 0 mais n/1000, ou n est le nombre de
    conditions franchies. Sans cela le critere est plat a zero sur presque tout
    l'espace et le PSO n'a aucun gradient : avec la seule valeur 0, la
    recherche LQG ne sortait jamais de son point de depart.

--constraint sat  n'impose que la non-saturation (reproduit l'effondrement).
--constraint none restitue le critere sans contrainte, pour reproduire le
resultat degenere en butee permanente.

Ce que cela donne pour le LQG modal, sur la base a couplage identifie ET sous
le critere de stabilite corrige (F12) :

    reglage                       pire cas nominal   perturbations   crete V
    d'origine (publie a l'origine)     0.2386 mm       -- sature --   150
    optimum --constraint sat           0.4038 mm        3 sur 6        56
    reglage precedent, mesure a neuf   0.2542 mm        4 sur 6        39
    optimum --constraint rob           0.3241 mm      **6 sur 6**      33

Le dernier DOMINE tous les autres : meilleure limite nominale que le reglage
precedent (+27 %), la premiere tenue COMPLETE sous les six perturbations
reservees, et la tension crete la plus basse des quatre. C'est celui que
run_demo publie.

Un detail d'implementation a decide de ce resultat : l'ORDRE de PERTURB. Avec
les plus meurtrieres en tete, tout candidat echouait des la premiere, tous
rendaient 1/1000, et la recherche restait collee a sa graine -- c'est ce qu'on
observait avant. Du plus facile au plus dur, le score compte reellement combien
de perturbations le candidat encaisse ; la recherche a progresse 4 -> 5 -> 6 et
a franchi le seuil de faisabilite a la 12e iteration. Le critere n'a pas change,
seule sa lisibilite par l'optimiseur.

ET L'ESO, LUI, N'A RIEN TROUVE. Meme recherche, memes contraintes, meme ordre
de PERTURB, 20 particules x 20 iterations : le meilleur reste la GRAINE, c.-a-d.
le reglage deja publie, a 4 sur 6 (echecs kc x2.9 et H x2.92). Le score n'a pas
bouge d'un millieme du premier au vingtieme tour.

    architecture   reglage retenu        pire cas nominal   perturbations
    LQG modal      nouveau (iteration 12)    0.3241 mm         6 sur 6
    ESO propose    la graine, inchangee      0.2561 mm         4 sur 6

RESERVE SUR CES DEUX LIGNES. Elles ont ete obtenues avec le crible d'AVANT
l'elargissement, c.-a-d. H x0.50 / H x2.00 au lieu de H x0.44 / H x2.92. Le
tableau de run_demo --full a ete refait sur le crible elargi et les deux
verdicts TIENNENT -- LQG 6 sur 6 (marge la plus faible 0.1104 mm sous kc x2.9),
ESO 4 sur 6 -- mais la RECHERCHE, elle, n'a pas ete relancee sur le crible
elargi. Les vecteurs publies restent donc valides ; leur OPTIMALITE, elle, se
rapporte a l'ancien crible.

Ce n'est pas un echec de la recherche, c'est un RESULTAT, et il contredit
l'attente naturelle : la memoire de l'article donnerait l'avantage a
l'architecture la plus riche. Sous le critere corrige c'est l'inverse -- le
LQG, plus pauvre, domine sur les deux colonnes a la fois. Deux lectures
restent ouvertes et ce script ne les separe pas : ou l'ESO n'a pas de reglage
robuste dans les bornes de BOUNDS, ou son espace a sept dimensions est trop
creux pour que 400 evaluations le trouvent. La seconde se testerait en
elargissant la recherche ; la premiere, en elargissant les bornes.

Methode. PSO, meme famille que l'optimisation d'origine. Le reglage courant est
injecte comme particule 0, ce qui garantit de ne pas faire pire que l'existant.
Les particules sont evaluees en parallele (une par coeur).

Usage :
    python retune.py lqg            # ~7 min
    python retune.py eso            # ~15 min
    python retune.py both
    python retune.py both --constraint sat    # sans la contrainte de robustesse
"""
import argparse
import copy
import multiprocessing as mp
import sys
import time

import numpy as np

sys.path.insert(0, "sim_kit")
sys.path.insert(0, ".")
import simulation_base as SB                                    # noqa: E402
import run_demo as RD                                           # noqa: E402
from model_v2 import make_sim                                   # noqa: E402

SPEEDS = RD.SPEEDS
TOL_SEARCH = 5e-5      # 0.05 mm : suffisant pour classer des candidats
TOL_FINAL = 2e-6       # 0.002 mm : pour la valeur publiee

# (nom, bornes) — bornes centrees genereusement sur le reglage actuel
BOUNDS = {
    'lqg': np.array([(-12.0, -4.0),     # log10_rho
                     (-2.0, 5.0),       # log10_qw
                     (-12.0, -4.0)]),   # log10_R
    'eso': np.array([(-12.0, -4.0),     # log10_rho
                     (-4.0, 4.0),       # log10_qw
                     (-4.0, 6.0),       # log10_sigd
                     (-12.0, -4.0),     # log10_R
                     (0.0, 2.0),        # beta
                     (0.0, 1.0e6),      # g_reg
                     (1.0, 4.0)]),      # log10_fhp
}
SEED = {'lqg': np.array(RD.X_LQG, float), 'eso': np.array(RD.X_ESO, float)}
FACTORY = {'lqg': RD.factory_lqg, 'eso': RD.factory_eso}

_SIM = None
_PLATE = None
_K0 = None
_C0 = None
_H0 = None
_KC0 = None

# Perturbations reservees, identiques a celles de run_demo --full.
# (nom, facteur sur Kp, sur Cp, sur les coefficients de coupe, sur H_Pe)
# Ordre : de la PLUS FACILE a la plus dure, la boucle sortant au premier echec.
#
# L'ordre inverse -- les plus meurtrieres d'abord -- minimise le cout, et c'est
# ce qu'il faisait tant que le critere de stabilite etait optimiste. Sous le
# critere corrige il DETRUIT le gradient : presque tout candidat echoue des la
# premiere perturbation, tous rendent donc 1/1000, et le PSO n'a plus rien a
# suivre -- verifie, la recherche restait collee a sa graine. Du plus facile au
# plus dur, le score n compte reellement combien de perturbations le candidat
# encaisse avant de ceder, et la recherche progresse.
#
# L'ordre n'est PAS choisi a vue : il suit la marge mesuree par
# run_demo --full, prise au pire des deux architectures (limite du pire cas,
# mm) -- c'est la difficulte reelle, pas une intuition.
#
#     C x0.80  0.2542     K x0.90  0.2522     H x0.44  0.1628
#     K x1.10  0.1590     kc x2.9  0.0870     H x2.92  0.0000
#
# Il a fallu le refaire en elargissant le balayage de gain actionneur a
# [0.44, 2.92] : H x0.50 etait la perturbation la plus DOUCE et ouvrait la
# liste, H x0.44 est la troisieme plus dure. La laisser en tete aurait
# reintroduit exactement la pathologie decrite ci-dessus.
#
# RESERVE, et elle est reelle : la difficulte DEPEND DE L'ARCHITECTURE. H x2.92
# est la perturbation la plus douce pour le LQG (0.3669 mm, mieux qu'au
# nominal) et elle est LETALE pour l'ESO (zero aux cinq vitesses). Aucun ordre
# unique ne peut donc etre optimal pour les deux, et celui-ci -- fonde sur le
# pire des deux -- protege le gradient de l'architecture la plus fragile.
# Le remede de fond serait d'evaluer les six conditions au lieu de sortir a la
# premiere, rendant le score independant de l'ordre ; il coute ~30 simulations
# courtes par candidat au lieu de ~10, contre 200 a 275 pour une bissection.
PERTURB = [("C x0.80", 1.0, 0.80, 1.0, 1.0),
           ("K x0.90", 0.90, 1.0, 1.0, 1.0),
           ("H x0.44", 1.0, 1.0, 1.0, 0.44),
           ("K x1.10", 1.10, 1.0, 1.0, 1.0),
           ("kc x2.9", 1.0, 1.0, 2.9, 1.0),
           ("H x2.92", 1.0, 1.0, 1.0, 2.92)]

# Profondeur a laquelle on exige la survie SOUS PERTURBATION. AP_TEST (0.25 mm)
# serait absurde : sous kc x2.9 meme la meilleure commande ne tient que 0.14 mm.
# 0.10 mm est environ le double de la limite libre nominale (0.058 mm) et reste
# au-dessus de la limite libre de chaque plaque perturbee.
AP_GATE = 0.10e-3


def _init_worker():
    global _SIM, _PLATE, _K0, _C0, _KC0, _H0
    _SIM = make_sim()
    _PLATE = copy.deepcopy(_SIM.plate)
    _K0 = np.array(_SIM.plate.Kp, float).copy()
    _C0 = np.array(_SIM.plate.Cp, float).copy()
    _H0 = np.array(_SIM.plate.H_Pe_modal, float).copy()
    _KC0 = (float(_SIM.k1c), float(_SIM.k2c))


def _set_plant(ks=1.0, cs=1.0, kcs=1.0, hs=1.0):
    """Perturbe la plaque SIMULEE. Le correcteur, lui, reste synthetise sur la
    plaque nominale (_PLATE) : c'est donc bien un essai d'erreur de modele."""
    _SIM.plate.Kp = _K0*ks
    _SIM.plate.Cp = _C0*cs
    _SIM.plate.H_Pe_modal = _H0*hs
    _SIM.k1c, _SIM.k2c = _KC0[0]*kcs, _KC0[1]*kcs
    _SIM._cache.clear()


def _survives(mk, ks, cs, kcs, hs, ap, check_sat, T=None):
    """Le correcteur tient-il a la profondeur `ap` sur cette plaque-la ?

    T doit valoir T_LIMIT pour les essais de perturbation : c'est l'horizon
    auquel run_demo publie ses limites, et le critere de stabilite DEPEND de
    l'horizon (une instabilite lente a besoin de temps pour se voir). Une
    version precedente de ce fichier utilisait T_RUN ici : le crible etait donc
    plus indulgent que la grandeur publiee, et il laissait passer un reglage
    ESO qui rendait 0.0000 mm sous forte perturbation dans run_demo --full.
    """
    _set_plant(ks, cs, kcs, hs)
    T = SB.T_RUN if T is None else T
    try:
        for rpm in SPEEDS:
            tau = 60.0/(3*rpm)
            r = _SIM.run(mk(tau/82, tau), rpm=rpm, ap=ap, T=T)
            if not r['stable']:
                return False
            if check_sat and r['peak_u'] >= 0.999*SB.V_MAX:
                return False
    except Exception:
        return False
    finally:
        _set_plant()
    return True


def _worst_blim(arch, x, tol, cutoff=0.0, gate='rob'):
    """Limite de passe du pire cas, en mm. 0.0 si le correcteur echoue.

    cutoff : meilleure valeur connue. Des qu'une vitesse fait tomber le minimum
    courant en dessous, le candidat ne peut plus gagner et on s'arrete ; la
    valeur rendue est alors une BORNE SUPERIEURE, ce qui suffit au classement
    et divise le cout par ~3, la plupart des particules perdant.
    """
    try:
        mk = FACTORY[arch](_PLATE, list(x))
    except Exception:
        return 0.0

    # Contraintes evaluees d'abord : elles ne coutent que des simulations
    # courtes, la ou la bissection en coute 40 a 55.
    #
    # Un candidat INFAISABLE ne rend pas 0 mais n/1000, ou n est le nombre de
    # conditions franchies avant l'echec. C'est la gestion de contraintes
    # "faisabilite d'abord" : n/1000 <= 0.007 reste sous toute limite reelle
    # (>= 0.020 mm), donc tout candidat faisable bat tout candidat infaisable,
    # mais entre deux infaisables celui qui va plus loin gagne. Sans cela le
    # critere serait plat a zero sur presque tout l'espace et le PSO n'aurait
    # aucun gradient a suivre.
    if gate in ('sat', 'rob'):
        if not _survives(mk, 1.0, 1.0, 1.0, 1.0, SB.AP_TEST, check_sat=True):
            return 0.0
    if gate == 'rob':
        for n, (_, ks, cs, kcs, hs) in enumerate(PERTURB, start=1):
            if not _survives(mk, ks, cs, kcs, hs, AP_GATE, check_sat=False,
                             T=SB.T_LIMIT):
                return n/1000.0

    best = np.inf
    for rpm in SPEEDS:
        try:
            v = _SIM.stability_limit(mk, rpm=rpm, lo=0.02e-3, hi=1.5e-3,
                                     T=SB.T_LIMIT, tol=tol)*1e3
        except Exception:
            return 0.0
        if v < best:
            best = v
        if best <= max(cutoff, 0.021):
            return best
    return float(best)


def _eval(job):
    arch, x, tol, cutoff, gate = job
    return _worst_blim(arch, np.asarray(x), tol, cutoff, gate)


def pso(arch, n_part, n_iter, workers, rng, gate='rob'):
    lo, hi = BOUNDS[arch][:, 0], BOUNDS[arch][:, 1]
    d = len(lo)
    X = rng.uniform(lo, hi, size=(n_part, d))
    X[0] = np.clip(SEED[arch], lo, hi)          # reglage actuel comme graine
    V = rng.uniform(-1, 1, size=(n_part, d))*(hi - lo)*0.1
    with mp.Pool(workers, initializer=_init_worker) as pool:
        f = np.array(pool.map(
            _eval, [(arch, x, TOL_SEARCH, 0.0, gate) for x in X]))
        pbest, pbest_f = X.copy(), f.copy()
        g = int(np.argmax(f))
        gbest, gbest_f = X[g].copy(), f[g]
        print(f'  init   : meilleur = {gbest_f:.4f} mm '
              f'(graine = {f[0]:.4f} mm)', flush=True)
        w, c1, c2 = 0.72, 1.5, 1.5
        for it in range(n_iter):
            r1 = rng.random((n_part, d))
            r2 = rng.random((n_part, d))
            V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)
            V = np.clip(V, -(hi - lo)*0.25, (hi - lo)*0.25)
            X = np.clip(X + V, lo, hi)
            f = np.array(pool.map(
                _eval, [(arch, x, TOL_SEARCH, 0.97*gbest_f, gate)
                        for x in X]))
            imp = f > pbest_f
            pbest[imp], pbest_f[imp] = X[imp], f[imp]
            g = int(np.argmax(pbest_f))
            if pbest_f[g] > gbest_f:
                gbest, gbest_f = pbest[g].copy(), pbest_f[g]
            print(f'  iter {it+1:3d} : meilleur = {gbest_f:.4f} mm', flush=True)
    return gbest, gbest_f


def report(arch, x):
    """Evalue le reglage a la tolerance finale et rapporte la tension."""
    global _SIM, _PLATE, _K0, _C0, _H0, _KC0
    _init_worker()
    mk0 = FACTORY[arch](_PLATE, list(x))
    surv = [(nm, _survives(mk0, ks, cs, kcs, hs, AP_GATE, False,
                           T=SB.T_LIMIT))
            for nm, ks, cs, kcs, hs in PERTURB]
    sim = make_sim()
    plate = copy.deepcopy(sim.plate)
    mk = FACTORY[arch](plate, list(x))
    lim = [sim.stability_limit(mk, rpm=r, lo=0.02e-3, hi=4.0e-3,
                               T=SB.T_LIMIT, tol=TOL_FINAL)*1e3 for r in SPEEDS]
    u, pk, y = [], [], []
    for r in SPEEDS:
        tau = 60.0/(3*r)
        res = sim.run(mk(tau/82, tau), rpm=r, ap=SB.AP_TEST, T=SB.T_RUN)
        u.append(res['rms_u']); pk.append(res['peak_u']); y.append(res['rms_um'])
    return lim, u, pk, y, surv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arch', choices=['lqg', 'eso', 'both'])
    ap.add_argument('--particles', type=int, default=0)
    ap.add_argument('--iters', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--constraint', choices=['rob', 'sat', 'none'],
                    default='rob')
    a = ap.parse_args()
    archs = ['lqg', 'eso'] if a.arch == 'both' else [a.arch]
    rng = np.random.default_rng(20240809)
    t0 = time.time()
    for arch in archs:
        npart = a.particles or (12 if arch == 'lqg' else 20)
        nit = a.iters or (15 if arch == 'lqg' else 20)
        print(f'=== {arch.upper()} : PSO {npart} particules x {nit} iterations '
              f'sur {a.workers} coeurs ===', flush=True)
        x, fx = pso(arch, npart, nit, a.workers, rng, gate=a.constraint)
        lim, u, pk, y, surv = report(arch, x)
        print(f'\n  X_{arch.upper()} = [' + ', '.join(f'{v:.4f}' for v in x) + ']')
        print(f'  limites (mm)  : {[round(v, 4) for v in lim]}')
        print(f'  pire cas      : {min(lim):.4f} mm   (recherche : {fx:.4f})')
        print(f'  tension RMS V : {[round(v, 1) for v in u]}')
        print(f'  tension crete : {[round(v, 1) for v in pk]}   '
              f'(V_MAX = {SB.V_MAX})')
        print(f'  vibration um  : {[round(v, 3) for v in y]}')
        print(f'  survie sous perturbation reservee, a '
              f'{AP_GATE*1e3:.2f} mm :')
        for nm, ok in surv:
            print(f'     {nm:10s} : {"OK" if ok else "ECHEC"}')
        print(f'  ecoule        : {time.time()-t0:.0f} s\n', flush=True)


if __name__ == '__main__':
    main()
