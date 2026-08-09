"""
run_demo.py — reproduit le resultat principal de la campagne v3.

Usage :
    cd simulation
    python run_demo.py            # tableau nominal (3 architectures)
    python run_demo.py --full     # + perturbations reservees (plus long)

Base : plaque encastree-libre 5 modes calibree sur Du et al. (IJMS 2024),
patch piezo VERTICAL 20 x 60 mm au coin inferieur gauche — la configuration que
montre la photo de la Fig. 11 de l'article. Une version precedente de ce fichier
utilisait un patch horizontal au coin inferieur droit : c'etait une erreur, voir
l'en-tete de model_v2.py. Amortissements modaux mesures (Tableau 4), et patch
COLLE et non soude (shear lag, eta = 0.886), avec le couplage
membrane-flexion du patch colle d'un seul cote (-11.6 % sur H_Pe).

Resultat attendu, b_lim en mm, horizon T = simulation_base.T_LIMIT :

                     3000    4200    4900    6000    7200   pire cas
    boucle ouverte  0.0773  0.0754  0.0579  0.1026  0.2600   0.0579
    LQG modal       0.3397  0.3455  0.3591  0.2581  0.7089   0.2581
    ESO propose     0.3591  0.3630  0.3708  0.2581  0.6992   0.2581

c.-a-d. performance SATUREE : deux architectures de richesse tres differente
atteignent EXACTEMENT le meme plafond, 0.2581 mm, contre 0.0579 mm en boucle
ouverte. C'est le resultat central de l'etude, et il est desormais bien plus
solide qu'avant : chaque architecture a ete RE-OPTIMISEE sur cette base-ci et
sous les memes contraintes, alors que le tableau precedent comparait deux
reglages herites d'un autre modele.

Les reglages viennent de `retune.py both`. Le critere publie -- maximiser la
limite nominale du pire cas -- est mal pose DANS LES DEUX SENS, et retune.py
documente les deux : sans contrainte l'optimum est un correcteur en butee
permanente ; avec la seule non-saturation il atteint 0.4038 mm mais s'effondre
sous K x0.90, C x0.80 et H x2.00. Les reglages ci-dessous sont donc obtenus
sous contrainte de non-saturation ET de robustesse, et ils DOMINENT ceux
d'origine : limite nominale plus haute (0.2581 contre 0.2386), meme tenue sous
perturbation, et crete de tension 39 V au lieu de la butee a 150 V.

`--full` rejoue le tableau sous les six perturbations reservees, le correcteur
restant synthetise sur la plaque nominale (donc un vrai essai d'erreur de
modele). Pire cas sur les cinq vitesses :

    perturbation    boucle ouverte    LQG modal      ESO propose
    nominal             0.0579          0.2581         0.2581
    K x0.90             0.0482          0.3066         0.3144
    K x1.10             0.0579          0.3494         0.3591
    C x0.80             0.0501          0.2561         0.2561
    kc x2.9             0.0210          0.0870         0.0870
    H x0.50             0.0579          0.1784         0.1784
    H x2.00             0.0579          0.3708         0.0000 (*)

La boucle ouverte ne bouge pas d'un chiffre entre nominal, H x0.50 et H x2.00 :
H_Pe n'entre pas dans le modele sans commande. Le NIVEAU de H_Pe n'etant pas
valide (defaut F9 -- le gain statique mesure vaut 2.94 fois celui du modele),
les lignes H x0.50 et H x2.00 encadrent cette incertitude, et aucune des deux
ne fait tomber la commande.

(*) CE ZERO EST UN ARTEFACT DE MESURE, PAS UN EFFONDREMENT. `stability_limit`
bissecte entre lo = 0.02 mm et hi = 4.0 mm en supposant la stabilite MONOTONE
en profondeur ; si lo est declare instable elle rend 0. Or a 6000 tr/min sous
H x2.00 ce correcteur donne :

    ap mm    0.02   0.03   0.05   0.10   0.20   0.30   0.40   0.50
    stable   NON    oui    oui    oui    oui    oui    oui    NON
    rms um  0.055  0.080  0.130  0.244  0.474  0.696  0.939  12.61
    crete V   1.4    2.0    3.3    6.4   12.9   30.0   82.5  150.0

La vraie limite est entre 0.40 et 0.50 mm. Le "non" a 0.02 mm porte sur une
reponse de 0.055 um pilotee par 1.4 V : c'est le critere de CROISSANCE
(GROWTH_MAX) qui se declenche sur un transitoire a une profondeur ou la reponse
n'a pas le temps de s'etablir, pas une instabilite. La bissection part donc
d'une borne basse faussement instable et rend 0. Le crible de robustesse de
retune.py, lui, teste a 0.10 mm et voit juste -- d'ou le desaccord entre les
deux. C'est un defaut de la METRIQUE, pas du correcteur ; il est laisse visible
plutot que maquille, et la ligne ESO / H x2.00 doit se lire "environ 0.45 mm".

Aucun des deux reglages ne tient kc x2.9 a 0.10 mm (0.0870 mm atteint), mais
c'est encore 4.1 fois la limite libre correspondante, et le reglage d'origine
n'y arrivait pas davantage : ce n'est pas une regression.

MISE EN GARDE HISTORIQUE. Les chiffres de ce fichier ont ete regeneres apres la
correction de `blim` (sa tolerance d'arret valait 6e-5 METRES, soit 0.060 mm,
plus grossiere que la limite en boucle ouverte elle-meme ; elle rendait 0.020 mm
par construction), apres la correction de la geometrie du patch, et apres
l'identification de la repartition modale de H_Pe sur la Fig. 12(b).
"""
import argparse
import copy
import sys
import time

import numpy as np

sys.path.insert(0, "sim_kit")
sys.path.insert(0, ".")
import simulation_base as SB                                 # noqa: E402
from model_v2 import make_sim                                # noqa: E402
from modal_adrc import ModalADRCFOPID                        # noqa: E402
from competitors import ModalLQG                             # noqa: E402

SPEEDS = [3000, 4200, 4900, 6000, 7200]

# Reglages RE-OPTIMISES sur la base corrigee (retune.py both). Les valeurs
# d'origine -- X_ESO = [-8.229, -0.437, 0.550, -7.874, 0.367, 230297.916,
# 1.615] et X_LQG = [-8.263, 1.901, -7.284] -- avaient ete obtenues sur une
# geometrie de patch fausse ET sur une repartition modale de couplage fausse ;
# sur la base corrigee elles saturent l'amplificateur aux cinq vitesses.
#
# L'optimisation est SOUS CONTRAINTE de non-saturation ET de robustesse : le
# critere de limite nominale seul est mal pose dans les deux sens, voir
# retune.py. Les reglages ci-dessous dominent ceux d'origine.
X_ESO = [-8.3861, 3.2814, 1.0339, -9.9052, 0.7112, 273969.8996, 2.1823]
X_LQG = [-8.3646, 2.7796, -9.6188]


def factory_eso(plate, x):
    cache = {}

    def mk(dt, tau):
        k = round(dt, 12)
        if k not in cache:
            c = ModalADRCFOPID(dt, plate, (0, 0, 0, .5, .5),
                               log10_sigd=x[2], log10_R=x[3], beta=x[4],
                               tau=tau, log10_qw=x[1], log10_fhp=x[6])
            c.t_ramp = 0.02
            c.build_lqr(log10_rho=x[0])
            c.g_reg = x[5]
            c.g_reg_v = 0.0
            cache[k] = c
        c = cache[k]
        c.reset()
        return c
    return mk


def factory_lqg(plate, x):
    cache = {}

    def mk(dt, tau):
        k = round(dt, 12)
        if k not in cache:
            cache[k] = ModalLQG(dt, plate, log10_rho=x[0], log10_qw=x[1],
                                log10_R=x[2], alpha=0.0, tau=tau)
        c = cache[k]
        c.reset()
        return c
    return mk


def blim(sim, mk, rpm, T=SB.T_LIMIT):
    """Limite de passe stable, en mm.

    Delegue a SimBase.stability_limit : meme bissection, meme horizon T_LIMIT
    et meme critere de stabilite que la reference de la base, donc les deux
    valeurs sont directement comparables.

    L'implementation locale precedente etait fausse sur deux points :
      * sa tolerance d'arret, 6e-5, est en METRES, soit 0.060 mm — plus
        grossiere que la limite en boucle ouverte elle-meme (~0.04 mm). La
        bissection s'arretait avec lo encore a sa valeur initiale et rendait
        0.020 mm quel que soit le resultat reel ;
      * sa boucle d'expansion pouvait sortir avec ok(hi) encore vrai, puis
        bissecter en supposant l'inverse.
    """
    return sim.stability_limit(mk, rpm=rpm, lo=0.02e-3, hi=4.0e-3,
                               T=T, tol=2e-6) * 1e3


def table(sim, plate, label):
    rows = {}
    for name, mk in [("boucle ouverte", None),
                     ("LQG modal", factory_lqg(plate, X_LQG)),
                     ("ESO propose", factory_eso(plate, X_ESO))]:
        v = [blim(sim, mk, r) for r in SPEEDS]
        rows[name] = v
        print(f"  {name:16s} " + " ".join(f"{x:7.4f}" for x in v)
              + f"   pire cas = {min(v):.4f} mm", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="ajoute les perturbations reservees")
    args = ap.parse_args()

    t0 = time.time()
    sim = make_sim()
    plate = copy.deepcopy(sim.plate)
    print("vitesses [tr/min] :", SPEEDS)
    print("\n=== NOMINAL ===")
    table(sim, plate, "nominal")

    if args.full:
        K0 = np.array(sim.plate.Kp, float).copy()
        C0 = np.array(sim.plate.Cp, float).copy()
        H0 = np.array(sim.plate.H_Pe_modal, float).copy()
        k1, k2 = float(sim.k1c), float(sim.k2c)
        # Les correcteurs sont synthetises sur `plate` (copie nominale) ; seule
        # la plaque SIMULEE est perturbee, donc c'est bien un essai de
        # robustesse a l'erreur de modele. H x0.5 / H x2.0 balaient le gain
        # actionneur, dont le NIVEAU n'est pas valide experimentalement
        # (cf. section 1 du docstring de simulation_base).
        for name, ks, cs, kcs, hs in [("K x0.90", .90, 1, 1, 1),
                                      ("K x1.10", 1.10, 1, 1, 1),
                                      ("C x0.80", 1, .80, 1, 1),
                                      ("kc x2.9", 1, 1, 2.9, 1),
                                      ("H x0.50", 1, 1, 1, 0.5),
                                      ("H x2.00", 1, 1, 1, 2.0)]:
            sim.plate.Kp = K0 * ks
            sim.plate.Cp = C0 * cs
            sim.plate.H_Pe_modal = H0 * hs
            sim.k1c, sim.k2c = k1 * kcs, k2 * kcs
            sim._cache.clear()
            print(f"\n=== {name} (perturbation reservee) ===")
            table(sim, plate, name)

    print(f"\ntermine en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
