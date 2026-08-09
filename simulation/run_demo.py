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

Resultat attendu, b_lim en mm, horizon T = simulation_base.T_LIMIT = 0.60 s :

                     3000    4200    4900    6000    7200   pire cas
    boucle ouverte  0.0676  0.0734  0.0501  0.1026  0.2153   0.0501
    LQG modal       0.3533  0.3416  0.3572  0.2542  0.6934   0.2542
    ESO propose     0.3552  0.3591  0.3688  0.2561  0.6856   0.2561

c.-a-d. performance SATUREE : deux architectures de richesse tres differente
atteignent le meme plafond a 0.7 % pres, contre 0.0501 mm en boucle ouverte.
C'est le resultat central de l'etude, et il survit a tout ce qui precede.

TOUS CES CHIFFRES ONT BAISSE D'ENVIRON 19 % avec la correction du critere de
stabilite (defaut F12). L'ancien critere -- rapport de RMS entre les deux
moities de la fenetre -- declarait stables des coupes qui divergent plus tard :
sur 144 cas dont la verite a ete etablie a T = 1.60 s, il se trompait 8 fois, et
TOUJOURS en declarant stable ce qui diverge. Le critere actuel, un taux de
croissance exponentiel sigma <= 0.05 /s mesure sur la reponse ETABLIE, se
trompe 0 fois sur les memes 144 cas. Toute valeur anterieure a cette correction
est optimiste et n'est PAS comparable a celles-ci. Voir verification/15.

`--full` rejoue le tableau sous les six perturbations reservees, le correcteur
restant synthetise sur la plaque nominale (donc un vrai essai d'erreur de
modele). Pire cas sur les cinq vitesses :

    perturbation    boucle ouverte    LQG modal      ESO propose
    nominal             0.0501          0.2542         0.2561
    K x0.90             0.0423          0.2503         0.2522
    K x1.10             0.0443          0.3455         0.3552
    C x0.80             0.0404          0.2542         0.2542
    kc x2.9             0.0000 (*)      0.0870         0.0870
    H x0.50             0.0501          0.1764         0.1764
    H x2.00             0.0501          0.0000         0.0000

LA LIGNE H x2.00 EST LE RESULTAT LE PLUS IMPORTANT DE CE TABLEAU. Les deux
correcteurs y perdent le controle. Or le NIVEAU de H_Pe n'est pas valide
(defaut F9) et le gain statique mesure sur la Fig. 12(b) vaut 2.94 fois celui du
modele : le facteur 2 n'est donc pas une marge academique, c'est le milieu de
l'incertitude reelle. Sous le critere precedent cette ligne affichait 0.3708 mm
pour le LQG -- une valeur rassurante et fausse.

Il faut le dire clairement : ces correcteurs n'ont pas de marge de gain
actionneur, et ce sont les memes reglages qu'avant. Ce n'est pas la correction
du critere qui les a degrades, c'est elle qui a cesse de le cacher.

(*) 0.0000 signifie "en dessous de la borne basse de la bissection", soit
0.02 mm : sous kc x2.9 la coupe libre est instable des la plus petite passe
testee. C'est un vrai zero, pas un artefact.

kc x2.9 reste hors d'atteinte pour les deux correcteurs a la profondeur de
crible de retune.py (0.10 mm) ; ils y tiennent 0.0870 mm, ce qui reste tres
au-dessus de la limite libre correspondante.

Les reglages viennent de `retune.py both`. Le critere qu'il maximise -- la
limite nominale du pire cas -- est mal pose dans les deux sens (butee
permanente sans contrainte ; effondrement sous perturbation avec la seule
non-saturation), d'ou les contraintes de non-saturation ET de robustesse ; voir
retune.py. Sous le critere corrige, la re-optimisation ne trouve rien de mieux
que ces reglages-la, et le crible de robustesse n'est franchi que par 2 des 6
perturbations : le "5 sur 6" annonce precedemment etait lui aussi un artefact du
critere optimiste.

MISE EN GARDE HISTORIQUE. Les chiffres de ce fichier ont ete regeneres apres la
correction de `blim`, apres la correction de la geometrie du patch, apres
l'identification de la repartition modale de H_Pe sur la Fig. 12(b), et enfin
apres la correction du critere de stabilite lui-meme.
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
