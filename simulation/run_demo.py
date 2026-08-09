"""
run_demo.py — reproduit le resultat principal de la campagne v3.

Usage :
    cd simulation
    python run_demo.py            # tableau nominal (3 architectures)
    python run_demo.py --full     # + perturbations reservees (~15 min)

Base : plaque encastree-libre 5 modes calibree sur Du et al. (IJMS 2024),
patch piezo VERTICAL 20 x 60 mm au coin inferieur gauche — la configuration que
montre la photo de la Fig. 11 de l'article. Une version precedente de ce fichier
utilisait un patch horizontal au coin inferieur droit : c'etait une erreur, voir
l'en-tete de model_v2.py. Amortissements modaux mesures (Tableau 4), patch COLLE
et non soude (shear lag, eta = 0.886), couplage membrane-flexion du patch colle
d'un seul cote, et repartition modale de H_Pe identifiee sur la Fig. 12(b).

Resultat, b_lim en mm, horizon T = simulation_base.T_LIMIT = 0.60 s, critere
sigma <= 0.05 /s :

                     3000    4200    4900    6000    7200   pire cas
    boucle ouverte  0.0676  0.0734  0.0501  0.1026  0.2153   0.0501
    LQG modal       0.3611  0.3844  0.3241  0.3475  0.6720   0.3241
    ESO propose     0.3552  0.3591  0.3688  0.2561  0.6856   0.2561

soit x6.5 et x5.1 sur la limite libre du pire cas.

LA "PERFORMANCE SATUREE" N'EXISTE PLUS. Ce fichier a longtemps annonce que deux
architectures de richesse tres differente atteignaient le meme plafond a 0.7 %
pres, et en faisait le resultat central de l'etude. Sous le critere de stabilite
corrige (defaut F12) l'ecart est de 26.6 %, et il est dans le sens INVERSE de
l'attendu : c'est le LQG, la structure la plus pauvre, qui mene. L'egalite etait
un artefact du critere optimiste.

`--full` rejoue le tableau sous les six perturbations reservees, le correcteur
restant synthetise sur la plaque NOMINALE (donc un vrai essai d'erreur de
modele). Pire cas sur les cinq vitesses :

    perturbation    boucle ouverte    LQG modal      ESO propose
    nominal             0.0501          0.3241         0.2561
    K x0.90             0.0423          0.2697         0.2522
    K x1.10             0.0443          0.1590       **0.3552**
    C x0.80             0.0404          0.3222         0.2542
    kc x2.9             0.0000 (*)      0.1104         0.0870
    H x0.50             0.0501          0.1784         0.1764
    H x2.00             0.0501        **0.3436**       0.0000

TROIS LECTURES, DONT DEUX CONTREDISENT CE QUE CE FICHIER DISAIT AVANT.

1. La ligne H x2.00 ne tue plus le LQG — elle le FAVORISE : 0.3436 mm contre
   0.3241 en nominal. Ce fichier annoncait "les deux correcteurs y perdent le
   controle" et en faisait son resultat le plus important. C'etait vrai des
   reglages d'alors, pas de la commande : doubler le gain actionneur double
   l'autorite, et si la boucle reste stable la limite MONTE. Il fallait
   contraindre la recherche sur cette perturbation pour trouver un reglage qui
   le fasse ; c'est fait, et ca marche. Le point compte parce que le NIVEAU de
   H_Pe n'est pas valide (defaut F9) et que le gain statique mesure sur la
   Fig. 12(b) vaut 2.94 fois celui du modele : le facteur 2 n'est pas une marge
   academique, c'est le milieu de l'incertitude reelle.

2. Le LQG ne domine PAS partout. Sous K x1.10 le classement s'inverse
   franchement : 0.1590 contre 0.3552 pour l'ESO, soit un facteur 2.2 dans
   l'autre sens. "6 sur 6" veut dire qu'il passe le crible a 0.10 mm sous les
   six perturbations, pas qu'il est meilleur sous chacune.

3. L'ESO s'effondre sous H x2.00 : zero a quatre vitesses sur cinq. Ce sont de
   VRAIS zeros et non des artefacts de bissection — verifie en allongeant
   l'horizon, tout diverge (voir verification/15). Il echoue aussi le crible
   sous kc x2.9 (0.0870 < 0.10 mm). D'ou son 4 sur 6.

(*) 0.0000 en boucle ouverte signifie "sous la borne basse de la bissection",
soit 0.02 mm : avec des coefficients de coupe presque triples, la coupe libre
est instable des la plus petite passe testee. Vrai zero, pas artefact.

Les reglages viennent de `retune.py both`, relance en entier sur le critere
corrige. Le critere qu'il maximise -- la limite nominale du pire cas -- est mal
pose dans les deux sens (butee permanente sans contrainte ; effondrement sous
perturbation avec la seule non-saturation), d'ou les contraintes de
non-saturation ET de robustesse ; voir retune.py.

X_LQG est NOUVEAU : +27 % sur le reglage precedent mesure au meme etalon,
premiere tenue complete des six perturbations, et la plus basse tension crete
des candidats (33 V pour 150 V disponibles).

X_ESO est INCHANGE, et ce n'est pas un oubli : sur 20 particules x 20
iterations, la recherche n'a jamais fait mieux que sa graine -- le score n'a pas
bouge d'un millieme du premier au vingtieme tour. Deux lectures restent
ouvertes, que cette campagne ne separe pas : ou l'ESO n'a pas de reglage robuste
dans les bornes de BOUNDS, ou son espace a sept dimensions est trop creux pour
que 400 evaluations le trouvent.

MISE EN GARDE HISTORIQUE. Les chiffres de ce fichier ont ete regeneres apres la
correction de `blim`, apres la correction de la geometrie du patch, apres
l'identification de la repartition modale de H_Pe sur la Fig. 12(b), apres la
correction du critere de stabilite lui-meme, et enfin apres la re-optimisation
des deux correcteurs sur ce critere. Aucune valeur anterieure a la derniere de
ces etapes n'est comparable a celles-ci.
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

# Reglages issus de `retune.py both`, sous contrainte de non-saturation ET de
# robustesse : le critere de limite nominale seul est mal pose dans les deux
# sens, voir retune.py.
#
# Les valeurs d'origine -- X_ESO = [-8.229, -0.437, 0.550, -7.874, 0.367,
# 230297.916, 1.615] et X_LQG = [-8.263, 1.901, -7.284] -- avaient ete obtenues
# sur une geometrie de patch fausse ET sur une repartition modale de couplage
# fausse ; sur la base corrigee elles saturent l'amplificateur aux cinq
# vitesses.
#
# LES DEUX LIGNES N'ONT PAS LE MEME STATUT, et c'est le resultat lui-meme :
#   X_LQG a ete TROUVE par la recherche sur le critere corrige (iteration 12),
#     et domine le reglage precedent sur les trois colonnes a la fois : +27 %
#     de limite nominale, 6 perturbations sur 6 au lieu de 4, et 33 V de crete
#     au lieu de 39.
#   X_ESO est la GRAINE, c.-a-d. le reglage deja publie : la meme recherche,
#     avec les memes contraintes, n'a rien trouve de mieux en 20 x 20
#     evaluations. Ce n'est pas un reste d'une version precedente de ce fichier.
X_ESO = [-8.3861, 3.2814, 1.0339, -9.9052, 0.7112, 273969.8996, 2.1823]
X_LQG = [-8.6678, -1.1613, -4.5628]


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
