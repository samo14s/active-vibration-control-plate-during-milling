"""
run_demo.py — reproduit le resultat principal de la campagne v3.

Usage :
    cd simulation
    python run_demo.py            # tableau nominal (3 architectures)
    python run_demo.py --full     # + perturbations reservees (plus long)

Base : plaque encastree-libre 5 modes calibree sur Du et al. (IJMS 2024),
avec les DEUX corrections validees contre la Fig. 12 et le Tableau 4 :
patch piezo inferieur droit horizontal, et amortissements modaux mesures
(0.31 / 0.17 / 0.27 / 0.56 / 0.35 %). Le patch y est COLLE et non soude
(shear lag, eta = 0.886) : l'autorite actionneur est 11 % plus faible qu'avec
l'hypothese de collage parfait utilisee auparavant.

Resultat attendu, b_lim en mm, horizon T = simulation_base.T_LIMIT :

                     3000    4200    4900    6000    7200   pire cas
    boucle ouverte  0.0598  0.0598  0.0462  0.0929  0.2114   0.0462
    LQG modal       0.5612  0.6720  0.6020  0.5262  0.8838   0.5262
    ESO propose     0.5398  0.6351  0.5845  0.5262  0.8255   0.5262

c.-a-d. performance SATUREE : trois architectures de richesse croissante
atteignent le meme plafond — les pires cas du LQG et de l'ESO sont ici
IDENTIQUES (0.5262 mm, tous deux a 6000 tr/min). C'est le resultat central
de l'etude.

Chiffres regeneres apres correction de `blim` : sa tolerance d'arret valait
6e-5 METRES, soit 0.060 mm, plus grossiere que la limite en boucle ouverte
elle-meme ; elle rendait 0.020 mm par construction, et les valeurs annoncees
auparavant (0.049 / 0.515 / 0.515) n'etaient pas celles que le code produisait.
`blim` delegue desormais a SimBase.stability_limit, donc ces limites sont
directement comparables a la reference de la base.

`--full` ajoute, aux perturbations K / C / kc, un balayage du gain actionneur
`H x0.50` et `H x2.00` : le NIVEAU de H_Pe n'est pas valide experimentalement
(cf. section 1 du docstring de simulation_base). Pire cas sur les cinq vitesses :

    gain H    boucle ouverte    LQG modal        ESO propose
    x0.50         0.0462        0.3183 (-42 %)   0.3144 (-42 %)
    x1.00         0.0462        0.5457           0.5457
    x2.00         0.0462        0.4582 (-16 %)   0.4796 (-12 %)

(balayage mesure AVANT la mise en place de la couche de colle ; les ordres de
grandeur et le caractere non monotone sont inchanges, le gain nominal a
simplement baisse de 11 %.)

La boucle ouverte ne bouge pas d'un chiffre : H_Pe n'entre pas dans le modele
sans commande. En boucle fermee le pire cas perd 42 % a gain divise par deux ET
16 % a gain double -- le comportement n'est pas monotone, car doubler le gain
double aussi le gain de boucle d'un correcteur synthetise sur le gain nominal.
Les valeurs du tableau du haut sont donc celles d'UN point d'un parametre non
calibre. L'egalite LQG / ESO, elle, tient aux trois gains (ecart <= 4 %).
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

# reglages retenus (voir data/phase*_results.json)
X_ESO = [-8.229, -0.437, 0.550, -7.874, 0.367, 230297.916, 1.615]
X_LQG = [-8.263, 1.901, -7.284]


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
