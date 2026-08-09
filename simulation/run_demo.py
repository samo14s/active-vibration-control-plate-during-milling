"""
run_demo.py — reproduit le resultat principal de la campagne v3.

Usage :
    cd simulation
    python run_demo.py            # tableau nominal (3 architectures)
    python run_demo.py --full     # + perturbations reservees (plus long)

Base : plaque encastree-libre 5 modes calibree sur Du et al. (IJMS 2024),
avec les DEUX corrections validees contre la Fig. 12 et le Tableau 4 :
patch piezo inferieur droit horizontal, et amortissements modaux mesures
(0.31 / 0.17 / 0.27 / 0.56 / 0.35 %).

Resultat attendu (b_lim pire cas sur les 5 vitesses) :
    boucle ouverte  ~ 0.049 mm
    LQG modal       ~ 0.515 mm
    ESO propose     ~ 0.515 mm
c.-a-d. performance SATUREE : trois architectures de richesse croissante
atteignent le meme plafond. C'est le resultat central de l'etude.
"""
import argparse
import copy
import sys
import time

import numpy as np

sys.path.insert(0, "sim_kit")
sys.path.insert(0, ".")
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


def blim(sim, mk, rpm, T=0.45):
    lo, hi = 0.02e-3, 2.0e-3

    def ok(ap):
        tau = 60.0 / (3 * rpm)
        return sim.run(None if mk is None else mk(tau / 82, tau),
                       rpm=rpm, ap=ap, T=T)["stable"]
    if not ok(lo):
        return 0.0
    while ok(hi) and hi < 4e-3:
        hi *= 1.4
    while hi - lo > 6e-5:
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if ok(m) else (lo, m)
    return lo * 1e3


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
        k1, k2 = float(sim.k1c), float(sim.k2c)
        for name, ks, cs, kcs in [("K x0.90", .90, 1, 1),
                                  ("K x1.10", 1.10, 1, 1),
                                  ("C x0.80", 1, .80, 1),
                                  ("kc x2.9", 1, 1, 2.9)]:
            sim.plate.Kp = K0 * ks
            sim.plate.Cp = C0 * cs
            sim.k1c, sim.k2c = k1 * kcs, k2 * kcs
            sim._cache.clear()
            print(f"\n=== {name} (perturbation reservee) ===")
            table(sim, plate, name)

    print(f"\ntermine en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
