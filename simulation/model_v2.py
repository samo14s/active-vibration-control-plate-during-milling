"""
model_v2 — base gelee CORRIGEE (campagne v2).

Correction validee contre Du 2024 Fig. 12b : le patch piezo est au coin
INFERIEUR DROIT, ORIENTE HORIZONTALEMENT (60 mm selon x, 20 mm selon z),
du meme cote que le capteur (100, 80) — configuration du banc de fraisage
(Section 5) et seule configuration reproduisant les 3 antiresonances
mesurees (787 / 1493 / 3609 Hz : motif 4/4).

Frequences recalibrees identiques (erreur <= 0.67 % vs mesures Du).
||H|| = 0.5546 (x1.86 vs ancienne base) : plus d'autorite actionneur.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/claude/work/sim_kit")
import simulation_base as SB

PATCH_V2 = dict(SB.PATCH)
PATCH_V2.update({"x1": 0.04, "x2": 0.10, "z1": 0.0, "z2": 0.02})

# Amortissements modaux : valeurs MESUREES par Du et al. (Tableau 4).
# La base d'origine utilisait 0.30 % aux modes 4 et 5 ; les mesures donnent
# 0.56 % et 0.35 %. Le mode 4 etait donc sous-amorti d'un facteur 1.87, ce
# qui exagerait le spillover de commande (phase 21) et rendait la boucle
# marginale. Correction appuyee sur la donnee publiee, pas sur un choix libre.
ZETA_DU = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]


def make_sim(verbose=False, zeta_du=True):
    SB.PATCH = PATCH_V2
    if zeta_du:
        SB.ZETA_MODES = list(ZETA_DU)
    sim = SB.SimBase(verbose=verbose)
    H = np.asarray(sim.plate.H_Pe_modal).ravel()
    D = np.asarray(sim.plate.D_obs).ravel()
    assert list(np.sign(H * D).astype(int)) == [-1, -1, -1, 1, 1], \
        "signature H*D inattendue — verifier la configuration v2"
    return sim
