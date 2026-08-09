"""
model_v2 — base gelee CORRIGEE (campagne v2).

Deux corrections par rapport a sim_kit/simulation_base.py :

  1. patch piezo au coin INFERIEUR DROIT, ORIENTE HORIZONTALEMENT
     (60 mm selon x, 20 mm selon z), du meme cote que le capteur (100, 80) ;
  2. amortissements modaux MESURES par Du et al. (Tableau 4).

--------------------------------------------------------------------------
POURQUOI CETTE POSITION DE PATCH — l'argument reel
--------------------------------------------------------------------------
Le seul jeu de donnees qui discrimine la position du patch est la Fig. 12(b)
de Du et al. (transfert tension -> deplacement, capteur au coin superieur
droit). Ce qu'elle contient de robuste n'est pas le niveau en dB (voir plus
bas), mais la REPARTITION des creux profonds entre les modes :

    creux profonds mesures : 788 Hz, 1493 Hz, 3609 Hz
    modes mesures          : 540, 1068, 2787, 3351, 4122 Hz
    -> occupation par intervalle inter-mode : (1, 1, 0, 1)

Cette occupation est une signature de SIGNE, insensible a l'amortissement et
a l'echelle : pour G = somme r_i/(w_i^2 - w^2), l'intervalle k contient un
nombre IMPAIR de zeros si sign(r_k) == sign(r_{k+1}), un nombre PAIR sinon.
Elle ne peut donc pas etre ajustee : soit une configuration la reproduit,
soit non. Sur les quatre configurations compatibles avec l'article
(coin gauche/droit x orientation verticale/horizontale) :

    configuration            occupation   accord
    bas-gauche 20x60 (v1)    (0, 0, 1, 0)   non
    bas-droit   20x60        (1, 1, 1, 1)   non
    bas-gauche  60x20        (0, 2, 0, 0)   non
    bas-droit   60x20        (1, 1, 0, 1)   OUI   <- retenue ici

Une seule convient, et c'est celle-ci. C'est verifie a l'execution par
`check_patch_position()`, appele depuis `make_sim()`.

Coherent avec la Section 5 de l'article (« pasted in the right lower corner
of the plate back »). La Section 4.1 dit « left lower corner » pour la meme
experience modale : l'article se contredit, probablement parce que gauche et
droite s'echangent selon la face observee. Les donnees tranchent, pas le texte.

--------------------------------------------------------------------------
CE QUE CETTE CONFIGURATION NE REPRODUIT PAS — a savoir avant de s'en servir
--------------------------------------------------------------------------
* Les FREQUENCES des creux sont fausses au milieu de la bande :

      mesure : 788    1493    3609 Hz
      modele : 818    2655    3750 Hz
      ecart  : +3.8 %  +77.8 %  +3.9 %

  Le premier et le dernier sont bons a 4 % pres, le deuxieme est hors de
  portee : aucune des quatre configurations, ni aucun decalage de patch ou de
  capteur teste, ne descend ce zero jusqu'a 1493 Hz. Il faudrait un rapport
  de residus |r2/r1| ~ 0.95 alors que le modele donne 1.52. C'est une limite
  du modele de couplage, pas du choix de position.

* Le premier mode monte a 572.3 Hz contre 540 Hz mesures (+6.0 %). Ce n'est
  PAS un argument contre cette position : `_add_patch_structure` suppose un
  collage PARFAIT, alors que la couche de colle ne transmet le cisaillement
  que partiellement. Les deux bornes physiques encadrent la mesure dans les
  deux orientations :

      sans raidissement            : 521.1 Hz
      horizontal, collage parfait  : 572.3 Hz     mesure : 540.0 Hz
      vertical,   collage parfait  : 541.1 Hz

  Le premier mode ne discrimine donc pas l'orientation ; il ne fait que fixer
  une raideur de collage inconnue. Que la configuration verticale tombe tout
  pres de 540 Hz est une coincidence : elle compense un modele de plaque nue
  deja 3.5 % trop souple.

* Le NIVEAU absolu de H_Pe reste non valide : la Fig. 12(a) implique une
  souplesse statique 6.36 fois celle d'une plaque de Kirchhoff aux dimensions
  du Tableau 1, et contredit les propres lobes de stabilite de l'article
  (Fig. 13). Voir VERIFICATION.md, point F9.

||H|| = 0.5541 (x1.87 vs la base v1) : plus d'autorite actionneur.
Frequences recalibrees identiques (erreur <= 0.67 % vs mesures Du).

--------------------------------------------------------------------------
AMORTISSEMENTS
--------------------------------------------------------------------------
La base d'origine utilisait 0.30 % aux modes 4 et 5 ; le Tableau 4 donne
0.56 % et 0.35 %. Le mode 4 etait donc sous-amorti d'un facteur 1.87, ce qui
exagerait le spillover de commande et rendait la boucle marginale. Correction
appuyee sur la donnee publiee, pas sur un choix libre.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "sim_kit"))
import simulation_base as SB                                    # noqa: E402

PATCH_V2 = dict(SB.PATCH)
PATCH_V2.update({"x1": 0.04, "x2": 0.10, "z1": 0.0, "z2": 0.02})

# Amortissements modaux : valeurs MESUREES par Du et al. (Tableau 4).
ZETA_DU = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]

# Fig. 12(b) numerisee : creux profonds, et leur repartition par intervalle
# inter-mode. C'est la signature qui identifie la position du patch.
NOTCH_HZ = (788.0, 1493.0, 3609.0)
NOTCH_OCCUPANCY = (1, 1, 0, 1)


def antiresonances(plate):
    """Zeros exacts (systeme non amorti) de la reception tension -> capteur.

    G(w) = somme_i r_i / (w_i^2 - w^2) avec r_i = D_obs,i * H_Pe,i. Les zeros
    sont les racines en s = w^2 du polynome somme_i r_i prod_{j!=i}(s_j - s) ;
    on les calcule exactement plutot que de chercher des minima locaux de |G|,
    car avec amortissement un intervalle SANS zero presente quand meme un
    minimum peu profond que l'on confondrait avec une antiresonance.

    Retour : frequences en Hz, croissantes (zeros positifs reels seulement).
    """
    r = np.asarray(plate.D_obs).ravel() * np.asarray(plate.H_Pe_modal).ravel()
    s2 = np.asarray(plate.omega_n).ravel() ** 2
    n = len(r)
    P = np.zeros(n)
    for i in range(n):
        P += r[i] * np.poly(np.delete(s2, i)) * ((-1) ** (n - 1))
    z = np.roots(P)
    z = z[np.abs(z.imag) <= 1e-6 * max(np.abs(z.real).max(), 1e-30)].real
    return np.sort(np.sqrt(z[z > 0])) / (2 * np.pi)


def notch_occupancy(plate):
    """Nombre de zeros dans chaque intervalle inter-mode (tuple de n_modes-1)."""
    f = np.asarray(plate.omega_n).ravel() / (2 * np.pi)
    z = antiresonances(plate)
    return tuple(int(((z > f[k]) & (z < f[k + 1])).sum())
                 for k in range(len(f) - 1))


def check_patch_position(plate, verbose=False):
    """Verifie que la configuration reproduit la signature de la Fig. 12(b).

    Leve AssertionError si l'occupation des intervalles differe de la mesure —
    c'est le seul critere qui distingue les quatre positions de patch
    compatibles avec l'article.
    """
    occ = notch_occupancy(plate)
    if verbose:
        z = antiresonances(plate)
        f = np.asarray(plate.omega_n).ravel() / (2 * np.pi)
        zi = z[(z > f[0]) & (z < f[-1])]
        print(f"[model_v2] occupation des intervalles : {occ} "
              f"(mesure {NOTCH_OCCUPANCY})")
        print(f"[model_v2] zeros en bande : {np.round(zi, 0)} Hz "
              f"(mesure {list(NOTCH_HZ)})")
    assert occ == NOTCH_OCCUPANCY, (
        f"occupation des creux {occ} != mesure {NOTCH_OCCUPANCY} : la position "
        "du patch ne reproduit plus la signature de la Fig. 12(b)")
    return occ


def make_sim(verbose=False, zeta_du=True):
    """Construit la base v2. N'ecrit PAS dans les constantes de module de
    `simulation_base` : la configuration est passee par arguments, donc un
    `SimBase()` construit ensuite dans le meme processus reste la base v1."""
    sim = SB.SimBase(verbose=verbose, patch=PATCH_V2,
                     zeta=list(ZETA_DU) if zeta_du else None)
    check_patch_position(sim.plate, verbose=verbose)
    return sim


if __name__ == "__main__":
    sim = make_sim(verbose=True)
    p = sim.plate
    H = np.asarray(p.H_Pe_modal).ravel()
    D = np.asarray(p.D_obs).ravel()
    print(f"\n||H|| = {np.linalg.norm(H):.4f} N/V")
    print("signes H*D_obs :", np.sign(H * D).astype(int))
    z = antiresonances(p)
    f = np.asarray(p.omega_n).ravel() / (2 * np.pi)
    zi = z[(z > f[0]) & (z < f[-1])]
    print("\ncreux tension -> deplacement :")
    for zm, zc in zip(NOTCH_HZ, zi):
        print(f"   mesure {zm:7.1f} Hz   modele {zc:7.1f} Hz   "
              f"ecart {100 * (zc / zm - 1):+6.1f} %")
    print("\nle creux central reste hors de portee du modele de couplage :")
    print("voir la section 'CE QUE CETTE CONFIGURATION NE REPRODUIT PAS'.")
