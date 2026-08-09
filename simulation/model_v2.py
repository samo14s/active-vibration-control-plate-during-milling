"""
model_v2 — RETRACTATION : ce module ne differe plus de la base figee.

--------------------------------------------------------------------------
CE QUI S'EST PASSE
--------------------------------------------------------------------------
Ce module deplacait le patch piezoelectrique au coin INFERIEUR DROIT, oriente
HORIZONTALEMENT (60 mm selon x, 20 mm selon z), au motif que c'etait la seule
configuration reproduisant la signature de la Fig. 12(b) de Du et al. :
l'occupation (1, 1, 0, 1) des intervalles inter-mode par les creux profonds.

C'ETAIT UNE ERREUR DE SELECTION DE MODELE. La Fig. 11 de l'article — la photo
de l'experience modale qui a PRODUIT la Fig. 12 — montre l'actionneur : un
rectangle nettement plus haut que large, au coin INFERIEUR GAUCHE, montant du
bord encastre. La Fig. 2 (schema) le dessine de la meme facon, et la Fig. 17
precise « Actuator is in the back » pour le banc de fraisage : le « right lower
corner » de la Section 5 et le « left lower corner » de la Section 4.1 sont donc
LE MEME coin physique vu des deux faces. L'article ne se contredit pas.

La bonne configuration est celle de la base figee : coin inferieur gauche,
VERTICAL 20 x 60 mm. Elle est corroboree par le premier mode, une fois la
couche de colle modelisee :

      vertical   (base)      : f1 = 538.3 Hz     mesure : 540.0 Hz   (-0.3 %)
      horizontal (ex-"v2")   : f1 = 567.5 Hz                         (+5.1 %)

--------------------------------------------------------------------------
CE QUE CELA A APPRIS SUR LE MODELE DE COUPLAGE -- ET COMMENT C'EST CORRIGE
--------------------------------------------------------------------------
Avec la bonne geometrie, l'occupation calculee par les elements finis est
(0, 0, 1, 0). Ce desaccord ne disait donc RIEN sur la position du patch : il
mesurait un defaut de H_Pe, dont la REPARTITION MODALE etait fausse.

    grandeur                     EF            mesure (Fig. 12b)
    occupation des intervalles   (0, 0, 1, 0)  (1, 1, 1, 1)
    creux en bande               2825 Hz       788 / 1493 / 2913 / 3609 Hz
    ecart courbe a courbe        14.3 dB RMS   --

Corriger cela demandait un terme DEPENDANT DU MODE : un facteur scalaire — gain
actionneur, rendement de collage — se simplifie dans les rapports de residus
qui fixent les zeros. C'est desormais fait, non pas en inventant un mecanisme,
mais en IDENTIFIANT la repartition sur la Fig. 12(b) elle-meme : ses poles sont
les frequences mesurees, ses zeros les creux mesures, et poles + zeros fixent
les residus a une constante pres. Voir simulation_base.H_IDENT et
verification/14_coupling_identification.py. La base rend maintenant

    occupation (1, 1, 1, 1), creux 788 / 1495 / 2894 / 3614 Hz, 5.5 dB RMS.

RECTIFICATION. Ce fichier a longtemps annonce NOTCH_OCCUPANCY = (1, 1, 0, 1),
d'apres les trois creux PROFONDS de la Fig. 12(b). C'etait une erreur de
lecture : la numerisation fournie ne compte que 76 points sur 0-5000 Hz, soit
~66 Hz de pas, et elle rate le fond du 4e creux, qu'elle rend comme un simple
epaulement a +11.9 dB vers 2913 Hz. Un ajustement libre des cinq residus (16
motifs de signe x 60 graines) tranche : la lecture a quatre creux tombe dans la
region de confiance sur ses cinq composantes, celle a trois creux en sort sur
quatre sur cinq (10.0 dB RMS contre 5.5). La courbe THEORIQUE de l'article,
tracee sur la meme figure, montre elle aussi quatre creux — elle etait d'accord
avec sa propre mesure, et c'est notre lecture qui ne l'etait pas.

--------------------------------------------------------------------------
POURQUOI CE MODULE EXISTE ENCORE
--------------------------------------------------------------------------
`run_demo.py` importe `make_sim`, et les diagnostics `antiresonances()` /
`notch_occupancy()` restent utiles pour surveiller le modele de couplage. Le
nom « v2 » n'a plus de sens : make_sim() renvoie exactement SimBase().
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "sim_kit"))
import simulation_base as SB                                    # noqa: E402

# Geometrie du patch : celle de la base, confirmee par la Fig. 11 de l'article.
PATCH_V2 = dict(SB.PATCH)

# Amortissements mesures (Tableau 4). Desormais portes par la base elle-meme ;
# conserve pour compatibilite d'appel.
ZETA_DU = list(SB.ZETA_MODES)

# Fig. 12(b) numerisee : creux mesures et leur repartition par intervalle
# inter-mode. Le 4e (2913 Hz) est un epaulement dans la numerisation fournie,
# pas un creux franc — voir la RECTIFICATION dans l'en-tete.
NOTCH_HZ = (788.0, 1493.0, 2913.0, 3609.0)
NOTCH_OCCUPANCY = (1, 1, 1, 1)


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
    """Nombre de zeros dans chaque intervalle inter-mode (tuple de n_modes-1).

    Signature de SIGNE, insensible a l'amortissement et a toute mise a
    l'echelle de H_Pe : l'intervalle k contient un nombre impair de zeros si
    sign(r_k) == sign(r_{k+1}), pair sinon.
    """
    f = np.asarray(plate.omega_n).ravel() / (2 * np.pi)
    z = antiresonances(plate)
    return tuple(int(((z > f[k]) & (z < f[k + 1])).sum())
                 for k in range(len(f) - 1))


def report_notch_signature(plate, verbose=True):
    """Compare la signature calculee a la mesure. NE LEVE PAS d'exception.

    Avec coupling='ident' (defaut) la signature est conforme ; avec
    coupling='fem' elle ne l'est pas, et c'est justement ce que mesure ce
    diagnostic. En faire une assertion reviendrait a maquiller un ecart de
    modele en contrainte de conception — ce qui avait conduit a deplacer le
    patch a tort.
    """
    occ = notch_occupancy(plate)
    if verbose:
        z = antiresonances(plate)
        f = np.asarray(plate.omega_n).ravel() / (2 * np.pi)
        zi = z[(z > f[0]) & (z < f[-1])]
        flag = "conforme" if occ == NOTCH_OCCUPANCY else "ECART CONNU"
        print(f"[model_v2] occupation {occ} vs mesure {NOTCH_OCCUPANCY} "
              f"-> {flag}")
        print(f"[model_v2] creux en bande {np.round(zi, 0)} Hz "
              f"vs mesure {list(NOTCH_HZ)}")
    return occ


def make_sim(verbose=False, zeta_du=True):
    """Construit la base. Identique a SimBase() : la geometrie de patch qui
    distinguait autrefois ce module etait fausse (voir l'en-tete)."""
    sim = SB.SimBase(verbose=verbose, patch=PATCH_V2,
                     zeta=list(ZETA_DU) if zeta_du else None)
    if verbose:
        report_notch_signature(sim.plate)
    return sim


if __name__ == "__main__":
    sim = make_sim(verbose=True)
    p = sim.plate
    H = np.asarray(p.H_Pe_modal).ravel()
    D = np.asarray(p.D_obs).ravel()
    print(f"\n||H|| = {np.linalg.norm(H):.4f} N/V")
    print("signes H*D_obs :", np.sign(H * D).astype(int))
    print(f"eta (colle)    : {p.eta_bond:.4f}")
    print(f"couplage       : {sim.coupling}")
    print("\nLa repartition modale de H_Pe est desormais identifiee sur la")
    print("Fig. 12(b) ; son NIVEAU reste non valide (F9) et se balaie par")
    print("gain_H. Voir VERIFICATION.md, sections F4 et 10.")
