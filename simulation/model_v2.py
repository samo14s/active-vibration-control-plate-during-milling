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
CE QUE CELA APPREND SUR LE MODELE DE COUPLAGE
--------------------------------------------------------------------------
Avec la bonne geometrie, l'occupation calculee est (0, 0, 1, 0) et non
(1, 1, 0, 1). Ce desaccord ne dit donc RIEN sur la position du patch : il
mesure un defaut de H_Pe. Il rejoint le creux mesure a 1493 Hz, que le modele
place a 2827 Hz — meme cause, un couplage dont la REPARTITION MODALE est
fausse.

    grandeur                     modele        mesure (Fig. 12b)
    occupation des intervalles   (0, 0, 1, 0)  (1, 1, 0, 1)
    creux en bande               2827 Hz       788 / 1493 / 3609 Hz

Corriger cela demande un terme DEPENDANT DU MODE dans H_Pe. Un facteur
scalaire — gain actionneur, rendement de collage — ne peut rien y faire : il se
simplifie dans les rapports de residus qui fixent les zeros. Ce chemin-la est
ferme, avec preuve (VERIFICATION.md, section 5).

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

# Fig. 12(b) numerisee : creux profonds mesures et leur repartition par
# intervalle inter-mode. LE MODELE NE LA REPRODUIT PAS — voir l'en-tete.
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

    Le desaccord est un ecart CONNU du modele de couplage, documente dans
    l'en-tete ; en faire une assertion reviendrait a le maquiller en contrainte
    de conception — ce qui avait justement conduit a deplacer le patch a tort.
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
    print("\nLa repartition modale de H_Pe est fausse : c'est le seul ecart")
    print("structurel qui subsiste. Voir VERIFICATION.md, sections F4 et 5.")
