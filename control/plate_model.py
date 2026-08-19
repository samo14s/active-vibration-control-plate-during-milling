"""
plate_model.py — Plaque + pastille piezoelectrique (Section 2 du papier)
========================================================================
Seule partie conservee de la couche de commande precedente : la construction
du modele. Tout le reste (mu-synthese, commande a retard) a ete retire au
profit de la comparaison FOPID / ADRC-FOPID optimisees par PSO.

POSITION ET ORIENTATION DE L'ACTIONNEUR — le papier ne les donne pas de facon
utilisable :
  * il se contredit sur le COTE (Sections 3 et 4.1 : "the left lower corner of
    the plate" ; Section 5 : "pasted in the right lower corner of the plate
    back", capteur au coin superieur droit) ;
  * il ne dit RIEN du SENS : la QDA60-20-0.7 mesure 60 x 20 mm et peut etre
    collee 60 mm le long de l'arete x ou 60 mm le long de la hauteur z.

C'est la Fig. 12(b) qui tranche, et sans aucun reglage : le nombre de zeros de
la fonction de transfert tension -> deplacement entre deux poles consecutifs
est fixe par les SIGNES des residus D_obs(i) H_Pe(i), donc par la geometrie
seule. Les creux profonds digitalises (788 / 1493 / 3609 Hz) contre les poles
mesures donnent l'occupation (1, 1, 0, 1), et des quatre geometries
admissibles une seule la reproduit — coin bas DROIT, 60 mm le long de x :

    droite 60x20   (1, 1, 0, 1)  <- retenue      zeros 818 / 2665 / 3751 Hz
    droite 20x60   (1, 1, 1, 1)
    gauche 60x20   (0, 2, 0, 0)
    gauche 20x60   (0, 0, 1, 0)

(verification/19_patch_orientation.py ; le modele elements finis independant
de simulation/ aboutit a la meme geometrie et aux memes zeros a 0.4 % pres.)

Avec cette geometrie les residus D_obs(i) H_Pe(i) ne sont PAS tous de meme
signe ([-1 -1 -1 +1 +1]) : la paire actionneur/capteur n'est pas duale, la
phase n'est pas bornee, et les zeros ne s'intercalent pas systematiquement
entre les poles. C'est une difficulte reelle de ce montage, pas un defaut du
modele — et c'est une des choses que la comparaison FOPID / ADRC-FOPID met a
l'epreuve. b0 = D_obs . H_Pe = +3.40 (positif).
"""
import numpy as np

from chebyshev_plate import ChebyshevPlate

F_MEASURED = [540.0, 1068.0, 2787.0, 3351.0, 4122.0]      # Tableau 4, mesure
F_THEORETICAL = [537.0, 1101.0, 2805.0, 3423.0, 4254.0]   # Tableau 4, theorie
# 60 mm le long de x, 20 mm le long de z (voir l'entete)
PATCH = dict(left=dict(x1=0.000, x2=0.060, z1=0.0, z2=0.020),
             right=dict(x1=0.040, x2=0.100, z1=0.0, z2=0.020))


def build_plate(patch='right', PX=14, PZ=14, n_modes=5, calibrate=True,
                freqs=None):
    """Plaque du Tableau 1 avec la pastille du Tableau 2.

    Calage par defaut sur les frequences THEORIQUES du Tableau 4 : c'est le
    jeu qui reproduit la Section 4 du papier (broutement simule a 1121 Hz
    contre 1135 Hz publie). `freqs=F_MEASURED` donne le modele recale sur les
    mesures, utilise pour les essais de robustesse.
    """
    p = ChebyshevPlate(PX=PX, PZ=PZ, n_modes=n_modes)
    p.add_piezo_patch(**PATCH[patch])
    if calibrate:
        tgt = list(F_THEORETICAL[:n_modes]) if freqs is None \
            else list(freqs)[:n_modes]
        p.calibrate_frequencies(tgt)
    p.patch_side = patch
    return p


def plant_vectors(plate, n_modes=2):
    """(omega, zeta, H, D_obs, sign_loop) du modele de synthese.

    `sign_loop` rend la boucle a retour NEGATIF quel que soit le signe du
    couplage : la commande appliquee est u = sign_loop * C(s) * y. Avec la
    geometrie retenue (coin bas droit, 60 mm le long de x) D_obs.H > 0, donc
    sign_loop = -1, et des gains POSITIFS de C(s) ajoutent bien de la raideur
    et de l'amortissement.
    """
    w = np.asarray(plate.omega_n[:n_modes], float)
    z = np.asarray(plate.zeta_modes[:n_modes], float)
    H = np.asarray(plate.H_Pe_modal, float)[:n_modes]
    D_obs = plate.D_row(plate.lp, plate.hp)[:n_modes]
    sign_loop = -float(np.sign(D_obs @ H))
    return w, z, H, D_obs, sign_loop


def plant_frf(plate, f, n_modes=2, x_force=None):
    """(P_u, P_f) : reponses de y_obs a la tension [m/V] et a la force de
    coupe appliquee en x_force [m/N]."""
    w, z, H, D_obs, _ = plant_vectors(plate, n_modes)
    om = 2 * np.pi * np.asarray(f, float)
    den = (w[:, None]**2 - om[None, :]**2
           + 2j * z[:, None] * w[:, None] * om[None, :])
    Pu = np.sum((D_obs * H)[:, None] / den, axis=0)
    if x_force is None:
        return Pu, None
    Dx = plate.D_row(x_force, plate.hp)[:n_modes]
    Pf = np.sum((D_obs * Dx)[:, None] / den, axis=0)
    return Pu, Pf
