"""
config.py — Point de fonctionnement et protocole de comparaison
================================================================
Comparaison EQUITABLE entre FOPID et ADRC-FOPID, tous deux optimises par
essaim particulaire (PSO) pour maximiser les limites de stabilite du fraisage.

Deux reglages se choisissent par VARIABLE D'ENVIRONNEMENT, pour que les deux
correcteurs subissent toujours exactement le meme protocole :

    PROTOCOL=A   synthese sur le modele REDUIT du papier (2 modes, Eq. 21),
                 evaluation finale sur les 5 modes -> mesure la robustesse au
                 defaut de modele ;
    PROTOCOL=B   synthese ET evaluation sur les 5 modes (defaut) -> mesure la
                 performance maximale a modele parfait.

    CALIB=measured      plaque calee sur les frequences MESUREES du Tableau 4
                        (la plaque reelle ; defaut pour la commande) ;
    CALIB=theoretical   plaque calee sur la colonne "theorique" du Tableau 4
                        (le modele du papier ; sert a reproduire sa Section 4).
"""
import os
import numpy as np

# ---------------------------------------------------------------- plaque [P]
ZETA = (0.0031, 0.0017, 0.0027, 0.0056, 0.0035)           # Tableau 4
F_MEASURED = [540.0, 1068.0, 2787.0, 3351.0, 4122.0]
F_THEORETICAL = [537.0, 1101.0, 2805.0, 3423.0, 4254.0]
PATCH_SIDE = 'right'          # configuration experimentale (Section 5)

# calage de la plaque nominale utilisee par TOUTE la couche de commande
CALIB = os.environ.get('CALIB', 'measured').lower()
assert CALIB in ('measured', 'theoretical')
F_NOMINAL = F_MEASURED if CALIB == 'measured' else F_THEORETICAL

# ------------------------------------------------- conditions de coupe [P]
AE = 0.1e-3
FZ = 0.02e-3
RPM_DESIGN = 4900             # vitesse de synthese
RPM_GRID = (4300, 4900, 5500, 6100, 6700)                 # validation
AP_T2 = 0.5e-3
RPM_T2 = 6100

# --------------------------------------------- convention de couplage [P!]
# Le papier est incoherent : en partant de ses Eqs. (1)(2)(5)(10) on obtient
# (K - a4 D^T D) q(t) + a4 D^T D q(t - tau), alors que ses Eqs. (12)-(13)
# donnent les signes inverses. Le choix n'est pas cosmetique : il echange les
# creux et les bosses des lobes, donc change d'un facteur ~10 la limite de
# coupe A UNE VITESSE DONNEE — et c'est cette limite que le PSO maximise.
#
# verification/18_sign_convention.py tranche par trois preuves independantes
# tirees du papier lui-meme (calage sur les frequences mesurees) :
#
#   P1  Fig. 13(b) annonce des maxima "autour de 3600 et 5400 tr/min" :
#       Eq. (13) -> 5400 (0.407 mm) et 3600 (0.267 mm)   <- exact
#       signe derive -> 5000 et 3400 tr/min              <- decale
#   P2  Fig. 14(a) : la condition S (4900 tr/min, a_p = 0.30 mm) DIVERGE :
#       Eq. (13) -> a_p,lim = 0.045 mm, divergence a 0.089 s   <- conforme
#       signe derive -> a_p,lim = 0.435 mm, reponse STABLE a 2.3 um  <- non
#   P3  Fig. 18 : voir le script (partiellement discriminant).
#
# => l'erreur de signe est dans les equations intermediaires du papier, pas
#    dans son Eq. (13) ; on suit l'Eq. (13) telle que publiee, exactement
#    comme la couche elements finis (simulation_base.FORCE_SIGN = +1).
SIGN_SIM = +1.0

# ------------------------------------------------------------ numerique
# --- modele de synthese contre modele d'evaluation --------------------------
# Le papier concoit sur DEUX modes (Eq. 12 simplifiee). Mais evaluer aussi sur
# deux modes laisse l'optimiseur EXPLOITER les modes absents : un premier essai
# a produit un ADRC-FOPID a w_o = 5.6e4 rad/s (8.9 kHz) qui dominait le FOPID
# sur tous les criteres a deux modes (3.00 mm contre 2.57) et s'effondrait a
# 0.00 mm des qu'on le rejugeait sur cinq modes — l'observateur prenait les
# modes 3-5 (2.8, 3.4, 4.2 kHz) pour de la "perturbation" et les excitait.
# Le protocole B evalue donc l'objectif ET les contraintes sur le modele
# COMPLET a cinq modes : aucune des deux structures ne peut etre recompensee
# pour avoir exploite les lacunes du modele de synthese.
# Le protocole A garde volontairement le modele reduit du papier PENDANT la
# synthese (objectif, contraintes et b0 a deux modes) et ne revele les modes
# 3 a 5 qu'a l'evaluation : c'est la mesure de robustesse au defaut de modele.
# Dans les deux cas les DEUX correcteurs voient exactement le meme protocole.
PROTOCOL = os.environ.get('PROTOCOL', 'B').upper()
assert PROTOCOL in ('A', 'B')

N_MODES = 5                   # modele d'evaluation = verite (lobes, temporel)
N_MODES_OBJ = 2 if PROTOCOL == 'A' else 5     # modele vu par l'optimiseur
N_MODES_DESIGN = N_MODES_OBJ  # ce que le correcteur "connait" (b0 nominal)
M_FLOQUET_PSO = 40            # sous-intervalles pendant l'optimisation
M_FLOQUET = 200               # sous-intervalles pour tous les resultats
# Nombre de periodes de l'iteration de puissance : None = critere d'arret
# ADAPTATIF (voir closed_loop.spectral_radius). Un nombre fixe et petit
# sous-estime rho, donc SURESTIME la stabilite : a 20 periodes la boucle
# ouverte a 4900 tr/min etait declaree stable jusqu'a 0.080 mm au lieu de
# 0.045 mm. Ne remettre une valeur entiere que pour un diagnostic.
N_PERIOD = None
# Pas d'integration temporelle. Les correcteurs contiennent des poles
# d'Oustaloup jusqu'a w_h = 2*pi*100 kHz : a n_sub = 164 (fs = 40 kHz) ces
# dynamiques se replient et la simulation diverge avec des tensions de 700 kV
# alors que l'analyse de Floquet (correcteur continu) donne stable. Ce n'est
# PAS une instabilite physique : a n_sub = 656 (161 kHz) et 2624 (643 kHz) les
# reponses sont identiques (4.07 um / 40.1 V pour le FOPID). Regle : fs doit
# depasser quelques fois w_h/2pi. Une implantation materielle a 40 kHz
# demanderait de retrecir la bande d'Oustaloup.
N_SUB = 656
POSITIONS_DESIGN = (0.0, 0.5, 1.0)          # fractions de l_P (synthese)
POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)     # fractions de l_P (validation)

# ------------------------------------- realisation d'ordre fractionnaire
# MEME filtre d'Oustaloup pour les deux correcteurs (condition d'equite) :
# bande [1 Hz, 100 kHz], N = 3 -> ordre 7 par operateur.
# Precision mesuree sur 100-5000 Hz pour s^0.5 : phase 1.45 deg, gain 0.04 dB.
OUST_WB = 2 * np.pi * 1.0
OUST_WH = 2 * np.pi * 1.0e5
OUST_N = 3
# lissage commun (anti-repliement) applique aux DEUX correcteurs
ROLLOFF_HZ = 8000.0
ROLLOFF_ORDER = 2

# ------------------------------------------------- protocole d'equite [!]
# Les deux correcteurs subissent EXACTEMENT :
#   * le meme modele, la meme pastille, le meme capteur, le meme signe ;
#   * la meme fonction objectif et les memes contraintes ;
#   * le meme PSO (taille, iterations, coefficients) et les memes graines ;
#   * la meme evaluation finale (Floquet m=200, memes simulations).
# Seule differe la STRUCTURE du correcteur (5 parametres contre 7) — c'est
# l'objet meme de la comparaison, et le nombre de parametres est rapporte.
MS_MAX = 2.0                  # marge de module : max |S| <= 2  (>= 0.5)
V_PER_N = 450.0               # effort : max |K S P_f| <= 450 V par newton
# Saturation de l'amplificateur, appliquee A L'IDENTIQUE aux deux correcteurs
# dans TOUTES les simulations temporelles. Le papier n'annonce pas de borne
# explicite mais l'etage de puissance est un PI E420.00 de gain 100 pilotant
# une pastille QDA60-20-0.7 ; +/-150 V est la limite retenue par l'audit du
# modele elements finis de ce depot (VERIFICATION.md, defaut F1). Un correcteur
# non sature dispose d'une autorite illimitee et toute comparaison perd son
# sens : la borne fait donc partie du protocole d'equite.
V_MAX = 150.0
# Profondeurs sondes de l'objectif. Elles doivent ENCADRER la zone utile,
# sinon le critere ne mesure plus la limite atteignable mais "qui est le moins
# instable" a une profondeur que personne n'atteint. La zone utile est fixee
# par le papier et par la plaque, pas par les correcteurs : la limite sans
# commande vaut 0.033 mm a 4900 tr/min, la condition S du papier est a
# 0.30 mm, et ses experiences montent a 0.6-0.8 mm sous commande (Fig. 18).
# Le jeu (0.5, 1, 2) mm herite de la version pre-correction du modele, ou la
# limite sans commande etait cinq fois plus haute.
AP_PROBE = (0.15e-3, 0.30e-3, 0.60e-3)

PSO = dict(n_particles=24, n_iter=24, w=0.72, c1=1.5, c2=1.5, v_max=0.25,
           seeds=(1, 2, 3))

# bornes de recherche (log10 pour les gains)
# Les bornes couvrent, pour CHAQUE structure, la plage physiquement utile de
# ses propres parametres (ils n'ont pas la meme signification des deux cotes :
# le FOPID agit sur le deplacement, l'ADRC-FOPID sur le double integrateur
# compense, ou les gains sont homogenes a des w_c^2 et 2 w_c). Imposer les
# memes intervalles numeriques a des grandeurs de natures differentes serait
# le choix INEQUITABLE. L'etendue relative est comparable : 5 a 7 decades.
BOUNDS_FOPID = dict(
    log_Kp=(2.0, 7.0), log_Ki=(2.0, 9.0), log_Kd=(0.0, 5.0),
    lam=(0.05, 1.0), mu=(0.05, 1.0))
BOUNDS_ADRC = dict(
    log_Kp=(3.0, 9.0), log_Ki=(3.0, 11.0), log_Kd=(0.0, 6.0),
    lam=(0.05, 1.0), mu=(0.05, 1.0),
    log_wo=(3.0, 5.5), b0_scale=(0.2, 5.0))
