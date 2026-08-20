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
M_FLOQUET_PSO = 24            # sous-intervalles pendant l'optimisation
M_FLOQUET = 200               # sous-intervalles pour tous les resultats
# Le rayon spectral de la monodromie est desormais obtenu par Arnoldi
# (paper_model/monodromy.py) : il est EXACT a 1e-15 pres, donc il n'y a plus
# ni nombre de periodes, ni tolerance, ni reglage degrade pour l'optimisation.
# Les quatre constantes N_PERIOD, N_PERIOD_MIN_PSO, N_PERIOD_TOL_PSO et
# N_PERIOD_MAX_PSO qui vivaient ici reglaient une iteration de puissance qui
# rendait des valeurs fausses (0.79107 pour un exact de 0.967392) et dont
# aucun reglage ne pouvait detecter l'erreur ; elles ont ete supprimees plutot
# que conservees inertes.
# Pas d'integration temporelle. Les correcteurs contiennent des poles
# d'Oustaloup jusqu'a w_h = 2*pi*100 kHz : a n_sub = 164 (fs = 40 kHz) ces
# dynamiques se replient et la simulation diverge avec des tensions de 700 kV
# alors que l'analyse de Floquet (correcteur continu) donne stable. Ce n'est
# PAS une instabilite physique : a n_sub = 656 (161 kHz) et 2624 (643 kHz) les
# reponses sont identiques (4.07 um / 40.1 V pour le FOPID). Regle : fs doit
# depasser quelques fois w_h/2pi. Une implantation materielle a 40 kHz
# demanderait de retrecir la bande d'Oustaloup.
N_SUB = 656
# Positions vues par l'objectif. Trois points (0, 1/2, 1) laissaient un angle
# mort : l'ADRC-FOPID retenu avant correction avait une explosion locale a
# x/l_P = 0.125 que la grille de synthese ne voyait pas (log rho = +2.11 la
# contre +0.57 sur la grille), alors que le FOPID n'en avait pas. La grille de
# synthese abritait donc une structure et pas l'autre.
POSITIONS_DESIGN = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
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
# Les sondes DEPENDENT DU PROTOCOLE, parce qu'elles doivent encadrer la zone
# atteignable DU MODELE SUR LEQUEL ON NOTE. Le modele reduit a deux modes du
# protocole A est bien plus facile a stabiliser que la plaque complete : avec
# les sondes du protocole B, le depistage FOPID y saturait deja le plafond de
# l'estimateur (2.1 mm), donc l'objectif ne discriminait plus rien. Les deux
# structures subissent evidemment les MEMES sondes dans un protocole donne.
AP_PROBE = ((0.5e-3, 1.5e-3, 3.0e-3) if PROTOCOL == 'A'
            else (0.12e-3, 0.30e-3, 0.70e-3))

# Taille d'essaim PROPORTIONNELLE A LA DIMENSION. A budget d'evaluations egal,
# un PSO de 24 particules n'a pas la meme qualite de recherche en 5 et en 7
# dimensions : mesure sur trois paysages de reference avec ce meme code, l'ecart
# a l'optimum est environ 2x plus grand en 7-D (sphere x1.91, rosenbrock x2.06,
# rastrigin x2.36, 12 graines chacun). "Meme nombre d'evaluations" n'est donc
# PAS une garantie d'equite ; c'est la qualite de recherche qu'il faut egaliser.
# n_particles = 10 + 4 n_dim donne 30 particules pour le FOPID et 38 pour
# l'ADRC-FOPID ; le nombre d'evaluations differe alors et est rapporte tel quel.
PSO = dict(n_particles=None, n_iter=20, w=0.72, c1=1.5, c2=1.5, v_max=0.25,
           seeds=(1, 2, 3, 4), n_particles_base=10, n_particles_per_dim=4)

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
# Le boitier ADRC doit contenir un COIN DE FAIBLE AUTORITE, sinon la moitie de
# son budget est morte d'avance. Meme a gains FOPID nuls, le terme b3 = w_o^3 de
# l'observateur survit et donne |K|_inf >= w_o,min^2 / (3 b0,max). Avec les
# anciennes bornes (w_o >= 1e3, b0_scale <= 5) ce plancher valait 1.97e4 V/m,
# contre 1.8e2 V/m pour le boitier FOPID : un facteur 108. L'optimiseur le
# confirmait en poussant b0_scale a 0.977 de son intervalle, c'est-a-dire en
# cherchant a REDUIRE l'autorite qu'on lui imposait. Avec w_o >= 1e2 et
# b0_scale <= 50 le plancher tombe a 2.0e1 V/m, donc sous celui du FOPID.
BOUNDS_ADRC = dict(
    log_Kp=(3.0, 9.0), log_Ki=(3.0, 11.0), log_Kd=(0.0, 6.0),
    lam=(0.05, 1.0), mu=(0.05, 1.0),
    log_wo=(2.0, 5.5), b0_scale=(0.05, 50.0))

# Boitier de la structure FOPID + observateur modal (control/fdob.py). Les
# gains gardent les bornes du FOPID : la structure CONTIENT le FOPID a
# alpha = 0, donc lui donner un boitier de gains different fausserait la
# comparaison avec son propre cas particulier. S'y ajoutent zeta_q (largeur
# relative des passe-bande, en log car elle couvre presque deux decades) et
# alpha (le melange). Soit SEPT parametres, autant que l'ADRC-FOPID.
#
# Le coin "zeta_q large x alpha grand" contient des correcteurs INSTABLES
# (1 - alpha sum Q_k s'approche de zero). On ne retrecit pas les bornes pour
# l'eviter : ce serait une faveur que ni le FOPID ni l'ADRC-FOPID n'ont
# recue. Le crible de stabilite nominale les rejette, exactement comme il
# rejette la moitie morte du boitier ADRC, et la part perdue est mesuree et
# rapportee par control/audit_fairness.py.
# La borne BASSE de zeta_q compte, et elle se CALCULE. Le bloc de
# l'observateur est W = sum_k Q_k P_k^-1, et P_k^-1 CROIT en s^2 : evalue au
# mode voisin, P_1^-1(j w_2) vaut 470 fois P_1^-1(j w_1). Pour que le terme du
# mode 1 ne domine pas celui du mode 2 chez lui, il faut |Q_1(j w_2)| < 1/470,
# ce qui exige zeta_q de l'ordre de zeta_k lui-meme (0.0017-0.0056 ici), voire
# moitie moins. Mesure directe : a zeta_q = 0.05 la contribution de
# l'observateur au gain de boucle au mode vaut 22 alpha au lieu de alpha, et
# 85 alpha a zeta_q = 0.2 — c'est le mode VOISIN qu'on entend, pas le sien.
# Avec une borne basse a 0.01 l'optimiseur n'aurait pas pu atteindre la region
# selective. On descend donc a 3.2e-4, un ordre de grandeur sous le mode le
# plus fin.
#
# MAIS la mesure contredit cette prescription, et c'est la mesure qui tranche.
# Balayage a gains FOPID figes, alpha = 0.8, |S| au mode 1 puis max|S| :
#     zeta_q = 0.0005 -> 0.0315 / 4.38     zeta_q = 0.010 -> 0.0220 / 48.6
#     zeta_q = 0.0030 -> 0.0235 / 11.3     zeta_q = 0.050 -> 0.0110 /  2.18
# Le zeta_q etroit cree un accident de phase tres peu amorti qui fait exploser
# la marge de module, pendant que le "diaphonie" entre modes voisins, elle,
# ajoute du gain LA OU ON EN VEUT — aux deux modes de broutement. L'argument
# algebrique ci-dessus dit ou la normalisation exacte du gain de boucle est
# atteinte ; il ne dit pas ou le meilleur correcteur se trouve.
# On garde donc les DEUX regions dans le boitier (3.2e-4 a 0.7) et on laisse
# l'optimiseur choisir, en rapportant si une borne devient active.
BOUNDS_FDOB = dict(
    log_Kp=(2.0, 7.0), log_Ki=(2.0, 9.0), log_Kd=(0.0, 5.0),
    lam=(0.05, 1.0), mu=(0.05, 1.0),
    log_zq=(-3.5, -0.155), alpha=(0.0, 0.9))

# Modes vises par l'observateur modal. '12' suit la prescription du maillon 6
# du diagnostic (concentrer le budget sur les deux modes de broutement, ne
# rien depenser au-dela du zero instable a 2459 Hz) ; '12345' exerce en plus
# la conscience du SIGNE, puisque les residus des modes 4-5 sont de signe
# oppose a ceux des modes 1-3. Les deux sont essayes.
FDOB_MODES = os.environ.get('FDOB_MODES', '12')
FDOB_TARGETS = tuple(int(c) - 1 for c in FDOB_MODES)
FDOB_WC = 2 * np.pi * 8000.0      # = coupure d'anti-repliement, non ajustee


# ---------------------------------------------------------------------------
# Boitiers des structures de SYNTHESE MODELE (control/hinf.py, control/musyn.py)
#
# Ces deux-la ne se reglent pas par des gains mais par des PONDERATIONS : c'est
# la nature meme de la synthese H-infini et mu. Le protocole d'equite ne dit
# pas "les memes parametres pour tous" — il serait alors impossible de comparer
# des structures differentes — mais "le meme budget de recherche, les memes
# graines, les memes contraintes, la meme fonction objectif". Chaque structure
# cherche donc sur SES propres poignees.
#
# Cinq parametres, exactement comme le FOPID. Ce n'est pas un hasard qu'on
# s'autorise : c'est une contrainte qu'on s'impose, pour que l'ecart mesure ne
# puisse pas etre attribue a une dimension de recherche plus riche.
#
#   kw    gain de la ponderation de performance (passe-bande)
#   f_w   centre de la bande — LIBRE entre les deux modes de broutement et
#         au-dela : l'optimiseur decide s'il vaut mieux viser un mode ou
#         couvrir les deux
#   zw    largeur relative de la bande
#   w2    ponderation de l'effort de commande (scalaire : voir hinf.py, c'est
#         ce qui donne D12'C1 = 0 sans decalage de boucle)
#   eps   ponderation du bruit de mesure
BOUNDS_HINF = dict(
    log_kw=(0.0, 8.0), f_w=(350.0, 1500.0), log_zw=(-1.5, 0.3),
    log_w2=(-4.0, 2.0), log_eps=(-6.0, -1.0))

# La mu-synthese partage EXACTEMENT le meme boitier : la ponderation
# d'incertitude, elle, n'est pas reglee — c'est celle du papier (Eqs. 18-19),
# figee. La difference entre les deux structures est donc uniquement
# l'iteration D-K, ce qui est precisement ce qu'on veut mesurer.
BOUNDS_MU = dict(BOUNDS_HINF)

#: nombre de tours D-K. Fixe, hors PSO, pour la meme raison que les reglages du
#: superviseur de l'observateur modal : on mesure ce qu'apporte D-K, sans le
#: melanger a un gain d'optimisation.
N_DK = 3


# ---------------------------------------------------------------------------
# Boitiers des trois references classiques (control/classical.py).
#
# Les DIMENSIONS DIFFERENT ici, et c'est voulu. Le DVF n'a que deux poignees
# parce qu'il n'en a que deux : lui en inventer une troisieme pour "egaliser"
# serait truquer la comparaison dans l'autre sens. L'essaim etant
# proportionnel a la dimension (10 + 4 x n_dim), chaque structure recoit un
# effectif adapte a son espace — c'est la regle deja appliquee aux quatre
# premieres, pas une faveur nouvelle.
BOUNDS_DVF = dict(log_g=(-2.0, 6.0), f_d=(200.0, 8000.0))

# Un absorbeur par mode de broutement. Les centres sont libres autour des deux
# modes nominaux (540 et 1068 Hz) : on ne suppose pas que l'accord optimal est
# l'accord exact, c'est justement une question que l'optimiseur tranche.
BOUNDS_VPA = dict(
    log_g1=(-2.0, 8.0), f_a1=(350.0, 800.0), log_z1=(-2.5, -0.3),
    log_g2=(-2.0, 8.0), f_a2=(800.0, 1500.0), log_z2=(-2.5, -0.3))

# LQG : quatre ponderations plus la frequence de mise en forme. Seuls les
# rapports q/r et w/v pilotent le resultat, mais on laisse les quatre libres
# plutot que d'imposer a l'optimiseur une reparametrisation qu'il n'a pas
# demandee — et le cout en est nul, l'essaim etant dimensionne par n_dim.
BOUNDS_LQG = dict(
    log_q=(0.0, 16.0), log_r=(-6.0, 2.0), log_w=(0.0, 12.0),
    log_v=(-6.0, 2.0), f_w=(350.0, 1500.0))

# MPC explicite : rigoureusement les memes cinq ponderations que le LQG, plus
# l'HORIZON — c'est la condition pour que l'ecart mesure entre les deux porte
# sur l'horizon et sur rien d'autre. La borne basse (10 us) est bien en deca
# de la periode du mode le plus haut (0.24 ms a 4122 Hz), donc d'un correcteur
# vraiment myope ; la borne haute (100 ms) est le regime ou la Riccati a
# horizon fini a deja rejoint l'equation algebrique — mesure : ecart relatif
# au gain LQR de 0.38 a 1 ms, 0.16 a 10 ms, 0.030 a 100 ms. Laisser
# l'optimiseur atteindre ce bord est voulu : si le meilleur MPC est le LQG,
# c'est un resultat, pas un echec de reglage.
BOUNDS_MPC = dict(
    log_q=(0.0, 16.0), log_r=(-6.0, 2.0), log_w=(0.0, 12.0),
    log_v=(-6.0, 2.0), f_w=(350.0, 1500.0), log_T=(-5.0, -1.0))

# Mode glissant a couche limite. Seuls K_s/phi et lambda entrent dans la loi
# LINEAIRE de l'interieur de la couche (u = -(K_s/phi)(ydot + lambda y)) ;
# K_s et phi sont neanmoins laisses separes parce qu'ils se separent hors de
# la couche, ou K_s est l'autorite maximale et phi la largeur de la zone
# lineaire — et c'est le critere TEMPOREL qui les distingue.
#   lambda : pente de la surface, soit la pulsation de coupure du glissement.
#            De 100 a 1e5 rad/s encadre les cinq modes (3.4e3 a 2.6e4 rad/s).
#   K_s    : autorite, en volts. Bornee par V_MAX cote simulation.
#   phi    : largeur de la couche limite, en unites de la surface s
#            (m/s + lambda*m). Une couche trop fine ramene le broutement de
#            commande, une couche trop large annule le mode glissant.
BOUNDS_SMC = dict(log_lam=(2.0, 5.0), log_ks=(-1.0, 3.0), log_phi=(-8.0, -1.0))


# Boitier de l'observateur conscient du zero instable (control/nmp_dob.py).
# Memes bornes de gains que le FOPID — la structure le CONTIENT a alpha = 0,
# donc lui donner d'autres bornes casserait cette inclusion. Sept parametres,
# comme l'ADRC-FOPID et comme l'observateur modal : les trois structures qui
# ajoutent un observateur a l'epine dorsale FOPID sont ainsi comparables a
# dimension egale.
#
# `wq` est la coupure du filtre Q. Sa borne haute n'est pas libre : au-dela de
# la bande d'Oustaloup le filtre ne fait plus que de l'amplification de bruit,
# et au-dela du zero instable (2459 Hz) il tente d'inverser ce qui, meme
# apres factorisation, ne rend rien d'utile.
BOUNDS_NMPDOB = dict(
    log_Kp=(2.0, 7.0), log_Ki=(2.0, 9.0), log_Kd=(0.0, 5.0),
    lam=(0.05, 1.0), mu=(0.05, 1.0),
    log_wq=(2.5, 4.6), alpha=(0.0, 0.9))
