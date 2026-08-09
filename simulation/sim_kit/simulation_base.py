"""
================================================================================
BASE DE SIMULATION VALIDEE — fraisage d'une paroi mince avec patch piezo
================================================================================
Plaque, patch, capteur, efforts de coupe. AUCUN CORRECTEUR.

Cette base est FIGEE. Elle ne doit pas etre modifiee entre deux essais de
commande, sinon les resultats ne sont plus comparables. Tout correcteur se
branche par l'interface decrite plus bas.

--------------------------------------------------------------------------------
1. CE QUI EST VALIDE, ET COMMENT
--------------------------------------------------------------------------------
Le modele est cale sur deux courbes experimentales numerisees (reception au choc
et transfert tension -> deplacement, coin superieur droit). Ces courbes valident
la FORME de la reponse, PAS son niveau : voir la mise en garde plus bas.

CE QUI EST VALIDE
  * cinq resonances mesurees   : 536.4 / 1071.8 / 2778.6 / 3358.7 / 4122.6 Hz
    predites par le modele EF  : +0.4 / +1.5 / -1.9 / +0.8 / +0.7 %
  * quatre antiresonances mesurees : 734 / 2161 / 3001 / 3805 Hz
    modele a cinq modes calibre    : 731 / 2123 / 3009 / 3938 Hz
    ecarts                         : +0.5 / -1.8 / +0.3 / +3.5 %
    (comparaison faite sur la reception AU POINT DE FRAPPE, residus D_obs^2 :
     c'est la grandeur mesuree en Fig. 12a, marteau et capteur au meme coin.
     Un modele a TROIS modes ne donne que deux antiresonances -- 741 / 2407 Hz,
     dont la seconde est fausse de 11 % : d'ou les cinq modes retenus.)
  * les deux premiers modes sont QUASI A EGALITE pour la criticite : par la
    theorie moyennee d'ordre 0 sur ce meme modele, mode 1 seul -> 0.0495 mm,
    mode 2 seul -> 0.0463 mm (7.1 % d'ecart), contre 0.94 a 1.61 mm pour les
    modes 3 a 5. Selon l'analyse menee, c'est donc l'un ou l'autre qui sort :
      - lobes moyennes d'ordre 0        : mode 2, broutement a 1070 Hz ;
      - simulation temporelle non lineaire (4900 tr/min, ap = 0.30 mm, FFT sur
        la seconde moitie du signal, fenetre de Hann) : mode 1, a 524.6 Hz ;
      - experience de Du (Fig. 20)      : LES DEUX, fc1 = 580 et fc2 = 1123 Hz
        sont les deux plus grandes raies.
    La valeur de 1135 Hz publiee correspond au mode que la theorie lineaire de
    ce modele designe elle aussi : elle n'est pas refutable ici.
  * la REPARTITION MODALE du couplage piezoelectrique, par les quatre creux de
    la Fig. 12(b) -- le transfert tension -> capteur :
      mesures : 788 / 1493 / 2913 / 3609 Hz
      modele  : 787 / 1495 / 2907 / 3613 Hz   (ecart max 0.19 %)
    Elle n'est pas PREDITE par les elements finis, elle y est IDENTIFIEE : le
    H_Pe elements finis ne produit qu'un seul creux, a 2818 Hz, et s'ecarte de
    14.3 dB RMS de la Fig. 12(b) contre 5.7 dB pour la version identifiee.
    Voir la constante H_IDENT plus bas et verification/14. SimBase(coupling=
    'fem') restitue la repartition elements finis pour comparaison.

CE QUI N'EST PAS VALIDE : LE NIVEAU ABSOLU, ET DONC LE GAIN ACTIONNEUR
  Le plateau basse frequence de la Fig. 12(a), lu avec la reference annoncee
  (1 um/N), vaut 43.8 um/N. Une plaque de Kirchhoff aux dimensions du Tableau 1
  donne 6.92 um/N (resolution statique EF exacte, tous ddl) : un facteur 6.34.
  Ce n'est pas une erreur du modele. Les memes matrices K et M reproduisent les
  CINQ frequences mesurees a 2 % pres ; or la souplesse est K^-1. Une raideur
  6.34 fois trop faible donnerait f1 = 540/sqrt(6.34) = 214 Hz, pas 540 Hz. En
  termes modaux, 43.8 um/N exigerait D_obs[0]^2 = 504 alors que la normalisation
  en masse du premier mode plafonne ce carre vers 45 : un facteur 11 sur une
  quantite qui n'a aucune liberte. De plus les propres lobes de stabilite de
  l'article (Fig. 13, 0.03-0.05 mm) s'accordent avec la valeur RAIDE ; une
  plaque 6.34 fois plus souple donnerait ~0.008 mm.

  Conclusion : l'echelle en dB de la Fig. 12 n'est pas exploitable. Comme
  H_Pe ne peut se calibrer que par le rapport des deux courbes, LE NIVEAU DE
  H_Pe RESTE NON VALIDE. Toute conclusion de performance en boucle
  fermee -- limite de passe atteinte, tension necessaire -- herite donc de
  cette incertitude, et doit etre accompagnee d'un balayage :

      SimBase(gain_H=0.5) ... SimBase(gain_H=2.0)

  en synthetisant le correcteur sur la plaque nominale et en le simulant sur la
  plaque perturbee (voir `run_demo.py --full`, lignes H x0.5 et H x2.0).

Verification a l'execution : appeler check_model() ; il refait ces controles et
leve une exception si un ecart depasse la tolerance.

--------------------------------------------------------------------------------
2. CE QUI EST FIGE, ET POURQUOI
--------------------------------------------------------------------------------
  * dt = tau/82 exactement. Le retard regeneratif doit tomber sur la grille
    d'echantillonnage, sinon q(t-tau) demande une interpolation qui fausse la
    stabilite. dt n'est donc PAS un reglage libre.
  * cinq modes retenus. Justifie par les antiresonances ci-dessus. La limite de
    stabilite en boucle ouverte est identique a 1, 2, 3 ou 5 modes (les modes 4
    et 5 sont fortement couples au patch mais quasiment pas excites par la
    coupe) : le nombre de modes ne change donc pas la frontiere libre, mais il
    change ce que l'observateur peut voir.
  * amortissements modaux mesures (Tableau 4) : 0.31 / 0.17 / 0.27 / 0.56 / 0.35 %.
  * patch QDA60-20-0.7 dans le coin inferieur gauche, capteur au coin superieur
    droit oppose.
  * patch COLLE, pas soude : couche d'epoxy G_adh = 1 GPa, t_adh = 30 um, d'ou
    1/Gamma = 0.874 mm et un rendement de transfert eta = 0.886 qui reduit a la
    fois le raidissement composite (58.3 % -> 51.7 %) et H_Pe (-11.4 %). Ni
    G_adh ni t_adh ne sont donnes par l'article ; les annuler restitue le
    collage parfait supposé auparavant.

--------------------------------------------------------------------------------
3. INTERFACE CORRECTEUR
--------------------------------------------------------------------------------
Un correcteur est un objet exposant :

    step(x_hat_prev, u_prev, y_meas) -> (x_hat, u)
        appele une fois par pas. y_meas est le deplacement mesure en metres,
        u la tension renvoyee en volts. x_hat_prev est passe tel quel et peut
        etre ignore ; il n'existe que pour compatibilite.
    reset()
        remet les etats internes a zero. Appele avant chaque simulation.

Le simulateur ecrete lui-meme la commande a +/- V_MAX avant de l'appliquer
(newmark_solver.NewmarkSimulator, parametre v_max). La tension renvoyee au
correcteur au pas suivant (u_prev) est la tension SATUREE, donc un observateur
voit toujours ce qui a reellement ete applique. Un correcteur qui sature en
interne doit utiliser la MEME borne, faute de quoi il calculera sa commande
suivante a partir d'une valeur qu'il croit appliquee et qui ne l'est pas.

--------------------------------------------------------------------------------
4. UTILISATION
--------------------------------------------------------------------------------
    from simulation_base import SimBase, check_model

    check_model()                       # a lancer une fois
    sim = SimBase()                     # plaque + patch + capteur

    r = sim.run(MonCorrecteur(...), rpm=4900, ap=0.25e-3, T=0.20)
    print(r['rms_um'], r['stable'])

    J = sim.multi_speed_cost(lambda dt, tau: MonCorrecteur(dt, ...))

    lim = sim.stability_limit(lambda dt, tau: MonCorrecteur(dt, ...), rpm=4900)

--------------------------------------------------------------------------------
5. DEPENDANCES
--------------------------------------------------------------------------------
Ce fichier suppose presents, dans le meme dossier ou sur le PYTHONPATH :
    plate_model.py, milling_force.py, newmark_solver.py
Ils constituent le modele physique et ne doivent pas etre modifies non plus.
================================================================================
"""
import numpy as np

from plate_model import PlateModel
from milling_force import cutting_coefficients, precompute_alpha_periodic
from newmark_solver import NewmarkSimulator

# ============================================================================
# CONSTANTES FIGEES
# ============================================================================
PLATE_L, PLATE_H, PLATE_T = 0.100, 0.080, 0.004      # m
RHO, YOUNG, POISSON = 2830.0, 69e9, 0.33             # AL6061
MESH_N1, MESH_N2 = 36, 30
N_MODES = 5
F_MEASURED = [536.4, 1071.8, 2778.6, 3358.7, 4122.6]   # Hz, numerisees
# Amortissements MESURES, Tableau 4 de l'article. Les modes 4 et 5 valaient
# 0.30 % / 0.30 % dans la version precedente, contre 0.56 % / 0.35 % publies :
# le mode 4 etait sous-amorti d'un facteur 1.87, ce qui exagerait le spillover
# de commande a 3359 Hz.
ZETA_MODES = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]

# Le patch n'est pas soude a la plaque : il est COLLE. La couche de colle ne
# transmet le cisaillement que sur ~1/Gamma depuis chaque bord, d'ou un
# rendement de transfert eta < 1 qui reduit A LA FOIS le raidissement composite
# et le couplage piezoelectrique (plate_model.shear_lag_efficiency, theorie de
# Crawley & de Luis 1987). Ni G_adh ni t_adh ne sont donnes par l'article :
# valeurs d'un epoxy structural courant. Les mettre a None restitue le collage
# parfait suppose auparavant.
PATCH = dict(x1=0.0, x2=0.020, z1=0.0, z2=0.060,
             d31=175e-12, thickness=0.7e-3, E=63e9, nu=0.35,
             G_adh=1.0e9, t_adh=30e-6)
SENSOR_XZ = (0.100, 0.080)                            # coin superieur droit
V_MAX = 150.0                                         # V, borne amplificateur

# ----------------------------------------------------------------------------
# Repartition modale du couplage piezoelectrique, IDENTIFIEE sur la Fig. 12(b).
#
# H_Pe etait la seule grandeur modale encore prise telle quelle des elements
# finis : les frequences sont recalees sur la mesure (F_MEASURED), les
# amortissements sont mesures (Tableau 4), et D_obs est valide par la Fig. 12(a)
# -- une FRF au point d'impact, dont les residus valent D_i^2 -- a 30 % pres sur
# les cinq modes. H_Pe, lui, n'avait jamais ete confronte a une mesure, et ne la
# reproduisait pas :
#
#     creux de la Fig. 12(b)   mesure : 788 / 1493 / 2913 / 3609 Hz
#                              EF     : 2825 Hz, et lui seul
#     ecart courbe a courbe    EF     : 14.3 dB RMS
#
# La Fig. 12(b) donne pourtant exactement ce qu'il faut. G_b(w) = somme_i
# r_i/(w_i^2 - w^2 + ...) avec r_i = D_obs,i * H_Pe,i : ses poles sont les
# frequences mesurees, ses zeros sont les creux mesures, et poles + zeros
# determinent les r_i A UNE CONSTANTE PRES. Les valeurs ci-dessous sont ces
# r_i/r_1, calcules exactement (racines du polynome), pas ajustes :
#
#     r_i/r_1 identifie      1   0.959   1.646   3.728   10.111  -> 5.7 dB RMS
#     r_i/r_1 elements finis 1  -1.538   0.164   3.625   -4.035  -> 14.3 dB
#
# Ils tombent dans la region de confiance d'un ajustement LIBRE des cinq residus
# (16 motifs de signe x 60 graines, toutes les solutions a moins de 0.5 dB du
# meilleur) sur leurs cinq composantes ; la lecture concurrente, qui ne compte
# que les trois creux profonds et rejette le 4e hors bande, en sort sur quatre
# composantes sur cinq. Voir verification/14_coupling_identification.py.
#
# Le NIVEAU n'est PAS identifie (defaut F9 : le niveau absolu des figures de
# l'article n'est pas exploitable) ; set_identified_coupling conserve donc la
# norme du couplage EF et ne corrige que la repartition. `gain_H` balaie le
# niveau, comme avant.
H_IDENT = (1.0, 0.95943, 1.64565, 3.72754, 10.11124)
COUPLING = 'ident'                                    # 'ident' | 'fem'

N_TEETH, D_TOOL = 3, 0.010                            # fraise 3 dents, 10 mm
HELIX = np.deg2rad(35.0)
RAKE = np.deg2rad(15.0)
KT, KN, MU_C = 925e6, 0.26, 0.20
AE_NOM, FZ_NOM = 0.1e-3, 0.02e-3                      # m, m/dent
STEPS_PER_TOOTH = 82                                  # dt = tau/82, NON MODIFIABLE

SPEEDS_DEFAULT = [3000, 4200, 4900, 6000, 7200]       # tr/min
AP_TEST = 0.25e-3                                     # m, profondeur de reference
GROWTH_MAX = 1.15                                     # garde de croissance (F12)

# Critere de stabilite : TAUX DE CROISSANCE, et non rapport de deux demi-fenetres.
#
# Defaut F12. Le critere d'origine comparait la RMS de la seconde moitie de la
# fenetre a celle de la premiere et declarait instable si le rapport depassait
# GROWTH_MAX. Ce test est AVEUGLE a un mode qui croit lentement sous une reponse
# forcee plus grande que lui : le rapport reste proche de 1 tant que le mode
# instable n'a pas rattrape le regime force. Sur 144 cas dont la verite a ete
# etablie en rejouant chaque simulation jusqu'a T = 1.20 s, il se trompait 16
# fois -- et TOUJOURS dans le meme sens, en declarant STABLE ce qui diverge
# ensuite. Exemple : ESO sous H x2.00 a 6000 tr/min, ap = 0.10 mm : rapport
# 1.070 donc "stable" a T = 0.28 s, et 22.6 um a T = 1.20 s.
#
# On ajuste desormais une DROITE sur log(RMS par blocs) le long de la fenetre,
# ce qui donne un taux de croissance exponentiel sigma [1/s]. C'est une PENTE :
# elle voit un mode lent bien avant qu'il ne domine l'amplitude. Sur la meme
# base de 144 cas, sigma <= SIGMA_MAX se trompe 3 fois, et les trois erreurs
# sont CONSERVATRICES (declare instable ce qui tenait) : l'erreur bascule du
# cote sur.
#
# SIGMA_MAX = 0.05 /s : un mode qui croit de moins de 5 % par seconde est tenu
# pour stable. Sur l'horizon T_LIMIT cela fait 1.4 % de croissance, sous la
# resolution de l'essai. La valeur est au milieu du palier [0.034, 0.060] ou le
# nombre d'erreurs est minimal. Voir verification/15_stability_criterion.py.
SIGMA_MAX = 0.05                                      # 1/s

# Horizons d'integration. CE SONT DES CONSTANTES DE MESURE, pas des details :
# pres du seuil, allonger T abaisse la limite mesuree (~20 % entre 0.30 et
# 0.40 s), parce qu'une instabilite lente a besoin de temps pour se voir. Une
# limite de stabilite n'a donc de sens qu'accompagnee de son horizon. Tout
# resultat destine a etre compare a un autre doit utiliser LE MEME.
# UN SEUL horizon (F12). Il y en avait deux, 0.20 s pour les essais courts et
# 0.28 s pour la bissection, et cet ecart a produit un resultat faux : le crible
# de robustesse de retune.py, qui tournait a T_RUN, laissait passer un reglage
# que run_demo declarait ensuite instable a T_LIMIT. Le verdict de stabilite
# DEPEND de l'horizon ; deux horizons, c'est deux criteres.
# 0.60 s, et non 0.28 : le verdict de stabilite exige que la reponse forcee
# soit ETABLIE, faute de quoi on mesure sa montee. A 0.28 s les cas vraiment
# stables et vraiment instables ne se separent pas (sigma 0.19 contre -0.07) ;
# a 0.60 s ils sont separes de deux decades (|sigma| < 0.02 contre > 2.6).
# Le cout est 2.1 fois celui d'avant. C'est le prix d'un verdict fiable.
T_RUN = 0.60
T_LIMIT = 0.60

# Convention de signe des efforts de coupe. L'article de Du et al. se contredit :
#   * son Eq. (13) s'ecrit  M q" + C q' + (K + a4 DtD) q(t) - a4 DtD q(t-tau)
#                            = ft a3 Dt + H u   ;
#   * mais en enchainant ses propres Eqs. (1), (2), (5), (10) et (A.4) -- avec
#     f_y = -F_y et y_r = -y_P(t) + y_P(t-tau) -- on obtient les memes termes
#     TOUS DE SIGNE INVERSE, ce qui revient a a3 -> -a3, a4 -> -a4.
#
# Ce n'est PAS un detail : les deux conventions donnent des diagrammes de lobes
# quasiment complementaires (limite libre en mm, horizon T_LIMIT) :
#
#     tr/min      3000   3600   4200   4900   5500   6000   7200
#     Eq. (13)   0.0785 0.2650 0.0771 0.0597 0.3705 0.1031 0.2621
#     derivee    0.5237 0.0959 0.4211 0.6003 0.1118 0.0655 0.0684
#
# L'Eq. (13) telle que publiee est celle qui reproduit les donnees de l'article :
# les pics de sa Fig. 13(b) a ~3600 et ~5500 tr/min, le fait que le point S
# (4900 tr/min, ap = 0.30 mm) diverge en Fig. 14(a), et sa Fig. 18 ou la limite
# experimentale sans controle est sous 0.1 mm partout SAUF a 5500 tr/min. La
# convention derivee inverse ce motif et rendrait le point S stable. L'erreur de
# signe est donc dans les equations intermediaires de l'article, pas dans son
# Eq. (13) : la base suit l'Eq. (13), et c'est le bon choix.
#
# SimBase(force_sign=-1.0) reproduit le tableau ci-dessus.
FORCE_SIGN = +1.0


def _growth_rate(y, t, nblock=12):
    """Taux de croissance exponentiel sigma [1/s] sur la fenetre d'analyse.

    On decoupe la fenetre en blocs, on prend la RMS de chaque bloc, et on ajuste
    au moindre carre log(RMS) contre le temps : la pente EST sigma. Passer au
    log rend la mesure insensible a l'amplitude, donc au niveau de la reponse
    forcee -- c'est ce qui permet de voir un mode instable encore petit devant
    elle. Voir SIGMA_MAX pour la calibration et le defaut F12 qu'il corrige.
    """
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    n = len(y)
    if n < 8:
        return 0.0
    nb = max(3, min(nblock, n//4))
    idx = np.array_split(np.arange(n), nb)
    tb = np.array([t[k].mean() for k in idx])
    rb = np.array([np.sqrt(np.mean(y[k]**2)) for k in idx])
    m = rb > 0
    if int(m.sum()) < 3 or (tb[m].max() - tb[m].min()) <= 0:
        return 0.0
    A = np.vstack([tb[m], np.ones(int(m.sum()))]).T
    return float(np.linalg.lstsq(A, np.log(rb[m]), rcond=None)[0][0])


class SimBase:
    """Plaque + patch + capteur + coupe. Le correcteur est fourni a l'appel.

    patch / zeta : surchargent PATCH et ZETA_MODES pour cette instance
        seulement. Passer par ces arguments plutot que de reaffecter les
        constantes du module : une reaffectation reste active pour toutes les
        instances construites ensuite dans le meme processus.
    force_sign : +1 pour l'Eq. (13) telle que publiee (defaut), -1 pour la
        convention obtenue en enchainant les Eqs. (1), (2), (5), (10), (A.4) du
        meme article. Voir le commentaire de FORCE_SIGN : l'article se
        contredit, la base suit la forme publiee, et ce parametre existe pour
        verifier qu'un resultat ne depend pas de ce choix.
    gain_H : facteur multiplicatif sur H_Pe_modal. Le NIVEAU du couplage
        piezoelectrique n'est PAS valide experimentalement (voir la section 1
        du docstring de module) : ce facteur existe pour balayer cette
        incertitude. Construire la plaque nominale (gain_H = 1) pour SYNTHETISER
        le correcteur, puis une plaque perturbee pour le SIMULER, donne un vrai
        essai de robustesse a l'erreur de gain actionneur.
    coupling : 'ident' (defaut) prend la repartition modale de H_Pe identifiee
        sur la Fig. 12(b) de l'article (constante H_IDENT) ; 'fem' garde celle
        des elements finis, qui ne reproduit aucun des creux mesures. Dans les
        deux cas la NORME de H_Pe est la meme, seule sa repartition change.
    """

    def __init__(self, verbose=False, patch=None, zeta=None, gain_H=1.0,
                 force_sign=None, coupling=None):
        patch = PATCH if patch is None else patch
        zeta = ZETA_MODES if zeta is None else zeta
        coupling = COUPLING if coupling is None else coupling
        if coupling not in ('ident', 'fem'):
            raise ValueError("coupling doit valoir 'ident' ou 'fem'")
        p = PlateModel(PLATE_L, PLATE_H, PLATE_T, RHO, YOUNG, POISSON,
                       N1=MESH_N1, N2=MESH_N2, n_modes=N_MODES,
                       zeta_modes=zeta, verbose=False)
        p.precompute_Dp(zp_pos=PLATE_H - 0.15e-3, n_pos=2001)
        p.set_observation(*SENSOR_XZ)
        p.add_piezo_patch(patch['x1'], patch['x2'], patch['z1'], patch['z2'],
                          patch['d31'], patch['thickness'], patch['E'], patch['nu'],
                          G_adh=patch.get('G_adh'), t_adh=patch.get('t_adh'))
        if coupling == 'ident':
            if p.n_modes > len(H_IDENT):
                raise ValueError(
                    f"H_IDENT n'est identifie que sur {len(H_IDENT)} modes "
                    f"(les cinq de la Fig. 12) ; construire une base a "
                    f"{p.n_modes} modes impose coupling='fem'")
            # Troncature : sur un modele reduit a n < 5 modes, on garde les n
            # premiers residus identifies. C'est la meme troncature que celle
            # deja subie par les frequences et les amortissements.
            p.set_identified_coupling(H_IDENT[:p.n_modes])
        if gain_H != 1.0:
            p.H_Pe_modal = np.asarray(p.H_Pe_modal, float)*gain_H
        p.calibrate_frequencies(F_MEASURED)
        self.plate = p
        self.patch_cfg = patch
        self.coupling = coupling
        self.gain_H = gain_H
        self.force_sign = FORCE_SIGN if force_sign is None else float(force_sign)
        self.k1c, self.k2c = cutting_coefficients(KN, MU_C, HELIX, RAKE)
        self._cache = {}
        if verbose:
            self.describe()

    # -- proprietes modales utiles au reglage d'un observateur ---------------
    @property
    def freqs(self):
        """Frequences propres calibrees, Hz."""
        return np.asarray(self.plate.omega_n).ravel()/(2*np.pi)

    @property
    def modal(self):
        """Vecteurs modaux : H (entree patch), D_obs (capteur), D_tool (outil)."""
        return dict(H=np.asarray(self.plate.H_Pe_modal).ravel(),
                    D_obs=np.asarray(self.plate.D_obs).ravel(),
                    D_tool=np.asarray(self.plate.Dp_array[:, 0]).ravel(),
                    zeta=np.asarray(self.plate.zeta_modes).ravel(),
                    omega=np.asarray(self.plate.omega_n).ravel())

    def describe(self):
        m = self.modal
        print("frequences (Hz) :", np.round(self.freqs, 1))
        print("H (patch)       :", np.round(m['H'], 5))
        print("D_obs (capteur) :", np.round(m['D_obs'], 3))
        print("D_tool (outil)  :", np.round(m['D_tool'], 3))
        raw = np.abs(m['H']*m['D_obs'])
        print("|H D| brut      :", np.round(raw, 4))
        nrm = raw/m['omega']**2
        print("|H D|/w^2 norm. :", np.array2string(nrm/nrm.max(), precision=4))

    # -- setup d'une condition de coupe -------------------------------------
    def _setup(self, rpm, ap, T, ae, fz, tool_pos):
        key = (rpm, round(ap, 9), round(T, 4), round(ae, 9), round(fz, 9), tool_pos)
        if key not in self._cache:
            tau = 60.0/(N_TEETH*rpm)
            dt = tau/STEPS_PER_TOOTH
            s = NewmarkSimulator(self.plate, dt=dt, T_end=T, ft=fz, tau=tau,
                                 verbose=False, v_max=V_MAX)
            phi = np.pi - np.arccos(1 - ae/(D_TOOL/2))
            a3, a4 = precompute_alpha_periodic(
                dt, STEPS_PER_TOOTH, s.nstep, 2*np.pi*rpm/60, N_TEETH, D_TOOL/2,
                HELIX, phi, np.pi, PLATE_H - ap, PLATE_H, self.k1c, self.k2c, KT)
            a3 = self.force_sign*a3
            a4 = self.force_sign*a4
            self._cache[key] = (s, a3, a4, dt, tau)
        return self._cache[key]

    # -- simulation elementaire ---------------------------------------------
    def run(self, controller, rpm=4900, ap=AP_TEST, T=T_RUN, ae=AE_NOM,
            fz=FZ_NOM, tool_pos=0, full_pass=False, keep_signals=False):
        """Simule une coupe. controller = objet step/reset, ou None.

        tool_pos : indice de station de l'outil (0 = debut d'arete, station la
                   moins stable). Ignore si full_pass=True.
        full_pass : l'outil parcourt les 100 mm ; T est alors recalcule.
        Retour : dict avec stable, rms_um, peak_um, rms_u, peak_u, growth, t_end.
        """
        if full_pass:
            T = PLATE_L/(fz*N_TEETH*rpm/60.0)
        s, a3, a4, dt, tau = self._setup(rpm, ap, T, ae, fz, tool_pos)
        if full_pass:
            v = fz*N_TEETH*rpm/60.0
            kp = np.clip(np.round(np.minimum(v*s.t_vec, PLATE_L)/PLATE_L*2000
                                  ).astype(int), 0, 2000)
        else:
            kp = np.full(s.nstep, int(tool_pos), dtype=int)
        if controller is not None and hasattr(controller, 'reset'):
            controller.reset()
        r = s.simulate(a3, a4, kp, controller=controller, progress=False,
                       stop_threshold=5e-3 if full_pass else 1e-4)
        i = r['stop_idx']
        diverged = r['diverged_at'] > 0
        i0 = min(s.nstep//3, max(0, i - 1))
        y2 = r['y'][i0:i+1]
        if y2.size < 4:
            y2 = r['y'][:max(4, i+1)]
        n2 = max(1, len(y2)//2)
        growth = (np.sqrt(np.mean(y2[n2:]**2))
                  / max(np.sqrt(np.mean(y2[:n2]**2)), 1e-18))
        # sigma se mesure sur la DERNIERE MOITIE du releve : la reponse forcee
        # d'une plaque a zeta = 0.17 % met ~0.3 s a s'etablir, et tant qu'elle
        # monte encore une droite ajustee sur log(RMS) lit cette montee comme
        # une croissance. Sur la premiere moitie sigma vaut ~1 /s meme pour une
        # coupe parfaitement stable ; sur la seconde il tombe a |sigma| < 0.02.
        yv = r['y'][:i+1]
        tv = r['t'][:i+1]
        h = len(yv)//2
        sigma = _growth_rate(yv[h:], tv[h:])
        stable = (not diverged) and sigma <= SIGMA_MAX
        out = dict(stable=bool(stable), diverged=bool(diverged), sigma=sigma,
                   rms_um=float(np.sqrt(np.mean(y2**2))*1e6),
                   peak_um=float(np.abs(r['y'][:i+1]).max()*1e6),
                   rms_u=float(np.sqrt(np.mean(r['u'][i0:i+1]**2))),
                   peak_u=float(np.abs(r['u'][:i+1]).max()),
                   growth=float(growth), t_end=float(r['t'][i]), dt=dt, tau=tau)
        if keep_signals:
            out['t'] = r['t'][:i+1]
            out['y'] = r['y'][:i+1]
            out['u'] = r['u'][:i+1]
        return out

    # -- cout multi-vitesses, identique pour tous les correcteurs -----------
    def multi_speed_cost(self, make_ctrl, speeds=None, ap=AP_TEST, T=T_RUN,
                         penalty=12.0, w_worst=0.5):
        """make_ctrl(dt, tau) -> correcteur. Retourne un scalaire a minimiser.

        Une vitesse instable coute `penalty`. Le terme worst-case evite les
        reglages excellents a une vitesse et mediocres ailleurs.
        """
        speeds = speeds or SPEEDS_DEFAULT
        tot = 0.0
        worst = 0.0
        for rpm in speeds:
            tau = 60.0/(N_TEETH*rpm)
            dt = tau/STEPS_PER_TOOTH
            try:
                c = make_ctrl(dt, tau)
            except Exception:
                tot += penalty
                continue
            r = self.run(c, rpm=rpm, ap=ap, T=T)
            if not r['stable']:
                tot += penalty
            else:
                tot += r['rms_um'] + 0.05*r['rms_u']
                worst = max(worst, r['rms_um'])
        return tot/len(speeds) + w_worst*worst

    # -- limite de stabilite par bissection ---------------------------------
    def stability_limit(self, make_ctrl, rpm=4900, lo=0.02e-3, hi=1.5e-3,
                        T=T_LIMIT, tol=2e-5):
        """Plus grande profondeur de passe stable. make_ctrl=None -> boucle ouverte."""
        def ok(ap):
            tau = 60.0/(N_TEETH*rpm)
            dt = tau/STEPS_PER_TOOTH
            c = None
            if make_ctrl is not None:
                try:
                    c = make_ctrl(dt, tau)
                except Exception:
                    return False
            return self.run(c, rpm=rpm, ap=ap, T=T)['stable']
        if not ok(lo):
            return 0.0
        if ok(hi):
            return hi
        while hi - lo > tol:
            mid = 0.5*(lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return 0.5*(lo + hi)

    # -- reponse frequentielle mesuree, sans hypothese de linearite ---------
    def receptance(self, make_ctrl, rpm=4900, f_lo=64.0, f_hi=1600.0,
                   amp=5e-3, n_period=3):
        """|Y/F| mesuree par injection d'un multisinus PERIODIQUE a la place de
        l'effort de coupe, couplage regeneratif desactive.

        Cette voie est la seule fiable : l'extraction lineaire par reponse
        impulsionnelle echoue pour les correcteurs dont la dynamique propre est
        instable (le cas de la plupart des schemas a observateur), leur reponse
        impulsionnelle divergeant.
        Retour : (frequences, |Y/F| en um/N) ou (None, None) si divergence.
        """
        tau = 60.0/(N_TEETH*rpm)
        dt = tau/STEPS_PER_TOOTH
        NP = 5022
        s = NewmarkSimulator(self.plate, dt=dt, T_end=(NP*n_period + 2)*dt,
                             ft=1.0, tau=tau, verbose=False, v_max=V_MAX)
        n = s.nstep
        NP = n//n_period
        df = 1.0/(NP*dt)
        ks = np.arange(int(round(f_lo/df)), int(round(f_hi/df)) + 1)
        tp = np.arange(NP)*dt
        ph = np.pi*np.arange(len(ks))**2/len(ks)     # phases de Schroeder
        base = np.zeros(NP)
        for i, k in enumerate(ks):
            base += np.cos(2*np.pi*k*df*tp + ph[i])
        base /= np.abs(base).max()
        F = np.tile(base, n_period + 1)[:n]*amp
        c = None
        if make_ctrl is not None:
            try:
                c = make_ctrl(dt, tau)
            except Exception:
                return None, None
        r = s.simulate(F, np.zeros(n), np.zeros(n, dtype=int), controller=c,
                       progress=False, stop_threshold=1e-2)
        if r['diverged_at'] > 0:
            return None, None
        k0 = n - NP - 2
        Y = np.fft.rfft(r['y'][k0:k0+NP])
        Ff = np.fft.rfft(F[k0:k0+NP])
        fr = np.fft.rfftfreq(NP, dt)
        return fr[ks], np.abs(Y[ks]/Ff[ks])*1e6


# ============================================================================
# VERIFICATION DU MODELE
# ============================================================================
def check_model(verbose=True, sim=None):
    """Refait les controles de validation. Leve AssertionError si un ecart
    depasse la tolerance. A lancer une fois au debut d'une session.

    sim : base a controler. Par defaut SimBase(), c.-a-d. la base nominale ;
        passer SimBase(coupling='fem') montre ce que le controle 4 attrape."""
    if sim is None:
        sim = SimBase()
    ok = True

    f = sim.freqs
    err = np.abs(f/np.array(F_MEASURED) - 1)
    if verbose:
        print("1. frequences apres calibration")
        for i in range(N_MODES):
            print(f"   mode {i+1} : {f[i]:8.1f} Hz  (mesure {F_MEASURED[i]:8.1f}, "
                  f"ecart {100*err[i]:+.3f} %)")
    assert err.max() < 1e-3, "calibration des frequences incorrecte"

    # Les antiresonances de reference proviennent de la Fig. 12(a) de Du et al.,
    # ou le marteau ET le capteur sont au MEME point (coin superieur droit) :
    # c'est une reception AU POINT DE FRAPPE. Les residus sont donc D_obs*D_obs,
    # tous positifs, et les zeros s'intercalent strictement entre les poles --
    # ce que la mesure confirme (734 < 1072, 2161 < 2779, 3001 < 3359, 3805 <
    # 4123). Utiliser D_obs*D_tool comparerait une reception de TRANSFERT
    # (outil en x=0 -> capteur) a une mesure au point de frappe : residus de
    # signes mixtes, motif de zeros sans rapport.
    ff = np.linspace(60, 4600, 60000)
    w = 2*np.pi*ff
    m = sim.modal
    den = ((m['omega'][:, None]**2 - w[None, :]**2)
           + 2j*m['zeta'][:, None]*m['omega'][:, None]*w[None, :])
    G = np.abs(np.sum((m['D_obs']**2)[:, None]/den, axis=0))
    lg = np.log(G)
    anti = [ff[i] for i in range(1, len(G)-1) if lg[i] < lg[i-1] and lg[i] < lg[i+1]]
    target = [734, 2161, 3001, 3805]
    if verbose:
        print("2. antiresonances (reception au point de frappe, Fig. 12a)")
        print(f"   mesurees : {target}")
        print(f"   modele   : {[round(a) for a in anti]}")
    assert len(anti) >= 4, "le modele ne produit pas quatre antiresonances"
    for t in target:
        d = min(abs(a - t) for a in anti)
        if verbose:
            print(f"   {t:5d} Hz -> ecart {d:5.0f} Hz ({100*d/t:+.1f} %)")
        if d/t > 0.05:
            ok = False
            print(f"   ATTENTION : antiresonance {t} Hz mal reproduite (ecart {d:.0f} Hz)")

    # Niveau absolu. On ne peut PAS le confronter a la Fig. 12 (son echelle en
    # dB est inexploitable, cf. section 1) ; on verifie donc la coherence
    # INTERNE du modele, ce qui attrape une erreur de normalisation modale ou de
    # D_obs sans rien emprunter a l'article :
    #   - la souplesse statique du modele modal tronque doit approcher la
    #     resolution statique EF exacte (memes K, tous les ddl) ;
    #   - cette souplesse doit depasser celle d'une poutre encastree de pleine
    #     largeur, qui est une borne inferieure stricte pour une charge au coin.
    from scipy.sparse.linalg import spsolve
    from kirchhoff_q4 import shape_at_point
    pl = sim.plate
    N_obs = shape_at_point(*SENSOR_XZ, MESH_N1, MESH_N2,
                           pl.lex, pl.ley, pl.n1)[pl.DOFf]
    C_exact = float(N_obs @ spsolve(pl.K_free.tocsc(), N_obs))
    C_modal = float(np.sum(np.asarray(pl.D_obs)**2 / np.asarray(pl.omega_n)**2))
    D_flex = YOUNG*PLATE_T**3/(12*(1 - POISSON**2))
    C_beam = PLATE_H**3/(3*D_flex*PLATE_L)
    if verbose:
        print("3. niveau absolu (coherence interne, sans recours a la Fig. 12)")
        print(f"   souplesse statique EF exacte      : {C_exact*1e6:7.3f} um/N")
        print(f"   modele modal a {N_MODES} modes           : {C_modal*1e6:7.3f} um/N"
              f"  ({100*C_modal/C_exact:.1f} % de l'exacte)")
        print(f"   borne inf. poutre pleine largeur  : {C_beam*1e6:7.3f} um/N"
              f"  (rapport {C_exact/C_beam:.2f})")
        print(f"   Fig. 12(a) lue a 1 um/N           :   43.83 um/N"
              f"  -> facteur {43.83e-6/C_exact:.2f}, ECHELLE INEXPLOITABLE")
        print("   => le NIVEAU de H_Pe n'est pas valide ; balayer gain_H.")
    assert 0.80 < C_modal/C_exact < 1.05, \
        "troncature modale incoherente avec la statique exacte"
    assert 1.0 < C_exact/C_beam < 3.0, \
        "souplesse statique hors des bornes physiques d'une plaque encastree"

    # Repartition modale du couplage. Les checks 2 et 3 ne portent que sur
    # D_obs (leurs residus valent D_i^2) : ils passeraient tels quels avec un
    # H_Pe entierement faux, ce qui a longtemps ete le cas. Ce test-ci ferme
    # cette porte. Les zeros de la fonction de transfert tension -> capteur
    # sont une signature de SIGNE, insensible a toute mise a l'echelle de H_Pe
    # -- donc au niveau non valide de la section 1.
    r_c = np.asarray(sim.plate.H_Pe_modal).ravel()*np.asarray(sim.plate.D_obs)
    s2 = np.asarray(sim.plate.omega_n).ravel()**2
    P = sum(r_c[i]*np.poly(np.delete(s2, i)) for i in range(len(r_c)))
    zr = np.roots(P)
    zr = zr[np.abs(zr.imag) <= 1e-6*max(np.abs(zr.real).max(), 1e-30)].real
    zc = np.sort(np.sqrt(zr[zr > 0]))/(2*np.pi)
    NOTCH_MEAS = np.array([787.5, 1493.4, 2912.8, 3608.9])   # Fig. 12(b)
    if verbose:
        print("4. creux de la Fig. 12(b) (tension -> capteur) : "
              "repartition modale de H_Pe")
        print(f"   mesures : {np.round(NOTCH_MEAS, 0)}")
        print(f"   modele  : {np.round(zc, 0)}")
    if len(zc) != len(NOTCH_MEAS):
        ok = False
        if verbose:
            print(f"   ATTENTION : {len(zc)} creux calcules contre "
                  f"{len(NOTCH_MEAS)} mesures — la repartition modale de H_Pe "
                  f"est fausse (coupling='fem' ?)")
    else:
        e = np.abs(zc - NOTCH_MEAS)/NOTCH_MEAS*100
        if verbose:
            print(f"   ecarts  : {np.round(e, 2)} %  (max {e.max():.2f} %)")
        if e.max() > 3.0:
            ok = False
            if verbose:
                print("   ATTENTION : creux mal reproduits")

    lim = sim.stability_limit(None, rpm=4900)
    if verbose:
        print(f"5. limite de stabilite en boucle ouverte a 4900 tr/min : "
              f"{lim*1e3:.4f} mm  (reference de cette base : 0.0489 mm)")
        print("   cette valeur depend de l'horizon T et de la fenetre de mesure ;")
        print("   c'est la reference de CETTE base, a ne pas comparer a une valeur")
        print("   obtenue avec d'autres reglages d'evaluation.")
    assert abs(lim*1e3 - 0.0489) < 0.008, "limite libre hors tolerance"

    if verbose:
        print("\nbase valide." if ok else "\nbase utilisable, voir avertissements.")
    return sim


if __name__ == "__main__":
    sim = check_model()
    print()
    sim.describe()
    print("\nessai en boucle ouverte a 0.05 mm (coupe stable) :")
    r = sim.run(None, rpm=4900, ap=0.05e-3, T=0.20)
    print(f"   stable={r['stable']}  RMS={r['rms_um']:.4f} um  "
          f"croissance={r['growth']:.3f}")
