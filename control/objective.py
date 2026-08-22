"""
objective.py — Fonction objectif et contraintes, IDENTIQUES aux deux
correcteurs
=====================================================================
C'est ici que se joue l'equite de la comparaison : FOPID et ADRC-FOPID sont
notes par le MEME code, avec les MEMES contraintes. Seule change la fonction
qui fabrique le (A, B, C, D) a partir du vecteur de decision.

OBJECTIF — maximiser la profondeur axiale limite du fraisage.
Le critere exact est a_p,lim obtenu par bissection sur le rayon spectral de
Floquet, soit 12 a 15 evaluations par position : trop cher dans une boucle
PSO. On l'ESTIME donc a partir de quelques profondeurs sondes, en interpolant
le premier croisement de max_x log rho(a_p, x) par zero (voir
_ap_from_margins). J est ainsi en millimetres, dans l'unite de la grandeur
visee, et son classement coincide avec elle a la resolution des sondes pres.
Les a_p,lim VRAIS sont ensuite bissectes a pleine resolution (m = 200, cinq
positions) pour les seuls correcteurs retenus.

CONTRAINTES (penalisees, identiques des deux cotes) :
  1. stabilite nominale sans coupe : max Re(lambda) <= -1 s^-1 ;
  2. marge de module : max_w |S| <= MS_MAX, avec S = 1/(1 - P_u K) ;
  3. effort d'actionneur : max_w |K S P_f| <= V_PER_N volts par newton de
     force de coupe, evalue a toutes les positions de synthese.
Les contraintes 1-3 sont peu couteuses et servent de crible : une particule
qui les viole est rejetee AVANT tout calcul de Floquet.
"""
import numpy as np

import config as C
from plate_model import plant_vectors, plant_frf
from fopid import ss_frf
from closed_loop import period_maps, spectral_radius
from milling_dynamics import N_TEETH

# Fond logarithmique [Hz]. La borne BASSE compte autant que la finesse : avec
# une grille demarrant a 3.16 Hz, le FOPID retenu au protocole B affichait
# Ms = 1.985 sur la grille et |S| = 28.7 EN DESSOUS d'elle (pole de boucle
# fermee a 0.11 Hz : le gain de boucle au continu vaut +0.965, la boucle perd
# presque toute sa raideur statique). L'ADRC-FOPID, lui, valait 0.227 au meme
# endroit : la contrainte de marge de module ne pesait donc pas du tout de la
# meme facon sur les deux structures. On couvre desormais toute la bande ou la
# realisation d'Oustaloup a un sens, de 0.01 Hz a 126 kHz (w_b = 0.159 Hz,
# w_h = 100 kHz), et un peu au-dela des deux cotes.
_F_LOG = np.logspace(-2.0, 5.1, 300)


def _con_grid(plate, poles=None, n_modes=None):
    """Grille de frequences pour Ms et l'effort, RESOLUE sur les resonances.

    Une grille logarithmique a 140 points a un pas de 6.1 %, alors que la
    largeur a mi-puissance des modes vaut 2*zeta*f = 0.34 % a 1.12 %, et
    qu'un pole de boucle fermee peut etre bien plus fin encore : le correcteur
    ADRC-FOPID retenu laissait un pole a 2804 Hz avec zeta = 2.2e-4, soit
    0.044 %, cent fois plus etroit qu'un pas de grille. Les pics de |S| et de
    |K S P_f| n'etaient donc tout simplement pas echantillonnes : les deux
    correcteurs etaient declares conformes (Ms = 1.79 et 1.87) alors que leurs
    vrais maxima valaient 6.1 et 36.0, et l'effort de l'ADRC 2620 V/N contre
    une limite de 450. La contrainte ne contraignait rien, et pas de la meme
    facon pour les deux structures.

    On ajoute donc des grappes serrees autour de chaque mode de la plaque ET
    autour de chaque pole de boucle fermee peu amorti (dont on dispose deja :
    le crible de stabilite nominale les calcule).
    """
    g = [_F_LOG]
    n = C.N_MODES_OBJ if n_modes is None else n_modes
    fs = [float(f) for f in np.asarray(plate.freq_n)[:n]]
    if poles is not None:
        for lam in np.atleast_1d(poles):
            wi, si = abs(lam.imag), abs(lam.real)
            if wi > 6.0 and si < 0.3 * wi:          # resonant, peu amorti
                fs.append(wi / (2 * np.pi))
    for f0 in fs:
        if 3.0 < f0 < 2.0e4:
            g.append(f0 * np.linspace(0.97, 1.03, 121))
    return np.unique(np.concatenate(g))


# ---------------------------------------------------------------------------
def nominal_poles(plate, ss, n_modes=None):
    """Valeurs propres de la boucle fermee NOMINALE (sans coupe).

    `n_modes` vaut par defaut le modele VU PAR L'OPTIMISEUR (C.N_MODES_OBJ) :
    en protocole A c'est le modele reduit du papier, en protocole B le modele
    complet. Les rapports finals passent explicitement C.N_MODES.
    """
    n = C.N_MODES_OBJ if n_modes is None else n_modes
    w, z, H, D_obs, _ = plant_vectors(plate, n)
    if ss is None:
        # BOUCLE OUVERTE. Meme convention que closed_loop.build_matrices, qui
        # accepte ctrl=None depuis toujours : sans elle, la ligne « boucle
        # ouverte » du tableau des poles leve TypeError — et c'est exactement
        # ce qu'elle a fait des que `robust_poles` est passe par ici.
        Ac = np.zeros((0, 0))
        Bc = np.zeros((0, 1))
        Cc = np.zeros((1, 0))
        Dc = np.zeros((1, 1))
    else:
        Ac, Bc, Cc, Dc = [np.atleast_2d(np.asarray(m, float)) for m in ss]
    nc = Ac.shape[0]
    A = np.zeros((2 * n + nc, 2 * n + nc))
    A[:n, n:2 * n] = np.eye(n)
    A[n:2 * n, :n] = -np.diag(w**2) + float(Dc[0, 0]) * np.outer(H, D_obs)
    A[n:2 * n, n:2 * n] = -np.diag(2 * z * w)
    if nc:
        A[n:2 * n, 2 * n:] = np.outer(H, Cc.ravel())
        A[2 * n:, :n] = np.outer(Bc.ravel(), D_obs)
        A[2 * n:, 2 * n:] = Ac
    return np.linalg.eigvals(A)


def nominal_max_re(plate, ss, pd=None, n_modes=None):
    """max Re du spectre de la boucle fermee SANS coupe, terme retarde compris.

    Sans `pd`, c'est exactement `nominal_poles(...).real.max()`.

    Avec `pd`, ce n'est plus un probleme aux valeurs propres : les gains de
    l'Eq. (30) vivent sur l'etat retarde, la boucle nominale est une equation
    a retard et son spectre est infini. On passe par la monodromie sur une
    periode de dent — rho = |e^{lambda tau}| pour le mode dominant, donc
    log(rho)/tau EST le maximum cherche. Pour un systeme sans retard la
    formule redonne l'autre a l'identique, ce que `tests/` verifie.
    """
    n = C.N_MODES_OBJ if n_modes is None else n_modes
    if pd is None:
        return float(nominal_poles(plate, ss, n_modes=n).real.max())
    m = C.M_FLOQUET
    maps, tau = period_maps(plate, C.RPM_DESIGN, 0.0, 0.5 * plate.lp,
                            ctrl=ss, pd=pd, n_modes=n, m=m,
                            coeff_mode='time', coeff_scale=C.SIGN_SIM,
                            ae=C.AE)
    rho = spectral_radius(maps, m, maps[0][0].shape[0])
    if not np.isfinite(rho):
        return np.inf
    return float(np.log(max(rho, 1e-300)) / tau)


def frequency_metrics(plate, ss, positions=None, f=None, n_modes=None,
                      poles=None, pd=None, tau=None):
    """(Ms, Vmax) : marge de module et effort maximal en V/N.

    `pd = (K_Pp, K_Pd)` ajoute le CONTROLE A RETARD ACTIF de l'Eq. (30).
    Cette loi, u_d = K_Pp y(t - tau) + K_Pd y'(t - tau), est parfaitement LTI :
    sa reponse frequentielle vaut (K_Pp + K_Pd jw) e^{-jw tau}, exactement. Elle
    passe donc par les MEMES contraintes Ms et effort que toutes les autres
    structures — le terme retarde n'est pas exempte du protocole commun, il y
    entre par sa vraie FRF. C'est la condition pour que la reference « robuste
    + retard » soit comparee au meme prix que le reste.
    """
    f = _con_grid(plate, poles, n_modes) if f is None else np.asarray(f, float)
    pos = C.POSITIONS_DESIGN if positions is None else positions
    n = C.N_MODES_OBJ if n_modes is None else n_modes
    w = 2 * np.pi * f
    K = ss_frf(ss, w)
    if pd is not None:
        if tau is None:
            raise ValueError('tau est obligatoire avec pd : sans lui le terme '
                             'retarde serait note sans son retard')
        K = K + (float(pd[0]) + float(pd[1]) * 1j * w) * np.exp(-1j * w * tau)
    Pu, _ = plant_frf(plate, f, n)
    S = 1.0 / (1.0 - Pu * K)
    Ms = float(np.max(np.abs(S)))
    v = 0.0
    for fr in pos:
        _, Pf = plant_frf(plate, f, n, x_force=fr * plate.lp)
        v = max(v, float(np.max(np.abs(K * S * Pf))))
    return Ms, v


def floquet_margin(plate, ss, rpm, ap, x_pos, m=None, pd=None):
    """log(rho) de la monodromie pour un correcteur LTI donne.

    Le calcul est le MEME pendant l'optimisation et pour les resultats
    publies : Arnoldi rend le spectre exact, il n'y a donc plus de reglage
    "economique" a degrader (les anciens C.N_PERIOD_* n'ont plus d'objet).
    Seul C.M_FLOQUET_PSO < C.M_FLOQUET distingue encore les deux regimes, et
    c'est une finesse de DISCRETISATION, pas une precision d'estimateur.
    """
    m = C.M_FLOQUET_PSO if m is None else m
    maps, _ = period_maps(plate, rpm, ap, x_pos, ctrl=ss, pd=pd,
                          n_modes=C.N_MODES_OBJ, m=m, coeff_mode='time',
                          coeff_scale=C.SIGN_SIM, ae=C.AE)
    rho = spectral_radius(maps, m, maps[0][0].shape[0])
    if not np.isfinite(rho):        # divergence violente : borne haute graduee
        return 50.0
    return float(np.clip(np.log(max(rho, 1e-300)), -50.0, 50.0))


# ---------------------------------------------------------------------------
def evaluate(plate, ss, rpm=None, probes=None, positions=None, m=None,
             detail=False, pd=None):
    if ss is None:
        # `Design.build` rend None quand la SYNTHESE echoue — cas propre aux
        # structures H-infini et mu, ou il existe des ponderations pour
        # lesquelles aucun correcteur n'atteint le gamma demande. On le note
        # comme un echec au meme titre qu'un correcteur nominalement instable,
        # plutot que de le contourner : c'est une propriete de la structure,
        # et l'escamoter fausserait la comparaison en sa faveur.
        info = dict(feasible=False, reason='synthese impossible', Ms=np.nan,
                    V=np.nan, J=-1e4, max_re=np.nan)
        return (info['J'], info) if detail else info['J']
    try:
        return _evaluate(plate, ss, rpm, probes, positions, m, detail, pd)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        info = dict(feasible=False, reason='echec numerique', Ms=np.nan,
                    V=np.nan, J=-1e4, max_re=np.nan)
        return (info['J'], info) if detail else info['J']


def _evaluate(plate, ss, rpm, probes, positions, m, detail, pd=None):
    """Note d'un correcteur. Retourne J (a MAXIMISER) et, si demande, le
    detail des contraintes."""
    rpm = C.RPM_DESIGN if rpm is None else rpm
    probes = C.AP_PROBE if probes is None else probes
    pos = C.POSITIONS_DESIGN if positions is None else positions
    info = dict(feasible=False, reason='', Ms=np.nan, V=np.nan, J=-np.inf)

    # -- crible 1 : stabilite nominale
    # Le seuil etait -1 s^-1. Il n'est PAS neutre vis-a-vis des structures :
    # l'integrateur fractionnaire du FOPID est realise par Oustaloup, donc
    # borne en bande a w_b = 2 pi rad/s, et son pole le plus lent est
    # structurellement loin de zero ; l'etat z3 de l'observateur etendu est un
    # VRAI integrateur en s = 0, dont l'image en boucle fermee se place
    # typiquement entre -0.2 et -0.5 s^-1, c'est-a-dire dans la bande rejetee.
    # Mesure par echantillonnage (2500 tirages par boitier) : le seuil -1 tuait
    # 58.3 % des ADRC-FOPID nominalement STABLES contre 1.0-1.7 % des FOPID, et
    # la penalite y etait plate (-1000 exactement), donc sans gradient. On ne
    # rejette plus que l'instabilite reelle, avec une penalite graduee.
    ev = nominal_poles(plate, ss)
    if pd is None:
        mre = float(np.max(ev.real))
        info['max_re'] = mre
        if not np.isfinite(mre) or mre > 0.0:
            info['reason'] = 'boucle nominale instable'
            info['J'] = -1e3 - (mre if np.isfinite(mre) else 1e3)
            return (info['J'], info) if detail else info['J']
    else:
        # AVEC RETARD, LE NOMINAL N'EST PLUS UN PROBLEME AUX VALEURS PROPRES.
        # La boucle fermee devient x' = A x + A_tau x(t - tau) : une equation
        # differentielle A RETARD, de spectre infini. `nominal_poles` ne voit
        # que la partie non retardee et declarerait stable une boucle que le
        # terme retarde deteriore — exactement le mode de panne que ce
        # controleur est cense exploiter.
        #
        # Le bon test existe deja : le rayon spectral de la monodromie A
        # PROFONDEUR NULLE, ou le terme regeneratif s'annule et ou il ne reste
        # que la plaque et la loi retardee. C'est le meme moteur que
        # l'objectif, donc aucun estimateur supplementaire n'est introduit.
        g0 = max(floquet_margin(plate, ss, rpm, 0.0, fr * plate.lp, m=m, pd=pd)
                 for fr in pos)
        info['max_re'] = float(g0)          # log rho, pas un pole : voir plus haut
        if not np.isfinite(g0) or g0 > 0.0:
            info['reason'] = 'boucle nominale instable (retard, log rho > 0)'
            info['J'] = -1e3 - (g0 if np.isfinite(g0) else 1e3)
            return (info['J'], info) if detail else info['J']

    # -- crible 2/3 : marge de module et effort, sur une grille resolue par
    #    les poles de boucle fermee qu'on vient justement de calculer
    tau = 60.0 / (N_TEETH * rpm)
    Ms, V = frequency_metrics(plate, ss, pos, poles=ev, pd=pd, tau=tau)
    info['Ms'], info['V'] = Ms, V
    pen = 0.0
    if Ms > C.MS_MAX:
        pen += 10.0 * (Ms / C.MS_MAX - 1.0)
    if V > C.V_PER_N:
        pen += 10.0 * (V / C.V_PER_N - 1.0)
    if pen > 0.0:
        info['reason'] = f'contrainte violee (Ms={Ms:.2f}, V={V:.0f} V/N)'
        info['J'] = -100.0 - pen
        return (info['J'], info) if detail else info['J']

    # -- objectif : estimation de a_p,lim par interpolation du croisement
    margins = [max(floquet_margin(plate, ss, rpm, ap, fr * plate.lp, m=m,
                                  pd=pd)
                   for fr in pos) for ap in probes]
    J = _ap_from_margins(np.asarray(probes, float),
                         np.asarray(margins, float)) * 1e3      # en mm
    info.update(feasible=True, J=J, margins=margins)
    return (J, info) if detail else J


def _ap_from_margins(probes, g):
    """Estimation de a_p,lim [m] a partir de log rho aux profondeurs sondes.

    L'ancien critere etait J = -moyenne(log rho). Il etait presente comme
    "strictement monotone" en a_p,lim ; il ne l'est pas. Des qu'une sonde est
    au-dessus de la limite, son terme mesure a quel point la boucle est
    INSTABLE a une profondeur que personne n'atteint, et cette quantite n'a
    aucune relation monotone avec le premier croisement de rho = 1. Mesure sur
    dix correcteurs faisables : tau de Kendall entre J et le vrai a_p,lim
    bissecte = 0.809 seulement, avec des inversions y compris ENTRE STRUCTURES
    (un FOPID note J = -0.132 contre un ADRC a -0.223, alors que les limites
    vraies etaient 0.219 mm contre 0.231 mm).

    On estime donc directement la profondeur ou max_x log rho croise zero, par
    interpolation lineaire du PREMIER croisement montant (et extrapolation
    lineaire si toutes les sondes sont du meme cote). Le critere est alors dans
    la meme unite que la grandeur visee, et son classement coincide avec elle
    par construction a la resolution des sondes pres.
    """
    ap = np.asarray(probes, float)
    g = np.asarray(g, float)
    if g[0] > 0.0:                       # deja instable a la plus petite sonde
        if g[1] > g[0]:                  # extrapolation vers le bas
            sl = (g[1] - g[0]) / (ap[1] - ap[0])
            return max(ap[0] - g[0] / sl, 0.0)
        return 0.0
    for k in range(len(ap) - 1):         # premier croisement montant
        if g[k] <= 0.0 < g[k + 1]:
            t = -g[k] / (g[k + 1] - g[k])
            return float(ap[k] + t * (ap[k + 1] - ap[k]))
    if g[-1] <= 0.0:                     # stable partout : extrapolation haute
        # BORNEE. Sans borne, une pente quasi nulle envoie l'estimation a
        # l'infini : un correcteur du protocole A, stable a toutes les sondes
        # sur le modele reduit a deux modes, a produit J = 3.1e5 mm. Au-dela de
        # 3 x la sonde la plus profonde on ne pretend plus discriminer — c'est
        # a la bissection finale (cinq modes, m = 200) de le faire.
        cap = 3.0 * ap[-1]
        if g[-1] > g[-2]:
            sl = (g[-1] - g[-2]) / (ap[-1] - ap[-2])
            return float(min(ap[-1] - g[-1] / sl, cap))
        return float(cap)
    return float(ap[0])


# ---------------------------------------------------------------------------
def limits(plate, ss, rpm, positions=None, m=None, n_modes=None,
           lo=0.005e-3, hi=4.0e-3, rtol=1e-3, pd=None):
    """Profondeurs limites VRAIES (bissection de Floquet), pleine resolution.

    `pd` porte les gains de l'Eq. (30). Il valait None en dur ici, ce qui
    etait exact tant qu'aucune structure n'avait de terme retarde ; pour
    `musyn_td` cela reviendrait a tracer les fossoles de mu tout seul."""
    from closed_loop import limit
    pos = C.POSITIONS if positions is None else positions
    kw = dict(n_modes=C.N_MODES if n_modes is None else n_modes,
              m=C.M_FLOQUET if m is None else m,
              coeff_mode='time', coeff_scale=C.SIGN_SIM, ae=C.AE)
    return np.array([limit(plate, rpm, fr * plate.lp, ctrl=ss, pd=pd,
                           lo=lo, hi=hi, rtol=rtol, **kw) for fr in pos])
