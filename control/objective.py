"""
objective.py — Fonction objectif et contraintes, IDENTIQUES aux deux
correcteurs
=====================================================================
C'est ici que se joue l'equite de la comparaison : FOPID et ADRC-FOPID sont
notes par le MEME code, avec les MEMES contraintes. Seule change la fonction
qui fabrique le (A, B, C, D) a partir du vecteur de decision.

OBJECTIF — maximiser les limites de stabilite du fraisage.
Le critere exact serait la profondeur axiale limite a_p,lim obtenue par
bissection sur le rayon spectral de Floquet ; c'est 12 a 15 evaluations par
position, trop cher dans une boucle PSO. On maximise donc une mesure
STRICTEMENT MONOTONE de la meme quantite : la marge logarithmique de Floquet

    J = - moyenne_{a_p sondes} [ max_{x} log rho(a_p, x) ]

rho <= 1 signifie stable ; J > 0 signifie donc "stable a toutes les
profondeurs sondes et a toutes les positions", et J croit avec la marge. Les
profondeurs sondes (0.5, 1.0, 2.0 mm) encadrent la zone utile, la limite en
boucle ouverte etant ~0.15 mm. Les a_p,lim VRAIS sont ensuite calcules par
bissection, a pleine resolution, pour les seuls correcteurs retenus.

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

_F_LOG = np.logspace(0.5, 4.1, 140)          # fond logarithmique [Hz]


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


def frequency_metrics(plate, ss, positions=None, f=None, n_modes=None,
                      poles=None):
    """(Ms, Vmax) : marge de module et effort maximal en V/N."""
    f = _con_grid(plate, poles, n_modes) if f is None else np.asarray(f, float)
    pos = C.POSITIONS_DESIGN if positions is None else positions
    n = C.N_MODES_OBJ if n_modes is None else n_modes
    K = ss_frf(ss, 2 * np.pi * f)
    Pu, _ = plant_frf(plate, f, n)
    S = 1.0 / (1.0 - Pu * K)
    Ms = float(np.max(np.abs(S)))
    v = 0.0
    for fr in pos:
        _, Pf = plant_frf(plate, f, n, x_force=fr * plate.lp)
        v = max(v, float(np.max(np.abs(K * S * Pf))))
    return Ms, v


def floquet_margin(plate, ss, rpm, ap, x_pos, m=None, n_period=None):
    """log(rho) de la monodromie pour un correcteur LTI donne."""
    m = C.M_FLOQUET_PSO if m is None else m
    npd = C.N_PERIOD if n_period is None else n_period
    maps, _ = period_maps(plate, rpm, ap, x_pos, ctrl=ss, pd=None,
                          n_modes=C.N_MODES_OBJ, m=m, coeff_mode='time',
                          coeff_scale=C.SIGN_SIM, ae=C.AE)
    rho = spectral_radius(maps, m, maps[0][0].shape[0], npd)
    if not np.isfinite(rho):        # divergence violente : borne haute graduee
        return 50.0
    return float(np.clip(np.log(max(rho, 1e-300)), -50.0, 50.0))


# ---------------------------------------------------------------------------
def evaluate(plate, ss, rpm=None, probes=None, positions=None, m=None,
             detail=False):
    try:
        return _evaluate(plate, ss, rpm, probes, positions, m, detail)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        info = dict(feasible=False, reason='echec numerique', Ms=np.nan,
                    V=np.nan, J=-1e4, max_re=np.nan)
        return (info['J'], info) if detail else info['J']


def _evaluate(plate, ss, rpm, probes, positions, m, detail):
    """Note d'un correcteur. Retourne J (a MAXIMISER) et, si demande, le
    detail des contraintes."""
    rpm = C.RPM_DESIGN if rpm is None else rpm
    probes = C.AP_PROBE if probes is None else probes
    pos = C.POSITIONS_DESIGN if positions is None else positions
    info = dict(feasible=False, reason='', Ms=np.nan, V=np.nan, J=-np.inf)

    # -- crible 1 : stabilite nominale
    ev = nominal_poles(plate, ss)
    mre = float(np.max(ev.real))
    info['max_re'] = mre
    if not np.isfinite(mre) or mre > -1.0:
        info['reason'] = 'boucle nominale instable'
        info['J'] = -1e3 - max(mre, 0.0)
        return (info['J'], info) if detail else info['J']

    # -- crible 2/3 : marge de module et effort, sur une grille resolue par
    #    les poles de boucle fermee qu'on vient justement de calculer
    Ms, V = frequency_metrics(plate, ss, pos, poles=ev)
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

    # -- objectif : marge de Floquet moyennee sur les profondeurs sondes
    margins = []
    for ap in probes:
        worst = max(floquet_margin(plate, ss, rpm, ap, fr * plate.lp, m=m)
                    for fr in pos)
        margins.append(worst)
    J = -float(np.mean(margins))
    info.update(feasible=True, J=J, margins=margins)
    return (J, info) if detail else J


# ---------------------------------------------------------------------------
def limits(plate, ss, rpm, positions=None, m=None, n_modes=None,
           lo=0.01e-3, hi=3.0e-3, tol=1e-5):
    """Profondeurs limites VRAIES (bissection de Floquet), pleine resolution."""
    from closed_loop import limit
    pos = C.POSITIONS if positions is None else positions
    kw = dict(n_modes=C.N_MODES if n_modes is None else n_modes,
              m=C.M_FLOQUET if m is None else m, n_period=C.N_PERIOD,
              coeff_mode='time', coeff_scale=C.SIGN_SIM, ae=C.AE)
    return np.array([limit(plate, rpm, fr * plate.lp, ctrl=ss, pd=None,
                           lo=lo, hi=hi, tol=tol, **kw) for fr in pos])
