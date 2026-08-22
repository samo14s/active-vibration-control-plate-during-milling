"""
surface_error.py — QUALITE DE SURFACE : l'erreur de position de surface (SLE)
=============================================================================
Jusqu'ici ce depot ne mesurait QUE la stabilite : rho, a_p,lim, marges. Le mot
« surface » n'y designait que la surface de glissement du SMC. Or la moitie de
l'objectif — « minimiser la degradation de qualite » — n'a aucun sens sans une
grandeur de qualite. C'est ce que ce module apporte.

CE QUE MESURE LA SLE. La surface finie est engendree a l'instant precis ou une
dent passe a l'angle de generation. Tout ecart de la piece a cet instant-la est
imprime dans la piece. En avalant (phi_st = pi - acos(1 - ae/R), phi_ex = pi),
la dent quitte la coupe en phi = pi, ou son rayon pointe exactement selon la
normale : c'est la que la surface est engendree.

    SLE(x) = w(x, t*)   avec   Omega t* + 2 pi j / N == pi  [2 pi]

POURQUOI IL FAUT UN NOUVEAU CALCUL. Le modele du depot est HOMOGENE : c'est la
dynamique de PERTURBATION (Eq. 13), dont la solution nominale est zero. La SLE
vit dans la solution nominale, pas dans la perturbation — il faut donc le terme
de CORSAGE statique, absent jusqu'ici.

L'EFFORT STATIQUE, EXACTEMENT. Dans ce modele l'effort normal vaut

    dF_n = kt (k2 sin phi - k1 cos phi) . h . dz

L'epaisseur dynamique est h_dyn = Delta . sin phi (projection du deplacement
normal), d'ou alpha4 = kt ∫ (k2 sin^2 - k1 sin cos) dz : c'est exactement ce
que calcule alpha34(). L'epaisseur STATIQUE est h_s = f_z sin phi — le MEME
sin phi. Donc

    F_n,statique(t) = f_z . alpha4(t)                                  (exact)

sans aucune integrale nouvelle. L'effort modal est alpha4(t) . f_z . D.

LE RETARD DISPARAIT SUR LA SOLUTION NOMINALE. Sans broutement la surface se
repete : x(t - tau) = x(t). L'equation nominale devient donc

    x'_p(t) = [A(t) + A_tau(t)] x_p(t) + b(t) ,    b = [0 ; alpha4 f_z D ; 0]

et la raideur regenerative s'annule exactement — A porte -(K0 + a4 D'D), A_tau
porte +a4 D'D. Il ne reste que K0. C'est la traduction du fait qu'un tour sans
broutement ne regenere rien.

ET C'EST LA QUE LE CORRECTEUR DE L'EQ. (30) SE DISTINGUE. van Dijk et al.
(2014) emploient une retroaction de PERTURBATION (Pyragas, en y(t) - y(t-tau))
qui S'ANNULE sur la solution nominale, et ils avertissent explicitement que
sans cela « the controller may affect the nominal solution ». Or l'Eq. (30) de
Du et al. est un terme RETARDE PUR, K_Pp y(t-tau) + K_Pd y'(t-tau) : sur la
solution nominale il vaut K_Pp y(t) + K_Pd y'(t), donc il NE s'annule PAS. Il
entre dans A + A_tau et deplace la solution nominale — donc la SLE.

L'avertissement de van Dijk est ici RENDU MESURABLE : ce module calcule de
combien.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'paper_model'), HERE]

from milling_dynamics import alpha4_series, N_TEETH, FZ_NOM, AE_NOM  # noqa
from step_integrals import step_integrals                             # noqa
from closed_loop import build_matrices                                # noqa


def generation_time(rpm):
    """Instant EXACT, dans [0, tau), ou une dent atteint l'angle pi.

    On ne prend PAS le noeud de grille le plus proche. Pour N = 3 l'instant de
    generation tombe a t* = tau/2, donc a k = m/2 - 0.5 sur une grille de
    points milieux : exactement ENTRE deux noeuds. Arrondir au plus proche fait
    sauter l'echantillon d'un demi-pas selon la parite de m, et comme la
    reponse nominale a ici une amplitude crete-a-crete de l'ordre de 90 um, ce
    demi-pas se voyait — la SLE « convergeait » vers 44.0, 43.7 puis 42.5 um en
    raffinant. Ce n'etait pas de la discretisation, c'etait du jitter
    d'echantillonnage. On interpole donc a t*.
    """
    Omega = 2.0 * np.pi * rpm / 60.0
    tau = 60.0 / (N_TEETH * rpm)
    ts = [((np.pi - 2.0 * np.pi * j / N_TEETH) / Omega) % tau
          for j in range(N_TEETH)]
    return float(min(ts)), tau


def echantillonne(w, t_grid, t_star, tau):
    """w interpole lineairement a t_star, avec repliement periodique."""
    m = len(w)
    h = tau / m
    u = (t_star - t_grid[0]) / h                 # grille de points milieux
    i = int(np.floor(u)) % m
    f = u - np.floor(u)
    return float((1.0 - f) * w[i] + f * w[(i + 1) % m])


def periodic_response(plate, rpm, ap, x_pos, ctrl=None, pd=None, n_modes=2,
                      m=40, fz=FZ_NOM, ae=AE_NOM, coeff_scale=1.0):
    """Solution tau-periodique nominale (sans broutement).

    Retourne dict(w, q, t, k_gen, sle, pv, cond) ou w est le deplacement
    normal de la plaque au point de coupe, echantillonne sur la periode.
    """
    tau = 60.0 / (N_TEETH * rpm)
    h = tau / m
    D = np.asarray(plate.D_row(x_pos, plate.hp), float)[:n_modes]
    DtD = np.outer(D, D)
    D_obs = np.asarray(plate.D_row(plate.lp, plate.hp), float)[:n_modes]
    H = np.asarray(plate.H_Pe_modal, float)[:n_modes]
    a4 = coeff_scale * alpha4_series(rpm, ap, plate.hp, m, ae=ae,
                                     midpoint=True)[1]

    A0, At0 = build_matrices(plate, DtD, D_obs, H, a4[0], ctrl, pd, n_modes)
    nx = A0.shape[0]
    b = np.zeros((m, nx))
    Phi = np.empty((m, nx, nx))
    G = np.empty((m, nx))
    Ms = []
    for k in range(m):
        A, At = build_matrices(plate, DtD, D_obs, H, a4[k], ctrl, pd, n_modes)
        Ms.append(A + At)                       # le retard tombe : x(t-tau)=x(t)
        b[k, n_modes:2 * n_modes] = a4[k] * fz * D
    for k in range(m):
        P0, J1, J2 = step_integrals(Ms[k], h)
        Phi[k] = P0
        db = b[(k + 1) % m] - b[k]              # corsage interpole lineairement
        G[k] = J1 @ b[k] + (J2 / h) @ db

    # x_{k+1} = Phi_k x_k + G_k ; periodicite x_m = x_0.
    Mon = np.eye(nx)
    acc = np.zeros(nx)
    for k in range(m):
        acc = Phi[k] @ acc + G[k]
        Mon = Phi[k] @ Mon
    S = np.eye(nx) - Mon
    cond = float(np.linalg.cond(S))
    x0 = np.linalg.solve(S, acc)

    x = np.empty((m + 1, nx))
    x[0] = x0
    for k in range(m):
        x[k + 1] = Phi[k] @ x[k] + G[k]
    q = x[:m, :n_modes]
    w = q @ D
    t_grid = (np.arange(m) + 0.5) * h
    t_star, _ = generation_time(rpm)
    return dict(w=w, q=q, x=x[:m], t=t_grid, t_star=t_star,
                sle=echantillonne(w, t_grid, t_star, tau),
                pv=float(w.max() - w.min()), cond=cond, tau=tau,
                fz=fz)


def sle(plate, rpm, ap, x_pos, **kw):
    """Raccourci : la seule SLE, en metres."""
    return periodic_response(plate, rpm, ap, x_pos, **kw)['sle']


# ---------------------------------------------------------------------------
# LIMITE DE VALIDITE — a lire avant d'utiliser ce module au-dela du finition
# ---------------------------------------------------------------------------
# La reponse nominale calculee ici suppose, comme alpha34(), un engagement
# CONTINU de la dent sur [phi_st, phi_ex]. Cette hypothese tombe des que la
# variation de deplacement DANS la periode de dent depasse la charge par dent :
# la dent decroche alors une partie du temps (fly-over), le retard devient
# dependant de l'etat, et alpha4(t) lui-meme n'est plus le bon coefficient.
#
# Mesure a 5200 tr/min, x = lp/2, deux modes, m = 192 :
#
#     a_p [mm]   SLE [um]   crete-a-crete [um]   (c-a-c)/f_z
#       0.02       1.27            3.15             0.16
#       0.10       6.55           15.86             0.79
#       0.15       9.97           23.81             1.19   <- croisement
#       0.60      42.97           94.42             4.72
#       1.00      73.03          154.75             7.74
#
# Le croisement tombe vers a_p ~ 0.12 mm, soit environ CINQ FOIS SOUS la limite
# de stabilite que ce depot publie (0.6-0.8 mm). L'immersion radiale nominale y
# est tres faible (ae = 0.1 mm, soit ae/R = 0.02 : la dent n'est engagee que
# ~10 % de la periode), donc l'effort est une impulsion breve et la plaque
# resonne entre les dents — d'ou une crete-a-crete tres superieure a la SLE.
#
# DEUX CONSEQUENCES, et la seconde touche tout le depot :
#
#  1. Pour cette plaque, c'est la QUALITE qui sature en premier, pas la
#     stabilite. « Maximiser a_p » n'est donc pas le bon objectif : la surface
#     est perdue bien avant la limite de broutement.
#  2. Au-dela du croisement, les coefficients alpha4(t) qui alimentent l'analyse
#     de Floquet sont eux-memes hors de leur domaine. Les a_p,lim publies ici —
#     et la comparaison des douze structures qui en depend — sont calcules dans
#     un regime ou l'engagement continu n'est plus garanti.
#
# CE QUE CE CHIFFRE N'EST PAS. Le croisement est un INDICATEUR, pas une preuve
# de decrochage : l'etablir demande un modele a retard dependant de l'etat
# (Niu, Ding, Zhu & Ding, IJMS 2021), que ce depot ne possede pas. Il indique ou
# le modele actuel cesse d'etre defendable, pas ce qui s'y passe reellement.


# ---------------------------------------------------------------------------
def nominal_control(plate, rpm, ap, x_pos, ctrl=None, pd=None, n_modes=2,
                    m=192, fz=FZ_NOM, ae=AE_NOM, coeff_scale=1.0):
    """Tension d'actionneur exigee par la SOLUTION NOMINALE elle-meme.

    Grandeur que ce depot n'avait jamais calculee, et qui n'existe que parce
    que la phase precedente a construit la solution nominale.

    Un correcteur de PERTURBATION (Pyragas, en y(t) - y(t-tau)) demande ZERO
    volt sur la solution nominale : il ne s'active que sur le broutement.
    L'Eq. (30) de Du et al. est un terme retarde pur, donc sur la solution
    nominale il vaut K_Pp y(t) + K_Pd y'(t) : il tire du courant PENDANT TOUTE
    LA COUPE, sans qu'il y ait le moindre broutement a supprimer.

    Ce n'est pas une subtilite comptable. Le budget est V_MAX ; ce qui part
    dans la solution nominale n'est plus disponible pour la perturbation. On
    rend donc u_nom (crete et efficace) ET sa part dans V_MAX.
    """
    r = periodic_response(plate, rpm, ap, x_pos, ctrl=ctrl, pd=pd,
                          n_modes=n_modes, m=m, fz=fz, ae=ae,
                          coeff_scale=coeff_scale)
    D_obs = np.asarray(plate.D_row(plate.lp, plate.hp), float)[:n_modes]
    q = r['x'][:, :n_modes]
    qd = r['x'][:, n_modes:2 * n_modes]
    y = q @ D_obs
    yd = qd @ D_obs
    u_pd = np.zeros(m)
    if pd is not None:
        u_pd = float(pd[0]) * y + float(pd[1]) * yd      # le retard tombe
    u_rob = np.zeros(m)
    if ctrl is not None:
        Ac, Bc, Cc, Dc = [np.atleast_2d(np.asarray(z, float)) for z in ctrl]
        nc = Bc.shape[0] if Bc.size else 0
        if nc:
            xc = r['x'][:, 2 * n_modes:]
            u_rob = xc @ Cc.ravel()
        u_rob = u_rob + float(Dc[0, 0]) * y
    u = u_rob + u_pd
    f = lambda v: dict(peak=float(np.max(np.abs(v))),
                       rms=float(np.sqrt(np.mean(v**2))))
    return dict(u=u, total=f(u), rob=f(u_rob), pd=f(u_pd),
                sle=r['sle'], pv=r['pv'], t=r['t'])
