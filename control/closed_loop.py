"""
closed_loop.py — Stabilite de la coupe en boucle fermee (Floquet) avec
correcteur LTI ET commande a retard.
======================================================================
Le module `lti_floquet.py` du paquet plant ne retarde que les POSITIONS q :
il ne peut donc pas evaluer la commande a retard de l'Eq. (30), qui utilise
y(t - tau) ET y'(t - tau). Ici l'etat AUGMENTE COMPLET est retarde :

    x = [q ; q' ; x_c],     x'(t) = A(t) x(t) + A_tau(t) x(t - tau)

    A(t)     : plaque + correcteur robuste (LTI, non retarde)
    A_tau(t) : effet regeneratif   alpha4(t) D^T D q(t - tau)
             + commande a retard   H (K_Pp y(t-tau) + K_Pd y'(t-tau)),
                                   y = D_obs q

Sur chaque sous-intervalle h = tau/m, A est gelee au point milieu et le terme
retarde est interpole lineairement, exactement comme dans stability_fdm.py
(methode de discretisation complete, Ding et al. [79]). Le rayon spectral de
la monodromie est obtenu par iteration de puissance sur l'application d'une
periode de dent, sans jamais assembler la matrice (m+1)nx augmentee.
"""
import numpy as np

from milling_dynamics import alpha4_series, alpha4_average, N_TEETH
import monodromy
from step_integrals import step_integrals


# ---------------------------------------------------------------------------
def build_matrices(plate, DtD, D_obs, H, a4, ctrl=None, pd=None, n=2):
    """(A, A_tau) de l'etat augmente pour une valeur figee de alpha4.

    ctrl : (Ac, Bc, Cc, Dc) correcteur LTI y -> u   (ou None)
    pd   : (K_Pp, K_Pd) commande a retard, Eq. (30) (ou None)
    """
    if ctrl is None:
        Ac = np.zeros((0, 0)); Bc = np.zeros((0, 1))
        Cc = np.zeros((1, 0)); Dc = np.zeros((1, 1))
    else:
        Ac, Bc, Cc, Dc = [np.atleast_2d(np.asarray(m, float)) for m in ctrl]
        Ac = Ac.reshape(Bc.shape[0], Bc.shape[0]) if Bc.size else Ac
    nc = Ac.shape[0]
    nx = 2 * n + nc
    K0 = np.diag(plate.omega_n[:n]**2)
    C0 = np.diag(2 * plate.zeta_modes[:n] * plate.omega_n[:n])
    A = np.zeros((nx, nx))
    At = np.zeros((nx, nx))
    A[:n, n:2 * n] = np.eye(n)
    A[n:2 * n, :n] = -(K0 + a4 * DtD) + float(Dc[0, 0]) * np.outer(H, D_obs)
    A[n:2 * n, n:2 * n] = -C0
    At[n:2 * n, :n] = a4 * DtD
    if nc:
        A[n:2 * n, 2 * n:] = np.outer(H, Cc.ravel())
        A[2 * n:, :n] = np.outer(Bc.ravel(), D_obs)
        A[2 * n:, 2 * n:] = Ac
    if pd is not None:                                   # Eq. (30)
        Kp, Kd = float(pd[0]), float(pd[1])
        At[n:2 * n, :n] += Kp * np.outer(H, D_obs)
        At[n:2 * n, n:2 * n] += Kd * np.outer(H, D_obs)
    return A, At


def period_maps(plate, rpm, ap, x_pos, ctrl=None, pd=None, n_modes=2, m=40,
                coeff_mode='time', coeff_scale=1.0, ae=None):
    """Applications elementaires sur une periode de dent."""
    from milling_dynamics import AE_NOM
    ae = AE_NOM if ae is None else ae
    tau = 60.0 / (N_TEETH * rpm)
    h = tau / m
    D = plate.D_row(x_pos, plate.hp)[:n_modes]
    DtD = np.outer(D, D)
    D_obs = plate.D_row(plate.lp, plate.hp)[:n_modes]
    H = np.asarray(plate.H_Pe_modal, float)[:n_modes]
    if coeff_mode == 'time':
        _, a4 = alpha4_series(rpm, ap, plate.hp, m, ae=ae, midpoint=True)
        a4 = coeff_scale * a4
    else:
        a4 = np.full(m, coeff_scale * alpha4_average(rpm, ap, plate.hp, ae))
    maps = []
    for k in range(m):
        A, At = build_matrices(plate, DtD, D_obs, H, a4[k], ctrl, pd, n_modes)
        # Le rattrapage sans inversion vit desormais dans
        # paper_model/step_integrals.py : les trois moteurs de Floquet du
        # depot en avaient besoin, et deux d'entre eux ne l'avaient pas.
        P0, J1, J2 = step_integrals(A, h)
        maps.append((P0, (J1 - J2 / h) @ At, (J2 / h) @ At))
    return maps, tau


def spectral_radius(maps, m, nx, seed=0, k=4):
    """Rayon spectral de la monodromie — Arnoldi, voir paper_model/monodromy.py.

    REMPLACE UNE ITERATION DE PUISSANCE ADAPTATIVE QUI RENDAIT DES VALEURS
    FAUSSES. L'ancien texte de cette fonction expliquait deux pieges reels (un
    nombre de periodes fixe et trop petit ; un critere d'arret a fenetres
    recouvrantes et tolerance relative) et les corrigeait tous les deux — mais
    il restait le troisieme, qu'aucun reglage du critere ne peut atteindre :
    la monodromie de ce systeme est tres non normale (conditionnement de la
    base propre : 1.35e29 sur le cas FDOB), le facteur de croissance traverse
    une longue HUITIEME PLATE a une valeur qui n'est pas la bonne, et deux
    fenetres non recouvrantes s'y accordent aussi bien que sur la vraie
    limite. Mesure, FOPID stocke, 5 modes, m = 24, a_p = 0.10 mm : la
    monodromie assemblee donne 0.967392 ; l'iteration rendait 0.96690 /
    0.79107 / 0.96756 selon la graine. Sur le FDOB au meme point : 0.9142 /
    0.9138 / 0.9108 pour un exact de 0.959809.

    L'erreur etait TOUJOURS PAR DEFAUT sur rho, donc TOUJOURS PAR EXCES sur
    a_p,lim : chaque limite publiee par ce depot etait optimiste.

    Arnoldi rend le meme spectre a 1e-15 pres que la monodromie assemblee, sur
    tous les m ou l'assemblage reste calculable, et coute 3 a 15 fois moins.
    """
    return monodromy.spectral_radius(maps, m, nx, seed=seed, k=k)


def dominant_eig(maps, m, nx, k=6, seed=0):
    """Multiplicateurs dominants AVEC leur phase (frequence de broutement).

    L'iteration de sous-espace qui vivait ici souffrait de la meme huitieme
    plate que l'iteration de puissance ; Arnoldi rend les valeurs propres
    complexes directement.
    """
    return monodromy.dominant(maps, m, nx, k=k, seed=seed)


def is_stable(plate, rpm, ap, x_pos, ctrl=None, pd=None, n_modes=2, m=40,
              coeff_mode='time', coeff_scale=1.0, seed=0, ae=None):
    maps, _ = period_maps(plate, rpm, ap, x_pos, ctrl, pd, n_modes, m,
                          coeff_mode, coeff_scale, ae)
    rho = spectral_radius(maps, m, maps[0][0].shape[0], seed)
    return rho <= 1.0, rho


def limit(plate, rpm, x_pos, ctrl=None, pd=None, lo=0.005e-3, hi=4.0e-3,
          tol=None, rtol=2e-3, atol=1e-6, **kw):
    """Profondeur axiale limite [m] par bissection (0 si deja instable).

    ARRET RELATIF. Avec un arret absolu a 0.02 mm, les limites de 0.03 a
    0.27 mm rencontrees ici etaient quantifiees sur la grille dyadique de la
    bissection : la boucle ouverte a 4900 tr/min rendait 0.0334 mm au lieu de
    0.0363 mm (-8 %), et jusqu'a +18 % a 4600 tr/min. L'erreur ne se compense
    pas entre structures — son signe depend de la position de la vraie limite
    dans le dernier intervalle — et elle contamine les facteurs d'amelioration,
    qui divisent tous par la limite en boucle ouverte. `tol` reste accepte
    comme alias de `atol` pour les appels existants.
    """
    if tol is not None:
        atol = float(tol)
    ok = lambda ap: is_stable(plate, rpm, ap, x_pos, ctrl, pd, **kw)[0]
    if not ok(lo):
        return 0.0
    if ok(hi):
        return hi
    while hi - lo > max(atol, rtol * hi):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return 0.5 * (lo + hi)
def limit_over_pass(plate, rpm, ctrl=None, pd=None, positions=(0.0, 0.25,
                    0.5, 0.75, 1.0), **kw):
    """Minimum de la limite le long du bord superieur (fractions de l_P)."""
    vals = [limit(plate, rpm, f * plate.lp, ctrl, pd, **kw) for f in positions]
    return float(min(vals)), np.array(vals)


def chatter_frequency(plate, rpm, ap, x_pos, ctrl=None, pd=None, n_modes=2,
                      m=40, kmax=8, ae=None, coeff_scale=1.0):
    """Frequences de broutement en boucle fermee (phase du multiplicateur)."""
    maps, tau = period_maps(plate, rpm, ap, x_pos, ctrl, pd, n_modes, m,
                            'time', coeff_scale, ae)
    ev = dominant_eig(maps, m, maps[0][0].shape[0])
    ev = ev[np.isfinite(ev)]
    if ev.size == 0:
        return None, np.inf
    i = int(np.argmax(np.abs(ev)))
    lam = ev[i]
    f_pv = abs(np.angle(lam)) / (2 * np.pi * tau)
    cand = sorted({round(abs(s * f_pv + j / tau), 2)
                   for s in (1, -1) for j in range(0, kmax + 1)})
    out = [min(cand, key=lambda c: abs(c - plate.omega_n[k] / (2 * np.pi)))
           for k in range(n_modes)]
    return out, float(abs(lam))
