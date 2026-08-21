"""lti_floquet.py — stabilite de la coupe en boucle fermee avec un correcteur
LTI QUELCONQUE (etat-espace), et non plus seulement la structure PPF.

Systeme augmente x = [q ; q' ; x_c] :

    q"  = -(K + a4(t) D^T D) q - C q' + H u + a4(t) D^T D q(t - tau) + ...
    u   = C_c x_c + D_c y ,        y = D_obs . q
    x_c'= A_c x_c + B_c y

Le rayon spectral de la monodromie est obtenu par methode de puissance sur
l'application d'une periode de dent (aucune matrice augmentee (m+1)*nx n'est
assemblee), ce qui autorise des correcteurs d'ordre eleve — R-ESO + FOPID
approche par Oustaloup atteint typiquement l'ordre 20.
"""
import numpy as np

from milling_dynamics import alpha4_series, alpha4_average, N_TEETH
import monodromy
from step_integrals import step_integrals


def augmented(plate, ctrl, DtD, D_obs, H, a4, n):
    """(A, A_tau) du systeme augmente pour une valeur figee de alpha4.

    ctrl : (Ac, Bc, Cc, Dc) du correcteur y -> u, ou None (boucle ouverte)."""
    if ctrl is None:
        Ac = np.zeros((0, 0)); Bc = np.zeros((0, 1))
        Cc = np.zeros((1, 0)); Dc = np.zeros((1, 1))
    else:
        Ac, Bc, Cc, Dc = ctrl
    nc = Ac.shape[0]
    nx = 2 * n + nc
    K0 = np.diag(plate.omega_n[:n] ** 2)
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
    return A, At


def period_maps(plate, rpm, ap, x_pos, ctrl=None, n_modes=2, m=60,
                coeff_mode='time', coeff_scale=1.0):
    tau = 60.0 / (N_TEETH * rpm)
    h = tau / m
    D = plate.D_row(x_pos, plate.hp)[:n_modes]
    DtD = np.outer(D, D)
    D_obs = plate.D_row(plate.lp, plate.hp)[:n_modes]
    H = np.asarray(plate.H_Pe_modal, float)[:n_modes]
    if coeff_mode == 'time':
        _, a4 = alpha4_series(rpm, ap, plate.hp, m, midpoint=True)
        a4 = coeff_scale * a4
    else:
        a4 = np.full(m, coeff_scale * alpha4_average(rpm, ap, plate.hp))
    maps = []
    for k in range(m):
        A, At = augmented(plate, ctrl, DtD, D_obs, H, a4[k], n_modes)
        # Meme rattrapage que dans les deux autres moteurs : A peut etre
        # singuliere sans que les integrales cessent d'exister.
        P0, J1, J2 = step_integrals(A, h)
        maps.append((P0, (J1 - J2 / h) @ At, (J2 / h) @ At))
    return maps, tau


def spectral_radius(maps, m, nx, seed=0, k=4):
    """Rayon spectral de la monodromie — Arnoldi, voir monodromy.py.

    REMPLACE UNE ITERATION DE PUISSANCE. Le texte precedent documentait deux
    pieges reels et corriges (nombre de periodes fixe et trop petit ; critere
    d'arret a fenetres recouvrantes et tolerance relative) ; il en restait un
    troisieme, hors d'atteinte de tout reglage du critere. La monodromie est
    tres non normale, le facteur de croissance traverse une HUITIEME PLATE a
    une valeur fausse, et deux fenetres non recouvrantes s'y accordent aussi
    bien que sur la vraie limite : sur le FOPID stocke a a_p = 0.10 mm,
    l'iteration rendait 0.79107 (graine 1) la ou la monodromie assemblee donne
    0.967392. L'erreur allait TOUJOURS dans le sens d'une coupe plus stable
    qu'elle ne l'est.
    """
    return monodromy.spectral_radius(maps, m, nx, seed=seed, k=k)


def dominant_eigs(maps, m, nx, q=4, seed=0):
    """Multiplicateurs dominants AVEC leur phase.

    L'iteration de sous-espace qui vivait ici avait deja ete corrigee deux
    fois (reorthonormalisation du bloc, projection de Rayleigh-Ritz sur une
    base reellement orthonormee) ; elle restait sujette a la meme huitieme
    plate que l'iteration de puissance — mesure : 0.902429 la ou la monodromie
    assemblee donne 0.939341 (ADRC stocke, a_p = 0.10 mm). Arnoldi rend le
    spectre dominant a 1e-15 pres.
    """
    return monodromy.dominant(maps, m, nx, k=q, seed=seed)


def closed_loop_chatter(plate, rpm, ap, x_pos, ctrl, n_modes=2, m=40,
                        coeff_scale=1.0, kmax=8):
    """Frequences de broutement EN BOUCLE FERMEE, deduites de la phase du
    multiplicateur dominant, repliees puis rapportees a chaque mode."""
    maps, tau = period_maps(plate, rpm, ap, x_pos, ctrl, n_modes, m,
                            'time', coeff_scale)
    ev = dominant_eigs(maps, m, maps[0][0].shape[0])
    ev = ev[np.isfinite(ev)]
    if ev.size == 0:
        return None, np.inf
    i = int(np.argmax(np.abs(ev)))
    lam = ev[i]
    rho = float(abs(lam))
    f_pv = abs(np.angle(lam)) / (2 * np.pi * tau)
    cand = sorted({round(abs(s * f_pv + j / tau), 2)
                   for s in (1, -1) for j in range(0, kmax + 1)})
    out = [min(cand, key=lambda c: abs(c - plate.omega_n[k] / (2 * np.pi)))
           for k in range(n_modes)]
    return out, rho


def is_stable(plate, rpm, ap, x_pos, ctrl=None, n_modes=2, m=60,
              coeff_mode='time', coeff_scale=1.0, seed=0):
    maps, _ = period_maps(plate, rpm, ap, x_pos, ctrl, n_modes, m,
                          coeff_mode, coeff_scale)
    rho = spectral_radius(maps, m, maps[0][0].shape[0], seed)
    return rho <= 1.0, rho


def limit(plate, rpm, x_pos, ctrl=None, lo=0.005e-3, hi=4.0e-3, tol=None,
          rtol=2e-3, atol=1e-6, **kw):
    """Profondeur axiale limite [m] par bissection, arret RELATIF.

    Voir control/closed_loop.limit : un arret absolu a 0.02 mm quantifiait les
    limites de 0.03-0.27 mm sur la grille dyadique de la bissection (jusqu'a
    18 % d'erreur, de signe dependant de la position de la vraie limite dans le
    dernier intervalle). `tol` reste accepte comme alias de `atol`.
    """
    if tol is not None:
        atol = float(tol)
    ok = lambda ap: is_stable(plate, rpm, ap, x_pos, ctrl, **kw)[0]
    if not ok(lo):
        return 0.0
    if ok(hi):
        return hi
    while hi - lo > max(atol, rtol * hi):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return 0.5 * (lo + hi)
