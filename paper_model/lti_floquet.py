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
from scipy.linalg import expm

from milling_dynamics import alpha4_series, alpha4_average, N_TEETH


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
        nx = A.shape[0]
        P0 = expm(A * h)
        J1 = np.linalg.solve(A, P0 - np.eye(nx))
        J2 = h * J1 - np.linalg.solve(A, h * P0 - J1)
        maps.append((P0, (J1 - J2 / h) @ At, (J2 / h) @ At))
    return maps, tau


def spectral_radius(maps, m, nx, n_period=None, seed=0, tol=1e-3,
                    n_min=40, n_max=400):
    """Rayon spectral de la monodromie (iteration de puissance ADAPTATIVE).

    Deux pieges, tous deux mesures sur ce modele :

    1. Un nombre de periodes FIXE et petit sous-estime rho, donc SURESTIME la
       stabilite (boucle ouverte, 4900 tr/min, a_p = 0.05 mm, m = 200) :

           n_period      10       20       50      100      200
           rho        0.8270   0.9207   1.0126   1.0173   1.0164

    2. Un critere d'arret mal construit s'arrete trop tot. La version
       precedente comparait deux moyennes cumulees se recouvrant a 90 % et
       tolerait une variation RELATIVE de log rho : pres de la frontiere de
       stabilite log rho -> 0, la tolerance relative degenere, et sur des cas
       de boucle fermee reels l'iteration s'arretait a 80 periodes en rendant
       0.8887 la ou la valeur convergee (1500 periodes) vaut 0.9753 — avec une
       dispersion de 9 % selon la graine.

    D'ou : fenetres NON RECOUVRANTES (les dix derniers facteurs de croissance
    contre les dix precedents), tolerance ABSOLUE sur log rho, et estimateur
    pris sur la queue seule (le transitoire initial de l'iteration biaise une
    moyenne sur toute la seconde moitie). `n_period` force un nombre fixe.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((m + 1, nx))
    Z /= np.linalg.norm(Z)
    g = []
    n_fixed = None if n_period is None else int(n_period)
    limit_it = n_fixed if n_fixed is not None else n_max
    ok = 0
    for it in range(limit_it):
        for P0, C_lo, C_hi in maps:
            new = P0 @ Z[0] + C_lo @ Z[m] + C_hi @ Z[m - 1]
            Z = np.roll(Z, 1, axis=0)
            Z[0] = new
        nz = np.linalg.norm(Z)
        if not np.isfinite(nz):
            return np.inf
        if nz == 0.0:
            return 0.0
        g.append(np.log(nz))
        Z /= nz
        if n_fixed is None and len(g) >= max(n_min, 20):
            a = float(np.mean(g[-10:]))
            b = float(np.mean(g[-20:-10]))
            ok = ok + 1 if abs(a - b) <= tol else 0
            if ok >= 2:                       # deux passages consecutifs
                break
    w = max(10, len(g) // 4)
    return float(np.exp(np.mean(np.array(g[-w:]))))
def dominant_eigs(maps, m, nx, q=4, n_period=40, seed=0):
    """Valeurs propres dominantes de la monodromie par ITERATION DE SOUS-ESPACE
    (bloc de q vecteurs + projection de Rayleigh-Ritz), donc avec leur PHASE :
    c'est elle qui donne la frequence de broutement EN BOUCLE FERMEE."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((m + 1, nx, q))
    Z /= np.linalg.norm(Z)
    prev = None
    for it in range(n_period):
        prev = Z.copy()
        for P0, C_lo, C_hi in maps:
            new = P0 @ Z[0] + C_lo @ Z[m] + C_hi @ Z[m - 1]
            Z = np.roll(Z, 1, axis=0)
            Z[0] = new
        nz = np.linalg.norm(Z)
        if not np.isfinite(nz) or nz == 0.0:
            return np.array([np.inf if np.isfinite(nz) else np.inf])
        Z /= nz
        if it >= n_period - 2:
            break
    # base orthonormee du sous-espace courant et projection
    V = prev.reshape(-1, q)
    Qb, _ = np.linalg.qr(V)
    W = Z.reshape(-1, q) * nz
    Hm = Qb.T @ W
    S = Qb.T @ Qb
    try:
        Hm = np.linalg.solve(S, Hm)
    except np.linalg.LinAlgError:
        pass
    return np.linalg.eigvals(Hm)


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
              coeff_mode='time', coeff_scale=1.0, n_period=None):
    maps, _ = period_maps(plate, rpm, ap, x_pos, ctrl, n_modes, m,
                          coeff_mode, coeff_scale)
    rho = spectral_radius(maps, m, maps[0][0].shape[0], n_period)
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
