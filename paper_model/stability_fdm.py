"""
stability_fdm.py — Lobes de stabilite par discretisation complete (type Ding
et al. [79] du papier) pour le systeme retarde a coefficients periodiques

    q" + C0 q' + (K0 + a4(t) DtD) q(t) - a4(t) DtD q(t - tau) = f(t)

avec q les DEUX premiers modes (M0 = I, modes normalises en masse), DtD la
deformee au point de contact (position donnee sur le bord superieur), a4(t)
periodique de periode de dent tau, et la convention de signe de l'Eq. (13)
telle que publiee (celle qui reproduit les figures du papier — voir l'en-tete
de milling_plant.py).

Methode : sur chaque sous-intervalle h = tau/m, la matrice d'etat A_k est
gelee a sa valeur au point milieu, le terme retarde est interpole lineairement
entre q_{k-m} et q_{k-m+1}, et la solution exacte par exponentielle de matrice
fournit l'application

    x_{k+1} = Phi0 x_k + C_lo q_{k-m} + C_hi q_{k-m+1}.

La matrice de monodromie sur une periode est le produit des m applications sur
l'etat augmente z = [x ; q_{k-1} ; ... ; q_{k-m}] ; la coupe est stable si le
rayon spectral est <= 1 (Floquet).
"""
import numpy as np

from milling_dynamics import alpha4_series, alpha4_average, N_TEETH
from step_integrals import step_integrals


def floquet_matrix(omega, zeta, DtD, a4_mid, tau, n=None):
    """Monodromie du systeme a 2 modes. a4_mid : (m,) valeurs de alpha4 aux
    points milieux des m sous-intervalles."""
    n = DtD.shape[0] if n is None else n
    m = len(a4_mid)
    h = tau / m
    K0 = np.diag(omega[:n]**2)
    C0 = np.diag(2 * zeta[:n] * omega[:n])
    dim = 2 * n + n * m
    Phi = np.eye(dim)
    I2 = np.eye(n)
    for k in range(m):
        Kk = K0 + a4_mid[k] * DtD
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = I2
        A[n:, :n] = -Kk
        A[n:, n:] = -C0
        B = np.zeros((2 * n, n))
        B[n:, :] = a4_mid[k] * DtD
        # La raideur effective K0 + a4(t).D^T D s'annule a certaines
        # profondeurs : A devient alors singuliere et le `solve` direct
        # levait une exception au lieu de rendre des integrales qui, elles,
        # existent. Voir step_integrals.py.
        P0, J1, J2 = step_integrals(A, h)
        C_hi = (J2 / h) @ B                                  # coeff q_{k-m+1}
        C_lo = (J1 - J2 / h) @ B                             # coeff q_{k-m}
        Dk = np.zeros((dim, dim))
        Dk[:2 * n, :2 * n] = P0
        # q_{k-m}   est le DERNIER bloc de z, q_{k-m+1} l'avant-dernier
        Dk[:2 * n, dim - n:] = C_lo
        # ... SAUF a m = 1, ou "l'avant-dernier bloc" n'existe pas : q_{k-m+1}
        # vaut alors q_k, la position COURANTE, qui vit dans les n premieres
        # colonnes de l'etat. Comme dim - 2n = n a m = 1, l'ecriture generale
        # tombait exactement sur le bloc des VITESSES et assemblait
        # silencieusement une autre monodromie que celle voulue.
        j_hi = 0 if m == 1 else dim - 2 * n
        Dk[:2 * n, j_hi:j_hi + n] += C_hi
        # decalage de l'historique : nouveau q_{k} = positions de x_k
        Dk[2 * n:3 * n, :n] = I2
        if m > 1:
            Dk[3 * n:, 2 * n:dim - n] = np.eye(n * (m - 1))
        Phi = Dk @ Phi
    return Phi


def spectral_radius_and_freq(Phi, tau):
    """(rho, f_c) : rayon spectral et frequence de broutement associee au
    multiplicateur dominant, f_c = |angle(mu)| / (2 pi tau) (valeur
    principale, a completer modulo 1/tau)."""
    mu = np.linalg.eigvals(Phi)
    i = int(np.argmax(np.abs(mu)))
    return float(np.abs(mu[i])), float(abs(np.angle(mu[i])) / (2 * np.pi * tau))


def is_stable(plate, rpm, ap, x_pos, m=60, coeff_mode='time', coeff_scale=1.0,
              n_modes=2, return_freq=False):
    """Stabilite d'une coupe (rpm, ap) au point de contact (x_pos, h_P).

    coeff_mode : 'time'  -> alpha4(t) exact, periodique (courbe a4(t), Fig. 6)
                 'avg'   -> alpha4 = coeff_scale * abar4 constant
                            (courbes 0.3x, 1x, 2.9x de la Fig. 6)
    """
    tau = 60.0 / (N_TEETH * rpm)
    D = plate.D_row(x_pos, plate.hp)[:n_modes]
    DtD = np.outer(D, D)
    if coeff_mode == 'time':
        _, a4 = alpha4_series(rpm, ap, plate.hp, m, midpoint=True)
        a4 = coeff_scale * a4
    else:
        a4 = np.full(m, coeff_scale * alpha4_average(rpm, ap, plate.hp))
    Phi = floquet_matrix(plate.omega_n[:n_modes], plate.zeta_modes[:n_modes],
                         DtD, a4, tau)
    rho, fc = spectral_radius_and_freq(Phi, tau)
    return (rho <= 1.0, fc, rho) if return_freq else rho <= 1.0


def stability_limit(plate, rpm, x_pos, lo=0.005e-3, hi=1.5e-3, tol=2e-6,
                    m=60, coeff_mode='time', coeff_scale=1.0):
    """Plus grande profondeur axiale stable [m] par bissection ; renvoie 0 si
    lo est deja instable, hi si hi tient encore (vraies bornes)."""
    ok = lambda ap: is_stable(plate, rpm, ap, x_pos, m, coeff_mode, coeff_scale)
    if not ok(lo):
        return 0.0
    if ok(hi):
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return 0.5 * (lo + hi)
