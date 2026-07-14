"""
fdm_stability.py
=================
Calcul du diagramme de lobes de stabilité (SLD) par la méthode
Full-Discretization Method (FDM) d'Insperger-Stépán (2002, 2004).

Théorie :
   Le système avec retard régénératif est équivalent à un système
   linéaire à coefficients périodiques (LPP). On discrétise une
   période tau en m sous-intervalles, on construit la matrice de
   transition Phi sur une période, puis on calcule les multiplicateurs
   de Floquet (rayons spectraux) :
      stable    si max|eig(Phi)| < 1
      instable  sinon

Pour chaque (RPM, ap), on construit Phi(RPM, ap) puis on évalue
le rayon spectral. La frontière |λ|=1 délimite les lobes.

Référence :
   T. Insperger, G. Stépán, "Updated semi-discretization method for
   periodic delay-differential equations with discrete delay",
   Int. J. Numer. Meth. Eng. 61 (2004) 117–141.

Le système modal est :
   q̈ + 2 ζ ω q̇ + ω² q + (a4(t)/m_eff) Dp² q = (a4(t)/m_eff) Dp² q(t-τ)
"""
import numpy as np
from scipy.linalg import expm

from milling_force import milling_force_coeffs


# ====================================================================
# Rigorous CLOSED-LOOP, COUPLED, multi-mode monodromy SLD (P2)
# ====================================================================
def closed_loop_rho(omega_n, zeta_n, Dp_vec, H_Pe, D_obs,
                    A_ctrl, B_ctrl, K_lqr, L_kal,
                    RPM, NT, RT, eta_h, phi_st, phi_ex, ap, hp,
                    k1, k2, kt, m_div=25):
    """
    Spectral radius of the monodromy matrix of the FULL closed-loop, COUPLED,
    time-periodic delayed system over one tooth-passing period, via semi-
    discretization (Insperger–Stépán). Unlike the per-mode `fdm_stability_one_mode`,
    this keeps the rank-1 inter-modal regenerative coupling α4(t)·Dp·Dpᵀ AND embeds
    the actual LQG controller (state feedback + Kalman observer) in the loop, instead
    of feeding a closed-loop "equivalent damping" into an open-loop formula.

    Continuous augmented state ξ = [q; q̇; x̂]  (dim 4n if a controller is given,
    else 2n for the open loop). The regenerative term couples q(t) to q(t-τ):

        q̈ = -(K + α4·DpDpᵀ) q - C q̇ - H_Pe·(K_lqr x̂) + α4·DpDpᵀ q(t-τ)
        x̂̇ = (A_ctrl - B_ctrl K_lqr - L C_ctrl) x̂ + L·D_obs·q         (C_ctrl=[D_obs 0])

    Pass A_ctrl=None to get the open-loop coupled system (no control) — used for
    verification against the per-mode open-loop SLD.
    """
    n = len(omega_n)
    K = np.diag(np.asarray(omega_n, float) ** 2)
    Cdamp = np.diag(2.0 * np.asarray(zeta_n, float) * np.asarray(omega_n, float))
    DpDpT = np.outer(Dp_vec, Dp_vec)

    closed = A_ctrl is not None
    nx = 4 * n if closed else 2 * n           # continuous state dimension

    tau = 60.0 / (NT * RPM)
    dt = tau / m_div
    za_low, za_high = hp - ap, hp
    Omega_spin = 2 * np.pi * RPM / 60.0

    # Periodic milling coefficient α4(t) over one period
    a4 = np.empty(m_div)
    for k in range(m_div):
        _, a4k = milling_force_coeffs(k * dt, Omega_spin, NT, RT, eta_h,
                                      phi_st, phi_ex, za_low, za_high, k1, k2, kt)
        a4[k] = a4k

    # Constant (α4-independent) blocks
    if closed:
        H_Pe = np.asarray(H_Pe, float).reshape(n)
        D_obs = np.asarray(D_obs, float).reshape(n)
        K_lqr = np.asarray(K_lqr, float).reshape(2 * n)
        L = np.asarray(L_kal, float).reshape(2 * n)
        A_xh = A_ctrl - np.outer(B_ctrl.reshape(2 * n), K_lqr) \
             - np.outer(L, np.concatenate([D_obs, np.zeros(n)]))   # A - B K - L C
        LD = np.outer(L, D_obs)                                    # (2n × n)
        HK = np.outer(H_Pe, K_lqr)                                 # (n × 2n)

    N = nx + n * m_div                        # augmented (with delay buffer)
    Phi = np.eye(N)
    Iv = slice(n, 2 * n)                       # velocity rows of ξ

    for k in range(m_div):
        Ac = np.zeros((nx, nx))
        Ac[0:n, n:2*n] = np.eye(n)                        # q̇ = v
        Ac[n:2*n, 0:n] = -(K + a4[k] * DpDpT)             # v̇ stiffness (self-coupling)
        Ac[n:2*n, n:2*n] = -Cdamp                         # v̇ damping
        if closed:
            Ac[n:2*n, 2*n:4*n] = -HK                      # v̇ from control
            Ac[2*n:4*n, 0:n] = LD                         # x̂̇ from measured q
            Ac[2*n:4*n, 2*n:4*n] = A_xh                   # x̂̇ observer/feedback

        Ad = expm(Ac * dt)
        # Ic = ∫₀^dt exp(Ac s) ds  (delayed term held constant over the step)
        try:
            Ic = np.linalg.solve(Ac, Ad - np.eye(nx))
        except np.linalg.LinAlgError:
            Ic = np.eye(nx) * dt + Ac * dt**2 / 2.0
        # delayed q enters the v-block with gain α4·DpDpᵀ
        Bd = Ic[:, Iv] @ (a4[k] * DpDpT)                  # (nx × n)

        # Augmented one-step map T
        T = np.zeros((N, N))
        T[0:nx, 0:nx] = Ad
        T[0:nx, nx + (m_div - 1) * n: nx + m_div * n] = Bd   # uses oldest buffered q
        T[nx:nx + n, 0:n] = np.eye(n)                        # newest buffer entry = q_k
        for i in range(1, m_div):                            # shift buffer
            T[nx + i * n: nx + (i + 1) * n,
              nx + (i - 1) * n: nx + i * n] = np.eye(n)
        Phi = T @ Phi

    return float(np.max(np.abs(np.linalg.eigvals(Phi))))


def compute_closed_loop_SLD(RPM_array, ap_array,
                            omega_n, zeta_n, Dp_vec, H_Pe, D_obs,
                            A_ctrl, B_ctrl, K_lqr, L_kal,
                            NT, RT, eta_h, phi_st, phi_ex,
                            k1, k2, kt, hp, m_div=25, verbose=True):
    """Grid of closed-loop (or open-loop if A_ctrl is None) spectral radii."""
    import time
    n_RPM, n_ap = len(RPM_array), len(ap_array)
    rho = np.zeros((n_ap, n_RPM))
    if verbose:
        tag = "closed-loop" if A_ctrl is not None else "open-loop (coupled)"
        print(f"[SLD-CL] {tag} monodromy, {n_ap}x{n_RPM} grid, m_div={m_div}")
        t0 = time.time()
    for i_rpm, RPM in enumerate(RPM_array):
        for i_ap, ap in enumerate(ap_array):
            rho[i_ap, i_rpm] = closed_loop_rho(
                omega_n, zeta_n, Dp_vec, H_Pe, D_obs,
                A_ctrl, B_ctrl, K_lqr, L_kal,
                RPM, NT, RT, eta_h, phi_st, phi_ex, ap, hp,
                k1, k2, kt, m_div=m_div)
        if verbose and (i_rpm + 1) % max(1, n_RPM // 10) == 0:
            el = time.time() - t0
            print(f"   {100*(i_rpm+1)/n_RPM:5.1f}%  RPM={RPM:.0f}  ({el:5.1f}s)")
    if verbose:
        print(f"[SLD-CL] done in {time.time()-t0:.1f}s")
    return rho


def expm_2x2_analytical(A, dt):
    """
    Exponentielle matricielle exacte d'une matrice 2x2 par valeurs propres.
    Beaucoup plus rapide que scipy.linalg.expm pour des appels répétés.
    """
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    tr = a + d
    det = a*d - b*c
    disc = tr**2 - 4*det

    Adt = A * dt

    if abs(disc) < 1e-14:
        # Cas dégénéré : valeurs propres confondues -> Padé
        # Pour notre système, ce cas n'arrive jamais (oscillateur)
        # Fallback : développement Taylor d'ordre 6
        I = np.eye(2)
        Adt2 = Adt @ Adt
        Adt3 = Adt2 @ Adt
        Adt4 = Adt2 @ Adt2
        return I + Adt + Adt2/2 + Adt3/6 + Adt4/24 \
                 + Adt4 @ Adt / 120 + Adt4 @ Adt2 / 720

    sqrt_disc = np.sqrt(complex(disc))
    lam1 = (tr + sqrt_disc) / 2
    lam2 = (tr - sqrt_disc) / 2

    # exp(A*dt) = (exp(λ1*dt) - exp(λ2*dt))/(λ1 - λ2) * A*dt
    #           + (λ1*exp(λ2*dt) - λ2*exp(λ1*dt))/(λ1 - λ2) * I
    e1 = np.exp(lam1 * dt)
    e2 = np.exp(lam2 * dt)
    diff = lam1 - lam2

    coef1 = (e1 - e2) / diff
    coef0 = (lam1 * e2 - lam2 * e1) / diff

    Ad = coef0 * np.eye(2) + coef1 * A
    return Ad.real      # partie réelle (les complexes annulent)


def integral_expA_2x2(A, dt):
    """
    Calcule ∫₀^dt exp(A*s) ds  pour A 2x2.
    Si A inversible : = A^-1 (exp(A*dt) - I)
    Sinon : développement en série
    """
    Ad = expm_2x2_analytical(A, dt)
    det = A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]
    if abs(det) > 1e-12:
        return np.linalg.solve(A, Ad - np.eye(2))
    else:
        # Fallback Taylor
        return np.eye(2)*dt + A*dt**2/2 + A@A*dt**3/6


def build_FDM_Phi(omega_n, zeta_n, Dp_modal, m_modal,
                   a4_array, n_tau, m_div, dt_int):
    """
    Construit la matrice de transition Phi pour un système modal SISO
    avec retard régénératif sur une période tau = m_div * dt_int.

    Parameters
    ----------
    omega_n  : pulsation propre [rad/s]
    zeta_n   : amortissement [/]
    Dp_modal : valeur Dp scalaire pour ce mode
    m_modal  : masse modale (=1 si normalisée)
    a4_array : (m_div,) valeurs de a4(t) sur 1 période
    n_tau    : retard en pas de temps
    m_div    : nombre de subdivisions
    dt_int   : pas d'intégration

    Returns
    -------
    Phi : matrice (2*(n_tau+1), 2*(n_tau+1))
    """
    n_state = 2
    N = (n_tau + 1) * n_state
    Phi = np.eye(N)

    Dp2 = float(Dp_modal**2 / m_modal)

    # Pré-allocation
    D = np.zeros((N, N))
    # Identité dans les blocs de décalage (constante)
    for j in range(n_tau):
        D[2*(j+1):2*(j+2), 2*j:2*(j+1)] = np.eye(n_state)

    for k in range(m_div):
        a4_k = a4_array[k]
        alpha_k = a4_k * Dp2

        # A continu et B
        A = np.array([[0.0,                     1.0],
                      [-omega_n**2 - alpha_k, -2*zeta_n*omega_n]])

        # Discrétisation analytique 2x2
        Ad = expm_2x2_analytical(A, dt_int)
        # Bd = ∫₀^dt exp(A*(dt-s)) ds * B = ∫₀^dt exp(A*r) dr * B (r=dt-s)
        # B applique alpha_k * x(t-tau)[0] sur la 2ème équation
        # B = [[0,0],[alpha_k, 0]]
        intExp = integral_expA_2x2(A, dt_int)
        # Bd = intExp @ B
        # Ne calcule que les colonnes utiles (B colonne 0)
        Bd_col0 = intExp[:, 1] * alpha_k     # car B[1,0]=alpha, donc intExp @ B = intExp[:,1] * alpha
        Bd = np.zeros((2, 2))
        Bd[:, 0] = Bd_col0

        # Matrice de transition élémentaire
        D[0:2, :] = 0.0      # reset bloc supérieur
        D[0:2, 0:2] = Ad
        D[0:2, 2*n_tau:2*(n_tau+1)] = Bd

        Phi = D @ Phi

    return Phi


def fdm_stability_one_mode(omega_n, zeta_n, Dp_modal, m_modal,
                            RPM, NT, RT, eta_h,
                            phi_st, phi_ex, ap, hp,
                            k1, k2, kt,
                            m_div=40, scale_a4=1.0):
    """
    Evalue le rayon spectral pour un (RPM, ap) donné.
    """
    tau = 60.0 / (NT * RPM)
    Omega_spin = 2 * np.pi * RPM / 60
    dt_int = tau / m_div
    n_tau = m_div

    za_low = hp - ap
    za_high = hp

    a4_array = np.zeros(m_div)
    for k in range(m_div):
        _, a4 = milling_force_coeffs(
            k * dt_int, Omega_spin, NT, RT, eta_h,
            phi_st, phi_ex, za_low, za_high, k1, k2, kt
        )
        a4_array[k] = a4 * scale_a4

    Phi = build_FDM_Phi(omega_n, zeta_n, Dp_modal, m_modal,
                         a4_array, n_tau, m_div, dt_int)

    eigvals = np.linalg.eigvals(Phi)
    rho = float(np.max(np.abs(eigvals)))
    return rho


def compute_SLD(RPM_array, ap_array,
                 omega_n_list, zeta_list, Dp_list, m_list,
                 NT, RT, eta_h, phi_st, phi_ex,
                 k1, k2, kt, hp,
                 m_div=40, verbose=True):
    """
    Calcule la grille de rayons spectraux pour un SLD multi-modes.
    """
    import time
    n_RPM = len(RPM_array)
    n_ap  = len(ap_array)
    n_modes = len(omega_n_list)

    rho_grid = np.zeros((n_ap, n_RPM))
    rho_per_mode = np.zeros((n_modes, n_ap, n_RPM))

    if verbose:
        print(f"[SLD-FDM] grille {n_ap} x {n_RPM} = {n_ap*n_RPM} cas, "
              f"{n_modes} modes, m_div={m_div}")
        t0 = time.time()

    n_total = n_RPM * n_ap

    for i_rpm, RPM in enumerate(RPM_array):
        for i_ap, ap in enumerate(ap_array):
            for i_mode in range(n_modes):
                rho_i = fdm_stability_one_mode(
                    omega_n_list[i_mode], zeta_list[i_mode],
                    Dp_list[i_mode], m_list[i_mode],
                    RPM, NT, RT, eta_h, phi_st, phi_ex, ap, hp,
                    k1, k2, kt, m_div=m_div
                )
                rho_per_mode[i_mode, i_ap, i_rpm] = rho_i
            rho_grid[i_ap, i_rpm] = np.max(rho_per_mode[:, i_ap, i_rpm])

        if verbose and (i_rpm + 1) % max(1, n_RPM // 10) == 0:
            n_done = (i_rpm + 1) * n_ap
            pct = (n_done / n_total) * 100
            elapsed = time.time() - t0
            eta = elapsed * (n_total - n_done) / max(n_done, 1)
            print(f"   {pct:5.1f}%  RPM={RPM:.0f}  "
                  f"({elapsed:5.1f}s écoulées, ETA {eta:5.1f}s)")

    if verbose:
        print(f"[SLD-FDM] terminé en {time.time()-t0:.1f}s")

    return rho_grid, rho_per_mode
