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

from milling_force import milling_force_coeffs


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


# ===========================================================================
#  ANALYSE DE FLOQUET DE LA BOUCLE FERMÉE (multi-modes, compensateur inclus)
# ===========================================================================
def build_Phi_closed_loop(A0, Bu_col, Dp_vec, C_row, a4_array,
                          m_div, dt_int, ctrl=None):
    """
    Matrice de monodromie sur une période de passage de dent pour le système
    MULTI-MODES couplé (rang-1 régénératif Dp·Dpᵀ), avec — optionnellement —
    le COMPENSATEUR LQG NUMÉRIQUE (observateur de Kalman + retour d'état
    estimé) AUGMENTÉ dans l'état, échantillonné à dt_int.

    État augmenté : z = [ x (2n) ; x̂ (2n, si ctrl) ; w_hist (m_div) ]
    avec w = Dpᵀ q le déplacement au point outil et w_hist l'historique
    nécessaire au terme retardé w(t-τ) (semi-discrétisation d'ordre zéro).

    Dynamique par sous-intervalle k (coefficient a4_k gelé) :
        ẋ = [A0 - a4_k·E] x + Bu·u + a4_k·Bd·w(t-τ)
        E  = [[0,0],[Dp·Dpᵀ,0]],  Bd = [0 ; Dp]      (M = I, modes normalisés)
    Compensateur (exactement celui implémenté par LQGController) :
        u_j    = -K x̂_j
        x̂_{j+1} = A_obs_d x̂_j + G_u u_j + G_y (C x_j)

    Parameters
    ----------
    A0     : (2n, 2n) matrice d'état de la plaque seule
    Bu_col : (2n,)    colonne d'entrée piézo [0 ; H_Pe]
    Dp_vec : (n,)     forme propre au point outil
    C_row  : (2n,)    ligne de mesure [D_obs, 0]
    a4_array : (m_div,) coefficient régénératif sur une période
    ctrl   : None (boucle ouverte) ou dict(K, A_obs_d, G_u, G_y)
             discrétisé à dt_int.

    Returns
    -------
    Phi : matrice de monodromie ((2n [+2n]) + m_div carrée)
    """
    n_x = A0.shape[0]
    n = n_x // 2
    has_ctrl = ctrl is not None
    n_c = n_x if has_ctrl else 0
    N = n_x + n_c + m_div

    E = np.zeros((n_x, n_x))
    E[n:, :n] = np.outer(Dp_vec, Dp_vec)
    Bd = np.zeros(n_x)
    Bd[n:] = Dp_vec
    Dp_ext = np.zeros(n_x)
    Dp_ext[:n] = Dp_vec                      # w = Dpᵀ q = Dp_ext · x

    if has_ctrl:
        K = np.atleast_2d(ctrl['K'])
        K_row = K.flatten()
        A_obs_d = ctrl['A_obs_d']
        G_u = np.asarray(ctrl['G_u']).reshape(n_x)
        G_y = np.asarray(ctrl['G_y']).reshape(n_x)
        A_hat = A_obs_d - np.outer(G_u, K_row)         # x̂ -> x̂ (u = -K x̂)

    from scipy.linalg import expm as _expm
    Phi = np.eye(N)
    i_h0 = n_x + n_c                          # début du bloc historique

    # La matrice de pas T_k est creuse (bloc dense (n_x+n_c) lignes + décalage
    # de l'historique) : on applique T_k @ Phi de façon STRUCTURÉE en
    # O(m_div·N²) au lieu de O(m_div·N³), et on met en cache la
    # discrétisation (Ad, Γu, Γd) par valeur de a4_k — la dent est hors coupe
    # (a4 = 0) la majorité de la période à faible immersion radiale.
    disc_cache = {}
    for k in range(m_div):
        a4_k = float(a4_array[k])
        if a4_k in disc_cache:
            Ad, Gu_p, Gd_p = disc_cache[a4_k]
        else:
            A_k = A0 - a4_k * E
            Ad = _expm(A_k * dt_int)
            try:
                intExp = np.linalg.solve(A_k, Ad - np.eye(n_x))
            except np.linalg.LinAlgError:
                intExp = (np.eye(n_x) * dt_int + A_k * dt_int**2 / 2
                          + A_k @ A_k * dt_int**3 / 6)
            Gu_p = intExp @ Bu_col            # entrée commande (ZOH)
            Gd_p = intExp @ (a4_k * Bd)       # entrée retardée (gelée)
            disc_cache[a4_k] = (Ad, Gu_p, Gd_p)

        x_blk = Phi[:n_x]                     # vues sur l'état courant
        h_last = Phi[i_h0 + m_div - 1]        # ligne w_{j-m_div}
        new_x = Ad @ x_blk + np.outer(Gd_p, h_last)
        if has_ctrl:
            xh_blk = Phi[n_x:n_x + n_c]
            new_x = new_x - np.outer(Gu_p, K_row @ xh_blk)
            new_xh = np.outer(G_y, C_row @ x_blk) + A_hat @ xh_blk
        new_w = Dp_ext @ x_blk                # nouveau w_j (ligne)

        # décalage de l'historique (copie explicite : recouvrement)
        Phi[i_h0 + 1:i_h0 + m_div] = Phi[i_h0:i_h0 + m_div - 1].copy()
        Phi[i_h0] = new_w
        Phi[:n_x] = new_x
        if has_ctrl:
            Phi[n_x:n_x + n_c] = new_xh

    return Phi


def compute_SLD_closed_loop(RPM_array, ap_array, plate,
                            NT, RT, eta_h, phi_st, phi_ex,
                            k1, k2, kt, hp, Dp_vec,
                            lqg=None, dt_c=5e-5, verbose=True):
    """
    SLD par multiplicateurs de Floquet de la boucle COMPLÈTE :
    plaque multi-modes couplée (Dp·Dpᵀ) + retard régénératif
    [+ compensateur LQG numérique si ``lqg`` est fourni].

    - lqg : instance de LQGController (attributs A, B, C, K_lqr, L_kal),
      conçue sur la plaque NOMINALE ; l'observateur est re-discrétisé à
      dt_int = τ/m_div pour chaque RPM (m_div = round(τ/dt_c), de sorte que
      dt_int ≈ dt_c à ±0.7 %).
    - lqg=None : boucle ouverte (même cadre, mêmes hypothèses -> courbes
      directement comparables).

    NOTE : contrairement à l'ancienne approche « substitution de pôles »
    (amortissements de eig(A-BK) réinjectés dans le SLD boucle ouverte),
    cette analyse contient l'observateur, l'échantillonnage et le couplage
    retard/compensateur.  Elle reste LINÉAIRE : saturation ±150 V exclue —
    la frontière n'est valide que là où la tension requise reste < u_max.
    """
    import time
    from scipy.linalg import expm as _expm

    n = plate.n_modes
    A0 = np.zeros((2 * n, 2 * n))
    A0[:n, n:] = np.eye(n)
    A0[n:, :n] = -np.linalg.solve(plate.Mp, plate.Kp)
    A0[n:, n:] = -np.linalg.solve(plate.Mp, plate.Cp)
    Bu_col = np.zeros(2 * n)
    Bu_col[n:] = np.linalg.solve(plate.Mp, plate.H_Pe_modal)
    C_row = np.zeros(2 * n)
    C_row[:n] = plate.D_obs

    n_RPM = len(RPM_array)
    n_ap = len(ap_array)
    rho_grid = np.zeros((n_ap, n_RPM))

    if verbose:
        mode = "boucle fermée LQG" if lqg is not None else "boucle ouverte"
        print(f"[SLD-Floquet-BF] grille {n_ap} x {n_RPM} ({mode})")
        t0 = time.time()

    for i_rpm, RPM in enumerate(RPM_array):
        tau = 60.0 / (NT * RPM)
        Omega_spin = 2 * np.pi * RPM / 60
        m_div = max(20, int(round(tau / dt_c)))
        dt_int = tau / m_div

        ctrl = None
        if lqg is not None:
            # Re-discrétisation exacte de l'observateur implémenté à dt_int
            A_obs = lqg.A - lqg.L_kal @ lqg.C
            A_obs_d = _expm(A_obs * dt_int)
            try:
                G_u = np.linalg.solve(A_obs, (A_obs_d - np.eye(2 * n)) @ lqg.B)
                G_y = np.linalg.solve(A_obs, (A_obs_d - np.eye(2 * n)) @ lqg.L_kal)
            except np.linalg.LinAlgError:
                G_u = lqg.B * dt_int
                G_y = lqg.L_kal * dt_int
            ctrl = dict(K=lqg.K_lqr, A_obs_d=A_obs_d, G_u=G_u, G_y=G_y)

        for i_ap, ap in enumerate(ap_array):
            za_low, za_high = hp - ap, hp
            a4_array = np.zeros(m_div)
            for k in range(m_div):
                _, a4 = milling_force_coeffs(
                    k * dt_int, Omega_spin, NT, RT, eta_h,
                    phi_st, phi_ex, za_low, za_high, k1, k2, kt)
                a4_array[k] = a4
            Phi = build_Phi_closed_loop(A0, Bu_col, Dp_vec, C_row,
                                        a4_array, m_div, dt_int, ctrl=ctrl)
            rho_grid[i_ap, i_rpm] = float(np.max(np.abs(np.linalg.eigvals(Phi))))

        if verbose and (i_rpm + 1) % max(1, n_RPM // 5) == 0:
            print(f"   {100*(i_rpm+1)/n_RPM:5.1f}%  "
                  f"({time.time()-t0:5.1f}s écoulées)")

    if verbose:
        print(f"[SLD-Floquet-BF] terminé en {time.time()-t0:.1f}s")
    return rho_grid


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
