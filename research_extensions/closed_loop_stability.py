"""
closed_loop_stability.py
========================
RIGOROUS CLOSED-LOOP STABILITY LOBE DIAGRAM via full-system semi-discretization
==============================================================================

Research gap addressed (#4)
---------------------------
The package computes its stability-lobe diagram (SLD) with ``fdm_stability.py``,
which builds the monodromy matrix of the **open-loop**, mode-decoupled
regenerative system.  The controlled curves in the article's SLD are then drawn
with a *heuristic effective-damping boost* — the README states plainly:

    "the DARC curve uses a heuristic effective-damping boost … (an assumption,
     not a Floquet result for the feedforward-controlled NDDE)."

So the headline "stability improved 28×–40×" is **not** a stability result: it
is an assumed damping multiplier fed into the open-loop tool.

What this module does
---------------------
It computes the SLD of the **actual closed loop** by semi-discretising the full
coupled system — plant + Kalman observer + controller (+ internal-model
resonators) — including the regenerative delay τ and the periodic cutting
coefficient a4(t).  The controller is treated exactly as the linear time-
invariant compensator it is::

    [q; q̇] plant  (2n)      q̈ = -(Kp + a4 Dp Dpᵀ) q - Cp q̇
                                 + a4 Dp Dpᵀ q(t-τ) + H_Pe u
    x_ctrl        (nc)      ẋ_ctrl = A_ctrl x_ctrl + B_ctrl y ,   y = D_obs·q
    u             = C_ctrl x_ctrl

Because the observer and the modes are coupled through the controller, the
closed loop is **not** mode-decoupled — the per-mode 2×2 construction of the
article's tool is not valid here, which is exactly why a full-system monodromy
is required.  The Floquet criterion max|eig(Φ)| < 1 then gives a *genuine*
closed-loop stability boundary for LQG and for the repetitive/internal-model
LQG.

Method: 0th-order semi-discretisation (Insperger & Stépán, 2004) of the coupled
state, with the delayed positions q(t-τ) carried as m history blocks.

References
----------
- T. Insperger, G. Stépán, "Updated semi-discretization method for periodic
  delay-differential equations", Int. J. Numer. Meth. Eng. 61 (2004) 117-141.
- Y. Altintas, E. Budak, "Analytical prediction of stability lobes in milling",
  CIRP Annals 44 (1995) 357-362.
"""
import numpy as np
from scipy.linalg import expm

import _pkg_path  # noqa: F401  (puts the base package on sys.path)
from milling_force import milling_force_coeffs


# ======================================================================
#  Controller state-space extraction (continuous LTI compensator)
# ======================================================================
def controller_ss(ctrl):
    """
    Return the continuous compensator (A_ctrl, B_ctrl, C_ctrl) with

        ẋ_ctrl = A_ctrl x_ctrl + B_ctrl y ,   u = C_ctrl x_ctrl

    from a fitted ``LQGController`` or ``RepetitiveLQG`` — i.e. exactly the
    controller used in the time-domain simulations (fair, no re-tuning).
    ``None`` → open loop (nc = 0).
    """
    if ctrl is None:
        return (np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)))

    cls = ctrl.__class__.__name__
    A, B, C = ctrl.A, ctrl.B, ctrl.C
    L = ctrl.L_kal

    if cls == "LQGController":
        K = ctrl.K_lqr
        A_ctrl = A - B @ K - L @ C
        B_ctrl = L
        C_ctrl = -K
        return A_ctrl, B_ctrl, C_ctrl

    if cls == "RepetitiveLQG":
        # x_ctrl = [x_hat ; xi]
        nx, nim = ctrl.n_x, ctrl.n_im
        Kx, Kxi = ctrl.K_x, ctrl.K_xi
        S, Lim = ctrl.S, ctrl.L_im
        A_ctrl = np.zeros((nx + nim, nx + nim))
        A_ctrl[:nx, :nx] = A - B @ Kx - L @ C
        A_ctrl[:nx, nx:] = -B @ Kxi
        A_ctrl[nx:, nx:] = S
        B_ctrl = np.zeros((nx + nim, 1))
        B_ctrl[:nx, :] = L
        B_ctrl[nx:, :] = Lim
        C_ctrl = np.zeros((1, nx + nim))
        C_ctrl[0, :nx] = -Kx.flatten()
        C_ctrl[0, nx:] = -Kxi.flatten()
        return A_ctrl, B_ctrl, C_ctrl

    raise TypeError(f"controller_ss: unsupported controller {cls}")


# ======================================================================
#  Closed-loop continuous matrices (frozen a4)
# ======================================================================
def _closed_loop_matrices(plate, A_ctrl, B_ctrl, C_ctrl):
    """
    Build the parts of the closed-loop generator that do NOT depend on a4,
    plus the constant pieces needed to assemble A_c(a4) quickly.

    State  w = [q; q̇; x_ctrl]  (dim N = 2n + nc).
    Delayed coupling acts on q(t-τ) (dim n).
    """
    n = plate.n_modes
    nc = A_ctrl.shape[0]
    N = 2 * n + nc

    Kp = plate.Kp
    Cp = plate.Cp
    Dp = plate.Dp_ctrl                     # mode shape at the tool (n,)
    DpDpT = np.outer(Dp, Dp)
    H = plate.H_Pe_modal
    D_obs = plate.D_obs

    B_plant = np.zeros((2 * n, 1))
    B_plant[n:, 0] = H
    C_plant = np.zeros((1, 2 * n))
    C_plant[0, :n] = D_obs

    # constant (a4-independent) skeleton of A_c
    A0 = np.zeros((N, N))
    A0[:n, n:2 * n] = np.eye(n)            # q̇
    A0[n:2 * n, :n] = -Kp                  # -Kp q
    A0[n:2 * n, n:2 * n] = -Cp             # -Cp q̇
    if nc > 0:
        A0[:2 * n, 2 * n:] = B_plant @ C_ctrl        # plant ← controller
        A0[2 * n:, :2 * n] = B_ctrl @ C_plant        # controller ← output
        A0[2 * n:, 2 * n:] = A_ctrl                  # controller dynamics

    # a4-dependent structure: q̈ += -a4 DpDpᵀ q (into A_c), and +a4 DpDpᵀ q_τ (delayed)
    reg = np.zeros((N, N))
    reg[n:2 * n, :n] = -DpDpT              # multiplied by a4 at runtime

    A_d_unit = np.zeros((N, n))
    A_d_unit[n:2 * n, :] = DpDpT           # multiplied by a4 at runtime

    return N, n, nc, A0, reg, A_d_unit


# ======================================================================
#  Semi-discretised monodromy of the closed loop
# ======================================================================
def closed_loop_floquet(plate, ctrl, RPM, ap, hp,
                        NT, RT, eta_h, phi_st, phi_ex, k1, k2, kt,
                        m_div=30):
    """
    Floquet spectral radius ρ = max|eig(Φ)| of the closed loop for one
    (RPM, ap).  ρ < 1 → asymptotically stable (chatter-free); ρ ≥ 1 → chatter.
    """
    A_ctrl, B_ctrl, C_ctrl = controller_ss(ctrl)
    N, n, nc, A0, reg, A_d_unit = _closed_loop_matrices(
        plate, A_ctrl, B_ctrl, C_ctrl)

    tau = 60.0 / (NT * RPM)
    Omega = 2 * np.pi * RPM / 60.0
    dt = tau / m_div
    m = m_div

    za_low, za_high = hp - ap, hp

    # a4(t) sampled over one tooth-passing period
    a4 = np.empty(m)
    for k in range(m):
        _, a4k = milling_force_coeffs(k * dt, Omega, NT, RT, eta_h,
                                      phi_st, phi_ex, za_low, za_high,
                                      k1, k2, kt)
        a4[k] = a4k

    Cq = np.zeros((n, N))
    Cq[:, :n] = np.eye(n)                  # picks q out of w

    naug = N + n * m
    Phi = np.eye(naug)

    # index layout of augmented state: [ w(0:N) , h1(N:N+n), ..., hm ]
    for k in range(m):
        A_c = A0 + a4[k] * reg
        A_dk = a4[k] * A_d_unit

        P = expm(A_c * dt)
        # R = (∫_0^dt e^{A_c s} ds) A_dk
        try:
            R = np.linalg.solve(A_c, (P - np.eye(N)) @ A_dk)
        except np.linalg.LinAlgError:
            # series fallback
            S_int = np.eye(N) * dt + A_c * dt**2 / 2.0 + (A_c @ A_c) * dt**3 / 6.0
            R = S_int @ A_dk

        G = np.zeros((naug, naug))
        # w' = P w + R h_m
        G[:N, :N] = P
        G[:N, N + n * (m - 1):N + n * m] = R
        # h_1' = Cq w ; h_j' = h_{j-1}
        G[N:N + n, :N] = Cq
        if m > 1:
            G[N + n:N + n * m, N:N + n * (m - 1)] = np.eye(n * (m - 1))

        Phi = G @ Phi

    ev = np.linalg.eigvals(Phi)
    return float(np.max(np.abs(ev)))


# ======================================================================
#  Critical depth of cut (bisection on ap) and full SLD grid
# ======================================================================
def critical_ap(plate, ctrl, RPM, hp, NT, RT, eta_h, phi_st, phi_ex,
                k1, k2, kt, ap_lo=1e-5, ap_hi=8e-3, n_scan=60, tol=5e-5,
                m_div=30):
    """
    First (smallest) ap at which ρ crosses 1 — the true chatter limit.

    ρ(ap) can be non-monotonic (lobe structure), so a coarse **upward scan**
    locates the first stable→unstable bracket, then bisection refines it.  This
    avoids the classic pitfall of bisecting a non-monotonic boundary.
    """
    aps = np.linspace(ap_lo, ap_hi, n_scan)
    r_prev = closed_loop_floquet(plate, ctrl, RPM, aps[0], hp, NT, RT, eta_h,
                                 phi_st, phi_ex, k1, k2, kt, m_div)
    if r_prev >= 1.0:
        return aps[0]                      # unstable even at the tiniest ap
    for a in aps[1:]:
        r = closed_loop_floquet(plate, ctrl, RPM, a, hp, NT, RT, eta_h,
                                phi_st, phi_ex, k1, k2, kt, m_div)
        if r >= 1.0:
            lo, hi = a - (aps[1] - aps[0]), a       # bracket found
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                rm = closed_loop_floquet(plate, ctrl, RPM, mid, hp, NT, RT,
                                         eta_h, phi_st, phi_ex, k1, k2, kt,
                                         m_div)
                if rm < 1.0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < tol:
                    break
            return 0.5 * (lo + hi)
        r_prev = r
    return ap_hi                           # stable across the whole search range


def compute_closed_loop_SLD(plate, ctrl, RPM_array, ap_array, hp,
                            NT, RT, eta_h, phi_st, phi_ex, k1, k2, kt,
                            m_div=30, verbose=True, label=""):
    """
    ρ(RPM, ap) grid for the closed loop.  Returns rho_grid (n_ap × n_RPM).
    """
    import time
    n_RPM, n_ap = len(RPM_array), len(ap_array)
    rho = np.zeros((n_ap, n_RPM))
    t0 = time.time()
    for i, RPM in enumerate(RPM_array):
        for j, ap in enumerate(ap_array):
            rho[j, i] = closed_loop_floquet(plate, ctrl, RPM, ap, hp, NT, RT,
                                            eta_h, phi_st, phi_ex, k1, k2, kt,
                                            m_div)
        if verbose and (i + 1) % max(1, n_RPM // 8) == 0:
            el = time.time() - t0
            print(f"   [{label}] {100*(i+1)/n_RPM:4.0f}%  RPM={RPM:.0f}  "
                  f"({el:4.0f}s, ETA {el*(n_RPM-i-1)/(i+1):4.0f}s)")
    return rho
