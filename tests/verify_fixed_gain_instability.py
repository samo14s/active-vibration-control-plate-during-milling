"""
verify_fixed_gain_instability.py
================================
The path study reports that a fixed-gain LQG designed on the blank becomes
WORSE than no control at all over the second half of the machining
programme. That is a strong claim, so it is checked here three independent
ways before it is believed:

  1. Continuous-time closed-loop poles. If K and L were designed for a plant
     whose modes have since moved by a factor of two, is A_cl = A - B K
     (and the observer loop) actually unstable on the current plant? This
     involves no delay, no periodicity and no discretisation, so it isolates
     pure design mistuning.

  2. The certified monodromy at a small depth of cut, where the regenerative
     term is negligible. It must agree with (1).

  3. Direct time-domain simulation of the sampled loop on the current plant.
     The response must actually grow.

If all three agree, the effect is a real mistuning instability and not an
artefact of the stability solver.
"""
import numpy as np
import _bootstrap  # noqa: F401

from scipy.linalg import solve_continuous_are, expm

from evolving_plate import EvolvingPlateModel
from closed_loop_sld import spectral_radius

LP, HP = 0.100, 0.080
H_BLANK, H_FINAL = 0.008, 0.003
RHO, E_AL, NU = 2830, 69e9, 0.33
ZETA = np.array([0.0031, 0.0017, 0.0027])
N_MODES, N1, N2 = 3, 30, 40
PIEZO = dict(d31=175e-12, h_Pa=0.7e-3, E_Pe=63e9, nu_Pe=0.35)
PATCH = (0.0, 0.020, 0.0, 0.060)
OBS = (LP, 0.5 * HP)
TS = 5e-5
NT, RPM = 3, 4900
D_TOOL = 0.010
ETA_H, GAMMA_N = np.deg2rad(35), np.deg2rad(15)
KT, KN, MU_C, AE = 925e6, 0.26, 0.20, 0.1e-3
RT = D_TOOL / 2
PHI_ST = np.pi - np.arccos(1 - AE / RT)
PHI_EX = np.pi
K1 = KN * np.cos(ETA_H)
K2 = 1 + MU_C * np.tan(ETA_H) * np.cos(GAMMA_N) - KN * np.sin(ETA_H)
COMMON = dict(NT=NT, RT=RT, eta_h=ETA_H, phi_st=PHI_ST, phi_ex=PHI_EX,
              hp=HP, k1=K1, k2=K2, kt=KT)


def build(plate):
    n = N_MODES
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -np.diag(plate.omega_n ** 2)
    A[n:, n:] = -np.diag(2 * ZETA * plate.omega_n)
    B = np.zeros((2 * n, 1))
    B[n:, 0] = plate.H_Pe_modal
    C = np.zeros((1, 2 * n))
    C[0, :n] = plate.D_obs
    return A, B, C


def design(A, B, C, D_obs, w_q=1e14, w_qd=1e8):
    n = N_MODES
    Q = np.block([[w_q * np.outer(D_obs, D_obs) + 1e-3 * np.eye(n),
                   np.zeros((n, n))],
                  [np.zeros((n, n)), w_qd * np.eye(n) + 1e-3 * np.eye(n)]])
    R = np.array([[1.0]])
    K = np.linalg.solve(R, B.T @ solve_continuous_are(A, B, Q, R))
    V = np.array([[1e-12]])
    L = solve_continuous_are(A.T, C.T, 1e-6 * np.eye(2 * n), V) @ C.T / 1e-12
    return K, L


def make_plate(n_layers_removed):
    """Wall after `n_layers_removed` full-height radial layers of 1.667 mm."""
    p = EvolvingPlateModel(LP, HP, H_BLANK, RHO, E_AL, NU, N1=N1, N2=N2,
                           n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
    ae = (H_BLANK - H_FINAL) / 3
    for _ in range(n_layers_removed):
        p.remove_material(0.0, LP, 0.0, HP, ae)
    p.solve_modes()
    p.precompute_Dp(zp_pos=0.75 * HP, n_pos=201)
    p.set_observation(*OBS)
    p.add_piezo_patch(*PATCH, **PIEZO)
    return p


print("=" * 78)
print(" Is the fixed-gain instability real? Three independent checks.")
print("=" * 78)

blank = make_plate(0)
Ab, Bb, Cb = build(blank)
K_fix, L_fix = design(Ab, Bb, Cb, blank.D_obs)
print(f"  controller designed on the blank: f = {np.round(blank.freq_n,1)} Hz")
print(f"  ||K|| = {np.linalg.norm(K_fix):.3e}")
print()
print(f"  {'layers':>7} {'f1 [Hz]':>9} {'max Re(A-BK)':>14} "
      f"{'max Re(full LQG)':>18} {'rho cert (ap=1um)':>19} {'time-domain':>13}")

for nl in (0, 1, 2, 3):
    p = make_plate(nl)
    A, B, C = build(p)

    # (1) continuous-time closed loop with the FIXED gains on this plant
    ev_sf = np.linalg.eigvals(A - B @ K_fix)
    # full LQG loop: [x; xhat] with the fixed observer too
    Acl = np.block([[A, -B @ K_fix],
                    [L_fix @ C, Ab - Bb @ K_fix - L_fix @ Cb]])
    ev_full = np.linalg.eigvals(Acl)

    # (2) certified monodromy at a depth where regeneration is negligible
    des = dict(A=Ab, B=Bb, C=Cb, K=K_fix, L=L_fix)
    rho = spectral_radius(p.omega_n, ZETA, p.Dp_array[:, 100],
                          p.H_Pe_modal, p.D_obs, RPM=RPM, ap=1e-6,
                          Ts=TS, ctrl_design=des, **COMMON)

    # (3) time-domain: sampled loop, no cutting force, impulse start
    n = N_MODES
    Ad = expm(A * TS)
    Bd = np.linalg.solve(A, (Ad - np.eye(2 * n)) @ B)
    Ao = expm((Ab - L_fix @ Cb) * TS)
    Gu = np.linalg.solve(Ab - L_fix @ Cb,
                         (Ao - np.eye(2 * n)) @ Bb)
    Gy = np.linalg.solve(Ab - L_fix @ Cb,
                         (Ao - np.eye(2 * n)) @ L_fix)
    x = np.zeros(2 * n)
    x[0] = 1e-6
    xh = np.zeros(2 * n)
    u = 0.0
    amp = []
    for k in range(8000):
        y = float((C @ x).item())
        xh = Ao @ xh + Gu.flatten() * u + Gy.flatten() * y
        u = float((-K_fix @ xh).item())
        x = Ad @ x + Bd.flatten() * u
        amp.append(abs(y))
        if abs(y) > 1e3:
            break
    a = np.array(amp)
    half = len(a) // 2
    grow = (np.max(a[half:]) / max(np.max(a[:half]), 1e-300)
            if len(a) > 10 else np.inf)

    print(f"  {nl:7d} {p.freq_n[0]:9.1f} {ev_sf.real.max():+14.1f} "
          f"{ev_full.real.max():+18.1f} {rho:19.4f} "
          f"{'GROWS' if grow > 10 else 'decays':>13}")

print()
print("  A positive max Re means the continuous closed loop is unstable with")
print("  no delay and no periodicity involved -- i.e. pure design mistuning,")
print("  not an artefact of the stability solver. The certified Floquet")
print("  radius and the time-domain run must agree with it.")
