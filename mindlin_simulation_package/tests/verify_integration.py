"""
verify_integration.py
=====================
End-to-end drop-in test: the Mindlin `PlateModel` must run unchanged with the
article package-12 stack (milling_force + LQG controller + Newmark solver).
Confirms every interface attribute the downstream code relies on is present and
that closed-loop LQG reduces vibration on the Mindlin plate.

Run:  python verify_integration.py
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "01_core"))
sys.path.insert(0, os.path.join(_HERE, "..", "02_controllers"))

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from newmark_solver import NewmarkSimulator

# ---- article parameters (main_simulation.py) ----
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT = 925e6; KN = 0.26; MU_C = 0.20
RPM = 4900
FT = 0.02e-3; AE = 0.1e-3
D31 = 175e-12; H_PA = 0.7e-3
E_PE = 63e9; NU_PE = 0.35
N1, N2 = 30, 24
N_MODES = 3
ZETA = [0.0031, 0.0017, 0.0027]
DT = 5e-5
ap = 0.3e-3
T_END = 0.03


def rms(x):
    return np.sqrt(np.mean(x**2))


def main():
    print(">> Building Mindlin plate (article geometry)...")
    plate = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2,
                       n_modes=N_MODES, zeta_modes=ZETA, verbose=True)
    plate.precompute_Dp(zp_pos=HP - ap/2, n_pos=2001)
    plate.set_observation(x_obs=LP, z_obs=HP)
    plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)

    # required interface attributes for the package-12 stack
    required = ["Mp", "Kp", "Cp", "omega_n", "freq_n", "V", "n_modes",
                "D_obs", "H_Pe_modal", "Dp_array", "DpT_Dp_array", "xp_array"]
    print("\n>> Interface attributes:")
    missing = [a for a in required if not hasattr(plate, a)]
    for a in required:
        print(f"   {a:12s} {'ok' if hasattr(plate, a) else 'MISSING'}")
    assert not missing, f"missing interface attributes: {missing}"

    phi_st = np.pi - np.arccos(1 - AE/(DT_TOOL/2)); phi_ex = np.pi
    Omega = 2*np.pi*RPM/60; RT = DT_TOOL/2; tau = 60/(NT*RPM)
    k1 = KN*np.cos(ETA_H)
    k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)

    sim = NewmarkSimulator(plate, dt=DT, T_end=T_END, ft=FT, tau=tau, verbose=True)
    v_feed = FT*NT*RPM/60
    kp_idx = np.clip(np.round(np.minimum(v_feed*sim.t_vec, LP)/LP*2000).astype(int), 0, 2000)
    n_per = int(np.round(tau/DT))
    a3, a4 = precompute_alpha_periodic(DT, n_per, sim.nstep, Omega, NT, RT, ETA_H,
                                       phi_st, phi_ex, HP-ap, HP, k1, k2, KT)

    print("\n>> Building LQG on the Mindlin modal model...")
    lqg = LQGController(plate, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14], w_qd_list=[1e6, 1e8], w_r=1.0)
    lqg.discretize_observer()

    res_ol = sim.simulate(a3, a4, kp_idx, controller=None, progress=False)
    res_cl = sim.simulate(a3, a4, kp_idx, controller=lqg, progress=False)

    i_ol, i_cl = res_ol['stop_idx'], res_cl['stop_idx']
    y_ol = rms(res_ol['y'][:i_ol+1]) * 1e6
    y_cl = rms(res_cl['y'][:i_cl+1]) * 1e6
    u_max = np.max(np.abs(res_cl['u'][:i_cl+1]))
    print("\n>> RESULTS (Mindlin plate, package-12 stack):")
    print(f"   open-loop y_rms = {y_ol:.4f} um")
    print(f"   LQG       y_rms = {y_cl:.4f} um   u_max = {u_max:.2f} V")
    red = (1 - y_cl/y_ol) * 100
    print(f"   -> LQG reduction vs open-loop: {red:.1f}%")

    assert np.all(np.isfinite(res_cl['y'])), "non-finite output"
    assert y_cl < y_ol, "LQG did not reduce vibration"
    assert u_max <= 150.0 + 1e-9, "voltage exceeded piezo saturation"
    print("\n  [integration] PASS")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
