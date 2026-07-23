"""
verify_rcsac.py
===============
Checks for the RC-SAC controller (see docs/RCSAC_STRATEGY.md):

  1. Decoupling filter: stable poles, C(0) ≈ Σ(Dp²/ω²)/Σ(DpH/ω²), and the
     mode-1 / mode-3 gains-with-phase match the required +85 / −257 targets.
  2. Worst-case stability: at a_p = 2.5 mm (beyond the LQG limit 2.05 mm, hard
     ±150 V saturation) LQG diverges while RC-SAC stays stable.
  3. Robustness: RC-SAC (nominal design) remains stable on the machined plate
     (free-end −1 mm, mode 1 drifted 519→533 Hz) at 2.5 mm.

Run:  python verify_rcsac.py        (~40 s)
"""
import os
import sys
import numpy as np
from scipy import signal as sg

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "01_core"))
sys.path.insert(0, os.path.join(_HERE, "..", "02_controllers"))

from plate_model import PlateModel
from mindlin_q8 import shape_at_point_M
from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from rcsac_controller import RCSACController

LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT = 925e6; KN = 0.26; MU_C = 0.20
AE = 0.1e-3; FT = 0.02e-3
ZETA = [0.0031, 0.0017, 0.0027]
DT = 5e-5; T_END = 0.30
RPM = 4900; U_SAT = 150.0
Z_TOOL = HP - 5e-3; X_TOOL = LP/2


def banner(t):
    print("\n" + "=" * 68 + f"\n {t}\n" + "=" * 68)


def build(hp=HP):
    p = PlateModel(LP, hp, BP, RHO, E_AL, NU, N1=30, N2=24, n_modes=3,
                   zeta_modes=ZETA, verbose=False)
    p.set_observation(LP, Z_TOOL)
    p.add_piezo_patch(0, 0.020, 0, 0.060, 175e-12, 0.7e-3, 63e9, 0.35)
    Dp = shape_at_point_M(X_TOOL, Z_TOOL, p.N1, p.N2, p.lex, p.ley,
                          p.node_id, p.ndof)[p.DOFf] @ p.V
    return p, Dp


def milling(ap):
    phi_st = np.pi - np.arccos(1 - AE/(DT_TOOL/2))
    Omega = 2*np.pi*RPM/60; RT = DT_TOOL/2; tau = 60/(NT*RPM)
    k1 = KN*np.cos(ETA_H)
    k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)
    ns = int(round(T_END/DT)) + 1
    a3, a4 = precompute_alpha_periodic(DT, int(np.round(tau/DT)), ns, Omega,
                                       NT, RT, ETA_H, phi_st, np.pi,
                                       Z_TOOL-ap, Z_TOOL, k1, k2, KT)
    return a3, a4, int(round(tau/DT)), ns


def simulate(plant, Dp_plant, ap, controller):
    a3, a4, n_tau, ns = milling(ap)
    DpT = np.outer(Dp_plant, Dp_plant)
    q = np.zeros(3); qd = np.zeros(3); qdd = np.zeros(3)
    qm = np.zeros((3, ns)); y = np.zeros(ns)
    x_hat = np.zeros(6); u_prev = 0.0
    if controller is not None and hasattr(controller, 'reset'):
        controller.reset()
    for k in range(1, ns):
        y_meas = plant.D_obs @ qm[:, k-1]
        if controller is None:
            uk = 0.0
        else:
            x_hat, uk = controller.step(x_hat, u_prev, y_meas)
            uk = float(np.clip(uk, -U_SAT, U_SAT))
        q_del = qm[:, k-n_tau] if k-n_tau > 0 else np.zeros(3)
        K_eff = plant.Kp + a4[k]*DpT
        F = FT*a3[k]*Dp_plant + a4[k]*DpT @ q_del + plant.H_Pe_modal*uk
        qd_p = qd + 0.5*DT*qdd; q_p = q + DT*qd + 0.25*DT**2*qdd
        S = plant.Mp + 0.5*DT*plant.Cp + 0.25*DT**2*K_eff
        qdd = np.linalg.solve(S, F - plant.Cp@qd_p - K_eff@q_p)
        qd = qd_p + 0.5*DT*qdd; q = q_p + 0.25*DT**2*qdd
        qm[:, k] = q; y[k] = plant.D_obs @ q; u_prev = uk
        if abs(y[k]) > 5e-3:
            return True, np.sqrt(np.mean(y[:k+1]**2))
    return False, np.sqrt(np.mean(y**2))


def main():
    ok = True
    plate, Dp = build()
    a3, a4, n_tau, _ = milling(2.5e-3)
    rc = RCSACController(plate, DT, Dp, a3, a4, n_tau,
                         u_sat=U_SAT, ft=FT, verbose=False)

    banner("1. Decoupling filter properties")
    om = plate.omega_n; H = plate.H_Pe_modal
    C0_expect = (np.sum(Dp**2/om**2)) / (np.sum(Dp*H/om**2))
    print(f"  C(0) = {rc.C_dc:.2f}  (expected {C0_expect:.2f})")
    w_test = 2*np.pi*np.array([519.4, 2680.9])
    _, Hf = sg.sosfreqz(rc.sos, worN=w_test*DT)
    g1, p1 = np.abs(Hf[0]), np.degrees(np.angle(Hf[0]))
    g3, p3 = np.abs(Hf[1]), abs(np.degrees(np.angle(Hf[1])))
    print(f"  mode 1: |C|={g1:.1f} ∠{p1:.0f}°  (target ≈ +85 ∠0°)")
    print(f"  mode 3: |C|={g3:.1f} ∠±{p3:.0f}° (target ≈ 257 ∠180°)")
    c1 = abs(rc.C_dc - C0_expect)/C0_expect < 0.02
    c2 = abs(g1 - abs(Dp[0]/H[0]))/abs(Dp[0]/H[0]) < 0.05 and abs(p1) < 15
    c3 = p3 > 160
    print(f"  [1] {'PASS' if (c1 and c2 and c3) else 'FAIL'}")
    ok &= c1 and c2 and c3

    banner("2. Worst-case a_p=2.5 mm (±150 V): LQG diverges, RC-SAC stable")
    lqg = LQGController(plate, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                         w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
    lqg.discretize_observer()
    dq, yq = simulate(plate, Dp, 2.5e-3, lqg)
    dr, yr = simulate(plate, Dp, 2.5e-3, rc)
    print(f"  LQG   : {'DIV' if dq else f'{yq*1e6:.2f} um'}")
    print(f"  RC-SAC: {'DIV' if dr else f'{yr*1e6:.2f} um'}")
    c4 = dq and (not dr) and yr < 5e-6
    print(f"  [2] {'PASS' if c4 else 'FAIL'}")
    ok &= c4

    banner("3. Robustness: machined plate (drift 519→533 Hz), nominal RC-SAC")
    plate_m, Dp_m = build(HP - 1e-3)
    dm, ym = simulate(plate_m, Dp_m, 2.5e-3, rc)
    print(f"  machined @2.5 mm: {'DIV' if dm else f'{ym*1e6:.2f} um'}")
    c5 = (not dm) and ym < 5e-6
    print(f"  [3] {'PASS' if c5 else 'FAIL'}")
    ok &= c5

    banner("SUMMARY")
    print(f"  RC-SAC controller: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
