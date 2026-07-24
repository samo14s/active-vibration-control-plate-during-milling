"""
verify_digital_twin.py
=====================
Checks for the physics-guided digital twin (see docs/DIGITAL_TWIN.md):

  1. Probe calibration: the FEM-guided chirp probe recovers the true mode-1
     frequency to < 2 % across a ±12 % model error.
  2. Robust control: at a deep cut (a_p = 2.5 mm) with +12 % model error the
     FIXED-model RC-SAC diverges, while the TWIN-calibrated RC-SAC is stable.
  3. Fusion: from ONE calibration, the twin (FEM trend × scale) tracks the true
     in-process frequency along the material-removal path to < 2 %.

Run:  python verify_digital_twin.py     (~40 s)
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "01_core"))
sys.path.insert(0, os.path.join(_HERE, "..", "02_controllers"))

from digital_twin import MillingDigitalTwin
from milling_force import precompute_alpha_periodic
from rcsac_controller import RCSACController

LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT = 925e6; KN = 0.26; MU_C = 0.20
AE = 0.1e-3; FT = 0.02e-3
ZETA = [0.0031, 0.0017, 0.0027]
DT = 5e-5; T_END = 0.25; RPM = 4900; U_SAT = 150.0
Z_TOOL = HP - 5e-3; X_TOOL = LP/2
PATCH = dict(xP1=0, xP2=0.020, zP1=0, zP2=0.060,
             d31=175e-12, h_Pa=0.7e-3, E_Pe=63e9, nu_Pe=0.35)


def banner(t):
    print("\n" + "=" * 66 + f"\n {t}\n" + "=" * 66)


twin = MillingDigitalTwin(LP, HP, BP, RHO, E_AL, NU, X_TOOL, Z_TOOL, LP, Z_TOOL,
                          PATCH, N1=30, N2=24, n_modes=3, zeta_modes=ZETA)


def milling(ap):
    phi = np.pi - np.arccos(1 - AE/(DT_TOOL/2))
    Om = 2*np.pi*RPM/60; RT = DT_TOOL/2; tau = 60/(NT*RPM)
    k1 = KN*np.cos(ETA_H); k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)
    ns = int(round(T_END/DT)) + 1
    a3, a4 = precompute_alpha_periodic(DT, int(round(tau/DT)), ns, Om, NT, RT,
                                       ETA_H, phi, np.pi, Z_TOOL-ap, Z_TOOL, k1, k2, KT)
    return a3, a4, int(round(tau/DT)), ns


def true_plant(removal, beta):
    p = twin.predict(removal); om = p.omega_fem*(1+beta)
    return (p.Dp_tool.copy(), p.D_obs.copy(), p.H_Pe_modal.copy(),
            np.diag(om**2), np.diag(2*np.array(ZETA)*om))


def probe_true(removal, beta):
    u = twin.design_probe(DT, T_probe=0.10, amp=20.0)
    Dp, Do, Hpz, Kp, Cp = true_plant(removal, beta)
    ns = len(u); q = np.zeros(3); qd = np.zeros(3); qdd = np.zeros(3); y = np.zeros(ns)
    for k in range(1, ns):
        S = np.eye(3)+0.5*DT*Cp+0.25*DT**2*Kp
        qd_p = qd+0.5*DT*qdd; q_p = q+DT*qd+0.25*DT**2*qdd
        qdd = np.linalg.solve(S, Hpz*u[k]-Cp@qd_p-Kp@q_p); qd = qd_p+0.5*DT*qdd; q = q_p+0.25*DT**2*qdd
        y[k] = Do@q
    return u, y + np.random.default_rng(1).normal(0, 3e-9, ns)


def simulate(removal, beta, ap, controller):
    a3, a4, n_tau, ns = milling(ap)
    Dp, Do, Hpz, Kp, Cp = true_plant(removal, beta); DpT = np.outer(Dp, Dp)
    q = np.zeros(3); qd = np.zeros(3); qdd = np.zeros(3); qm = np.zeros((3, ns))
    y = np.zeros(ns); x_hat = np.zeros(6); u_prev = 0.0; controller.reset()
    for k in range(1, ns):
        x_hat, uk = controller.step(x_hat, u_prev, Do@qm[:, k-1])
        uk = float(np.clip(uk, -U_SAT, U_SAT))
        q_del = qm[:, k-n_tau] if k-n_tau > 0 else np.zeros(3)
        K_eff = Kp + a4[k]*DpT
        F = FT*a3[k]*Dp + a4[k]*DpT@q_del + Hpz*uk
        qd_p = qd+0.5*DT*qdd; q_p = q+DT*qd+0.25*DT**2*qdd
        S = np.eye(3)+0.5*DT*Cp+0.25*DT**2*K_eff
        qdd = np.linalg.solve(S, F-Cp@qd_p-K_eff@q_p); qd = qd_p+0.5*DT*qdd; q = q_p+0.25*DT**2*qdd
        qm[:, k] = q; y[k] = Do@q; u_prev = uk
        if abs(y[k]) > 5e-3:
            return True, np.sqrt(np.mean(y[:k+1]**2))
    return False, np.sqrt(np.mean(y**2))


def rcsac_on(plate, ap):
    a3, a4, n_tau, _ = milling(ap)
    return RCSACController(plate, DT, plate.Dp_tool, a3, a4, n_tau,
                           u_sat=U_SAT, ft=FT, verbose=False)


def main():
    ok = True

    banner("1. FEM-guided probe recovers the true frequency (±12%)")
    err_max = 0.0
    for b in [-0.12, -0.06, 0.0, 0.06, 0.12]:
        u, y = probe_true(0.0, b); theta, f_est = twin.calibrate(u, y, DT)
        f_true = twin.f_fem_intact[0]*(1+b); err = abs(f_est-f_true)/f_true*100
        err_max = max(err_max, err)
        print(f"  beta={b*100:+4.0f}%: f_true={f_true:6.1f}  f_probe={f_est:6.1f}  err={err:+.2f}%")
    c1 = err_max < 2.0
    print(f"  [1] {'PASS' if c1 else 'FAIL'} (max err {err_max:.2f}%)"); ok &= c1

    banner("2. Deep cut, +12% error: fixed diverges, twin stable")
    beta = 0.12; ap = 2.5e-3
    d_fx, y_fx = simulate(0.0, beta, ap, rcsac_on(twin.corrected_model(0.0, theta=1.0), ap))
    u, y = probe_true(0.0, beta); twin.calibrate(u, y, DT)
    d_tw, y_tw = simulate(0.0, beta, ap, rcsac_on(twin.corrected_model(0.0), ap))
    print(f"  fixed-model : {'DIV' if d_fx else f'{y_fx*1e6:.2f} um'}")
    print(f"  twin        : {'DIV' if d_tw else f'{y_tw*1e6:.2f} um'}")
    c2 = d_fx and (not d_tw)
    print(f"  [2] {'PASS' if c2 else 'FAIL'}"); ok &= c2

    banner("3. One calibration tracks the in-process frequency along the path")
    beta = 0.09
    u, y = probe_true(0.0, beta); twin.calibrate(u, y, DT)
    errs = []
    for r in np.linspace(0, 2.0e-3, 5):
        f_true = twin.predict(r).freq_n[0]*(1+beta)
        f_twin = twin.predicted_frequency(r, corrected=True)
        errs.append(abs(f_twin-f_true)/f_true*100)
    c3 = max(errs) < 2.0
    print(f"  tracking error along path (max) = {max(errs):.2f}%")
    print(f"  [3] {'PASS' if c3 else 'FAIL'}"); ok &= c3

    banner("SUMMARY")
    print(f"  Digital twin: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
