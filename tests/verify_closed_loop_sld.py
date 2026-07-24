"""
Cross-check the coupled closed-loop monodromy against the existing per-mode
open-loop solver, verify mesh convergence, and quantify the error of the
'equivalent damping substitution' the package currently uses for closed-loop
lobes. Finally cross-validate the certified lobe against direct time-domain
simulation, which is the only independent check available.
"""
import _bootstrap  # noqa: F401
import numpy as np

from plate_model import PlateModel
from lqg_controller import LQGController
from fdm_stability import fdm_stability_one_mode
from milling_force import precompute_alpha_periodic
from newmark_solver import NewmarkSimulator
from closed_loop_sld import spectral_radius, critical_depth

LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT, DT_TOOL = 3, 0.010
ETA_H, GAMMA_N = np.deg2rad(35), np.deg2rad(15)
KT, KN, MU_C = 925e6, 0.26, 0.20
AE, FT = 0.1e-3, 0.02e-3
D31, H_PA, E_PE, NU_PE = 175e-12, 0.7e-3, 63e9, 0.35
ZETA = [0.0031, 0.0017, 0.0027]
N_MODES = 3
AP = 0.3e-3
TS = 5e-5

plate = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=30, N2=24,
                   n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
plate.precompute_Dp(zp_pos=HP - AP / 2, n_pos=2001)
plate.set_observation(x_obs=LP, z_obs=HP)
plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)

phi_st = np.pi - np.arccos(1 - AE / (DT_TOOL / 2))
phi_ex = np.pi
RT = DT_TOOL / 2
k1 = KN * np.cos(ETA_H)
k2 = 1 + MU_C * np.tan(ETA_H) * np.cos(GAMMA_N) - KN * np.sin(ETA_H)

Dp_mid = plate.Dp_array[:, 1000]
m_list = np.diag(plate.Mp)

COMMON = dict(NT=NT, RT=RT, eta_h=ETA_H, phi_st=phi_st, phi_ex=phi_ex,
              hp=HP, k1=k1, k2=k2, kt=KT)

print("=" * 78)
print(" TEST 1 : open loop -- new coupled solver vs existing per-mode solver")
print("=" * 78)
print(f"{'RPM':>7} {'ap[mm]':>8} {'rho coupled':>14} {'rho max-per-mode':>18} {'diff':>9}")
for RPM in (3500, 4900, 6200):
    for ap in (0.1e-3, 0.3e-3, 1.0e-3):
        rho_new = spectral_radius(plate.omega_n, ZETA, Dp_mid,
                                  plate.H_Pe_modal, plate.D_obs,
                                  RPM=RPM, ap=ap, Ts=TS, refine=1,
                                  ctrl_design=None, **COMMON)
        rho_old = max(
            fdm_stability_one_mode(plate.omega_n[i], ZETA[i], Dp_mid[i],
                                   m_list[i], RPM, NT, RT, ETA_H,
                                   phi_st, phi_ex, ap, HP, k1, k2, KT,
                                   m_div=40)
            for i in range(N_MODES))
        print(f"{RPM:7.0f} {ap*1e3:8.2f} {rho_new:14.5f} {rho_old:18.5f} "
              f"{(rho_new-rho_old)/rho_old*100:+8.2f}%")
print("  (agreement at small a_p where modes decouple; divergence at larger")
print("   a_p is the inter-modal coupling the per-mode maximum discards)")

print()
print("=" * 78)
print(" TEST 2 : mesh convergence in `refine` at fixed controller rate")
print("=" * 78)
lqg = LQGController(plate, dt=TS, verbose=False)
lqg.optimize_weights(w_q_list=[1e14], w_qd_list=[1e8], w_r=1.0)
lqg.discretize_observer()
design = dict(A=lqg.A, B=lqg.B, C=lqg.C, K=lqg.K_lqr, L=lqg.L_kal)

for refine in (1, 2, 4):
    r_ol = spectral_radius(plate.omega_n, ZETA, Dp_mid, plate.H_Pe_modal,
                           plate.D_obs, RPM=4900, ap=0.3e-3, Ts=TS,
                           refine=refine, ctrl_design=None, **COMMON)
    r_cl = spectral_radius(plate.omega_n, ZETA, Dp_mid, plate.H_Pe_modal,
                           plate.D_obs, RPM=4900, ap=0.3e-3, Ts=TS,
                           refine=refine, ctrl_design=design, **COMMON)
    print(f"   refine = {refine:2d}  (m_div = {82*refine:4d})   "
          f"rho_OL = {r_ol:.6f}   rho_CL = {r_cl:.6f}")

print()
print("=" * 78)
print(" TEST 3 : certified closed loop vs equivalent-damping substitution")
print("=" * 78)


def extract_modes(ev, n_modes):
    """The routine the package uses to turn closed-loop poles into (w, zeta)."""
    ev = np.array(ev)
    pos = ev[np.imag(ev) > 1e-9]
    pos = pos[np.argsort(np.abs(pos))][:n_modes]
    return np.real(np.abs(pos)), np.real(-np.real(pos) / np.abs(pos))


w_cl, z_cl = extract_modes(lqg.ev_cl, N_MODES)
print(f"  closed-loop poles -> zeta = {np.round(z_cl*100,2)} %   "
      f"(open loop {np.round(np.array(ZETA)*100,2)} %)")

RPM_T = 4900
ap_ol = critical_depth(lambda ap: spectral_radius(
    plate.omega_n, ZETA, Dp_mid, plate.H_Pe_modal, plate.D_obs,
    RPM=RPM_T, ap=ap, Ts=TS, ctrl_design=None, **COMMON))
ap_sub = critical_depth(lambda ap: spectral_radius(
    w_cl, z_cl, Dp_mid, plate.H_Pe_modal, plate.D_obs,
    RPM=RPM_T, ap=ap, Ts=TS, ctrl_design=None, **COMMON))
ap_cert = critical_depth(lambda ap: spectral_radius(
    plate.omega_n, ZETA, Dp_mid, plate.H_Pe_modal, plate.D_obs,
    RPM=RPM_T, ap=ap, Ts=TS, ctrl_design=design, **COMMON))

print()
print(f"  a_p,crit at {RPM_T} RPM")
print(f"    open loop                             : {ap_ol*1e3:8.4f} mm")
print(f"    LQG, equivalent-damping substitution  : {ap_sub*1e3:8.4f} mm "
      f"({ap_sub/ap_ol:6.1f}x)")
print(f"    LQG, CERTIFIED closed-loop monodromy  : {ap_cert*1e3:8.4f} mm "
      f"({ap_cert/ap_ol:6.1f}x)")
print()
err = (ap_sub - ap_cert) / ap_cert * 100
print(f"  >> equivalent-damping substitution is {err:+.1f}% "
      f"{'OPTIMISTIC' if err > 0 else 'PESSIMISTIC'} vs the certified loop")

print()
print("=" * 78)
print(" TEST 4 : independent cross-validation by time-domain simulation")
print("=" * 78)
print("  (the certified a_p,crit should separate decaying from growing runs)")

tau = 60 / (NT * RPM_T)
Omega = 2 * np.pi * RPM_T / 60
n_per = int(round(tau / TS))


def sim_rms(ap, controller, T_end=0.25):
    p = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=30, N2=24,
                   n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
    p.precompute_Dp(zp_pos=HP - ap / 2, n_pos=2001)
    p.set_observation(x_obs=LP, z_obs=HP)
    p.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    s = NewmarkSimulator(p, dt=TS, T_end=T_end, ft=FT, tau=tau, verbose=False)
    a3, a4 = precompute_alpha_periodic(TS, n_per, s.nstep, Omega, NT, RT,
                                       ETA_H, phi_st, phi_ex,
                                       HP - ap, HP, k1, k2, KT)
    # hold the tool at mid-path so the SLD's frozen-position assumption holds
    kp = np.full(s.nstep, 1000, dtype=int)
    c = None
    if controller is not None:
        c = LQGController(p, dt=TS, verbose=False)
        c.optimize_weights(w_q_list=[1e14], w_qd_list=[1e8], w_r=1.0)
        c.discretize_observer()
    r = s.simulate(a3, a4, kp, controller=c, progress=False)
    y = r['y'][:r['stop_idx'] + 1]
    half = len(y) // 2
    # growth ratio between the second and first half of the record
    g1 = np.sqrt(np.mean(y[:half] ** 2))
    g2 = np.sqrt(np.mean(y[half:] ** 2))
    return g2 / max(g1, 1e-30), r['diverged_at']


print(f"\n  open loop  (certified a_p,crit = {ap_ol*1e3:.4f} mm)")
for ap in (0.5 * ap_ol, 0.9 * ap_ol, 1.2 * ap_ol, 2.0 * ap_ol):
    g, dv = sim_rms(ap, None)
    print(f"    ap = {ap*1e3:7.4f} mm  growth ratio = {g:10.3e}  "
          f"{'GROWING' if g > 1.5 else 'decaying'}"
          f"{'  (hit divergence guard)' if dv else ''}")

print(f"\n  LQG        (certified a_p,crit = {ap_cert*1e3:.4f} mm)")
for ap in (0.5 * ap_cert, 0.9 * ap_cert, 1.2 * ap_cert, 2.0 * ap_cert):
    g, dv = sim_rms(ap, True)
    print(f"    ap = {ap*1e3:7.4f} mm  growth ratio = {g:10.3e}  "
          f"{'GROWING' if g > 1.5 else 'decaying'}"
          f"{'  (hit divergence guard)' if dv else ''}")
