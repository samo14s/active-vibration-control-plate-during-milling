"""
gen_controller_limits.py
========================
STRESS TEST — where does the twin-calibrated RC-SAC stop working, and why?

A Q1 contribution must state its operating envelope and its failure boundary,
not only its successes. This script pushes three INDEPENDENT stressors to the
breaking point and identifies the mechanism that bounds each:

  (a) OPERATING ENVELOPE  a_p^crit(beta): stable depth of cut vs model error for
      fixed-model vs twin-calibrated RC-SAC. The twin hugely enlarges the
      envelope, but it is NOT unbounded — it saturates against a ceiling.
  (b) AUTHORITY LIMIT  max|u| vs a_p (well-calibrated, beta=+9%): the required
      piezo voltage rises with depth and hits the ±150 V saturation. THAT is
      the ceiling in (a): a fundamental actuator-authority limit, not a model
      limit — no tuning can beat it.
  (c) CALIBRATION LIMIT  probe recovery vs beta: the FEM-guided NARROW-band
      probe recovers the true frequency to <2 % only while the true mode stays
      inside the (0.80, 1.22)·f_FEM search band, i.e. beta in [-20 %, +22 %].
      Past the band edge the estimate clamps and the correction degrades — the
      price of the FEM prior that makes the probe sharp and cheap.
  (d) BREAKDOWN  a time trace just beyond the ceiling (beta=+9 %, a_p=5 mm):
      the piezo saturates at ±150 V, loses authority, and even the twin-
      calibrated controller diverges — an honest picture of the hard limit.

Output: docs/controller_limits.png (+ .pdf)
Run:  python gen_controller_limits.py     (~2 min)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("01_core", "02_controllers"):
    sys.path.insert(0, os.path.join(_HERE, "..", sub))

from digital_twin import MillingDigitalTwin
from milling_force import precompute_alpha_periodic
from rcsac_controller import RCSACController

# ------------------------------------------------------------------ params
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT = 925e6; KN = 0.26; MU_C = 0.20
AE = 0.1e-3; FT = 0.02e-3
D31 = 175e-12; H_PA = 0.7e-3; E_PE = 63e9; NU_PE = 0.35
ZETA = [0.0031, 0.0017, 0.0027]
DT = 5e-5; T_SWEEP = 0.25; U_SAT = 150.0
RPM = 4900
Z_TOOL = HP - 5e-3; X_TOOL = LP/2

OUT = os.path.join(_HERE, "..", "docs")
os.makedirs(OUT, exist_ok=True)
PATCH = dict(xP1=0, xP2=0.020, zP1=0, zP2=0.060,
             d31=D31, h_Pa=H_PA, E_Pe=E_PE, nu_Pe=NU_PE)

twin = MillingDigitalTwin(LP, HP, BP, RHO, E_AL, NU,
                          X_TOOL, Z_TOOL, LP, Z_TOOL, PATCH,
                          N1=30, N2=24, n_modes=3, zeta_modes=ZETA)
F_FEM = twin.f_fem_intact[0]
PROBE_BAND = (0.80, 1.22)                       # matches design_probe default


def milling(ap, t_end):
    phi_st = np.pi - np.arccos(1 - AE/(DT_TOOL/2))
    Omega = 2*np.pi*RPM/60; RT = DT_TOOL/2; tau = 60/(NT*RPM)
    k1 = KN*np.cos(ETA_H)
    k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)
    ns = int(round(t_end/DT)) + 1
    a3, a4 = precompute_alpha_periodic(DT, int(np.round(tau/DT)), ns, Omega, NT,
                                       RT, ETA_H, phi_st, np.pi, Z_TOOL-ap, Z_TOOL,
                                       k1, k2, KT)
    return a3, a4, int(round(tau/DT)), ns


def true_plant(beta):
    p = twin.predict(0.0)
    om = p.omega_fem * (1 + beta)
    return (p.Dp_tool.copy(), p.D_obs.copy(), p.H_Pe_modal.copy(),
            np.diag(om**2), np.diag(2*np.array(ZETA)*om))


def probe_true(beta):
    u = twin.design_probe(DT, T_probe=0.10, amp=20.0, band=PROBE_BAND)
    Dp, Do, Hpz, Kp, Cp = true_plant(beta)
    ns = len(u); q = np.zeros(3); qd = np.zeros(3); qdd = np.zeros(3); y = np.zeros(ns)
    for k in range(1, ns):
        S = np.eye(3) + 0.5*DT*Cp + 0.25*DT**2*Kp
        qd_p = qd + 0.5*DT*qdd; q_p = q + DT*qd + 0.25*DT**2*qdd
        qdd = np.linalg.solve(S, Hpz*u[k] - Cp@qd_p - Kp@q_p)
        qd = qd_p + 0.5*DT*qdd; q = q_p + 0.25*DT**2*qdd
        y[k] = Do @ q
    return u, y + np.random.default_rng(1).normal(0, 3e-9, ns)


def simulate(beta, ap, controller, t_end=T_SWEEP):
    a3, a4, n_tau, ns = milling(ap, t_end)
    Dp, Do, Hpz, Kp, Cp = true_plant(beta)
    DpT = np.outer(Dp, Dp)
    q = np.zeros(3); qd = np.zeros(3); qdd = np.zeros(3)
    qm = np.zeros((3, ns)); y = np.zeros(ns); u = np.zeros(ns)
    x_hat = np.zeros(6); u_prev = 0.0; controller.reset()
    umax = 0.0
    for k in range(1, ns):
        x_hat, uk = controller.step(x_hat, u_prev, Do @ qm[:, k-1])
        uk = float(np.clip(uk, -U_SAT, U_SAT)); u[k] = uk
        umax = max(umax, abs(uk))
        q_del = qm[:, k-n_tau] if k-n_tau > 0 else np.zeros(3)
        K_eff = Kp + a4[k]*DpT
        F = FT*a3[k]*Dp + a4[k]*DpT @ q_del + Hpz*uk
        qd_p = qd + 0.5*DT*qdd; q_p = q + DT*qd + 0.25*DT**2*qdd
        S = np.eye(3) + 0.5*DT*Cp + 0.25*DT**2*K_eff
        qdd = np.linalg.solve(S, F - Cp@qd_p - K_eff@q_p)
        qd = qd_p + 0.5*DT*qdd; q = q_p + 0.25*DT**2*qdd
        qm[:, k] = q; y[k] = Do @ q; u_prev = uk
        if abs(y[k]) > 5e-3:
            return dict(t=np.arange(ns)*DT, y=y, u=u, stop=k, diverged=True,
                        umax=umax, yrms=np.sqrt(np.mean(y[:k+1]**2)))
    return dict(t=np.arange(ns)*DT, y=y, u=u, stop=ns-1, diverged=False,
                umax=umax, yrms=np.sqrt(np.mean(y**2)))


def rcsac_on(plate, ap):
    a3, a4, n_tau, _ = milling(ap, T_SWEEP)
    return RCSACController(plate, DT, plate.Dp_tool, a3, a4, n_tau,
                           u_sat=U_SAT, ft=FT, verbose=False)


AP_SCAN = np.round(np.arange(0.5, 6.01, 0.25), 3) * 1e-3


def a_p_crit(plate, beta):
    """Deepest STABLE depth of cut: the last a_p that stays stable before the
    first one that diverges (0 if even the shallowest diverges; the scan max if
    none diverges — censored)."""
    last_stable = 0.0
    for ap in AP_SCAN:
        if simulate(beta, ap, rcsac_on(plate, ap))['diverged']:
            return last_stable
        last_stable = ap
    return AP_SCAN[-1]                           # censored: stable over whole range


# =================================================== (a) operating envelope
print("(a) operating envelope a_p_crit(beta) fixed vs twin ...")
BETAS_ENV = np.round(np.arange(-0.20, 0.201, 0.04), 3)
apc_fixed, apc_twin = [], []
p_fixed = twin.corrected_model(0.0, theta=1.0)
for b in BETAS_ENV:
    u, y = probe_true(b); twin.calibrate(u, y, DT)
    p_twin = twin.corrected_model(0.0)
    af = a_p_crit(p_fixed, b); at = a_p_crit(p_twin, b)
    apc_fixed.append(af); apc_twin.append(at)
    print(f"    beta={b*100:+4.0f}%: fixed={af*1e3:.2f}mm  twin={at*1e3:.2f}mm")
apc_fixed = np.array(apc_fixed); apc_twin = np.array(apc_twin)

# =================================================== (b) authority limit
print("(b) authority limit: max|u| vs a_p at beta=+9% ...")
BETA_AUTH = 0.09
u, y = probe_true(BETA_AUTH); twin.calibrate(u, y, DT)
p_auth = twin.corrected_model(0.0)
AP_AUTH = np.round(np.arange(0.5, 5.01, 0.5), 3) * 1e-3
umax_auth, div_auth, rms_auth = [], [], []
for ap in AP_AUTH:
    r = simulate(BETA_AUTH, ap, rcsac_on(p_auth, ap))
    umax_auth.append(r['umax']); div_auth.append(r['diverged'])
    rms_auth.append(r['yrms'])
    print(f"    a_p={ap*1e3:.1f}mm  max|u|={r['umax']:.0f}V  {'DIV' if r['diverged'] else 'stable'}")
umax_auth = np.array(umax_auth); div_auth = np.array(div_auth)
stable_aps = AP_AUTH[~div_auth]
ap_ceiling = stable_aps.max() if stable_aps.size else AP_AUTH[0]   # deepest STABLE depth
ap_fail = AP_AUTH[div_auth].min() if div_auth.any() else AP_AUTH[-1]

# =================================================== (c) calibration limit
print("(c) calibration limit: probe recovery vs beta ...")
BETAS_PROBE = np.round(np.arange(-0.30, 0.301, 0.02), 3)
err_probe = []
for b in BETAS_PROBE:
    u, y = probe_true(b); th, fe = twin.calibrate(u, y, DT)
    ft = F_FEM * (1 + b); err_probe.append(abs(fe - ft) / ft * 100)
err_probe = np.array(err_probe)
band_lo, band_hi = PROBE_BAND[0] - 1, PROBE_BAND[1] - 1     # beta band edges

# =================================================== (d) breakdown trace
print("(d) breakdown time trace at a_p=5 mm, beta=+9% ...")
u, y = probe_true(0.09); twin.calibrate(u, y, DT)
res_brk = simulate(0.09, 5.0e-3, rcsac_on(twin.corrected_model(0.0), 5.0e-3), t_end=0.10)

# =========================================================== plot
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(2, 2, figsize=(15, 9.6))
C_FIX, C_TW, C_U = '#1f4e79', '#c0392b', '#8e44ad'

# (a) envelope
a = ax[0, 0]
a.fill_between(BETAS_ENV*100, 0, apc_twin*1e3, color=C_TW, alpha=0.10)
a.plot(BETAS_ENV*100, apc_fixed*1e3, 's-', color=C_FIX, lw=1.8, ms=5, label='fixed-model RC-SAC')
a.plot(BETAS_ENV*100, apc_twin*1e3, 'D-', color=C_TW, lw=1.8, ms=5, label='twin-calibrated RC-SAC')
a.axhline(ap_ceiling*1e3, ls=':', color='k', lw=1.4, alpha=0.7,
          label=f'authority ceiling ≈{ap_ceiling*1e3:.1f} mm (±{U_SAT:.0f} V)')
a.set_title("(a) Operating envelope — stable depth vs model error\n"
            "twin enlarges it, but a saturation ceiling remains", fontweight='bold')
a.set_xlabel("model error $\\beta$ (%)"); a.set_ylabel("$a_p^{crit}$ (mm)")
a.grid(alpha=0.4); a.legend(fontsize=8.5, loc='lower center')

# (b) authority limit
a = ax[0, 1]
ok = ~div_auth
a.plot(AP_AUTH[ok]*1e3, umax_auth[ok], 'o-', color=C_U, lw=1.8, ms=6, label='max |u| (stable)')
if div_auth.any():
    a.plot(AP_AUTH[div_auth]*1e3, umax_auth[div_auth], 'x', color=C_FIX, ms=9, mew=2,
           label='diverged (saturated)')
a.axhline(U_SAT, ls='--', color='k', lw=1.3, alpha=0.7, label=f'piezo saturation ±{U_SAT:.0f} V')
a.axvspan(ap_fail*1e3, AP_AUTH[-1]*1e3 + 0.25, color=C_FIX, alpha=0.06)
a.set_title("(b) Why the ceiling: actuator-authority limit\n"
            "required voltage hits ±150 V, then control fails", fontweight='bold')
a.set_xlabel("depth of cut $a_p$ (mm)"); a.set_ylabel("peak control voltage |u| (V)")
a.grid(alpha=0.4); a.legend(fontsize=8.5, loc='lower right')

# (c) calibration limit
a = ax[1, 0]
a.axvspan(band_lo*100, band_hi*100, color='green', alpha=0.08,
          label=f'probe search band [{band_lo*100:.0f}%, {band_hi*100:+.0f}%]')
a.plot(BETAS_PROBE*100, err_probe, 'o-', color=C_TW, lw=1.6, ms=4, label='probe recovery error')
a.axhline(2.0, ls='--', color='k', lw=1, alpha=0.6, label='2 % tolerance')
a.axvline(band_lo*100, ls=':', color='green', lw=1.2)
a.axvline(band_hi*100, ls=':', color='green', lw=1.2)
a.set_title("(c) Calibration limit — the FEM-guided probe band\n"
            "clamps past the band edge (β outside [−20%, +22%])", fontweight='bold')
a.set_xlabel("model error $\\beta$ (%)"); a.set_ylabel("mode-1 recovery error (%)")
a.grid(alpha=0.4); a.legend(fontsize=8.5, loc='upper center')

# (d) breakdown
a = ax[1, 1]
i = res_brk['stop']
a2 = a.twinx()
lu, = a2.plot(res_brk['t'][:i+1]*1e3, res_brk['u'][:i+1], color=C_U, lw=0.5, alpha=0.55,
              label='u(t) — saturated ±150 V')
a2.axhline(U_SAT, ls='--', color='k', lw=1, alpha=0.5)
a2.axhline(-U_SAT, ls='--', color='k', lw=1, alpha=0.5)
a2.set_ylabel("u (V)", color=C_U); a2.tick_params(axis='y', labelcolor=C_U)
a2.set_ylim(-U_SAT*1.35, U_SAT*1.35)
ly, = a.plot(res_brk['t'][:i+1]*1e3, res_brk['y'][:i+1]*1e6, color=C_TW, lw=1.1,
             label='$y_p$ (diverges)', zorder=5)
a.set_xlabel("time (ms)"); a.set_ylabel("$y_p$ (µm)", color=C_TW)
a.tick_params(axis='y', labelcolor=C_TW)
a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)
a.set_title(f"(d) Breakdown beyond the ceiling — $a_p$=5 mm, β=+9%\n"
            f"piezo saturates, twin diverges @{res_brk['t'][i]*1e3:.0f} ms", fontweight='bold')
a.grid(alpha=0.3)
a.legend([ly, lu], [ly.get_label(), lu.get_label()], fontsize=8.5, loc='lower left')

fig.suptitle("Limits of the twin-calibrated RC-SAC — operating envelope and its two boundaries\n"
             f"depth ceiling ≈{ap_ceiling*1e3:.1f} mm set by ±{U_SAT:.0f} V authority; "
             "model-error window β∈[−20%, +22%] set by the FEM-guided probe band",
             fontsize=12.5, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.92])
png = os.path.join(OUT, "controller_limits.png")
fig.savefig(png, dpi=150, bbox_inches='tight')
fig.savefig(png.replace(".png", ".pdf"), bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {png}")
print(f"\nSUMMARY  depth ceiling≈{ap_ceiling*1e3:.1f}mm (saturation)  "
      f"model window β∈[{band_lo*100:.0f}%,{band_hi*100:+.0f}%] (probe band)")
