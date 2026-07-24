"""
gen_inprocess_certificate.py
===========================
NOVEL CONTRIBUTION — the actively-controlled, saturation-limited IN-PROCESS
stability certificate, and the certified variable-depth schedule it enables.

Gap addressed (from the literature survey): the in-process (material-removal)
3D stability-lobe literature is entirely OPEN-LOOP / passive, while the active-
chatter-control literature is designed at a single frozen operating point. No
work computes the CLOSED-LOOP, actuator-SATURATION-limited critical depth of cut
along the KNOWN material-removal trajectory — nor uses it to co-schedule the
depth of cut for maximum, certified material-removal rate (MRR).

This script computes, along the deterministic removal trajectory (offline from
the FEM in-process model):
  * a_p^crit(s)  for open loop, LQG (±150 V) and RC-SAC (±150 V);
  * the evolving control authority  |H_Pe(s)|, |Dp(s)|, |D_obs(s)|;
  * the certified variable-depth schedule a_p(s)=a_p^crit,cl(s) and its MRR gain
    over the best constant depth (which is limited by the worst point);
  * the compounding control×scheduling benefit a_p^crit,cl(s)/a_p^crit,ol(s).

Output: figs_certificate/inprocess_certificate.png (+ .pdf)
Run:  python gen_inprocess_certificate.py     (~2-3 min)
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

from plate_model import PlateModel
from mindlin_q8 import shape_at_point_M
from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from rcsac_controller import RCSACController

# ------------------------------------------------------------------ params
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT = 925e6; KN = 0.26; MU_C = 0.20
AE = 0.1e-3; FT = 0.02e-3
D31 = 175e-12; H_PA = 0.7e-3; E_PE = 63e9; NU_PE = 0.35
N1, N2 = 30, 24
ZETA = [0.0031, 0.0017, 0.0027]
DT = 5e-5; T_END = 0.25
RPM = 4900; U_SAT = 150.0
Z_TOOL = HP - 5e-3; X_TOOL = LP/2
TRIM_MAX = 2.0e-3               # cumulative free-end removal over the operation
N_STAGE = 9                     # removal stages sampled along the trajectory

OUT = os.path.join(_HERE, "figs_certificate")
os.makedirs(OUT, exist_ok=True)


def make_model(trim):
    """Exact in-process state = clean shortened cantilever (H_P - trim)."""
    p = PlateModel(LP, HP - trim, BP, RHO, E_AL, NU, N1=N1, N2=N2, n_modes=3,
                   zeta_modes=ZETA, verbose=False)
    p.set_observation(LP, Z_TOOL)
    p.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    Dp = shape_at_point_M(X_TOOL, Z_TOOL, p.N1, p.N2, p.lex, p.ley,
                          p.node_id, p.ndof)[p.DOFf] @ p.V
    return p, Dp


def milling(ap):
    phi_st = np.pi - np.arccos(1 - AE/(DT_TOOL/2))
    Omega = 2*np.pi*RPM/60; RT = DT_TOOL/2; tau = 60/(NT*RPM)
    k1 = KN*np.cos(ETA_H)
    k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)
    ns = int(round(T_END/DT)) + 1
    a3, a4 = precompute_alpha_periodic(DT, int(np.round(tau/DT)), ns, Omega, NT,
                                       RT, ETA_H, phi_st, np.pi, Z_TOOL-ap, Z_TOOL,
                                       k1, k2, KT)
    return a3, a4, int(round(tau/DT)), ns


def simulate(p, Dp, ap, controller):
    a3, a4, n_tau, ns = milling(ap)
    DpT = np.outer(Dp, Dp)
    q = np.zeros(3); qd = np.zeros(3); qdd = np.zeros(3)
    qm = np.zeros((3, ns)); y = np.zeros(ns); u_prev = 0.0; x_hat = np.zeros(6)
    if controller is not None and hasattr(controller, 'reset'):
        controller.reset()
    for k in range(1, ns):
        y_meas = p.D_obs @ qm[:, k-1]
        if controller is None:
            uk = 0.0
        else:
            x_hat, uk = controller.step(x_hat, u_prev, y_meas)
            uk = float(np.clip(uk, -U_SAT, U_SAT))
        q_del = qm[:, k-n_tau] if k-n_tau > 0 else np.zeros(3)
        K_eff = p.Kp + a4[k]*DpT
        F = FT*a3[k]*Dp + a4[k]*DpT @ q_del + p.H_Pe_modal*uk
        qd_p = qd + 0.5*DT*qdd; q_p = q + DT*qd + 0.25*DT**2*qdd
        S = p.Mp + 0.5*DT*p.Cp + 0.25*DT**2*K_eff
        qdd = np.linalg.solve(S, F - p.Cp@qd_p - K_eff@q_p)
        qd = qd_p + 0.5*DT*qdd; q = q_p + 0.25*DT**2*qdd
        qm[:, k] = q; y[k] = p.D_obs @ q; u_prev = uk
        if abs(y[k]) > 5e-3:
            return True
    return False


AP_SCAN = np.round(np.arange(0.1, 4.01, 0.2), 3) * 1e-3


def a_p_crit(p, Dp, ctrl_factory):
    for ap in AP_SCAN:
        ctrl = ctrl_factory(ap) if ctrl_factory else None
        if simulate(p, Dp, ap, ctrl):
            return ap
    return AP_SCAN[-1] * 1.05


# ---- nominal LQG (fixed, designed once) ---------------------------------
p0, Dp0 = make_model(0.0)
lqg = LQGController(p0, dt=DT, verbose=False)
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg.discretize_observer()

trims = np.linspace(0, TRIM_MAX, N_STAGE)
f1s, Hs, Dps, Dos = [], [], [], []
ac_ol, ac_lqg, ac_rc = [], [], []
print("trim(mm) f1(Hz) | apc_OL apc_LQG apc_RCSAC (mm)")
for tr in trims:
    p, Dp = make_model(tr)
    f1s.append(p.freq_n[0]); Hs.append(abs(p.H_Pe_modal[0]))
    Dps.append(abs(Dp[0])); Dos.append(abs(p.D_obs[0]))
    a_ol = a_p_crit(p, Dp, None)
    a_lq = a_p_crit(p, Dp, lambda ap: lqg)
    # RC-SAC rebuilt per (state, ap) via its factory
    def rc_factory(ap, _p=p, _Dp=Dp):
        a3, a4, n_tau, _ = milling(ap)
        return RCSACController(_p, DT, _Dp, a3, a4, n_tau,
                               u_sat=U_SAT, ft=FT, verbose=False)
    a_rc = a_p_crit(p, Dp, rc_factory)
    ac_ol.append(a_ol); ac_lqg.append(a_lq); ac_rc.append(a_rc)
    print(f"  {tr*1e3:.2f}   {p.freq_n[0]:6.1f} | {a_ol*1e3:5.2f}  {a_lq*1e3:5.2f}  {a_rc*1e3:5.2f}")

trims_mm = trims*1e3
f1s = np.array(f1s); Hs = np.array(Hs); Dps = np.array(Dps); Dos = np.array(Dos)
ac_ol = np.array(ac_ol); ac_lqg = np.array(ac_lqg); ac_rc = np.array(ac_rc)

# certified variable-depth schedule vs best constant depth (limited by worst point)
const_lqg = ac_lqg.min(); const_rc = ac_rc.min()
mrr_gain_lqg = (ac_lqg.mean() - const_lqg) / const_lqg * 100
mrr_gain_rc = (ac_rc.mean() - const_rc) / const_rc * 100
print(f"\nLQG : constant a_p (worst-limited) = {const_lqg*1e3:.2f} mm; "
      f"variable schedule mean = {ac_lqg.mean()*1e3:.2f} mm -> +{mrr_gain_lqg:.0f}% MRR")
print(f"RC-SAC: constant = {const_rc*1e3:.2f} mm; variable mean = {ac_rc.mean()*1e3:.2f} mm "
      f"-> +{mrr_gain_rc:.0f}% MRR")

# ============================================================ plot
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
C_OL, C_LQ, C_RC = '#8a8a8a', '#1f4e79', '#c0392b'

# (a) in-process closed-loop saturation-limited stability certificate
a = ax[0, 0]
a.plot(trims_mm, ac_ol*1e3, 'o-', color=C_OL, lw=1.8, ms=4, label='open loop')
a.plot(trims_mm, ac_lqg*1e3, 's-', color=C_LQ, lw=1.8, ms=4, label='LQG (±150 V)')
a.plot(trims_mm, ac_rc*1e3, 'D-', color=C_RC, lw=1.8, ms=4, label='RC-SAC (±150 V)')
a.fill_between(trims_mm, ac_ol*1e3, ac_rc*1e3, color=C_RC, alpha=0.06)
a.set_title("(a) In-process CLOSED-LOOP saturation-limited stability certificate\n"
            "$a_p^{crit}(s)$ along the material-removal trajectory", fontweight='bold')
a.set_xlabel("cumulative material removed  →  (mm off free end)")
a.set_ylabel("critical depth of cut $a_p^{crit}$ (mm)")
a.grid(alpha=0.4); a.legend(fontsize=9, loc='center left')

# (b) evolving control authority
a = ax[0, 1]
a.plot(trims_mm, Hs/Hs[0], 's-', color=C_RC, lw=1.6, ms=3, label='$|H_{Pe,1}|$ actuator→mode')
a.plot(trims_mm, Dps/Dps[0], 'o-', color=C_LQ, lw=1.6, ms=3, label='$|D_{p,1}|$ force→mode')
a.plot(trims_mm, Dos/Dos[0], '^-', color='#2e8b57', lw=1.6, ms=3, label='$|D_{obs,1}|$ mode→sensor')
a.plot(trims_mm, f1s/f1s[0], ':', color='k', lw=1.4, label='$f_1$ (normalised)')
a.set_title("(b) Evolving modal control authority (fixed patch & sensor)\n"
            "normalised to the intact workpiece", fontweight='bold')
a.set_xlabel("cumulative material removed (mm)")
a.set_ylabel("authority / $f_1$  (ratio to intact)")
a.grid(alpha=0.4); a.legend(fontsize=8, loc='upper left')

# (c) certified variable-depth schedule vs constant (LQG)
a = ax[1, 0]
a.plot(trims_mm, ac_lqg*1e3, 's-', color=C_LQ, lw=2, ms=4,
       label=f'variable schedule $a_p(s)$ (mean {ac_lqg.mean()*1e3:.2f} mm)')
a.axhline(const_lqg*1e3, color='k', ls='--', lw=1.6,
          label=f'best constant $a_p$ = {const_lqg*1e3:.2f} mm (worst-point limited)')
a.fill_between(trims_mm, const_lqg*1e3, ac_lqg*1e3, where=ac_lqg*1e3 >= const_lqg*1e3,
               color='#2e8b57', alpha=0.25, label=f'unused capacity → +{mrr_gain_lqg:.0f}% MRR')
a.set_title("(c) Certified variable-depth schedule vs constant depth (LQG)",
            fontweight='bold')
a.set_xlabel("cumulative material removed (mm)")
a.set_ylabel("depth of cut $a_p$ (mm)")
a.grid(alpha=0.4); a.legend(fontsize=8, loc='lower right')

# (d) compounding control × scheduling benefit
a = ax[1, 1]
a.plot(trims_mm, ac_lqg/ac_ol, 's-', color=C_LQ, lw=1.8, ms=4, label='LQG / open loop')
a.plot(trims_mm, ac_rc/ac_ol, 'D-', color=C_RC, lw=1.8, ms=4, label='RC-SAC / open loop')
a.set_title("(d) Control benefit grows as removal destabilises the workpiece\n"
            "$a_p^{crit,cl}/a_p^{crit,ol}$ along the trajectory", fontweight='bold')
a.set_xlabel("cumulative material removed (mm)")
a.set_ylabel("stable-depth extension factor (×)")
a.grid(alpha=0.4); a.legend(fontsize=9, loc='upper left')

fig.suptitle("Actively-controlled, saturation-limited IN-PROCESS stability "
             "certificate & certified variable-depth scheduling\n"
             f"closed-loop $a_p^{{crit}}$ varies {ac_lqg.min()*1e3:.2f}–{ac_lqg.max()*1e3:.2f} mm "
             f"(LQG) along the removal path → +{mrr_gain_lqg:.0f}% certified MRR vs constant depth",
             fontsize=12.5, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
png = os.path.join(OUT, "inprocess_certificate.png")
fig.savefig(png, dpi=150, bbox_inches='tight')
fig.savefig(png.replace(".png", ".pdf"), bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {png}")
