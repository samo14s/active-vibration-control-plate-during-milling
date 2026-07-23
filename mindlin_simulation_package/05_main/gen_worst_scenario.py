"""
gen_worst_scenario.py
====================
Worst-case investigation: regenerative CHATTER (open-loop divergence), how the
machining-induced frequency drift changes the chatter stability, and the ultimate
limit of the LQG controller under a realistic +-150 V piezo saturation.

Spindle speed RPM = 4900 (an open-loop-unstable regime for this plate). Two
machining states are compared:
  * intact          (free end at 80 mm, f1 = 519 Hz)
  * machined 1 mm   (free-end pass removed the top 1 mm -> 79 mm, f1 = 533 Hz)
The machined state is modelled exactly as the clean shortened plate (validated
earlier to 0.003 %), avoiding any spurious modes.

Panels:
  (a) depth-of-cut sweep: open-loop vs LQG y_rms; critical a_p (chatter onset) of
      each, for both machining states, under +-150 V saturation.
  (b) LQG peak voltage vs a_p -> where it hits +-150 V is the controller's limit.
  (c) time response at a_p between the OL and LQG critical depths (LQG rescues a
      cut that chatters open-loop).
  (d) time response BEYOND the LQG critical depth (voltage saturates, even LQG
      diverges) — the absolute worst case.

Output: figs_worst/worst_scenario.png (+ .pdf)
Run:  python gen_worst_scenario.py
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

# ------------------------------------------------------------------ params
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT = 925e6; KN = 0.26; MU_C = 0.20
AE = 0.1e-3; FT = 0.02e-3
D31 = 175e-12; H_PA = 0.7e-3; E_PE = 63e9; NU_PE = 0.35
N1, N2 = 30, 24
ZETA = np.array([0.0031, 0.0017, 0.0027])
DT = 5e-5
T_END = 0.30
RPM = 4900                         # open-loop-unstable (chatter) regime
U_SAT = 150.0                      # piezo saturation (decisive for the worst case)
Z_TOOL = HP - 5e-3
X_TOOL = LP / 2

OUT = os.path.join(_HERE, "figs_worst")
os.makedirs(OUT, exist_ok=True)


def rms(x):
    return np.sqrt(np.mean(x**2))


def make_model(trim):
    """Exact machined state = clean shortened cantilever (H_P - trim)."""
    hp = HP - trim
    p = PlateModel(LP, hp, BP, RHO, E_AL, NU, N1=N1, N2=N2, n_modes=3,
                   zeta_modes=list(ZETA), verbose=False)
    p.set_observation(LP, Z_TOOL)
    p.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    DOFf = p.DOFf
    Dp = shape_at_point_M(X_TOOL, Z_TOOL, p.N1, p.N2, p.lex, p.ley,
                          p.node_id, p.ndof)[DOFf] @ p.V
    return dict(f1=p.freq_n[0], Mp=p.Mp, Kp=p.Kp, Cp=p.Cp,
                D_obs=p.D_obs.copy(), H_Pe=p.H_Pe_modal.copy(), Dp=Dp, plate=p)


# ---- nominal LQG designed once on the intact plate ----------------------
intact = make_model(0.0)
lqg = LQGController(intact['plate'], dt=DT, verbose=False)
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg.discretize_observer()
mach = make_model(1.0e-3)
print(f"RPM={RPM} (2*f_t={2*NT*RPM/60:.0f} Hz), u_sat=±{U_SAT:.0f} V")
print(f"intact f1={intact['f1']:.1f} Hz;  machined-1mm f1={mach['f1']:.1f} Hz\n")


def milling_force(ap):
    phi_st = np.pi - np.arccos(1 - AE/(DT_TOOL/2))
    Omega = 2*np.pi*RPM/60; RT = DT_TOOL/2; tau = 60/(NT*RPM)
    k1 = KN*np.cos(ETA_H)
    k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)
    nstep = int(round(T_END/DT)) + 1
    n_per = int(np.round(tau/DT))
    a3, a4 = precompute_alpha_periodic(DT, n_per, nstep, Omega, NT, RT, ETA_H,
                                       phi_st, np.pi, Z_TOOL - ap, Z_TOOL, k1, k2, KT)
    return a3, a4, int(round(tau/DT)), nstep


def simulate(m, ap, controller):
    Mp, Kp, Cp = m['Mp'], m['Kp'], m['Cp']
    D_obs, H_Pe, Dp = m['D_obs'], m['H_Pe'], m['Dp']
    DpTDp = np.outer(Dp, Dp)
    a3, a4, n_tau, nstep = milling_force(ap)
    t = np.arange(nstep)*DT
    q = np.zeros(3); qd = np.zeros(3); qdd = np.zeros(3)
    qm = np.zeros((3, nstep)); y = np.zeros(nstep); u = np.zeros(nstep)
    x_hat = np.zeros(6); u_prev = 0.0
    gNM, bNM = 0.5, 0.25
    stop = nstep - 1
    for k in range(1, nstep):
        y_meas = D_obs @ qm[:, k-1]
        if controller is not None:
            x_hat, uk = controller.step(x_hat, u_prev, y_meas)
            uk = float(np.clip(uk, -U_SAT, U_SAT))
        else:
            uk = 0.0
        u[k] = uk
        q_delay = qm[:, k-n_tau] if k-n_tau > 0 else np.zeros(3)
        K_eff = Kp + a4[k]*DpTDp
        F = FT*a3[k]*Dp + a4[k]*DpTDp @ q_delay + H_Pe*uk
        qd_pred = qd + (1-gNM)*DT*qdd
        q_pred = q + DT*qd + (0.5-bNM)*DT**2*qdd
        S = Mp + gNM*DT*Cp + bNM*DT**2*K_eff
        qdd = np.linalg.solve(S, F - Cp@qd_pred - K_eff@q_pred)
        qd = qd_pred + gNM*DT*qdd; q = q_pred + bNM*DT**2*qdd
        qm[:, k] = q; y[k] = D_obs @ q; u_prev = uk
        if abs(y[k]) > 5e-3:
            stop = k; break
    return dict(t=t, y=y, u=u, stop=stop, diverged=(stop < nstep-1),
                yrms=rms(y[:stop+1]), umax=np.max(np.abs(u[:stop+1])))


# ---- depth-of-cut sweeps -------------------------------------------------
ap_list = np.round(np.arange(0.1, 3.01, 0.15), 3) * 1e-3
data = {}
for name, m in [("intact", intact), ("machined", mach)]:
    olr, clr, clu, oldiv, cldiv = [], [], [], [], []
    for ap in ap_list:
        ol = simulate(m, ap, None); cl = simulate(m, ap, lqg)
        olr.append(np.nan if ol['diverged'] else ol['yrms']*1e6)
        clr.append(np.nan if cl['diverged'] else cl['yrms']*1e6)
        clu.append(cl['umax']); oldiv.append(ol['diverged']); cldiv.append(cl['diverged'])
    apc_ol = next((ap for ap, d in zip(ap_list, oldiv) if d), ap_list[-1]*1.1)
    apc_cl = next((ap for ap, d in zip(ap_list, cldiv) if d), ap_list[-1]*1.1)
    data[name] = dict(m=m, olr=np.array(olr), clr=np.array(clr), clu=np.array(clu),
                      apc_ol=apc_ol, apc_cl=apc_cl)
    print(f"{name:9s} f1={m['f1']:.1f}Hz:  a_p^crit OL={apc_ol*1e3:.2f}mm  "
          f"LQG={apc_cl*1e3:.2f}mm  (x{apc_cl/apc_ol:.1f})")

# worst machining state = smaller OL critical depth
worst_name = min(data, key=lambda n: data[n]['apc_ol'])
W = data[worst_name]; mW = W['m']
ap_rescue = round((W['apc_ol'] + min(W['apc_cl'], ap_list[-1]))/2, 4)   # OL-crit < ap < LQG-crit
ap_beyond = round(min(W['apc_cl'] + 0.4e-3, ap_list[-1]), 4)            # beyond LQG-crit
tr_ol_r = simulate(mW, ap_rescue, None); tr_cl_r = simulate(mW, ap_rescue, lqg)
tr_ol_b = simulate(mW, ap_beyond, None); tr_cl_b = simulate(mW, ap_beyond, lqg)
print(f"\nworst state = {worst_name} (f1={mW['f1']:.1f}Hz); "
      f"rescue a_p={ap_rescue*1e3:.2f}mm, beyond-limit a_p={ap_beyond*1e3:.2f}mm")

# ===================================================== plot
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
C_OL, C_CL = '#1f4e79', '#c0392b'
apx = ap_list*1e3

# (a) depth sweep, both states
a = ax[0, 0]
a.plot(apx, data['intact']['olr'], 'o-', color=C_OL, lw=1.6, ms=3, label='OL intact')
a.plot(apx, data['intact']['clr'], 's-', color=C_CL, lw=1.6, ms=3, label='LQG intact')
a.plot(apx, data['machined']['olr'], 'o--', color='#5b8fbf', lw=1.4, ms=3, label='OL machined 1mm')
a.plot(apx, data['machined']['clr'], 's--', color='#e08a8a', lw=1.4, ms=3, label='LQG machined 1mm')
a.axvline(data['intact']['apc_ol']*1e3, color=C_OL, ls=':', lw=1.5)
a.axvline(data['intact']['apc_cl']*1e3, color=C_CL, ls=':', lw=1.5)
a.set_yscale('log')
a.set_title(f"(a) Depth-of-cut sweep at RPM={RPM}\n"
            "y_rms (gaps = chatter divergence)", fontweight='bold')
a.set_xlabel("axial depth of cut $a_p$ (mm)"); a.set_ylabel("$y_{rms}$ (µm, log)")
a.grid(alpha=0.4, which='both'); a.legend(fontsize=8, loc='lower right')

# (b) LQG voltage vs a_p
a = ax[0, 1]
a.plot(apx, data['intact']['clu'], '^-', color=C_CL, lw=1.6, ms=3, label='LQG intact')
a.plot(apx, data['machined']['clu'], '^--', color='#e08a8a', lw=1.4, ms=3, label='LQG machined')
a.axhline(U_SAT, color='k', ls='--', lw=1.5, label=f"saturation ±{U_SAT:.0f} V")
a.axvline(data['intact']['apc_cl']*1e3, color=C_CL, ls=':', lw=1.2,
          label=f"LQG limit ≈ {data['intact']['apc_cl']*1e3:.1f} mm")
a.set_title("(b) LQG peak voltage vs depth of cut → saturation = limit",
            fontweight='bold')
a.set_xlabel("axial depth of cut $a_p$ (mm)"); a.set_ylabel("$|u|_{max}$ (V)")
a.grid(alpha=0.4); a.legend(fontsize=8, loc='lower right')

# (c) rescue: OL chatters, LQG stable
a = ax[1, 0]
i1 = tr_ol_r['stop']; i2 = tr_cl_r['stop']
a.plot(tr_ol_r['t'][:i1+1]*1e3, tr_ol_r['y'][:i1+1]*1e6, color=C_OL, lw=0.6,
       label="open-loop (chatter, diverges @ %dms)" % (tr_ol_r['t'][i1]*1e3))
a.plot(tr_cl_r['t'][:i2+1]*1e3, tr_cl_r['y'][:i2+1]*1e6, color=C_CL, lw=0.6,
       label="LQG (stable, rms=%.3fµm, u_max=%.0fV)" % (tr_cl_r['yrms']*1e6, tr_cl_r['umax']))
a.set_title(f"(c) LQG rescue: $a_p$={ap_rescue*1e3:.2f} mm "
            f"(OL-crit<{ap_rescue*1e3:.2f}<LQG-crit)", fontweight='bold')
a.set_xlabel("time (ms)"); a.set_ylabel("$y_p$ (µm)")
a.grid(alpha=0.4); a.legend(fontsize=8, loc='upper left')

# (d) beyond LQG limit: both diverge
a = ax[1, 1]
i1 = tr_ol_b['stop']; i2 = tr_cl_b['stop']
a.plot(tr_ol_b['t'][:i1+1]*1e3, tr_ol_b['y'][:i1+1]*1e6, color=C_OL, lw=0.6,
       label="open-loop (diverges @ %dms)" % (tr_ol_b['t'][i1]*1e3))
lbl_cl = ("LQG (diverges @ %dms, u saturated)" % (tr_cl_b['t'][i2]*1e3)
          if tr_cl_b['diverged'] else "LQG (stable, u_max=%.0fV)" % tr_cl_b['umax'])
a.plot(tr_cl_b['t'][:i2+1]*1e3, tr_cl_b['y'][:i2+1]*1e6, color=C_CL, lw=0.6, label=lbl_cl)
a.set_title(f"(d) Beyond LQG limit: $a_p$={ap_beyond*1e3:.2f} mm "
            "(voltage saturates)", fontweight='bold')
a.set_xlabel("time (ms)"); a.set_ylabel("$y_p$ (µm)")
a.grid(alpha=0.4); a.legend(fontsize=8, loc='upper left')

fig.suptitle("Worst case — regenerative chatter under ±150 V: LQG extends the "
             f"stable depth {data['intact']['apc_ol']*1e3:.1f} → "
             f"{data['intact']['apc_cl']*1e3:.1f} mm "
             f"(×{data['intact']['apc_cl']/data['intact']['apc_ol']:.1f}) then "
             "saturates; material removal shifts the chatter limit",
             fontsize=12.5, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
png = os.path.join(OUT, "worst_scenario.png")
fig.savefig(png, dpi=150, bbox_inches='tight')
fig.savefig(png.replace(".png", ".pdf"), bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {png}")
