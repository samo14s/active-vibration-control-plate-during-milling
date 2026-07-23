"""
gen_freeend_lqg.py
=================
LQG control DURING the free-end machining pass (a_p = 1 mm).

As the cutter removes the top 1 mm along its path, the workpiece dynamics drift
(mode 1: 519 -> 533 Hz). A single LQG controller — designed once on the intact
(nominal) plate — is applied at successive machining stages (0 %, 50 %, 100 %),
with the plant's modal matrices taken from a fixed-basis reduced-order model of
the machined structure. The demo shows the controller suppresses the milling
vibration robustly across the evolving dynamics.

Output: figs_freeend_lqg/freeend_lqg.png (+ .pdf)
Run:  python gen_freeend_lqg.py
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

from inprocess_plate import InProcessPlate
from lqg_controller import LQGController
from milling_force import precompute_alpha_periodic

# ------------------------------------------------------------------ params
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT = 925e6; KN = 0.26; MU_C = 0.20
RPM = 4900
FT = 0.02e-3; AE = 0.1e-3
D31 = 175e-12; H_PA = 0.7e-3
E_PE = 63e9; NU_PE = 0.35
N1, N2 = 30, 80                 # ley = 1 mm so the removed a_p = 1 mm is one row
ZETA = np.array([0.0031, 0.0017, 0.0027])
DT = 5e-5
T_END = 0.30
A_P_REM = 1.0e-3                # free-end removal depth
AP_FORCE = 0.3e-3              # axial depth of cut for the milling force (chatter)
Z_TOOL = HP - 2e-3            # cut / sensor kept just below the removed free edge
F_TOOTH = NT * RPM / 60.0

OUT = os.path.join(_HERE, "figs_freeend_lqg")
os.makedirs(OUT, exist_ok=True)


def fresh():
    return InProcessPlate(LP, HP, BP, RHO, E_AL, NU, N1=N1, N2=N2, n_modes=3,
                          zeta_modes=list(ZETA), verbose=False)


def rms(x):
    return np.sqrt(np.mean(x**2))


# ---- nominal (intact) plate: fixed basis, projections, LQG -------------
p0 = fresh()
p0.precompute_Dp(zp_pos=Z_TOOL, n_pos=2001)
p0.set_observation(x_obs=LP, z_obs=Z_TOOL)
p0.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
V0 = p0.V.copy()
D_obs = p0.D_obs.copy(); H_Pe = p0.H_Pe_modal.copy()
omega0 = p0.omega_n.copy(); Cp = np.diag(2*ZETA*omega0)
kp_fixed = 1000                                  # stationary tool at X = 50 mm
Dp = p0.Dp_array[:, kp_fixed].copy()
DpTDp = p0.DpT_Dp_array[:, :, kp_fixed].copy()

lqg = LQGController(p0, dt=DT, verbose=False)
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg.discretize_observer()
print(f"nominal mode-1 = {omega0[0]/(2*np.pi):.1f} Hz, LQG designed.")

# ---- milling force (stationary tool) ------------------------------------
phi_st = np.pi - np.arccos(1 - AE/(DT_TOOL/2)); phi_ex = np.pi
Omega = 2*np.pi*RPM/60; RT = DT_TOOL/2; tau = 60/(NT*RPM)
k1 = KN*np.cos(ETA_H)
k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)
t_vec = np.arange(0, T_END + DT/2, DT); nstep = len(t_vec)
n_per = int(np.round(tau/DT))
a3, a4 = precompute_alpha_periodic(DT, n_per, nstep, Omega, NT, RT, ETA_H,
                                   phi_st, phi_ex, Z_TOOL - AP_FORCE, Z_TOOL,
                                   k1, k2, KT)
n_tau = int(round(tau/DT))


def simulate(Mr, Kr, controller):
    """Self-contained modal Newmark with regenerative delay; optional LQG."""
    n = 3
    q = np.zeros(n); qd = np.zeros(n); qdd = np.zeros(n)
    qm = np.zeros((n, nstep)); y = np.zeros(nstep); u = np.zeros(nstep)
    x_hat = np.zeros(2*n); u_prev = 0.0
    gNM, bNM = 0.5, 0.25
    stop = nstep - 1
    for k in range(1, nstep):
        y_meas = D_obs @ qm[:, k-1]
        if controller is not None:
            x_hat, uk = controller.step(x_hat, u_prev, y_meas)
        else:
            uk = 0.0
        u[k] = uk
        q_delay = qm[:, k-n_tau] if k-n_tau > 0 else np.zeros(n)
        K_eff = Kr + a4[k]*DpTDp
        F = FT*a3[k]*Dp + a4[k]*DpTDp @ q_delay + H_Pe*uk
        qd_pred = qd + (1-gNM)*DT*qdd
        q_pred = q + DT*qd + (0.5-bNM)*DT**2*qdd
        S = Mr + gNM*DT*Cp + bNM*DT**2*K_eff
        qdd = np.linalg.solve(S, F - Cp@qd_pred - K_eff@q_pred)
        qd = qd_pred + gNM*DT*qdd
        q = q_pred + bNM*DT**2*qdd
        qm[:, k] = q; y[k] = D_obs @ q; u_prev = uk
        if abs(y[k]) > 5e-3:
            stop = k; break
    return dict(t=t_vec, y=y, u=u, stop=stop,
                diverged=(stop < nstep - 1))


# ---- run across machining stages ----------------------------------------
stages = [(0.0, "0 % (intact)"), (0.5, "50 % machined"), (1.0, "100 % machined")]
results = []
print("\nstage           f1(Hz)   OL           LQG")
for frac, lbl in stages:
    p = fresh(); p.new_pass()
    p.machine_to(frac*LP, a_p=A_P_REM, a_e=BP, h_min_frac=1e-3); p.rebuild()
    f1 = p.instantaneous_modes(1)[0][0]
    Mr, Kr = p.reduced_on_basis(V0)
    ol = simulate(Mr, Kr, None)
    cl = simulate(Mr, Kr, lqg)
    results.append(dict(frac=frac, lbl=lbl, f1=f1, ol=ol, cl=cl, Mr=Mr, Kr=Kr))
    def tag(r):
        return (f"DIVERGED@{r['t'][r['stop']]*1e3:.0f}ms" if r['diverged']
                else f"y_rms={rms(r['y'][:r['stop']+1])*1e6:.3f}um")
    print(f"  {lbl:15s} {f1:6.1f}   {tag(ol):22s} {tag(cl)}  "
          f"u_max={np.max(np.abs(cl['u'][:cl['stop']+1])):.1f}V")


# ============================================================ plot
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
C_OL, C_CL = '#1f4e79', '#c0392b'

# (a) time response OL vs LQG at final stage
rf = results[-1]
a = ax[0, 0]
iOL = rf['ol']['stop']; iCL = rf['cl']['stop']
a.plot(rf['ol']['t'][:iOL+1]*1e3, rf['ol']['y'][:iOL+1]*1e6, color=C_OL, lw=0.7,
       label=("open-loop (chatter, diverges @ "
              f"{rf['ol']['t'][iOL]*1e3:.0f} ms)" if rf['ol']['diverged']
              else f"open-loop (y_rms={rms(rf['ol']['y'])*1e6:.2f} µm)"))
a.plot(rf['cl']['t'][:iCL+1]*1e3, rf['cl']['y'][:iCL+1]*1e6, color=C_CL, lw=0.7,
       label=f"LQG (y_rms={rms(rf['cl']['y'][:iCL+1])*1e6:.3f} µm)")
a.set_title(f"(a) Response at 100 % machined ($f_1$={rf['f1']:.0f} Hz)\n"
            "open-loop vs LQG", fontweight='bold')
a.set_xlabel("time (ms)"); a.set_ylabel("$y_p$ (µm)")
a.grid(alpha=0.4); a.legend(fontsize=9, loc='upper left')

# (b) LQG y(t) at the three stages
a = ax[0, 1]
sc = ['#1f4e79', '#e08a1e', '#c0392b']
for r, c in zip(results, sc):
    i = r['cl']['stop']
    a.plot(r['cl']['t'][:i+1]*1e3, r['cl']['y'][:i+1]*1e6, color=c, lw=0.7,
           label=f"{r['lbl']} ($f_1$={r['f1']:.0f} Hz, "
                 f"rms={rms(r['cl']['y'][:i+1])*1e6:.3f} µm)")
a.set_title("(b) LQG-controlled response across machining stages",
            fontweight='bold')
a.set_xlabel("time (ms)"); a.set_ylabel("$y_p$ (µm)")
a.grid(alpha=0.4); a.legend(fontsize=8, loc='upper right')

# (c) bar: OL vs LQG per stage (y_rms; OL divergence shown at cap)
a = ax[1, 0]
labels = [r['lbl'] for r in results]
xb = np.arange(len(results)); w = 0.35
ol_rms = [rms(r['ol']['y'][:r['ol']['stop']+1])*1e6 for r in results]
cl_rms = [rms(r['cl']['y'][:r['cl']['stop']+1])*1e6 for r in results]
b1 = a.bar(xb - w/2, ol_rms, w, color=C_OL, alpha=0.85, edgecolor='k',
           label='open-loop')
b2 = a.bar(xb + w/2, cl_rms, w, color=C_CL, alpha=0.85, edgecolor='k',
           label='LQG')
for x_, r, v in zip(xb - w/2, results, ol_rms):
    a.text(x_, v, "diverged" if r['ol']['diverged'] else f"{v:.2f}",
           ha='center', va='bottom', fontsize=8, rotation=90 if r['ol']['diverged'] else 0)
for x_, v in zip(xb + w/2, cl_rms):
    a.text(x_, v, f"{v:.3f}", ha='center', va='bottom', fontsize=8, color='darkred')
a.set_yscale('log')
a.set_xticks(xb); a.set_xticklabels([r['lbl'] for r in results], fontsize=9)
a.set_ylabel("$y_{rms}$ (µm, log)")
a.set_title("(c) Vibration: open-loop vs LQG at each stage", fontweight='bold')
a.legend(fontsize=9); a.grid(alpha=0.4, axis='y', which='both')

# (d) control voltage at final stage
a = ax[1, 1]
i = rf['cl']['stop']
a.plot(rf['cl']['t'][:i+1]*1e3, rf['cl']['u'][:i+1], color=C_CL, lw=0.7)
a.axhline(150, color='gray', ls='--', lw=1); a.axhline(-150, color='gray', ls='--', lw=1,
                                                       label='piezo sat. ±150 V')
a.set_title("(d) LQG control voltage $u(t)$ at 100 % machined", fontweight='bold')
a.set_xlabel("time (ms)"); a.set_ylabel("u (V)")
a.grid(alpha=0.4); a.legend(fontsize=9, loc='upper right')

fig.suptitle("LQG control during the free-end pass ($a_p$=1 mm): one nominal "
             "controller suppresses milling vibration as the dynamics drift "
             f"{results[0]['f1']:.0f} → {results[-1]['f1']:.0f} Hz",
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
png = os.path.join(OUT, "freeend_lqg.png")
fig.savefig(png, dpi=150, bbox_inches='tight')
fig.savefig(png.replace(".png", ".pdf"), bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {png}")
