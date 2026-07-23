"""
gen_freeend_resonance.py
=======================
Controller efficiency when the machining-induced frequency drift approaches the
tool cutting (tooth-passing) frequency.

The spindle speed is set so the 2nd tooth-passing harmonic 2*f_t = 530 Hz sits
inside the drift range of mode 1. As the free-end pass removes material (trim
depth 0 -> 2 mm), mode 1 drifts 519 -> 546 Hz and CROSSES 530 Hz — a resonance
between the natural mode and the periodic cutting force. Open-loop vibration
spikes at the crossing; a single nominal LQG controller keeps it suppressed,
and its efficiency (reduction ratio) is largest exactly where the drift meets
the cutting frequency.

Output: figs_resonance/freeend_resonance.png (+ .pdf)
Run:  python gen_freeend_resonance.py
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
N1, N2 = 30, 80
ZETA = np.array([0.0031, 0.0017, 0.0027])
DT = 5e-5
T_END = 0.35
RPM = 5300                      # -> 2*f_t = 530 Hz sits in the drift band
F_T = NT * RPM / 60.0           # tooth-passing frequency = 265 Hz
F_CUT = 2 * F_T                 # resonant cutting harmonic = 530 Hz
AP_FORCE = 0.08e-3             # light cut: clean forced resonance
Z_TOOL = HP - 5e-3            # excitation / sensor, safely below the trimmed band
X_TOOL = LP / 2

OUT = os.path.join(_HERE, "figs_resonance")
os.makedirs(OUT, exist_ok=True)


def fresh():
    return InProcessPlate(LP, HP, BP, RHO, E_AL, NU, N1=N1, N2=N2, n_modes=3,
                          zeta_modes=list(ZETA), verbose=False)


def rms(x):
    return np.sqrt(np.mean(x**2))


def projections(p):
    """D_obs, Dp at the fixed tool/sensor points, on the plate's current modes."""
    DOFf = p.DOFf
    No = shape_at_point_M(LP, Z_TOOL, p.N1, p.N2, p.lex, p.ley, p.node_id, p.ndof)[DOFf]
    Nf = shape_at_point_M(X_TOOL, Z_TOOL, p.N1, p.N2, p.lex, p.ley, p.node_id, p.ndof)[DOFf]
    return No @ p.V, Nf @ p.V


# ---- nominal plate + LQG (designed once) --------------------------------
p0 = fresh()
p0.set_observation(LP, Z_TOOL)
p0.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
lqg = LQGController(p0, dt=DT, verbose=False)
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg.discretize_observer()
print(f"RPM={RPM}: f_t={F_T:.0f} Hz, resonant harmonic 2*f_t={F_CUT:.0f} Hz")
print(f"nominal mode-1 = {p0.freq_n[0]:.1f} Hz; LQG designed.\n")

# ---- milling force (stationary tool at X_TOOL) --------------------------
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


def simulate(Mp, Kp, Cp, D_obs, H_Pe, Dp, controller):
    n = 3
    DpTDp = np.outer(Dp, Dp)
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
        K_eff = Kp + a4[k]*DpTDp
        F = FT*a3[k]*Dp + a4[k]*DpTDp @ q_delay + H_Pe*uk
        qd_pred = qd + (1-gNM)*DT*qdd
        q_pred = q + DT*qd + (0.5-bNM)*DT**2*qdd
        S = Mp + gNM*DT*Cp + bNM*DT**2*K_eff
        qdd = np.linalg.solve(S, F - Cp@qd_pred - K_eff@q_pred)
        qd = qd_pred + gNM*DT*qdd
        q = q_pred + bNM*DT**2*qdd
        qm[:, k] = q; y[k] = D_obs @ q; u_prev = uk
        if abs(y[k]) > 5e-3:
            stop = k; break
    return dict(t=t_vec, y=y, u=u, stop=stop, diverged=(stop < nstep-1),
                yrms=rms(y[:stop+1]), umax=np.max(np.abs(u[:stop+1])))


# ---- sweep trim depth so mode 1 drifts across the cutting harmonic -------
depths = np.linspace(0, 2.0e-3, 17)
f1s, ol_r, cl_r, u_r, ol_div = [], [], [], [], []
res_cross = None
print("trim(mm)  f1(Hz)   OL_rms(um)    LQG_rms(um)  u_max(V)")
for d in depths:
    p = fresh(); p.new_pass()
    p.machine_to(LP, a_p=d, a_e=BP, h_min_frac=1e-3); p.rebuild(); p._modal_analysis()
    p.set_observation(LP, Z_TOOL); p.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    D_obs, Dp = projections(p)
    p.D_obs = D_obs                       # keep consistent projection
    H_Pe = p.H_Pe_modal
    Mp, Kp, Cp = p.Mp, p.Kp, p.Cp
    f1 = p.freq_n[0]
    ol = simulate(Mp, Kp, Cp, D_obs, H_Pe, Dp, None)
    cl = simulate(Mp, Kp, Cp, D_obs, H_Pe, Dp, lqg)
    f1s.append(f1); ol_r.append(ol['yrms']*1e6); cl_r.append(cl['yrms']*1e6)
    u_r.append(cl['umax']); ol_div.append(ol['diverged'])
    if res_cross is None and f1 >= F_CUT:
        res_cross = dict(d=d, f1=f1, ol=ol, cl=cl)
    print(f"  {d*1e3:.2f}    {f1:6.1f}   "
          f"{('DIV@%dms'%(ol['t'][ol['stop']]*1e3)) if ol['diverged'] else '%.3f'%(ol['yrms']*1e6):>11s}  "
          f"{cl['yrms']*1e6:10.4f}  {cl['umax']:7.2f}")

f1s = np.array(f1s); ol_r = np.array(ol_r); cl_r = np.array(cl_r); u_r = np.array(u_r)
if res_cross is None:
    res_cross = dict(d=depths[-1], f1=f1s[-1],
                     ol=simulate(Mp, Kp, Cp, D_obs, H_Pe, Dp, None),
                     cl=simulate(Mp, Kp, Cp, D_obs, H_Pe, Dp, lqg))
eff = (1 - cl_r/ol_r) * 100

# ============================================================ plot
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
C_OL, C_CL = '#1f4e79', '#c0392b'

# (a) vibration vs mode-1 frequency (drift), cutting harmonic marked
a = ax[0, 0]
a.plot(f1s, ol_r, 'o-', color=C_OL, lw=1.8, ms=4, label='open-loop')
a.plot(f1s, cl_r, 's-', color=C_CL, lw=1.8, ms=4, label='LQG')
a.axvline(F_CUT, color='k', ls='--', lw=1.5)
a.text(F_CUT+0.5, a.get_ylim()[1]*0.6, f"cutting harmonic\n$2f_t$={F_CUT:.0f} Hz",
       fontsize=9, color='k')
div_mask = np.array(ol_div)
if div_mask.any():
    a.plot(f1s[div_mask], ol_r[div_mask], 'x', color='k', ms=10,
           label='open-loop diverged')
a.set_yscale('log')
a.set_title("(a) Vibration vs machining-drifted mode-1 frequency", fontweight='bold')
a.set_xlabel("mode-1 frequency $f_1$ (Hz)  →  (deeper cut)")
a.set_ylabel("$y_{rms}$ (µm, log)")
a.grid(alpha=0.4, which='both'); a.legend(fontsize=9, loc='upper right')

# (b) controller efficiency vs frequency proximity
a = ax[0, 1]
a.plot(f1s, eff, 'D-', color='#2e8b57', lw=1.8, ms=4)
a.axvline(F_CUT, color='k', ls='--', lw=1.5)
a.set_ylim(min(90, eff.min()-2), 100.3)
a.set_title("(b) Controller efficiency $(1-y_{LQG}/y_{OL})$", fontweight='bold')
a.set_xlabel("mode-1 frequency $f_1$ (Hz)")
a.set_ylabel("vibration reduction (%)")
a.grid(alpha=0.4)
a.text(F_CUT+0.5, eff.min(), "resonance\ncrossing", fontsize=9)

# (c) time response at the crossing
rc = res_cross
a = ax[1, 0]
iOL = rc['ol']['stop']; iCL = rc['cl']['stop']
a.plot(rc['ol']['t'][:iOL+1]*1e3, rc['ol']['y'][:iOL+1]*1e6, color=C_OL, lw=0.7,
       label=("open-loop (diverges @ %dms)" % (rc['ol']['t'][iOL]*1e3)
              if rc['ol']['diverged'] else
              "open-loop (rms=%.2f µm)" % (rc['ol']['yrms']*1e6)))
a.plot(rc['cl']['t'][:iCL+1]*1e3, rc['cl']['y'][:iCL+1]*1e6, color=C_CL, lw=0.7,
       label="LQG (rms=%.3f µm)" % (rc['cl']['yrms']*1e6))
a.set_title(f"(c) Response at the crossing $f_1$={rc['f1']:.0f} Hz ≈ $2f_t$={F_CUT:.0f} Hz",
            fontweight='bold')
a.set_xlabel("time (ms)"); a.set_ylabel("$y_p$ (µm)")
a.grid(alpha=0.4); a.legend(fontsize=9, loc='upper left')

# (d) control voltage vs frequency (effort near resonance)
a = ax[1, 1]
a.plot(f1s, u_r, '^-', color=C_CL, lw=1.8, ms=4)
a.axvline(F_CUT, color='k', ls='--', lw=1.5)
a.axhline(150, color='gray', ls=':', lw=1, label='piezo sat. ±150 V')
a.set_title("(d) LQG peak control voltage vs frequency", fontweight='bold')
a.set_xlabel("mode-1 frequency $f_1$ (Hz)"); a.set_ylabel("$|u|_{max}$ (V)")
a.grid(alpha=0.4); a.legend(fontsize=9, loc='upper left')

fig.suptitle("Controller efficiency when the machining frequency drift meets the "
             f"cutting frequency\nRPM={RPM}, $2f_t$={F_CUT:.0f} Hz; free-end removal "
             f"drifts $f_1$ {f1s[0]:.0f} → {f1s[-1]:.0f} Hz through resonance",
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
png = os.path.join(OUT, "freeend_resonance.png")
fig.savefig(png, dpi=150, bbox_inches='tight')
fig.savefig(png.replace(".png", ".pdf"), bbox_inches='tight')
plt.close(fig)
print(f"\nAt crossing f1={rc['f1']:.0f} Hz: OL "
      f"{'diverged' if rc['ol']['diverged'] else '%.2f um'%(rc['ol']['yrms']*1e6)}, "
      f"LQG {rc['cl']['yrms']*1e6:.3f} um")
print(f"Saved: {png}")
