"""
gen_response_figures.py
=======================
Temporal and frequency response of the Mindlin-plate milling model,
WITHOUT control (open-loop) vs WITH control (LQG).

Two operating points (article geometry, 4900 RPM):
  * Nominal a_p = 0.3 mm  -> open loop is UNSTABLE (chatter/divergence);
    LQG stabilises it. Shows chatter suppression in time + control effort.
  * Stable a_p = 0.1 mm   -> open loop is a bounded forced vibration; the
    clean time signal gives a meaningful FFT: LQG attenuates the resonant
    peak. Shows the frequency response.

Output: figs_response/response_OL_vs_LQG.png (+ .pdf)
Run from 05_main/ (or anywhere): python gen_response_figures.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("01_core", "02_controllers", "03_analysis"):
    sys.path.insert(0, os.path.join(_HERE, "..", sub))

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from newmark_solver import NewmarkSimulator

# ------------------------------------------------------------------ params
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
T_END = 0.5
F_TOOTH = NT * RPM / 60.0                # tooth-passing frequency = 245 Hz

OUT = os.path.join(_HERE, "figs_response")
os.makedirs(OUT, exist_ok=True)


def build_plate(ap):
    p = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2,
                   n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
    p.precompute_Dp(zp_pos=HP - ap/2, n_pos=2001)
    p.set_observation(x_obs=LP, z_obs=HP)
    p.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    return p


def run(ap, with_control):
    plate = build_plate(ap)
    phi_st = np.pi - np.arccos(1 - AE/(DT_TOOL/2)); phi_ex = np.pi
    Omega = 2*np.pi*RPM/60; RT = DT_TOOL/2; tau = 60/(NT*RPM)
    k1 = KN*np.cos(ETA_H)
    k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)
    sim = NewmarkSimulator(plate, dt=DT, T_end=T_END, ft=FT, tau=tau, verbose=False)
    v_feed = FT*NT*RPM/60
    kp = np.clip(np.round(np.minimum(v_feed*sim.t_vec, LP)/LP*2000).astype(int), 0, 2000)
    n_per = int(np.round(tau/DT))
    a3, a4 = precompute_alpha_periodic(DT, n_per, sim.nstep, Omega, NT, RT, ETA_H,
                                       phi_st, phi_ex, HP-ap, HP, k1, k2, KT)
    ctrl = None
    if with_control:
        ctrl = LQGController(plate, dt=DT, verbose=False)
        ctrl.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                              w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
        ctrl.discretize_observer()
    res = sim.simulate(a3, a4, kp, controller=ctrl, progress=False)
    res['t'] = sim.t_vec
    res['plate'] = plate
    return res


def fft_amp(y, dt, i_start_t=0.05):
    i0 = int(i_start_t/dt)
    y = y[i0:]
    if len(y) < 8:
        return np.array([0.0]), np.array([0.0])
    win = np.hanning(len(y))
    NFFT = 2**int(np.ceil(np.log2(len(y))))
    Y = np.fft.fft(y*win, NFFT) / len(y)
    f = (1/dt)/2 * np.linspace(0, 1, NFFT//2 + 1)
    return f, 2*np.abs(Y[:NFFT//2 + 1]) * 1e6      # microns


def rms(x):
    return np.sqrt(np.mean(x**2))


# =================================================================== run
print("Running open-loop / LQG at two operating points (Mindlin plate)...")
print(f"  tooth-passing frequency f_t = {F_TOOTH:.0f} Hz")

# nominal (chatter) point
ap_nom = 0.3e-3
ol_nom = run(ap_nom, with_control=False)
cl_nom = run(ap_nom, with_control=True)

# stable (forced) point
ap_stab = 0.1e-3
ol_stab = run(ap_stab, with_control=False)
cl_stab = run(ap_stab, with_control=True)

modes = ol_nom['plate'].freq_n

# metrics
def summarize(tag, r):
    i = r['stop_idx']
    y = r['y'][:i+1]
    print(f"  {tag:26s} t_end={r['t'][i]*1e3:6.1f}ms  "
          f"y_rms={rms(y)*1e6:8.3f}um  y_max={np.max(np.abs(y))*1e6:9.3f}um  "
          f"u_max={np.max(np.abs(r['u'][:i+1])):6.2f}V"
          + ("  [DIVERGED]" if r['diverged_at'] > 0 else ""))

print("\nResults:")
summarize("ap=0.3mm  OPEN-LOOP", ol_nom)
summarize("ap=0.3mm  LQG",       cl_nom)
summarize("ap=0.1mm  OPEN-LOOP", ol_stab)
summarize("ap=0.1mm  LQG",       cl_stab)

# =================================================================== plot
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.4})
fig, ax = plt.subplots(2, 2, figsize=(15, 9))

C_OL, C_CL = '#1f4e79', '#c0392b'      # open-loop blue, controlled red

# ---- (0,0) temporal, nominal ap=0.3mm : chatter suppression ----
a = ax[0, 0]
iOL = ol_nom['stop_idx']; iCL = cl_nom['stop_idx']
a.plot(ol_nom['t'][:iOL+1]*1e3, ol_nom['y'][:iOL+1]*1e6, color=C_OL, lw=0.7,
       label=f"open-loop (chatter, diverges @ {ol_nom['t'][iOL]*1e3:.0f} ms)")
a.plot(cl_nom['t'][:iCL+1]*1e3, cl_nom['y'][:iCL+1]*1e6, color=C_CL, lw=0.7,
       label=f"LQG (stable, y_rms={rms(cl_nom['y'][:iCL+1])*1e6:.3f} µm)")
a.set_title(f"(a) Temporal response — nominal $a_p$=0.3 mm (article point)",
            fontweight='bold')
a.set_xlabel("time (ms)"); a.set_ylabel("$y_p$ (µm)")
a.legend(loc='upper left', fontsize=9)

# ---- (1,0) control voltage, nominal ----
a = ax[1, 0]
a.plot(cl_nom['t'][:iCL+1]*1e3, cl_nom['u'][:iCL+1], color=C_CL, lw=0.7)
a.axhline(150, color='gray', ls='--', lw=1, alpha=0.7)
a.axhline(-150, color='gray', ls='--', lw=1, alpha=0.7, label='piezo sat. ±150 V')
a.set_title("(c) Control voltage $u(t)$ — LQG @ $a_p$=0.3 mm", fontweight='bold')
a.set_xlabel("time (ms)"); a.set_ylabel("u (V)")
a.legend(loc='upper right', fontsize=9)

# ---- (0,1) temporal, stable ap=0.1mm (zoom) : forced vibration ----
a = ax[0, 1]
t0, t1 = 0.20, 0.26
m = (ol_stab['t'] >= t0) & (ol_stab['t'] <= t1)
a.plot(ol_stab['t'][m]*1e3, ol_stab['y'][m]*1e6, color=C_OL, lw=1.1,
       label=f"open-loop (y_rms={rms(ol_stab['y'])*1e6:.3f} µm)")
a.plot(cl_stab['t'][m]*1e3, cl_stab['y'][m]*1e6, color=C_CL, lw=1.1,
       label=f"LQG (y_rms={rms(cl_stab['y'])*1e6:.3f} µm)")
a.set_title("(b) Temporal response — stable $a_p$=0.1 mm (zoom)", fontweight='bold')
a.set_xlabel("time (ms)"); a.set_ylabel("$y_p$ (µm)")
a.legend(loc='upper right', fontsize=9)

# ---- (1,1) FFT, stable ap=0.1mm : frequency response ----
a = ax[1, 1]
fO, YO = fft_amp(ol_stab['y'], DT)
fC, YC = fft_amp(cl_stab['y'], DT)
a.plot(fO, YO, color=C_OL, lw=1.3, label='open-loop')
a.plot(fC, YC, color=C_CL, lw=1.3, label='LQG')
for h in range(1, 6):                       # tooth-passing harmonics
    fh = h*F_TOOTH
    if fh <= 1400:
        a.axvline(fh, color='gray', ls=':', lw=0.8, alpha=0.6)
        a.text(fh, a.get_ylim()[1]*0.92, f"{h}·$f_t$", rotation=90,
               fontsize=7, color='gray', ha='right', va='top')
for fc in modes:                            # plate modes
    if fc <= 1400:
        a.axvline(fc, color='green', ls='--', lw=0.9, alpha=0.5)
a.text(modes[0], 0, "  mode 1", color='green', fontsize=8, rotation=90,
       va='bottom', ha='left')
a.set_xlim(0, 1400)
a.set_title("(d) Frequency response FFT[$y_p$] — stable $a_p$=0.1 mm",
            fontweight='bold')
a.set_xlabel("frequency (Hz)"); a.set_ylabel("amplitude (µm)")
a.legend(loc='upper right', fontsize=9)

peak_red = (1 - YC.max()/YO.max()) * 100
fig.suptitle("Mindlin plate — milling vibration response: WITHOUT control (open-loop) "
             "vs WITH control (LQG)\n"
             f"4900 RPM, $f_t$={F_TOOTH:.0f} Hz, mode 1={modes[0]:.0f} Hz   |   "
             f"peak FFT reduction @0.1 mm: {peak_red:.0f}%",
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
png = os.path.join(OUT, "response_OL_vs_LQG.png")
fig.savefig(png, dpi=150, bbox_inches='tight')
fig.savefig(png.replace(".png", ".pdf"), bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {png}")
print(f"       {png.replace('.png', '.pdf')}")
print(f"Peak FFT reduction (stable point): {peak_red:.1f}%")
