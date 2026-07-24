"""
gen_controlled_vibration.py
===========================
Milling vibration WITHOUT control vs WITH control (twin-calibrated RC-SAC).

This is the closed-loop pay-off of the physics-guided digital twin. Under a
realistic +9 % modal-frequency error, a single FEM-guided probe keeps the
RC-SAC correctly tuned, and the controller both (i) ENABLES a deep cut that is
catastrophically unstable open-loop, and (ii) strongly REDUCES the forced /
chatter vibration at a stable cut — all inside the ±150 V piezo limit.

Two depths are shown, exactly as in the LQG reference figure, so every
comparison is fair:
  * a DEEP cut (a_p = 0.6 mm) where the open loop diverges  -> stability story
  * a STABLE cut (a_p = 0.1 mm) where the open loop is bounded -> reduction story

Panels:
  (a) Temporal response y_p(t) at the DEEP cut: no control diverges; control stable.
  (b) Temporal zoom at the STABLE cut: no control vs control (amplitude reduction).
  (c) Control voltage u(t) at the deep cut — inside the ±150 V saturation band.
  (d) Frequency response FFT[y_p] at the STABLE cut: mode-1 peak suppressed (fair %).

Output: docs/controlled_vibration.png (+ .pdf)
Run:  python gen_controlled_vibration.py     (~1 min)
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
DT = 5e-5; T_END = 0.35
RPM = 4900; U_SAT = 150.0
Z_TOOL = HP - 5e-3; X_TOOL = LP/2

# scenario: realistic in-process error; two depths for a fair comparison
BETA = 0.09          # +9 % true modal-frequency error (E / BC uncertainty + drift)
AP_DEEP = 0.6e-3     # deep cut: open-loop DIVERGES, twin-controlled stable
AP_LOW = 0.1e-3      # stable cut: open-loop BOUNDED -> fair reduction %

OUT = os.path.join(_HERE, "..", "docs")
os.makedirs(OUT, exist_ok=True)

PATCH = dict(xP1=0, xP2=0.020, zP1=0, zP2=0.060,
             d31=D31, h_Pa=H_PA, E_Pe=E_PE, nu_Pe=NU_PE)

twin = MillingDigitalTwin(LP, HP, BP, RHO, E_AL, NU,
                          X_TOOL, Z_TOOL, LP, Z_TOOL, PATCH,
                          N1=30, N2=24, n_modes=3, zeta_modes=ZETA)


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


def true_plant(removal, beta):
    """True-plant modal data: FEM at `removal`, frequencies scaled by (1+beta)."""
    p = twin.predict(removal)
    om = p.omega_fem * (1 + beta)
    return (p.Dp_tool.copy(), p.D_obs.copy(), p.H_Pe_modal.copy(),
            np.diag(om**2), np.diag(2*np.array(ZETA)*om))


def probe_true(removal, beta):
    """Apply the FEM-guided probe to the true plant; return (u, y)."""
    u = twin.design_probe(DT, T_probe=0.10, amp=20.0)
    Dp, Do, Hpz, Kp, Cp = true_plant(removal, beta)
    ns = len(u); q = np.zeros(3); qd = np.zeros(3); qdd = np.zeros(3); y = np.zeros(ns)
    for k in range(1, ns):
        S = np.eye(3) + 0.5*DT*Cp + 0.25*DT**2*Kp
        qd_p = qd + 0.5*DT*qdd; q_p = q + DT*qd + 0.25*DT**2*qdd
        qdd = np.linalg.solve(S, Hpz*u[k] - Cp@qd_p - Kp@q_p)
        qd = qd_p + 0.5*DT*qdd; q = q_p + 0.25*DT**2*qdd
        y[k] = Do @ q
    rng = np.random.default_rng(1)
    return u, y + rng.normal(0, 3e-9, ns)


def simulate(removal, beta, ap, controller=None):
    """Closed loop; controller=None -> open-loop (no control, u=0)."""
    a3, a4, n_tau, ns = milling(ap)
    Dp, Do, Hpz, Kp, Cp = true_plant(removal, beta)
    DpT = np.outer(Dp, Dp)
    q = np.zeros(3); qd = np.zeros(3); qdd = np.zeros(3)
    qm = np.zeros((3, ns)); y = np.zeros(ns); u = np.zeros(ns)
    x_hat = np.zeros(6); u_prev = 0.0
    if controller is not None:
        controller.reset()
    for k in range(1, ns):
        if controller is None:
            uk = 0.0
        else:
            x_hat, uk = controller.step(x_hat, u_prev, Do @ qm[:, k-1])
            uk = float(np.clip(uk, -U_SAT, U_SAT))
        u[k] = uk
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
                        yrms=np.sqrt(np.mean(y[:k+1]**2)))
    return dict(t=np.arange(ns)*DT, y=y, u=u, stop=ns-1, diverged=False,
                yrms=np.sqrt(np.mean(y**2)))


def rcsac_on(plate, ap):
    a3, a4, n_tau, _ = milling(ap)
    return RCSACController(plate, DT, plate.Dp_tool, a3, a4, n_tau,
                           u_sat=U_SAT, ft=FT, verbose=False)


def fft_amp(y, t, t0, t1):
    """Single-sided FFT amplitude of y over the window [t0, t1]."""
    m = (t >= t0) & (t <= t1)
    yy = y[m]
    if len(yy) < 16:
        return np.array([0.0]), np.array([0.0])
    w = np.hanning(len(yy))
    Y = np.fft.rfft(yy * w)
    f = np.fft.rfftfreq(len(yy), DT)
    amp = 2.0 * np.abs(Y) / np.sum(w)
    return f, amp


# ===================================================================== run
print(f"scenario: beta={BETA*100:+.0f}%   T={T_END*1e3:.0f} ms")

# ONE FEM-guided probe / calibration, reused for both depths
u_pr, y_pr = probe_true(0.0, BETA)
theta, f_true = twin.calibrate(u_pr, y_pr, DT)
p_twin = twin.corrected_model(0.0)
f1 = twin.f_fem_intact[0] * (1 + BETA)      # true mode-1 freq
print(f"probe: theta={theta:.4f}  f_true={f_true:.1f} Hz  (FEM {twin.f_fem_intact[0]:.1f} Hz)")

print(f"deep cut  a_p={AP_DEEP*1e3:.1f} mm ...")
ol_deep = simulate(0.0, BETA, AP_DEEP, controller=None)
cl_deep = simulate(0.0, BETA, AP_DEEP, rcsac_on(p_twin, AP_DEEP))
t_div = ol_deep['t'][ol_deep['stop']] if ol_deep['diverged'] else T_END
print(f"    open-loop {'DIVERGES @ %.0f ms' % (t_div*1e3) if ol_deep['diverged'] else 'bounded'}"
      f"   controlled rms={cl_deep['yrms']*1e6:.3f} um   max|u|={np.max(np.abs(cl_deep['u'])):.0f} V")

print(f"stable cut a_p={AP_LOW*1e3:.1f} mm ...")
ol_low = simulate(0.0, BETA, AP_LOW, controller=None)
cl_low = simulate(0.0, BETA, AP_LOW, rcsac_on(p_twin, AP_LOW))

# fair FFT over the transient-free window (both bounded at the stable cut)
t = ol_low['t']
f_ol, A_ol = fft_amp(ol_low['y'], t, 0.05, T_END)
f_cl, A_cl = fft_amp(cl_low['y'], t, 0.05, T_END)
band = (f_ol > 0.7*f1) & (f_ol < 1.3*f1)
peak_ol = A_ol[band].max() if A_ol.size else 0.0
peak_cl = A_cl[(f_cl > 0.7*f1) & (f_cl < 1.3*f1)].max() if A_cl.size else 0.0
red_pk = 100*(1 - peak_cl/peak_ol) if peak_ol > 0 else 0.0
red_rms = 100*(1 - cl_low['yrms']/ol_low['yrms']) if ol_low['yrms'] > 0 else 0.0
print(f"    open-loop {'bounded' if not ol_low['diverged'] else 'DIVERGES'}"
      f"   rms {ol_low['yrms']*1e6:.3f}->{cl_low['yrms']*1e6:.3f} um "
      f"({red_rms:.0f}%)   mode-1 peak {red_pk:.0f}%")

# ===================================================================== plot
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
C_OL, C_CL = '#1f4e79', '#c0392b'

# (a) deep cut: open-loop diverges, control stable
a = ax[0, 0]
i_ol = ol_deep['stop']
a.plot(t[:i_ol+1]*1e3, ol_deep['y'][:i_ol+1]*1e6, color=C_OL, lw=0.6,
       label=("open-loop — no control (diverges @ %.0f ms)" % (t_div*1e3))
       if ol_deep['diverged'] else "open-loop — no control")
a.plot(cl_deep['t']*1e3, cl_deep['y']*1e6, color=C_CL, lw=0.7,
       label=f"twin-calibrated RC-SAC (stable, rms={cl_deep['yrms']*1e6:.3f} µm)")
a.set_title(f"(a) Deep cut $a_p$={AP_DEEP*1e3:.1f} mm — control ENABLES stability\n"
            f"β=+{BETA*100:.0f}% model error", fontweight='bold')
a.set_xlabel("time (ms)"); a.set_ylabel("$y_p$ (µm)")
a.grid(alpha=0.4); a.legend(fontsize=9, loc='upper left')

# (b) stable cut: amplitude reduction (zoom)
a = ax[0, 1]
zt0, zt1 = 0.20, 0.26
m = (t >= zt0) & (t <= zt1)
a.plot(t[m]*1e3, ol_low['y'][m]*1e6, color=C_OL, lw=0.9,
       label=f"no control (rms={ol_low['yrms']*1e6:.3f} µm)")
a.plot(t[m]*1e3, cl_low['y'][m]*1e6, color=C_CL, lw=1.0,
       label=f"RC-SAC (rms={cl_low['yrms']*1e6:.3f} µm)")
a.axhline(0, color='k', lw=0.5, alpha=0.4)
a.set_title(f"(b) Stable cut $a_p$={AP_LOW*1e3:.1f} mm — control REDUCES vibration\n"
            f"rms reduction {red_rms:.0f}% (zoom)", fontweight='bold')
a.set_xlabel("time (ms)"); a.set_ylabel("$y_p$ (µm)")
a.grid(alpha=0.4); a.legend(fontsize=9, loc='upper right')

# (c) control voltage (deep cut, demanding case)
a = ax[1, 0]
a.plot(cl_deep['t']*1e3, cl_deep['u'], color=C_CL, lw=0.6, label='RC-SAC $u(t)$ (deep cut)')
a.axhline(U_SAT, ls='--', color='k', lw=1, alpha=0.6, label=f'piezo sat. ±{U_SAT:.0f} V')
a.axhline(-U_SAT, ls='--', color='k', lw=1, alpha=0.6)
a.set_title(f"(c) Control voltage $u(t)$ — max |u|={np.max(np.abs(cl_deep['u'])):.0f} V "
            f"(within limit)", fontweight='bold')
a.set_xlabel("time (ms)"); a.set_ylabel("u (V)")
a.set_ylim(-U_SAT*1.15, U_SAT*1.15)
a.grid(alpha=0.4); a.legend(fontsize=9, loc='upper right')

# (d) FFT at the stable cut (fair reduction)
a = ax[1, 1]
a.plot(f_ol, A_ol*1e6, color=C_OL, lw=1.0, label='no control')
a.plot(f_cl, A_cl*1e6, color=C_CL, lw=1.0, label='twin-calibrated RC-SAC')
a.axvline(f1, ls='--', color='green', lw=1, alpha=0.6)
a.text(f1, peak_ol*1e6*0.95, ' mode 1', color='green', fontsize=8, rotation=90, va='top')
a.set_title(f"(d) FFT[$y_p$] at stable cut — mode-1 peak −{red_pk:.0f}%",
            fontweight='bold')
a.set_xlabel("frequency (Hz)"); a.set_ylabel("amplitude (µm)")
a.set_xlim(0, 1400)
a.grid(alpha=0.4); a.legend(fontsize=9, loc='upper right')

fig.suptitle("Vibration under control — twin-calibrated RC-SAC (β=+9% model error)\n"
             f"deep cut: control turns divergence into a stable {cl_deep['yrms']*1e6:.2f} µm signal; "
             f"stable cut: −{red_rms:.0f}% rms — all within ±{U_SAT:.0f} V",
             fontsize=12.5, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.92])
png = os.path.join(OUT, "controlled_vibration.png")
fig.savefig(png, dpi=150, bbox_inches='tight')
fig.savefig(png.replace(".png", ".pdf"), bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {png}")
