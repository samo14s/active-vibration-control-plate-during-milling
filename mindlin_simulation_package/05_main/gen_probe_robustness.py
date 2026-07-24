"""
gen_probe_robustness.py
=======================
STRESS TEST OF THE PROBE ITSELF — the twin's single point of failure.

The whole digital-twin advantage rests on ONE short FEM-guided probe: if the
receptance calibration cannot recover the true mode-1 frequency under realistic
sensor noise, a short measurement window, or weak excitation, the twin silently
degrades to a (wrong) fixed model. This script pushes the three probe-design
knobs to the breaking point at a realistic +9 % model error and finds each
threshold — then shows the CLOSED-LOOP consequence.

Physics that sets the limits: mode 1 (~566 Hz, zeta~0.003) has a settling time
1/(zeta*omega) ~ 93 ms, so the mode needs ~90 ms to ring up. A probe that is
too short (fast chirp) never excites it fully; a probe that is too weak, or a
sensor too noisy, buries the receptance peak.

Panels:
  (a) Noise robustness:     calibration error vs sensor noise sigma (log).
  (b) Duration robustness:  calibration error vs probe length T_probe.
  (c) Excitation robustness: calibration error vs probe amplitude (SNR).
  (d) Closed-loop consequence: stable RMS at a deep cut (a_p=3.5 mm) vs sigma —
      once the probe error grows, the deep-cut control destabilises.

Each 1-D point is the mean over many noise seeds; the shaded band is the
worst case over seeds (what a single real probe might hit).

Output: docs/probe_robustness.png (+ .pdf)
Run:  python gen_probe_robustness.py     (~2 min)
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
DT = 5e-5; U_SAT = 150.0
RPM = 4900
Z_TOOL = HP - 5e-3; X_TOOL = LP/2

BETA = 0.09                    # realistic in-process model error
N_SEED = 20                    # noise realisations per point
SIG_NOM = 3e-9                 # nominal sensor noise (3 nm)
T_NOM = 0.10                   # nominal probe length (100 ms)
AMP_NOM = 20.0                 # nominal probe amplitude (V)
AP_DEEP = 3.5e-3               # deep cut for the closed-loop consequence

OUT = os.path.join(_HERE, "..", "docs")
os.makedirs(OUT, exist_ok=True)
PATCH = dict(xP1=0, xP2=0.020, zP1=0, zP2=0.060,
             d31=D31, h_Pa=H_PA, E_Pe=E_PE, nu_Pe=NU_PE)

twin = MillingDigitalTwin(LP, HP, BP, RHO, E_AL, NU,
                          X_TOOL, Z_TOOL, LP, Z_TOOL, PATCH,
                          N1=30, N2=24, n_modes=3, zeta_modes=ZETA)
F_FEM = twin.f_fem_intact[0]
F_TRUE = F_FEM * (1 + BETA)
PROBE_BAND = (0.80, 1.22)


def true_plant():
    p = twin.predict(0.0)
    om = p.omega_fem * (1 + BETA)
    return (p.Dp_tool.copy(), p.D_obs.copy(), p.H_Pe_modal.copy(),
            np.diag(om**2), np.diag(2*np.array(ZETA)*om))

DP, DO, HPZ, KP, CP = true_plant()


def probe_response(T_probe, amp):
    """Noise-free probe response y0(t) and the input u(t)."""
    u = twin.design_probe(DT, T_probe=T_probe, amp=amp, band=PROBE_BAND)
    ns = len(u); q = np.zeros(3); qd = np.zeros(3); qdd = np.zeros(3); y = np.zeros(ns)
    for k in range(1, ns):
        S = np.eye(3) + 0.5*DT*CP + 0.25*DT**2*KP
        qd_p = qd + 0.5*DT*qdd; q_p = q + DT*qd + 0.25*DT**2*qdd
        qdd = np.linalg.solve(S, HPZ*u[k] - CP@qd_p - KP@q_p)
        qd = qd_p + 0.5*DT*qdd; q = q_p + 0.25*DT**2*qdd
        y[k] = DO @ q
    return u, y


def calib_err(T_probe, amp, sigma, seed):
    """Calibration error (%) for one noisy probe realisation."""
    u, y0 = probe_response(T_probe, amp)
    y = y0 + np.random.default_rng(seed).normal(0, sigma, len(y0))
    th, fe = twin.calibrate(u, y, DT)
    return abs(fe - F_TRUE) / F_TRUE * 100, th


def sweep(values, fn):
    """Return mean and worst-case (max) error over seeds for each value."""
    mean, worst = [], []
    for v in values:
        errs = [fn(v, s)[0] for s in range(N_SEED)]
        mean.append(np.mean(errs)); worst.append(np.max(errs))
    return np.array(mean), np.array(worst)


def threshold(values, worst, tol=2.0):
    """Largest/most-demanding value keeping worst-case error < tol (or None)."""
    ok = worst < tol
    return None if not ok.any() else values[ok]


# ---------- milling / closed-loop for the consequence panel ----------
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


def simulate_cl(theta, ap, t_end=0.22):
    plate = twin.corrected_model(0.0, theta=theta)
    a3, a4, n_tau, ns = milling(ap, t_end)
    ctrl = RCSACController(plate, DT, plate.Dp_tool, a3, a4, n_tau,
                           u_sat=U_SAT, ft=FT, verbose=False)
    DpT = np.outer(DP, DP)
    q = np.zeros(3); qd = np.zeros(3); qdd = np.zeros(3)
    qm = np.zeros((3, ns)); y = np.zeros(ns); x_hat = np.zeros(6); u_prev = 0.0
    ctrl.reset()
    for k in range(1, ns):
        x_hat, uk = ctrl.step(x_hat, u_prev, DO @ qm[:, k-1])
        uk = float(np.clip(uk, -U_SAT, U_SAT))
        q_del = qm[:, k-n_tau] if k-n_tau > 0 else np.zeros(3)
        K_eff = KP + a4[k]*DpT
        F = FT*a3[k]*DP + a4[k]*DpT @ q_del + HPZ*uk
        qd_p = qd + 0.5*DT*qdd; q_p = q + DT*qd + 0.25*DT**2*qdd
        S = np.eye(3) + 0.5*DT*CP + 0.25*DT**2*K_eff
        qdd = np.linalg.solve(S, F - CP@qd_p - K_eff@q_p)
        qd = qd_p + 0.5*DT*qdd; q = q_p + 0.25*DT**2*qdd
        qm[:, k] = q; y[k] = DO @ q; u_prev = uk
        if abs(y[k]) > 5e-3:
            return True, np.sqrt(np.mean(y[:k+1]**2))
    return False, np.sqrt(np.mean(y**2))


# =================================================================== sweeps
print(f"probe robustness at beta=+{BETA*100:.0f}%  (f_true={F_TRUE:.1f} Hz, "
      f"settling ~{1/(ZETA[0]*2*np.pi*F_TRUE)*1e3:.0f} ms)")

print("(a) noise ...")
SIGMAS = np.logspace(np.log10(3e-9), np.log10(1e-4), 16)   # 3 nm .. 100 µm
mA, wA = sweep(SIGMAS, lambda v, s: calib_err(T_NOM, AMP_NOM, v, s))
thA = threshold(SIGMAS, wA); sig_max = thA.max() if thA is not None else None

print("(b) duration ...")
TPROBES = np.array([0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15])
mB, wB = sweep(TPROBES, lambda v, s: calib_err(v, AMP_NOM, SIG_NOM, s))
thB = threshold(TPROBES, wB); t_min = thB.min() if thB is not None else None

print("(c) amplitude ...")
AMP_NOISE = 1e-7               # stress the SNR at 100 nm (10x a good sensor)
AMPS = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 150.0])
mC, wC = sweep(AMPS, lambda v, s: calib_err(T_NOM, v, AMP_NOISE, s))
thC = threshold(AMPS, wC); amp_min = thC.min() if thC is not None else None

print("(d) closed-loop consequence vs noise ...")
SIG_D = np.logspace(np.log10(3e-9), np.log10(1e-4), 14)
rms_d, div_d, err_d = [], [], []
for v in SIG_D:
    # representative single realisation (worst-ish seed) -> theta -> deep-cut control
    e_seeds = [calib_err(T_NOM, AMP_NOM, v, s) for s in range(N_SEED)]
    # pick the seed giving the median error as the 'typical' probe
    order = np.argsort([e for e, _ in e_seeds]); e_med, th_med = e_seeds[order[len(order)//2]]
    diverged, rms = simulate_cl(th_med, AP_DEEP)
    rms_d.append(rms); div_d.append(diverged); err_d.append(e_med)
    print(f"    sigma={v*1e9:6.0f} nm  calib_err={e_med:5.2f}%  "
          f"{'DIV' if diverged else f'{rms*1e6:.2f} um'}")
rms_d = np.array(rms_d); div_d = np.array(div_d); err_d = np.array(err_d)

# =========================================================== plot
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(2, 2, figsize=(15, 9.6))
C_M, C_W, C_TH = '#c0392b', '#e59866', '#1f4e79'


def draw(a, x, mean, worst, xlab, title, logx=True, thr=None, thr_lab=None,
         real_band=None):
    if logx:
        a.set_xscale('log')
    if real_band is not None:
        a.axvspan(real_band[0], real_band[1], color='green', alpha=0.12,
                  label='real sensor noise (1–10 nm)')
    a.fill_between(x, mean, worst, color=C_W, alpha=0.30, label='worst case (over seeds)')
    a.plot(x, mean, 'o-', color=C_M, lw=1.8, ms=5, label='mean error')
    a.axhline(2.0, ls='--', color='k', lw=1, alpha=0.6, label='2 % tolerance')
    if thr is not None:
        a.axvline(thr, ls=':', color=C_TH, lw=1.6, label=thr_lab)
    a.set_title(title, fontweight='bold')
    a.set_xlabel(xlab); a.set_ylabel("mode-1 recovery error (%)")
    a.grid(alpha=0.4, which='both'); a.legend(fontsize=8.5, loc='upper left')


draw(ax[0, 0], SIGMAS*1e9, mA, wA, "sensor noise σ (nm)",
     "(a) Noise robustness — receptance peak buried by noise",
     thr=(sig_max*1e9 if sig_max else None),
     thr_lab=(f"noise limit ≈{sig_max*1e9/1000:.1f} µm" if sig_max else None),
     real_band=(1, 10))

draw(ax[0, 1], TPROBES*1e3, mB, wB, "probe length $T_{probe}$ (ms)",
     "(b) Duration robustness — mode needs ~90 ms to ring up", logx=False,
     thr=(t_min*1e3 if t_min else None),
     thr_lab=(f"min length ≈{t_min*1e3:.0f} ms" if t_min else None))

draw(ax[1, 0], AMPS, mC, wC, "probe amplitude (V)",
     "(c) Excitation robustness — weak drive → poor SNR (at 100 nm noise)",
     logx=True, thr=amp_min,
     thr_lab=(f"min amplitude ≈{amp_min:g} V" if amp_min else None))

# (d) closed-loop consequence
a = ax[1, 1]
a.set_xscale('log')
a.axvspan(1, 10, color='green', alpha=0.12, label='real sensor noise (1–10 nm)')
ok = ~div_d
a.plot(SIG_D[ok]*1e9, rms_d[ok]*1e6, 'o-', color=C_M, lw=1.8, ms=6, label='stable RMS (deep cut)')
if div_d.any():
    yd = np.full(div_d.sum(), np.nanmax(rms_d[ok])*1e6*1.15 if ok.any() else 1.0)
    a.plot(SIG_D[div_d]*1e9, yd, 'x', color=C_TH, ms=11, mew=2.5, label='DIVERGED')
a.set_title(f"(d) Closed-loop consequence — deep cut $a_p$={AP_DEEP*1e3:.1f} mm\n"
            "a noisy probe → wrong θ → the deep cut destabilises", fontweight='bold')
a.set_xlabel("sensor noise σ (nm)"); a.set_ylabel("closed-loop $y_p$ RMS (µm)")
a.grid(alpha=0.4, which='both'); a.legend(fontsize=8.5, loc='upper left')

sig_txt = (f"σ≈{sig_max*1e9/1000:.1f} µm" if sig_max and sig_max >= 1e-6
           else (f"σ≈{sig_max*1e9:.0f} nm" if sig_max else "σ (n/a)"))
amp_txt = f"amp≥{amp_min:g} V" if amp_min else "amp (n/a)"
fig.suptitle("Probe robustness — the twin's single point of failure, stress-tested (β=+9%)\n"
             f"calibration holds <2 % up to {sig_txt}, T≥{t_min*1e3:.0f} ms, {amp_txt} — "
             "orders of magnitude beyond any real sensor; the probe is not the bottleneck",
             fontsize=12.5, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.92])
png = os.path.join(OUT, "probe_robustness.png")
fig.savefig(png, dpi=150, bbox_inches='tight')
fig.savefig(png.replace(".png", ".pdf"), bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {png}")
print(f"SUMMARY  noise≤{sig_max*1e9:.0f}nm  T≥{t_min*1e3:.0f}ms  amp≥{amp_min:g}V  (worst-case <2%)")
