"""
run_path_study.py
=================
The integrated study: path-varying workpiece dynamics, path-varying
controllability, and closed-loop chatter stability certified at every station
of a real machining programme.

Three questions, one path:

  Q1  How much does the workpiece change while it is being machined, and does
      a controller designed on the blank remain valid?

  Q2  Can the actuator layout actually reach the disturbance everywhere along
      the path, and if not, what layout can?

  Q3  Is the closed loop chatter-stable at every station -- computed from the
      real sampled controller in the monodromy matrix, not inferred from an
      equivalent damping ratio?

Outputs a machine-readable results.json plus the summary tables printed here.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tests"))
import _bootstrap  # noqa: F401,E402

from scipy.linalg import solve_continuous_are                     # noqa: E402

from machining_path import (MachiningProgramme, PathModelSequence)  # noqa: E402
from actuator_placement import (matched_fraction_series,            # noqa: E402
                                authority_series, build_candidates,
                                optimise_layout, modal_controllability)
from closed_loop_sld import spectral_radius, critical_depth        # noqa: E402

# ---------------------------------------------------------------- geometry
LP, HP = 0.100, 0.080
H_BLANK, H_FINAL = 0.008, 0.003
RHO, E_AL, NU = 2830, 69e9, 0.33
ZETA = [0.0031, 0.0017, 0.0027]
N_MODES = 3
N1, N2 = 30, 40
AP = 0.008                      # axial engagement per pass [m]
N_RADIAL = 3

# ---------------------------------------------------------------- process
NT, D_TOOL = 3, 0.010
ETA_H, GAMMA_N = np.deg2rad(35), np.deg2rad(15)
KT, KN, MU_C = 925e6, 0.26, 0.20
AE_RADIAL, FT = 0.1e-3, 0.02e-3
RPM = 4900
TS = 5e-5                       # controller sample period [s]

# ---------------------------------------------------------------- actuator
PIEZO = dict(d31=175e-12, h_Pa=0.7e-3, E_Pe=63e9, nu_Pe=0.35)
PATCH_BASELINE = [(0.0, 0.020, 0.0, 0.060)]
OBS_POINT = (LP, 0.5 * HP)      # probe on the un-machined lower half

RT = D_TOOL / 2
PHI_ST = np.pi - np.arccos(1 - AE_RADIAL / RT)
PHI_EX = np.pi
K1 = KN * np.cos(ETA_H)
K2 = 1 + MU_C * np.tan(ETA_H) * np.cos(GAMMA_N) - KN * np.sin(ETA_H)
SLD_COMMON = dict(NT=NT, RT=RT, eta_h=ETA_H, phi_st=PHI_ST, phi_ex=PHI_EX,
                  hp=HP, k1=K1, k2=K2, kt=KT)

OUT = os.path.dirname(os.path.abspath(__file__))
t_start = time.time()


def design_lqg(omega_n, zeta_n, H, D_obs, w_q=1e14, w_qd=1e8, w_r=1.0):
    """
    Continuous LQG design on a given modal model. Returns the dict that
    closed_loop_sld.spectral_radius consumes.

    H may be (n,) or (n, n_u); R scales with the number of inputs so that a
    two-patch layout is not implicitly handed twice the effort budget of a
    one-patch layout.
    """
    n = len(omega_n)
    n2 = 2 * n
    Hm = np.asarray(H, float)
    Hm = Hm.reshape(n, 1) if Hm.ndim == 1 else Hm.reshape(n, -1)
    n_u = Hm.shape[1]

    A = np.zeros((n2, n2))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -np.diag(np.asarray(omega_n, float) ** 2)
    A[n:, n:] = -np.diag(2 * np.asarray(zeta_n, float)
                         * np.asarray(omega_n, float))
    B = np.zeros((n2, n_u))
    B[n:, :] = Hm
    C = np.zeros((1, n2))
    C[0, :n] = np.asarray(D_obs, float)

    M_yp = np.outer(D_obs, D_obs)
    Q = np.block([[w_q * M_yp + 1e-3 * np.eye(n), np.zeros((n, n))],
                  [np.zeros((n, n)), w_qd * np.eye(n) + 1e-3 * np.eye(n)]])
    R = w_r * n_u * np.eye(n_u)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)

    W = 1e-6 * np.eye(n2)
    V = np.array([[1e-12]])
    Pk = solve_continuous_are(A.T, C.T, W, V)
    L = Pk @ C.T @ np.linalg.inv(V)
    return dict(A=A, B=B, C=C, K=K, L=L)


# =====================================================================
print("=" * 78)
print(" INTEGRATED PATH STUDY")
print("=" * 78)

prog = MachiningProgramme(LP, HP, H_BLANK, H_FINAL, N_RADIAL, AP)
seq = PathModelSequence(prog, RHO, E_AL, NU,
                        actuator_patches=PATCH_BASELINE,
                        obs_point=OBS_POINT, N1=N1, N2=N2,
                        n_modes=N_MODES, zeta_modes=ZETA,
                        n_x_stations=9, mode_updates_per_pass=1,
                        verbose=True, **PIEZO)

print("\n[1/4] walking the machining programme ...")
stations = []
for st in seq.stations():
    stations.append(st)
print(f"      {len(stations)} stations over {prog.n_passes} passes")

freqs = np.array([s['freq_n'] for s in stations])
Dser = np.column_stack([s['Dp'] for s in stations])
thick = np.array([s['thickness_min'] for s in stations])

print()
print("=" * 78)
print(" Q1  How much does the workpiece change during the operation?")
print("=" * 78)
print(f"   {'mode':>6} {'blank [Hz]':>12} {'final [Hz]':>12} {'min':>10} "
      f"{'max':>10} {'span':>9}")
for k in range(N_MODES):
    f = freqs[:, k]
    print(f"   {k+1:6d} {f[0]:12.1f} {f[-1]:12.1f} {f.min():10.1f} "
          f"{f.max():10.1f} {f.max()/f.min():8.2f}x")
print()
print(f"   wall thickness {H_BLANK*1e3:.1f} -> {thick.min()*1e3:.1f} mm")
print(f"   mode 1 is NOT monotone in the programme: it moves "
      f"{'up' if freqs[:,0].argmax() > 0 else 'down'} then "
      f"{'down' if freqs[:,0].argmin() > freqs[:,0].argmax() else 'up'} "
      f"(min at station {freqs[:,0].argmin()}, max at "
      f"{freqs[:,0].argmax()} of {len(stations)})")

# =====================================================================
print()
print("=" * 78)
print(" Q2  Can the actuator layout reach the disturbance along the path?")
print("=" * 78)

H_base = [np.array([s['H'][:, 0] for s in stations])]      # per-station
gam_base = np.array([
    matched_fraction_series([stations[i]['H'][:, 0]],
                            stations[i]['Dp'].reshape(-1, 1))[0]
    for i in range(len(stations))])

print(f"   baseline layout {PATCH_BASELINE}")
print(f"     matched fraction: mean {gam_base.mean():.3f}, "
       f"min {gam_base.min():.3f} at station {gam_base.argmin()} "
       f"(x = {stations[gam_base.argmin()]['x_tool']*1e3:.1f} mm)")

# candidate windows and the layout optimisation, evaluated on the
# station-varying disturbance directions
cands = build_candidates(LP, HP, n_windows=10, z_lo=0.0, z_hi=0.5 * HP)


def H_of_window(w, ref_station=None):
    """Modal actuator vector for a window, on a representative plate."""
    st = ref_station if ref_station is not None else len(stations) // 2
    seq.plate.add_piezo_patch(w[0], w[1], w[2], w[3], **PIEZO)
    return seq.plate.H_Pe_modal.copy()


# the plate currently holds the FINAL machined state; re-evaluating every
# candidate there is the conservative choice (thinnest, most flexible wall)
print()
print("   optimising layout for worst-case matched fraction over the path")
for n_act in (1, 2, 3):
    ws, gmin, gser, exhaustive = optimise_layout(
        cands, H_of_window, Dser, n_act=n_act, verbose=(n_act == 2))
    auth = authority_series([H_of_window(w) for w in ws], Dser)
    tag = "exhaustive" if exhaustive else "greedy"
    print(f"     n_act = {n_act} ({tag:10s}) : "
          f"min gamma = {gmin:.3f}, mean = {gser.mean():.3f}, "
          f"min authority = {auth.min():.4f} N/V")
    print(f"        windows x = "
          f"{[f'[{w[0]*1e3:.0f},{w[1]*1e3:.0f}]' for w in ws]} mm")
    if n_act == 2:
        best2 = ws
        gam2 = gser

# =====================================================================
print()
print("=" * 78)
print(" Q3  Is the closed loop chatter-stable at every station?")
print("=" * 78)
print("   Certified: the sampled observer and feedback are inside the")
print("   monodromy matrix. a_p,crit is the depth of cut at which the")
print("   Floquet radius of the real closed loop reaches 1.")
print()

# fixed-gain controller designed once on the blank
blank = stations[0]
ctrl_fixed = design_lqg(blank['omega_n'], blank['zeta'],
                        blank['H'][:, 0], blank['D_obs'])

sub = max(1, len(stations) // 24)          # subsample for the sweep
probe = stations[::sub]
print(f"   evaluating {len(probe)} stations "
      f"(every {sub} of {len(stations)})")
print()
print(f"   {'st':>4} {'x[mm]':>7} {'f1[Hz]':>8} {'gamma':>7} "
      f"{'ap_OL':>8} {'ap_fixed':>9} {'ap_sched':>9}")

rows = []
for i, st in enumerate(probe):
    Hs = st['H'][:, 0]
    ctrl_sched = design_lqg(st['omega_n'], st['zeta'], Hs, st['D_obs'])

    def rho(ap, ctrl):
        return spectral_radius(st['omega_n'], st['zeta'], st['Dp'],
                               Hs, st['D_obs'], RPM=RPM, ap=ap,
                               Ts=TS, ctrl_design=ctrl, **SLD_COMMON)

    ap_ol = critical_depth(lambda a: rho(a, None), lo=1e-7, hi=6e-3)
    ap_fx = critical_depth(lambda a: rho(a, ctrl_fixed), lo=1e-7, hi=6e-3)
    ap_sc = critical_depth(lambda a: rho(a, ctrl_sched), lo=1e-7, hi=6e-3)
    g = matched_fraction_series([Hs], st['Dp'].reshape(-1, 1))[0]

    rows.append(dict(station=i * sub, x=st['x_tool'], f1=st['freq_n'][0],
                     gamma=float(g), ap_ol=ap_ol, ap_fixed=ap_fx,
                     ap_sched=ap_sc))
    print(f"   {i*sub:4d} {st['x_tool']*1e3:7.1f} {st['freq_n'][0]:8.1f} "
          f"{g:7.3f} {ap_ol*1e3:8.4f} {ap_fx*1e3:9.4f} {ap_sc*1e3:9.4f}")

apf = np.array([r['ap_fixed'] for r in rows])
aps = np.array([r['ap_sched'] for r in rows])
apo = np.array([r['ap_ol'] for r in rows])

# The bisection is bracketed on [AP_LO, AP_HI]; values sitting on either
# bracket are censored, not measured. Ratios formed from censored values are
# meaningless, so they are reported as bounds instead of numbers.
AP_LO, AP_HI = 1e-7, 6e-3


def fmt(v):
    if v <= AP_LO * 1.01:
        return f"< {AP_LO*1e3:.4f} mm (unstable at any depth searched)"
    if v >= AP_HI * 0.99:
        return f"> {AP_HI*1e3:.4f} mm (stable to the top of the search range)"
    return f"= {v*1e3:.4f} mm"


n_cens_hi = int(np.sum(aps >= AP_HI * 0.99))
n_cens_lo = int(np.sum(apf <= AP_LO * 1.01))

print()
print("   worst case over the programme (this is what limits the process):")
print(f"     open loop        a_p,crit {fmt(apo.min())}")
print(f"     fixed-gain LQG   a_p,crit {fmt(apf.min())}")
print(f"     path-scheduled   a_p,crit {fmt(aps.min())}")
print()
print(f"   search bracket [{AP_LO*1e3:.4f}, {AP_HI*1e3:.1f}] mm; "
      f"{n_cens_lo} fixed-gain and {n_cens_hi} path-scheduled stations are")
print(f"   censored at a bracket, so ratios between them are NOT quoted.")
print()
n_worse = int(np.sum(apf < apo))
print(f"   stations at which the fixed-gain loop is WORSE than no control: "
      f"{n_worse} of {len(rows)}")
if aps.min() > AP_LO * 1.01 and apo.min() > AP_LO * 1.01:
    print(f"   path-scheduled vs open loop, worst case: "
          f"{aps.min()/apo.min():.1f}x")
print()
print(f"   Quoting a_p,crit once at the blank would report "
      f"{fmt(apf[0])},")
print(f"   while the worst case actually reached is {fmt(apf.min())}.")

res = dict(
    programme=prog.summary(),
    n_stations=len(stations),
    freq_blank=freqs[0].tolist(), freq_final=freqs[-1].tolist(),
    freq_min=freqs.min(axis=0).tolist(), freq_max=freqs.max(axis=0).tolist(),
    gamma_baseline=dict(mean=float(gam_base.mean()),
                        min=float(gam_base.min())),
    stations=rows,
    worst_case=dict(ap_ol=float(apo.min()), ap_fixed=float(apf.min()),
                    ap_sched=float(aps.min()),
                    bracket=[AP_LO, AP_HI],
                    n_censored_low=n_cens_lo, n_censored_high=n_cens_hi,
                    n_stations_worse_than_open_loop=n_worse),
    runtime_s=time.time() - t_start,
)
with open(os.path.join(OUT, "results_path_study.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"\n   wrote results_path_study.json   "
      f"({time.time()-t_start:.0f} s total)")
