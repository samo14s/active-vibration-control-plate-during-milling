"""
main_adaptive_removal.py
========================
Material-removal drift + robustness stress benchmark:
LQG (fixed) vs ESO-ADRC (certified fixed design) vs A-ESO-ADRC (supervised ladder).

The plant's modal frequencies drift DURING the pass (solver schedule Kp·s(t)²,
Cp·s(t); ramp over 60 % of the pass, then hold), or are statically perturbed, or
the actuator effectiveness is degraded. All controllers are designed once on the
nominal model; A-ESO-ADRC additionally switches between its two pre-designed rungs
online, driven by the measured cost only (no identification, no probe — see
02_controllers/adrc_controller.py).

Scenarios:
  D0  no drift (sanity — adaptation must cost ~nothing)
  D1  ramp to +15 %  (the article's stiffening direction)
  D2  ramp to -12 %  (beyond the fixed-LQG margin)
  D3  static -12 %   (whole pass at the drifted plant)
  D4  actuator effectiveness x0.25 (piezo d31/bonding degradation)

Output: figs_lqg_vs_adrc/fig_adaptive_removal.png|pdf + console table.
"""
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic, cutting_constants
from lqg_controller import LQGController
from adrc_controller import ESO_ADRC_Controller, AdaptiveESO_ADRC_Controller
from newmark_solver import NewmarkSimulator

# ---------------- physical parameters (article) ----------------
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3;  DT_TOOL = 0.010
ETA_H = np.deg2rad(35);  GAMMA_N = np.deg2rad(15)
KT_NOMINAL = 925e6;  KN = 0.26;  MU_C = 0.20
RPM = 4900
FT = 0.02e-3;  AE = 0.1e-3
D31 = 175e-12;  H_PA = 0.7e-3;  E_PE = 63e9;  NU_PE = 0.35
N1, N2 = 30, 24
N_MODES_PLANT, N_MODES_CTRL = 5, 3
ZETA_PLANT = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]
MEAS_NOISE_STD = 1e-8
KALMAN_V = 1e-12
DT = 5e-5
T_END = 0.5
AP = 0.3e-3

OUT_DIR = "figs_lqg_vs_adrc"
os.makedirs(OUT_DIR, exist_ok=True)

PHI_ST = np.pi - np.arccos(1 - AE/(DT_TOOL/2))
PHI_EX = np.pi
OMEGA = 2*np.pi*RPM/60
RT = DT_TOOL/2
TAU = 60/(NT*RPM)
K1, K2 = cutting_constants(KN, MU_C, ETA_H, GAMMA_N)
N_PER = int(np.round(TAU/DT))


def build_plant(freq_perturb=0.0):
    p = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2,
                   n_modes=N_MODES_PLANT, zeta_modes=ZETA_PLANT, verbose=False)
    p.precompute_Dp(zp_pos=HP - AP/2, n_pos=2001)
    p.set_observation(x_obs=LP, z_obs=HP)
    p.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    if abs(freq_perturb) > 1e-9:
        p = p.perturbed_copy(freq_perturb)
    return p


print("[drift] building plants and frozen controllers...")
plant = build_plant()
design = plant.truncated_view(N_MODES_CTRL)

lqg = LQGController(design, dt=DT, verbose=False, kalman_V=KALMAN_V, u_max=150.0)
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg.discretize_observer()

PERF_DESIGN = (1e16, 1e8, 3e3)      # from the main_simulation design grid
CERT_DESIGN = (1e14, 1e8, 1e4)
adrc = ESO_ADRC_Controller(design, DT, *CERT_DESIGN, kalman_V=KALMAN_V,
                           verbose=False)
aadrc = AdaptiveESO_ADRC_Controller(design, DT, rungs=(PERF_DESIGN, CERT_DESIGN),
                                    perf_rung=0, robust_rung=1,
                                    kalman_V=KALMAN_V, verbose=False)
CONTROLLERS = [("LQG", lqg), ("ESO-ADRC", adrc), ("A-ESO-ADRC", aadrc)]
COL = {"LQG": '#2E8B57', "ESO-ADRC": '#1E5AA8', "A-ESO-ADRC": '#DC143C'}

NSTEP = int(np.round(T_END/DT)) + 1


def ramp_hold(s_end, frac=0.6):
    """Frequency-scale schedule: 1 -> s_end over the first `frac` of the pass."""
    k_ramp = int(frac*NSTEP)
    s = np.ones(NSTEP)
    s[:k_ramp] = np.linspace(1.0, s_end, k_ramp)
    s[k_ramp:] = s_end
    return s


def eff_plant(kappa):
    p = copy.copy(plant)
    p.H_Pe_modal = plant.H_Pe_modal * kappa
    return p


def run_case(pl, ctrl, sched=None):
    if hasattr(ctrl, 'reset_adaptation'):
        ctrl.reset_adaptation()
    sim = NewmarkSimulator(pl, dt=DT, T_end=T_END, ft=FT, tau=TAU, verbose=False)
    v_feed = FT*NT*RPM/60
    xp_path = np.minimum(v_feed*sim.t_vec, LP)
    kp_idx = np.clip(np.round(xp_path/LP*2000).astype(int), 0, 2000)
    a3, a4 = precompute_alpha_periodic(DT, N_PER, sim.nstep, OMEGA, NT, RT,
                                       ETA_H, PHI_ST, PHI_EX, HP-AP, HP,
                                       K1, K2, KT_NOMINAL)
    r = sim.simulate(a3, a4, kp_idx, controller=ctrl,
                     meas_noise_std=MEAS_NOISE_STD, omega_scale_t=sched,
                     rng=np.random.default_rng(1234), progress=False)
    i = r['stop_idx']
    out = dict(t=sim.t_vec, y=r['y'], u=r['u'], stop_idx=i,
               rms=np.sqrt(np.mean(r['y'][:i+1]**2))*1e6,
               umax=np.max(np.abs(r['u'][:i+1])),
               diverged=i < sim.nstep - 2)
    if hasattr(ctrl, 'history_rung') and len(ctrl.history_rung):
        out['rung'] = np.array(ctrl.history_rung)
        out['n_switch'] = int(np.sum(np.diff(out['rung']) != 0))
    return out


cases = [
    ("D0 no drift",           plant,           None),
    ("D1 ramp +15%",          plant,           ramp_hold(1.15)),
    ("D2 ramp -12%",          plant,           ramp_hold(0.88)),
    ("D3 static -12%",        build_plant(-0.12), None),
    ("D4 effectiveness x0.25", eff_plant(0.25), None),
]

results = {}
print("\n=== drift / stress benchmark (T = 0.5 s each) ===")
for cname, pl, sched in cases:
    print(f"\n--- {cname} ---")
    results[cname] = {}
    for lbl, ctrl in CONTROLLERS:
        res = run_case(pl, ctrl, sched=sched)
        results[cname][lbl] = res
        extra = ""
        if 'rung' in res:
            extra = (f"  [{res['n_switch']} switches, "
                     f"{np.mean(res['rung'] == 1)*100:.0f}% certified rung]")
        print(f"  {lbl:<12} rms={res['rms']:10.4f}µm  umax={res['umax']:7.2f}V  "
              f"{'DIVERGED' if res['diverged'] else 'OK'}{extra}")

# ---------------- figure ----------------
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0], hspace=0.45,
                      wspace=0.25)

# (a) y(t) for D2 ramp -12% (the discriminating case)
ax = fig.add_subplot(gs[0, :])
case = "D2 ramp -12%"
for lbl, _ in CONTROLLERS:
    res = results[case][lbl]
    i = res['stop_idx']
    tag = 'DIVERGED' if res['diverged'] else f"{res['rms']:.2f} µm"
    ax.plot(res['t'][:i+1]*1e3, np.abs(res['y'][:i+1])*1e6, color=COL[lbl],
            linewidth=0.7, alpha=0.9, label=f"{lbl}: {tag}")
ax.set_yscale('log')
ax.set_ylim([1e-3, 5e3])
ax.set_xlabel("Temps (ms)")
ax.set_ylabel("|y_p| (µm, log)")
ax.set_title(f"{case} — frequency drift DURING the pass (ramp over 60 %, "
             "then hold)", fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.4)

# (b) rung trace + drift schedule for D2
ax = fig.add_subplot(gs[1, 0])
res = results[case]["A-ESO-ADRC"]
if 'rung' in res:
    t_ms = res['t'][1:len(res['rung'])+1]*1e3
    ax.step(t_ms, res['rung'], where='post', color=COL["A-ESO-ADRC"],
            linewidth=1.5, label='active rung')
ax2 = ax.twinx()
ax2.plot(res['t']*1e3, ramp_hold(0.88)*100 - 100, color='gray', linestyle='--',
         linewidth=1.5, label='true drift (%)')
ax2.set_ylabel("plant frequency drift (%)", color='gray')
ax.set_yticks([0, 1])
ax.set_yticklabels(['perf', 'certified'])
ax.set_ylim([-0.2, 1.2])
ax.set_xlabel("Temps (ms)")
ax.set_title("A-ESO-ADRC rung supervision under D2", fontweight='bold')
ax.grid(True, alpha=0.4)

# (c) rung trace for D1
ax = fig.add_subplot(gs[1, 1])
res = results["D1 ramp +15%"]["A-ESO-ADRC"]
if 'rung' in res:
    t_ms = res['t'][1:len(res['rung'])+1]*1e3
    ax.step(t_ms, res['rung'], where='post', color=COL["A-ESO-ADRC"],
            linewidth=1.5)
ax2 = ax.twinx()
ax2.plot(res['t']*1e3, ramp_hold(1.15)*100 - 100, color='gray', linestyle='--',
         linewidth=1.5)
ax2.set_ylabel("plant frequency drift (%)", color='gray')
ax.set_yticks([0, 1])
ax.set_yticklabels(['perf', 'certified'])
ax.set_ylim([-0.2, 1.2])
ax.set_xlabel("Temps (ms)")
ax.set_title("A-ESO-ADRC rung supervision under D1", fontweight='bold')
ax.grid(True, alpha=0.4)

# (d) summary bars (log scale, DIV hatched at plot ceiling)
ax = fig.add_subplot(gs[2, :])
cnames = [c[0] for c in cases]
xb = np.arange(len(cnames))
wb = 0.26
CEIL = 1e3
for j, (lbl, _) in enumerate(CONTROLLERS):
    vals, hats = [], []
    for cname in cnames:
        res = results[cname][lbl]
        vals.append(min(res['rms'], CEIL) if res['diverged'] else res['rms'])
        hats.append('//' if res['diverged'] else None)
    bars = ax.bar(xb + (j-1)*wb, vals, wb, color=COL[lbl], alpha=0.85,
                  edgecolor='k', label=lbl, linewidth=1.2)
    for bar, v, h, cname in zip(bars, vals, hats, cnames):
        if h:
            bar.set_hatch(h)
            ax.text(bar.get_x()+bar.get_width()/2, v*1.1, 'DIV', ha='center',
                    fontsize=9, fontweight='bold', color='red')
        else:
            ax.text(bar.get_x()+bar.get_width()/2, v*1.1, f'{v:.2f}',
                    ha='center', fontsize=8, fontweight='bold')
ax.set_yscale('log')
ax.set_xticks(xb)
ax.set_xticklabels(cnames, fontsize=10)
ax.set_ylabel("y_RMS (µm, log)")
ax.set_title("Drift / stress benchmark — fixed designs vs supervised adaptation",
             fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, axis='y', which='both', alpha=0.4)

plt.suptitle("Material-removal drift & robustness stress: "
             "LQG vs ESO-ADRC vs A-ESO-ADRC", fontsize=15, fontweight='bold')
plt.savefig(f"{OUT_DIR}/fig_adaptive_removal.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig_adaptive_removal.pdf", bbox_inches='tight')
plt.close()
print(f"\n  ✓ {OUT_DIR}/fig_adaptive_removal.png|pdf")

# ---------------- console summary table ----------------
print("\n=== SUMMARY (y_RMS µm; DIV = diverged) ===")
hdr = "".join(f"{lbl:<14}" for lbl, _ in CONTROLLERS)
print(f"  {'case':<24}{hdr}")
for cname in cnames:
    row = ""
    for lbl, _ in CONTROLLERS:
        res = results[cname][lbl]
        row += f"{'DIV':<14}" if res['diverged'] else f"{res['rms']:<14.4f}"
    print(f"  {cname:<24}{row}")
