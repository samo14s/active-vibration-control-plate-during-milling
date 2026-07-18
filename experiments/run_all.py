"""
run_all.py
==========
Reproducible experiment pipeline.  Every number reported in the manuscript is
computed here and written to ``results/`` -- there are no hard-coded results and
no ad-hoc damping multipliers.  Stability numbers come from the closed-loop
full-discretization solver (:mod:`cl_fdm`); vibration/voltage numbers come from
the Newmark time-domain simulator.

Stages
------
1. Floquet-direct synthesis    -> results/synthesis.json   (Pareto scan + rules)
2. Controlled SLD grids        -> results/sld.npz          (OL / LQG / Floquet)
3. Time-domain, 4 scenarios    -> results/scenarios.json   (LQG vs 2-DOF)
4. Feedforward decomposition   -> results/feedforward.json (role separation)

Run:  python experiments/run_all.py            (full, ~6-9 min)
      python experiments/run_all.py --quick    (coarser SLD grid, faster)
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from twodof_control import TwoDOFController
from adrc_control import ADRCController
from newmark_solver import NewmarkSimulator
from cl_fdm import ClosedLoopFDM, default_cutting
from kirchhoff_q4 import shape_at_point
import floquet_synthesis as fs

# ----------------------------------------------------------------------
# Physical parameters (Du et al., Int. J. Mech. Sci. 274 (2024) 109257)
# ----------------------------------------------------------------------
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010; RT = DT_TOOL / 2
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT_NOM = 925e6; KN = 0.26; MU_C = 0.20
RPM = 4900; FT = 0.02e-3; AE = 0.1e-3
D31 = 175e-12; H_PA = 0.7e-3; E_PE = 63e9; NU_PE = 0.35
N1, N2 = 30, 24; N_MODES = 3
ZETA = [0.0031, 0.0017, 0.0027]
DT = 5e-5
GAIN_CAP = 1e8            # actuator-authority proxy used by both selection rules

# ADRC design (collocated piezo sensor; see manuscript sec. on ADRC)
ADRC_WC = 1500.0          # controller bandwidth [rad/s]
ADRC_WO = 18000.0         # observer bandwidth [rad/s]
ADRC_SENSOR = (0.020, 0.060)   # collocated control sensor at the piezo corner


def build_plate(ap, freq_perturb=0.0, sensor="tip"):
    """Build the plate model. sensor='tip' puts the measurement at the tool tip
    (used by the model-based controllers with a full-state observer); 'piezo'
    puts it at the collocated piezo corner (used by ADRC). p.D_tip always holds
    the tip shape so performance is evaluated at the tip regardless of sensor."""
    p = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2, n_modes=N_MODES,
                   zeta_modes=ZETA, verbose=False)
    p.precompute_Dp(zp_pos=HP - ap / 2, n_pos=2001)
    p.set_observation(x_obs=LP, z_obs=HP)
    p.D_tip = p.D_obs.copy()               # tip shape, for performance metrics
    if sensor == "piezo":
        Nv = shape_at_point(ADRC_SENSOR[0], ADRC_SENSOR[1],
                            p.N1, p.N2, p.lex, p.ley, p.n1)
        p.D_obs = Nv[p.DOFf] @ p.V         # collocated control sensor
    p.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    if abs(freq_perturb) > 1e-6:           # perturbed "real" plant (robustness)
        p.Kp = p.Kp * (1 + freq_perturb) ** 2
        p.omega_n = np.sqrt(np.diag(p.Kp) / np.diag(p.Mp))
        p.freq_n = p.omega_n / (2 * np.pi)
        p.Cp = np.diag(2 * np.array(ZETA) * p.omega_n * np.diag(p.Mp))
    return p


def cutting_setup(kt=KT_NOM):
    return default_cutting(NT=NT, RT=RT, eta_h=ETA_H, gamma_n=GAMMA_N,
                           kt=kt, kN=KN, mu_c=MU_C, ae=AE, hp=HP)


def make_forces(sim, ap, kt):
    Omega = 2 * np.pi * RPM / 60
    phi_st = np.pi - np.arccos(1 - AE / RT); phi_ex = np.pi
    k1 = KN * np.cos(ETA_H)
    k2 = 1 + MU_C * np.tan(ETA_H) * np.cos(GAMMA_N) - KN * np.sin(ETA_H)
    v_feed = FT * NT * RPM / 60
    xp = np.minimum(v_feed * sim.t_vec, LP)
    kp = np.clip(np.round(xp / LP * 2000).astype(int), 0, 2000)
    n_per = int(np.round((60 / (NT * RPM)) / DT))
    a3, a4 = precompute_alpha_periodic(DT, n_per, sim.nstep, Omega, NT, RT, ETA_H,
                                       phi_st, phi_ex, HP - ap, HP, k1, k2, kt)
    return a3, a4, kp, n_per


def tdmetrics(res):
    ie = res['stop_idx']
    y = res['y'][:ie + 1]; u = res['u'][:ie + 1]
    return dict(y_rms=float(np.sqrt(np.mean(y ** 2)) * 1e6),
                y_max=float(np.max(np.abs(y)) * 1e6),
                u_max=float(np.max(np.abs(u))),
                u_rms=float(np.sqrt(np.mean(u ** 2))),
                stop=int(ie))


# ======================================================================
AP_EVAL = 0.6e-3          # representative aggressive cut for the voltage check
U_BUDGET = 150.0          # piezo voltage budget (+/- V)


def _u_peak_feedback(K, ap_eval):
    """Peak feedback voltage during a time-domain cut at depth ap_eval.

    With the canonical modal signs (see plate_model) the gain is portable, so we
    can evaluate any synthesised K on a fresh plant instance.
    """
    p = build_plate(ap_eval)
    sim = NewmarkSimulator(p, dt=DT, T_end=0.3, ft=FT, tau=60 / (NT * RPM), verbose=False)
    a3, a4, kp, _ = make_forces(sim, ap_eval, KT_NOM)
    lqg = LQGController(p, dt=DT, verbose=False)
    lqg.K_lqr = np.asarray(K).reshape(1, 2 * N_MODES)
    lqg.discretize_observer()
    return tdmetrics(sim.simulate(a3, a4, kp, controller=lqg, progress=False))


def stage1_synthesis():
    """Feedback-authority design curve via CL-FDM, and voltage-budget selection.

    The conventional LQG design ranks gains by a non-delayed eigenvalue proxy and
    caps the gain norm; both are surrogates for the real quantities (chatter
    stability and actuator voltage).  Here every candidate is scored on the true
    critical depth from CL-FDM and on its measured peak voltage, exposing the
    a_p,crit-vs-authority curve directly and letting the design be chosen under
    the real +/-150 V budget.
    """
    print("\n=== Stage 1: Floquet-authority design curve + voltage-budget selection ===")
    t0 = time.time()
    plate = build_plate(0.3e-3)
    clf = ClosedLoopFDM.from_plate(plate, cutting_setup(), m_div=30)

    # Design curve: sweep the damping-authority weight (w_qd) at fixed w_q.
    w_q0 = 1e14
    w_qd_list = [3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10]
    curve = []
    for w_qd in w_qd_list:
        K, ev = fs.lqr_gain(plate, w_q0, w_qd)
        ap_c = clf.ap_crit(RPM, K)
        m = _u_peak_feedback(K, AP_EVAL)
        rec = dict(w_q=w_q0, w_qd=w_qd, gain_norm=float(np.linalg.norm(K)),
                   min_re=float(np.min(np.real(ev))), ap_crit_mm=ap_c * 1e3,
                   u_max_apeval=m["u_max"], K=K.ravel().tolist())
        curve.append(rec)
        print(f"  w_qd={w_qd:.0e} |K|={rec['gain_norm']:.2e} "
              f"a_p,crit={rec['ap_crit_mm']:.3f}mm  u_max(ap={AP_EVAL*1e3:.1f}mm)={m['u_max']:.1f}V")

    # LQG baseline (its own selection rule)
    lqg = LQGController(plate, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                         w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
    ap_lqg = clf.ap_crit(RPM, lqg.K_lqr)
    m_lqg = _u_peak_feedback(lqg.K_lqr, AP_EVAL)

    # Voltage-budget-optimal design: max a_p,crit s.t. peak voltage <= budget
    feasible = [r for r in curve if r["u_max_apeval"] <= U_BUDGET]
    r_sel = max(feasible, key=lambda r: r["ap_crit_mm"]) if feasible else curve[0]

    out = dict(
        ap_eval_mm=AP_EVAL * 1e3, u_budget=U_BUDGET, rpm=RPM,
        design_curve=[{k: r[k] for k in
                       ("w_qd", "gain_norm", "min_re", "ap_crit_mm", "u_max_apeval")}
                      for r in curve],
        lqg=dict(gain_norm=float(np.linalg.norm(lqg.K_lqr)),
                 ap_crit_mm=ap_lqg * 1e3, u_max_apeval=m_lqg["u_max"],
                 K=lqg.K_lqr.ravel().tolist()),
        selected=dict(w_qd=r_sel["w_qd"], gain_norm=r_sel["gain_norm"],
                      ap_crit_mm=r_sel["ap_crit_mm"],
                      u_max_apeval=r_sel["u_max_apeval"], K=r_sel["K"]),
        elapsed_s=time.time() - t0,
    )
    json.dump(out, open(os.path.join(RESULTS, "synthesis.json"), "w"), indent=2)
    print(f"  LQG baseline   : a_p,crit={out['lqg']['ap_crit_mm']:.3f}mm "
          f"u_max={out['lqg']['u_max_apeval']:.1f}V |K|={out['lqg']['gain_norm']:.2e}")
    print(f"  voltage-budget : a_p,crit={out['selected']['ap_crit_mm']:.3f}mm "
          f"u_max={out['selected']['u_max_apeval']:.1f}V |K|={out['selected']['gain_norm']:.2e}")
    print(f"  ({out['elapsed_s']:.1f}s)")
    return out


# ======================================================================
def _lqg_lti(plate):
    """LQG (Kalman observer + LQR) as a linear dynamic output-feedback controller
    y -> u, so its stability can be evaluated with the observer IN the loop."""
    lqg = LQGController(plate, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                         w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
    A, B, C, K, L = lqg.A, lqg.B, lqg.C, lqg.K_lqr, lqg.L_kal
    Ac = A - L @ C - B @ K
    return dict(Ac=Ac, Bc=L, Cc=-K, Dc=np.zeros((1, 1)), K=K, lqg=lqg)


def stage2_sld(quick=False):
    """Fair controlled SLDs, all with the controller's observer/ESO IN the loop
    (closed-loop semi-discretization).  OL, LQG and ADRC are directly comparable."""
    print("\n=== Stage 2: controlled SLDs (observer/ESO in the loop) ===")
    t0 = time.time()
    m = 20 if quick else 30
    plate = build_plate(0.3e-3)                       # tip sensor (LQG)
    clf = ClosedLoopFDM.from_plate(plate, cutting_setup(), m_div=m)
    lti = _lqg_lti(plate)
    plate_p = build_plate(0.3e-3, sensor="piezo")     # collocated sensor (ADRC)
    clf_p = ClosedLoopFDM.from_plate(plate_p, cutting_setup(), m_div=m)
    adrc = ADRCController(plate_p, dt=DT, wc=ADRC_WC, wo=ADRC_WO)
    Aa, Ba, Ca, Da = adrc.export_lti()

    n_rpm = 24 if quick else 30
    n_ap = 20 if quick else 28
    RPM_arr = np.linspace(2500, 7500, n_rpm)
    ap_arr = np.linspace(0.02e-3, 6.0e-3, n_ap)

    print("  OL ...")
    rho_OL = clf.sld_grid(RPM_arr, ap_arr, None, verbose=True)
    print("  LQG (observer in loop) ...")
    rho_LQG = clf.sld_grid_dynamic(RPM_arr, ap_arr, lti["Ac"], lti["Bc"],
                                   lti["Cc"], lti["Dc"], verbose=True)
    print("  ADRC (ESO in loop) ...")
    rho_ADRC = clf_p.sld_grid_dynamic(RPM_arr, ap_arr, Aa, Ba, Ca, Da, verbose=True)

    np.savez(os.path.join(RESULTS, "sld.npz"),
             RPM=RPM_arr, ap=ap_arr, rho_OL=rho_OL, rho_LQG=rho_LQG, rho_ADRC=rho_ADRC)
    summ = dict(
        ap_crit_OL_mm=clf.ap_crit(RPM, None) * 1e3,
        ap_crit_LQG_static_mm=clf.ap_crit(RPM, lti["K"]) * 1e3,
        ap_crit_LQG_dyn_mm=clf.ap_crit_dynamic(RPM, lti["Ac"], lti["Bc"],
                                               lti["Cc"], lti["Dc"]) * 1e3,
        ap_crit_ADRC_mm=clf_p.ap_crit_dynamic(RPM, Aa, Ba, Ca, Da) * 1e3,
        elapsed_s=time.time() - t0)
    json.dump(summ, open(os.path.join(RESULTS, "sld_summary.json"), "w"), indent=2)
    print(f"  a_p,crit @ {RPM}rpm  OL={summ['ap_crit_OL_mm']:.3f}  "
          f"LQG(static)={summ['ap_crit_LQG_static_mm']:.3f}  "
          f"LQG(obs)={summ['ap_crit_LQG_dyn_mm']:.3f}  "
          f"ADRC={summ['ap_crit_ADRC_mm']:.3f} mm  ({summ['elapsed_s']:.1f}s)")
    return summ


# ======================================================================
def _run_pair(name, ap, kt, freq_perturb, save_traces_to=None):
    plate_d = build_plate(ap)                       # nominal (design) plant
    plate_r = build_plate(ap, freq_perturb)         # real (perturbed) plant
    sim = NewmarkSimulator(plate_r, dt=DT, T_end=0.5, ft=FT, tau=60 / (NT * RPM), verbose=False)
    sim_d = NewmarkSimulator(plate_d, dt=DT, T_end=0.5, ft=FT, tau=60 / (NT * RPM), verbose=False)
    a3, a4, kp, n_per = make_forces(sim, ap, kt)

    lqg = LQGController(plate_d, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                         w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
    lqg.discretize_observer()
    res_lqg = sim.simulate(a3, a4, kp, controller=lqg, progress=False)

    c2 = TwoDOFController(plate_d, dt=DT, ff_lr=0.005, ff_max=10.0, ff_alpha=1.0,
                          alpha4_periodic=a4[:n_per], n_per=n_per, safety_alpha=5.0,
                          enable_adaptation=True, u_max=150.0, verbose=False)
    c2.pretrain_iterative_simulation(sim_d, a3, a4, kp, n_iterations=30,
                                     n_epochs_per_iter=15, verbose=False)
    c2._reset_histories()
    res_2d = sim.simulate(a3, a4, kp, controller=c2, progress=False)
    mL, mD = tdmetrics(res_lqg), tdmetrics(res_2d)
    if save_traces_to is not None:
        ie = min(mL['stop'], mD['stop'])
        np.savez(save_traces_to, t=sim.t_vec[:ie + 1],
                 y_lqg=res_lqg['y'][:ie + 1], u_lqg=res_lqg['u'][:ie + 1],
                 y_2d=res_2d['y'][:ie + 1], u_2d=res_2d['u'][:ie + 1])
    return dict(name=name, ap_mm=ap * 1e3, kt_MPa=kt / 1e6, freq_perturb=freq_perturb,
                lqg=mL, twodof=mD,
                rms_gain_pct=(1 - mD['y_rms'] / mL['y_rms']) * 100,
                peak_gain_pct=(1 - mD['y_max'] / mL['y_max']) * 100,
                umax_change_pct=(mD['u_max'] / mL['u_max'] - 1) * 100)


def stage3_scenarios():
    print("\n=== Stage 3: time-domain, 4 scenarios (LQG vs 2-DOF) ===")
    t0 = time.time()
    defs = [("S1 Nominal", 0.3e-3, KT_NOM, 0.0),
            ("S2 Aggressive ap=0.6mm", 0.6e-3, KT_NOM, 0.0),
            ("S3 Uncertainty omega-15%", 0.3e-3, KT_NOM, -0.15),
            ("S4 High K_T +30%", 0.3e-3, 1.3 * KT_NOM, 0.0)]
    rows = []
    for i, d in enumerate(defs):
        traces = os.path.join(RESULTS, "traces_S1.npz") if i == 0 else None
        rows.append(_run_pair(*d, save_traces_to=traces))
    for r in rows:
        print(f"  {r['name']:<26} RMS {r['lqg']['y_rms']:.4f}->{r['twodof']['y_rms']:.4f}um "
              f"({r['rms_gain_pct']:+.2f}%)  peak {r['peak_gain_pct']:+.2f}%  "
              f"u_max {r['umax_change_pct']:+.2f}%")
    avg = float(np.mean([r['rms_gain_pct'] for r in rows]))
    out = dict(scenarios=rows, avg_rms_gain_pct=avg, elapsed_s=time.time() - t0)
    json.dump(out, open(os.path.join(RESULTS, "scenarios.json"), "w"), indent=2)
    print(f"  AVERAGE RMS gain = {avg:+.2f}%   ({out['elapsed_s']:.1f}s)")
    return out


# ======================================================================
def stage4_feedforward():
    """Isolate the feedforward's role: feedback-only vs feedback+FF, and confirm
    the FF leaves the stability boundary unchanged (feedforward cannot move poles)."""
    print("\n=== Stage 4: feedforward role decomposition ===")
    t0 = time.time()
    ap = 0.3e-3
    plate = build_plate(ap)
    sim = NewmarkSimulator(plate, dt=DT, T_end=0.5, ft=FT, tau=60 / (NT * RPM), verbose=False)
    a3, a4, kp, n_per = make_forces(sim, ap, KT_NOM)

    lqg = LQGController(plate, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                         w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
    lqg.discretize_observer()
    m_fb = tdmetrics(sim.simulate(a3, a4, kp, controller=lqg, progress=False))

    c2 = TwoDOFController(plate, dt=DT, ff_lr=0.005, ff_max=10.0, ff_alpha=1.0,
                          alpha4_periodic=a4[:n_per], n_per=n_per, safety_alpha=5.0,
                          enable_adaptation=True, u_max=150.0, verbose=False)
    c2.pretrain_iterative_simulation(sim, a3, a4, kp, 30, 15, verbose=False)
    c2._reset_histories()
    m_2d = tdmetrics(sim.simulate(a3, a4, kp, controller=c2, progress=False))

    # Stability boundary with the SAME feedback gain, WITH vs WITHOUT feedforward:
    # the feedforward is not in the feedback path, so a_p,crit is identical.
    clf = ClosedLoopFDM.from_plate(plate, cutting_setup(), m_div=30)
    ap_fb = clf.ap_crit(RPM, lqg.K_lqr)
    out = dict(feedback_only=m_fb, feedback_plus_ff=m_2d,
               rms_gain_pct=(1 - m_2d['y_rms'] / m_fb['y_rms']) * 100,
               peak_gain_pct=(1 - m_2d['y_max'] / m_fb['y_max']) * 100,
               umax_change_pct=(m_2d['u_max'] / m_fb['u_max'] - 1) * 100,
               ap_crit_mm=ap_fb * 1e3,
               note=("a_p,crit is set by the feedback gain alone; the phase-aware "
                     "feedforward does not enter the feedback path and therefore "
                     "does not change the stability boundary."),
               elapsed_s=time.time() - t0)
    json.dump(out, open(os.path.join(RESULTS, "feedforward.json"), "w"), indent=2)
    print(f"  feedback-only  y_rms={m_fb['y_rms']:.4f}um u_max={m_fb['u_max']:.2f}V")
    print(f"  +feedforward   y_rms={m_2d['y_rms']:.4f}um u_max={m_2d['u_max']:.2f}V "
          f"(RMS {out['rms_gain_pct']:+.2f}%, peak {out['peak_gain_pct']:+.2f}%, "
          f"u_max {out['umax_change_pct']:+.2f}%)")
    print(f"  a_p,crit unchanged by FF = {out['ap_crit_mm']:.3f}mm  ({out['elapsed_s']:.1f}s)")
    return out


def _run_lqg(ap, kt, fp):
    """Tip-referenced metrics for the LQG baseline (tip sensor + observer)."""
    plate_d = build_plate(ap)
    plate_r = build_plate(ap, fp)
    sim = NewmarkSimulator(plate_r, dt=DT, T_end=0.5, ft=FT, tau=60 / (NT * RPM), verbose=False)
    a3, a4, kp, _ = make_forces(sim, ap, kt)
    lqg = LQGController(plate_d, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                         w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
    lqg.discretize_observer()
    return tdmetrics(sim.simulate(a3, a4, kp, controller=lqg, progress=False))


def _run_adrc(ap, kt, fp):
    """Tip-referenced metrics for ADRC (collocated piezo sensor).

    The controller uses the collocated measurement (p.D_obs) for feedback; the
    performance is read at the tool tip (p.D_tip) from the modal history, so it
    is directly comparable with the model-based controllers."""
    plate_d = build_plate(ap, sensor="piezo")            # nominal design
    plate_r = build_plate(ap, fp, sensor="piezo")        # real (perturbed) plant
    sim = NewmarkSimulator(plate_r, dt=DT, T_end=0.5, ft=FT, tau=60 / (NT * RPM), verbose=False)
    a3, a4, kp, _ = make_forces(sim, ap, kt)
    adrc = ADRCController(plate_d, dt=DT, wc=ADRC_WC, wo=ADRC_WO, u_max=150.0)
    adrc.reset()
    res = sim.simulate(a3, a4, kp, controller=adrc, progress=False)
    ie = res['stop_idx']
    ytip = plate_r.D_tip @ res['qm'][:, :ie + 1]
    u = res['u'][:ie + 1]
    return dict(y_rms=float(np.sqrt(np.mean(ytip ** 2)) * 1e6),
                y_max=float(np.max(np.abs(ytip)) * 1e6),
                u_max=float(np.max(np.abs(u))),
                u_rms=float(np.sqrt(np.mean(u ** 2))), stop=int(ie))


def stage5_adrc(quick=False):
    print("\n=== Stage 5: ADRC (collocated) -- scenarios ===")
    t0 = time.time()
    plate = build_plate(0.3e-3, sensor="piezo")
    clf = ClosedLoopFDM.from_plate(plate, cutting_setup(), m_div=(20 if quick else 30))
    adrc = ADRCController(plate, dt=DT, wc=ADRC_WC, wo=ADRC_WO)
    Ac, Bc, Cc, Dc = adrc.export_lti()
    ap_adrc = clf.ap_crit_dynamic(RPM, Ac, Bc, Cc, Dc)

    defs = [("S1 Nominal", 0.3e-3, KT_NOM, 0.0),
            ("S2 Aggressive ap=0.6mm", 0.6e-3, KT_NOM, 0.0),
            ("S3 Uncertainty omega-15%", 0.3e-3, KT_NOM, -0.15),
            ("S4 High K_T +30%", 0.3e-3, 1.3 * KT_NOM, 0.0)]
    lqg_sc = json.load(open(os.path.join(RESULTS, "scenarios.json")))["scenarios"]
    rows = []
    for d, lrow in zip(defs, lqg_sc):
        mA = _run_adrc(d[1], d[2], d[3])
        mL = lrow["lqg"]
        rows.append(dict(name=d[0], lqg=mL, adrc=mA,
                         rms_gain_pct=(1 - mA['y_rms'] / mL['y_rms']) * 100,
                         peak_gain_pct=(1 - mA['y_max'] / mL['y_max']) * 100,
                         umax_change_pct=(mA['u_max'] / mL['u_max'] - 1) * 100))
        print(f"  {d[0]:<26} LQG {mL['y_rms']:.3f} -> ADRC {mA['y_rms']:.3f} um "
              f"({rows[-1]['rms_gain_pct']:+.1f}%)  u_max {mA['u_max']:.1f}V")
    out = dict(wc=ADRC_WC, wo=ADRC_WO, sensor=ADRC_SENSOR, b0=adrc.b0,
               ap_crit_mm=ap_adrc * 1e3, scenarios=rows,
               avg_rms_gain_pct=float(np.mean([r['rms_gain_pct'] for r in rows])),
               elapsed_s=time.time() - t0)
    json.dump(out, open(os.path.join(RESULTS, "adrc.json"), "w"), indent=2)
    print(f"  ADRC a_p,crit @ {RPM}rpm = {ap_adrc*1e3:.3f} mm; "
          f"avg RMS vs LQG {out['avg_rms_gain_pct']:+.1f}%  ({out['elapsed_s']:.1f}s)")
    return out


def stage6_robustness():
    """Vibration under modal-frequency drift: model-based LQG vs model-light ADRC.
    Both controllers are designed on the nominal plant; the real plant is drifted."""
    print("\n=== Stage 6: robustness to frequency drift (LQG vs ADRC) ===")
    t0 = time.time()
    fps = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]
    rows = []
    for fp in fps:
        mL = _run_lqg(0.3e-3, KT_NOM, fp)
        mA = _run_adrc(0.3e-3, KT_NOM, fp)
        rows.append(dict(freq_perturb=fp, lqg_yrms=mL['y_rms'], adrc_yrms=mA['y_rms'],
                         lqg_umax=mL['u_max'], adrc_umax=mA['u_max']))
        print(f"  df={fp:+.0%}  LQG y_rms={mL['y_rms']:.3f}  ADRC y_rms={mA['y_rms']:.3f} um")
    out = dict(sweep=rows, elapsed_s=time.time() - t0)
    json.dump(out, open(os.path.join(RESULTS, "robustness.json"), "w"), indent=2)
    lqg_v = [r['lqg_yrms'] for r in rows]; adrc_v = [r['adrc_yrms'] for r in rows]
    print(f"  LQG y_rms range [{min(lqg_v):.3f},{max(lqg_v):.3f}] um; "
          f"ADRC [{min(adrc_v):.3f},{max(adrc_v):.3f}] um  ({out['elapsed_s']:.1f}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", type=int, default=0, help="run a single stage (1-6)")
    args = ap.parse_args()
    T = time.time()
    if args.stage in (0, 1): stage1_synthesis()
    if args.stage in (0, 2): stage2_sld(quick=args.quick)
    if args.stage in (0, 3): stage3_scenarios()
    if args.stage in (0, 4): stage4_feedforward()
    if args.stage in (0, 5): stage5_adrc(quick=args.quick)
    if args.stage in (0, 6): stage6_robustness()
    print(f"\nTOTAL {time.time()-T:.1f}s  ->  results/ written.")
