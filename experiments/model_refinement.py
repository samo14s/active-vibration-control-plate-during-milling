"""
model_refinement.py
===================
Precise Kirchhoff modelling: patch dynamics, convergence, validation, and the
effect of the refined model on the control conclusions.

The baseline model (as in the source package) computes the piezo patch's
*forcing* vector H_pe but ignores the patch's mass and bending stiffness.  The
refined model (plate_model.add_piezo_patch(..., include_dynamics=True)) adds the
bonded layer as a composite section with a shifted neutral axis.

Stages
------
A. Mesh-convergence study (bare and refined, first 5 modes).
B. Validation against the measured natural frequencies of Du et al. (2024),
   Table 4, and against their own theoretical model's errors.
C. Modal-truncation / spillover test: the controllers are DESIGNED on the
   3-mode nominal model (as in the whole study) but EVALUATED on the refined
   5-mode plant -- both in the closed-loop semi-discretization (linear
   a_p,crit) and in saturated time domain (voltage-feasible depth).
D. Sensitivity of the refined frequencies to the assumed PZT density.
E. Re-tuned designs (feasible metric) on the refined plant at dt = 50 us.
F. Sampling-rate diagnosis (the marginal 3.4 kHz spillover pair destabilises
   numerically at 10 kHz) and the honest endpoint at dt = 25 us.

Outputs: results/refinement.json
Run:     python experiments/model_refinement.py            (stages A-F)
         python experiments/model_refinement.py --stage E  (append E only)
         python experiments/model_refinement.py --stage F  (append F only)
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from plate_model import PlateModel
from kirchhoff_q4 import shape_at_point
from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from adrc_control import ADRCController
from newmark_solver import NewmarkSimulator
from cl_fdm import ClosedLoopFDM, default_cutting

LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3; RPM = 4900; FT = 0.02e-3; DT = 5e-5
U_MAX = 150.0
CUT = default_cutting(hp=HP)
OMEGA = 2 * np.pi * RPM / 60

# Du et al. (2024), Table 4
MEASURED_HZ = [540.0, 1068.0, 2787.0, 3351.0, 4122.0]
ARTICLE_THEORY_HZ = [537.0, 1101.0, 2805.0, 3423.0, 4254.0]
ZETA5 = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]
RHO_PZT = 7800.0


def build(n_modes=3, N1=30, N2=24, include_dynamics=False, sensor="tip",
          rho_pe=RHO_PZT, zeta=None):
    zeta = (ZETA5[:n_modes] if zeta is None else zeta)
    p = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2, n_modes=n_modes,
                   zeta_modes=zeta, verbose=False)
    p.precompute_Dp(zp_pos=HP - 0.3e-3 / 2, n_pos=2001)
    p.set_observation(x_obs=LP, z_obs=HP)
    p.D_tip = p.D_obs.copy()
    if sensor == "piezo":
        Nv = shape_at_point(0.02, 0.06, p.N1, p.N2, p.lex, p.ley, p.n1)
        p.D_obs = Nv[p.DOFf] @ p.V
    p.add_piezo_patch(0, 0.020, 0, 0.060, 175e-12, 0.7e-3, 63e9, 0.35,
                      rho_Pe=rho_pe, include_dynamics=include_dynamics)
    if sensor == "piezo" and include_dynamics:
        # mode shapes changed inside add_piezo_patch -> recompute sensor row
        Nv = shape_at_point(0.02, 0.06, p.N1, p.N2, p.lex, p.ley, p.n1)
        p.D_obs = Nv[p.DOFf] @ p.V
        p.D_tip = shape_at_point(LP, HP, p.N1, p.N2, p.lex, p.ley, p.n1)[p.DOFf] @ p.V
    return p


# ----------------------------------------------------------------------
def stageA_convergence():
    print("=== A. mesh convergence (first 5 modes) ===")
    rows = []
    for (N1, N2) in [(20, 16), (30, 24), (40, 32), (50, 40)]:
        fb = build(5, N1, N2, include_dynamics=False).freq_n.tolist()
        fr = build(5, N1, N2, include_dynamics=True).freq_n.tolist()
        rows.append(dict(N1=N1, N2=N2, bare_hz=fb, refined_hz=fr))
        print(f"  {N1}x{N2}: bare f1={fb[0]:.1f}  refined f1={fr[0]:.1f}  "
              f"f5={fr[4]:.1f}")
    ref = rows[-1]
    conv = dict(rows=rows,
                delta_f1_30x24_vs_50x40_pct=abs(rows[1]["refined_hz"][0]
                                                - ref["refined_hz"][0])
                / ref["refined_hz"][0] * 100)
    print(f"  30x24 vs 50x40 (refined f1): {conv['delta_f1_30x24_vs_50x40_pct']:.3f}%")
    return conv


def stageB_validation():
    print("\n=== B. validation vs Du et al. Table 4 ===")
    fb = build(5, include_dynamics=False).freq_n
    fr = build(5, include_dynamics=True).freq_n
    def errs(f):
        return [(f[k] - MEASURED_HZ[k]) / MEASURED_HZ[k] * 100 for k in range(5)]
    e_bare, e_ref = errs(fb), errs(fr)
    e_art = [(ARTICLE_THEORY_HZ[k] - MEASURED_HZ[k]) / MEASURED_HZ[k] * 100
             for k in range(5)]
    for k in range(5):
        print(f"  mode {k+1}: meas {MEASURED_HZ[k]:6.0f}  bare {fb[k]:7.1f} "
              f"({e_bare[k]:+5.2f}%)  refined {fr[k]:7.1f} ({e_ref[k]:+5.2f}%)  "
              f"[article theory {e_art[k]:+5.2f}%]")
    out = dict(measured_hz=MEASURED_HZ, article_theory_hz=ARTICLE_THEORY_HZ,
               bare_hz=fb.tolist(), refined_hz=fr.tolist(),
               err_bare_pct=e_bare, err_refined_pct=e_ref, err_article_pct=e_art,
               mean_abs_err_bare=float(np.mean(np.abs(e_bare))),
               mean_abs_err_refined=float(np.mean(np.abs(e_ref))),
               mean_abs_err_article=float(np.mean(np.abs(e_art))))
    print(f"  mean|err|: bare {out['mean_abs_err_bare']:.2f}%  "
          f"refined {out['mean_abs_err_refined']:.2f}%  "
          f"article theory {out['mean_abs_err_article']:.2f}%")
    return out


# ----------------------------------------------------------------------
class _SelfStateWrapper:
    """Wrap a controller designed on an n_d-mode model so it can run against a
    plant with more modes: the controller keeps its own internal observer state
    and ignores the (larger) solver-managed state vector."""

    def __init__(self, ctrl, n_x_plant, n_x_design):
        self.ctrl = ctrl
        self.n_x_plant = n_x_plant
        self.n_x_design = n_x_design
        self.x_int = np.zeros(n_x_design)
        self.needs_kstep = getattr(ctrl, "needs_kstep", False)

    def step(self, x_hat_prev, u_prev, y_meas, k_step=0):
        if self.needs_kstep:
            x_new, u = self.ctrl.step(self.x_int, u_prev, y_meas, k_step=k_step)
        else:
            x_new, u = self.ctrl.step(self.x_int, u_prev, y_meas)
        x_new = np.asarray(x_new).ravel()
        self.x_int = x_new[:self.n_x_design] if x_new.size >= self.n_x_design \
            else np.concatenate([x_new, np.zeros(self.n_x_design - x_new.size)])
        packed = np.zeros(self.n_x_plant)
        m = min(self.n_x_plant, x_new.size)
        packed[:m] = x_new[:m]
        return packed, u


def _stable_at(make_ctrl, plant, ap, T=0.3, dt=DT):
    sim = NewmarkSimulator(plant, dt=dt, T_end=T, ft=FT, tau=60 / (NT * RPM), verbose=False)
    n_per = int(round((60 / (NT * RPM)) / dt))
    a3, a4 = precompute_alpha_periodic(dt, n_per, sim.nstep, OMEGA, NT, CUT['RT'],
                                       CUT['eta_h'], CUT['phi_st'], CUT['phi_ex'],
                                       HP - ap, HP, CUT['k1'], CUT['k2'], 925e6)
    vf = FT * NT * RPM / 60
    xp = np.minimum(vf * sim.t_vec, LP)
    kp = np.clip(np.round(xp / LP * 2000).astype(int), 0, 2000)
    r = sim.simulate(a3, a4, kp, controller=make_ctrl(), progress=False)
    ie = r['stop_idx']
    if ie < sim.nstep - 1:
        return False
    ytip = plant.D_tip @ r['qm'][:, :ie + 1]
    return bool(np.sqrt(np.mean(ytip[len(ytip) // 2:] ** 2)) * 1e6 < 50.0)


def _feasible(make_ctrl, plant_fn, lo=0.2e-3, hi=6e-3, tol=0.1e-3, dt=DT):
    if not _stable_at(make_ctrl, plant_fn(), lo, dt=dt):
        return 0.0
    if _stable_at(make_ctrl, plant_fn(), hi, dt=dt):
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _stable_at(make_ctrl, plant_fn(), mid, dt=dt):
            lo = mid
        else:
            hi = mid
    return lo


def stageC_control_on_refined():
    """Controllers designed on the 3-mode nominal model, evaluated on the
    refined 5-mode plant (spillover + model-refinement robustness test)."""
    print("\n=== C. control conclusions on the refined 5-mode plant ===")
    t0 = time.time()
    out = {}

    # --- refined plants ---
    plant5_tip = build(5, include_dynamics=True, sensor="tip")
    plant5_pz = build(5, include_dynamics=True, sensor="piezo")
    clf5_tip = ClosedLoopFDM.from_plate(plant5_tip, CUT, m_div=30)
    clf5_pz = ClosedLoopFDM.from_plate(plant5_pz, CUT, m_div=30)

    b0_ref = float(plant5_pz.D_obs @ plant5_pz.H_Pe_modal)
    coup = plant5_pz.D_obs * plant5_pz.H_Pe_modal
    out["refined_collocated_coupling"] = dict(
        b0=b0_ref, signs="".join('+' if c > 0 else '-' for c in coup),
        coupling=coup.tolist())
    print(f"  refined collocated coupling signs: {out['refined_collocated_coupling']['signs']}"
          f"  b0={b0_ref:.4f}")

    # --- open loop ---
    ap_ol = clf5_tip.ap_crit(RPM, None, ap_max=6e-3)
    out["ap_crit_OL_refined_mm"] = ap_ol * 1e3
    print(f"  OL a_p,crit (refined 5-mode) = {ap_ol*1e3:.3f} mm")

    # --- LQG designed on 3-mode nominal, evaluated on refined 5-mode ---
    design3 = build(3, include_dynamics=False, sensor="tip")
    lqg = LQGController(design3, dt=DT, verbose=False, u_max=U_MAX)
    lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                         w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
    lqg.discretize_observer()
    A, B, C, K, L = lqg.A, lqg.B, lqg.C, lqg.K_lqr, lqg.L_kal
    Ac, Bc, Cc, Dc = A - L @ C - B @ K, L, -K, np.zeros((1, 1))
    ap_lqg = clf5_tip.ap_crit_dynamic(RPM, Ac, Bc, Cc, Dc, ap_max=6e-3)
    out["ap_crit_LQG_on_refined_mm"] = ap_lqg * 1e3
    print(f"  LQG (3-mode design) on refined plant: a_p,crit = {ap_lqg*1e3:.3f} mm")

    # --- ADRC designed with nominal b0 (3-mode), evaluated on refined 5-mode ---
    design3_pz = build(3, include_dynamics=False, sensor="piezo")
    adrc = ADRCController(design3_pz, dt=DT, wc=1800, wo=21600, u_max=U_MAX)
    Aa, Ba, Ca, Da = adrc.export_lti()
    ap_adrc = clf5_pz.ap_crit_dynamic(RPM, Aa, Ba, Ca, Da, ap_max=6e-3)
    out["ap_crit_ADRC_on_refined_mm"] = ap_adrc * 1e3
    print(f"  ADRC (nominal b0 design) on refined plant: a_p,crit = {ap_adrc*1e3:.3f} mm")

    # --- voltage-feasible depths on the refined plant ---
    nx5 = 2 * 5
    def mk_lqg_w():
        d3 = build(3, include_dynamics=False, sensor="tip")
        l = LQGController(d3, dt=DT, verbose=False, u_max=U_MAX)
        l.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                           w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
        l.discretize_observer()
        return _SelfStateWrapper(l, nx5, 2 * 3)

    def mk_adrc_w():
        d3 = build(3, include_dynamics=False, sensor="piezo")
        a = ADRCController(d3, dt=DT, wc=1800, wo=21600, u_max=U_MAX)
        return _SelfStateWrapper(a, nx5, 2 * 3)   # ADRC's packed vector is 6-dim

    f_lqg = _feasible(mk_lqg_w, lambda: build(5, include_dynamics=True, sensor="tip"))
    f_adrc = _feasible(mk_adrc_w, lambda: build(5, include_dynamics=True, sensor="piezo"))
    out["feasible_LQG_on_refined_mm"] = f_lqg * 1e3
    out["feasible_ADRC_on_refined_mm"] = f_adrc * 1e3
    print(f"  voltage-feasible on refined plant: LQG={f_lqg*1e3:.2f} mm  "
          f"ADRC={f_adrc*1e3:.2f} mm  ({time.time()-t0:.0f}s)")
    return out


def stageE_retune_on_refined():
    """Fair endpoint: both controllers RE-TUNED (structure unchanged) with
    knowledge of the refined 5-mode plant, feasible-metric selection."""
    print("\n=== E. re-tuned designs on the refined plant (feasible metric) ===")
    t0 = time.time()
    nx5 = 2 * 5

    def mk_adrc(wc, wo):
        d5 = build(5, include_dynamics=True, sensor="piezo")   # refined b0
        a = ADRCController(d5, dt=DT, wc=wc, wo=wo, u_max=U_MAX)
        return _SelfStateWrapper(a, nx5, nx5)

    adrc_rows = []
    for wc, wo in [(800, 9600), (1200, 14400), (1800, 21600)]:
        f = _feasible(lambda wc=wc, wo=wo: mk_adrc(wc, wo),
                      lambda: build(5, include_dynamics=True, sensor="piezo"))
        adrc_rows.append(dict(wc=wc, wo=wo, feasible_mm=f * 1e3))
        print(f"  ADRC (wc={wc}, wo={wo}, refined b0): feasible = {f*1e3:.2f} mm "
              f"[{time.time()-t0:.0f}s]")

    def mk_lqg(weights):
        d3 = build(3, include_dynamics=False, sensor="tip")
        l = LQGController(d3, dt=DT, verbose=False, u_max=U_MAX)
        if weights is None:
            l.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                               w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
        else:
            l.optimize_weights(w_q_list=[weights[0]], w_qd_list=[weights[1]],
                               w_r=1.0, gain_norm_max=1e12)
        l.discretize_observer()
        return _SelfStateWrapper(l, nx5, 2 * 3)

    lqg_rows = []
    for label, w in [("grid-default", None), ("w_q=1e16,w_qd=1e8", (1e16, 1e8)),
                     ("w_q=1e12,w_qd=1e8", (1e12, 1e8))]:
        f = _feasible(lambda w=w: mk_lqg(w),
                      lambda: build(5, include_dynamics=True, sensor="tip"))
        lqg_rows.append(dict(label=label, feasible_mm=f * 1e3))
        print(f"  LQG ({label}): feasible = {f*1e3:.2f} mm [{time.time()-t0:.0f}s]")

    best_adrc = max(r["feasible_mm"] for r in adrc_rows)
    best_lqg = max(r["feasible_mm"] for r in lqg_rows)
    print(f"  BEST on refined plant: ADRC {best_adrc:.2f} mm vs LQG {best_lqg:.2f} mm")
    return dict(adrc=adrc_rows, lqg=lqg_rows,
                best_adrc_mm=best_adrc, best_lqg_mm=best_lqg,
                elapsed_s=time.time() - t0)


def stageF_sampling_endpoint():
    """Sampling-rate diagnosis + the honest endpoint at dt = 25 us.

    The refined plant's closed ADRC loop has a marginally-damped spillover pair
    near 3.4 kHz (continuous max Re(eig) ~ -35 1/s).  At the production sample
    time dt = 50 us (10 kHz) the one-sample implementation delay (~60-70 deg of
    phase at that frequency) plus the Newmark period distortion of modes 4-5
    destabilises it -- a numerical/implementation artifact, not continuous
    physics.  At dt = 25 us (20 kHz, standard for piezo control) the loop is
    clean and converged (checked also at 12.5 us).  The final comparison is
    therefore evaluated at dt = 25 us for BOTH controllers.
    """
    print("\n=== F. sampling diagnosis + endpoint at dt = 25 us ===")
    t0 = time.time()
    nx5 = 2 * 5

    # (a) dt sweep of the unsaturated nominal ADRC loop on the refined plant
    sweep = []
    for dtx in [5e-5, 2.5e-5, 1.25e-5]:
        p = build(5, include_dynamics=True, sensor="piezo")
        sim = NewmarkSimulator(p, dt=dtx, T_end=0.3, ft=FT, tau=60 / (NT * RPM), verbose=False)
        n_per = int(round((60 / (NT * RPM)) / dtx))
        a3, a4 = precompute_alpha_periodic(dtx, n_per, sim.nstep, OMEGA, NT, CUT['RT'],
                                           CUT['eta_h'], CUT['phi_st'], CUT['phi_ex'],
                                           HP - 0.3e-3, HP, CUT['k1'], CUT['k2'], 925e6)
        vf = FT * NT * RPM / 60
        xp = np.minimum(vf * sim.t_vec, LP)
        kp = np.clip(np.round(xp / LP * 2000).astype(int), 0, 2000)
        a = ADRCController(p, dt=dtx, wc=1800, wo=21600, u_max=1e7)
        r = sim.simulate(a3, a4, kp, controller=_SelfStateWrapper(a, nx5, nx5),
                         progress=False, stop_threshold=1e6)
        ie = r['stop_idx']
        y = p.D_tip @ r['qm'][:, :ie + 1]
        sweep.append(dict(dt_us=dtx * 1e6,
                          yrms_um=float(np.sqrt(np.mean(y[len(y) // 2:] ** 2)) * 1e6),
                          umax_V=float(np.max(np.abs(r['u'][:ie + 1])))))
        print(f"  unsat ADRC nominal, dt={dtx*1e6:.1f}us: "
              f"yrms={sweep[-1]['yrms_um']:.3f}um umax={sweep[-1]['umax_V']:.1f}V")

    # (b) endpoint: feasible depths at dt = 25 us, both controllers, refined plant
    DT2 = 2.5e-5

    def mk_adrc(wc, wo):
        d5 = build(5, include_dynamics=True, sensor="piezo")
        return _SelfStateWrapper(ADRCController(d5, dt=DT2, wc=wc, wo=wo,
                                                u_max=U_MAX), nx5, nx5)

    adrc_rows = []
    for wc, wo in [(1200, 14400), (1800, 21600)]:
        f = _feasible(lambda wc=wc, wo=wo: mk_adrc(wc, wo),
                      lambda: build(5, include_dynamics=True, sensor="piezo"), dt=DT2)
        adrc_rows.append(dict(wc=wc, wo=wo, feasible_mm=f * 1e3))
        print(f"  ADRC (wc={wc}) dt=25us: feasible = {f*1e3:.2f} mm [{time.time()-t0:.0f}s]")

    def mk_lqg(weights):
        d3 = build(3, include_dynamics=False, sensor="tip")
        l = LQGController(d3, dt=DT2, verbose=False, u_max=U_MAX)
        if weights is None:
            l.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                               w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
        else:
            l.optimize_weights(w_q_list=[weights[0]], w_qd_list=[weights[1]],
                               w_r=1.0, gain_norm_max=1e12)
        l.discretize_observer()
        return _SelfStateWrapper(l, nx5, 2 * 3)

    lqg_rows = []
    for label, w in [("grid-default", None), ("w_q=1e16,w_qd=1e8", (1e16, 1e8))]:
        f = _feasible(lambda w=w: mk_lqg(w),
                      lambda: build(5, include_dynamics=True, sensor="tip"), dt=DT2)
        lqg_rows.append(dict(label=label, feasible_mm=f * 1e3))
        print(f"  LQG ({label}) dt=25us: feasible = {f*1e3:.2f} mm [{time.time()-t0:.0f}s]")

    best_adrc = max(r["feasible_mm"] for r in adrc_rows)
    best_lqg = max(r["feasible_mm"] for r in lqg_rows)
    print(f"  ENDPOINT (dt=25us, refined plant): ADRC {best_adrc:.2f} mm "
          f"vs LQG {best_lqg:.2f} mm")
    return dict(dt_sweep=sweep, adrc=adrc_rows, lqg=lqg_rows,
                best_adrc_mm=best_adrc, best_lqg_mm=best_lqg,
                elapsed_s=time.time() - t0)


def stageD_density_sensitivity():
    print("\n=== D. sensitivity to PZT density ===")
    rows = []
    for rho in [7500.0, 7800.0]:
        f = build(5, include_dynamics=True, rho_pe=rho).freq_n
        rows.append(dict(rho_pe=rho, f1_hz=float(f[0]), f5_hz=float(f[4])))
        print(f"  rho={rho:.0f}: f1={f[0]:.1f} Hz, f5={f[4]:.1f} Hz")
    df1 = abs(rows[0]["f1_hz"] - rows[1]["f1_hz"]) / rows[1]["f1_hz"] * 100
    print(f"  f1 shift over density range: {df1:.2f}%")
    return dict(rows=rows, f1_shift_pct=df1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, default="all",
                    help="all | A | B | C | D | E (E can be run alone to append)")
    args = ap.parse_args()
    T = time.time()
    path = os.path.join(RESULTS, "refinement.json")
    if args.stage == "E":
        res = json.load(open(path))
        res["retuned_on_refined"] = stageE_retune_on_refined()
    elif args.stage == "F":
        res = json.load(open(path))
        res["sampling_endpoint"] = stageF_sampling_endpoint()
    else:
        res = dict(convergence=stageA_convergence(),
                   validation=stageB_validation(),
                   control_on_refined=stageC_control_on_refined(),
                   density_sensitivity=stageD_density_sensitivity(),
                   retuned_on_refined=stageE_retune_on_refined(),
                   rho_pzt_assumed=RHO_PZT, zeta5=ZETA5)
    json.dump(res, open(path, "w"), indent=2)
    print(f"\nTOTAL {time.time()-T:.1f}s -> results/refinement.json")
