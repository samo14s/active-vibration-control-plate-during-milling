"""
material_removal.py
===================
In-process material removal: physically-generated varying dynamics, and
control through the machining process.

The FEM now supports in-process thinning (plate_model.remove_material): covered
elements are re-assembled at their reduced thickness (K ~ h^3, M ~ h,
partial-coverage weighted) and the modal basis, tool-path shapes, sensor rows
and actuator projection are recomputed (piecewise-frozen assumption: removal is
much slower than the vibrations).

Stages
------
A. Single pass at the article's conditions (a_e = 0.1 mm over the top 0.3 mm
   strip, full length): honest bounding result -- the modal drift is < 0.02 %,
   so within ONE pass the constant-dynamics assumption of the production
   studies is justified.
B. Thin-walling scenario: the top 20 mm band is milled from 4.0 mm down to
   3.0 mm in four 0.25 mm passes.  Removing mass at the free edge RAISES f1
   (kinetic energy dominates there): ~+10 % physically-generated drift, the
   realistic counterpart of the synthetic +/-20 % robustness sweep.
C. Control through the process: both controllers are designed ONCE at the
   initial state (as in Sec. 4.8's endpoint: dt = 25 us, LQG on the 3-mode
   nominal model, ADRC with the initial refined b0 and the clean-response
   tuning wc = 1800; the wc = 1200 alternative reaches a deeper feasibility
   boundary but exhibits a saturated limit cycle at the operating point) and
   then
   evaluated at every machining state -- linear a_p,crit (CL-SD, controller
   in the monodromy of the CURRENT plant) and saturated time domain at the
   operating depth 0.3 mm.

Outputs: results/material_removal.json
Run:     python experiments/material_removal.py
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from kirchhoff_q4 import shape_at_point
from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from adrc_control import ADRCController
from newmark_solver import NewmarkSimulator
from cl_fdm import ClosedLoopFDM, default_cutting
sys.path.insert(0, HERE)
from model_refinement import build, _SelfStateWrapper, ZETA5

LP, HP = 0.100, 0.080
NT = 3; RPM = 4900; FT = 0.02e-3
DT2 = 2.5e-5                      # converged sample time (Sec. 4.8)
U_MAX = 150.0
CUT = default_cutting(hp=HP)
OMEGA = 2 * np.pi * RPM / 60
BAND = (0.060, 0.080)             # thin-walled band [z1, z2]
PASS_DEPTH = 0.25e-3              # thickness removed per pass
N_PASS = 4                        # 4.0 -> 3.0 mm


def refresh_sensors(p, sensor="piezo"):
    """Re-derive the sensor rows from the CURRENT modal basis (they go stale
    whenever remove_material / patch dynamics re-run the modal analysis)."""
    p.D_tip = shape_at_point(LP, HP, p.N1, p.N2, p.lex, p.ley, p.n1)[p.DOFf] @ p.V
    if sensor == "piezo":
        p.D_obs = shape_at_point(0.02, 0.06, p.N1, p.N2,
                                 p.lex, p.ley, p.n1)[p.DOFf] @ p.V
    else:
        p.D_obs = p.D_tip.copy()
    return p


def plant_at_state(n_pass, sensor="piezo"):
    """Refined 5-mode plant after n_pass thinning passes on the top band."""
    p = build(5, include_dynamics=True, sensor=sensor)
    for _ in range(n_pass):
        p.remove_material(0.0, LP, BAND[0], BAND[1], PASS_DEPTH)
    return refresh_sensors(p, sensor)


# ----------------------------------------------------------------------
def stageA_single_pass():
    print("=== A. single pass at article conditions (bounding result) ===")
    p = build(5, include_dynamics=True, sensor="piezo")
    refresh_sensors(p)
    f0 = p.freq_n.copy(); b0_0 = float(p.D_obs @ p.H_Pe_modal)
    p.remove_material(0.0, LP, HP - 0.3e-3, HP, 0.1e-3)
    refresh_sensors(p)
    f1 = p.freq_n.copy(); b0_1 = float(p.D_obs @ p.H_Pe_modal)
    drift = [(f1[k] / f0[k] - 1) * 100 for k in range(5)]
    for k in range(5):
        print(f"  mode {k+1}: {f0[k]:8.2f} -> {f1[k]:8.2f} Hz ({drift[k]:+.4f} %)")
    print(f"  b0: {b0_0:+.4f} -> {b0_1:+.4f}")
    return dict(f_before=f0.tolist(), f_after=f1.tolist(), drift_pct=drift,
                b0_before=b0_0, b0_after=b0_1,
                max_abs_drift_pct=float(np.max(np.abs(drift))))


def stageB_thinning():
    print("\n=== B. thin-walling 4.0 -> 3.0 mm (top 20 mm band) ===")
    rows = []
    for s in range(N_PASS + 1):
        p = plant_at_state(s)
        b0 = float(p.D_obs @ p.H_Pe_modal)
        coup = p.D_obs * p.H_Pe_modal
        rows.append(dict(n_pass=s, h_band_mm=4.0 - 0.25 * s,
                         freq_hz=p.freq_n.tolist(), b0=b0,
                         signs="".join('+' if c > 0 else '-' for c in coup)))
        print(f"  h={rows[-1]['h_band_mm']:.2f} mm: f1={p.freq_n[0]:7.2f} Hz "
              f"({(p.freq_n[0]/rows[0]['freq_hz'][0]-1)*100:+5.2f}%)  "
              f"b0={b0:+.3f} [{rows[-1]['signs']}]")
    return rows


# ----------------------------------------------------------------------
def _op_point(make_ctrl, plant, ap=0.3e-3, T=0.4):
    sim = NewmarkSimulator(plant, dt=DT2, T_end=T, ft=FT, tau=60 / (NT * RPM), verbose=False)
    n_per = int(round((60 / (NT * RPM)) / DT2))
    a3, a4 = precompute_alpha_periodic(DT2, n_per, sim.nstep, OMEGA, NT, CUT['RT'],
                                       CUT['eta_h'], CUT['phi_st'], CUT['phi_ex'],
                                       HP - ap, HP, CUT['k1'], CUT['k2'], 925e6)
    vf = FT * NT * RPM / 60
    xp = np.minimum(vf * sim.t_vec, LP)
    kp = np.clip(np.round(xp / LP * 2000).astype(int), 0, 2000)
    r = sim.simulate(a3, a4, kp, controller=make_ctrl(), progress=False)
    ie = r['stop_idx']
    y = plant.D_tip @ r['qm'][:, :ie + 1]
    u = r['u'][:ie + 1]
    return dict(stable=bool(ie == sim.nstep - 1),
                y_rms_um=float(np.sqrt(np.mean(y ** 2)) * 1e6),
                u_max_V=float(np.max(np.abs(u))),
                sat_frac=float(np.mean(np.abs(u) > 0.99 * U_MAX)))


def stageC_control_through_process():
    print("\n=== C. control through the machining process (designs frozen at state 0) ===")
    t0 = time.time()
    nx5 = 2 * 5

    # --- designs frozen at the initial state (Sec. 4.8 endpoint choices) ---
    lqg0 = LQGController(build(3, include_dynamics=False, sensor="tip"),
                         dt=DT2, verbose=False, u_max=U_MAX)
    lqg0.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                          w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
    lqg0.discretize_observer()
    A, B, C = lqg0.A, lqg0.B, lqg0.C
    K, L = lqg0.K_lqr, lqg0.L_kal
    lqg_lti = (A - L @ C - B @ K, L, -K, np.zeros((1, 1)))

    adrc0 = ADRCController(plant_at_state(0), dt=DT2, wc=1800, wo=21600, u_max=U_MAX)
    adrc_lti = adrc0.export_lti()
    b0_frozen = adrc0.b0

    def mk_lqg():
        l = LQGController(build(3, include_dynamics=False, sensor="tip"),
                          dt=DT2, verbose=False, u_max=U_MAX)
        l.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                          w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
        l.discretize_observer()
        return _SelfStateWrapper(l, nx5, 2 * 3)

    def mk_adrc():
        d0 = plant_at_state(0)
        a = ADRCController(d0, dt=DT2, wc=1800, wo=21600, b0=b0_frozen, u_max=U_MAX)
        return _SelfStateWrapper(a, nx5, nx5)

    rows = []
    for s in range(N_PASS + 1):
        plant_pz = plant_at_state(s, "piezo")
        plant_tip = plant_at_state(s, "tip")
        clf_pz = ClosedLoopFDM.from_plate(plant_pz, CUT, m_div=30)
        clf_tip = ClosedLoopFDM.from_plate(plant_tip, CUT, m_div=30)
        ap_lqg = clf_tip.ap_crit_dynamic(RPM, *lqg_lti, ap_max=6e-3)
        ap_adrc = clf_pz.ap_crit_dynamic(RPM, *adrc_lti, ap_max=6e-3)
        op_lqg = _op_point(mk_lqg, plant_tip)
        op_adrc = _op_point(mk_adrc, plant_pz)
        rows.append(dict(n_pass=s, h_band_mm=4.0 - 0.25 * s,
                         f1_hz=float(plant_pz.freq_n[0]),
                         ap_crit_lqg_mm=ap_lqg * 1e3,
                         ap_crit_adrc_mm=ap_adrc * 1e3,
                         op_lqg=op_lqg, op_adrc=op_adrc))
        print(f"  h={rows[-1]['h_band_mm']:.2f}: a_p,crit LQG={ap_lqg*1e3:5.2f} "
              f"ADRC={ap_adrc*1e3:5.2f} mm | op 0.3mm: "
              f"LQG {'ok' if op_lqg['stable'] else 'CHATTER'} "
              f"{op_lqg['y_rms_um']:8.3f}um/{op_lqg['u_max_V']:3.0f}V  "
              f"ADRC {'ok' if op_adrc['stable'] else 'CHATTER'} "
              f"{op_adrc['y_rms_um']:8.3f}um/{op_adrc['u_max_V']:3.0f}V "
              f"[{time.time()-t0:.0f}s]")
    return rows


if __name__ == "__main__":
    T = time.time()
    res = dict(single_pass=stageA_single_pass(),
               thinning=stageB_thinning(),
               control_through_process=stageC_control_through_process(),
               band_mm=[BAND[0] * 1e3, BAND[1] * 1e3],
               pass_depth_mm=PASS_DEPTH * 1e3, dt_us=DT2 * 1e6)
    json.dump(res, open(os.path.join(RESULTS, "material_removal.json"), "w"), indent=2)
    print(f"\nTOTAL {time.time()-T:.1f}s -> results/material_removal.json")
