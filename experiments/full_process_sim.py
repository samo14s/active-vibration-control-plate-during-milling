"""
full_process_sim.py
===================
Continuous full-process milling simulation: N passes END-TO-END (x = 0 -> L,
20.41 s each at the article feed), with in-process material removal applied
BEHIND the advancing tool and the vibration state carried continuously across
every plant update.

This answers the limitation of the piecewise-frozen study
(material_removal.py stage C), whose time-domain checks covered only ~2.5 mm of
travel per state: here the machining process is simulated end to end,
4 x 20.41 s = 81.6 s of physical time at dt = 25 us (~3.3 M Newmark steps per
controller).

Mechanics of the moving update
------------------------------
* The tool advances at v_feed = f_t N_T Omega = 4.9 mm/s.  Every SEG = 5 mm
  segment completed, the just-machined span of the band [60, 80] mm is thinned
  by the per-pass depth (0.25 mm), the modal basis is recomputed, and the
  state is REPROJECTED onto the new basis with the mass-orthonormal transform

      T = V_new^T  M_new  V_old,     q+ = T q,  q'+ = T q',  q''+ = T q'',

  applied also to the whole regenerative delay history (the physical
  displacement field is continuous; the projection error is O(removal per
  segment)).  Sensor rows, tool-path shapes and the actuator projection are
  re-derived per update.
* Controllers are FROZEN designs from the initial state (Sec. 4.8 endpoint:
  dt = 25 us; LQG on the 3-mode nominal model; ADRC clean tuning wc = 1800
  with the initial b0).  Their internal states are basis-independent (ADRC's
  ESO is physical; LQG's observer lives in its own frozen design frame), so
  they carry over unchanged -- as they would on the real machine.
* Between passes the tool re-enters at x = 0 instantly (retract time
  neglected; free vibrations decay within ~0.1 s and the comparison is
  identical for both controllers).

Scenario abstraction (disclosed): each "pass" thins the whole 20 mm band by
0.25 mm, aggregating the many small axial step-downs of a real wall-finishing
cycle; the cutting force itself acts at the article's engagement (top 0.3 mm,
a_e = 0.1 mm) throughout.

Usage:  python experiments/full_process_sim.py --controller lqg
        python experiments/full_process_sim.py --controller adrc
Output: results/full_process_<name>.json  (+ decimated traces .npz)
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from adrc_control import ADRCController
from cl_fdm import default_cutting
sys.path.insert(0, HERE)
from model_refinement import build, _SelfStateWrapper

LP, HP = 0.100, 0.080
NT = 3; RPM = 4900; FT = 0.02e-3
DT = 2.5e-5
U_MAX = 150.0
CUT = default_cutting(hp=HP)
OMEGA = 2 * np.pi * RPM / 60
AP = 0.3e-3                        # article axial engagement (cut at top edge)
BAND = (0.060, 0.080)
PASS_DEPTH = 0.25e-3
N_PASS = 4
SEG = 0.005                        # plant-update segment length [m]
V_FEED = FT * NT * RPM / 60        # 4.9 mm/s
DECIM = 16                         # trace decimation (~2.5 kHz)
ZETA5 = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]


def refresh_rows(p):
    from kirchhoff_q4 import shape_at_point
    p.D_tip = shape_at_point(LP, HP, p.N1, p.N2, p.lex, p.ley, p.n1)[p.DOFf] @ p.V
    p.D_col = shape_at_point(0.02, 0.06, p.N1, p.N2, p.lex, p.ley, p.n1)[p.DOFf] @ p.V


def modal_matrices(p):
    n = p.n_modes
    Kp = np.diag(p.omega_n ** 2)
    Cp = np.diag(2 * np.array(ZETA5[:n]) * p.omega_n)
    return np.eye(n), Kp, Cp


def make_controller(which):
    if which == "lqg":
        l = LQGController(build(3, include_dynamics=False, sensor="tip"),
                          dt=DT, verbose=False, u_max=U_MAX)
        l.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                           w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
        l.discretize_observer()
        return _SelfStateWrapper(l, 10, 6), "tip"
    elif which == "adrc":
        d0 = build(5, include_dynamics=True, sensor="piezo")
        refresh_rows(d0)
        d0.D_obs = d0.D_col
        a = ADRCController(d0, dt=DT, wc=1800, wo=21600, u_max=U_MAX)
        return _SelfStateWrapper(a, 10, 10), "piezo"
    else:  # afc
        from afc_adrc import AFCADRCController
        d0 = build(5, include_dynamics=True, sensor="piezo")
        refresh_rows(d0)
        d0.D_obs = d0.D_col
        n_per = int(round(60.0 / (NT * RPM) / DT))
        a = AFCADRCController(d0, dt=DT, wc=1800, wo=21600, n_per=n_per,
                              K=5, tau_a=0.3, uff_max=30.0, u_max=U_MAX)
        return a, "piezo"


def run(which):
    t_wall = time.time()
    ctrl, sensor = make_controller(which)
    plate = build(5, include_dynamics=True, sensor="tip")
    refresh_rows(plate)

    n = plate.n_modes
    tau = 60.0 / (NT * RPM)
    n_tau = int(round(tau / DT))
    n_per = n_tau
    alpha3, alpha4 = precompute_alpha_periodic(
        DT, n_per, n_per, OMEGA, NT, CUT['RT'], CUT['eta_h'],
        CUT['phi_st'], CUT['phi_ex'], HP - AP, HP, CUT['k1'], CUT['k2'], 925e6)

    gNM, bNM = 0.5, 0.25
    q = np.zeros(n); qd = np.zeros(n); qdd = np.zeros(n)
    hist = np.zeros((n_tau, n))            # regenerative delay ring buffer
    u_prev = 0.0
    Mp, Kp, Cp = modal_matrices(plate)
    Hpe = plate.H_Pe_modal.copy()

    steps_per_seg = int(round(SEG / V_FEED / DT))
    segs_per_pass = int(round(LP / SEG))
    trace_y, trace_u, trace_x, trace_pass = [], [], [], []
    seg_rows = []
    k_global = 0

    for np_pass in range(1, N_PASS + 1):
        for s in range(segs_per_pass):
            x0 = s * SEG
            seg_y2 = 0.0; seg_umax = 0.0; seg_sat = 0
            for k in range(steps_per_seg):
                x_tool = min(x0 + V_FEED * k * DT, LP - 1e-9)
                kp_idx = min(int(round(x_tool / LP * 2000)), 2000)
                Dp_now = plate.Dp_array[:, kp_idx]
                DpT_Dp = np.outer(Dp_now, Dp_now)
                ph = k_global % n_per
                a3 = alpha3[ph]; a4 = alpha4[ph]

                if getattr(ctrl, "dual", False):
                    u = ctrl.step_dual(u_prev, plate.D_col @ q,
                                       plate.D_tip @ q, k_global)
                else:
                    y_ctrl = (plate.D_col if sensor == "piezo" else plate.D_tip) @ q
                    _, u = ctrl.step(None, u_prev, y_ctrl, k_step=k_global)

                q_del = hist[k_global % n_tau]     # q(t - tau)
                K_eff = Kp + a4 * DpT_Dp
                F = FT * a3 * Dp_now + a4 * DpT_Dp @ q_del + Hpe * u
                qd_pr = qd + (1 - gNM) * DT * qdd
                q_pr = q + DT * qd + (0.5 - bNM) * DT ** 2 * qdd
                S = Mp + gNM * DT * Cp + bNM * DT ** 2 * K_eff
                qdd = np.linalg.solve(S, F - Cp @ qd_pr - K_eff @ q_pr)
                qd = qd_pr + gNM * DT * qdd
                q = q_pr + bNM * DT ** 2 * qdd
                hist[k_global % n_tau] = q
                u_prev = u
                k_global += 1

                y_tip = plate.D_tip @ q
                seg_y2 += y_tip * y_tip
                au = abs(u)
                if au > seg_umax: seg_umax = au
                if au > 0.99 * U_MAX: seg_sat += 1
                if (k_global % DECIM) == 0:
                    trace_y.append(y_tip); trace_u.append(u)
                    trace_x.append(x_tool); trace_pass.append(np_pass)
                if abs(y_tip) > 5e-3:
                    raise RuntimeError(f"divergence at pass {np_pass}, x={x_tool*1e3:.1f}mm")

            seg_rows.append(dict(n_pass=np_pass, x_mm=(x0 + SEG / 2) * 1e3,
                                 y_rms_um=float(np.sqrt(seg_y2 / steps_per_seg) * 1e6),
                                 u_max_V=float(seg_umax),
                                 sat_frac=float(seg_sat / steps_per_seg)))

            # --- material removal behind the tool + basis reprojection ---
            V_old = plate.V.copy()
            plate.remove_material(x0, x0 + SEG, BAND[0], BAND[1], PASS_DEPTH)
            refresh_rows(plate)
            T = plate.V.T @ (plate.M_free @ V_old)          # (n, n)
            q = T @ q; qd = T @ qd; qdd = T @ qdd
            hist = hist @ T.T
            Mp, Kp, Cp = modal_matrices(plate)
            Hpe = plate.H_Pe_modal.copy()

        print(f"  [{which}] pass {np_pass}/{N_PASS} done  "
              f"f1={plate.freq_n[0]:.1f} Hz  "
              f"pass y_rms={np.sqrt(np.mean([r['y_rms_um']**2 for r in seg_rows if r['n_pass']==np_pass])):.3f} um  "
              f"({time.time()-t_wall:.0f}s)", flush=True)

    out = dict(controller=which, n_pass=N_PASS, seg_mm=SEG * 1e3,
               dt_us=DT * 1e6, pass_depth_mm=PASS_DEPTH * 1e3,
               segments=seg_rows,
               pass_summary=[dict(
                   n_pass=pp,
                   y_rms_um=float(np.sqrt(np.mean([r['y_rms_um'] ** 2
                                                   for r in seg_rows if r['n_pass'] == pp]))),
                   u_max_V=float(max(r['u_max_V'] for r in seg_rows if r['n_pass'] == pp)),
                   sat_frac=float(np.mean([r['sat_frac'] for r in seg_rows if r['n_pass'] == pp])))
                   for pp in range(1, N_PASS + 1)],
               f1_final_hz=float(plate.freq_n[0]),
               elapsed_s=time.time() - t_wall)
    json.dump(out, open(os.path.join(RESULTS, f"full_process_{which}.json"), "w"), indent=2)
    np.savez(os.path.join(RESULTS, f"full_process_{which}_traces.npz"),
             y=np.array(trace_y), u=np.array(trace_u),
             x=np.array(trace_x), npass=np.array(trace_pass), decim=DECIM, dt=DT)
    print(f"  [{which}] TOTAL {out['elapsed_s']:.0f}s -> results/full_process_{which}.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["lqg", "adrc", "afc"], required=True)
    args = ap.parse_args()
    run(args.controller)
