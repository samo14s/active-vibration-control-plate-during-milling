"""
augmentation_study.py
=====================
Honest negative results: two natural augmentations of the delay-tuned ADRC are
evaluated with the closed-loop semi-discretization (observer in the monodromy)
and shown NOT to improve the linear chatter boundary at the nominal placement.

(1) Regeneration-aware delayed channel (RA-ADRC):
        u = u_ADRC + g * [ y(t - tau) - y(t) ]
    The bracket is the dynamic-chip-thickness-like signal and tau is known
    exactly from the spindle speed.  Scanning g on the true Floquet criterion
    shows only a marginal gain (< 1 %) before rapid destabilisation: the ESO
    already reconstructs the total disturbance with lag ~1/wo << tau, and the
    delayed channel injects wrong-phase feedback at all other frequencies.

(2) Resonant (frequency-shaped) ESO:
        disturbance model  d = f0 + f1,  f1'' + 2 zr wr f1' + wr^2 f1 = 0
    embedding an internal model at/near the dominant mode.  Over a grid of
    (wr, zr, observer pole patterns) the best case matches -- but does not
    beat -- the plain 3rd-order ESO, because the chatter frequency shifts with
    depth and lobe number, so a fixed internal model provides no robust margin.

Conclusion recorded in the manuscript: at fixed transducer hardware the plain
bandwidth-parameterised ESO with delay-aware tuning is already near the ceiling
of single-channel output-feedback augmentations; the remaining lever is the
transducer placement itself (see placement_study.py) -- and there, the binding
constraint is the actuator-voltage budget, not the linear boundary.

Outputs: results/augmentation.json
Run:     python experiments/augmentation_study.py
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
from adrc_control import ADRCController
from cl_fdm import ClosedLoopFDM, default_cutting
from scipy.signal import place_poles

LP, HP = 0.100, 0.080
RPM = 4900; DT = 5e-5
ZETA = [0.0031, 0.0017, 0.0027]
WC, WO = 1500.0, 18000.0            # delay-tuned plain ADRC (nominal placement)


def build():
    p = PlateModel(LP, HP, 0.004, 2830, 69e9, 0.33, N1=30, N2=24, n_modes=3,
                   zeta_modes=ZETA, verbose=False)
    p.precompute_Dp(zp_pos=HP - 0.3e-3 / 2, n_pos=2001)
    p.set_observation(x_obs=LP, z_obs=HP)
    p.D_tip = p.D_obs.copy()
    Nv = shape_at_point(0.02, 0.06, p.N1, p.N2, p.lex, p.ley, p.n1)
    p.D_obs = Nv[p.DOFf] @ p.V
    p.add_piezo_patch(0, 0.020, 0, 0.060, 175e-12, 0.7e-3, 63e9, 0.35)
    return p


def study_delayed_channel(p, clf, adrc):
    """RA-ADRC: scan the delayed-difference gain g on the Floquet criterion."""
    print("=== (1) RA-ADRC delayed channel u_tau = g*[y(t-tau)-y(t)] ===")
    Ac, Bc, Cc, Dc = adrc.export_lti()
    Bo = adrc.B_o.reshape(3, 1)
    rows = []
    for g in [0.0, -3e6, -1e6, -3e5, 3e5, 1e6, 3e6, 1e7]:
        Bc_g = Bc + Bo * (-g)              # ESO sees the extra -g*y through u
        Bc_tau = Bo * g
        Dc_g = np.array([[-g]])
        D_tau = np.array([[g]])
        ap = clf.ap_crit_dynamic(RPM, Ac, Bc_g, Cc, Dc_g,
                                 Bc_tau=Bc_tau, D_tau=D_tau, n_coarse=30)
        rows.append(dict(g=g, ap_crit_mm=ap * 1e3))
        print(f"  g={g:+.1e}  a_p,crit={ap*1e3:7.3f} mm")
    return rows


def study_resonant_eso(p, clf):
    """Resonant-ESO ADRC: disturbance model bias + resonant pair at wr."""
    print("\n=== (2) Resonant-ESO ADRC ===")
    b0 = float(p.D_obs @ p.H_Pe_modal)
    w1 = p.omega_n[0]
    rows = []
    for wc, wo in [(1500, 18000), (2000, 20000)]:
        for om_fac in [0.8, 1.0, 1.2]:
            for zr in [0.05, 0.3]:
                wr = om_fac * w1
                Aw = np.array([[0, 1, 0, 0, 0],
                               [0, 0, 1, 1, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1],
                               [0, 0, 0, -wr ** 2, -2 * zr * wr]], float)
                Bw = np.array([0, b0, 0, 0, 0], float)
                Cw = np.array([[1, 0, 0, 0, 0]], float)
                poles = -wo * np.array([1, 1.01, 1.02, 1.03, 1.04])
                try:
                    L = place_poles(Aw.T, Cw.T, poles).gain_matrix.T
                    Cc = np.array([[-wc ** 2, -2 * wc, -1, -1, 0]]) / b0
                    Ac = (Aw - L @ Cw) + np.outer(Bw, Cc.ravel())
                    ap = clf.ap_crit_dynamic(RPM, Ac, L, Cc, np.zeros((1, 1)),
                                             n_coarse=30)
                except Exception:
                    ap = 0.0
                rows.append(dict(wc=wc, wo=wo, omega_r=wr, zeta_r=zr,
                                 ap_crit_mm=ap * 1e3))
                print(f"  wc={wc} wo={wo} wr={om_fac:.1f}*w1 zr={zr}: "
                      f"a_p,crit={ap*1e3:7.3f} mm")
    return rows


if __name__ == "__main__":
    t0 = time.time()
    p = build()
    clf = ClosedLoopFDM.from_plate(p, default_cutting(hp=HP), m_div=30)
    adrc = ADRCController(p, dt=DT, wc=WC, wo=WO)
    Ac, Bc, Cc, Dc = adrc.export_lti()
    base = clf.ap_crit_dynamic(RPM, Ac, Bc, Cc, Dc, n_coarse=30)
    print(f"baseline plain ADRC (wc={WC:.0f}, wo={WO:.0f}): {base*1e3:.3f} mm\n")
    d_rows = study_delayed_channel(p, clf, adrc)
    r_rows = study_resonant_eso(p, clf)
    best_d = max(r["ap_crit_mm"] for r in d_rows)
    best_r = max(r["ap_crit_mm"] for r in r_rows)
    out = dict(baseline_ap_crit_mm=base * 1e3,
               delayed_channel=d_rows, resonant_eso=r_rows,
               best_delayed_mm=best_d, best_resonant_mm=best_r,
               conclusion=("Neither augmentation improves the linear boundary "
                           "materially over the plain delay-tuned ESO "
                           f"(baseline {base*1e3:.2f} mm, best delayed {best_d:.2f} mm, "
                           f"best resonant {best_r:.2f} mm)."),
               elapsed_s=time.time() - t0)
    json.dump(out, open(os.path.join(RESULTS, "augmentation.json"), "w"), indent=2)
    print(f"\nbest delayed={best_d:.3f} mm, best resonant={best_r:.3f} mm "
          f"(baseline {base*1e3:.3f})")
    print(f"TOTAL {time.time()-t0:.1f}s -> results/augmentation.json")
