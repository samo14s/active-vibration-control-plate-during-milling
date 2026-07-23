"""
bifurcation_analysis.py
=======================
NONLINEAR-REGIME / LIMIT-CYCLE BIFURCATION STUDY
================================================

Research gap addressed (#3)
---------------------------
The plant is advertised as a *nonlinear* NDDE (Von Kármán geometric cubic +
3rd-order regenerative cutting + tool–workpiece separation, after Nasiri &
Moradi 2025).  But the README itself notes the nonlinear terms are

    "dormant at the nominal µm amplitudes (the model then reduces exactly to the
     previous linear backbone)."

We verified this numerically: at the reported operating point LQG on the
*nonlinear* plant and on the *linear* plant give identical RMS to 4 decimals
(0.6141 µm, Δ ≈ 3·10⁻³ %).  So **every headline result in the article is
effectively a linear-plant result** and the elaborate nonlinear model is never
exercised — a reviewer will ask "why model the nonlinearity at all?"

What this module does
---------------------
It drives the plant into the regime where the nonlinearity is the whole story —
finite-amplitude *chatter* — and quantifies it:

  * **Limit-cycle branch** A(ap): the steady-state chatter amplitude vs depth of
    cut.  Below the Hopf point the amplitude is ≈0 (dormant nonlinearity); above
    it a **bounded limit cycle** appears whose amplitude is set *entirely* by the
    nonlinear terms.
  * **Nonlinear vs linearized plant**: the linear model **diverges** past the
    stability limit, while the nonlinear model **saturates** into a bounded limit
    cycle — the physically essential effect (e.g. at ap = 1.0 mm: nonlinear
    ≈ 3.1 mm p2p bounded, linear → ∞).  This is the justification for the
    nonlinear model that the article never shows.
  * **Controlled bifurcation**: how LQG / repetitive-LQG push the Hopf point to
    much larger ap and cut the limit-cycle amplitude — i.e. control acting on the
    genuine nonlinear dynamics, not the dormant linear backbone.

Method note
-----------
The bifurcation is taken at a **fixed tool position** (constant engagement), so
ap enters only through the cutting-force integration limits; the modal model
(and hence any controller) is built once.  The steady-state amplitude is the RMS
/ peak-to-peak of ``y`` after the start-up transient, from the package's own
nonlinear Newmark integrator (physics identical to the article).

References
----------
- G. Stépán, T. Kalmár-Nagy, "Nonlinear regenerative machine tool vibrations",
  ASME DETC (1997).
- K. Nasiri, H. Moradi, MSSP 224 (2025) 112198  (the plant model).
"""
import numpy as np

import _pkg_path  # noqa: F401
from milling_force import precompute_nonlinear_periodic
from newmark_solver import NewmarkSimulator


def limit_cycle_amplitude(plate, ctrl, ap, cutting, dt=5e-5, T=0.35,
                          t_settle=0.20, kp_fixed=1000, nonlinear=True,
                          sensor_noise=0.0, diverge_thresh=1.0e-2):
    """
    Steady-state chatter amplitude at a fixed operating point (ap).

    Returns dict with RMS [µm], peak-to-peak [µm], a ``diverged`` flag and the
    end time reached (short end time ⇒ the run tripped the divergence guard, i.e.
    an unbounded/linear blow-up).

    ``cutting`` bundles the constant milling parameters (see ``make_cutting``).
    ``nonlinear=False`` disables the cubic cutting and the Von Kármán term to
    expose the *linearised* behaviour for comparison.
    """
    sim = NewmarkSimulator(plate, dt=dt, T_end=T, ft=cutting['ft'],
                           tau=cutting['tau'], verbose=False)
    kp = np.full(sim.nstep, kp_fixed, dtype=int)
    a3, a4, a42, a43 = precompute_nonlinear_periodic(
        dt, cutting['n_per'], sim.nstep, cutting['Omega'], cutting['NT'],
        cutting['RT'], cutting['eta'], cutting['phi_st'], cutting['phi_ex'],
        plate.hp - ap, plate.hp, cutting['k1'], cutting['k2'], cutting['kt'])

    if nonlinear:
        a42_use, a43_use, lam = a42, a43, None
    else:
        a42_use, a43_use, lam = None, None, np.zeros(plate.n_modes)

    if ctrl is not None and hasattr(ctrl, 'reset'):
        ctrl.reset()
    for h in ('history_u_lqg', 'history_u_ff', 'history_u_total',
              'history_phase', 'history_safety'):
        if hasattr(ctrl, h):
            setattr(ctrl, h, [])

    res = sim.simulate(a3, a4, kp, controller=ctrl,
                       alpha4_2_t=a42_use, alpha4_3_t=a43_use, lam_modal=lam,
                       sensor_noise=sensor_noise,
                       sensor_rng=np.random.default_rng(1),
                       progress=False, stop_threshold=diverge_thresh)

    i = res['stop_idx']
    k0 = int(t_settle / dt)
    t_end = res['t'][i]
    diverged = t_end < T - 5 * dt          # tripped the guard before T
    if i <= k0 + 10:
        return dict(rms=np.nan, p2p=np.nan, umax=np.nan,
                    diverged=diverged, t_end=t_end)
    y = res['y'][k0:i + 1]
    u = res['u'][:i + 1]
    return dict(rms=float(np.sqrt(np.mean(y**2)) * 1e6),
                p2p=float((y.max() - y.min()) * 1e6),
                umax=float(np.max(np.abs(u))),
                diverged=diverged, t_end=t_end)


def make_cutting(RPM, NT, RT, eta, phi_st, phi_ex, k1, k2, kt, ft, dt):
    """Bundle the constant milling parameters used by the bifurcation sweep."""
    tau = 60.0 / (NT * RPM)
    return dict(RPM=RPM, NT=NT, RT=RT, eta=eta, phi_st=phi_st, phi_ex=phi_ex,
                k1=k1, k2=k2, kt=kt, ft=ft, tau=tau,
                Omega=2 * np.pi * RPM / 60.0, n_per=int(round(tau / dt)))


def bifurcation_diagram(plate, ctrl, ap_array, cutting, nonlinear=True,
                        sensor_noise=0.0, dt=5e-5, T=0.35, verbose=True,
                        label=""):
    """
    Sweep ap and return the steady-state amplitude branch.

    Returns dict of arrays: ap, rms, p2p, umax, diverged.
    """
    rms = np.full(len(ap_array), np.nan)
    p2p = np.full(len(ap_array), np.nan)
    umax = np.full(len(ap_array), np.nan)
    div = np.zeros(len(ap_array), dtype=bool)
    for j, ap in enumerate(ap_array):
        out = limit_cycle_amplitude(plate, ctrl, ap, cutting,
                                    dt=dt, T=T, nonlinear=nonlinear,
                                    sensor_noise=sensor_noise)
        rms[j], p2p[j], umax[j], div[j] = (out['rms'], out['p2p'],
                                           out['umax'], out['diverged'])
        if verbose:
            tag = "diverged" if out['diverged'] else "bounded"
            print(f"   [{label}] ap={ap*1e3:5.2f} mm  RMS={out['rms']:9.2f} µm "
                  f"p2p={out['p2p']:9.2f} µm  {tag}")
    return dict(ap=np.asarray(ap_array), rms=rms, p2p=p2p, umax=umax,
                diverged=div)


def hopf_point(branch, amp_thresh_um=1.0):
    """
    Estimate the chatter-onset depth of cut (Hopf point) from a from-rest branch:
    the smallest ap whose steady amplitude exceeds ``amp_thresh_um`` [µm].
    """
    ap, rms = branch['ap'], branch['rms']
    for a, r in zip(ap, rms):
        if np.isfinite(r) and r > amp_thresh_um:
            return float(a)
    return float(ap[-1])
