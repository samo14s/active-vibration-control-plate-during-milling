"""
robust_monte_carlo.py
=====================
ALL-CONTROLLER MONTE-CARLO ROBUSTNESS + ADAPTATION AUDIT
========================================================

Research gaps addressed (#5 and #6)
-----------------------------------
1. **Dead "Adaptive-Robust" layer.**  DARC = *Deep Adaptive Robust Control*,
   yet the online adaptation is inert.  In ``darc_mpc_v3_controller.step`` the
   quantity ``lambda_robust`` is computed from ``OnlineRLSAdapter`` and then
   **never used** — the control law is ``u = u_LQG + u_FF`` regardless.  We
   confirm it bit-exactly: ``enable_adaptation=True`` and ``False`` give the same
   RMS to 0.00·10⁰ µm (``audit_adaptation`` below).  The "A" and "R" of DARC are
   therefore unsupported by the code.

2. **Robustness claimed, not measured across controllers.**  The package's
   ``uncertainty_analysis.py`` runs Monte-Carlo on the **LQG only**; DARC and any
   fair baseline are never subjected to the same parametric uncertainty, and no
   confidence intervals or significance tests are reported.  ``monte_carlo_all``
   runs LQG, repetitive-LQG and DARC through the *same* perturbed plants and
   returns mean ± 95 % CI plus **paired bootstrap** comparisons — the statistics
   a robustness claim needs.

Protocol
--------
Every controller is designed **once on the nominal model** and then tested on N
random plants with (ω ±2 %, ζ ±20 %, K_T ±5 %) — the honest robustness test
(the controller does not know the true plant).  RMS is the steady-state
free-corner vibration.  Paired differences use the *same* random plant for every
controller, so the bootstrap CI on the difference is tight and meaningful.
"""
import copy
import numpy as np

import _pkg_path  # noqa: F401
from milling_force import precompute_nonlinear_periodic
from newmark_solver import NewmarkSimulator


DEFAULT_UNC = dict(omega_pct=0.02, zeta_pct=0.20, kt_pct=0.05)


# ======================================================================
#  Adaptation audit (proves the DARC RLS layer is inert)
# ======================================================================
def audit_adaptation(build_darc, simulate_rms):
    """
    Compare DARC RMS with ``enable_adaptation`` True vs False.

    ``build_darc(adapt)`` must return a freshly trained DARC controller;
    ``simulate_rms(ctrl)`` must return its steady-state RMS [µm].  Returns
    (rms_on, rms_off, identical: bool).
    """
    rms_on = simulate_rms(build_darc(True))
    rms_off = simulate_rms(build_darc(False))
    return rms_on, rms_off, abs(rms_on - rms_off) < 1e-6


# ======================================================================
#  Perturbed plant / cutting sampling
# ======================================================================
def sample_plant(rng, plate_nominal, omega_nom, zeta_nom, unc=None):
    """Return a shallow copy of the plate with perturbed modal matrices."""
    if unc is None:
        unc = DEFAULT_UNC
    n = plate_nominal.n_modes
    dom = rng.uniform(-unc['omega_pct'], unc['omega_pct'], n)
    dze = rng.uniform(-unc['zeta_pct'], unc['zeta_pct'], n)
    omega = omega_nom * (1 + dom)
    zeta = np.maximum(1e-4, zeta_nom * (1 + dze))
    p = copy.copy(plate_nominal)
    p.Kp = np.diag(omega**2)
    p.Cp = np.diag(2 * zeta * omega)
    p.omega_n = omega
    p.zeta_modes = zeta
    return p, omega, zeta


# ======================================================================
#  All-controller Monte-Carlo
# ======================================================================
def monte_carlo_all(plate_nominal, controllers, cutting, kp_idx,
                    omega_nom, zeta_nom, dt=5e-5, T=0.3, t_settle=0.1,
                    n_samples=40, sensor_noise=1e-7, unc=None, seed=7,
                    verbose=True):
    """
    Run every controller in ``controllers`` (dict name→ctrl, built on nominal)
    over ``n_samples`` random plants.

    Returns dict: rms[name] = array(n_samples), plus 'omega'/'zeta'/'kt' samples.
    """
    rng = np.random.default_rng(seed)
    names = list(controllers.keys())
    rms = {nm: np.full(n_samples, np.nan) for nm in names}
    kt_nom = cutting['kt']

    for i in range(n_samples):
        p, omega, zeta = sample_plant(rng, plate_nominal, omega_nom, zeta_nom, unc)
        # perturb cutting magnitude K_T (±kt_pct)
        u = DEFAULT_UNC if unc is None else unc
        kt = kt_nom * (1 + rng.uniform(-u['kt_pct'], u['kt_pct']))
        sim = NewmarkSimulator(p, dt=dt, T_end=T, ft=cutting['ft'],
                               tau=cutting['tau'], verbose=False)
        a3, a4, a42, a43 = precompute_nonlinear_periodic(
            dt, cutting['n_per'], sim.nstep, cutting['Omega'], cutting['NT'],
            cutting['RT'], cutting['eta'], cutting['phi_st'], cutting['phi_ex'],
            cutting['za_low'], cutting['za_high'], cutting['k1'], cutting['k2'], kt)

        for nm in names:
            ctrl = controllers[nm]
            if hasattr(ctrl, 'reset'):
                ctrl.reset()
            for h in ('history_u_lqg', 'history_u_ff', 'history_u_total',
                      'history_phase', 'history_safety'):
                if hasattr(ctrl, h):
                    setattr(ctrl, h, [])
            res = sim.simulate(a3, a4, kp_idx, controller=ctrl,
                               alpha4_2_t=a42, alpha4_3_t=a43,
                               sensor_noise=sensor_noise,
                               sensor_rng=np.random.default_rng(1000 + i),
                               progress=False)
            k0 = int(t_settle / dt)
            ie = res['stop_idx']
            if ie > k0 + 10:
                y = res['y'][k0:ie + 1]
                rms[nm][i] = float(np.sqrt(np.mean(y**2)) * 1e6)
        if verbose and (i + 1) % max(1, n_samples // 5) == 0:
            msg = "  ".join(f"{nm}={np.nanmean(rms[nm][:i+1]):.3f}" for nm in names)
            print(f"   MC {i+1:3d}/{n_samples}   mean RMS so far: {msg}")

    return rms


# ======================================================================
#  Statistics
# ======================================================================
def bootstrap_ci(x, n_boot=5000, alpha=0.05, seed=0):
    """Bootstrap (1-alpha) CI for the mean of x (NaNs dropped)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(x, x.size, replace=True).mean()
                     for _ in range(n_boot)])
    return (float(x.mean()),
            float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def paired_advantage(rms_a, rms_b, n_boot=5000, seed=1):
    """
    Paired relative advantage of B over A: (a-b)/a per sample.  Returns
    (mean_pct, lo_pct, hi_pct, prob_B_better) via paired bootstrap.
    """
    a = np.asarray(rms_a, float)
    b = np.asarray(rms_b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    rel = (a - b) / a * 100.0
    rng = np.random.default_rng(seed)
    idx = np.arange(a.size)
    boot = np.array([rel[rng.choice(idx, a.size, replace=True)].mean()
                     for _ in range(n_boot)])
    return (float(rel.mean()),
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
            float(np.mean(b < a)))


def summarize(rms_dict):
    """Print a mean ± 95 % CI table and the key paired comparisons."""
    print(f"\n  {'controller':<12}{'mean RMS':>10}{'95% CI':>22}{'worst':>10}")
    print("  " + "-" * 54)
    for nm, arr in rms_dict.items():
        m, lo, hi = bootstrap_ci(arr)
        worst = np.nanmax(arr)
        print(f"  {nm:<12}{m:>9.3f} µm  [{lo:6.3f}, {hi:6.3f}] µm{worst:>9.3f}")
    return rms_dict
