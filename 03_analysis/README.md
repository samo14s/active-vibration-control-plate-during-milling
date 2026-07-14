# 03_analysis — Stability & Robustness Analysis

This directory contains tools for **post-simulation analysis**: SLD computation
and Monte Carlo robustness tests.

## Files

| File | Analysis | Method |
|---|---|---|
| `fdm_stability.py` | Stability Lobe Diagram | Per-mode SLD (`compute_SLD`) **and** rigorous closed-loop coupled monodromy (`compute_closed_loop_SLD`) |
| `uncertainty_analysis.py` | Robustness | Monte-Carlo LQG-vs-PALF (`run_mc_lqg_vs_palf`) |
| `mesh_convergence.py` | FEM verification | Natural-frequency convergence vs article Table 4 |

## Stability Lobe Diagram (FDM/Floquet)

**Theory**: The milling system with regenerative chatter is a **delay
differential equation (DDE)** with periodic coefficients:

```
q̈ + 2ζω·q̇ + ω²·q + α₄(t)·D_p²/m·q(t) = α₄(t)·D_p²/m·q(t-τ)
```

Where:
- `α₄(t)` = periodic cutting coefficient (period τ = 60/(N_T·RPM))
- `q(t-τ)` = regenerative term (previous tooth pass)

**Floquet theory**: Stability is determined by the **monodromy matrix Φ**
mapping state from t to t+τ:
- Stable if max|eig(Φ)| < 1
- Chatter if max|eig(Φ)| ≥ 1

**FDM (Insperger-Stépán 2004)**:
1. Discretize period τ into m_div subintervals (m_div = 40)
2. Build augmented state vector to handle delay
3. Compute Φ as product of m_div elementary transition matrices
4. Evaluate eigenvalues

```python
from fdm_stability import compute_SLD

rho_grid, _ = compute_SLD(
    RPM_array, ap_array,
    omega_n_list, zeta_list, Dp_list, m_list,
    NT, RT, eta_h, phi_st, phi_ex,
    k1, k2, kt, hp,
    m_div=40
)
# rho_grid.shape = (n_ap, n_RPM)
# Stability boundary: contour where rho_grid == 1.0
```

## Closed-loop coupled monodromy SLD (rigorous)

`compute_closed_loop_SLD` builds the semi-discretization monodromy of the FULL,
coupled, time-periodic delayed system with the LQG controller (state feedback +
Kalman observer) embedded — no "equivalent damping" surrogate, no per-mode
decoupling. Pass `A_ctrl=None` for the coupled open-loop system. Run it from
`main_simulation.py` (all three SLD panels use it). The PALF-LQG boundary equals the
LQG boundary rigorously, because the phase-only feedforward has `∂u_FF/∂x̂ = 0` and so
does not enter the closed-loop Jacobian.

## Monte-Carlo robustness (LQG vs PALF-LQG)

```python
from uncertainty_analysis import run_mc_lqg_vs_palf
st = run_mc_lqg_vs_palf(plant_nominal, lqg, palf, nominal_params, kp_idx,
                        dt, T_end, ft, tau, n_per,
                        unc=dict(kt_pct=0.15, omega_pct=0.03, zeta_pct=0.20),
                        n_samples=50, meas_noise_std=1e-8)
# st['n_conv_lqg'], st['n_conv_palf']  -> convergence counts (divergence NOT hidden)
# st['gain_median'], st['gain_p05/p95'], st['pct_palf_better'] over both-converged
```
Driver: `python 05_main/main_robustness_mc.py` (frozen held-out controllers; both run
with the same measurement-noise realisation; diverged samples are reported, not
dropped silently).

## FEM mesh convergence

`python 03_analysis/mesh_convergence.py` — shows the FEM frequencies are converged to
<0.1 % by a 30×24 mesh and reconciles the ~2.6 % offset vs the article's Chebyshev-Ritz
theory (a discretisation-model difference; the FEM is within 0.2–0.6 % of the *measured*
modes 2, 4, 5).

## Reference

- **FDM / semi-discretization**: T. Insperger, G. Stépán, "Updated semi-discretization
  method for periodic delay-differential equations with discrete delay",
  *Int. J. Numer. Meth. Eng.* 61 (2004) 117–141.

## Performance

| Operation | Time |
|---|---:|
| Closed-loop monodromy SLD (30×25 grid, 3 modes, m_div=20) | ~8 s |
| Monte-Carlo (50 samples, 0.25 s each, ×2 controllers) | ~1 min |
| Mesh-convergence study | ~10 s |
