# 03_analysis — Stability & Robustness Analysis

This directory contains tools for **post-simulation analysis**: SLD computation
and Monte Carlo robustness tests.

## Files

| File | Analysis | Method |
|---|---|---|
| `fdm_stability.py` | Stability Lobe Diagram | Per-mode SLD (`compute_SLD`) **and** rigorous closed-loop coupled monodromy — LQG adapter (`compute_closed_loop_SLD`) + **generic controller realization** (`compute_closed_loop_SLD_generic`) |
| `uncertainty_analysis.py` | Robustness | Monte-Carlo over an arbitrary controller set (`run_mc_controllers`) |
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

**FDM (Insperger-Stépán 2004)**: discretize τ into m_div subintervals, build the
augmented (delay-buffered) transition matrix product, evaluate eigenvalues.

## Closed-loop coupled monodromy SLD (rigorous, generic)

`closed_loop_rho_generic` / `compute_closed_loop_SLD_generic` embed an ARBITRARY
LTI output-feedback controller in realization form

```
ż = A_con z + B_con_y · y ,   u = -K_con z ,   y = D_obs · q      (z ∈ R^m)
```

in the FULL coupled, time-periodic delayed system — no "equivalent damping"
surrogate, no per-mode decoupling. LQG (m = 2n) goes through the
`closed_loop_rho` adapter; ESO-ADRC (m = 3n) supplies
`controller_realization()` directly:

```python
from fdm_stability import compute_closed_loop_SLD_generic
A_con, B_con_y, K_con = adrc.controller_realization()
rho = compute_closed_loop_SLD_generic(RPM_arr, ap_arr,
                                      omega_n, zeta, Dp_vec, H_Pe, D_obs,
                                      A_con, B_con_y, K_con,
                                      NT, RT, eta_h, phi_st, phi_ex,
                                      k1, k2, kt, hp, m_div=20)
```

`A_con=None` gives the coupled open-loop grid. The same function powers the
**design-time certification** of A-ESO-ADRC's robust rung in
`main_simulation.py` (worst-case ρ over a frequency-mismatch × depth × tool-
position ball). Note: ρ values near 1 are marginal and m_div-sensitive — the
certification is a comparative selection criterion, always cross-checked by
time simulation, not a formal stability proof.

## Monte-Carlo robustness (generic controller set)

```python
from uncertainty_analysis import run_mc_controllers
st = run_mc_controllers(plant_nominal,
                        {"LQG": lqg, "ESO-ADRC": adrc, "A-ESO-ADRC": aadrc},
                        nominal_params, kp_idx, dt, T_end, ft, tau, n_per,
                        unc=dict(kt_pct=0.15, kn_pct=0.15, mu_c_pct=0.15,
                                 omega_pct=0.03, zeta_pct=0.20, E_pct=0.0),
                        n_samples=50, meas_noise_std=1e-8)
# st['n_conv'][name]        -> convergence counts (divergence NOT hidden)
# st['gain_stats'][name]    -> median/p05/p95 gain vs the FIRST (baseline) entry
```

The first dict entry is the baseline for pairwise gains. Controllers exposing
`reset_adaptation()` (A-ESO-ADRC) are reset before every sample, so adaptation
runs but never carries state across samples. All controllers see the same
measurement-noise realisation. Driver: `python 05_main/main_robustness_mc.py`.

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
| Closed-loop monodromy SLD (30×25 grid, 3 modes, m_div=20, 3 positions × 3 configs) | ~35 s |
| Certification grid (18 designs × 12 ball points, 5-mode plant) | ~2 min |
| Monte-Carlo (50 samples, 0.5 s each, ×3 controllers) | ~3 min |
| Mesh-convergence study | ~10 s |
