# 03_analysis — Stability & Robustness Analysis

| File | Analysis | Method |
|---|---|---|
| `fdm_stability.py` | Stability Lobe Diagram | Floquet multipliers (semi-discretization; NEW: full closed-loop monodromy) |
| `uncertainty_analysis.py` | Robustness harness (Monte Carlo) | **Available tool — NOT executed in any reported result** |

## Stability Lobe Diagram

Two tools coexist:

### 1. `compute_SLD_closed_loop` — the honest tool (use this)

Floquet analysis of the COMPLETE system: coupled multi-mode plant
(rank-1 regenerative coupling Dp·Dpᵀ) + delay + — optionally — the ACTUAL
digital LQG compensator (Kalman observer + discrete state feedback)
augmented into the monodromy matrix.

```python
from fdm_stability import compute_SLD_closed_loop

rho_OL  = compute_SLD_closed_loop(RPM_arr, ap_arr, plate, NT, RT, eta_h,
                                  phi_st, phi_ex, k1, k2, kt, hp, Dp_vec,
                                  lqg=None,  dt_c=5e-5)      # open loop
rho_LQG = compute_SLD_closed_loop(..., lqg=lqg_controller)   # closed loop
# Stability boundary: contour rho == 1.0
```

Notes:
- **No separate DARC curve exists**: a phase-locked periodic feedforward is
  an exogenous input; it does not change the homogeneous variational
  equation, so the Floquet multipliers — and the SLD — of DARC are exactly
  those of its LQG base. (The former hard-coded 1.30× "effective damping"
  multiplier was removed as unfounded.)
- LINEAR analysis: the ±150 V saturation is excluded — the boundary is only
  meaningful where the required voltage stays below the limit.
- Tool coupling uses the path-averaged Dp by convention in the study
  scripts; the antisymmetric torsion mode (mode 2) averages to ZERO there —
  a worst-case-position SLD is a known open item (see audit, P1).
- If the boundary does not cross ρ=1 inside the ap grid, report the grid
  top as a LOWER BOUND ("> x mm"), never as the critical depth.

### 2. `compute_SLD` — legacy per-mode tool

Zeroth-order semi-discretization of each mode SEPARATELY (SISO, no
compensator, no mode coupling). Kept for reference; do not use it to make
closed-loop claims.

## Monte Carlo Robustness — `uncertainty_analysis.py`

**Status: the harness is implemented but is NOT called by any script in the
package; no reported figure or table is a Monte Carlo result.** The
implemented protocol is the correct one (controller designed on the nominal
model, run unchanged on each perturbed plant, fixed seed), so executing it
is a ready P1 item.

Actual API and defaults (do not cite other values):

```python
from uncertainty_analysis import run_monte_carlo
# defaults: n_samples=50; uniform ranges: kt/kn/mu_c ±5 %, omega_n ±2 %,
# zeta ±20 %.  envelope_stats() returns mean/std/min/max and the 5th/95th
# sample percentiles (a prediction envelope — NOT a 95 % confidence
# interval of an estimator).
```

## Reference

- T. Insperger, G. Stépán, "Updated semi-discretization method for periodic
  delay-differential equations with discrete delay",
  *Int. J. Numer. Meth. Eng.* 61 (2004) 117–141. (The legacy per-mode tool
  is the zeroth-order 2002 variant; the closed-loop tool follows the same
  semi-discretization logic with the compensator states augmented.)

## Performance (measured on this package)

| Operation | Time |
|---|---:|
| Closed-loop Floquet, one (RPM, a_p) point | ~10 ms |
| Full closed-loop SLD (30 × 25 grid) | ~10 s |
