# 03_analysis — Stability & Robustness Analysis

This directory contains tools for **post-simulation analysis**: SLD computation
and Monte Carlo robustness tests.

## Files

| File | Analysis | Method |
|---|---|---|
| `fdm_stability.py` | Stability Lobe Diagram | Floquet multipliers (FDM) |
| `uncertainty_analysis.py` | Robustness | Monte Carlo sampling |

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

## Monte Carlo Robustness

```python
from uncertainty_analysis import run_uncertainty_analysis

results = run_uncertainty_analysis(
    plate_nominal, controller_factory,
    n_samples=100,
    omega_perturb=0.15,    # ±15% on natural frequencies
    zeta_perturb=0.30,     # ±30% on damping
    KT_perturb=0.30,       # ±30% on cutting coefficient
)
# Returns: mean, std, percentiles for each metric
```

## Reference

- **FDM**: T. Insperger, G. Stépán, "Updated semi-discretization method for
  periodic delay-differential equations with discrete delay",
  *Int. J. Numer. Meth. Eng.* 61 (2004) 117–141.

## Performance

| Operation | Time |
|---|---:|
| Single (RPM, a_p) point with FDM | ~25 ms |
| Full SLD (60 × 50 grid, 3 modes) | ~80 s |
| Monte Carlo (100 samples, 0.5 s each) | ~50 s |
