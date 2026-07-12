# 01_core — Physical Models

This directory contains the **physical model** of the system: the analytical
plate model, piezo actuator, cutting force, and time integration solver.

## Files

| File | Description | Key class/function |
|---|---|---|
| `kirchhoff_q4.py` | LEGACY Q4 plate element (Kirchhoff/ACM, non-conforming). Only used historically to produce the calibration anchors [521.06, 1069.95, 2733.02] Hz. | `compute_K_elem()`, `compute_M_elem()` |
| `plate_model.py` | Analytical Galerkin modal plate (beam-function products), frequencies calibrated to the legacy FEM anchors, Von Kármán cubic (energy form), piezo modal coupling | `PlateModel` class |
| `piezo_actuator.py` | QDA60-200.7 actuator/sensor realism wrapper (saturation, slew, amplifier lag, linear material phase-lag, sensor noise + delay) | `PiezoActuator` class |
| `milling_force.py` | 3-tooth cutting force coefficients (linear backbone + heuristic quadratic/cubic rescaling) | `precompute_alpha_periodic()`, `precompute_nonlinear_periodic()` |
| `newmark_solver.py` | Newmark-β time integration of the delayed modal equation | `NewmarkSimulator` class |

## Honest-model notes (read before writing the thesis chapter)

- **Mode shapes vs frequencies**: the analytical single-term Rayleigh basis
  over-predicts modes 2–3 by 8–10 % (calibration factors 0.987/0.918/0.898);
  the frequencies are overwritten by the FEM anchors while the SHAPES (hence
  `D_obs`, `Dp`, `H_Pe`) remain analytical. This hybrid must be stated.
- **Von Kármán cubic** `lam_modal` uses the variationally consistent
  membrane-ENERGY form under u₀=v₀=0 (positive by construction). It is an
  UPPER BOUND of the real hardening (no in-plane relief). Cross-mode cubic
  couplings are neglected (diagonal truncation).
- **Dormant at reported amplitudes**: at the µm levels of every controlled
  result the cubic and the quadratic/cubic cutting terms are numerically
  negligible — the reported comparisons live on the linear backbone.
- **Open-loop chatter amplitudes are NOT quantitative**: beyond the
  stability limit the "limit cycle" is set by the numerical clamp
  `chip_sat` (default 10·f_t) and the separation thresholds, not by a
  complete separation physics (no multiple regeneration 2τ, 3τ…). Use the
  stability BOUNDARY, never the saturated amplitude.
- **Not features of any reported result**: process damping
  (`set_process_damping`, never called → ζ_p = 0 everywhere), flank-wear
  edge force (`tool_wear_edge_force`, never called), and the Eq.-6
  separation helper (`chip_separation_factor`, re-implemented inline by the
  solver) are optional tools, not part of the simulated plant.
- **Piezo patch mass/stiffness** are not added to the plate model (the
  actuation force only). The ~6.5 g patch (~7 % of plate mass) shift is
  neglected; acknowledge it.
- The `PiezoActuator` "hysteresis" block is an exactly LINEAR first-order
  phase lag (~1° at the controlled modes) — no hysteresis loop. No
  temperature model exists.
- Material label: ρ = 2830 kg/m³ does not match AL6061 (2700 kg/m³); it is
  a 2024/7075-class density. Relabel the alloy or re-run with 2700.

## Typical usage

```python
from plate_model import PlateModel
from milling_force import precompute_nonlinear_periodic
from newmark_solver import NewmarkSimulator

# Build plate (analytical Galerkin, calibrated frequencies)
plate = PlateModel(lp=0.1, hp=0.08, bp=0.004,
                   rho=2830, E=69e9, nu=0.33,
                   n_modes=3, zeta_modes=[0.0031, 0.0017, 0.0027])
plate.precompute_Dp(zp_pos=0.0799, n_pos=2001)
plate.set_observation(x_obs=0.1, z_obs=0.08)
plate.add_piezo_patch(0, 0.020, 0, 0.060,
                      d31=175e-12, h_Pa=0.7e-3,
                      E_Pe=63e9, nu_Pe=0.35)

# Cutting force coefficients (linear + nonlinear rescalings)
a3, a4, a4_2, a4_3 = precompute_nonlinear_periodic(...)

# Run simulation
sim = NewmarkSimulator(plate, dt=5e-5, T_end=0.5, ft=2e-5, tau=tau)
results = sim.simulate(a3, a4, kp_idx, controller=my_controller,
                       alpha4_2_t=a4_2, alpha4_3_t=a4_3)
```

## Key parameters

- **n_modes**: number of modes retained (3 captures the dominant dynamics)
- **dt**: time step (5×10⁻⁵ s). NOTE: mode 3 (2733 Hz) then has only
  7.3 samples/period (~6 % Newmark period elongation) — a dt-convergence
  check (e.g. 10 µs) is advised before quantitative claims about mode 3.
- The regenerative delay is rounded to the grid (n_τ = round(τ/dt) = 82),
  i.e. the simulated spindle speed is effectively 4878 RPM, not 4900.
