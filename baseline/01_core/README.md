# 01_core — Physical Models (Mindlin edition)

This directory contains the **physical model** of the system: the FEM plate,
piezo actuator, cutting force, and time integration solver.

> **Plate theory: Reissner–Mindlin.** In this edition the plate FEM is the
> **8-node Serendipity Mindlin element**, ported *literally* from the MATLAB
> repository `Plate-FEM/Mindlin_plate`. See [`MINDLIN_PORT.md`](MINDLIN_PORT.md)
> for the exact `.m` ↔ `.py` correspondence. `plate_model.py` keeps the **same
> public `PlateModel` interface** as the original Kirchhoff version, so the rest
> of the package (controllers, Newmark solver, FDM stability, figures) is
> unchanged — only the underlying plate theory differs (Kirchhoff → Mindlin).

## Files

| File | Description | Key class/function |
|---|---|---|
| `mindlin_q8.py` | **Reissner–Mindlin** Serendipity Q8 plate element (3 DOF/node: `w, θx, θy`) — literal port of Plate-FEM | `stiffness_matrix_M()`, `mass_matrix_M()` |
| `plate_model.py` | Full plate FEM + modal reduction + piezo (now Mindlin) | `PlateModel` class |
| `kirchhoff_q4.py` | Q4 plate element (Kirchhoff theory) — kept for reference/comparison, **not used** by `plate_model.py` | `stiffness_matrix_K()`, `mass_matrix_K()` |
| `piezo_actuator.py` | QDA60-200.7 piezo model | `PiezoActuator` class |
| `milling_force.py` | 3-tooth cutting force coefficients | `precompute_alpha_periodic()` |
| `newmark_solver.py` | Newmark-β time integration | `NewmarkSimulator` class |

## Dependencies

- `mindlin_q8.py` → no internal deps
- `plate_model.py` → uses `mindlin_q8.py`
- `kirchhoff_q4.py` → no internal deps (reference only)
- `piezo_actuator.py` → no internal deps
- `milling_force.py` → no internal deps
- `newmark_solver.py` → uses `plate_model.py`, `milling_force.py`

## Mindlin element in brief (`mindlin_q8.py`)

- **8-node Serendipity** quadrilateral, 3 DOF per node: transverse deflection
  `w` and two section rotations `θx, θy` → 24 DOF/element.
- **Bending** (`Bf`, `Hf`) and **transverse-shear** (`Bs`, `Hs`) strain
  contributions: `Ke = h·(Kf + Ks)`.
- **Consistent mass** with rotary inertia: `Ie = diag(1, h²/12, h²/12)`.
- **Uniform reduced 2×2 Gauss integration** for both bending and shear (shear-
  locking treatment) — exactly as in Plate-FEM. Shear correction factor
  `κ = 5/6`.
- Clamped edge → `w = θx = θy = 0` (cantilever: bottom edge `z = 0`).
- Piezo coupling: consistent Mindlin **moment analogy**
  `H_Pe = m_piezo · ∫_patch Bfᵀ [1,1,0]ᵀ dA` (same physical bending moment as
  the article's Kirchhoff `m_piezo · ∫∇²N dA`).

## Typical usage

```python
from plate_model import PlateModel          # now a Mindlin plate
from milling_force import precompute_alpha_periodic
from newmark_solver import NewmarkSimulator

# Build plate (identical call signature to the Kirchhoff version)
plate = PlateModel(lp=0.1, hp=0.08, bp=0.004,
                   rho=2830, E=69e9, nu=0.33,
                   N1=30, N2=24, n_modes=3,
                   zeta_modes=[0.0031, 0.0017, 0.0027],
                   kappa=5/6)               # shear-correction factor (Mindlin)
plate.precompute_Dp(zp_pos=0.0795, n_pos=2001)
plate.set_observation(x_obs=0.1, z_obs=0.08)
plate.add_piezo_patch(0, 0.020, 0, 0.060,
                       d31=175e-12, h_Pa=0.7e-3,
                       E_Pe=63e9, nu_Pe=0.35)

# Compute cutting force coefficients
a3, a4 = precompute_alpha_periodic(...)

# Run simulation
sim = NewmarkSimulator(plate, dt=5e-5, T_end=0.5, ft=2e-5, tau=tau)
results = sim.simulate(a3, a4, kp_idx, controller=my_controller)
```

## Key parameters

- **N1, N2**: mesh density (30 × 24 = 720 Serendipity elements is sufficient)
- **n_modes**: number of modes retained (3 captures dominant dynamics)
- **kappa**: shear-correction factor (5/6 for isotropic plates)
- **dt**: time step (5×10⁻⁵ s for high-resolution; 1×10⁻⁴ s for full path)

## Validation (see `../tests/`)

| Benchmark | Result | Reference |
|---|---|---|
| CCCC thin square plate, λ₁ = ω a²√(ρh/D) | **35.98** | 35.99 (Leissa) — err −0.02% |
| CCCC modes 2–6 | 73.39, 73.39, 108.27, 131.59, 132.25 | 73.41, 73.41, 108.27, 131.64, 132.24 |
| Cantilever AL6061 (article geometry) mode 1 | **519.4 Hz** | ~521 Hz (article) / 528 Hz (beam strip) |
| Drop-in with package-12 LQG + Newmark | runs, LQG reduces vibration | — |
