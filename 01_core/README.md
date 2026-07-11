# 01_core — Physical Models

This directory contains the **physical model** of the system: the FEM plate,
piezo actuator, cutting force, and time integration solver.

## Files

| File | Description | Key class/function |
|---|---|---|
| `kirchhoff_q4.py` | Q4 plate element (Kirchhoff theory) | `compute_K_elem()`, `compute_M_elem()` |
| `plate_model.py` | Full plate FEM + modal reduction + piezo | `PlateModel` class |
| `piezo_actuator.py` | QDA60-200.7 piezo model | `PiezoActuator` class |
| `milling_force.py` | 3-tooth cutting force coefficients | `precompute_alpha_periodic()` |
| `newmark_solver.py` | Newmark-β time integration | `NewmarkSimulator` class |

## Dependencies

- `kirchhoff_q4.py` → no internal deps
- `plate_model.py` → uses `kirchhoff_q4.py`, `piezo_actuator.py`
- `piezo_actuator.py` → no internal deps
- `milling_force.py` → no internal deps
- `newmark_solver.py` → uses `plate_model.py`, `milling_force.py`

## Typical usage

```python
from plate_model import PlateModel
from milling_force import precompute_alpha_periodic
from newmark_solver import NewmarkSimulator

# Build plate
plate = PlateModel(LP=0.1, HP=0.08, BP=0.004,
                   rho=2830, E=69e9, nu=0.33,
                   N1=30, N2=24, n_modes=3,
                   zeta_modes=[0.0031, 0.0017, 0.0027])
plate.precompute_Dp(zp_pos=0.0795, n_pos=2001)
plate.add_piezo_patch(0, 0.020, 0, 0.060,
                       d31=175e-12, h_Pa=0.7e-3,
                       E_Pe=63e9, nu_Pe=0.35)

# Compute cutting force coefficients
a3, a4 = precompute_alpha_periodic(...)

# Run simulation
sim = NewmarkSimulator(plate, dt=5e-5, T_end=0.5,
                        ft=2e-5, tau=tau)
results = sim.simulate(a3, a4, kp_idx, controller=my_controller)
```

## Key parameters

- **N1, N2**: mesh density (30 × 24 = 720 elements is sufficient)
- **n_modes**: number of modes retained (3 captures dominant dynamics)
- **dt**: time step (5×10⁻⁵ s for high-resolution; 1×10⁻⁴ s for full path)
