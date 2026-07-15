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
| `material_removal.py` | **P6**: material-removal-aware time-varying FEM (per-element thickness field, MAC mode tracking, finishing sequence) | `MillingWorkpiece`, `finishing_sequence()` |

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


## Material removal (P6) — `material_removal.py`

The physically accurate, time-varying model of the workpiece as it is thinned by a
MULTI-PASS finishing sequence, replacing the phenomenological uniform
frequency-scale drift (`PlateModel.perturbed_copy` / the solver's `omega_scale_t`).

Honest physics (verified in `docs/REPRODUCED_RESULTS.md` P6):
- A single pass at a_p = 0.3 mm removes **0.0094 % of the volume** → ~0.000 %/mode
  drift, MAC = 1.0: within one pass the plant is essentially CONSTANT (the dominant
  within-pass variation is the spatial Dp(x), already modelled).
- Uniform full-face thinning gives **omega ∝ h exactly** (D ∝ h³, mass ∝ h), so the
  old uniform-scale heuristic is the correct BETWEEN-pass model; the article's
  9-17 % drift = 4-7 finishing layers (wall 4 → 3.3-3.6 mm).
- The genuinely new physics is **non-uniform** removal: thinning only part of the
  height gives per-mode drift of OPPOSITE signs and reshapes the modes (MAC < 1) —
  a single scale cannot capture it, which is what makes online ID necessary.

Efficiency: K_e = h³·K_unit, M_e = h·M_unit (uniform mesh), so one unit element
matrix is precomputed and each snapshot only rescales per element (0.06 s/eigensolve).
Mode tracking: each snapshot is MAC-matched and sign-aligned to a reference basis
so mode k stays mode k with consistent sign (0 sign flips over 24 snapshots).

P7 adds `remove_moving_front(x_tool, a_p, a_e)` — the physically accurate X-RESOLVED
within-pass thinning front (thins only the top band of height a_p, and only the
element-columns BEHIND the feeding tool), replacing the end-of-pass all-x
`remove_layer_band`. Call `begin_pass()` once at the start of a pass to snapshot the
baseline: the front then SETS `h = max(baseline − a_e, 1e-4)` behind the tool, so a_e
bites exactly ONCE per pass however many times the front is stepped (idempotent — a
column already behind the tool is not re-thinned on each crossing). Documented caveats:
(1) peripheral milling removes from ONE face, offsetting the neutral surface by ~a_e/2
(membrane-bending coupling that symmetric h³/h scaling ignores — negligible at
a_e = 0.1 mm, ~12 % at a rough 0.5 mm); (2) the band `ez ≥ hp − a_p` resolves no
element centroid until a_p ≈ 1.67 mm on the 24-row height mesh, so sub-mm depths report
exactly 0 % within-pass drift as a mesh-resolution floor (true drift there is ≲0.2 %).

```python
wp = MillingWorkpiece(LP, HP, BP, RHO, E, NU, N1=30, N2=24, n_modes=5)
wp.set_observation(x_obs=LP, z_obs=HP); wp.add_piezo_patch(...)
snaps = finishing_sequence(wp, n_layers=6, a_e=0.1e-3, a_p=0.020)  # 24 passes
# each snap: snap['freq'], snap['removed_frac'], snap['plant'] (solver-compatible)
```
