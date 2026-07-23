# tests — Mindlin port validation

Validation suite for the literal Reissner–Mindlin element port
(`01_core/mindlin_q8.py`) and its drop-in integration with the article
simulation package (package 12).

## Run

```bash
cd tests
bash run_tests.sh
# or individually:
python verify_mindlin.py        # structural / analytical benchmarks
python verify_geometry.py       # setup-diagram conformance (frame, BC, patch, sensor)
python verify_integration.py    # end-to-end drop-in with LQG + Newmark
```

Only `numpy` and `scipy` are required. No MATLAB needed.

## What is checked

### `verify_mindlin.py`
1. **Element sanity** — partition of unity (`ΣNᵢ=1`, `Σ∂Nᵢ=0`), matrix symmetry,
   and the zero-energy-mode count of a single free element (**4** = 3 rigid-body
   + 1 hourglass, as expected for the uniform-reduced 2×2 Q8 element).
2. **CCCC clamped square plate** — fundamental frequency parameter
   `λ₁ = ω a²√(ρh/D)` vs Leissa's reference **35.99** (and modes 2–6).
3. **Cantilever AL6061** (article geometry) — mode 1 vs the article's ~521 Hz
   and the beam-strip estimate (528 Hz).
4. **Mesh convergence** of the cantilever fundamental frequency.
5. **Thermal load** (`thermal_stress_M`) smoke test (net transverse force = 0).

### `verify_geometry.py`
Asserts the Mindlin model conforms to the **article setup diagram**
([`../docs/GEOMETRY.md`](../docs/GEOMETRY.md)): coordinate frame
`O_P–X_P Y_P Z_P`, encastrement at the bottom edge `Z_P=0` (all 3 DOFs),
cantilever mode-1 shape (w=0 at base → max at free top), piezo patch corners
`(x_P1,z_P1)-(x_P2,z_P2)`, displacement-sensor location, and tool path along
`X_P` at fixed height.

### `verify_inprocess.py`
Checks the in-process material-removal model (`InProcessPlate`, see
[`../docs/MATERIAL_REMOVAL.md`](../docs/MATERIAL_REMOVAL.md)): uniform-thickness
consistency with `PlateModel`, per-pass idempotency of `machine_to`, monotonic
mass reduction, the correct physical sign (thinning the free tip raises the
cantilever fundamental), and fixed-basis ROM accuracy.

### `verify_integration.py`
Builds the Mindlin `PlateModel` with the exact article parameters, checks every
interface attribute the downstream code relies on, then runs the **package-12
stack unchanged** (milling force → LQG controller → Newmark solver) and asserts
the closed loop reduces vibration within the piezo voltage limit.

## Expected output (abridged)

```
CCCC lambda_1 = 35.98   (Leissa 35.99, err -0.02%)
Cantilever mode 1 = 519.4 Hz   (article ~521 Hz)
LQG reduction vs open-loop: ~74 %   u_max ~13 V
ALL PASS
```
