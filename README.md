# Active Vibration Control of a Plate during Milling — Mindlin edition

This repository hosts the **Reissner–Mindlin plate edition** of the article
simulation package for active vibration control (LQG vs DARC-MPC) of a
thin-walled cantilever AL6061 plate during peripheral milling.

## What is in here

```
mindlin_simulation_package/     ← the article package (package 12), Mindlin plate
├── 01_core/
│   ├── mindlin_q8.py           ← ⭐ Reissner–Mindlin Q8 element, LITERAL port of Plate-FEM
│   ├── plate_model.py          ← plate assembly + modal reduction + piezo (Mindlin)
│   ├── MINDLIN_PORT.md         ← verbatim MATLAB(.m) ↔ Python(.py) correspondence
│   ├── kirchhoff_q4.py         ← original Kirchhoff element (reference only, unused)
│   ├── piezo_actuator.py / milling_force.py / newmark_solver.py   (unchanged)
├── 02_controllers/  03_analysis/  04_figures/  05_main/           (unchanged)
└── tests/                      ← validation suite (no MATLAB needed)
```

## The change in one sentence

The plate finite-element model was switched from the **Kirchhoff Q4** element to
the **Reissner–Mindlin 8-node Serendipity (Q8)** element — taken *literally* from
the MATLAB `Plate-FEM/Mindlin_plate` package — while keeping the `PlateModel`
public interface identical, so the rest of the article package (controllers,
Newmark solver, FDM stability, figures) runs **without any modification**.

## Mindlin element (`mindlin_q8.py`)

- 8-node Serendipity quad, 3 DOF/node `(w, θx, θy)`, 24 DOF/element.
- Bending (`Bf`,`Hf`) + transverse shear (`Bs`,`Hs`): `Ke = h·(Kf + Ks)`.
- Consistent mass with rotary inertia `Ie = diag(1, h²/12, h²/12)`.
- Uniform reduced 2×2 Gauss integration, shear-correction `κ = 5/6` — exactly as
  in Plate-FEM.
- Piezo coupling by the consistent Mindlin bending-moment analogy.

## Validation (`tests/`)

| Benchmark | Result | Reference |
|---|---|---|
| CCCC thin square plate, λ₁ = ω a²√(ρh/D) | **35.98** | 35.99 (Leissa), err −0.02% |
| Cantilever AL6061 (article geometry) mode 1 | **519.4 Hz** | ~521 Hz (article) |
| Drop-in with package-12 LQG + Newmark | LQG cuts y_rms 2.19→0.57 µm | — |

```bash
cd mindlin_simulation_package/tests && bash run_tests.sh
```

## Quick start

```bash
pip install numpy scipy matplotlib
cd mindlin_simulation_package
python setup_workspace.py            # flatten modules into workspace/
cd workspace && python main_simulation.py
```

Sources of the two input archives: the MATLAB FEM library `Plate-FEM` (Mindlin
theory) and the Python `article_simulation_package` (package 12).
