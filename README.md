# Active Vibration Control of a Thin-Walled Plate During Milling

**Topic**: LQG feedback vs. **PALF-LQG** — LQG augmented with a learned,
tooth-passing-phase-locked neural feedforward (renamed from the earlier over-claiming
"DARC-MPC v3") — for chatter mitigation in peripheral milling of cantilever AL6061
plates. The plant model is anchored to Du, Liu, Dai & Long (2024),
*Int. J. Mech. Sci.* 274:109257.

> **⚠️ Read before citing any number from this repository**
>
> - [`docs/CONTRIBUTION.md`](docs/CONTRIBUTION.md) — what this work contributes,
>   literature positioning, and the publication roadmap.
> - [`docs/REPRODUCED_RESULTS.md`](docs/REPRODUCED_RESULTS.md) — verified numbers from
>   the committed code (the historical results table of this README did **not**
>   reproduce and has been replaced below).
> - [`docs/AUDIT_FINDINGS.md`](docs/AUDIT_FINDINGS.md) — 44 verified findings
>   (integrity, naming, methodology) that must be addressed before submission.
>
> **P0 integrity fixes have been applied** (see `docs/AUDIT_FINDINGS.md` §"P0"):
> the fabricated ×1.30 "DARC" stability lobe is removed (PALF shares the LQG
> boundary); the feedforward is now trained **once on the nominal scenario and frozen**,
> then evaluated on held-out scenarios; the baseline is **symmetric** (identical
> grid-searched LQG weights for both); dead/mislabeled components ("adaptive RLS",
> the anti-disturbance pretrainer) are removed; and the controller is renamed from the
> over-claiming "DARC-MPC" to **PALF-LQG** (Phase-Aware Learned Feedforward + LQG).
>
> **P1 fixes now applied too:** the cutting constants k1, k2 are corrected to the
> article's Eq. (3) and deduplicated into `milling_force.cutting_constants`; the
> inverse crime is removed (the simulated plant carries **5 modes**, the controllers
> are designed on the **first 3** — spillover); **10 nm measurement noise** is injected;
> and both controllers clip identically to ±150 V.
>
> **Still open (P2):** periodic-gain closed-loop SLD, coupled/position-resolved SLD,
> Eq. (15) piezo coefficient, Monte-Carlo wiring, FEM mesh-convergence note.

---

## 📂 Package Structure

```
.
├── 01_core/              ← Physical models (FEM + dynamics)
│   ├── kirchhoff_q4.py           # Kirchhoff Q4 plate element FEM (consistent mass)
│   ├── plate_model.py            # Plate assembly + modal reduction
│   ├── piezo_actuator.py         # Piezo patch model (QDA60-20-0.7), modal force only
│   ├── milling_force.py          # Helical-engagement force kernels (article Eq. 4)
│   └── newmark_solver.py         # Newmark-β time integration with regenerative delay
│
├── 02_controllers/        ← Control algorithms
│   ├── lqg_controller.py         # LQG with Kalman observer (grid-searched weights)
│   └── palf_lqg_controller.py    # LQG + phase-locked NN feedforward (ILC-trained)
│
├── 03_analysis/           ← Stability & robustness analysis
│   ├── fdm_stability.py          # Floquet multipliers (0th-order semi-discretization)
│   └── uncertainty_analysis.py   # Monte Carlo sampling (not yet wired into main)
│
├── 04_figures/            ← Figure generators
├── 05_main/               ← Main simulation scripts
│   ├── main_simulation.py        # Full LQG vs feedforward-augmented comparison
│   └── main_realistic_piezo.py   # With realistic piezo non-linearities
│
└── docs/                  ← Contribution, audit, reproduced results
```

---

## 🎯 Quick Start

### Prerequisites

```bash
pip install numpy scipy matplotlib
```

### Setup

All Python files must be in the **same directory** to allow imports:

```bash
cp 01_core/*.py 02_controllers/*.py 03_analysis/*.py 04_figures/*.py 05_main/*.py ./
```

### Run main simulation

```bash
# Full comparison (4 scenarios + 3 SLDs, ~80 s)
python main_simulation.py
```

---

## 📐 Physical Setup (identical to Du et al. 2024, Tables 1–3)

### Plate (AL6061, vertical cantilever)

| Parameter | Symbol | Value |
|---|---|---:|
| Length | L_P | 100 mm |
| Height | H_P | 80 mm |
| Thickness | B_P | 4 mm |
| Density | ρ | 2830 kg/m³ |
| Young's modulus | E | 69 GPa |
| Poisson ratio | ν | 0.33 |
| Damping (modes 1–3, measured in the article) | ζ₁, ζ₂, ζ₃ | 0.31%, 0.17%, 0.27% |

### Piezo patch (QDA60-20-0.7, SINOCERA)

| Parameter | Symbol | Value |
|---|---|---:|
| Position | — | lower-left corner (x: 0–20 mm, z: 0–60 mm) |
| Thickness | h_Pa | 0.7 mm |
| Piezo coefficient | d₃₁ | 175 pm/V |
| Young's modulus | E_Pe | 63 GPa |
| Voltage saturation | u_max | ±150 V |

Note: the patch couples to the plate through a modal force vector only (no added
stiffness/mass in the current FEM assembly), and the coupling scalar is a simplified
induced-moment constant, not the article's Eq. (15) — see audit finding #4.

### Tool and cutting parameters (article condition T1)

| Parameter | Value | Parameter | Value |
|---|---:|---|---:|
| Diameter | 10 mm | Spindle speed | 4900 RPM |
| Teeth | 3 | Feed per tooth | 0.02 mm |
| Helix angle | 35° | Axial depth a_p | 0.3 mm |
| Rake angle | 15° | Radial depth a_e | 0.1 mm |
| K_T | 925 MPa | k_N | 0.26 |

---

## 🔬 What the controller actually is

```
u(t) = u_LQG(x̂(t)) + α · NN_FF(φ(t), x̂(t))
```

- **Feedback**: LQG (LQR gain + Kalman observer), weights grid-searched.
- **Feedforward**: a small MLP, (n_x+2) → 16 → 1 with tanh activations (~161
  parameters), input = estimated modal state + (cos φ, sin φ) of the tooth-passing
  phase, output saturated to ±30 V. Trained by hand-coded SGD through an iterative
  learning loop (simulate → collect → retrain, 30 iterations). In the current
  training protocol the state inputs receive zero gradient (samples use x = 0), so
  the trained object is effectively a **learned periodic map u_FF(φ)** — i.e.
  repetitive-control-like feedforward.
- **Safety filter**: a CLF-style voltage governor on the nominal delay-free model
  (heuristic — not a stability proof for the true delayed periodic loop).
- The historical name "DARC-MPC" (Deep Adaptive Robust Control with MPC) does not
  describe this architecture; renaming (e.g. **PALF-LQG**, Phase-Aware Learned
  Feedforward LQG) is part of the pre-submission roadmap.

---

## 📊 Verified results (committed code, 2 runs, seeds fixed — see docs/REPRODUCED_RESULTS.md)

### RMS vibration vs LQG baseline (T = 0.5 s) — held-out, symmetric baseline, spillover + noise

| Scenario | LQG y_RMS | PALF y_RMS | Gain |
|---|---:|---:|---:|
| S1 Nominal | 0.744 µm | 0.706 µm | +5.1 % |
| S2 Aggressive (a_p = 0.6 mm) | 1.491 µm | 1.433 µm | +3.9 % |
| S3 Model mismatch (ω −8 %) | 0.853 µm | 0.744 µm | **+12.7 %** |
| S4 High K_T (+30 %) | 0.970 µm | 0.927 µm | +4.5 % |

The learned feedforward buys little on the nominal plant and most under model
mismatch — that robustness asymmetry is the honest result. Trained once on the nominal
scenario and frozen; S2/S3/S4 are held-out. The plant carries 5 modes, the controller
sees 3 (spillover), and the measurement has 10 nm noise. With the corrected (stronger)
forces the controlled stability margin at 0.3 mm is ~−9 % frequency mismatch, so S3
uses −8 % (see `docs/REPRODUCED_RESULTS.md`).

### Stability lobes at 4900 RPM (Floquet)

| Configuration | a_p critical |
|---|---:|
| Open-Loop | 0.100 mm — **matches the article's experimental limit** |
| LQG closed loop | 2.05 mm (20.5× OL) |
| PALF-LQG | 2.05 mm — **= LQG** (a phase-locked feedforward does not shift the boundary) |

(The controlled critical depth is lower than the earlier 2.54 mm because the corrected
k1/k2 give ~15–20 % stronger cutting forces.)

---

## 📚 Reference

The plant model, parameters, and experimental anchors come from:

> Du J., Liu X., Dai H., Long X. (2024). Robust combined time delay control for
> milling chatter suppression of flexible workpieces. *International Journal of
> Mechanical Sciences* 274:109257. https://doi.org/10.1016/j.ijmecsci.2024.109257

Key methods: Insperger & Stépán (2004) semi-discretization; Altintas (2012)
*Manufacturing Automation*; Anderson & Moore (2007) *Optimal Control*.

---

## 💻 Computational requirements

| Task | Time |
|---|---:|
| Full 4-scenario comparison + 3 SLDs | ~80 s |
| Single simulation (T = 0.5 s, dt = 50 µs) | ~10 s |

Tested on Python 3.11–3.12, NumPy ≥1.26, SciPy ≥1.13. No GPU required.

---

## 📝 License & Disclaimer

This code is provided for academic research purposes, without warranty. Do not cite
performance numbers from this repository that are not listed in
`docs/REPRODUCED_RESULTS.md`.
