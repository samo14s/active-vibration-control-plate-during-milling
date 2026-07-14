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
> **P1 applied:** cutting constants k1, k2 corrected to Eq. (3) and deduplicated; the
> inverse crime is removed (5-mode plant, controllers on the first 3 — spillover);
> 10 nm measurement noise; identical ±150 V clipping.
>
> **P2 applied:** the article's **Eq. (15) piezo coupling** C_P0; a **rigorous
> closed-loop coupled monodromy SLD** (controller in the loop — no surrogate, no
> per-mode decoupling), making PALF = LQG a rigorous consequence of ∂u_FF/∂x̂ = 0; a
> **Monte-Carlo LQG-vs-PALF** robustness driver (divergence reported, no survivorship
> bias); and a **FEM mesh-convergence** study reconciling the frequencies vs Table 4.
>
> **P2.5 applied (improvement pass):** the ILC is upgraded to a **frequency-domain
> model-inverse harmonic update** (gains jump to double digits, still held-out); the
> SLD is evaluated at the **worst of 3 tool positions** (article Fig. 6 treatment);
> and the article's **Eq. (30) delayed PD** is added as a third baseline — it cannot
> stabilize these conditions, reproducing the article's own Fig. 14 finding.
>
> **Remaining (P3):** experimental validation on a physical plate — everything here is
> simulation.

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
│   ├── fdm_stability.py          # Per-mode SLD + rigorous closed-loop coupled monodromy
│   ├── uncertainty_analysis.py   # Monte-Carlo LQG-vs-PALF (run_mc_lqg_vs_palf)
│   └── mesh_convergence.py       # FEM natural-frequency convergence vs Table 4
│
├── 04_figures/            ← Figure generators (illustrative)
├── 05_main/               ← Main simulation scripts
│   ├── main_simulation.py        # Authoritative LQG vs PALF-LQG comparison + SLD
│   ├── main_robustness_mc.py     # Monte-Carlo robustness driver
│   └── main_realistic_piezo.py   # With realistic piezo non-linearities
│
├── figures/               ← Curated publication figures (vector PDF, 300 DPI)
└── docs/                  ← Contribution, audit, reproduced results
```

The `figures/` directory holds a committed snapshot of the publication set (vector PDF);
see `figures/README.md` for provenance and regeneration commands. All figures are
regenerable from the scripts above.

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
stiffness/mass in the current FEM assembly); the coupling scalar is the article's
**Eq. (15) C_P0** coefficient (P2 fix).

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

### RMS vibration vs LQG baseline (T = 0.5 s) — held-out, symmetric, spillover + noise, Eq. (15) piezo, model-inverse ILC

| Scenario | LQG y_RMS | PALF y_RMS | Gain |
|---|---:|---:|---:|
| S1 Nominal | 0.777 µm | 0.625 µm | **+19.5 %** |
| S2 Aggressive (a_p = 0.6 mm) | 1.558 µm | 1.386 µm | **+11.0 %** |
| S3 Model mismatch (ω −8 %) | 0.900 µm | 0.769 µm | **+14.6 %** |
| S4 High K_T (+30 %) | 1.013 µm | 0.853 µm | **+15.8 %** |
| **Average** | 1.062 µm | 0.908 µm | **+14.4 %** |

The frequency-domain model-inverse ILC cancels most of the tooth-passing-periodic
residual that survives LQG: +19.5 % nominal, and the FROZEN feedforward keeps
double-digit gains on every held-out perturbation (graceful degradation). Trained once
on the nominal scenario; S2/S3/S4 held-out; 5-mode plant / 3-mode controllers
(spillover); 10 nm noise; ~8 % more RMS voltage (6.0 vs 5.6 V).

**Monte-Carlo robustness** (`main_robustness_mc.py`, 50 samples, ±15 % cutting / ±3 %
freq / ±20 % damping): 50/50 converged for both controllers; **PALF beats LQG in 100 %**
of samples, median RMS gain **+17.8 %** [p05 +15.8 %, p95 +19.5 %].

**Four-way benchmark** (`main_delayed_pd_baseline.py`): the article's Eq. (30) delayed
PD — even grid-tuned — cannot stabilize these conditions (hundreds of µm or divergence),
reproducing the article's own Fig. 14 finding; OL ✗ → PD ✗ → LQG 0.78 µm → PALF 0.62 µm.

### Stability lobes at 4900 RPM — closed-loop coupled monodromy, worst of 3 tool positions

| Configuration | a_p critical |
|---|---:|
| Open-Loop | 0.100 mm — **matches the article's experimental limit** |
| LQG (closed-loop monodromy) | 1.08 mm (10.8× OL) |
| PALF-LQG | 1.08 mm — **= LQG, rigorously** (∂u_FF/∂x̂ = 0 → identical monodromy) |

All panels come from the monodromy of the full coupled, time-periodic delayed loop
with the LQG controller embedded, evaluated at x = 0, L/4, L/2 (worst case — the
article's Fig. 6 treatment). The controlled critical depth is now the same order as
the article's experimentally achieved 0.6–0.8 mm limits.

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
