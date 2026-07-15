# Active Vibration Control of a Thin-Walled Plate During Milling

**Topic**: **LQG** (benchmark baseline) vs **ESO-ADRC** (modal extended-state-observer
active disturbance rejection control) vs **A-ESO-ADRC** (its adaptive, cost-supervised
development) for chatter mitigation in peripheral milling of cantilever AL6061
plates. The plant model is anchored to Du, Liu, Dai & Long (2024),
*Int. J. Mech. Sci.* 274:109257.

> **⚠️ Read before citing any number from this repository**
>
> - [`docs/CONTRIBUTION.md`](docs/CONTRIBUTION.md) — what this work contributes,
>   positioning, and the publication roadmap.
> - [`docs/REPRODUCED_RESULTS.md`](docs/REPRODUCED_RESULTS.md) — verified numbers from
>   the committed code (every table below is copied from there).
> - [`docs/AUDIT_FINDINGS.md`](docs/AUDIT_FINDINGS.md) — the historical 44-finding
>   audit of the earlier learned-feedforward phase of this package (kept as record).
>
> **Package history:** P0–P2 turned this repository into an audited, article-anchored
> benchmark (train-once/held-out protocol, symmetric baselines, corrected Eq. 3
> forces, spillover + 10 nm measurement noise, Eq. 15 piezo coupling, rigorous
> closed-loop monodromy SLD). **P4 (2026-07-15): the earlier learned-feedforward
> controller family (PALF-LQG / A-PALF-LQG) was removed at the author's request and
> replaced by a new strategy developed to be adaptive** — the ESO-ADRC family below.
> **P5 adds the harmonic resonant cancellation (HRC) layer and the 4-rung
> supervised ladder — A-ESO-ADRC now beats LQG by ~50 % on the nominal plant and
> by a +48 % Monte-Carlo median (better in 94 % of samples) while remaining the
> only controller that never diverges.** All protocol/integrity fixes carry over
> unchanged. Remaining: P3 (experimental validation) — everything here is
> simulation.

---

## 🎛️ The controllers

| | Structure | Tuning |
|---|---|---|
| **LQG** | output-weighted LQR + Kalman filter (3 design modes) | weight grid search on the nominal model |
| **ESO-ADRC** | the SAME output-weighted LQR fed by a 9-state **modal extended state observer** that estimates a per-mode *total disturbance* d(t) ∈ R³ (regenerative force + feed forcing + spillover + drift) | grid over (w_q, w_qd, σ_d); the fixed design is **certification-selected**: smallest worst-case closed-loop Floquet radius over a design-time uncertainty ball |
| **ESO-ADRC+HRC** | + **harmonic resonant cancellation**: per-tooth-harmonic LTI resonant compensators (h = 1..4) with inverse-closed-loop-FRF phase — the online causal counterpart of model-inverse ILC, exploiting only the KNOWN spindle frequency | HRC grid (n_harm, g_base) on the performance base; n_harm capped by the mode-2 drift band |
| **A-ESO-ADRC** | a supervised **4-rung ladder** [HRC / performance-ESO / quasi-Kalman-ESO / certified-ESO] sharing one physical observer state — bumpless switching driven by the measured y²-cost only (rising-energy cascade panic with severity-based target, recovery-trend holds, per-pass failure flags, probe aborts, desperation probing, escalating locks). **No identification, no probe.** | rungs from the same grids |

The LQG-vs-ESO-ADRC comparison isolates exactly one ingredient: replace the Kalman
filter with a disturbance-estimating ESO. Four **documented design findings** (module
docstring of `02_controllers/adrc_controller.py`, all reproducible): canonical
output LADRC is structurally inapplicable here (non-collocated, non-minimum-phase
piezo→sensor transfer — it destabilizes for *every* bandwidth pair); the ESO gain
must come from a scaled Riccati equation (pole placement is numerically hopeless);
matched disturbance cancellation does not pay (actuator only ~19 % aligned with the
tool-force direction); and closed-loop actuator-effectiveness self-identification
is biased by the periodic cutting force (persistent excitation would be required) —
which is why the adaptive layer is identification-free.

---

## 📂 Package Structure

```
.
├── 01_core/              ← Physical models (FEM + dynamics)
│   ├── kirchhoff_q4.py           # Kirchhoff Q4 plate element FEM (consistent mass)
│   ├── plate_model.py            # Plate assembly + modal reduction
│   ├── piezo_actuator.py         # Piezo patch model (QDA60-20-0.7), modal force only
│   ├── milling_force.py          # Helical-engagement force kernels (article Eq. 4)
│   └── newmark_solver.py         # Newmark-β integration with regenerative delay
│
├── 02_controllers/        ← Control algorithms
│   ├── lqg_controller.py         # LQG baseline (grid-searched weights)
│   └── adrc_controller.py        # ESO-ADRC + A-ESO-ADRC (+ canonical LADRC
│                                 #   kept only for the negative result)
│
├── 03_analysis/           ← Stability & robustness analysis
│   ├── fdm_stability.py          # Per-mode SLD + rigorous closed-loop coupled
│   │                             #   monodromy (LQG adapter + GENERIC realization)
│   ├── uncertainty_analysis.py   # Monte-Carlo over any controller set
│   └── mesh_convergence.py       # FEM natural-frequency convergence vs Table 4
│
├── 04_figures/            ← Geometry + academic-style SLD generators
├── 05_main/               ← Main simulation scripts
│   ├── main_simulation.py        # Authoritative 3-way comparison + certification
│   │                             #   + worst-position closed-loop SLD
│   ├── main_robustness_mc.py     # Monte-Carlo robustness driver
│   ├── main_adaptive_removal.py  # Drift / stress benchmark (fixed vs adaptive)
│   └── main_realistic_piezo.py   # LQG with realistic piezo non-linearities
│
├── figures/               ← Curated publication figures (vector PDF, 300 DPI)
└── docs/                  ← Contribution, audit (historical), reproduced results
```

The `figures/` directory holds a committed snapshot of the publication set (vector
PDF); see `figures/README.md` for provenance and regeneration commands.

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

### Run

```bash
python main_simulation.py         # 3-way comparison + certification + SLD (~4.5 min)
python main_robustness_mc.py      # Monte-Carlo (~4 min)
python main_adaptive_removal.py   # drift / stress benchmark (~1.5 min)
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

The patch couples through the article's **Eq. (15) C_P0** coefficient (modal force
only; no added stiffness/mass in the FEM assembly).

### Tool and cutting parameters (article condition T1)

| Parameter | Value | Parameter | Value |
|---|---:|---|---:|
| Diameter | 10 mm | Spindle speed | 4900 RPM |
| Teeth | 3 | Feed per tooth | 0.02 mm |
| Helix angle | 35° | Axial depth a_p | 0.3 mm |
| Rake angle | 15° | Radial depth a_e | 0.1 mm |
| K_T | 925 MPa | k_N | 0.26 |

Evaluation model: **5-mode plant, 3-mode controllers** (control + observation
spillover), 10 nm measurement noise, identical ±150 V clipping, corrected Eq. (3)
cutting constants (k₁ = 0.3174, k₂ = 1.1258).

---

## 📊 Verified results (committed code, seeds fixed — see docs/REPRODUCED_RESULTS.md)

### Held-out scenarios (y_RMS, µm; T = 0.5 s; designs frozen on the nominal model)

| Scenario | LQG | ESO-ADRC+HRC | ESO-ADRC (certified) | A-ESO-ADRC |
|---|---:|---:|---:|---:|
| S1 Nominal | 0.777 | **0.381 (+51 %)** | 0.826 | **0.381 (+51 %)** |
| S2 Aggressive (a_p = 0.6 mm) | **1.558** | DIVERGES | 1.824 | 2.43 |
| S3 Model mismatch (ω −8 %) | **0.900** | 251 (bounded) | 20.8 (bounded hole) | 12.0 (bounded, recovering) |
| S4 High K_T (+30 %) | **1.013** | 1.135 | 1.078 | 1.135 |

### Drift / stress benchmark (`main_adaptive_removal.py`)

| Case | LQG | ESO-ADRC+HRC | ESO-ADRC | A-ESO-ADRC |
|---|---:|---:|---:|---:|
| D0 no drift | 0.777 | **0.381** | 0.826 | **0.381** |
| D1 ramp to +15 % during the pass | **0.682** | 0.955 | 1.276 | 1.424 |
| D2 ramp to −12 % during the pass | DIVERGES | 33.6 (bounded) | **0.898** | 1.209 |
| D3 static −12 % | DIVERGES | DIVERGES | 1.140 | **1.057** |
| D4 piezo effectiveness ×0.25 | 1.241 | 2.096 | 1.221 | **1.200** |

**The headline (P5):** the tooth-harmonic resonant layer (HRC — the online,
causal counterpart of model-inverse ILC, exploiting only the KNOWN spindle
frequency) cuts the nominal RMS **~50 % below LQG** (0.381 vs 0.777 µm) at lower
peak voltage, and wins the Monte-Carlo (±15 % cutting, ±3 % frequency, ±20 %
damping): medians 0.410 vs 0.788 µm, **+48 % median gain, better in 94 % of the
50 samples, everything 50/50 converged**. No fixed tuning covers the whole range
(the HRC design diverges at a_p = 0.6, LQG beyond −10 % drift, the certified ESO
rings at −8 %): **A-ESO-ADRC — the 4-rung supervised ladder — is the only
controller that never diverges across all 9 scenarios** while matching the HRC
rung nominally. Honest weak spot: at ω−8 % its recovery is bounded but slow
(12 µm over the 0.5 s pass vs LQG's 0.90 µm).

### Stability lobes at 4900 RPM — closed-loop coupled monodromy, worst of 3 tool positions

| Configuration | a_p critical |
|---|---:|
| Open-Loop | 0.100 mm — **matches the article's experimental limit** |
| LQG | 1.075 mm (10.8× OL) |
| ESO-ADRC (certified rung = A-ESO-ADRC's fallback boundary) | 0.913 mm (9.1× OL) |

All panels come from the monodromy of the full coupled, time-periodic delayed loop
with the controller embedded (generic realization — works for any LTI output
feedback controller), evaluated at x = 0, L/4, L/2 (worst case). Both controlled
depths are the same order as the article's experimentally achieved 0.6–0.8 mm.

---

## 📚 Reference

The plant model, parameters, and experimental anchors come from:

> Du J., Liu X., Dai H., Long X. (2024). Robust combined time delay control for
> milling chatter suppression of flexible workpieces. *International Journal of
> Mechanical Sciences* 274:109257. https://doi.org/10.1016/j.ijmecsci.2024.109257

Key methods: Insperger & Stépán (2004) semi-discretization; Altintas (2012)
*Manufacturing Automation*; Han (2009) / Gao (2003) ADRC & bandwidth
parametrisation; Anderson & Moore (2007) *Optimal Control*.

---

## 💻 Computational requirements

| Task | Time |
|---|---:|
| Full comparison + certification + 3 SLDs (`main_simulation.py`) | ~4.5 min |
| Monte-Carlo (50 samples × 3 controllers) | ~4 min |
| Drift benchmark (5 cases × 3 controllers) | ~1.5 min |
| Single simulation (T = 0.5 s, dt = 50 µs) | ~1 s |

Tested on Python 3.11–3.12, NumPy ≥1.26, SciPy ≥1.13. No GPU required.

---

## 📝 License & Disclaimer

This code is provided for academic research purposes, without warranty. Do not cite
performance numbers from this repository that are not listed in
`docs/REPRODUCED_RESULTS.md`.
