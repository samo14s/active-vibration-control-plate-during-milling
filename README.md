# Simulation Package — Active Vibration Control of a Flexible Plate in Milling

**Topic**: numerical comparison of feedback and feedforward architectures for
chatter/vibration mitigation in peripheral milling of a flexible cantilever
plate with a bonded piezoelectric actuator:

| Controller | Information it uses | File |
|---|---|---|
| **LQG** (baseline) | displacement sensor | `02_controllers/lqg_controller.py` |
| **IMC-LQG** (feedback baseline, internal-model principle) | displacement sensor + spindle period (encoder) | `02_controllers/imc_lqg_controller.py` |
| **DARC** (proposed: Deep Anticipative Residual Control) | spindle phase (encoder) + nominal cutting model + sensor (for its LQG base and NN) | `02_controllers/darc_controller.py` |

**Thesis context**: *Contribution au contrôle actif des vibrations en
fraisage des pièces flexibles* — a purely theoretical (simulation-based)
study. Because no experimental validation is available, the package follows
a strict **honest evaluation protocol** (below), and every known modeling
limitation is documented rather than hidden. A full independent audit of
this package, with the prioritized correction plan that produced the current
version, is in **`docs/AUDIT_SCIENTIFIQUE.md`** — read it before writing any
manuscript chapter from these results.

---

## Honest evaluation protocol

All reported numbers obey the following rules (enforced in the scripts):

1. **No oracle information.** Controllers are designed from the NOMINAL
   model only (nominal K_T = 925 MPa, nominal modal frequencies, commanded
   engagement). In the robustness scenarios the plant is perturbed but the
   controllers are not re-informed (S4's +30 % K_T is *unknown* to DARC's
   feedforward).
2. **No train/test contamination.** DARC's NN residual is trained in the
   controller's nominal world with training-noise realizations
   (seeds 100+iter) and best-checkpoint selection on a separate VALIDATION
   episode (seed 200). The evaluation episode (seed 1) is never used for
   training or model selection.
3. **Identical base.** LQG and DARC share the byte-identical LQG design
   (w_q = 1e14, w_qd = 1e8, R = 1, same Kalman tuning, same ZOH
   discretization) in the main study AND in the figure pipeline.
4. **Honest stability analysis.** SLDs are Floquet analyses of the complete
   coupled multi-mode system with the actual digital compensator (observer +
   discrete feedback) augmented in the monodromy matrix. There is no
   separate "DARC" SLD: a phase-locked periodic feedforward is an exogenous
   input and provably does not move Floquet multipliers → SLD(DARC) =
   SLD(LQG). Linear analysis: boundaries are only valid where u < 150 V.
5. **Paired noise.** All controller comparisons use common random numbers.

## Modeling — scope and limits

The plant follows the nonlinear analytical modeling style of Nasiri &
Moradi, *MSSP* 224 (2025) 112198: assumed-modes Galerkin plate
(clamped-free × free-free beam products), nonlinear delayed modal equation,
3-tooth regenerative milling force, piezo patch actuation. Deviations and
limits — all deliberate and documented (details in `01_core/README.md`):

- Modal frequencies are **calibrated** to legacy FEM anchors
  [521.06, 1069.95, 2733.02] Hz (factors 0.987/0.918/0.898); mode SHAPES
  remain analytical → hybrid model, to be stated as such.
- The Von Kármán cubic uses the variationally consistent membrane-energy
  form under u₀=v₀=0 (an upper bound; diagonal truncation). At the µm
  amplitudes of all controlled results the nonlinear terms are **dormant**:
  the comparisons live on the linear backbone.
- Open-loop post-instability amplitudes are bounded by a numerical chip
  clamp, **not** by complete separation physics — only the stability
  boundary is quantitative, never the saturated chatter amplitude.
- NOT included in any reported result: process damping, flank-wear edge
  force, piezo patch mass/stiffness, tool-side dynamics, multiple
  regeneration. The sensor/actuator realism module models saturation, slew,
  amplifier lag, delay and noise — its "material lag" block is linear (no
  true hysteresis) and there is no temperature model.
- Time step 50 µs → mode 3 (2733 Hz) has 7.3 samples/period; the
  regenerative delay is grid-rounded (effective spindle speed 4878 RPM).

## 📂 Package structure

```
├── 01_core/                ← physical models (see 01_core/README.md)
│   ├── plate_model.py            # analytical Galerkin plate + piezo coupling
│   ├── milling_force.py          # 3-tooth regenerative force coefficients
│   ├── newmark_solver.py         # Newmark-β NDDE integrator
│   ├── piezo_actuator.py         # actuator/sensor realism wrapper
│   └── kirchhoff_q4.py           # LEGACY FEM element (calibration anchors)
├── 02_controllers/
│   ├── lqg_controller.py         # LQG baseline
│   ├── imc_lqg_controller.py     # IMC-LQG internal-model feedback baseline
│   └── darc_controller.py        # DARC (proposed)
├── 03_analysis/
│   ├── fdm_stability.py          # closed-loop Floquet SLD (+ legacy per-mode)
│   └── uncertainty_analysis.py   # Monte Carlo harness (NOT yet executed)
├── 04_figures/                   # publication figure generators
├── 05_main/
│   ├── main_simulation.py        # LQG vs DARC, 4 scenarios + SLD + sensor sweep
│   ├── main_imc_baseline.py      # LQG vs IMC-LQG vs DARC-FF
│   ├── main_fullpath_comparison.py  # 20.4 s full feed pass (held-out NN test)
│   └── main_realistic_piezo.py   # LQG under realistic actuator/sensor
├── docs/AUDIT_SCIENTIFIQUE.md    # independent audit + correction plan
└── setup_workspace.py            # flatten modules into workspace/
```

## 🎯 Quick start

```bash
pip install numpy scipy matplotlib
python setup_workspace.py && cd workspace
python main_simulation.py           # ~5 min : 4 scenarios + SLD + sensor sweep
python main_imc_baseline.py         # ~30 s  : the internal-model baseline
python main_fullpath_comparison.py  # ~10 min: held-out full-pass evaluation
python main_realistic_piezo.py      # ~1 min : realistic actuator study (LQG)
```

## 📐 Physical setup

### Plate (vertical cantilever)

| Parameter | Symbol | Value |
|---|---|---:|
| Length | L_P | 100 mm |
| Height | H_P | 80 mm |
| Thickness | B_P | 4 mm |
| Density | ρ | 2830 kg/m³ ⚠ (2024/7075-class; NOT AL6061 = 2700) |
| Young's modulus | E | 69 GPa |
| Poisson ratio | ν | 0.33 |
| Damping (modes 1–3) | ζ | 0.31 %, 0.17 %, 0.27 % |
| Modal frequencies (calibrated) | f | 521.06, 1069.95, 2733.02 Hz |

### Piezo patch (QDA60-200.7) · Tool · Cutting

| | | | | | |
|---|---:|---|---:|---|---:|
| Patch position | 0–20 × 0–60 mm | Tool diameter | 10 mm | Spindle speed | 4900 RPM |
| Patch thickness | 0.7 mm | Teeth | 3 | Feed/tooth | 0.02 mm |
| d₃₁ | 175 pm/V | Helix | 35° | a_p nominal | 0.3 mm |
| E_Pe | 63 GPa | K_T | 925 MPa | a_e | 0.1 mm |
| Voltage limit | ±150 V | k_N | 0.26 | Sensor noise | 0.1 µm RMS |

---

## 📊 Key results (honest protocol)

### LQG vs DARC — 4 scenarios (T = 0.5 s, `main_simulation.py`)

| Scenario | LQG y_RMS | DARC y_RMS | DARC gain |
|---|---:|---:|---:|
| S1 Nominal | 0.605 µm | **0.294 µm** | +51.5 % |
| S2 Aggressive (a_p = 0.6 mm) | 1.206 µm | **0.566 µm** | +53.1 % |
| S3 Detuned plant (ω −15 %) | 0.923 µm | **0.334 µm** | +63.8 % |
| S4 Unknown K_T (+30 %) | 0.788 µm | **0.458 µm** | +41.9 % |
| **AVERAGE** | **0.881 µm** | **0.413 µm** | **+53.1 %** |

S4 is the number to quote for force-model robustness: the feedforward is
designed on nominal K_T and still delivers +41.9 % (an *informed* FF would
give ~+52 % — the gap is the honest price of model error).

### The internal-model question (`main_imc_baseline.py`)

"A feedback controller cannot reject a periodic disturbance" is FALSE when
the spindle period is known — IMC-LQG estimates the tooth-passing harmonics
online (augmented Kalman) and cancels them **without any cutting-force
model**:

| Scenario | LQG | IMC-LQG | DARC-FF (no NN) |
|---|---:|---:|---:|
| S1 Nominal | 0.605 µm | **0.223 µm** | 0.363 µm |
| S2 Aggressive | 1.206 µm | **0.475 µm** | 0.728 µm |
| S3 Detuned ω −15 % | 0.923 µm | 15.8 µm ⚠ | **0.592 µm** |
| S4 Unknown K_T | 0.788 µm | **0.293 µm** | 0.536 µm |

**No architecture dominates.** IMC-LQG wins whenever its plant model is
good (and needs no force model), but its nominal-model inversion mis-phases
near resonance under −15 % structural detuning and destabilizes (voltage
saturation). DARC's feedforward degrades gracefully under detuning but pays
for force-model error, and its NN adds a few honest points on top. This
trade-off (information structure ↔ robustness direction) is a genuine
contribution of the study.

### Sensor robustness (fig10)

DARC's feedforward is phase-indexed (encoder), so it survives displacement-
sensor degradation better than pure feedback — but note the symmetric
caveat: its own reference (spindle phase) is assumed perfect here, and a
measured ~30–40° tooth-passing phase error makes the FF *worse* than LQG.

| Sensor noise | LQG | DARC | DARC gain |
|---|---:|---:|---:|
| ideal | 0.604 µm | 0.291 µm | +52 % |
| 0.6 µm | 0.657 µm | 0.388 µm | +41 % |
| 1.0 µm | 0.736 µm | 0.509 µm | +31 % |
| 2.0 µm | 1.022 µm | 0.866 µm | +15 % |

### Full feed pass — held-out evaluation (`main_fullpath_comparison.py`)

The NN is trained on a 0.5 s mid-path segment and evaluated over the entire
20.4 s pass (genuine out-of-sample data): the three-layer ablation holds,
with the FF providing the bulk of the gain and the NN a few honest points:

| Controller (full path, T = 20.4 s) | RMS | vs LQG |
|---|---:|---:|
| LQG (feedback) | 0.471 µm | — |
| DARC-FF (LQG + feedforward) | 0.286 µm | **+39.4 %** |
| DARC (LQG + FF + NN) | 0.266 µm | **+43.5 %** |

The NN's out-of-sample marginal gain (+4 points) — not the in-sample
per-scenario figure — is the number to quote for the "Deep" layer.

### Stability lobes (closed-loop Floquet, at 4900 RPM)

| Configuration | a_p critical | vs open loop |
|---|---:|---:|
| Open loop | 0.100 mm | 1× |
| LQG (observer + sampling in the monodromy) | **2.375 mm** | 23.8× |
| DARC | = LQG (exogenous FF: Floquet unchanged) | 23.8× |

The former claims (LQG 2.86 mm / DARC 4.00 mm "40×") were artifacts of a
pole-substitution shortcut and a hard-coded damping multiplier — both
removed. Caveat: linear analysis; the boundary is only physical where the
required voltage stays below ±150 V.

### Realistic actuator/sensor study (`main_realistic_piezo.py`, LQG)

After fixing a sensor-delay off-by-one bug (the delay buffer applied 100 µs
instead of the specified 50 µs), the LQG loop is stable under the realistic
actuator model: 98.98 % reduction vs 99.01 % ideal (0.04 % degradation,
max real voltage 13.2 V). ⚠ The loop's delay margin is thin (stable at
50 µs sensor delay, divergent at 100 µs): any deployment claim must budget
total loop latency. DARC under the realistic actuator is an open item (P1).

---

## Known open items (P1 — see docs/AUDIT_SCIENTIFIQUE.md)

1. Spindle phase-error/jitter sweep for the feedforward (symmetric to the
   sensor-noise sweep).
2. DARC under the realistic actuator model.
3. dt-convergence study (50 µs → 10 µs) for mode-3 content.
4. Worst-case tool-position SLD (path-averaged Dp hides the torsion mode).
5. Actually execute the Monte Carlo harness (protocol is implemented).
6. Kalman covariances derived from the true noise levels; internal-model +
   feedforward hybrid controller.
7. Alloy/density consistency (2830 kg/m³ vs "AL6061" label).

## 📚 Citation

```bibtex
@phdthesis{darc2026,
  title  = {Contribution au contr{\^o}le actif des vibrations en fraisage
            des pi{\`e}ces flexibles},
  note   = {Simulation study. DARC (LQG + spindle-synchronized inverse-model
            feedforward + ILC-trained neural residual) reduces RMS vibration
            by ~53% below an identically-tuned LQG across nominal and
            perturbed scenarios under a no-oracle protocol (+42% when the
            cutting coefficient is unknown by +30%); an internal-model LQG
            baseline (no cutting model needed) is stronger nominally but
            fragile under structural detuning, delineating the information
            structure trade-off between feedback and feedforward chatter
            rejection.},
  year   = {2026}
}
```

### Key references

1. Nasiri, K. & Moradi, H. (2025). *MSSP* **224**: 112198 — plant-model style.
2. Francis, B.A. & Wonham, W.M. (1976) — internal model principle (IMC-LQG baseline).
3. Insperger, T. & Stépán, G. (2004). *IJNME* 61:117–141 — semi-discretization.
4. Altintas, Y. (2012). *Manufacturing Automation*, 2nd ed. — milling forces.
5. Anderson & Moore (2007). *Optimal Control: Linear Quadratic Methods* — LQG.

## 💻 Computational cost (measured, 4-core container)

| Task | Time |
|---|---:|
| main_simulation.py (4 scenarios + SLD + sensor sweep) | ~5 min |
| main_imc_baseline.py | ~30 s |
| main_fullpath_comparison.py | ~10 min |
| Closed-loop Floquet SLD (30 × 25 grid, 2 curves) | ~10 s |

*Tested on Python 3.11, NumPy 2.4, SciPy 1.17 — results are bit-exact
reproducible (fixed seeds).*

## 📝 License & disclaimer

Academic research code, provided as-is. The honest-protocol and audit
documentation are part of the scientific record of this thesis — keep them
in sync with any code change.
