# Article Simulation Package — Active Vibration Control of Thin-Walled Milling
### Reissner–Mindlin plate edition

**Topic**: Comparison between LQG and DARC-MPC controllers for chatter mitigation in peripheral milling of cantilever AL6061 plates.

**Target journals**: IEEE TCST · MSSP · Mechatronics · Automatica · CIRP Annals · JSV

> **Plate theory in this edition: Reissner–Mindlin.** This is the article
> simulation package (package 12) with its plate FEM replaced by the **8-node
> Serendipity Mindlin element**, ported *literally* from the MATLAB repository
> `Plate-FEM/Mindlin_plate`. The `PlateModel` public interface is unchanged, so
> **every other module (controllers, Newmark solver, FDM stability, figures)
> runs exactly as before** — only the plate theory differs (Kirchhoff → Mindlin).
> See [`01_core/MINDLIN_PORT.md`](01_core/MINDLIN_PORT.md) for the `.m` ↔ `.py`
> correspondence, [`docs/GEOMETRY.md`](docs/GEOMETRY.md) for the setup-diagram ↔
> model mapping, and [`tests/`](tests/) for the validation suite (CCCC benchmark
> error 0.02% vs Leissa; cantilever mode 1 = 519 Hz vs the article's ~521 Hz;
> full setup-geometry conformance).

---

## 📂 Package Structure

```
article_simulation_package/
│
├── 01_core/              ← Physical models (FEM + dynamics)
│   ├── mindlin_q8.py             # Reissner–Mindlin Q8 plate element FEM (literal Plate-FEM port)
│   ├── kirchhoff_q4.py           # Kirchhoff Q4 plate element FEM (reference only, unused)
│   ├── plate_model.py            # Plate assembly + modal reduction (Mindlin)
│   ├── inprocess_plate.py        # In-process material removal (time-varying dynamics)
│   ├── digital_twin.py           # ⭐ physics-guided predictive–corrective digital twin
│   ├── piezo_actuator.py         # Piezoelectric actuator model (QDA60-200.7)
│   ├── milling_force.py          # Cutting force model (3-tooth end-mill)
│   └── newmark_solver.py         # Newmark-β time integration
│
├── 02_controllers/        ← Control algorithms
│   ├── lqg_controller.py         # LQG with Kalman observer
│   ├── rcsac_controller.py       # ⭐ RC-SAC: novel regeneration-cancelling
│   │                             #    saturation-aware control (this work)
│   └── darc_mpc_v3_controller.py # DARC-MPC: LQG + NN feedforward
│
├── 03_analysis/           ← Stability & robustness analysis
│   ├── fdm_stability.py          # Floquet multipliers (FDM, Insperger-Stépán)
│   └── uncertainty_analysis.py   # Monte Carlo robustness analysis
│
├── 04_figures/            ← Publication-quality figure generators
│   ├── gen_article_complete_figures.py    # 14 main figures
│   ├── gen_SLD_academic_style.py          # SLD academic style
│   ├── gen_geometry_figure.py             # Setup geometry (3D + views)
│   ├── gen_geometry_custom.py             # Customizable geometry
│   └── gen_control_strategy_diagram.py    # Control architecture diagram
│
├── 05_main/               ← Main simulation scripts
│   ├── main_simulation.py        # Full LQG vs DARC-MPC comparison
│   └── main_realistic_piezo.py   # With realistic piezo non-linearities
│
└── README.md             ← This file
```

---

## 🎯 Quick Start

### Prerequisites

```bash
pip install numpy scipy matplotlib
```

### Setup

All Python files must be in the **same directory** to allow imports.
Concatenate the package or use a single working folder:

```bash
# Option 1: flat structure (recommended for first run)
cp 01_core/*.py 02_controllers/*.py 03_analysis/*.py 04_figures/*.py 05_main/*.py ./
```

### Run main simulation

```bash
# Full simulation (LQG vs DARC-MPC, 4 scenarios, ~4 min)
python main_simulation.py

# Generate all 14 publication figures (~4 min)
python gen_article_complete_figures.py

# Generate SLD with FDM/Floquet method
python gen_SLD_academic_style.py

# Generate geometry figures
python gen_geometry_figure.py

# Generate control architecture diagram
python gen_control_strategy_diagram.py
```

---

## 📐 Physical Setup

### Plate (AL6061, vertical cantilever)

| Parameter | Symbol | Value |
|---|---|---:|
| Length | L_P | 100 mm |
| Height | H_P | 80 mm |
| Thickness | B_P | 4 mm |
| Density | ρ | 2830 kg/m³ |
| Young's modulus | E | 69 GPa |
| Poisson ratio | ν | 0.33 |
| Damping (Mode 1, 2, 3) | ζ₁, ζ₂, ζ₃ | 0.31%, 0.17%, 0.27% |

### Piezo patch (QDA60-200.7)

| Parameter | Symbol | Value |
|---|---|---:|
| Position (X) | — | 0–20 mm |
| Position (Z) | — | 0–60 mm (lower-left) |
| Thickness | h_Pa | 0.7 mm |
| Piezo coefficient | d₃₁ | 175 pm/V |
| Young's modulus | E_Pe | 63 GPa |
| Voltage saturation | u_max | ±150 V |

### Tool (end-mill, peripheral milling)

| Parameter | Symbol | Value |
|---|---|---:|
| Diameter | D | 10 mm |
| Number of teeth | N_T | 3 |
| Helix angle | η | 35° |
| Rake angle | γ_n | 15° |
| Friction coefficient | μ_c | 0.20 |
| Tangential cutting coeff | K_T | 925 MPa |
| Normal cutting ratio | k_N | 0.26 |

### Cutting parameters

| Parameter | Symbol | Value |
|---|---|---:|
| Spindle speed | Ω | 4900 RPM |
| Feed per tooth | f_t | 0.02 mm/tooth |
| Axial engagement | a_p | 0.3 mm |
| Radial engagement | a_e | 0.1 mm |
| Feed velocity | v_feed | 4.9 mm/s |
| Path duration | T_path | 20.4 s |

---

## 🔬 Module Descriptions

### 01_core — Physical Models

#### `mindlin_q8.py`  ⭐ (this edition)
Reissner–Mindlin plate **Q8 (8-node Serendipity)** finite element with 3 DOF/node
(w, θ_x, θ_y) → 24 DOF/element. **Literal port** of `Plate-FEM/Mindlin_plate`.
Provides:
- Shape functions `shape_function_M` and derivatives (Serendipity)
- Bending / shear strain matrices `matrix_der_M` → (`Bf`, `Bs`)
- Element stiffness `stiffness_matrix_M` = `h·(Kf + Ks)` (uniform reduced 2×2 Gauss)
- Consistent mass `mass_matrix_M` with rotary inertia `diag(1, h²/12, h²/12)`
- Thermal load `thermal_stress_M`
- Serendipity mesh numbering, point evaluation, and piezo moment-load helpers

See [`MINDLIN_PORT.md`](01_core/MINDLIN_PORT.md) for the verbatim `.m` ↔ `.py` map.

#### `inprocess_plate.py` — in-process material removal
`InProcessPlate` (subclass of `PlateModel`) raises the fidelity so the cutter
**removes material as it feeds along its path**: per-element thickness field,
`machine_to(x_tool, a_p, a_e)` removal law, `rebuild()`, `instantaneous_modes()`
and a fixed-basis ROM `reduced_on_basis(V0)`. With nothing removed it equals
`PlateModel`. See [`docs/MATERIAL_REMOVAL.md`](docs/MATERIAL_REMOVAL.md) and
`05_main/gen_material_removal_sim.py` (roughing pass: mode 1 519 → 542 Hz).

#### `kirchhoff_q4.py` (reference only)
Original Kirchhoff plate Q4 (4-node) element with Hermite shape functions.
Retained for Kirchhoff-vs-Mindlin comparison; **not imported** by `plate_model.py`.

#### `plate_model.py`
Full plate assembly (now **Mindlin**), same public `PlateModel` API:
- Serendipity Q8 mesh generation (N1 × N2 elements, default 30 × 24 = 720 elements)
- Cantilever boundary conditions (clamped edge at z = 0: w = θ_x = θ_y = 0)
- Modal reduction to N_modes (default 3)
- Pre-computation of D_p(x_p) shape function values along tool path
- Piezo patch addition via consistent Mindlin bending-moment coupling

#### `piezo_actuator.py`
QDA60-200.7 piezoelectric patch model:
- d₃₁ formulation (transverse mode)
- Stress-charge constitutive equation
- Linear voltage-to-moment mapping (with optional non-linearity)

#### `milling_force.py`
Cutting force model for 3-tooth end-mill:
- Per-tooth force computation with helix angle
- Lehmann-Engin model for tangential and normal forces
- Numba-compiled `precompute_alpha_periodic()` for fast simulation
- Returns α₃(t), α₄(t) coefficients used by FEM

#### `newmark_solver.py`
Newmark-β time integration scheme (β = 1/4, γ = 1/2):
- Average acceleration method (unconditionally stable)
- dt = 5×10⁻⁵ s (50 µs) for high-resolution
- Supports any controller via `controller.compute(x_hat, y_meas, k)`
- Tracks both physical states and observer estimates

### 02_controllers — Control Algorithms

#### `rcsac_controller.py`  ⭐ (novel strategy of this work)
**RC-SAC — Regeneration-Cancelling Saturation-Aware Control.** Targets the
root cause of chatter (the delayed regenerative force) instead of its energy:
1. **Exact decoupling filter** `C(s) = [DpᵀG(s)Dp]/[DpᵀG(s)H]` — dynamic
   inversion of the actuator→tool-point channel, resolving the opposite-sign
   static-gain conflict between modes (+85 vs −257) that defeats static
   cancellation;
2. **Cutting-force-aware Kalman predictor** — the known force model
   (α₃, α₄, τ) is injected into the observer propagation;
3. **Saturation-aware priority allocation** — the stability-critical
   feedforward gets the ±150 V budget first, LQG damping uses the headroom.

Worst-case result (RPM 4900, ±150 V): critical depth of cut
**0.40 (OL) → 2.05 (LQG) → 3.85 mm (RC-SAC)** — ×1.9 vs LQG, ×9.6 vs open loop,
with lower vibration and voltage than LQG at every depth. See
[`docs/RCSAC_STRATEGY.md`](docs/RCSAC_STRATEGY.md).

#### `lqg_controller.py`
Linear Quadratic Gaussian controller:
- LQR design via Riccati equation (`scipy.linalg.solve_continuous_are`)
- Kalman observer for state estimation
- Grid search over (w_q, w_qd, w_r) weights
- Discretization for real-time implementation

**Key parameters**:
- `w_q`: state penalty (default 1×10¹³ for sub-optimal, 1×10¹⁴ for optimal)
- `w_qd`: state derivative penalty (1×10⁸)
- `w_r`: control penalty (1.0)

#### `darc_mpc_v3_controller.py`
**DARC-MPC**: Deep Adaptive Robust Control with MPC, our novel method:

**Architecture**:
```
u(t) = u_LQG(x̂) + α · NN_FF(φ, x̂)
```

**Components**:
1. **Reactive baseline**: LQG controller (optimal weights)
2. **Anticipative feedforward**: Phase-aware Neural Network
   - Inputs: tool phase φ ∈ [0, 2π], state estimate x̂
   - Architecture: 3 → 16 → 16 → 1 (sigmoid activation)
   - Output: u_FF ∈ [-10, 10] V (saturated)
3. **Lyapunov safety filter**: rejects unsafe u* commands
4. **Iterative Learning Control**:
   - 30 iterations of simulation + NN training
   - Target: u_target = -K_corr · y_residual
   - Adam optimizer with lr = 5×10⁻³

**Pre-training**:
```python
darc.pretrain_iterative_simulation(
    simulator, alpha3, alpha4, kp_idx,
    n_iterations=30,
    n_epochs_per_iter=15
)
```

### 03_analysis — Stability & Robustness

#### `fdm_stability.py`
Stability Lobe Diagram (SLD) via Full-Discretization Method:

**Theory**: Insperger-Stépán (2004) method computes monodromy matrix Φ over
one tooth-passing period τ, then evaluates Floquet multipliers ρ = max|λ(Φ)|.
- Stable if ρ < 1
- Unstable (chatter) if ρ ≥ 1

**Implementation**:
- m_div = 40 subdivisions per period
- Augmented state for time delay handling
- Analytical 2×2 matrix exponential (fast)
- Multi-mode support (3 modes superimposed)

```python
rho_grid, _ = compute_SLD(
    RPM_array, ap_array,
    omega_n_list, zeta_list, Dp_list, m_list,
    NT, RT, eta_h, phi_st, phi_ex,
    k1, k2, kt, hp,
    m_div=40
)
```

#### `uncertainty_analysis.py`
Monte Carlo robustness analysis:
- Random sampling of (ω_n, ζ, K_T) within ±15%
- 100 simulation samples
- Statistical metrics: mean, std, 95% confidence intervals

### 04_figures — Figure Generators

All scripts produce **PNG (300 DPI) + PDF** in `figs_article_publication/`.

#### `gen_article_complete_figures.py`
Generates **14 main figures**:
1. Global summary (3-panel)
2. Temporal y(t) full path (4 scenarios)
3. Temporal u(t) full path (4 scenarios)
4. Time + FFT side-by-side (S1)
5. FFT y(t) (4 scenarios with annotations)
6. FFT u(t) (log scale)
7. Modal damping + complex plane poles
8. SLD 3 panels (OL/LQG/DARC)
9. SLD overlay
10. 6-panel multi-metric
11. DARC internal blocks
12. Tool position + envelope
13. Zoom 3 phases
14. Robustness comparison

#### `gen_SLD_academic_style.py`
Three SLD styles:
- Overlay (single panel with all 3 controllers)
- 3-panel (separated)
- Hatched (Insperger-Stépán style)

#### `gen_geometry_figure.py` and `gen_geometry_custom.py`
Setup geometry visualization:
- 3D isometric perspective
- Front view (Y=0 plane)
- Top view (Z=H_P plane, **key for peripheral milling**)

The `_custom.py` version has all parameters at the top for easy modification.

#### `gen_control_strategy_diagram.py`
Control architecture diagrams:
- Detailed block diagram (Fig. 15)
- Algorithm flowchart (Fig. 16)
- Compact summary (Fig. 17)

### 05_main — Simulation Scripts

#### `main_simulation.py`
Complete simulation pipeline:
1. Build plate model (FEM + modal reduction)
2. Build LQG controller (sub-optimal weights)
3. Build DARC-MPC controller (optimal LQG base + NN)
4. Pre-train NN via iterative learning (30 iterations)
5. Run 4 scenarios:
   - S1: Nominal (a_p = 0.3 mm)
   - S2: Aggressive (a_p = 0.6 mm)
   - S3: Uncertainty (ω - 15%)
   - S4: High K_T (+30%)
6. Compute metrics (RMS, peak, voltage, etc.)

#### `main_realistic_piezo.py`
Same pipeline but with realistic piezo non-linearities:
- Hysteresis modeling
- Rate-dependent saturation
- Temperature drift

---

## 📊 Key Results

### RMS Vibration Reduction (T = 0.5 s)

| Scenario | LQG | DARC-MPC | Gain |
|---|---:|---:|---:|
| S1 Nominal | 0.628 µm | 0.507 µm | **+19.20%** |
| S2 Aggressive | 1.253 µm | 1.009 µm | **+19.51%** |
| S3 Uncertainty | 0.604 µm | 0.488 µm | **+19.22%** |
| S4 High K_T | 0.817 µm | 0.661 µm | **+19.17%** |
| **AVERAGE** | **0.825 µm** | **0.666 µm** | **+19.31%** 🏆 |

### Stability Domain (SLD at 4900 RPM)

| Method | a_p critical | vs Open-Loop |
|---|---:|---:|
| Open-Loop | 0.14 mm | 1× |
| LQG | 2.17 mm | 15.5× |
| DARC-MPC | **3.05 mm** | **21.7×** (+41% vs LQG) |

### Modal Damping (Mode 1, 521 Hz)

| Configuration | ζ |
|---|---:|
| Open-Loop | 0.31% |
| LQG (sub-optimal) | 13.2% |
| DARC base (optimal LQG) | 23.9% |
| DARC-MPC effective | **31.1%** |

---

## 📚 Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{darcmpc2026,
  title   = {Deep Adaptive Robust Control with Phase-Aware Neural Feedforward 
             for Chatter Mitigation in Thin-Walled Milling},
  author  = {[Author names]},
  journal = {[Journal name]},
  year    = {2026},
  note    = {DARC-MPC achieves +19% RMS vibration reduction and +41% stability 
             domain extension compared to LQG baseline.}
}
```

### Key references

1. **FEM**: Bathe, K.-J. (2014). *Finite Element Procedures*. Prentice Hall.
2. **LQG**: Anderson & Moore (2007). *Optimal Control: Linear Quadratic Methods*.
3. **FDM/Floquet**: Insperger, T. & Stépán, G. (2004). "Updated semi-discretization 
   method for periodic delay-differential equations". *Int. J. Numer. Meth. Eng.* 61: 117–141.
4. **Milling forces**: Altintas, Y. (2012). *Manufacturing Automation*. 2nd ed.

---

## 💻 Computational Requirements

| Task | Time | Memory |
|---|---:|---:|
| Single simulation (T = 0.5 s, dt = 50 µs) | ~10 s | <500 MB |
| Single simulation full path (T = 20.4 s) | ~30 s | <1 GB |
| DARC-MPC pre-training (30 iter × 15 epochs) | ~3 min | <1 GB |
| SLD computation (60 × 50 = 3000 grid points) | ~80 s | <500 MB |
| Full 14-figure generation | ~4 min | <2 GB |

**Tested on**: Python 3.12, NumPy 1.26, SciPy 1.13, Matplotlib 3.9
**OS**: Linux/macOS/Windows

---

## 🛠️ Troubleshooting

### Common issues

**Issue**: `ImportError: No module named 'plate_model'`
**Solution**: All Python files must be in the **same directory**.

**Issue**: SLD takes too long
**Solution**: Reduce grid resolution in `gen_SLD_academic_style.py`:
```python
RPM_arr = np.linspace(2500, 7500, 30)  # was 60
ap_arr = np.linspace(0.005e-3, 4e-3, 30)  # was 60
```

**Issue**: NN training diverges
**Solution**: Reduce learning rate in DARC-MPC:
```python
darc = DARC_MPC_v3_Controller(plate_d, dt=dt,
    ff_lr=0.001,  # was 0.005
    ...
)
```

**Issue**: Controller saturates voltage
**Solution**: Use sub-optimal LQG weights:
```python
lqg.optimize_weights(w_q_list=[1e13],  # was 1e14
                     w_qd_list=[1e8], w_r=1.0)
```

---

## 📝 License & Disclaimer

This code is provided for academic research purposes. The authors make no
warranty regarding fitness for any particular purpose. Use at your own risk
in industrial applications.

---

## 📧 Contact

For questions or collaboration:
- [Author email]
- [Lab website]

---

*Last updated: April 2026*
