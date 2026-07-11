# Article Simulation Package — Active Vibration Control of Thin-Walled Milling

**Topic**: Comparison between LQG and DARC-MPC controllers for chatter mitigation in peripheral milling of cantilever AL6061 plates.

**Target journals**: IEEE TCST · MSSP · Mechatronics · Automatica · CIRP Annals · JSV

---

## 📂 Package Structure

```
article_simulation_package/
│
├── 01_core/              ← Physical models (FEM + dynamics)
│   ├── kirchhoff_q4.py           # Kirchhoff Q4 plate element FEM
│   ├── plate_model.py            # Plate assembly + modal reduction
│   ├── piezo_actuator.py         # Piezoelectric actuator model (QDA60-200.7)
│   ├── milling_force.py          # Cutting force model (3-tooth end-mill)
│   ├── newmark_solver.py         # Newmark-β time integration (linear)
│   ├── von_karman_rom.py             # Geometrically nonlinear (von Kármán) ROM ★
│   └── newmark_nonlinear_solver.py   # Newmark–Newton-Raphson (nonlinear) ★
│
├── 02_controllers/        ← Control algorithms
│   ├── lqg_controller.py              # LQG with Kalman observer
│   ├── darc_mpc_v3_controller.py      # DARC-MPC: LQG + NN feedforward
│   └── darc_mpc_v4_plad_controller.py # DARC-MPC v4 PLAD: phase-locked FF ★
│
├── 03_analysis/           ← Stability & robustness analysis
│   ├── fdm_stability.py          # Floquet multipliers (FDM, Insperger-Stépán)
│   ├── uncertainty_analysis.py   # Monte Carlo robustness analysis
│   ├── validate_phase_observer.py # v4 phase-observer test suite ★
│   └── validate_von_karman.py     # von Kármán ROM validation (backbone) ★
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
│   ├── main_realistic_piezo.py   # With realistic piezo non-linearities
│   ├── main_gap_spindle_sync.py  # Spindle-speed-uncertainty experiment ★
│   └── main_geometric_nonlinear.py # von Kármán geometric-nonlinearity study ★
│
├── docs/
│   ├── research_gap.md          ← Spindle-sync gap (PLAD) + literature ★
│   ├── verrou_scientifique.md   ← French thesis framing (PLAD) ★
│   └── verrou_nonlinearite.md   ← French thesis framing (von Kármán) ★
│
├── results_gap_sync/      ← Output of the spindle-sync experiment ★
├── results_geom_nl/       ← Output of the geometric-nonlinearity study ★
│
└── README.md             ← This file
```

★ = research-gap contribution (DARC-MPC v4 PLAD), see the section
"Research-gap contribution" below and `docs/research_gap.md`.

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

#### `kirchhoff_q4.py`
Kirchhoff plate Q4 (4-node quadrilateral) finite element with 3 DOF/node
(w, θ_x, θ_y). Provides:
- Element stiffness matrix `K_elem`
- Element mass matrix `M_elem` (consistent and lumped options)
- Shape functions for displacement and rotation

#### `plate_model.py`
Full plate assembly:
- Q4 mesh generation (N1 × N2 elements, default 30 × 24 = 720 elements)
- Cantilever boundary conditions (clamped at z = 0)
- Modal reduction to N_modes (default 3)
- Pre-computation of D_p(x_p) shape function values along tool path
- Piezo patch addition with stiffness/mass coupling

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

## ⭐ Research-gap contribution 2 — Geometric nonlinearity (von Kármán)

Active-control-of-thin-wall-milling models are **linear** (Kirchhoff/Mindlin
+ modal reduction). Yet at aggressive cuts the wall vibrates at a
significant fraction of its thickness, where von Kármán geometric
nonlinearity matters. A geometrically nonlinear, **FE-consistent** reduced
model (`01_core/von_karman_rom.py`, validated against the classical clamped-
plate backbone ω_nl/ω_l ≈ 1.17 at A/h=1) with a Newmark–Newton-Raphson
solver shows:

- **Bifurcation**: past the stability limit the *linear* model diverges
  exponentially, while von Kármán **bounds** the chatter into a **limit
  cycle** (Hopf) — the linear model cannot predict the post-critical
  amplitude.
- **Control**: the same LQG (±150 V) designed on the linear model **diverges
  on the linear plant** (design would reject the actuator) but is **bounded
  and stabilised on the true von Kármán plant** — the linear model gives a
  qualitatively wrong control assessment / actuator sizing.
- **Criterion**: the effect is strong for an in-plane-restrained "wall"
  (ω_nl/ω_l=1.27 @ A/h=1) and negligible for a free-edge cantilever
  (≈1.00) — delineating when linear models suffice.

Aligned with the thesis rapporteur's expertise (nonlinear smart-shell FE).
Details: `docs/verrou_nonlinearite.md`. Reproduce:

```bash
python 05_main/main_geometric_nonlinear.py       # ~2 min → results_geom_nl/
python 03_analysis/validate_von_karman.py        # validation suite
```

---

## ⭐ Research-gap contribution 1 — DARC-MPC v4 "PLAD"

The v3 feedforward is indexed by an **open-loop clock** `k mod n_per`
(exactly known, constant spindle speed assumed) and its adaptation factor
was computed but never applied. Realistic spindle-speed deviations
(droop under load, fluctuation, deliberate SSV) desynchronise the learned
feedforward, which then injects voltage at the wrong phase.

**DARC-MPC v4 PLAD** (`02_controllers/darc_mpc_v4_plad_controller.py`)
closes this gap with a *sensorless* synchronisation layer:

```
u(t) = u_LQG(x̂) + α · c_lock(t) · NN_FF(φ̂(t), x̂)
```

- band-pass + digital **PLL** locks onto the tooth-passing fundamental in
  the displacement measurement (no spindle encoder);
- **model-based phase referencing** (closed-loop FRF, scheduled over tool
  position via the solver's `enable_gs` hook, one-shot calibration);
- **confidence-gated feedforward**: graceful fallback to pure LQG when
  lock is lost — the adaptation is actually wired into the control law.

Measured impact (steady state, identical trained NN weights,
`05_main/main_gap_spindle_sync.py`):

| Speed error (effective) | DARC v3 gain vs LQG | DARC v4 gain vs LQG |
|---|---:|---:|
| 0 % (nominal, ap=0.3) | +4.6 % | +4.7 % |
| +1.23 % | −0.1 % | +4.7 % |
| +2.50 % | +0.8 % | +5.6 % |
| −1.20 % | −0.2 % | +4.8 % |
| ±1 % sinusoidal @ 2 Hz | −0.3 % | +5.1 % |
| +2.5 %, long pass 4 s | +0.8 % | +6.8 % |
| +2.5 %, noisy sensor (0.1 µm RMS) | +1.0 % | +5.6 % |
| +9.3 % (beyond pull-in ±7 %) | +1.7 % | −0.1 % (graceful fallback to LQG) |

A 1–2.5 % spindle-speed error **erases the learned-feedforward benefit**
of v3 (and wastes control effort injecting at the wrong phase); v4
retains the benefit consistently (lock confidence 0.98–1.00, measured
lock time 0.15–0.17 s) with no steady-state cost at nominal speed.
Beyond the PLL pull-in range the confidence gate retracts the
feedforward and v4 coincides with the LQG baseline. Full analysis,
disclosures and literature grounding: `docs/research_gap.md`;
reproducible via:

```bash
python 05_main/main_gap_spindle_sync.py         # ~2 min
python 03_analysis/validate_phase_observer.py   # test suite (12 checks)
```

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
