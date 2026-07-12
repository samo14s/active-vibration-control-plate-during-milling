# Article Simulation Package — Active Vibration Control of Thin-Walled Milling

**Topic**: Comparison between LQG and DARC-MPC controllers for chatter mitigation in peripheral milling of cantilever AL6061 plates.

**Target journals**: IEEE TCST · MSSP · Mechatronics · Automatica · CIRP Annals · JSV

---

## 🔄 Modeling — now following Nasiri & Moradi (MSSP 224 (2025) 112198)

The **physical plant model** has been switched from the previous FEM / linear
formulation to the **nonlinear analytical modeling** of *Nasiri & Moradi,
"Chatter suppression in nonlinear milling of a flexible plate-workpiece with
attached piezoelectric actuators", Mechanical Systems and Signal Processing 224
(2025) 112198*:

| Aspect | Before (FEM / linear) | Now (article's nonlinear modeling) |
|---|---|---|
| Plate kinematics | Kirchhoff Q4 finite elements | **Von Kármán** geometric nonlinearity (Eq. 18-19) |
| Modal reduction | FEM eigen-analysis | **Galerkin + mode summation** (Eq. 32-35): clamped-free φₘ × free-free ψₙ |
| Modal equation | Linear `M q̈+C q̇+K q` | **Nonlinear NDDE** `η̈+2(ζ+ζₚ)ωη̇+ω²η+λη³ = …` (Eq. 36) |
| Cutting force | Linearized coefficient | **3rd-order polynomial** in chip thickness ξ₁h³+ξ₂h²+ξ₃h+ξ₄ (Eq. 4/16) |
| Damping | Structural only | + **process damping** ζₚ and **tool flank wear** (Eq. 39, 44-46) |
| Large amplitude | — | **tool–workpiece separation** (Eq. 6, §4.1) |

**Main values are preserved.** The analytical Galerkin frequencies for this
plate (528 / 1165 / 2657 Hz) are calibrated to the original FEM values
**[521.06, 1069.95, 2733.02] Hz**; the AL6061 dimensions, AL6061 material,
3-tooth tool, QDA60 piezo, 4900 RPM, the cutting magnitude `K_T`, and the modal
damping ratios are kept unchanged. The Von Kármán cubic and the cutting
nonlinearity are **dormant at the nominal µm amplitudes** (the model then
reduces exactly to the previous linear backbone — e.g. LQG `y_RMS ≈ 0.63 µm` as
before) and only **activate at large chatter amplitudes**, exactly as in the
article.

**Controllers.** Only **LQG** and **DARC-MPC** are kept (model-based). They are
re-designed automatically from the modal model `(Mp, Kp, Cp, H_Pe_modal, D_obs)`
and run in closed loop on the new nonlinear plant.

## ⚖️ Fair LQG vs DARC-MPC — DARC's anticipative feedforward

`main_simulation.py` is set up so the comparison is **fair**: LQG and DARC-MPC use
the **identical LQG base** (`w_q=1e14`); DARC-MPC's *only* addition is an
**anticipative feedforward** that cancels the periodic tooth-passing excitation
(`design_periodic_feedforward`, inverse-model over the tooth-passing harmonics).
This is a genuine, theory-backed advantage: a feedback controller cannot fully
reject a periodic disturbance (phase lag), but a feedforward — synchronized to the
spindle **phase** — can. On top of this, a **neural network** learns the remaining
*nonlinear* residual (`train_nn_residual`, the "Deep" part of DARC). The result:
**DARC-MPC reduces RMS vibration ≈ 55 % below LQG in every scenario** (see Key
Results). Because the feedforward (model + NN) is phase-based, it is **independent
of the displacement sensor**, so DARC also stays ahead as the sensor degrades
(`fig10_sensor_robustness`). A fine sensor (0.1 µm noise) is used; the solver also
supports `sensor_floor` / `sensor_noise` to model coarser sensors.

---

## 📂 Package Structure

```
article_simulation_package/
│
├── 01_core/              ← Physical models (analytical Galerkin + nonlinear dynamics)
│   ├── kirchhoff_q4.py           # LEGACY Kirchhoff Q4 FEM (no longer used)
│   ├── plate_model.py            # Von Kármán + Galerkin nonlinear plate (Nasiri & Moradi)
│   ├── piezo_actuator.py         # Piezoelectric actuator model (QDA60-200.7)
│   ├── milling_force.py          # Nonlinear 3rd-order cutting force (3-tooth end-mill)
│   └── newmark_solver.py         # Nonlinear NDDE Newmark-β integrator
│
├── 02_controllers/        ← Control algorithms (LQG, DARC-MPC)
│   ├── lqg_controller.py         # LQG with Kalman observer (model-based)
│   └── darc_mpc_v3_controller.py # DARC-MPC: LQG + NN feedforward (model-based)
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
│   ├── main_simulation.py        # Fair LQG vs DARC-MPC (4 scenarios, 0.5 s)
│   ├── main_fullpath_comparison.py  # LQG vs DARC vs DARC-MPC over the full feed pass (20.4 s)
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

#### `kirchhoff_q4.py` — **LEGACY (no longer used)**
Former Kirchhoff plate Q4 (4-node quadrilateral) finite element with 3 DOF/node
(w, θ_x, θ_y). Superseded by the analytical Galerkin modeling in
`plate_model.py`; kept only as a reference for the original FEM discretization
whose frequencies serve to calibrate the "main values".

#### `plate_model.py` — **analytical Von Kármán / Galerkin plate (Nasiri & Moradi)**
Analytical modal assembly (no FEM):
- Mode shapes = clamped-free `φ_m(Z/H_P)` × free-free `ψ_n(X/L_P)` products (Eq. 33-35);
  for this plate the lowest 3 are (1,1) bending, (1,2) torsion, (1,3) width-bending
- Natural frequencies by Galerkin/Rayleigh quotient, **calibrated** to the
  preserved package values [521.06, 1069.95, 2733.02] Hz (`freq_calib`)
- **Von Kármán cubic coefficients** `lam_modal` (λ_i, Eq. 30/36) by Galerkin
  projection of the nonlinear plate terms
- Piezo coupling `H_Pe_modal = B_piezo·∫∫∇²W` (Eq. 30-31)
- `set_process_damping(Ω, Γ)` → process-damping ζ_p added to `Cp` (Eq. 44-46)
- Pre-computation of `Dp(X)` along the tool path; same public interface as before

#### `piezo_actuator.py`
QDA60-200.7 piezoelectric patch model:
- d₃₁ formulation (transverse mode)
- Stress-charge constitutive equation
- Linear voltage-to-moment mapping (with optional non-linearity)

#### `milling_force.py` — **nonlinear cutting force (Nasiri & Moradi, Eq. 4/16)**
Cutting force model for the 3-tooth end-mill:
- `precompute_alpha_periodic()` → linear backbone `α₃` (mean), `α₄` (linear
  regenerative coeff, ∝ `K_T`) — **unchanged**, preserves the main values
- `precompute_nonlinear_periodic()` → also returns the **quadratic/cubic**
  regenerative coefficients `α₄₂ = α₄·(ξ₂/ξ₃)`, `α₄₃ = α₄·(ξ₁/ξ₃)` from the
  article's full 3rd-order chip-thickness polynomial (Table 2 ratios)
- `tool_wear_edge_force()` → flank-wear edge term ξ₄ ∝ VB (Eq. 39)
- `chip_separation_factor()` → tool–workpiece separation (Eq. 6)

#### `newmark_solver.py` — **nonlinear NDDE integrator**
Newmark-β time integration (β = 1/4, γ = 1/2) of the nonlinear delayed
modal equation (Eq. 36):
- Linear regenerative backbone treated **implicitly** (identical to before →
  nominal behaviour preserved)
- Adds **Von Kármán cubic** `λ·η³`, **nonlinear cutting** `α₄₂Δ²+α₄₃Δ³`,
  **separation** and **chip saturation** (bounded chatter limit cycle)
- dt = 5×10⁻⁵ s (50 µs); process damping read from `plate.Cp`
- Controller interface (LQG / DARC-MPC `step(...)`) **unchanged**

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
u(t) = u_LQG(x̂)  +  α · u_FF(φ)
```
reactive feedback (= standalone LQG)  +  anticipative feedforward  +  NN residual.

**Components**:
1. **Reactive baseline**: LQG controller (= the standalone LQG → fair comparison)
2. **Anticipative feedforward** (`design_periodic_feedforward`): cancels the
   periodic tooth-passing excitation via an **inverse-model design** —
   `U_FF(ω_h) = −G_wy(ω_h)·W(ω_h)/G_uy(ω_h)` at each tooth-passing harmonic ω_h,
   with `G_wy, G_uy` the closed-loop disturbance→y and command→y responses and
   `W = FFT(f_t·a3)`. Phase-synchronized (spindle encoder) → **sensor-independent**.
3. **Neural-network residual** (`train_nn_residual`, the "Deep" part): a small NN
   `u_NN(φ, x̂)` learns, by iterative learning, the **nonlinear residual** that the
   *linear* inverse-model leaves — adding a further ~15–25 % reduction. Best-policy
   checkpointing guarantees it never degrades the inverse-model result.
4. **Lyapunov safety filter**: kept for the pure-NN mode.

**Feedforward design + NN residual training**:
```python
Dp_ff = plate.get_Dp_at(kp_mid)[0]                 # mode shape at the tool
darc.design_periodic_feedforward(FT, alpha3[:n_per], Dp_ff, n_harm=30)
darc.train_nn_residual(sim, alpha3, alpha4, kp_idx,
                       alpha4_2_t=a42, alpha4_3_t=a43, n_iter=20)   # learns the residual
```

Why it beats LQG fairly: a feedback controller cannot reject a periodic
disturbance (phase-lag / waterbed limit); the feedforward anticipates and cancels
it; a neural network (`train_nn_residual`) then learns the nonlinear residual on
top → ≈ 55 % lower RMS than the same LQG base, sensor-independent.

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
1. Build plate model (analytical Galerkin modal reduction)
2. Build LQG controller (w_q=1e14 base, **= DARC's base** → fair comparison)
3. Build DARC-MPC controller (same LQG base + anticipative feedforward)
4. Design DARC feedforward (`design_periodic_feedforward`) + train the NN
   residual (`train_nn_residual`, the "Deep" nonlinear correction)
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

> **FAIR comparison — DARC-MPC's genuine advantage.** LQG and DARC-MPC share the
> **identical LQG base** (`w_q=1e14, w_qd=1e8, R=1`, unconstrained); DARC-MPC's
> *only* addition is an **anticipative feedforward** that cancels the periodic
> tooth-passing excitation (inverse-model design over the tooth-passing harmonics,
> `design_periodic_feedforward`), **and a neural network** that learns the
> remaining *nonlinear* residual on top (`train_nn_residual` — the "Deep" part of
> DARC). A feedback controller cannot cancel a periodic disturbance (phase-lag /
> waterbed limit) — the anticipative feedforward can. Run on the nonlinear NDDE
> plant with a **fine displacement sensor** (0.1 µm RMS noise); modal
> frequencies/damping preserved exactly ([521.06, 1069.95, 2733.02] Hz; ζ = [0.31,
> 0.17, 0.27] %).

### RMS Vibration (T = 0.5 s) — fair comparison (same LQG base + DARC feedforward + NN)

| Scenario | LQG | DARC-MPC | DARC gain |
|---|---:|---:|---:|
| S1 Nominal | 0.605 µm | **0.295 µm** | **+51.2 %** |
| S2 Aggressive | 1.206 µm | **0.571 µm** | **+52.6 %** |
| S3 Uncertainty (ω−15%) | 0.923 µm | **0.338 µm** | **+63.4 %** |
| S4 High K_T | 0.788 µm | **0.374 µm** | **+52.6 %** |
| **AVERAGE** | **0.881 µm** | **0.395 µm** | **+55.2 %** 🏆 |

> **DARC-MPC reduces vibration ~55 % below LQG in every scenario**, fairly (same
> base controller). The advantage builds in two genuine, theory-backed layers: the
> inverse-model feedforward cancels the *periodic* regenerative excitation the
> feedback-only LQG cannot (~39 %), and the neural network learns the *nonlinear*
> residual on top (a further ~15–25 %, largest under detuning, S3 +63 %).

### Sensor robustness — the feedforward does not need the sensor (`fig10`)

DARC's feedforward (model + NN) is synchronized to the **spindle phase** (encoder),
so it is **independent of the displacement sensor**. As the sensor degrades
(noise ↑), the feedback-only LQG worsens steeply while DARC keeps its edge:

| Sensor noise | LQG | DARC-MPC | DARC gain |
|---|---:|---:|---:|
| ideal | 0.604 µm | 0.293 µm | +51 % |
| 0.6 µm | 0.657 µm | 0.390 µm | +41 % |
| 1.0 µm | 0.736 µm | 0.510 µm | +31 % |
| 2.0 µm | 1.022 µm | 0.869 µm | +15 % |

→ DARC-MPC is **more robust to sensor quality** — a second fair argument for it.

### Full feed pass — three-way ablation (`main_fullpath_comparison.py`, `fig fullpath`)

Run over the **entire feed path** (the tool travels the whole width `L_P`,
`T_path = L_P/v_feed ≈ 20.4 s`, 408 k time steps), the three controllers stay
ordered and stable the whole pass (the single-position feedforward is adequate
because the dominant chatter mode's shape `Dp` is constant along the width):

| Controller (full path, T = 20.4 s) | RMS | vs LQG |
|---|---:|---:|
| **LQG** (feedback) | 0.471 µm | — |
| **DARC** (LQG + feedforward) | 0.286 µm | **+39.4 %** |
| **DARC-MPC** (LQG + feedforward + NN) | 0.267 µm | **+43.4 %** |

→ The three-layer ablation **LQG < DARC < DARC-MPC** holds over the full machining
pass: the model-based feedforward provides the bulk (+39 %), the NN residual adds
on top (+43 % total). (The NN's marginal gain is smaller over the full path than on
a 0.5 s window because it is trained on one representative segment.)

### Stability Domain (SLD at 4900 RPM)

| Method | a_p critical | vs Open-Loop |
|---|---:|---:|
| Open-Loop | 0.10 mm | 1× |
| LQG | 2.86 mm | 28.6× |
| DARC-MPC | 4.00 mm | 40× |

> ⚠️ **Note:** the SLD is from the linear-stability tool (`fdm_stability.py`); the
> DARC curve uses a *heuristic* effective-damping boost to represent the feedforward
> (an assumption, not a Floquet result for the feedforward-controlled NDDE). The
> directly **measured** advantage is the time-domain RMS (≈ 55 % below LQG, above),
> which is the rigorous fair result; treat the SLD numbers as indicative.

### Modal Damping (Mode 1, 521 Hz) — preserved

| Configuration | ζ |
|---|---:|
| Open-Loop | 0.31% (exact) |
| LQG (optimal weights, = DARC base) | ≈ 28% |
| DARC-MPC | base + anticipative feedforward |

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
  note    = {Under a FAIR comparison (identical LQG base, the only additions being
             DARC's anticipative inverse-model feedforward and a neural network that
             learns the nonlinear residual), DARC-MPC reduces RMS vibration ~55% below
             LQG across all scenarios by cancelling the periodic tooth-passing
             excitation that feedback cannot reject; the phase-based feedforward is
             sensor-independent, so DARC also stays superior as the sensor degrades.}
}
```

### Key references

1. **Nonlinear plate / NDDE modeling (this package's plant model)**:
   Nasiri, K. & Moradi, H. (2025). "Chatter suppression in nonlinear milling of a
   flexible plate-workpiece with attached piezoelectric actuators: Comparison of
   soft-actor-critic-based controller vs optimized type-2 fuzzy controller".
   *Mechanical Systems and Signal Processing* **224**: 112198.
2. **Von Kármán plate theory**: Reddy, J.N. (2007). *Theory and Analysis of Elastic Plates and Shells*.
3. **LQG**: Anderson & Moore (2007). *Optimal Control: Linear Quadratic Methods*.
4. **FDM/Floquet**: Insperger, T. & Stépán, G. (2004). "Updated semi-discretization 
   method for periodic delay-differential equations". *Int. J. Numer. Meth. Eng.* 61: 117–141.
5. **Milling forces**: Altintas, Y. (2012). *Manufacturing Automation*. 2nd ed.
6. **FEM (legacy)**: Bathe, K.-J. (2014). *Finite Element Procedures*. Prentice Hall.

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
