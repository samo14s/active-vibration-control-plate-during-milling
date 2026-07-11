# 05_main — Main Simulation Scripts

This directory contains the **main entry points** for running the simulation.

## Files

| File | Description | Output |
|---|---|---|
| `main_simulation.py` | Full LQG vs DARC-MPC comparison | Console + figs |
| `main_realistic_piezo.py` | Same but with piezo non-linearities | Console + figs |

## main_simulation.py

The **primary script** for reproducing the article results.

### What it does

1. Builds plate FEM model (Q4 × 30×24 elements, 3 modes)
2. Adds piezoelectric patch (QDA60-200.7, 20×60 mm)
3. Computes cutting force coefficients (3-tooth end-mill)
4. Builds **LQG controller** with sub-optimal weights
5. Builds **DARC-MPC controller** with optimal LQG base + NN
6. **Pre-trains the NN** via 30 iterations of iterative learning control
7. Runs **4 scenarios**:
   - **S1 Nominal**: a_p = 0.3 mm, K_T nominal, ω nominal
   - **S2 Aggressive**: a_p = 0.6 mm (twice the depth)
   - **S3 Uncertainty**: ω - 15% (frequency mismatch)
   - **S4 High K_T**: K_T + 30% (harder material)
8. Computes metrics: RMS, peak, peak-to-peak, voltage stats
9. Prints summary table

### Run

```bash
python main_simulation.py
```

### Expected output

```
========================================================================
 LQG vs DARC-MPC v3 COMPLETE COMPARISON
========================================================================

[Setup]
  Plate model         : 30 × 24 elements, 3 modes
  Modal frequencies   : [521, 1070, 2733] Hz
  Cutting parameters  : 4900 RPM, 0.3 mm axial, 0.1 mm radial
  
[Phase 1] Building controllers ...
  LQG (sub-optimal)   : w_q=1e13, w_qd=1e8 → ζ_1 = 13.2%
  DARC-MPC (optimal)  : w_q=1e14, w_qd=1e8 → ζ_1 = 23.9% (base)

[Phase 2] Pre-training DARC-MPC NN ...
  Iter  1/30 : RMS_residual = 0.587 µm
  Iter  5/30 : RMS_residual = 0.412 µm
  Iter 10/30 : RMS_residual = 0.298 µm
  Iter 30/30 : RMS_residual = 0.107 µm  ✓ converged

[Phase 3] Running 4 scenarios ...
  S1 Nominal              : LQG 0.628 → DARC 0.507 µm  (+19.20%)
  S2 Aggressive ap=0.6mm  : LQG 1.253 → DARC 1.009 µm  (+19.51%)
  S3 Uncertainty ω-15%    : LQG 0.604 → DARC 0.488 µm  (+19.22%)
  S4 High K_T +30%        : LQG 0.817 → DARC 0.661 µm  (+19.17%)
  ─────────────────────────────────────────────────────────────────
  AVERAGE                 : LQG 0.825 → DARC 0.666 µm  (+19.31%)

Done. Total time: ~4 minutes
```

### Modifying parameters

Edit the configuration section near the top of `main_simulation.py`:

```python
# ────── PHYSICAL PARAMETERS ──────
LP = 0.100              # plate length (m)
HP = 0.080              # plate height (m)
BP = 0.004              # plate thickness (m)
RPM = 4900              # spindle speed
FT = 0.02e-3            # feed per tooth (m)
AP = 0.3e-3             # axial engagement (m)
AE = 0.1e-3             # radial engagement (m)

# ────── LQG WEIGHTS ──────
LQG_W_Q = 1e13          # state penalty (sub-optimal)
LQG_W_QD = 1e8          # state derivative penalty
LQG_W_R = 1.0           # control penalty

# ────── DARC-MPC PARAMETERS ──────
DARC_BASE_W_Q = 1e14    # base LQG (optimal)
DARC_FF_LR = 5e-3       # NN learning rate
DARC_FF_MAX = 10.0      # FF saturation (V)
DARC_N_ITER = 30        # ILC iterations
DARC_N_EPOCHS = 15      # epochs per iteration
```

## main_realistic_piezo.py

Same as main_simulation.py but uses a **non-linear piezo model** with:
- Hysteresis (Bouc-Wen model)
- Rate-dependent saturation
- Temperature drift compensation

This shows that DARC-MPC remains effective even with realistic actuator
non-linearities (typical for industrial piezoelectric actuators).

### Run

```bash
python main_realistic_piezo.py
```

Expected results: ~+15% improvement (slightly less than ideal case due to
non-linearities, but still very significant).

## Tips

1. **For first run**: use `main_simulation.py` (faster, ideal piezo)
2. **For thesis defense**: use `main_realistic_piezo.py` (more realistic)
3. **For figure regeneration**: use scripts in `04_figures/`
4. **For SLD analysis only**: use `gen_SLD_academic_style.py` directly

---

## main_gap_spindle_sync.py (research-gap experiment)

Quantifies the sensitivity of the learned feedforward to spindle-speed
uncertainty and demonstrates the v4 PLAD fix (see `docs/research_gap.md`).

Scenarios (controllers designed/trained at nominal speed only):
- **A1–A6** — constant effective speed offsets 0 / ±1.2 / +2.5 % at
  a_p = 0.3 and 0.6 mm (tiled periodic coefficients);
- **B** — sinusoidal spindle-speed fluctuation ±1 % @ 2 Hz
  (phase-continuous coefficients), T = 1 s;
- **C** — long pass T = 4 s with +2.5 % offset (sustained lock while the
  tool advances; position-scheduled phase reference).

```bash
python 05_main/main_gap_spindle_sync.py   # ~1 min, writes results_gap_sync/
python 03_analysis/validate_phase_observer.py   # phase-observer test suite
```

Outputs: `results_gap_sync/` — 5 figures, `metrics.json`, `summary.md`.
