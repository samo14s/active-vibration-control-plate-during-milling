# 05_main — Main Simulation Scripts

Main entry points for running the simulation. **Authoritative numbers** come from
`main_simulation.py` and are logged in `docs/REPRODUCED_RESULTS.md`.

## Files

| File | Description | Output |
|---|---|---|
| `main_simulation.py` | LQG vs PALF-LQG comparison (train-once / held-out) + SLD | Console + figs |
| `main_robustness_mc.py` | Monte-Carlo robustness (50 samples, divergence reported) | Console + fig |
| `main_realistic_piezo.py` | LQG with realistic piezo non-linearities | Console + figs |

## main_simulation.py

The **primary script**. Comparison protocol (see `docs/AUDIT_FINDINGS.md`):

1. Builds plate FEM model (Q4 × 30×24 elements, 3 modes).
2. Adds piezoelectric patch (QDA60-20-0.7, 20×60 mm).
3. Computes cutting force coefficients (3-tooth end-mill).
4. Builds a **full-order 5-mode plant**; the controllers are designed on the **first 3
   modes** (spillover — no inverse crime). One eigensolve feeds both.
5. Builds **one shared LQG feedback** (grid-searched weights) used by both the baseline
   LQG and PALF's internal LQR — a **symmetric** comparison — with 10 nm measurement
   noise and identical ±150 V clipping.
6. Builds **PALF-LQG** = the shared LQG + a phase-locked learned feedforward, and
   **pre-trains the feedforward ONCE on the nominal scenario (S1), then freezes it**.
7. Runs **4 scenarios**, evaluating the *frozen* controllers (S2/S3/S4 are held-out):
   - **S1 Nominal**: a_p = 0.3 mm, K_T nominal, ω nominal
   - **S2 Aggressive**: a_p = 0.6 mm
   - **S3 Uncertainty**: ω − 8 % (mismatch; ~−9 % is the controlled stability margin)
   - **S4 High K_T**: K_T + 30 %
8. Computes metrics (RMS, peak, peak-to-peak, voltage) and prints the summary.

### Run

```bash
python main_simulation.py
```

### Verified output (committed code, ~62 s)

```
========================================================================
 RÉSUMÉ FINAL : LQG vs PALF-LQG
========================================================================
  Scénario                    LQG y_RMS     PALF y_RMS    Gain
  S1 - Nominal article        0.7765        0.6252        +19.48%
  S2 - Aggressive ap=0.6mm    1.5580        1.3864        +11.01%
  S3 - Uncertainty ω-8%       0.9001        0.7689        +14.58%   <- model mismatch
  S4 - High K_T +30%          1.0127        0.8533        +15.75%
  MOYENNE                     1.0618        0.9084        +14.44%

  STABILITÉ (SLD, closed-loop monodromy, worst of 3 positions) - à RPM = 4900 :
     a_p crit OPEN-LOOP : 0.100 mm   (matches Du et al. 2024 experiment)
     a_p crit LQG       : 1.075 mm   (10.8x OL)
     a_p crit PALF-LQG  : 1.075 mm   (= LQG rigorously; du_FF/dx_hat = 0)
```

Monte-Carlo robustness (`main_robustness_mc.py`, 50 samples): PALF beats LQG in 100 %
of samples, median RMS gain **+17.8 %** [p05 +15.8 %, p95 +19.5 %], all converged.

The learned feedforward helps most under **model mismatch** (S3), because it is indexed
to the tooth-passing phase rather than to the (wrong) feedback model. It does not extend
the stability lobe — a phase-locked feedforward changes the periodic forcing, not the
closed-loop poles.

### Key parameters (top of `main_simulation.py`)

```python
LP, HP, BP = 0.100, 0.080, 0.004   # plate length/height/thickness (m)
RPM = 4900                         # spindle speed
FT  = 0.02e-3                      # feed per tooth (m)
AE  = 0.1e-3                       # radial engagement (m)

# One shared LQG design (grid search), reused by the baseline and by PALF:
LQG_SHARED.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                            w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)

# PALF feedforward: ff_lr=0.005, ff_max=10.0 V, 30 ILC iterations, trained once + frozen
```

## main_realistic_piezo.py

LQG with a realistic piezo model (saturation, slew rate, amplifier bandwidth,
hysteresis, sensor noise/delay). Uses the corrected Eq. (3) constants via
`cutting_constants`; it is an LQG-only actuator study (no PALF, no 5-mode spillover).

```bash
python main_realistic_piezo.py
```

## Tips

1. **First run / authoritative numbers**: `main_simulation.py`.
2. **Realistic actuator study**: `main_realistic_piezo.py`.
3. **Figures**: scripts in `04_figures/` (illustrative — see their header notes).
