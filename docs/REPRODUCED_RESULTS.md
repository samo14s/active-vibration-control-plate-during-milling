# Reproduced Results — Verification Log

**Date:** 2026-07-14 (updated after the P2.5 improvement pass)
**Environment:** Python 3.11, NumPy/SciPy/Matplotlib (latest), Linux x86-64
**Command:** `python main_simulation.py` (files flattened into one directory per README)

**Protocol now in force (P0 + P1 + P2 + P2.5):**
- Train-once / freeze / evaluate-held-out; **symmetric** baseline (standalone LQG and
  PALF's internal LQR share identical grid-searched weights); identical ±150 V clipping.
- **Corrected cutting constants** k1 = kn/cos η = 0.3174, k2 = 1.1258 (Eq. 3), from
  `milling_force.cutting_constants` (P1).
- **No inverse crime:** 5-mode PLANT, controllers designed on the first 3 (spillover);
  10 nm measurement noise (P1).
- **Eq. (15) piezo coupling** C_P0 (P2).
- **Rigorous closed-loop monodromy SLD** with the LQG controller in the loop, at the
  **worst of 3 tool positions** (x = 0, L/4, L/2 — the article's Fig. 6 treatment;
  P2 + P2.5).
- **Frequency-domain model-inverse ILC** (P2.5): the feedforward harmonics are updated
  per trial as U_h ← U_h − η·Y_h/G(jhω_τ) from the DFT of the periodic steady-state
  residual, with G the design closed-loop FRF; the best trial is frozen and the phase
  network is fit to the resulting periodic target.

---

## 1. Time-domain comparison (LQG vs PALF-LQG), T = 0.5 s

| Scenario | LQG y_RMS (µm) | PALF y_RMS (µm) | Gain |
|---|---:|---:|---:|
| S1 — Nominal (a_p = 0.3 mm) | 0.7765 | 0.6252 | **+19.48 %** |
| S2 — Aggressive (a_p = 0.6 mm) | 1.5580 | 1.3864 | **+11.01 %** |
| S3 — Model mismatch (ω −8 %) | 0.9001 | 0.7689 | **+14.58 %** |
| S4 — High K_T (+30 %) | 1.0127 | 0.8533 | **+15.75 %** |
| **Average** | 1.0618 | 0.9084 | **+14.44 %** |

Control effort (S1): LQG u_max = 23.04 V, u_RMS = 5.55 V; PALF u_max = 22.29 V,
u_RMS = 6.02 V — ~8 % more RMS voltage buys ~19.5 % less vibration; both far below
the ±150 V limit. Bit-reproducible across runs (all RNGs seeded).

### The result, honestly

With the principled model-inverse ILC, the phase-locked feedforward now cancels most
of the tooth-passing-periodic residual that survives LQG feedback: **+19.5 % on the
nominal plant**, and the FROZEN feedforward retains **double-digit gains on every
held-out perturbation** (+11 % at doubled depth, +14.6 % under −8 % frequency
mismatch, +15.8 % at +30 % cutting stiffness). The gains degrade gracefully rather
than collapsing — the phase-indexed compensation does not depend on the (wrong)
feedback model.

### Monte-Carlo robustness (50 samples; held-out; divergence reported)

`python main_robustness_mc.py` (±15 % cutting force, ±3 % modal frequency, ±20 %
damping):

- Converged: **LQG 50/50, PALF 50/50** (no survivorship bias — divergence would be
  counted, none occurred within these ranges).
- RMS gain over both-converged samples: **median +17.81 %, mean +17.66 %**
  [p05 +15.78 %, p95 +19.53 %].
- **PALF beats LQG in 100 %** of samples. (`figs_lqg_vs_palf/fig_robustness_mc.*`)

### A genuine robustness limit (surfaced by the corrected forces)

The **controlled** stability margin at a_p = 0.3 mm / 4900 RPM is ~**−9 % frequency
mismatch**: stable at −8 %, divergent at −10 %+, for both controllers, independent of
observer tuning. S3 therefore uses −8 % (≈ the article's measured 9 % 2nd-mode drift).

## 2. Stability lobe diagram — rigorous closed-loop monodromy, worst of 3 tool positions, at 4900 RPM

| Configuration | a_p critical | vs Open-Loop |
|---|---:|---:|
| Open-Loop (coupled monodromy, worst position) | 0.100 mm | 1× |
| LQG (closed-loop monodromy, worst position) | 1.075 mm | 10.8× |
| PALF-LQG | **1.075 mm (= LQG, rigorously)** | 10.8× |

Notes:
1. **Open-loop anchor still validated:** 0.10 mm at 4900 RPM = Du et al. (2024)
   Fig. 18 experimental limit.
2. **Worst-position analysis** (x = 0, L/4, L/2; elementwise-max ρ) replaces the
   path-averaged Dp — more conservative (1.08 mm vs 1.73 mm averaged) and aligned with
   the article's own position-resolved treatment. Notably, the controlled critical
   depth is now the **same order as the article's experimentally achieved 0.6–0.8 mm**
   controlled limits.
3. **PALF = LQG rigorously:** the feedforward is a phase-only map, ∂u_FF/∂x̂ ≡ 0, so it
   does not enter the closed-loop Jacobian — identical monodromy, multipliers, boundary.
4. Monodromy verification: no-cutting ρ = 0.56 < 1; open-loop coupled = per-mode
   open-loop = 0.079 mm on a fine grid.

## 3. FEM verification (mesh convergence)

`python 03_analysis/mesh_convergence.py`: frequencies converged to <0.1 % by 30×24
(521.1/1069.9/2732.8/3334/4145 Hz); uniform ~2.6 % below the article's Chebyshev-Ritz
*theory* (discretisation-model difference — the non-conforming element converges from
below) but within **0.2–0.6 % of the MEASURED** modes 2, 4, 5 (Table 4).

## 4. Reproducibility notes

- One eigensolve builds the nominal 5-mode plant; truncation feeds the controllers,
  perturbed copies feed the scenarios (consistent mode signs).
- All RNGs seeded (ILC/NN `42`; measurement noise `1234`, identical for both
  controllers). Bit-reproducible across runs.
- Runtime ≈ 62 s (`main_simulation.py`, incl. the 6 worst-position monodromy grids);
  ≈ 1 min Monte-Carlo. No GPU.

## 5. Remaining items (P3)

- Experimental validation on a physical plate (the article's rig is fully specified —
  Tables 1–3 + Fig. 17 list every instrument). Everything above is simulation.
