# Reproduced Results — Verification Log

**Date:** 2026-07-14 (updated after the P2 work)
**Environment:** Python 3.11, NumPy/SciPy/Matplotlib (latest), Linux x86-64
**Command:** `python main_simulation.py` (files flattened into one directory per README)

**Protocol now in force (P0 + P1 + P2):**
- Train-once / freeze / evaluate-held-out; **symmetric** baseline (standalone LQG and
  PALF's internal LQR share identical grid-searched weights).
- **Corrected cutting constants** k1 = kn/cos η = 0.3174, k2 = 1.1258 (Du et al.
  Eq. 3), from the single source `milling_force.cutting_constants` (P1).
- **No inverse crime:** 5-mode PLANT, controllers designed on the first 3 (spillover);
  10 nm measurement noise; identical ±150 V clipping (P1).
- **Eq. (15) piezo coupling** C_P0 (P2): the actual bending-coupling coefficient
  (~16 % weaker than the earlier simplified constant).
- **Rigorous closed-loop SLD** (P2): all three SLD panels are computed from the coupled
  monodromy matrix with the LQG controller in the loop — no "equivalent damping"
  surrogate, no per-mode decoupling.

---

## 1. Time-domain comparison (LQG vs PALF-LQG), T = 0.5 s

| Scenario | LQG y_RMS (µm) | PALF y_RMS (µm) | Gain |
|---|---:|---:|---:|
| S1 — Nominal (a_p = 0.3 mm) | 0.7765 | 0.7393 | **+4.79 %** |
| S2 — Aggressive (a_p = 0.6 mm) | 1.5580 | 1.5014 | **+3.63 %** |
| S3 — Model mismatch (ω −8 %) | 0.9001 | 0.8115 | **+9.84 %** |
| S4 — High K_T (+30 %) | 1.0127 | 0.9675 | **+4.47 %** |
| **Average** | 1.0618 | 1.0049 | **+5.36 %** |

Control effort (S1): LQG u_max = 23.04 V, u_RMS = 5.55 V; PALF u_max = 21.17 V,
u_RMS = 5.69 V. (Voltages rose ~15 % vs the P1 run because the Eq. 15 coupling is
weaker, so more volts are needed for the same modal force.)

### The result, honestly

The learned feedforward helps most under **model mismatch** (S3, +9.8 %) and modestly
on the nominal plant (+4.8 %) — the same robustness asymmetry throughout, now with
spillover, measurement noise, a symmetric baseline, the corrected Eq. (3) forces, and
the Eq. (15) piezo coupling. Because the feedforward is indexed to the tooth-passing
phase, it keeps cancelling the periodic disturbance when the feedback model is wrong.

### Monte-Carlo robustness (50 samples; held-out; divergence reported)

`python main_robustness_mc.py` (±15 % cutting force, ±3 % modal frequency, ±20 %
damping):

- Converged: **LQG 50/50, PALF 50/50** (no survivorship bias — divergence would be
  counted, none occurred within these ranges).
- RMS gain over both-converged samples: **median +5.05 %, mean +4.99 %**
  [p05 +3.22 %, p95 +6.88 %].
- **PALF beats LQG in 100 %** of samples. (Figure: `figs_lqg_vs_palf/fig_robustness_mc.png`.)

### A genuine robustness limit (surfaced by the corrected forces)

The **controlled** stability margin at a_p = 0.3 mm / 4900 RPM is ~**−9 % frequency
mismatch**: a sweep found stability at −8 % but divergence at −10 %+, for both
controllers, independent of observer tuning. S3 therefore uses −8 % (within margin, and
matching the article's ~9 % measured 2nd-mode drift), not the earlier −15 % (past the
boundary at this depth).

## 2. Stability lobe diagram — rigorous closed-loop monodromy, at 4900 RPM

| Configuration | a_p critical | vs Open-Loop |
|---|---:|---:|
| Open-Loop (coupled monodromy) | 0.100 mm | 1× |
| LQG (closed-loop monodromy) | 1.725 mm | 17.2× |
| PALF-LQG | **1.725 mm (= LQG, rigorously)** | 17.2× |

Notes:
1. **Open-loop anchor validated.** 0.10 mm at 4900 RPM matches Du et al. (2024),
   Fig. 18 (< 0.1 mm at most speeds; T1 at 0.3 mm chatters without control).
2. **No surrogate, no decoupling.** The LQG boundary is the spectral radius of the
   monodromy matrix of the full coupled, time-periodic delayed loop with the LQR gain
   AND the Kalman observer embedded. The critical depth (1.7 mm) is a bit lower than
   the old equivalent-damping surrogate (1.9 mm) because the rigorous method accounts
   for the observer dynamics and the true delayed feedback.
3. **PALF = LQG is now rigorous, not asserted.** The feedforward is a phase-only map
   u_FF(φ), so ∂u_FF/∂x̂ ≡ 0: it does not enter the closed-loop Jacobian, hence the
   monodromy matrix, all Floquet multipliers, and the boundary are identical to LQG.
4. Verification of the monodromy code: no-cutting → ρ = 0.56 < 1 (stable); open-loop
   coupled = per-mode open-loop = 0.079 mm on a fine grid.

## 3. FEM verification (mesh convergence)

`python 03_analysis/mesh_convergence.py`: the FEM frequencies converge to <0.1 % by a
30×24 mesh (521.1 / 1069.9 / 2732.8 / 3334 / 4145 Hz). They sit a uniform ~2.6 % below
the article's Chebyshev-Ritz *theory* (a discretisation-model difference — the
non-conforming Kirchhoff element converges from below), but are within **0.2–0.6 % of
the MEASURED** modes 2, 4, 5 (Table 4). This reconciles the earlier unexplained gap.

## 4. Reproducibility notes

- One eigensolve builds the nominal 5-mode plant; its truncation feeds the controllers
  and its perturbed copies feed each scenario (consistent mode signs).
- All RNGs seeded (NN/ILC `seed=42`; measurement noise `rng=1234`, identical for both
  controllers). Repeatable to <1 %.
- Runtime ≈ 35 s (`main_simulation.py`, incl. the closed-loop SLD in ~8 s);
  ≈ 1 min for the 50-sample Monte-Carlo. No GPU.

## 5. Remaining items (P3)

- Experimental validation on a physical plate (the article's rig is fully specified —
  Tables 1–3 + Fig. 17 list every instrument). This is the transformative next step;
  everything above is simulation.
