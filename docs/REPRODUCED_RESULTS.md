# Reproduced Results — Verification Log

**Date:** 2026-07-14
**Environment:** Python 3.11, NumPy/SciPy/Matplotlib (latest), Linux x86-64
**Command:** `python main_simulation.py` (files flattened into one directory per package README)
**Runs:** 2 independent runs, ~77 s each.

## 1. Time-domain comparison (LQG vs DARC v3), T = 0.5 s

Numbers below are from run 1; run 2 agreed to within <1% (NN init is seeded with
`seed=42`, residual variance comes from the iterative-learning pass).

| Scenario | LQG y_RMS (µm) | DARC y_RMS (µm) | Gain |
|---|---:|---:|---:|
| S1 — Nominal (a_p = 0.3 mm) | 0.5319 | 0.5073 | **+4.63 %** |
| S2 — Aggressive (a_p = 0.6 mm) | 1.0577 | 1.0112 | **+4.39 %** |
| S3 — Uncertainty (ω −15 %) | 0.6059 | 0.4886 | **+19.37 %** |
| S4 — High K_T (+30 %) | 0.6924 | 0.6605 | **+4.60 %** |
| **Average** | 0.7220 | 0.6669 | **+7.63 %** |

Control effort (S1): LQG u_max = 12.81 V, u_RMS = 3.63 V; DARC u_max = 12.60 V,
u_RMS = 3.70 V (DARC uses *slightly more* RMS voltage in the nominal case).

### ⚠️ Discrepancy vs the historical package README

The package README claimed +19.2…19.5 % for **all four** scenarios (average
+19.31 %). The current code reproduces that magnitude **only for S3**
(model-frequency mismatch). The nominal-scenario gain is ≈ +4.6 %. The README
figures evidently came from an earlier code version and are not reproducible;
all reported numbers must be regenerated from the committed code before any
manuscript submission.

The honest headline is arguably *more* interesting: the phase-aware
feedforward helps most exactly when the feedback model is wrong (S3), which is
the robustness story.

## 2. Stability lobe diagram (FDM/Floquet) at 4900 RPM

| Configuration | a_p critical | vs Open-Loop |
|---|---:|---:|
| Open-Loop | 0.100 mm | 1× |
| LQG (grid-searched weights) | 2.538 mm | 25.4× |
| "DARC v3" | 3.188 mm | 31.9× |

Two observations:

1. **Open-loop anchor is validated.** The reproduced open-loop critical depth
   (0.10 mm at 4900 RPM) matches the *experimental* stability limit reported by
   Du et al. (2024), Fig. 18 (< 0.1 mm at most speeds, milling condition T1 at
   0.3 mm chatters without control). This is a genuine validation point for the
   FEM + force model + FDM pipeline.
2. **The DARC SLD row is NOT a computed result.** In `main_simulation.py` the
   DARC lobe is produced by scaling the LQG closed-loop modal damping by a
   hard-coded factor 1.30 (`zeta_DARC_eff = zeta_LQG_sld * 1.30`) — an assumed
   "equivalent damping", not a Floquet analysis of the actual DARC loop. Since
   the trained network is (by construction of its training data) essentially a
   function of tool phase only, it acts as a *periodic feedforward*, and pure
   feedforward does not move closed-loop poles. The honest computation is to
   linearize the NN around the periodic orbit (periodic gain ∂u_FF/∂x̂) and run
   the FDM with that periodic feedback term. Until that is done, no SLD claim
   for DARC should appear in a manuscript.

## 3. Reproducibility notes

- NN initialization and ILC sampling are seeded (`seed=42`); results are
  repeatable across runs on the same machine to within ~1 %.
- Total runtime ≈ 77 s for the 4-scenario comparison + 3 SLDs (grid 60×50,
  m_div = 30) on a standard container. No GPU required.
