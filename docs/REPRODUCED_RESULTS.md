# Reproduced Results — Verification Log

**Date:** 2026-07-14 (updated after the P0 integrity fixes)
**Environment:** Python 3.11, NumPy/SciPy/Matplotlib (latest), Linux x86-64
**Command:** `python main_simulation.py` (files flattened into one directory per README)
**Protocol:** train-once / freeze / evaluate-held-out, with a **symmetric** feedback
baseline (the standalone LQG and PALF's internal LQR share identical grid-searched
weights). Two independent runs; agreement <1 % (RNGs seeded with 42).

---

## 1. Time-domain comparison (LQG vs PALF-LQG), T = 0.5 s

PALF-LQG = LQG feedback + phase-locked learned feedforward. The feedforward is trained
**once** on the nominal scenario (S1) and then **frozen**; S2/S3/S4 are held-out.

| Scenario | LQG y_RMS (µm) | PALF y_RMS (µm) | Gain |
|---|---:|---:|---:|
| S1 — Nominal (a_p = 0.3 mm) | 0.5319 | 0.5073 | **+4.62 %** |
| S2 — Aggressive (a_p = 0.6 mm) | 1.0577 | 1.0207 | **+3.49 %** |
| S3 — Model mismatch (ω −15 %) | 0.6059 | 0.4866 | **+19.69 %** |
| S4 — High K_T (+30 %) | 0.6924 | 0.6643 | **+4.06 %** |
| **Average** | 0.7220 | 0.6697 | **+7.23 %** |

Control effort (S1): LQG u_max = 12.81 V, u_RMS = 3.63 V; PALF u_max = 12.61 V,
u_RMS = 3.70 V (PALF uses marginally more RMS voltage in the nominal case).

### The result, honestly

The learned feedforward buys **little on the nominal plant** (+4.6 %) but **holds its
gain under model mismatch** (+19.7 % when the feedback design frequency is off by
−15 %). Because the feedforward is indexed to the tooth-passing phase rather than to
the model, it keeps cancelling the periodic disturbance even when the feedback model is
wrong. "Learned periodic feedforward buys robustness to model mismatch, not nominal
performance" is the defensible, reviewer-proof claim — and it is exactly what the held-
out data shows (S3 is the only large gain, and it survives the train-once protocol).

### Change vs the historical (pre-fix) numbers

The old package README claimed a uniform +19.2…19.5 % across **all four** scenarios
(average +19.31 %). That table came from an earlier configuration with a deliberately
weakened baseline and per-scenario retraining, and it did not reproduce. Under the
corrected symmetric / held-out protocol the honest picture is S1 +4.6 %, S2 +3.5 %,
S3 +19.7 %, S4 +4.1 %, average +7.2 %.

## 2. Stability lobe diagram (Floquet) at 4900 RPM

| Configuration | a_p critical | vs Open-Loop |
|---|---:|---:|
| Open-Loop | 0.100 mm | 1× |
| LQG (closed loop) | 2.538 mm | 25.4× |
| PALF-LQG | **2.538 mm (= LQG)** | 25.4× |

Two points:

1. **Open-loop anchor is validated.** The reproduced open-loop critical depth
   (0.10 mm at 4900 RPM) matches the *experimental* stability limit reported by
   Du et al. (2024), Fig. 18 (< 0.1 mm at most speeds; condition T1 at 0.3 mm chatters
   without control). This validates the FEM + force model + Floquet pipeline.
2. **The PALF SLD is no longer fabricated.** The previous code multiplied the LQG
   closed-loop damping by a hard-coded 1.30 and presented the result as the "DARC"
   lobe (the source of the retracted "+41 % stable depth" and "31.1 % effective
   damping" claims). That multiplier is removed. A phase-locked feedforward changes the
   periodic forcing, not the closed-loop poles, so PALF **shares the LQG boundary**.
   A genuine closed-loop SLD for the feedforward would require linearizing the trained
   network around the periodic orbit (periodic gain ∂u_FF/∂x̂) and running the
   discretization with that periodic feedback term — flagged as future work (P2).

## 3. Reproducibility notes

- Single shared feedback design + single feedforward pretraining (30 ILC iterations);
  all RNGs seeded (`seed=42`). Repeatable to within ~1 % across runs.
- Runtime ≈ 38 s for the 4-scenario comparison + SLDs (grid 60×50, m_div = 30) on a
  standard container — faster than before because the controller is trained once, not
  four times. No GPU required.

## 4. Known residuals still to address before submission (not P0)

- **k1, k2 cutting constants** still follow the current package convention and deviate
  from Du et al. Eq. (3) (k1 ≈ −33 %, k2 ≈ −12 %). Correcting them (P1) will shift all
  absolute force levels and every SLD depth; the *relative* LQG-vs-PALF comparison is
  unaffected.
- **Inverse crime**: the simulation plant still equals the controller design model
  (3 modes, near-noiseless Kalman). Add unmodeled modes + sensor noise (P1).
- The 04_figures/ scripts still retrain the feedforward per figure; they are
  illustrative. Authoritative numbers are those in this file.
