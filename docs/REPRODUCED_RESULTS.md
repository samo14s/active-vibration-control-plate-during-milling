# Reproduced Results — Verification Log

**Date:** 2026-07-14 (updated after the P1 fixes)
**Environment:** Python 3.11, NumPy/SciPy/Matplotlib (latest), Linux x86-64
**Command:** `python main_simulation.py` (files flattened into one directory per README)

**Protocol now in force (P0 + P1):**
- Train-once / freeze / evaluate-held-out; **symmetric** baseline (standalone LQG and
  PALF's internal LQR share identical grid-searched weights).
- **Corrected cutting constants** k1 = kn/cos η = 0.3174, k2 = 1 + µc·tan η·(cos γn −
  kn·sin γn) = 1.1258 (Du et al. Eq. 3), from the single source
  `milling_force.cutting_constants`.
- **No inverse crime:** the simulated PLANT carries **5 modes** (521/1070/2733/3334/
  4146 Hz), the controllers are designed on the **first 3** — the extra modes are
  unmodelled (control + observation spillover).
- **Measurement noise:** 10 nm RMS injected on the displacement; the Kalman keeps a
  conservative (robust) noise assumption.
- **Identical actuator clipping:** both controllers clip to ±150 V.

---

## 1. Time-domain comparison (LQG vs PALF-LQG), T = 0.5 s

| Scenario | LQG y_RMS (µm) | PALF y_RMS (µm) | Gain |
|---|---:|---:|---:|
| S1 — Nominal (a_p = 0.3 mm) | 0.7437 | 0.7060 | **+5.07 %** |
| S2 — Aggressive (a_p = 0.6 mm) | 1.4914 | 1.4331 | **+3.91 %** |
| S3 — Model mismatch (ω −8 %) | 0.8530 | 0.7443 | **+12.74 %** |
| S4 — High K_T (+30 %) | 0.9698 | 0.9265 | **+4.46 %** |
| **Average** | 1.0145 | 0.9525 | **+6.11 %** |

Control effort (S1): LQG u_max = 20.26 V, u_RMS = 5.03 V; PALF u_max = 18.46 V,
u_RMS = 5.20 V. (Voltages are higher than in the earlier, under-forced runs because
the corrected constants give ~15–20 % stronger cutting forces.)

### The result, honestly

The learned feedforward still helps most under **model mismatch** (S3, +12.7 %) and
modestly on the nominal plant (+5.1 %) — the same robustness asymmetry as before, now
demonstrated with spillover, measurement noise, a symmetric baseline, and the corrected
force model. Because the feedforward is indexed to the tooth-passing phase, it keeps
cancelling the periodic disturbance when the feedback model is wrong.

### A genuine robustness limit surfaced by the honest force model

With the corrected (stronger) forces, the **controlled** stability margin at
a_p = 0.3 mm / 4900 RPM is about **−9 % frequency mismatch**: a diagnostic sweep found
the closed loop stable at −8 % but divergent at −10 % and beyond, for **both**
controllers and independent of the observer tuning. This is why S3 uses −8 % (within
margin, and matching the article's ~9 % measured 2nd-mode drift after machining) rather
than the earlier arbitrary −15 %, which is simply past the stability boundary at this
depth. Material-removal drift beyond ~9 % would require re-tuning or gain scheduling —
exactly the motivation for the article's µ-synthesis robust design.

## 2. Stability lobe diagram (Floquet) at 4900 RPM

| Configuration | a_p critical | vs Open-Loop |
|---|---:|---:|
| Open-Loop | 0.100 mm | 1× |
| LQG (closed loop) | 2.050 mm | 20.5× |
| PALF-LQG | **2.050 mm (= LQG)** | 20.5× |

Notes:
1. **Open-loop anchor still validated.** 0.10 mm at 4900 RPM matches Du et al. (2024),
   Fig. 18 (< 0.1 mm at most speeds; T1 at 0.3 mm chatters without control).
2. **Corrected forces lowered the controlled critical depth** from 2.54 mm (old, wrong
   k1/k2) to 2.05 mm — a direct, expected consequence of the ~15–20 % stronger forces.
3. **PALF shares the LQG boundary** (no fabricated lobe): a phase-locked feedforward
   does not move the closed-loop poles. A rigorous periodic-gain Floquet closed-loop
   SLD remains future work (P2).

## 3. Reproducibility notes

- One eigensolve builds the nominal 5-mode plant; its truncation feeds the controllers
  and its perturbed copies feed each scenario, so all mode signs are consistent.
- All RNGs seeded (NN init and ILC `seed=42`; measurement noise `rng=1234`, identical
  for both controllers so they face the same noise). Repeatable to <1 %.
- Runtime ≈ 41 s for the 4-scenario comparison + SLDs on a standard container. No GPU.

## 4. Remaining items (P2, not yet done)

- Periodic-gain closed-loop SLD for the feedforward (linearize the NN around the
  periodic orbit); coupled multi-mode / position-resolved SLD.
- Eq. (15) piezo coupling coefficient (currently a simplified induced-moment constant).
- Wire the Monte Carlo module into the pipeline with the train-once/held-out protocol.
- Mesh-convergence note reconciling the 2.6–3.5 % FEM-vs-Table 4 frequency gap.
