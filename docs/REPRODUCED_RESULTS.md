# Reproduced Results — Verification Log

## P9 UPDATE (2026-07-17): plant MODELING refinement — piezo-patch structure in the FEM

Returning to the P5 modeling: the bonded PZT patch entered the model ONLY as an
actuation force (`add_piezo_patch` → `H_Pe_modal`); the modal analysis ran on the
**bare aluminium plate**. But a 20×60×0.7 mm surface-bonded PZT layer is not
massless/stiffnessless. `PlateModel.add_piezo_structure` now smears it into the patch
elements as an equivalent Kirchhoff element with the composite bending rigidity about
the laminate neutral axis (**×1.583**, +58 %) and the added areal mass (**×1.464**,
+46 %; neutral-axis shift 0.324 mm), then re-solves the modes (108 patch elements).

Natural frequencies (Hz) vs Du et al. (2024) Table 4 **measured**:

| mode | bare | +structure | shift | measured | bare err | instr err |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 521.1 | 541.1 | +3.8 % | 540 | −3.5 % | **+0.2 %** |
| 2 | 1069.9 | 1093.9 | +2.2 % | 1068 | +0.2 % | +2.4 % |
| 3 | 2733.0 | 2745.0 | +0.4 % | 2787 | −1.9 % | −1.5 % |
| 4 | 3334.4 | 3407.3 | +2.2 % | 3351 | −0.5 % | +1.7 % |
| 5 | 4145.6 | 4193.1 | +1.1 % | 4122 | +0.6 % | +1.7 % |

Near the clamped root (peak mode-1 curvature) the stiffening dominates → modes rise
0.4–3.8 %. The **mode-1** match improves markedly (−3.5 % → +0.2 %, 541 vs 540 Hz);
modes 2/4/5 shift up ~2 % (mode 2 the wrong way), so the **net RMS frequency error vs
measured improves modestly, 1.83 % → 1.67 %** — the gain is mode-1-driven, reported in
full, not cherry-picked. A-ESO-ADRC re-designed+evaluated consistently per plant holds
**0.381 (bare) vs 0.346 µm (instrumented)** at a_p = 0.3 mm.

Honest caveats (all in the driver docstring): (1) the pure-bending Kirchhoff element
neglects the bending–membrane coupling of the offset neutral axis (laminate B-matrix) —
standard for thin bonded patches; (2) PZT density `rho_Pe` is ASSUMED 7500 kg/m³
(PZT-5H) — the article gives d31/E_Pe but not density; sensitivity is negligible (mode 1
= 541.1 Hz at 7500 vs 540.8 at 7800); (3) whether the article's measured Table-4 values
were taken bare or instrumented is unstated — the improved mode-1 match is consistent
with, but not proof of, the patch having been bonded during the modal test. This is an
**opt-in higher-fidelity plant**; the committed P1–P8 numbers keep the bare-plate model
so they stay comparable. Driver: `05_main/main_piezo_structure.py` (~6 s). Figure:
`figures/16_piezo_structure.pdf`.

---

## P8 UPDATE (2026-07-15): improving the P5 HRC — certified robust resonator + adaptive negative result

The P5 fixed HRC uses NARROW resonators (lam = 5 rad/s) for the deepest nominal notch
(0.381 µm, ~2× LQG). Narrow notches are FRAGILE to plant drift. Two routes to improve
robustness were investigated; driver `05_main/main_hrc_robustness.py` (~105 s).

**Route 1 — adaptive (FxLMS/AFC) harmonic cancellation. REFUTED.** Adapt a complex
weight per tooth line online to re-null the drifted residual
(`ESO_ADRC_AdaptiveHRC_Controller`). Tuned on nominal only (mu0 = 0.01, leak = 1e-4) it
matches the fixed HRC nominally (0.389 vs 0.381 µm) but is WORSE under every drift:

| case | LQG | HRC perf (lam5) | **adaptive-HRC** | A-ESO-ADRC |
|---|---:|---:|---:|---:|
| S1 nominal | 0.777 | 0.381 | 0.389 | 0.381 |
| D1 ramp +15% | 0.682 | 0.955 | **3.407** | 1.424 |
| S3 static −8% | 0.900 | 251 | **302** | 12.03 |
| D2 ramp −12% | DIV | 33.6 | **DIV** | 1.209 |
| S4 K_T +30% | 1.013 | 1.135 | **DIV** | 1.135 |

Mechanism (physical, honest): this plant's modes sit CLOSE to the tooth lines (mode 1 =
521 Hz between lines h2 = 490 / h3 = 735; mode 2 = 1070 Hz near h4 = 980), so even modest
drift swings the secondary-path phase past the FxLMS ±90° convergence cone at a line
adjacent to a mode, and the low-margin integral action then INJECTS rather than cancels.
The optional `w_cap` self-protection does not rescue it (the nominal converged per-line
|W| is already O(100 V) because the lines partly cancel, so any cap tight enough to catch
a runaway also sheds healthy lines). Kept as a documented negative-result artifact.

**Route 2 — wider LTI resonator (larger lam). THE IMPROVEMENT.** Widening trades notch
depth for phase margin at each line. The design-ball worst-case closed-loop Floquet
radius (the SAME certification used for the ESO rung) decreases MONOTONICALLY with lam,
so a wider resonator is provably more robust by the project's own metric:

| lam (rad/s) | nominal RMS (µm) | worst-ρ (design ball) |
|---:|---:|---:|
| 5 (performance) | 0.381 | 1.3927 |
| 10 | 0.388 | 1.3884 |
| **20 (robust)** | **0.407** | **1.3800** |
| 40 | 0.451 | 1.3637 |
| 60 | 0.491 | 1.3481 |

Selection (honest — a smooth dial, NOT a threshold). The certification fixes the
DIRECTION (wider = more robust); the payoff on the in-ball −12 % frequency ramp then
improves SMOOTHLY and monotonically with lam, with no sharp "controls it" cutoff:

| lam | nominal (µm) | D2 −12 % ramp (µm) | S4 +30 % K_T (µm) |
|---:|---:|---:|---:|
| 8 | 0.385 | 6.40 | 1.094 |
| 10 | 0.388 | 3.82 | 1.069 |
| 12 | 0.391 | 2.28 | 1.045 |
| 15 | 0.397 | 1.14 | 1.013 |
| **20** | **0.407** | **0.591** | **0.968** |

So lam is a tunable robustness/performance dial. **lam = 20** is chosen as a balance
point: it brings the −12 % ramp (perf-HRC 33.6 µm, near-divergence) down to 0.591 µm —
below the supervised ladder's 1.209 µm and the LQG-envelope ~0.7 µm — at a modest +7 %
nominal cost (0.407 µm), and +30 % K_T improves too (1.135 → 0.968). A different operating
point (e.g. lam = 15: 0.397 µm nominal, 1.14 µm ramp) is equally defensible; the value is
the documented dial, not one magic width.
Held-out checks: S4 (+30 % K_T, outside the freq ball) improves; the two static-drift
failures (S3 −8 %, D3 −12 %) remain — these are the resonance-coincidence case (mode 2
drifts ONTO the h4 line) with no settling time, exactly what the supervised ladder covers.
The robust HRC stays fully LTI → the monodromy SLD / certification still apply.

**Benign-regime Monte-Carlo** (±3 % freq, ±15 % K_T, ±20 % damping, 25 samples, all
25/25 converged): LQG 0.747, perf-HRC 0.361, robust-HRC 0.388, A-ESO-ADRC 0.361 µm — the
robust width's small benign cost (+7 %), the price for the large-drift robustness.

**Deployment (honest):** inside the A-ESO-ADRC SUPERVISED ladder the wider rung is a WASH
(the supervisor already gets drift robustness by switching to the ESO rungs: a swapped
lam=20 rung helps −8 % 12.0→9.3 and +30 % 1.14→0.97 but costs nominal 0.381→0.407 and D2
1.21→2.03; an inserted 5th wide-HRC rung helps +15 % 1.42→1.21 but hurts −8 % and D2). So
the deployed 4-rung ladder is UNCHANGED; the robust HRC is the recommended STANDALONE HRC
(performance/certified duality, mirroring the ESO designs). Figure:
`figures/15_hrc_robustness.pdf`.

---

## P7 UPDATE (2026-07-15): material-removal-aware predictive control — feasibility study

The user requested a controller that accounts for the material removed DURING the cut
(precise cutting model -> plate properties per step -> predicted vibration ->
suppression). A rigorous investigation (4 parallel numerical studies + adversarial
review) established that a genuine PER-STEP material-removal-aware controller is not
physically realizable/beneficial on this plant, for two independent reasons, and the
honest deliverable is a feasibility study plus the concrete artifacts.

**WALL 1 — timescale.** At the article feed the tool advances 0.245 µm per 50-µs step
and takes **13,605 steps (167 tooth-periods, 0.68 s) to cross one FEM mesh column**, so
(M, K, C) is piecewise-constant at mesh resolution and cannot change between steps. A
per-step property update captures ~1e-5 %/step — over-engineering by ~1e4. Correct
cadence = event-driven (mesh-crossing).

**WALL 2 — regime conflict.** Open-loop chatter radius ρ(a_p) at 4900 RPM vs within-pass
modal drift (x-resolved moving front, idempotent single-bite per crossing):

| a_p (mm) | ρ open-loop | within-pass drift | regime |
|---:|---:|---:|---|
| 0.3 | 1.19 | ~0 %* | controllable |
| 1.0 | 1.64 | ~0 %* | controllable |
| 2.0 | 3.5 | 0.20 % | controllable (boundary) |
| 5.0 | 178 | 0.37 % | uncontrollable |
| 10 | 1.1e5 | 0.53 % | uncontrollable |
| 20 | 6.0e9 | 0.87 % | uncontrollable |
| 40 | 7.9e16 | 1.37 % | uncontrollable |

Within-pass drift first exceeds 1 % only at **a_p = 40 mm** (ρ ~ 8e16, utterly
uncontrollable); even at the controllable boundary (a_p = 2 mm, ρ ~ 3.5) it is just
0.20 %. \*The ~0 % for a_p ≤ 1 mm is a **mesh-resolution floor**: on the 24-row height
mesh (row ≈ 3.33 mm) the thinning band `ez ≥ hp − a_p` captures no element centroid
until a_p ≈ 1.67 mm, so sub-mm depths quantise to exactly zero removed elements; the
true drift there is bounded above by the 0.20 % resolved at 2 mm and is physically
negligible. Removal is significant only where ρ ≫ 1 (uncontrollable, beyond ±150 V
piezo authority); the controllable regime has ≲0.2 % removal. The two requirements are
mutually exclusive on this plant. (These magnitudes correct an earlier over-thinning
bug — repeated `remove_moving_front` calls re-subtracted a_e each crossing, inflating
drift ~10×; the fix baselines each pass with `begin_pass()` so a_e bites once.)

**Controller (a_p = 0.3 mm, T = 0.4 s).** The PreviewPredictiveController exploits that
the regenerative force uses the already-known q(k−n_tau) and the feed force is periodic
(previewable one tooth period ahead); a receding-horizon QP (H = 41 steps, r = 1e-14)
pre-empts the disturbance. Anti-inverse-crime: controller cutting model mismatched
(+15 % KT, +15 % kn, −15 % µc):

| controller | y_RMS (µm) |
|---|---:|
| LQG | 0.779 |
| ESO-ADRC | 0.829 |
| Preview (exact-model reference) | 0.720 |
| **Preview (mismatched — honest)** | **0.682** |
| ESO-ADRC+HRC | **0.390 (best)** |

The preview law is stable and modestly beats LQG (~10 % at a tuned horizon; the
mismatched model happening to edge the exact one is a horizon-dependent coincidence,
NOT a robustness claim), but is **dominated ~2× by the existing HRC** and, being
model-based, is inherently limited by cutting-model accuracy — weaker than P6's probe,
which measures the truth. Documented caveats: the a_p = 0.3 mm band is below the N2=24
mesh row height (moving-front is mesh-valid only for deep cuts); one-face removal
offsets the neutral surface by ~a_e/2 (symmetric h³/h scaling ignores it — negligible
at a_e = 0.1 mm). Artifacts: `01_core/material_removal.remove_moving_front`,
`02_controllers/predictive_removal.py`, `05_main/main_predictive_removal.py`
(~30 s runtime). Figure: `figs_predictive_removal/`.

---

## P6 UPDATE (2026-07-15): real-time identification over an accurate finishing sequence

New physical model `01_core/material_removal.MillingWorkpiece` — per-element
thickness-field FEM (K_e = h³·K_unit, M_e = h·M_unit; MAC mode tracking, 0 sign
flips / 24 snapshots). Honest physics (`phys_check`): one a_p = 0.3 mm pass removes
0.0094 % of the volume → ~0.000 %/mode drift, MAC = 1.0 (plant constant within a
pass); uniform full-face thinning gives ω ∝ h exactly (−2.5 %/layer); only
NON-UNIFORM removal reshapes the modes (top-band cut: per-mode drift
+0.87/+0.15/−0.57 %, MAC 0.9998). The 6-layer finishing sequence (24 passes, wall
4.0 → 3.4 mm) drifts the modes **−15.0 %** — the article's 9–17 % band.

Active-probe identifier `03_analysis/realtime_id.transit_probe_identify` (chirp
during the non-cutting transit, open-loop FRF, peak-pick): modal-frequency error
**≤ 1.09 % (mean 0.82 %)** across the whole −15 % sweep. Passive baseline
(`passive_frequency_estimate`, cutting spectrum): **biased — returns a tooth
harmonic** (mid-sequence 488 Hz near 2·f_tooth, not the true modes), confirming the
persistent-excitation finding.

Closed loop over the sequence (`main_realtime_id.py`, each pass = a cutting run on
that pass's true snapshot plant; T = 0.4 s):

| controller | passes with RMS > 1 µm (control lost) / 24 | worst-pass RMS | final-pass (−15 %) |
|---|---:|---:|---:|
| LQG fixed (pristine) | **7** | 10.9 µm | 10.9 µm |
| LQG ID-scheduled | **0** | 0.072 µm | 0.042 µm |
| LQG oracle (true freq) | 0 | ~0.07 µm | 0.042 µm |
| ESO-ADRC fixed (robust) | **0** | ~0.1 µm | 0.037 µm |

ID-scheduled LQG is **151× better than fixed at the worst pass** and matches the
true-frequency oracle to 3 decimals (frequency-only ID; shape reshaping is second
order). The robust ESO-ADRC survives WITHOUT ID (0 control-loss passes) and naive
ID-retuning can even hurt it — identification and disturbance-observer robustness
are complementary. New negative-result finding: an HRC resonator on a mode that has
drifted onto a tooth harmonic is hazardous regardless of phase; the `'hrc'`-kind
IDScheduledController drops that line (per-line `harmonics=` support added to
`ESO_ADRC_HRC_Controller`). Runtime ≈ 65 s. Figure: `figs_realtime_id/`.

---

## P5 UPDATE (2026-07-15, evening): HRC layer + 4-rung A-ESO-ADRC

New controller stage: **ESO-ADRC+HRC** (per-tooth-harmonic LTI resonant
compensators, inverse-closed-loop-FRF phase, n_harm = 4 — capped so the lines
stay clear of mode 2's drift band — g_base = 150, lam = 5, grid-selected on the
nominal model) and **A-ESO-ADRC v3** = 4-rung supervised ladder
[HRC / perf-ESO(1e16,1e8,3e3) / quasi-Kalman(1e14,1e8,1e3) / certified(1e14,1e8,1e4)].

Held-out scenarios (y_RMS µm): S1: LQG 0.777, HRC **0.381**, cert-ESO 0.826,
A- **0.381**; S2: 1.558 / DIV / 1.824 / 2.43; S3: 0.900 / 251 (bounded) / 20.8
(bounded) / 12.0 (bounded, ends recovering on the quasi-Kalman rung);
S4: 1.013 / 1.135 / 1.078 / 1.135.

Drift benchmark: D0: 0.777/0.381/0.826/**0.381**; D1(+15 % ramp):
0.682/0.955/1.276/1.424; D2(−12 % ramp): DIV/33.6/0.898/**1.209**;
D3(−12 % static): DIV/DIV/1.140/**1.057**; D4(effect. ×0.25):
1.241/2.096/1.221/**1.200**. **A-ESO-ADRC: zero divergences in all 9 cases.**

Monte-Carlo (50 samples, ±15 % cutting, ±3 % freq, ±20 % damping, all
controllers 50/50 converged): medians LQG 0.788, HRC 0.410, cert-ESO 0.850,
**A-ESO-ADRC 0.410 µm — +48.1 % median vs LQG, better in 94 % of samples**
[p05 −4.0 %, p95 +54.1 %].

Control effort (S1): A-ESO-ADRC u_max 20.3 V (vs LQG 23.0), u_RMS 7.2 V (vs 5.6).
SLD unchanged (certified rung = the A- fallback boundary, 0.913 mm at 4900 RPM).
Design finding #5 (negative): harmonic states INSIDE the ESO destabilise the
5-mode plant via estimator spillover (with or without LQ-optimal disturbance
feedthrough) — the resonant layer must sit in the controller output path. The
supervisor v3 mechanics (rising-energy cascade panic, severity-based target,
recovery-trend holds, per-pass failure flags, probe aborts, desperation probing,
escalating locks) are documented in `02_controllers/adrc_controller.py`.

The sections below describe the P4 state (pre-HRC); their protocol statements
remain in force and their numbers remain valid for the LQG / certified-ESO
columns.


**Date:** 2026-07-15 (P4 — ESO-ADRC controller family)
**Environment:** Python 3.11, NumPy/SciPy/Matplotlib (latest), Linux x86-64
**Commands:** `python main_simulation.py`, `python main_robustness_mc.py`,
`python main_adaptive_removal.py` (files flattened into one directory per README)

**Protocol in force (P0 + P1 + P2 protocol, P4 controllers):**
- Design-once / freeze / evaluate-held-out; adaptation state reset per run.
- **Symmetric design machinery:** LQG and ESO-ADRC use the same output-weighted LQR
  construction; both are grid-tuned on the nominal design model only. The ESO-ADRC
  fixed design is **certification-selected**: smallest worst-case coupled-monodromy
  Floquet radius over a design ball (mismatch −12/−8/0/+8/+15 % at a_p = 0.3 mm,
  plus 0 % at a_p = 0.6 mm; tool at x = 0 and L/2) — no held-out time simulations
  enter any design decision.
- **Corrected cutting constants** k1 = 0.3174, k2 = 1.1258 (Eq. 3), single source
  `milling_force.cutting_constants` (P1).
- **No inverse crime:** 5-mode PLANT, controllers designed on the first 3
  (spillover); 10 nm measurement noise; identical ±150 V clipping; identical noise
  realisations (P1).
- **Eq. (15) piezo coupling** C_P0 (P2).
- **Rigorous closed-loop coupled monodromy SLD** with the controller embedded
  (generic realization — LQG m = 6, ESO-ADRC m = 9), worst of 3 tool positions
  (x = 0, L/4, L/2 — the article's Fig. 6 treatment).

**Designs selected by the grid (printed by `main_simulation.py`):**
- LQG: w_q = 1e14, w_qd = 1e8 (grid-searched).
- ESO-ADRC **performance** design: (w_q, w_qd, σ_d) = (1e16, 1e8, 3e3),
  rms_nom = 0.798 µm, worst-ρ = 1.165.
- ESO-ADRC **certified** design: (1e14, 1e8, 1e4), rms_nom = 0.838 µm,
  worst-ρ = 1.046 → the fixed "ESO-ADRC" entry everywhere.
- A-ESO-ADRC ladder = [performance rung, certified rung].

---

## 1. Time-domain comparison (held-out scenarios, T = 0.5 s)

| Scenario | LQG y_RMS (µm) | ESO-ADRC (µm) | A-ESO-ADRC (µm) |
|---|---:|---:|---:|
| S1 — Nominal (a_p = 0.3 mm) | **0.7765** | 0.8256 | 0.7826 |
| S2 — Aggressive (a_p = 0.6 mm) | **1.5580** | 1.8237 | 3.4145 ¹ |
| S3 — Model mismatch (ω −8 %) | **0.9001** | 20.79 ² | 1.1233 |
| S4 — High K_T (+30 %) | **1.0127** | 1.0784 | 1.0402 |

¹ 7 rung switches (panic + escalating locks); the transient of the first escape
inflates the RMS; ends parked on the certified rung, stable.
² The certified design's −8 % sensitivity hole: a BOUNDED saturation limit cycle
(y_max 61 µm, u at ±150 V), not a divergence. A-ESO-ADRC avoids it by staying on
the performance rung (0 switches).

Control effort (S1): LQG u_max = 23.0 V, u_RMS = 5.6 V; ESO-ADRC 23.4/5.9 V;
A-ESO-ADRC 21.7/5.0 V — all far below ±150 V. Bit-reproducible across runs.

### The result, honestly

**Inside the fixed-design envelope the correctly-modelled LQG is the best
regulator** — the ESO trades ~6 % nominal RMS for its disturbance states, and the
supervised ladder brings that back to −0.8 % (S1: 0.783 vs 0.777). The ESO's
return shows up OUTSIDE the LQG envelope (§2). The certified fixed design's −8 %
hole and the performance design's a_p = 0.6 divergence are complementary — see the
certification figure (`fig06_certification`) — which is precisely the case for the
supervised ladder: **A-ESO-ADRC is the only controller in the study that never
diverges** (all 4 scenarios here + all 5 stress cases of §2).

## 2. Material-removal drift & robustness stress (`main_adaptive_removal.py`)

The plant's modal frequencies drift DURING the pass (solver schedule Kp·s(t)²,
Cp·s(t); ramp over 60 % of the pass, then hold), or are statically perturbed, or
the actuator effectiveness is degraded. All controllers frozen nominal designs;
A-ESO-ADRC additionally switches rungs online (measured cost only — no
identification, no probe).

| Case | LQG (fixed) | ESO-ADRC (fixed, certified) | **A-ESO-ADRC** |
|---|---:|---:|---:|
| D0 no drift (sanity) | **0.777** | 0.826 | 0.783 (adaptation costs ~nothing) |
| D1 ramp to +15 % (article's direction) | **0.682** | 1.276 | 1.256 |
| D2 ramp to −12 % (beyond LQG margin) | 260 µm ✗ | **0.898** | 1.151 — survives |
| D3 static −12 % | 483 µm ✗ | **1.140** | 1.708 — survives |
| D4 piezo effectiveness ×0.25 | 1.241 | 1.221 | **1.184** |

Key mechanism: the ESO's per-mode disturbance states absorb the stiffness-drift
model error that biases the plain Kalman filter — the fixed LQG diverges at −12 %
(4 % beyond its ~−9 % margin) while the certified ESO design rides through at
~1 µm. The A-ESO-ADRC rung traces (`fig_adaptive_removal`) show the supervisor
escalating to the certified rung as the drift crosses the performance rung's
comfort zone (D2: one switch at ~330 ms), and a probe-and-return pattern on D1.

**Rejected mechanism (documented negative result):** closed-loop
actuator-effectiveness self-identification from the d̂–u regression is BIASED (the
periodic cutting force correlates with u through the feedback path; the estimate
ran to its projection bounds in the wrong direction) — consistent with this
package's identifiability finding that parameter identification under stable
periodic cutting requires persistent excitation. D4 shows the loop tolerates
×0.25 effectiveness anyway (gain margin), so no adaptation is needed on this axis
within the envelope studied.

## 3. Monte-Carlo robustness (50 samples; held-out; divergence reported)

`python main_robustness_mc.py` (±15 % cutting constants, ±3 % modal frequency,
±20 % damping — the LQG-safe neighbourhood):

- Converged: **LQG 50/50, ESO-ADRC 50/50, A-ESO-ADRC 50/50** (no survivorship
  bias — divergence would be counted; none occurred within these ranges).
- RMS medians: LQG **0.788 µm** [p05 0.683, p95 0.882]; ESO-ADRC 0.850 µm
  [0.710, 0.977]; A-ESO-ADRC 0.886 µm [0.696, 1.361].
- Pairwise vs LQG: ESO-ADRC median −7.3 % (better in 0 % of samples);
  A-ESO-ADRC median −11.7 % (better in 26 %; heavy left tail from occasional
  panic transients).

Honest reading: within ±3 % frequency uncertainty — inside the LQG margin — **LQG
wins the Monte-Carlo**, as §1 predicts. The ESO family's advantage is confined to
where it claims it: drift/mismatch beyond the LQG envelope (§2), where the MC's
uniform draws never go. A wider-mismatch MC would mix divergences of LQG with
survivals of ESO-ADRC (see D2/D3).

## 4. Stability lobe diagram — rigorous closed-loop monodromy, worst of 3 tool positions, at 4900 RPM

| Configuration | a_p critical | vs Open-Loop |
|---|---:|---:|
| Open-Loop (coupled monodromy, worst position) | 0.100 mm | 1× |
| LQG (closed-loop monodromy) | **1.075 mm** | 10.8× |
| ESO-ADRC certified design (generic monodromy) | 0.913 mm | 9.1× |

Notes:
1. **Open-loop anchor still validated:** 0.10 mm at 4900 RPM = Du et al. (2024)
   Fig. 18 experimental limit.
2. Both controlled depths are the same order as the article's experimentally
   achieved 0.6–0.8 mm controlled limits. LQG's nominal-plant lobe is higher —
   consistent with §1 (nominal optimality) — while the certified ESO design's
   lobe is what A-ESO-ADRC's panic fallback guarantees at the nominal plant.
3. The ESO's d-state leakage (β_d = 10 rad/s) contributes a constant Floquet
   multiplier exp(−β_d τ) ≈ 0.96, irrelevant to the chatter boundary.
4. Certification caveat: ρ values within a few % of 1 are marginal and
   m_div-sensitive; the certification is a comparative design-selection criterion,
   always cross-checked in the time domain (S3 shows a bounded limit cycle at a
   point where ρ ≈ 0.99–1.10 depending on discretisation).

## 5. Canonical LADRC — the negative result (reproducible)

Textbook output-form LADRC (ÿ = f + b₀u, Gao bandwidth parametrisation,
`CanonicalLADRC_Controller`) destabilizes this plant for EVERY tested bandwidth
pair (wc = 2π·50–800 Hz × wo = 2π·800–2500 Hz), even without cutting. Cause: the
piezo→tip-sensor transfer is non-collocated with alternating modal residues
(D·H = −0.40/+0.65/−0.19), DC gain −2.4e-8 vs high-frequency gain +0.057 —
opposite signs ⟹ an odd number of real RHP zeros ⟹ no single b₀ sign is correct
across frequency. This is why the package's ADRC is formulated in modal space.

## 6. FEM verification (mesh convergence)

`python 03_analysis/mesh_convergence.py`: frequencies converged to <0.1 % by 30×24
(521.1/1069.9/2732.8/3334/4145 Hz); uniform ~2.6 % below the article's
Chebyshev-Ritz *theory* (discretisation-model difference — the non-conforming
element converges from below) but within **0.2–0.6 % of the MEASURED** modes
2, 4, 5 (Table 4).

## 7. Reproducibility notes

- One eigensolve builds the nominal 5-mode plant; truncation feeds the
  controllers, perturbed copies feed the scenarios (consistent mode signs).
- All RNGs seeded (measurement noise `1234`, identical for all controllers; the
  design grid is deterministic). Bit-reproducible across runs.
- Runtimes: `main_simulation.py` ≈ 4.5 min (design grid + certification ≈ 2 min,
  9 SLD monodromy grids ≈ 35 s); `main_robustness_mc.py` ≈ 4 min;
  `main_adaptive_removal.py` ≈ 1.5 min. No GPU.

## 8. Remaining items (P3)

- Experimental validation on a physical plate (the article's rig is fully
  specified — Tables 1–3 + Fig. 17 list every instrument). Everything above is
  simulation.
- A dedicated ADRC/ESO + supervisory-adaptation literature search before
  submission (see CONTRIBUTION.md §5).
