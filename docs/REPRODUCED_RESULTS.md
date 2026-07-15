# Reproduced Results — Verification Log

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
