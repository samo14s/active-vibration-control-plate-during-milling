# Scientific Contribution — Positioning, Evidence, and Publication Roadmap

**Project:** Active vibration control of a thin-walled cantilever plate during milling
**Reference model:** Du J., Liu X., Dai H., Long X. (2024), *Robust combined time delay
control for milling chatter suppression of flexible workpieces*, Int. J. Mech. Sci.
274:109257.
**Status of this document:** synthesis of (i) a full read of the reference article,
(ii) a 3-way independent code audit (44 findings, see `AUDIT_FINDINGS.md` —
historical record of the earlier PALF-LQG phase), (iii) verification runs of the
committed code (see `REPRODUCED_RESULTS.md`), and (iv) the ESO-ADRC design study
(P4, 2026-07-15).

> **Package history.** P0–P2.6 built an audited, article-anchored benchmark around a
> learned-feedforward controller (PALF-LQG / A-PALF-LQG). On 2026-07-15 the user
> requested that the entire learned-feedforward strategy family be REMOVED and a new
> control strategy be proposed and developed into an adaptive one. The current
> package is the result: **LQG (benchmark baseline) vs ESO-ADRC (modal
> extended-state-observer ADRC) vs A-ESO-ADRC (its adaptive, cost-supervised
> development)**. All protocol/integrity fixes of P0–P2 (train-once/held-out,
> symmetric baselines, corrected Eq. 3 forces, spillover + measurement noise,
> Eq. 15 piezo, rigorous monodromy SLD) carry over unchanged. Only P3
> (experimental validation) remains open.
>
> **P5 (2026-07-15, "beat LQG inside the envelope too"):** the ESO-ADRC family
> gains a **harmonic resonant cancellation (HRC) layer** — per-tooth-harmonic LTI
> resonant compensators with inverse-closed-loop-FRF phase (the online, causal
> counterpart of the earlier model-inverse ILC; n_harm capped at 4 so the lines
> stay clear of mode 2's drift band) — and A-ESO-ADRC becomes a **4-rung
> supervised ladder** [HRC / performance-ESO / quasi-Kalman-ESO / certified-ESO]
> with supervisor v3 (rising-energy cascade panic with severity-based target,
> recovery-trend holds, per-pass failure flags, probe aborts, desperation probing
> from the robust end, escalating locks). Result: **nominal RMS ~50 % BELOW LQG
> (0.38 vs 0.78 µm), Monte-Carlo median +48 % vs LQG (better in 94 % of samples,
> all 50/50 converged), still the only controller that never diverges across all
> 9 stress scenarios** (LQG diverges beyond −10 % drift). New negative result
> (DESIGN FINDING #5): harmonic states inside the ESO destabilise the true plant
> via estimator spillover — the resonant layer must live in the controller output
> path, driven by y. Honest remaining weak spot: at ω−8 % (S3) the adaptive
> recovery is bounded but slow (~5–15 µm over the 0.5 s pass vs LQG's 0.90 µm) —
> the certified rung's sensitivity hole; documented, not hidden. Verified numbers:
> `REPRODUCED_RESULTS.md`.
> **P6 (2026-07-15, "real-time ID along the tool path"):** the phenomenological
> drift is replaced by a **physically accurate material-removal FEM** (per-element
> thickness field; MAC mode tracking) driving a **multi-pass finishing sequence**
> that thins the wall non-uniformly to −15 % (article 9–17 %). An **active piezo
> probe** identifies the drifting modal frequencies UNBIASED (≤1.1 %; a passive
> estimate is biased — returns the tooth harmonics, per the documented
> persistent-excitation obstruction), and an **ID-scheduled controller** re-tunes
> per pass. Result: fixed LQG loses control on 7 of 24 passes (worst 10.9 µm);
> ID-scheduled LQG stays ≤0.072 µm (151× better, matches the true-frequency
> oracle); the robust ESO-ADRC survives WITHOUT ID — identification and
> disturbance-observer robustness are shown to be complementary. Verified numbers:
> `REPRODUCED_RESULTS.md`.

---

## 1. Executive summary

This package re-implements the *plant* of Du et al. (2024) — cantilever AL6061 plate
100×80×4 mm, QDA60-20-0.7 piezo patch, 3-tooth helical end-mill force model, single
regenerative delay — and develops a controller family the article does not have:
**active disturbance rejection control formulated in modal space** (a per-mode
total-disturbance extended state observer feeding the same output-weighted LQR used
by the LQG baseline), and its **adaptive development**: a supervised ladder of two
pre-designed tunings — a *performance* design and a *Floquet-certified robust*
design — switched online by the measured vibration cost alone (no identification,
no probe), with a panic fallback and escalating locks.

The two headline results, both held-out and bit-reproducible:

1. **Architectural robustness of the modal ESO.** The certified fixed ESO-ADRC
   survives frequency drift **4 % beyond the margin at which the fixed LQG
   diverges** (−12 % static AND ramped during the pass, ~0.9–1.1 µm), because the
   disturbance states absorb model error that biases a plain Kalman filter. The
   price is honest and stated: ~6 % higher RMS than LQG on the nominal plant, and a
   bounded (~21 µm) sensitivity hole at −8 % mismatch.
2. **The supervised ladder removes every fixed design's failure mode.**
   A-ESO-ADRC is the **only controller in the study that never diverges** across
   the 4 held-out scenarios + 5 drift/stress cases: it rides the performance rung
   to within 0.8 % of LQG nominally, escapes to the certified rung under drift
   beyond the LQG margin (where LQG diverges), and avoids the certified rung's −8 %
   hole by staying on the performance rung there.

A third contribution is methodological: the **generic closed-loop coupled monodromy
SLD** (any LTI output-feedback controller in realization form embedded in the full
time-periodic delayed system) — used both to compute rigorous closed-loop stability
lobes for the ESO controller and, at design time, to **certify the robust rung**
over a frequency-mismatch × depth × tool-position ball. The negative results are
documented as findings (see §3.4): canonical output LADRC is structurally
inapplicable to this non-collocated NMP plant, matched disturbance cancellation
does not pay, and closed-loop effectiveness self-identification is biased.

As before: **no public repository combines a FEM structural model + piezo actuation
+ controller synthesis + stability-lobe computation** — releasing this package
(SoftwareX route; the Mill+ precedent contains no active control) remains a second,
nearly-free contribution.

---

## 2. What is faithfully reproduced from the article (validation anchors)

These points anchor the package to published, experimentally validated results:

| Anchor | Package | Article (Du et al. 2024) | Status |
|---|---|---|---|
| Plate geometry/material (Table 1) | 100×80×4 mm, ρ=2830, E=69 GPa, ν=0.33 | identical | ✅ exact |
| Piezo patch (Table 2) | d31=175 pm/V, h=0.7 mm, E=63 GPa, lower-left corner, Eq. (15) C_P0 coupling | identical | ✅ exact |
| Tool & force coefficients (Table 3) | NT=3, D=10 mm, helix 35°, rake 15°, kt=925 MPa, kn=0.26, µc=0.2; k1, k2 per Eq. (3) | identical | ✅ exact |
| Cutting condition (T1) | 4900 rpm, ae=0.1 mm, ap=0.3 mm, ft=0.02 mm | identical | ✅ exact |
| Force kernels ss/sc/cc (Eq. 4) | exact closed-form helical-segment integrals | Eq. (4) | ✅ verified term-by-term |
| Governing delay equation (Eqs. 12–13) | reproduced literally in the Newmark solver | Eqs. (12)–(13) | ✅ |
| Natural frequencies | FEM: 521 / 1070 / 2733 Hz | measured: 540 / 1068 / 2787 Hz | ✅ converged <0.1 % by 30×24; within 0.2–0.6 % of measured modes 2, 4, 5 (see mesh_convergence) |
| Modal damping | 0.31 / 0.17 / 0.27 % | measured Table 4 | ✅ exact |
| **Open-loop stability limit @4900 rpm** | **0.10 mm (coupled monodromy, worst of 3 positions)** | **< 0.1 mm experimental (Fig. 18)** | ✅ **key validation point** |

## 3. What is genuinely new relative to the article

1. **Modal-space ESO-ADRC for flexible-workpiece chatter.** The article's
   controller is µ-synthesis + delayed PD — model-based feedback with the cutting
   force treated as bounded uncertainty. Here the regenerative force, feed forcing,
   spillover and drift are lumped into a per-mode **total disturbance d(t) ∈ R³**,
   estimated online by a 9-state ESO (Riccati-designed gain, disturbance-intensity
   knob σ_d) and fed to the same output-weighted LQR as the baseline. The
   comparison LQG vs ESO-ADRC therefore isolates exactly one ingredient — replace
   the Kalman filter with a disturbance-estimating ESO — and shows what it buys:
   survival at −12 % drift where LQG diverges.
2. **The adaptive development (A-ESO-ADRC): a certified design ladder under a
   measurement-cost supervisor.** Two rungs — the grid's nominal-best design and
   the design minimizing the worst-case Floquet radius over a design-time
   uncertainty ball — share one physical observer state (bumpless switching) and
   are switched by the measured y²-cost only: slow-EMA toggling with a running-min
   quiet level, dwell and hysteresis; fast-EMA **panic** jump to the certified rung
   (absolute floor active from step one); **escalating post-panic locks** that
   provably damp hole↔rung limit cycling. No parameter identification anywhere —
   which is precisely what the identifiability finding (below) demands.
3. **Generic closed-loop coupled monodromy SLD** (`closed_loop_rho_generic`):
   embeds ANY LTI output-feedback controller (LQG m=2n, ESO-ADRC m=9, …) in the
   full coupled, time-periodic delayed system — no averaged-LTI surrogate (supersedes
   Zhang et al. 2019's CLSLD methodology), evaluated at the worst of 3 tool
   positions (the article's Fig. 6 treatment). Also powers the design-time rung
   certification — a new use of Floquet analysis *inside* the adaptive-controller
   design loop.
4. **Documented negative results** (each reproducible from the committed code):
   - **Canonical output LADRC is structurally inapplicable** here: the
     piezo→sensor transfer is non-collocated with alternating modal residues
     (D·H = −0.40/+0.65/−0.19), DC and HF gains of opposite sign ⟹ real RHP zeros
     ⟹ the ÿ = f + b₀u premise fails and the loop is unstable for EVERY bandwidth
     pair (verified over wc = 2π·50–800 Hz × wo = 2π·800–2500 Hz, even without
     cutting). This motivates the modal-space formulation and is a useful warning
     for the ADRC-in-machining literature.
   - **Matched disturbance cancellation does not pay** on this plant: the actuator
     direction is only ~19 % aligned with the tool-force direction, so the γ·Hᵀd̂
     channel adds effort and spillover excitation for no RMS benefit — the grid
     selects γ = 0; the ESO's value is disturbance-aware state estimation.
   - **Closed-loop actuator-effectiveness self-identification is biased**: d̂
     contains the periodic cutting force, which correlates with u through the
     feedback path and swamps the (κ_true−κ̂)·H·u signal (the estimator ran to its
     projection bounds in the wrong direction). This extends the package's earlier
     identifiability finding — under stable periodic cutting, parameter
     identification requires persistent excitation — and motivates the
     identification-FREE supervisor of A-ESO-ADRC.
5. **FEM Kirchhoff Q4 discretization** with sparse modal extraction (vs the
   article's Chebyshev–Ritz), moving-tool time-domain simulation, spillover
   evaluation (5-mode plant / 3-mode controllers), measurement noise, and a
   realistic-actuator layer (`main_realistic_piezo.py`) — carried over from the
   earlier phases.

## 4. The honest numbers (committed code — see REPRODUCED_RESULTS.md for the full log)

Held-out scenarios (controllers designed/frozen on the nominal model; T = 0.5 s;
y_RMS in µm):

| Scenario | LQG | ESO-ADRC (certified) | A-ESO-ADRC |
|---|---:|---:|---:|
| S1 nominal | **0.777** | 0.826 | 0.783 |
| S2 a_p = 0.6 mm | **1.558** | 1.824 | 3.41 (panic transient) |
| S3 ω−8 % | **0.900** | 20.8 (bounded hole) | 1.123 |
| S4 K_T+30 % | **1.013** | 1.078 | 1.040 |

Drift / stress benchmark (`main_adaptive_removal.py`):

| Case | LQG | ESO-ADRC | A-ESO-ADRC |
|---|---:|---:|---:|
| D0 no drift | **0.777** | 0.826 | 0.783 |
| D1 ramp to +15 % during pass | **0.682** | 1.276 | 1.256 |
| D2 ramp to −12 % during pass | DIVERGES | **0.898** | 1.151 |
| D3 static −12 % | DIVERGES | **1.140** | 1.708 |
| D4 piezo effectiveness ×0.25 | 1.241 | 1.221 | **1.184** |

**Honest reading.** Inside the fixed-design envelope the correctly-modelled LQG is
the best regulator — the ESO gives up ~6 % nominal RMS for its disturbance states.
The ESO's return is *architectural robustness*: it survives −12 % drift (static and
ramped) where LQG diverges. Neither fixed design covers everything — LQG dies
beyond −10 %, the certified ESO rings (bounded) at −8 %, the performance ESO
diverges at a_p = 0.6 — and the design-time Floquet map shows these holes are
complementary (waterbed). **A-ESO-ADRC is the only controller that never diverges
across all 9 cases**, at a modest cost above the per-case best. That the
adaptation is driven by measured cost alone (no identification) is what makes it
compatible with the identifiability obstruction documented in §3.4.

Monte-Carlo (±15 % cutting constants, ±3 % modal frequencies, ±20 % damping,
50 samples, divergence reported): see `REPRODUCED_RESULTS.md` §3 for the verified
statistics of the committed run.

Stability lobes (rigorous closed-loop monodromy, worst of 3 tool positions,
4900 RPM): OL 0.100 mm (= article experiment); LQG 1.075 mm (10.8× OL); ESO-ADRC
certified design — see `REPRODUCED_RESULTS.md` §4 (its boundary also bounds
A-ESO-ADRC after a panic, since the certified rung is A-ESO-ADRC's fallback).

## 5. Literature positioning (to be refreshed for the ADRC framing)

The 28-work survey of the earlier phase remains valid for the *benchmark/plant*
positioning (Du & Long 2022 JMP; Du et al. 2023 IJMS; Du et al. 2024 IJMS = the
model-based state of the art on this plant class; Zhang et al. 2019 ASME JMSE =
the averaged-LTI closed-loop SLD to supersede; Mill+ SoftwareX 2025 = the
no-active-control software precedent; Nasiri & Moradi 2025 MSSP = the
simulation-only controller-benchmark precedent).

For the NEW controller family a dedicated search is still required before
submission (ADRC/ESO in milling chatter and thin-wall machining; supervisory /
switching adaptive control with certified fallback; multi-model adaptive control
MMAC). Known near neighbours to differentiate: ADRC for spindle/servo loops and
grinding chatter; ESO-based vibration rejection of flexible structures
(spacecraft, manipulators); MMAC with dwell-time switching. The expected exact
claim — to be verified against that search:

> *First modal-ESO ADRC formulation for regenerative chatter suppression of a
> flexible workpiece, with a Floquet-certified two-rung supervisory adaptation
> driven by measured cost only, benchmarked symmetrically against LQG on an
> article-anchored, open FEM benchmark.*

## 6. Protocol integrity (carried over from P0–P2, all still in force)

- Train-once / freeze / evaluate-held-out; adaptation state reset per run.
- **Symmetric design machinery**: both controllers use the same output-weighted
  LQR construction and grid-search tuning on the nominal design model only; the
  certified-rung selection uses the design model's Floquet radius — no held-out
  time simulations enter any design decision.
- Corrected Eq. (3) constants from the single source `milling_force.cutting_constants`.
- No inverse crime: 5-mode plant vs 3-mode controllers; 10 nm measurement noise;
  identical ±150 V clipping; identical noise realisations across controllers.
- Divergence reported explicitly everywhere (no survivorship bias).
- All RNGs seeded; bit-reproducible across runs.

## 7. Suggested manuscript

- **Title (suggestion):** *Adaptive modal extended-state-observer control of
  milling chatter in thin-walled workpieces: a Floquet-certified supervisory
  design on an experimentally anchored benchmark*
- **Contributions (4 bullets):** (1) the modal-ESO ADRC formulation + the
  non-collocation/NMP negative result that mandates it; (2) the A-ESO-ADRC
  supervisory ladder with design-time Floquet certification of the fallback rung;
  (3) the generic closed-loop coupled monodromy SLD; (4) the open, article-anchored
  benchmark (open-loop limit = the article's measured 0.1 mm).
- **Baselines:** open loop, grid-tuned LQG (symmetric machinery), fixed ESO-ADRC
  (both selection criteria), A-ESO-ADRC. µ-synthesis remains future work (the
  article's own method, not reproducible without their weights).
- **Target venues:** *Mechanical Systems and Signal Processing*, *Mechatronics*,
  *Journal of Sound and Vibration*, *ISA Transactions*; companion *SoftwareX*
  paper for the package.
- **What not to claim:** experimental validation (P3 open); "certified stability"
  (the Floquet certification is a comparative design-selection criterion —
  marginal ρ values are m_div-sensitive and are always cross-checked in time
  domain); improvement over LQG *inside* the nominal envelope (LQG wins there —
  the contribution is robustness + never-diverging adaptation, and the paper must
  say so plainly).

## 8. Reproducibility statement (current state)

All numbers in §4 come from the committed code with fixed seeds (design grid
deterministic; measurement noise rng 1234, identical across controllers).
Runtimes on a standard container (no GPU): `main_simulation.py` ≈ 4 min
(incl. the design-grid certification and 9 monodromy SLD grids),
`main_robustness_mc.py` ≈ 4 min, `main_adaptive_removal.py` ≈ 1 min.
Environment: Python 3.11, NumPy/SciPy/Matplotlib. See `REPRODUCED_RESULTS.md`.
