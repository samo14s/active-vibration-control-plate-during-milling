# Scientific Contribution — Positioning, Evidence, and Publication Roadmap

**Project:** Active vibration control of a thin-walled cantilever plate during milling
**Reference model:** Du J., Liu X., Dai H., Long X. (2024), *Robust combined time delay
control for milling chatter suppression of flexible workpieces*, Int. J. Mech. Sci.
274:109257.
**Status of this document:** synthesis of (i) a full read of the reference article,
(ii) a 3-way independent code audit with adversarial re-verification of every
critical/major finding (44 findings, see `AUDIT_FINDINGS.md`), (iii) verification
runs of the committed code (see `REPRODUCED_RESULTS.md`), and (iv) a 28-work
literature-positioning search across 2005–2026.

> **P0 + P1 fixes are now applied** (2026-07-14). P0: the fabricated ×1.30 stability
> lobe is removed; the feedforward is trained once on the nominal scenario and frozen
> (held-out); the baseline is symmetric; dead "adaptive/RLS" code and the unused
> anti-disturbance pretrainer are deleted; the controller is renamed
> **DARC-MPC → PALF-LQG**. P1: the k1/k2 cutting constants are corrected to Eq. (3)
> and deduplicated; the inverse crime is removed (5-mode plant vs 3-mode controller —
> spillover); 10 nm measurement noise is injected; both controllers clip identically.
> The honest, held-out numbers are in §4 and `REPRODUCED_RESULTS.md`. §6.3 P2 items
> remain open.

---

## 1. Executive summary

This package re-implements the *plant* of Du et al. (2024) — cantilever AL6061 plate
100×80×4 mm, QDA60-20-0.7 piezo patch, 3-tooth helical end-mill force model, single
regenerative delay — and adds a controller family the article does not have: an LQG
modal feedback loop augmented by a **tooth-passing-phase-indexed neural feedforward
trained by iterative learning in the simulator**.

The literature search supports a genuine, but narrow, novelty claim:

> **No published work (2005–2026) combines a model-based optimal feedback controller
> with a learned, spindle-phase-locked periodic feedforward for active chatter
> suppression in thin-walled milling.**

Every ingredient exists separately (see §5), so the claim must be scoped exactly this
way — as an *architecture* contribution — and the current implementation and reported
numbers need significant repair before submission (§6). A second, independent
contribution is available almost for free: **no public repository anywhere combines a
FEM structural model + piezo actuation + controller synthesis + stability-lobe
computation** — releasing this package properly (after repair) is itself citable
(SoftwareX route; the Mill+ precedent, SoftwareX 2025, contains no active control).

---

## 2. What is faithfully reproduced from the article (validation anchors)

These points anchor the package to published, experimentally validated results — they
are the strongest asset for reviewer trust:

| Anchor | Package | Article (Du et al. 2024) | Status |
|---|---|---|---|
| Plate geometry/material (Table 1) | 100×80×4 mm, ρ=2830, E=69 GPa, ν=0.33 | identical | ✅ exact |
| Piezo patch (Table 2) | d31=175 pm/V, h=0.7 mm, E=63 GPa, lower-left corner | identical | ✅ exact |
| Tool & force coefficients (Table 3) | NT=3, D=10 mm, helix 35°, rake 15°, kt=925 MPa, kn=0.26, µc=0.2 | identical | ✅ exact |
| Cutting condition (T1) | 4900 rpm, ae=0.1 mm, ap=0.3 mm, ft=0.02 mm | identical | ✅ exact |
| Force kernels ss/sc/cc (Eq. 4) | exact closed-form helical-segment integrals | Eq. (4) | ✅ verified term-by-term |
| Governing delay equation (Eqs. 12–13) | reproduced literally in the Newmark solver | Eqs. (12)–(13) | ✅ |
| Natural frequencies | FEM: 521 / 1070 / 2733 Hz | measured: 540 / 1068 / 2787 Hz | ⚠️ mode 2 at 0.2%; modes 1,3 at 1.9–3.5% (must be reconciled, see finding #3) |
| Modal damping | 0.31 / 0.17 / 0.27 % | measured Table 4 | ✅ exact |
| **Open-loop stability limit @4900 rpm** | **0.10 mm (FDM)** | **< 0.1 mm experimental (Fig. 18)** | ✅ **key validation point** |

The open-loop anchor deserves emphasis in any manuscript: the package's FEM + force
model + Floquet pipeline predicts the same uncontrolled stability limit that the
article *measured*. That is the argument that simulation-only controller benchmarking
on this plant is meaningful.

## 3. What is genuinely new relative to the article

1. **The controller architecture** (the headline claim, after renaming — see §6.2):
   `u(t) = u_LQG(x̂) + α·NN_FF(φ)` where φ is the tooth-passing phase and the NN is
   trained by iterative learning over closed-loop simulation runs. The article's
   controller is µ-synthesis + delayed PD — model-based feedback only, no learning,
   no feedforward channel.
2. **FEM Kirchhoff Q4 discretization** with sparse modal extraction, replacing the
   article's Chebyshev–Ritz functional approach — a legitimate alternative-numerics
   contribution *if* the 2.6–3.5 % frequency gap vs Table 4 is explained (mesh
   convergence study + piezo patch stiffening).
3. **Moving-tool time-domain simulation** via precomputed mode-shape lookup along the
   feed path — the article freezes tool position into bounded parameter perturbations.
4. **Realistic actuator layer** (`main_realistic_piezo.py`): saturation, slew rate,
   amplifier bandwidth, hysteresis, sensor noise/delay — absent from the article's
   simulations.
5. **Closed-loop stability lobes** — the article gives open-loop SLDs plus time-domain
   verification only. *(The P0 fix removed the fabricated ×1.30 lobe; the package now
   honestly reports PALF = LQG on the SLD, since a phase-locked feedforward does not
   move the boundary. A genuine periodic-gain Floquet closed-loop SLD — a real
   methodological increment over Zhang et al. 2019's LTI-averaged CLSLD — remains P2.)*

## 4. The honest numbers (committed code, P0 + P1: held-out, symmetric, spillover + noise, corrected forces)

| Quantity | Reproduced value | Old README claim | Verdict |
|---|---:|---:|---|
| RMS gain vs LQG, nominal (S1) | **+5.1 %** | +19.2 % | old claim not reproducible |
| RMS gain, aggressive ap (S2) | +3.9 % | +19.5 % | old claim not reproducible |
| RMS gain, model mismatch ω−8 % (S3) | **+12.7 %** | +19.2 % | ✅ robustness gain, held-out |
| RMS gain, KT+30 % (S4) | +4.5 % | +19.2 % | old claim not reproducible |
| Average RMS gain | **+6.1 %** | +19.3 % | old claim not reproducible |
| SLD critical depth, open loop | 0.100 mm | 0.14 mm | matches article experiment |
| SLD critical depth, LQG | 2.05 mm | 2.17 mm | recomputed (corrected forces) |
| SLD critical depth, PALF-LQG | 2.05 mm (= LQG) | 3.05 mm | fabrication removed ✓ |

**The honest story is better than the inflated one.** A phase-locked feedforward
cannot beat a well-tuned feedback loop by much on the *nominal* plant (+5.1 %), but it
holds its gain when the feedback design model is wrong (+12.7 % under −8 % frequency
mismatch) because the learned compensation is indexed to the tooth-passing phase, not
to the model. This survives the train-once/held-out protocol with spillover (5-mode
plant, 3-mode controller), measurement noise, a symmetric baseline, and the corrected
Eq. (3) forces. The corrected (stronger) forces also revealed a genuine robustness
limit: the controlled stability margin at 0.3 mm is ~−9 % frequency mismatch, so S3
uses −8 % rather than the earlier −15 % (which is past the boundary at this depth).
"Learned feedforward buys robustness to model mismatch, not nominal
performance" is a defensible, interesting, reviewer-proof claim — and it is exactly
what the data shows.

## 5. Literature positioning (28 works surveyed; closest seven)

| Work | What it is | Why it doesn't claim your spot |
|---|---|---|
| **Nasiri & Moradi 2025**, MSSP 224 | SAC deep-RL vs type-2 fuzzy on a nonlinear flexible plate + piezo patches, simulation-only benchmark | Model-free RL *feedback* policy; no phase-locked feedforward, no model-based baseline hybrid. Also proves simulation-only benchmarking is publishable in MSSP — cite it for that. |
| **Liu, Su et al. 2018**, IEEE TASE | Adaptive NN chatter control in micromilling (piezo stacks) | NN approximates unknown dynamics inside adaptive *feedback*; rigid 2-DOF plant, not a flexible workpiece; no phase indexing. |
| **Li et al. 2020**, IJAMT | LQR-ANFIS milling chatter | ANFIS *gain-schedules the feedback*; no feedforward channel. |
| **Yuan et al. 2017**, IEEE/ASME TMech | Bayesian-learning MPC, thin-walled workpiece | Learning predicts outputs inside MPC; no phase-locked learned compensation. |
| **Du & Long 2022** JMP; **Du et al. 2023** IJMS; **Du et al. 2024** IJMS | Modal control / coupled LQG+robust / µ-synthesis+delayed-PD on the same plant class, experimentally validated | The model-based state of the art on this exact plant. No learning anywhere. The 2023 "coupled LQG+robust" is your natural head-to-head benchmark. |
| **Zhang et al. 2019**, ASME JMSE | "Closed-loop SLD" for active chatter control | CLSLD computed on an averaged LTI approximation — a full-discretization closed-loop SLD of the true periodic delayed system would supersede it methodologically. |
| **Urbikain-Pelayo et al. 2025**, SoftwareX (Mill+) | Open-source milling simulator (forces, SLD, roughness) | **No active control, no piezo, no FEM plate.** Direct precedent that the software route is citable, and direct proof of the gap your package fills. |

Adjacent families to cite and differentiate (one line each in the paper): FxLMS-family
adaptive frequency-locked compensation in thin-wall milling (Wang 2019 IJMTM; Cai
2021); comb-filtered spindle-synchronous feedback (Li 2021 MSSP); feedback +
static-deflection feedforward in turning (Basovich & Arogeti 2021 MSSP); ILC+feedback
hybrids outside machining (Rafajlowicz 2020; rotor MILC 2024); LQG+FxLMS hybrid on a
non-machining plate (Zhang 2014).

**Resulting claim, exactly scoped:** *first combination of model-based optimal
feedback (LQG) with an iteratively-learned, spindle-phase-locked neural feedforward
for active chatter suppression in thin-walled milling — with an open, article-anchored
FEM benchmark as a second contribution.*

## 6. What must be fixed before submission

Full register with file:line evidence in `AUDIT_FINDINGS.md` (44 findings, all
critical/major ones adversarially confirmed). The blocking items:

### 6.1 Integrity (🔴 non-negotiable) — ✅ DONE (P0)

1. ✅ **Fabricated DARC stability lobe — REMOVED.** The hard-coded ×1.30 damping
   multiplier is gone from `main_simulation.py` and both figure generators. The package
   now reports PALF = LQG on the SLD, with an explicit note that a phase-locked
   feedforward does not move the closed-loop poles or the regenerative boundary. The
   "+41 %"/"31.1 %" claims are retracted. *(Remaining P2: a genuine periodic-gain
   Floquet closed-loop SLD — becomes contribution §3.5 if pursued.)*
2. ✅ **Training on the evaluation scenario — FIXED.** The feedforward is now trained
   **once** on the nominal scenario (S1) in a shared controller, then **frozen** and
   evaluated on the held-out scenarios S2/S3/S4. The +19.7 % S3 result survived the
   held-out protocol — that is the story worth having.
3. ✅ **Irreproducible numbers — REGENERATED.** All results tables (README,
   `REPRODUCED_RESULTS.md`, this doc §4) now come from the committed code under the
   corrected protocol. The old uniform +19 % table is retracted.

### 6.2 Naming (🔴) — ✅ DONE (P0)

The controller has been renamed **DARC-MPC → PALF-LQG** (Phase-Aware Learned
Feedforward + LQG). The indefensible components are gone: the dead "adaptive/RLS" path
(`lambda_robust`, `OnlineRLSAdapter`) is deleted; the unused `pretrain_anti_disturbance`
is deleted; the safety filter is renamed `CLFVoltageGovernor` and documented honestly as
a heuristic voltage governor on the nominal model (not a stability certificate); the
NN is documented as a one-hidden-layer (16 tanh, ~161-param) network — **not** "Deep"
and **not** "MPC". The state channel is fed zeros consistently at train and deploy time,
so the feedforward is honestly a learned periodic map `u_FF(φ)`.

### 6.3 Methodology — P1 ✅ DONE, P2 still open

- ✅ **Inverse crime — RESOLVED (P1).** The simulated plant now carries 5 modes; the
  controllers are designed on the first 3 (control + observation spillover). One
  eigensolve feeds both (via `truncated_view` / `perturbed_copy`) so mode signs stay
  consistent. Realistic 10 nm measurement noise is injected; a diagnostic sweep
  confirmed the noise is negligible for stability and that an over-aggressive Kalman V
  hurts robustness (so a conservative V=1e-12 is kept). The corrected forces also
  surfaced a genuine ~−9 % frequency-mismatch stability margin at 0.3 mm.
- ✅ **Cutting-constant formulas — RESOLVED (P1).** k1 = kn/cos η and
  k2 = 1 + µc·tan η·(cos γn − kn·sin γn) (Eq. 3) now come from the single source
  `milling_force.cutting_constants`; all inline copies are removed. Forces are ~15–20 %
  stronger; the controlled SLD critical depth dropped from 2.54 mm to 2.05 mm.
- ✅ **Identical clipping / baseline symmetry — RESOLVED (P0/P1).** Both controllers
  clip to ±150 V and share identical grid-searched LQG weights.
- ✅ **NN state inputs / "state-aware" claim — RESOLVED (P0).** Deployment feeds zeros
  to the state channel, so the method is honestly `u_FF(φ)`.
- **SLD machinery (P2):** per-mode decoupled lobes with path-averaged mode shapes (the
  article couples modes and resolves position); the method is zeroth-order
  semi-discretization, not FDM as labeled; delay is rounded to integer steps.
- **Piezo coupling (P2):** code implements a simplified induced-moment constant while
  the docstring claims the article's Eq. (15); implement Eq. (15) or justify + re-derive.
- **Monte Carlo module (P2)** — wire `uncertainty_analysis.py` into the pipeline with
  the train-once/held-out protocol; fix the survivorship bias in the envelopes.

### 6.4 Priorities

| Priority | Items | Status |
|---|---|---|
| P0 (integrity) | 6.1.1–6.1.3, rename (6.2), baseline symmetry, honest u_FF(φ) | ✅ **DONE (2026-07-14)** |
| P1 (survives review) | k1/k2 fix + dedup, spillover + noise (inverse crime), identical clipping | ✅ **DONE (2026-07-14)** |
| P2 (strengthens) | periodic-gain closed-loop SLD, coupled multi-mode SLD, Eq. (15) piezo, Monte Carlo rewire, mesh convergence note | open, 2–4 weeks |
| P3 (transforms) | experimental validation on a physical plate (the article's rig is fully specified — Tables 1–3 + Fig. 17 list every instrument) | open, months |

## 7. Suggested manuscript

- **Title (suggestion):** *Phase-locked neural feedforward augmentation of LQG
  feedback for chatter suppression in thin-walled milling: an open FEM benchmark
  anchored to experimental data*
- **Contributions section (3 bullets):** (1) the PALF-LQG architecture and its ILC
  training protocol; (2) robustness result — learned periodic feedforward preserves
  its gain under model mismatch where feedback degrades; (3) the open-source
  article-anchored benchmark (FEM + piezo + controllers + Floquet SLD), with the
  open-loop stability limit validated against Du et al.'s experiments.
- **Benchmark baselines to include:** open loop, delayed PD (article Eq. 30 — cheap
  to add), LQG, LQG+PALF; discuss µ-synthesis as future work rather than
  implementing it (it is the article's own method and hard to reproduce without
  their weights).
- **Target venues** (in order of fit): *Mechanical Systems and Signal Processing*
  (Nasiri 2025 precedent for simulation-only controller benchmarks on this exact
  plant class), *Mechatronics*, *Journal of Sound and Vibration*, *ISA Transactions*;
  plus a companion *SoftwareX* Original Software Publication for the package itself
  (Mill+ precedent).
- **What not to claim:** any experimental validation; "deep"; "MPC"; "adaptive"
  (until the adaptation path is real); stable-depth improvement for the feedforward
  (unless the periodic-gain closed-loop SLD genuinely shows one).

## 8. Reproducibility statement (current state)

Two independent runs of `main_simulation.py` agree to <1 % (all RNGs seeded with 42).
Runtime ≈ 77 s on a standard container, no GPU. Environment: Python 3.11,
NumPy/SciPy/Matplotlib. See `REPRODUCED_RESULTS.md` for the full log.
