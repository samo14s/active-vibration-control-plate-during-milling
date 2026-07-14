# Scientific Contribution — Positioning, Evidence, and Publication Roadmap

**Project:** Active vibration control of a thin-walled cantilever plate during milling
**Reference model:** Du J., Liu X., Dai H., Long X. (2024), *Robust combined time delay
control for milling chatter suppression of flexible workpieces*, Int. J. Mech. Sci.
274:109257.
**Status of this document:** synthesis of (i) a full read of the reference article,
(ii) a 3-way independent code audit with adversarial re-verification of every
critical/major finding (44 findings, see `AUDIT_FINDINGS.md`), (iii) two verification
runs of the committed code (see `REPRODUCED_RESULTS.md`), and (iv) a 28-work
literature-positioning search across 2005–2026.

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
   verification only. *(Currently the package's closed-loop SLD is not honestly
   computed — see §6.1; once fixed via periodic-gain Floquet analysis it becomes a
   real methodological increment over Zhang et al. 2019's LTI-averaged CLSLD.)*

## 4. The honest numbers (as of the committed code, 2 verification runs)

| Quantity | Reproduced value | Old README claim | Verdict |
|---|---:|---:|---|
| RMS gain vs LQG, nominal (S1) | **+4.6 %** | +19.2 % | claim not reproducible |
| RMS gain, aggressive ap (S2) | +4.4 % | +19.5 % | claim not reproducible |
| RMS gain, model mismatch ω−15 % (S3) | **+19.4 %** | +19.2 % | ✅ reproduces |
| RMS gain, KT+30 % (S4) | +4.6 % | +19.2 % | claim not reproducible |
| SLD critical depth, open loop | 0.100 mm | 0.14 mm | recomputed |
| SLD critical depth, LQG | 2.54 mm | 2.17 mm | recomputed |
| SLD critical depth, "DARC" | *(not a computed result — fabricated ×1.30 multiplier)* | 3.05 mm | **must be removed** |

**The honest story is better than the inflated one.** A phase-locked feedforward
cannot beat a well-tuned feedback loop by much on the *nominal* plant (+4.6 %), but it
holds its gain when the feedback design model is wrong (+19.4 % under −15 % frequency
mismatch) because the learned compensation is indexed to the tooth-passing phase, not
to the model. "Learned feedforward buys robustness to model mismatch, not nominal
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

### 6.1 Integrity (🔴 non-negotiable)

1. **Fabricated DARC stability lobe.** `main_simulation.py:636` multiplies the LQG
   closed-loop damping by a hard-coded 1.30 and presents the rerun lobe as the DARC
   SLD; the "+41 % stable depth" and "31.1 % effective damping" headline claims are
   artifacts of this multiplier (which also appears in two figure generators). The
   code's own pole plot admits the feedforward does not move closed-loop poles.
   *Fix:* either drop the DARC SLD claim, or do it properly — linearize the trained
   NN around the periodic orbit (periodic gain ∂u_FF/∂x̂(φ)) and run the discretization
   with that periodic feedback term. Done properly this becomes contribution §3.5.
2. **Training on the evaluation scenario.** Each scenario (including the
   "robustness" scenarios S3/S4) pre-trains the NN on the *exact* perturbed plant,
   force arrays, and trajectory it is then scored on — the +19 % robustness claims
   are circular as coded. *Fix:* train once on the nominal scenario, freeze the
   weights, evaluate on held-out perturbed scenarios. (The S3 result will likely
   survive — that is the story worth having.)
3. **Irreproducible README numbers.** The +19 % ×4-scenarios table cannot be produced
   by the committed code (§4). *Fix:* regenerate every number from the repository
   head; results tables in the README now point to `REPRODUCED_RESULTS.md`.

### 6.2 Naming (🔴 — reviewers will reject the vocabulary before reading the results)

"DARC-MPC: Deep Adaptive Robust Control with MPC" is indefensible: there is **no MPC**
(no receding horizon, no online optimization), the network is **not deep** (one hidden
layer, 16 tanh units, ~161 parameters), the **adaptive component is dead code**
(`lambda_robust` computed and never used; the "RLS" is a sign-gradient nudge), and the
**"Lyapunov" filter certifies the wrong plant** (nominal delay-free LTI, no milling
force, applied to the estimate, silent fallback to P=I). *Fix:* delete or repair the
dead/mislabeled components and rename honestly — e.g. **PALF-LQG** (Phase-Aware
Learned Feedforward LQG) or **ILC-FF/LQG**. The safety filter can stay if described
as a heuristic CLF-based voltage governor on the nominal model.

### 6.3 Methodology (🟠)

- **Inverse crime:** the simulation plant equals the controller design model (same 3
  modes, no unmodeled dynamics, near-noiseless Kalman with V=1e-12). Add ≥2 extra
  plant modes not given to the controller (spillover test) and realistic sensor noise.
- **NN state inputs are untrained:** training samples set x=0, so the state pathway
  carries its random initialization into deployment. Either train it or remove it and
  present the method honestly as u_FF(φ) — a learned periodic map.
- **Baseline symmetry:** give DARC's internal LQR the same grid search as the
  baseline LQG (or fix both to identical weights) so the delta is attributable to the
  feedforward alone; clip both controllers identically.
- **Cutting-constant formulas:** `k2` drops the parentheses of Eq. (3) and uses the
  helix angle where the rake angle belongs (−12.4 %); `k1` uses kn·cos η where the
  article has kn/cos η (−33 %). One-line fixes in `main_simulation.py:142-143` (and
  4 copies in other scripts — deduplicate into one module). All absolute force
  levels, and hence all SLD depths, shift when fixed.
- **SLD machinery:** per-mode decoupled lobes with path-averaged mode shapes (the
  article couples modes and resolves position); the method is zeroth-order
  semi-discretization, not FDM as labeled; delay is rounded to integer steps
  (effective 4878 rpm at the 4900 rpm operating point).
- **Piezo coupling:** code implements a simplified induced-moment constant while the
  docstring claims the article's Eq. (15); implement Eq. (15) or justify the model
  used and re-derive the gain.
- **Monte Carlo module is dead code** — wire `uncertainty_analysis.py` into the main
  pipeline (train-once/evaluate-perturbed protocol) and report honest confidence
  intervals; fix the survivorship bias in the envelopes.

### 6.4 Priorities

| Priority | Items | Effort |
|---|---|---|
| P0 (integrity) | 6.1.1–6.1.3, rename (6.2) | days |
| P1 (survives review) | train/test split, k1/k2 fix + dedup, baseline symmetry, spillover + noise | 1–2 weeks |
| P2 (strengthens) | periodic-gain closed-loop SLD, coupled multi-mode SLD, Eq. (15) piezo, Monte Carlo rewire, mesh convergence note | 2–4 weeks |
| P3 (transforms) | experimental validation on a physical plate (the article's rig is fully specified — Tables 1–3 + Fig. 17 list every instrument) | months |

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
