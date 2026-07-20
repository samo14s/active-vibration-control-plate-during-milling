# Position-Scheduled LPV Active Chatter Control with Regenerative-Targeted Delayed Feedback for Thin-Walled Plate Milling

**Working title of the manuscript:**
*"Position-scheduled linear parameter-varying control with regenerative-targeted delayed feedback for chatter suppression in thin-walled plate milling"*

This document is the complete mathematical development of the proposed control
strategy. It is the source from which the manuscript (`paper/main.tex`) and the
reference implementation (`avc/`) are derived.

---

## 1. Motivation and gap

The state of the art in active chatter control of flexible (thin-walled)
workpieces is represented by Du *et al.*, *Int. J. Mech. Sci.* 274 (2024)
109257, who combine a μ-synthesis robust controller with a delayed PD term.
Their controller treats **all** in-process variation — the position-dependent
modal participation of the milling point, the non-smooth (intermittent) cutting
force coefficients, and the modal drift caused by material removal — as
**norm-bounded uncertainty** around a single nominal plant.

The methodological observation that motivates this work:

> The dominant "uncertainty" in thin-wall milling is not uncertain at all.
> The tool position along the feed path is known in real time from the NC
> program and the drive encoders, and the material-removal state is a known
> monotone function of that position. A controller that *measures* these
> variables and *schedules* on them does not need to buy robustness against
> them — robustness that is paid for with conservatism (lower attainable
> depth of cut, higher control voltage).

**Proposed strategy (PS-LPV-DR):** a *position-scheduled linear
parameter-varying* (LPV) H∞ output-feedback controller, gain-scheduled on the
known tool position and material-removal state, combined with a
*spindle-synchronized delayed feedback* term that directly targets the
regenerative term of the cutting force. Norm-bounded uncertainty is retained
**only** for what is genuinely uncertain: truncated high-order modes
(spillover), cutting-coefficient dispersion, and modal-parameter tolerances.

Contributions (novelty claims vetted against the July-2026 adversarial
literature review, `docs/literature_positioning.md` — observe its list of
*forbidden claims*):

1. **C1 — Known variation as scheduling, residual variation as uncertainty.**
   A grid-based LPV H∞ controller for piezo-patch-actuated thin-walled
   workpiece milling scheduled on tool position along the feed path and
   material-removal state — both *known in real time from the NC program /
   CAM model* (not "measured") — while a residual additive uncertainty block
   covers scheduling-map error and truncated-mode spillover. This
   de-conservatizes the robust baseline (Du et al. 2024) that wraps the same
   known variation into uncertainty; the conservatism gap is quantified in
   stable depth of cut and γ-level as a function of position.
2. **C2 — Jointly scheduled regenerative-targeted delayed feedback.** The
   spindle-synchronized delayed displacement-difference gain — previously
   fixed (Du et al. 2022; Dong et al. 2023) or mapped over spindle speed
   (JMP 2026) — is scheduled on the same (position, removal) pair as the H∞
   controller and tuned offline by maximizing the *closed-loop* critical
   depth of cut computed by semi-discretization of the full time-periodic
   delayed closed loop (controller dynamics included).
3. **C3 — Certification of the scheduled, time-periodic, delayed closed
   loop.** Stability of the parameter-varying periodic delay-differential
   closed loop — with full controller/actuator/filter dynamics — is
   certified by semi-discretization stability lobes along the tool path
   under a bounded scheduling rate. This certificate (not the grid LPV
   synthesis itself) is the formal stability guarantee, extending
   closed-loop-SLD practice (Zhang et al. 2019; Lehotzky & Bachrathy 2021)
   to scheduled controllers.

---

## 2. Plant modeling

### 2.1 Geometry and physical data

Cantilever plate (clamped at bottom edge `z = 0`), milled in down-milling
along its free top edge `z = h_p`, feed in `+x`. Data chosen identical to the
experimental setup of Du et al. (2024) so that simulation results are directly
comparable with their published measurements:

| Quantity | Symbol | Value |
|---|---|---|
| Plate length (feed dir.) | `l_p` | 100 mm |
| Plate height | `h_p` | 80 mm |
| Plate thickness | `b_p` | 4 mm |
| Material | — | AL6061: `E_p` = 69 GPa, `ν_p` = 0.33, `ρ_p` = 2830 kg/m³ |
| Modal damping ratios (measured, modes 1–5) | `ζ_i` | 0.31 %, 0.17 %, 0.27 %, 0.56 %, 0.35 % |
| Piezo patch (QDA60-20-0.7) | — | 60 × 20 × 0.7 mm, `E_pe` = 63 GPa, `ν_pe` = 0.35, `d31` = −175 pm/V, bottom-left corner |
| Tool | — | 3 flutes, Ø10 mm, helix 35°, rake 15° |
| Cutting coefficients | — | `k_t` = 925 MPa, `k_n` = 0.26, `μ_c` = 0.2 |
| Milling mode | — | down-milling, radial depth `a_e` = 0.1 mm, feed `f_t` = 0.02 mm/tooth |

### 2.2 Finite-element plate model

Reissner–Mindlin 4-node quadrilateral elements (selective reduced integration
on the shear terms to avoid locking), DOFs per node `(w, θ_x, θ_z)`, mesh
`N_x × N_z` (convergence-checked; 40 × 32 is ample). Assembly gives

```
M q̈ + C q̇ + K q = f(t),        q ∈ R^{3N}
```

Damping is *not* assembled: it is imposed modally with the measured ratios
(Rayleigh damping cannot match five arbitrary ratios; modal damping is exact
by construction in the reduced model and standard practice).

**Validation targets:** natural frequencies of the cantilever plate must
reproduce (i) the analytical thin-cantilever-plate benchmarks and (ii) the
frequencies reported by Du et al. (Table 4: first modes ≈ 322 Hz, 1101 Hz,
1996 Hz region — exact values extracted from the paper are stored in
`avc/params.py` and asserted in `tests/`).

### 2.3 Piezoelectric actuation

Surface-bonded patch driven by voltage `u(t)`; induced in-plane strain
`Λ = d31 u / h_pa` acting at offset `z_off = (b_p + h_pa)/2` from the
mid-plane produces the consistent nodal load

```
f_pe = Θ u,     Θ = Σ_e ∫_{A_e∩A_patch} B_b^T D_pe Λ̂ z_off dA
```

(`B_b` = bending strain-displacement matrix, `D_pe` = patch plane-stress
matrix, `Λ̂ = [1, 1, 0]^T Λ` isotropic in-plane actuation). This is the
standard induced-moment model for a thin bonded patch and reduces to the
pin-force model used in the reference paper in the thin-patch limit.

### 2.4 Modal reduction and the LPV structure

Mass-normalized modes `Φ = [φ_1 … φ_n]` (design model: `n = 3`; full
evaluation model: `n_full = 12`). With `q = Φ η`:

```
η̈ + diag(2ζ_iω_i) η̇ + diag(ω_i²) η = Φ^T Θ u + Φ^T e_w(x_T) F_y(t)
y_s = φ_w(P_s)^T η + v(t)
```

* `e_w(x_T)` — nodal interpolation vector of the transverse deflection at the
  *milling point* `(x_T, h_p)`; **this is where the tool position enters**.
  Write `b_f(x_T) = Φ^T e_w(x_T) ∈ R^n`.
* `P_s` — fixed sensor point (eddy-current probe, top right corner region);
  the output matrix is **constant** — a structural advantage: scheduling
  enters only the dynamics matrix, not the measurement.

### 2.5 Regenerative milling force

Linearized single-frequency regenerative model, plate flexibility dominant in
the transverse direction (tool compliance negligible w.r.t. plate, as in the
reference):

```
F_y(t) = α_4(t) [ w(x_T, t) − w(x_T, t−τ) ] + α_3(t) f_t,     τ = 60/(N_t Ω)
```

with the standard helical multi-tooth engagement integrals for
`α_3(t), α_4(t)` (paper Eqs. (3)–(4); mean values `ᾱ_3, ᾱ_4` for design,
full periodic coefficients for evaluation). Substituting `w(x_T,t) =
b_f(x_T)^T η(t)`:

```
η̈ + D_c η̇ + [Ω_K − ᾱ_4 b_f b_f^T] η + ᾱ_4 b_f b_f^T η(t−τ) = Φ^TΘ u + b_f ᾱ_3 f_t
```

The cutting-stiffness dyad `ᾱ_4(a_p) b_f(x_T) b_f(x_T)^T` is **rank one,
position dependent, and proportional to the axial depth of cut** — the LPV
core of the problem.

### 2.6 Material removal

Milling the top edge at radial depth `a_e` per pass recedes the free edge:
after machining up to position `x_T`, height is `h_p − a_e` for `x < x_T`.
The FEM model is rebuilt on the receded geometry at a grid of removal states;
modal sensitivity `∂(ω_i, φ_i)/∂ϱ` is tabulated. For single-pass simulations
(0.1 mm off an 80 mm plate) the drift is small and is *also* covered by the
retained parametric uncertainty — the LPV machinery matters over multi-pass
sequences, which we simulate to demonstrate C1 at full strength.

### 2.7 Scheduling vector

```
θ(t) = [ x̃_T(t), ϱ(t) ] ∈ Θ ⊂ R²
```

`x̃_T` = normalized tool position (known from NC/encoders), `ϱ` = removed
material fraction (known from the process plan). Feed 0.02 mm/tooth × 3 teeth
× 4900 rpm ≈ 4.9 mm/s ⇒ the plate is crossed in ≈ 20 s while the slowest
structural period is ≈ 3 ms: `|θ̇|` is 3–4 orders of magnitude below the
plant dynamics. The frozen-parameter (quasi-LPV, slowly-varying) design regime
is therefore rigorously justified and quantified in the paper.

---

## 3. Controller synthesis

### 3.1 Design plant with retained uncertainty

At frozen `θ`, reduced design model `G_r(θ)`:

```
ẋ = A(θ) x + B_w w̃ + B_u u
z  = C_z x + D_z u          (performance channels)
y  = C_y x + n
```

* `x = [η, η̇] ∈ R^{2n}`, `A(θ)` contains the position-dependent cutting
  stiffness dyad at nominal depth.
* Exogenous input `w̃` collects: residual regenerative force (the part not
  cancelled by the delayed feedback term, entering through `b_f(θ)`), the
  static feed-force ripple, and sensor noise `n`.
* Performance channels: `z_1 = W_f(s) · w(x_T,·)` (weighted milling-point
  displacement — the chatter variable), `z_2 = W_u(s) · u` (voltage budget:
  the amplifier limit of the reference rig, ±150 V, sets `W_u`).
* **Spillover safety:** additive uncertainty weight `W_a(s)` overbounding the
  truncated modes (`n+1 … n_full`) of the *full FEM model over all θ ∈ Θ*
  (same construction as the reference's Eq. (18)–(19), but computed from our
  FEM sweep). Robust stability against `Δ_a W_a`, ‖Δ_a‖∞ < 1 is imposed in
  the synthesis; cutting-coefficient dispersion `α_4 ∈ [0.3, 2.9] ᾱ_4` and
  modal tolerances (±10 % mass/stiffness, −20 % damping) are *verified* a
  posteriori by Monte-Carlo over the closed-loop SLD (Sec. 4).

### 3.1b Implementability lessons baked into the synthesis (added after
integration testing — these are load-bearing, not cosmetic)

The naive weighted DGKF design fails three ways on this plant; each failure
mode and its structural fix is now part of the method (and of the paper's
synthesis section):

1. **Truncation boundary.** Truncating after mode 3 leaves a 1.23× frequency
   gap to mode 4 — no realizable filter separates that; 3-mode designs
   destabilize the 12-mode plant at any depth. The design model therefore
   retains the five measured modes (gap to mode 6: ×1.55), and a **4th-order
   actuator roll-off filter at 4.8 kHz** is composed in series with every
   deployed controller.
2. **What bounds the controller in the truncated band is the *sensor-noise
   weight*, not Wu.** With a flat (tiny) Wn, nothing in the γ-stack
   penalizes observer gain at 6–12 kHz and the central controller happily
   places huge gain there. Wn is a **lead cascade** (0.2 µm rising ×25
   between 1.2–6 kHz — physically consistent with eddy-current probe noise),
   Wu likewise (×16 between 1.5–6 kHz). Weight poles stay ≤ 6 kHz because
   weight states become controller states and must respect the real-time
   Nyquist rate.
3. **γ back-off.** At the feasibility edge the game Riccati solutions blow
   up and the central controller develops parasitic high-gain fast poles
   (observed: 69 kHz pole with ~70 % FRF residue) that no target can
   implement and that static residualization cannot remove. The controller
   is re-solved at **1.3 × γ_opt**; closed-loop critical depths change by
   < 2 % for back-off 1.3–3.0.
4. **Sampled-data awareness.** The design plant includes a 1st-order Padé
   of the implementation latency (1.5 control periods at 50 kHz = 30 µs);
   deployment adds two certificates evaluated on the *actual deployed
   controller*: (i) small-gain spillover margin ‖W_a K S‖∞ < 1 over all θ,
   (ii) exact spectral radius of the sampled closed loop (ZOH at 50 kHz +
   one-period computation delay) < 1. Continuous SLD analysis uses the
   latency-composed controller so the analyzed loop matches the designed
   one.

### 3.2 Grid-based gain-scheduled H∞ synthesis

* Grid `Θ_g = {θ_1 … θ_m}` (m ≈ 9 positions × 3 removal states).
* At each `θ_k`: standard two-Riccati (DGKF) H∞ output-feedback synthesis
  with γ-bisection on the weighted plant. The central controller has
  observer/state-feedback structure
  `K(θ_k) = { A_K = A + B_uF + Z L C_y (+γ⁻²…), F(θ_k), L(θ_k) }`,
  so scheduling is implemented by interpolating the *gains* `F(θ)`, `L(θ)`
  (piecewise-cubic in θ) inside the fixed observer structure — a
  well-conditioned interpolation (no state-basis ambiguity between grid
  points, unlike naive controller-matrix interpolation).
* A common γ is enforced across the grid (design at the max-γ point sets the
  achievable level; grid refinement where γ(θ) peaks).
* **A posteriori LPV validation** (Sec. 4) replaces a common-Lyapunov LMI
  certificate; with `|θ̇|` bounds three orders below the dynamics this is the
  standard and defensible route for machining applications, and is stated as
  such (limitation § in the paper, with the LMI extension flagged as outlook).

### 3.3 Regenerative-targeted delayed feedback (DR term)

The regenerative force is `ᾱ_4 b_f^T [η(t) − η(t−τ)]`. An actuation `u_d`
producing modal force `Φ^TΘ u_d ≈ −ᾱ_4 b_f b_f^T [η − η_τ]` would cancel it
exactly, but `Φ^TΘ` and `b_f(θ)` are not collinear (actuator is a patch, the
disturbance acts at the milling point) — perfect cancellation is impossible.

Structure adopted:

```
u_d(t) = k_r(θ) [ ŵ_T(t) − ŵ_T(t−τ) ]
```

`ŵ_T = b_f(θ)^T η̂` is the *estimated milling-point displacement*
reconstructed by the H∞ observer (already running — no new hardware), and τ
is locked to the spindle (encoder-synchronized), exactly like the delayed
term of the reference but: (i) acting on the reconstructed milling-point
motion instead of the fixed sensor point — so it targets the actual
regenerative variable; (ii) with `k_r(θ)` **scheduled**, and (iii) tuned by
direct optimization of the closed-loop critical depth:

```
k_r(θ_k) = argmax_k  a_lim^{CL}(θ_k, k)   s.t.  V_rms(k) ≤ V_budget
```

where `a_lim^{CL}` is computed by semi-discretization of the complete closed
loop (plant + controller + delayed term + regenerative delay) — a scalar
line-search per grid point, done offline.

Total control: `u = u_H∞ + u_d` with a shared anti-windup voltage saturation
at ±150 V.

### 3.4 Baselines implemented for comparison

| Label | Description |
|---|---|
| OL | no control |
| DPD | delayed PD on the sensor signal (single active time-delay control, as ref. Eq. (30)) |
| R-HINF | single robust H∞ point design: nominal = path midpoint, position variation wrapped into the uncertainty budget (emulates the conservatism of the μ approach with identical weights) |
| PS-LPV | contribution C1 only (no delayed term) |
| PS-LPV-DR | full proposed strategy |

Fairness protocol: identical weights `W_f, W_u, W_n, W_a`, identical noise
realizations, identical saturation, identical evaluation model (full-order
FEM plant, periodic coefficients, loss-of-contact nonlinearity).

---

## 4. Analysis and evaluation methodology

1. **Closed-loop stability lobe diagrams** — first-order semi-discretization
   (Insperger–Stépán) of the time-periodic delayed closed loop with the
   controller states augmented; delay grid resolved to the tooth-passing
   period. Open-loop code validated against the classical 1-DOF milling
   benchmark lobes before use. SLDs computed at start / ¼ / ½ / ¾ positions
   and for fresh vs. multi-pass removal states; headline metric:
   `min_Ω a_lim(Ω)` and area-under-lobe over 2–10 krpm.
2. **Time-domain LTV simulation** — full pass(es) with moving tool, periodic
   coefficients, loss-of-contact (unilateral chip) nonlinearity, saturation,
   measurement noise; metrics: milling-point RMS/peak displacement, surface
   location error proxy, `V_rms`, `V_peak`.
3. **Robustness Monte-Carlo** — dispersion of `α_4` in `[0.3, 2.9] ᾱ_4`,
   ±10 % modal mass/stiffness, −20 % damping: closed-loop SLD percentile
   bands per strategy.
4. **Spillover check** — controllers designed on `n = 3` evaluated on
   `n_full = 12` plant; sweep of sensor/actuator placement perturbations.

Expected headline results (to be confirmed by the runs, not asserted before):
PS-LPV-DR ≥ R-HINF in stable depth at *every* position (equal at the design
point of R-HINF, strictly better away from it), at strictly lower voltage;
improvement widens with material removal (R-HINF's nominal drifts away).

---

## 5. Honest limitations (stated in the paper)

* Frozen-grid LPV design with a posteriori Floquet/simulation validation, not
  an LMI global certificate (justified by the 10³ time-scale separation;
  LMI-based synthesis flagged as follow-up).
* Single-frequency regenerative linearization for design (full periodic +
  nonlinear model used for evaluation only).
* Simulation + model-in-the-loop study; the experimental rig of the reference
  paper is the intended validation platform (hardware interface documented in
  `docs/` for the follow-up experimental campaign).

---

## 6. Repository map

```
avc/params.py       physical data (Tables 1–3 of the reference + rig limits)
avc/fem_plate.py    Mindlin FEM, material-removal geometry map
avc/piezo.py        patch coupling vector Θ
avc/modal.py        modal reduction, LPV state-space builder, b_f(θ) sweep
avc/milling.py      engagement integrals α_i(t), mean coefficients, delay
avc/synthesis.py    H∞ DGKF + γ-bisection, gain grid, interpolation, LQG/DPD/R-HINF baselines
avc/delayed_feedback.py   k_r(θ) line-search on closed-loop a_lim
avc/sld.py          semi-discretization SLD (open & closed loop)
avc/simulate.py     LTV time-domain engine (nonlinear evaluation model)
scripts/fig_*.py    one script per manuscript figure
tests/              module validation (FEM benchmarks, SLD benchmark, H∞ sanity)
paper/              elsarticle manuscript
```
