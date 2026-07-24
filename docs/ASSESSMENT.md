# Assessment of the starting package, and what is publishable

This document records what was checked, how, and what the numbers actually
came out to. Everything below was reproduced by running the code in
`baseline/`; nothing is taken from the package README on trust. The scripts
that produce each number are named so the checks can be re-run.

---

## 1. The starting package

`baseline/` is the uploaded `mindlin_simulation_package`, unmodified.

### What is genuinely good

The structural core is sound and worth keeping:

- The 8-node Serendipity Reissner–Mindlin element is correctly formulated —
  shape functions, Jacobian, `Bf`/`Bs` operators, the `h³/12` bending and
  `κGh` shear scalings, and the rotary inertia `diag(1, h²/12, h²/12)`.
- The CCCC benchmark reproduces Leissa to **0.02 %**, and the cantilever
  fundamental comes out at **519.4 Hz** against the ~521 Hz reference, with
  mesh convergence to 0.001 % between the last two meshes
  (`baseline/tests/verify_mindlin.py`, re-run and passing).
- The helical milling-force integrals are correct in closed form, the
  regenerative sign convention is right (`K_eff = Kp + a4·Dp Dpᵀ` with
  `a4 < 0` through engagement, so the process correctly softens the plate),
  and the fly-over bookkeeping is right — a 12.2 % duty cycle, verified.
- The Newmark update itself is textbook average-acceleration and correct.

That core is the reason this work is worth continuing rather than restarting.

### What does not survive checking

| README claim | What the code actually does |
|---|---|
| "+19.31 % average RMS reduction vs LQG" | **+4.32 %** reproducible. And retuning the baseline to `w_q = 1e15` — a value excluded only by an arbitrary `‖K‖ < 1e8` cap, needing 18.5 V of the 150 V available — gives 0.4938 µm against DARC-MPC's 0.5273 µm. The advantage **reverses** in 3 of the 4 scenarios. |
| "Stability domain 21.7× open loop, +41 % vs LQG" | `zeta_DARC_eff = zeta_LQG * 1.30` — a hard-coded 30 % damping bonus. The DARC controller object never enters the stability computation. **And the claim is unattainable in principle**: a feedforward signal is input-additive, so it cannot appear in the monodromy matrix and cannot move the stability boundary. The correct value of that improvement is exactly zero — demonstrated in `tests/verify_feedforward_cannot_move_lobes.py`, where ρ is bit-identical up to 1000 V across five harmonics. |
| "Deep Adaptive Robust Control with MPC" | No prediction horizon, no finite-horizon cost, no QP. `lambda_robust` is computed and never read. `OnlineRLSAdapter` contains no recursive least squares and never fires once (`omega_hat/omega_nom = 1.000000` after a full run) — its trigger compares a ~3.6e-8 m displacement against a 1e-6 threshold. The network is one hidden layer of 16 units. |
| "Phase-aware neural feedforward" | 6 of the 8 inputs receive **exactly zero gradient**, because training states are appended as `np.zeros`. The learned function is DC 1.03 V + 2.02 V at the tooth-passing frequency, everything above the 2nd harmonic below 0.05 V — a one-harmonic Fourier series, three effective parameters, from 51 weights and 450 SGD epochs. |
| "Iterative learning control" | The update rule has a hard-coded `K_correction = 1e6`, giving a loop gain of 0.011 at the fundamental, and is **divergent at the 3rd and 4th tooth harmonics** (contraction factors 1.017 and 1.019). It is bounded only by the ±10 V clip. |
| "Lyapunov safety filter" | Certifies the nominal LTI loop. It omits the periodic parametric stiffness, the regenerative delay and the cutting force — whose combined contribution to `V̇` is 63 % of the control term's. Chatter *is* a Floquet/DDE instability, so the certificate is blind to the phenomenon it is invoked against. |

Two further problems affect every reported number:

- **The study covers 2.45 mm of a 100 mm pass.** At `v_feed = 4.9 mm/s` and
  `T_END = 0.5 s` the tool travels 2.5 % of the wall — and that window sits
  exactly where the actuator happens to be 95 % aligned with the
  disturbance (see §3).
- **The linear cutting law is applied 1000× outside its validity.** The
  maximum uncut chip is 3.98 µm at 1 % radial immersion, so the tooth
  separates from the workpiece beyond ~4 µm of vibration; the divergence
  guard is set at 5 mm. Every "chatter" trace lives far outside the regime
  where the model means anything.

**Verdict.** The DARC-MPC-versus-LQG story is not publishable, and should not
be submitted anywhere. The FEM core, the force model and the solver are worth
building on.

Reproduce with `tests/audit_baseline_claims.py`.

---

## 2. A defect in the ported element, and the fix

The Q8 consistent mass integrand `Nᵀ Ie N` is **quartic** in each parametric
direction, but the source element integrates it with the same 2×2 Gauss rule
it uses for the deliberately-reduced shear stiffness. Consequences, all
verified:

- element mass matrix has **rank 12 of 24**;
- the assembled constrained `M_free` carries exactly `3·N2` zero-mass
  directions (15 on a 6×5 mesh, 72 on the production 30×24 mesh);
- `scipy.linalg.cho_factor(M_free)` **fails**, and `eigh(K, M)` cannot factor
  it.

Total mass is nonetheless exact, which is why shift-invert Lanczos still
returns almost the right frequencies and the defect went unnoticed.

`src/evolving_plate.py` therefore defaults to **selective reduced
integration** — bending and mass at 3×3, shear at 2×2 — which restores full
rank, removes the spurious hourglass mechanism, and moves the benchmark
frequencies by **0.001 %**. So no published frequency depends on the change,
and the matrices become usable. `quadrature="uniform2"` reproduces the source
element bit-for-bit and is kept so the port can still be checked against its
origin.

Reproduce with `tests/verify_evolving.py` (sections 1, 1b, 1c).

---

## 3. The three findings the new work is built on

### 3.1 A single actuator cannot reach the disturbance everywhere

The milling disturbance enters the modal equations along `Dp(x_tool)`, the
mode shapes at the contact point, which rotates in modal space as the tool
feeds. The actuator pushes along `H`. The reachable fraction is

    γ(s) = ‖Qᵀ d̂(s)‖ ,   Q an orthonormal basis of range(H)

For the baseline patch over the full 100 mm pass:

| tool x [mm] | 0 | 2.45 | 10 | 20 | 30 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|---|---|
| γ | 0.944 | 0.961 | 0.996 | 0.914 | 0.633 | 0.180 | 0.006 | 0.037 |

Mean over the 2.45 mm actually simulated: **0.953**. Mean over the full pass:
**0.375**. Minimum: **0.000 at x = 69 mm** — a tool position at which that
actuator cannot oppose the disturbance at all, at any voltage.

This is not a placement mistake. **Every** single-window placement tested has
a worst case of ~0.000, because with 3 retained modes a single actuator spans
a line in ℝ³ while `Dp(x)` sweeps a curve that generically crosses the
orthogonal plane. Adding actuators fixes it:

| layout | min γ | mean γ |
|---|---|---|
| baseline 1 patch | 0.001 | 0.399 |
| best 1 patch | 0.015 | 0.523 |
| best 2 patches | 0.408 | 0.786 |
| best 3 patches | **1.000** | **1.000** |

Three independent actuators span ℝ³, so with three retained modes every
disturbance direction becomes reachable — a crisp design rule: **path-wide
input-matching requires at least as many independent actuators as retained
modes.**

Reproduce with `tests/analyze_actuator_alignment.py` and
`experiments/run_path_study.py`.

### 3.2 The workpiece changes by a factor of ~3 while being machined

Over a realistic programme (8 mm blank → 3 mm wall, 3 radial layers × 10
axial bands of 8 mm):

| mode | blank | final | span |
|---|---|---|---|
| 1 | 1069.5 Hz | 390.1 Hz | 2.90× |
| 2 | 2083.7 Hz | 793.1 Hz | 2.64× |
| 3 | 5110.7 Hz | 2024.3 Hz | 2.52× |

and the evolution is **not monotone** — f₁ first *rises* to 1130 Hz, because
early passes remove inertia near the free edge, before falling as stiffness
loss takes over. A model that only ever lowers frequencies is wrong.

Reproduce with `tests/verify_evolving.py` (section 4) and
`experiments/run_path_study.py`.

### 3.3 A fixed-gain LQG becomes worse than no control

This is the strongest result, and it was checked three independent ways
before being believed:

| layers removed | f₁ | max Re(A−BK) | max Re(**full LQG**) | ρ certified | time domain |
|---|---|---|---|---|---|
| 0 | 1030 Hz | −186.3 | −186.3 | 0.711 | decays |
| 1 | 819 Hz | −163.7 | −82.6 | 0.598 | decays |
| 2 | 605 Hz | −136.1 | **+271.7** | **2.365** | **grows** |
| 3 | 390 Hz | −103.0 | **+257.6** | **3.301** | **grows** |

The mechanism is specific and worth stating precisely: **the regulator stays
stable throughout** — `A − BK` has negative real parts at every stage — and it
is the **observer** that destabilises the loop. State feedback tolerates the
drift; observer-based output feedback does not. This is the classical absence
of a guaranteed LQG margin, triggered here by the workpiece thinning under a
stale model.

Since essentially all piezo-based workpiece AVC in the literature uses
observer-based output feedback designed once on the nominal workpiece, this
matters beyond this particular plate.

Reproduce with `tests/verify_fixed_gain_instability.py`.

---

## 4. Certified closed-loop stability lobes

Closed-loop lobe diagrams in this field are commonly produced by reading the
closed-loop damping ratios off the poles and substituting them into the
*open-loop* single-mode lobe formula. That substitution is not the closed
loop: a sampled dynamic output-feedback controller changes the dimension and
the structure of the monodromy operator, and the observer states are
themselves driven by the delayed regenerative term through the measurement.

`src/closed_loop_sld.py` builds the monodromy matrix of the actual sampled
loop — plant, zero-order hold, discrete observer and feedback — so the lobe
is certified rather than inferred. Two corrections come with it:

- **Multi-mode coupling.** `Dp Dpᵀ` is rank one and couples every retained
  mode through the single contact point. Taking one Floquet radius per mode
  and keeping the maximum, as the baseline does, is not the spectral radius
  of the coupled system: the two differ by up to **20.8 %** here.
- **Multi-rate timing.** The controller has a fixed sample period while the
  semi-discretisation step must divide the tooth period, which changes with
  spindle speed. Mesh convergence is taken by refining the plant *without*
  changing the controller rate; refining both together silently changes the
  system being analysed.

Validation, at 4900 RPM: the certified LQG lobe gives `a_p,crit = 2.01 mm`,
and direct time-domain simulation places the stability transition between
1.81 and 2.41 mm.

**How wrong is the substitution? It depends entirely on where you evaluate
it, and averaging hides the answer.**
(`tests/verify_substitution_error_along_path.py`, with `Ts = tau/82` exactly)

| evaluated with | substitution error |
|---|---|
| the path-averaged `Dp` the baseline SLD scripts use | **+1.1 %** |
| the local `Dp(x)` at each tool position | **−3.6 % to +45.4 %**, mean +10.2 %, optimistic at 7 of 10 stations |

So a single path-averaged number makes the shortcut look almost exact, while
the error actually incurred along the pass reaches **+45 % in the optimistic
direction** — it over-promises stability by nearly half.

The location of the worst error is the interesting part, and it is the
opposite of the intuitive guess. The +45.4 % occurs at x = 0, where the
matched fraction is γ = 0.944 — that is, where the actuator is *best* aligned
with the disturbance and the controller is doing the most work. Where the
controller has strong authority its own dynamics dominate the loop, and that
is precisely what collapsing it to a scalar damping ratio throws away. Where
γ → 0 the controller barely acts, the loop is nearly open, and the
substitution is nearly right for the trivial reason that there is nothing to
misrepresent.

*(The earlier figure of ~6 % quoted from `verify_closed_loop_sld.py` came
from a slightly different configuration — `Ts = 5e-5` s, which does not
divide `tau` exactly, and a grid-selected weight. The honest statement is the
range above, not any single number.)*

Reproduce with `tests/verify_closed_loop_sld.py`;
`tests/verify_monodromy_equivalence.py` checks the fast structured
implementation against the explicit dense product (agreement to 1e-19, 24×
faster).

---

## 4b. Minimising vibration and maximising stability are different objectives

At matched control effort on the nominal plate
(`experiments/run_benchmark.py`, 60 V budget, every law tuned against the
same reported objective):

| law | gain | u peak [V] | y_rms [µm] | certified a_p,crit [mm] |
|---|---|---|---|---|
| open loop | — | 0.00 | 28.487 | 0.071 |
| velocity feedback | 4.64e4 | **8.92** | 0.166 | **2.401** |
| static modal position fb | — | 0.00 | 28.488 | 0.071 |
| LQG | 1.00e18 | 28.53 | **0.134** | 1.983 |

LQG wins on RMS by 19 % and **loses on stability by 17 %**, using 3.2× the
voltage. Both selected gains are interior optima, not grid boundaries — the
search reports a flag when they are not, and static modal position feedback
is flagged as achieving nothing at all, which is correct: a stiffness shift
does not add damping, and chatter is a damping problem.

That is not a paradox, it is the point. The quadratic cost weights the
response to an assumed disturbance spectrum; chatter stability is governed by
how far the negative real part of the plant FRF is pushed back at the chatter
frequency. Velocity feedback attacks the second quantity directly. A
controller tuned for one need not win the other.

The consequence for the field is concrete: **an RMS reduction is not evidence
of chatter suppression**, and papers that report only "x % vibration
reduction" — including the package this work started from — have not
demonstrated the thing they claim. The stability boundary has to be computed,
and computed from the real loop.

It also connects to §3.3. Velocity feedback here is static in the state; the
liability in §3.3 was the *observer*. The two results point the same way: the
observer is the fragile element, and it is not buying stability margin in
exchange.

*(Reported from a single nominal operating point; the ordering should be
confirmed across the RPM range before it is stated as general.)*

---

## 5. What the paper should be

**Working title.** *Path-wide controllability and certified closed-loop
chatter stability for actively damped thin-walled milling.*

One argument, three legs, all measured on the same machining programme:

1. **The plant moves** (§3.2) — by a factor of ~3, non-monotonically.
2. **The actuation moves too** (§3.1) — reachability collapses to zero at
   some tool positions, and no single actuator avoids it.
3. **Ignoring either is not conservative** (§3.3) — a fixed-gain
   observer-based controller does not merely under-perform, it destabilises
   a process that was stable without it. And the usual way of drawing
   closed-loop lobes (§4) over-promises stability.

The constructive half is then: a worst-case-over-path actuator placement
criterion, a path-scheduled controller, and closed-loop stability certified
at every station rather than at one nominal point.

### What must still be done before submission

Ordered by how much a reviewer will care.

1. **Chip-thickness nonlinearity.** Enforce `h_j > 0` per tooth and axial
   slice so the tool can leave the material. *(not yet implemented)*

   **Scope note, and it matters for how the paper is defended.** This defect
   invalidates *large-amplitude time-domain* results — the baseline's
   "unstable" traces run to 5 mm, against a 3.98 µm maximum uncut chip, so
   they are meaningless. It does **not** invalidate the stability results in
   §3.3 and §4. A stability lobe diagram is a *linearised* statement about
   growth or decay in the neighbourhood of the nominal periodic motion, and
   that neighbourhood is exactly where the linear cutting law is valid: the
   tooth has not yet left the material. The certified Floquet radius, the
   critical depths, and the fixed-gain instability are therefore all inside
   the model's validity domain. The nonlinearity governs what happens *after*
   the boundary is crossed — limit-cycle amplitude, surface finish — which
   the paper should simply not claim without it.
2. **Fair benchmark at matched control effort.** LQG, DVF, PPF and a proper
   repetitive controller, each tuned against the reported metric under the
   same voltage budget. `src/baseline_controllers.py` provides the laws and a
   common tuning routine; the comparison harness is not yet written.
3. **Time-step and delay discretisation.** Choose `dt = τ/N` exactly (the
   baseline runs at an effective 4878 RPM while its own SLD uses 4900), and
   take `dt` small enough that the highest retained mode is not detuned —
   at `dt = 50 µs` mode 3 is detuned 5.65 %, about 10× its half-power
   bandwidth.
4. **Piezo patch as structure.** Implemented in `src/evolving_plate.py`
   (`add_bonded_layer`, plus the moment arm taken to the composite neutral
   axis rather than the bare mid-plane), but not yet used in the headline
   runs.
5. **Sensor placement.** The baseline probe sits on the machined top edge,
   coincident with the tool at the end of the pass — physically impossible.
   Moved to the un-machined lower half here; a sensitivity sweep would
   strengthen the paper.

### Validation without a laboratory — and a correction

**Matching a published FRF is *not* validation.** This needs stating plainly
because it was the original plan for this work. Reproducing someone else's
published FRF is **code verification / benchmarking**: it shows your solver
reproduces a solver-plus-specimen you did not build, and says nothing about
your plate, your clamp, your piezo bond or your force model. In ASME V&V 10
terms it is "comparison to a benchmark solution", which V&V 10 explicitly
separates from validation against physical experiment. Calling it validation
in a manuscript invites the reviewer to make the distinction for you.

What the venues actually say:

- **MSSP.** Aims & Scope seeks papers with "both theoretical and experimental
  aspects, or theoretical material of high relevance to practical
  applications". The clearest written statement is the *Guidelines for
  Machine Learning Papers in MSSP* (5 Dec 2024): benchmark on more than one
  dataset, "at least one dataset should usually be experimental", with an
  exception only "if simulated data is designed appropriately", and it notes
  such papers are often "rejected without review". *Caveat: that document is
  formally scoped to ML submissions, and the wording above was read out by a
  search engine rather than from the PDF — verify it before quoting.*
- **IEEE TCST.** Scope explicitly covers "analysis and design through
  simulation" and "novel modeling techniques", and there is **no written rule
  against simulation-only papers**. The barrier is editorial practice, not
  policy — which means it cannot be quoted, only anticipated.

**So the strategy is unchanged but its labelling must be exact:**

- **Verification (done, and it is genuinely strong).** Leissa CCCC to 0.02 %;
  mesh convergence; analytic `f ∝ h` scaling; first-order eigenvalue
  perturbation with demonstrated O(Δh) convergence; the certified monodromy
  cross-checked against independent time-domain simulation; the structured
  monodromy checked against the dense product to 1e-19.
- **Benchmarking against published results (to do).** Digitise the open-loop
  FRFs and force coefficients of two or three groups and reproduce their
  reported closed-loop improvements. Call it benchmarking, not validation.
  Its value is real but different: no paper in this field publishes
  machine-readable FRF, modal or SLD data, so releasing the digitised
  datasets plus a validated open model is itself a contribution — and one of
  the few genuinely available to a group without a rig.
- **Lead with the method.** A simulation-only paper survives in these venues
  when the deliverable is a *method and a certificate*, not a percentage.
  That is exactly what the certified closed-loop lobe and the placement
  criterion are. Do not lead with "x % improvement".

**Three parameter problems flagged by the same survey:**

1. `K_T = 925 MPa` is quoted as a constant, but specific cutting pressure is
   a power law in chip thickness, and at 2–4 µm this model is deep in the
   size-effect regime. The number is meaningless without a stated `h`. Fix
   the citation and state the chip thickness it corresponds to.
2. No published FRF exists at this plate's scale — verified open-access
   thin-wall cases sit at 2680 Hz or 20–325 Hz, and this plate at ~520 Hz
   falls in the gap. Expect to benchmark against a *different* geometry and
   say so.
3. The Leissa CFFF parameters are only partly verified here (λ₁ = 3.492 and
   λ₂ = 8.525 confirmed for ν = 0.3; modes 3–6 not independently confirmed).
   Check them in the original before tabulating.

Closest prior art that may already contain exactly the with/without-control
lobes needed: **Zhang & Sims (2005), *Smart Mater. Struct.* 14(6):N65–N70**
(PPF, 7× limiting depth). It could not be opened here. Read it first.
