# Positioning against the literature

From a survey across WebSearch, Consensus and Scholar Gateway. Read this
before writing the introduction: roughly half of what feels novel in this work
was established between 2005 and 2020, and claiming it would invite an easy
rejection.

> **Citation caveat.** Several entries below were surfaced by automated
> search and the key ASME/Elsevier/Springer sources returned 403 to automated
> fetch. Author lists for MSSP 159 (2021), JMP 84:1042–1053 (2022),
> *Machines* 13(6):524 (2025) and the SIAM slow-variation paper are
> **unverified**. Obtain every one of these through institutional access and
> check it before citing. Do not copy a citation from this file into a
> manuscript unread.

---

## What is already done — do NOT claim these

| Claim you might be tempted to make | Who did it |
|---|---|
| "Material removal changes the workpiece dynamics during machining" | Thevenot et al. 2006 (two companion papers); Budak, Tunç, Alan, Özgüven, CIRP Annals 2012 |
| "In-process workpiece dynamics can be predicted along the path" | Tuysuz & Altintas, ASME JMSE 2017 (frequency-domain reduced-order substructuring) and 2018 (time-domain, perturbation); Dang et al. IJMS 2019; Yang et al. MSSP 2019; Yang et al. IJMSD 2022 (GPR + POD surrogate) |
| "Stability lobes depend on tool position — 3D SLD (RPM × a_p × position)" | Bravo et al. 2005; Thevenot et al. 2006; Seguy et al. 2008 ("toolpath dependent" — the exact phrase); Tang & Liu 2009; Campa et al. 2011; Wang et al. IJMS 2019 |
| "The modes of a highly flexible workpiece drift enough to matter" | Stépán, Kiss, Ghalamchi, Sopanen, Bachrathy, CIRP Annals 2017; Kiss, Bachrathy, Stépán, ASME 2020 |
| "Frozen-time stability analysis is not exact for a slowly drifting plant" | Dombóvári, Munoa, Kuske, Stépán, Procedia CIRP 77:110–113, 2018 — *proved*, with a rigorous escape estimate in the companion SIAM paper |
| "Robust control can certify chatter-free operation over a region" | van Dijk, van de Wouw, Nijmeijer, IJRNC 2015 — over a region of (a_p, RPM), at **fixed** structural dynamics |

So §3.2 of `ASSESSMENT.md` (the workpiece moves by ~3×) is **motivation, not
contribution**. Present it as a quantified restatement of a known phenomenon
for this specific geometry, and cite Thevenot/Budak/Kiss when doing so.

---

## What is genuinely unclaimed — this is the paper

The survey's own words on the two central questions were *"Direct answer: no"*
both times.

### 1. A closed-loop stability certificate over a tool path

> "The two certificate types in the literature are disjoint. (a) Open-loop
> path-wide: 3D SLDs cover the whole path but contain no controller.
> (b) Closed-loop frozen: van Dijk et al. certifies over a region of
> (a_p, RPM) but at fixed structural dynamics. **The obvious unclaimed
> contribution is the intersection: a closed-loop 3D SLD**, certified
> chatter-free over RPM × a_p × tool-path coordinate with a controller in the
> loop."

That is exactly what `src/closed_loop_sld.py` + `src/machining_path.py`
produce. The distinguishing technical point is that the **controller
state-space is inside the monodromy matrix** — every path-wide chart in the
literature is open-loop, and every closed-loop chart substitutes an
equivalent damping ratio into an open-loop formula.

### 2. No path-parameterised controller exists

> "Every active chatter controller found for thin walls is synthesised at a
> single frozen operating point."

The strongest near-miss states its own method in one sentence: the varying
dynamics are *"overcome by designing controller with the parameters on the
maximum vibration position to stabilize the whole process"* — a worst-case
frozen design (JMP 84:1042–1053, 2022). The only genuinely position-dependent
controller found (Wang, Song, Liu, IJAMT 105:2843–2856, 2019) schedules a PD
gain heuristically on the first mode shape, with no material-removal-updated
model, no LPV synthesis and **no stability guarantee**.

### 3. Actuator placement for path-wide authority

> "As the mode shape migrates along the path, collocation, controllability
> and observability degrade — and no work jointly optimises actuator/sensor
> placement together with a scheduled controller over the whole path, nor
> certifies that authority is retained everywhere. This is a concrete,
> experimentally testable gap."

This is §3.1 of `ASSESSMENT.md`. Wang et al. IJMS 2019 documented that the
in-process mode *shapes* change, and the 2022 active-modal-control paper
optimises patch position at **one** configuration. Nobody has posed the
worst-case-over-path placement problem, and the finding that a single patch
has γ → 0 blind spots that **no** placement removes appears to be new.

### 4. Two further openings this work is positioned to take

- **Quantify the frozen-time error for material removal.** Dombóvári et al.
  proved frozen-time is wrong, but their slow parameter is the drifting
  *machine structure* in heavy-duty milling. Nobody has computed the error
  when the slow parameter is **wall thinning**, where the drift per pass is
  far larger. Self-contained and publishable, and it is the prerequisite for
  arguing a path-wide certificate is *needed*.
- **Use the scheduling-parameter rate bound.** LPV stability needs a bound on
  |dρ/dt|, and in milling that bound is free — it follows from the feed rate
  and the material removal rate, both known from the CL file *before* the
  cut. No milling paper found exploits this.

---

## Consequences for how the paper is written

1. **Lead with the certificate, not the phenomenon.** The title should be
   about closed-loop path-wide certification and actuator authority, not
   about material removal.
2. **Cite the IPW-prediction line as an enabler, not a competitor.** Budak
   2012 / Tuysuz & Altintas 2017–18 / Yang 2022 all output exactly the
   position-parameterised model family a scheduled controller needs, and all
   terminate in an open-loop chart. The natural sentence is: *these models
   exist and nobody has closed a loop around them.*
3. **Do not claim to be first to show the modes move.** Quantify it for this
   geometry and move on within a paragraph.
4. **The `git`-level honesty carries over.** The survey notes the
   IPW-prediction literature never propagates its reduction error into a
   robust synthesis, so any path-wide certificate is "only as good as an
   unquantified model". This work has the same exposure — a 3-mode modal
   truncation — and should say so rather than wait to be asked.

---

## The decisive one: feedforward cannot move the lobes

A third survey, of control synthesis where the delay and periodicity are part
of the *design* problem, settles the question the starting package's headline
depends on.

> "**Feedforward cannot move Floquet multipliers.** A signal added at the
> plant input, however cleverly learned, does not change the closed-loop
> monodromy operator of the milling LTP-DDE. It cancels exogenous periodic
> forcing — forced vibration, surface location error, tooth-pass harmonics —
> and therefore improves surface finish and forced-vibration amplitude. It
> does NOT enlarge the stability lobe diagram. **Any claim of lobe
> enlargement from feedforward alone is a fatal, and correct, reviewer
> objection.**"

`tests/verify_feedforward_cannot_move_lobes.py` demonstrates this in this
codebase, two ways: the monodromy matrix is bit-identical with and without a
feedforward (ρ = 0.666337362620496 unchanged at 2 V, 20 V, 150 V, and 1000 V
across five harmonics), and in time-domain simulation past the certified
boundary the loop diverges with the feedforward active exactly as without it.

So the starting package's **"+41 % stability domain"** is not merely
fabricated by its hard-coded ×1.30 multiplier. **The correct value of that
improvement is exactly zero**, and no amount of better learning would change
it.

### And the architecture itself is prior art

> "A feedforward signal indexed by the phase of a periodic process and
> updated pass-to-pass from measured error is, mathematically, one of three
> equivalent things: repetitive control; adaptive feedforward cancellation /
> higher-harmonic control; or FxLMS with a spindle-encoder-locked reference…
> **ILC run continuously over the tooth period, with forgetting, IS
> repetitive control** — this is textbook. Replacing the linear basis with a
> neural network is a function-approximator swap inside a known
> architecture, not a new control principle."

And specifically in machining: Tsao & Tomizuka (1994) for spindle-synchronised
repetitive control; Rashid & Nicolescu (2006, IJMTM) for adaptive feedforward
on milling workholding; and decisively **Chen, Zhang, Zhang & Ding (2014,
ASME JDSMC 136(2):021007)**, which expands the regenerative cutting force in a
**Fourier series and adapts the coefficients online** — adaptive phase-indexed
harmonic feedforward for milling chatter, twelve years ago.

### One more reason the internal-model route is limited

Regenerative chatter at a secondary-Hopf (Neimark–Sacker) bifurcation is
**quasi-periodic**: the chatter frequency is incommensurate with tooth
passing, so an internal model placed at tooth-pass harmonics has *zero gain*
at the chatter frequency and by the internal model principle cannot reject it.
This is why the spindle-speed-variation literature reports SSV works on
period-doubling but not on quasi-periodic chatter.

The exception is **flip (period-doubling) chatter**, which is 2T-periodic and
therefore genuinely within reach of a period-doubled internal model. The
survey found no publication doing that — *"a 2T phase-indexed learned internal
model aimed specifically at flip lobes is, as far as this search reaches,
unclaimed."*

### Closest prior art to our own monodromy work — cite and differentiate

**Nazari, Butcher & Bobrenkov (2014)** do periodic-gain delayed feedback with
Chebyshev collocation of the monodromy operator and spectral-radius
minimisation. That is the nearest thing to synthesis on the monodromy, and it
must be cited. Differentiators: they optimise (not schedule) gains, at fixed
structural dynamics, with no workpiece, no material removal, no tool path and
no hardware.

Also relevant: **Borgioli et al. (2020)** define a pseudospectral radius
directly on the LTP-DDE monodromy operator but use it only for *analysis* —
minimising it over controller parameters is flagged as unclaimed.

### A check we should run and almost nobody does

> "Insperger & Stépán establish a critical depth of cut above which **no
> digital controller at a given sampling period can stabilise** the process.
> Almost no applied active-chatter paper checks its claimed enlargement
> against this bound."

Our multi-rate formulation already carries the sampling period explicitly, so
this check is cheap here and would strengthen every reported a_p,crit.

### It also explains our own benchmark result

> "Active damping raises the lobe floor but does not move the lobes. The
> theoretical link (b_lim ∝ 1/|Re G|_min, so roughly proportional to added
> modal damping) is well understood via Ganguli / Deraemaeker / Preumont."

That is exactly the mechanism behind §4b of `ASSESSMENT.md`: velocity
feedback beats LQG on stability because it attacks |Re G|_min directly, while
the quadratic cost optimises something else.

---

## A second survey, of the active-control side specifically

A separate sweep of piezo-based AVC for thin-walled workpieces (2015–2026)
independently reaches the same conclusions and adds several that map directly
onto modules already built here.

**Field structure.** Three actuation loci, pursued by disjoint groups:
workpiece-side bonded patches (Zhang & Sims lineage; Wang/Song/Liu at
Shandong; Du & Long at SJTU/NPU), fixture-side piezo stacks (Rashid &
Nicolescu; Sallese/Scippa/Campatelli; INTEFIX), and tool/spindle-side
(Cao/Chen at XJTU; Munoa/Beudaert at Ideko). Much of what is indexed as
"milling chatter AVC" is tool-side and does not treat position-dependent
workpiece dynamics at all.

**Gaps that match what is in `src/`:**

| Survey finding | Module |
|---|---|
| "The regenerative delay is routinely treated as a **disturbance**… a rigorous delay-differential closed-loop analysis (semi-discretisation with the controller states appended)… benchmarked against the delay-as-disturbance approach" is open | `closed_loop_sld.py` |
| "**Closed-loop SLD does not evolve with material removal.** Every SLD in this corpus is computed for one workpiece state. A three-dimensional closed-loop stability map over speed × depth × tool position, with the controller in the loop, has not been published. **Simulation is the only practical way to produce it.**" | `closed_loop_sld.py` + `machining_path.py` |
| "**Everything is SISO.** Essentially all thin-wall workpiece AVC uses one patch and one accelerometer. MIMO modal control with several patches… plus the associated placement optimisation, has not been done." | `actuator_placement.py` |
| "**Placement is optimised without the process in the loop.** Nobody optimises placement against the closed-loop stability limit along a real toolpath — i.e. using the achievable depth of cut, not the modal Gramian, as the objective." | `actuator_placement.py` — currently maximises reachability; **extending the objective to certified a_p,crit is the obvious next step** |
| "**No common-plant controller benchmark for a thin wall.** Nobody has run PPF / DVF / LQG / H∞ / µ-synthesis / SMC / MPC / RL on one thin-plate-plus-bonded-piezo plant." | `run_benchmark.py` |
| "**Sampling, discretisation and filter delay are ignored.** Controllers are designed in continuous time and implemented on real-time hardware with no published analysis of how sample rate and group delay eat the phase margin." | the multi-rate treatment in `closed_loop_sld.py` |
| "**Spillover is never quantified**… no paper quantifies spillover-driven destabilisation in stability-lobe terms." | not yet done — a natural addition, and §3.3 is adjacent to it |

Two further notes that bear on our own results:

- **PPF works, and works well** — Zhang & Sims (2005) report a 7× limiting
  depth of cut, and it is the origin of the workpiece-side thread. Our
  benchmark's "static modal position feedback achieved nothing" is therefore
  *not* a verdict on PPF: it is the stiffness-shift limit without the
  second-order filter. Implementing and certifying real PPF is required
  before saying anything about it.
- µ-synthesis has **never** been applied to a thin-walled workpiece, despite
  it being the textbook structured-uncertainty problem (frequencies drift
  monotonically with removal). One clear route if a robust-control angle is
  wanted.

---

## Validation without a laboratory

The second survey names the strategy explicitly, and it is the single most
useful item here for a simulation-only group:

> "**Validation-by-reproduction has never been attempted.** Several
> independent experimental results are publicly digitisable… A simulation-only
> study that digitises the published open-loop FRFs and cutting-force
> coefficients and then reproduces several groups' results [would be a
> contribution]."

and

> "**No open data.** Not one paper in this corpus publishes machine-readable
> FRF, modal or SLD data; everything must be digitised from figures.
> Releasing a validated open benchmark model plus digitised reference
> datasets would itself be a contribution, and is **one of the few
> contributions in this field genuinely available to a simulation-only
> group.**"

Reported closed-loop improvements that are digitisable and reproducible:

| source | reported result |
|---|---|
| Du & Long (2022) | limiting depth 0.2 → 1 mm |
| Du, Liu & Long (2023) | 1.5 → 6 mm |
| Du, Liu, Dai & Long (2024) | 0.1 → 0.8 mm |
| Ozsoy, Sims & Ozturk (2022) | 2.6× |
| Aggogeri et al. (2021) | 96 % attenuation at 1130 Hz |
| Zhang & Sims (2005) | 7× limiting depth, PPF |

**Recommended plan:** digitise the open-loop FRF and force coefficients from
two or three of these, reproduce their reported closed-loop improvement with
this code, and release the digitised datasets alongside. That converts
"simulation-only" from a weakness into a stated contribution, and it is
exactly what these venues will accept in place of an experiment.

## Other validation sources worth chasing

Mostly paywalled; flagged as open-access with digitisable data:

- Yang et al., *Int. J. Mech. Syst. Dyn.* 2(1):117–130 (2022) — GPR+POD,
  predicted vs experimental SLD
- *Machines* 13(6):524 (2025) — 3D SLD for Ti thin walls, semi-discretization
  with process damping
- Wang, Wu, Wan, Dulikravich, *Math. Probl. Eng.* (2015) — explicit LPV
  state-space matrices, usable directly as a control benchmark
- Reviews: *Int. J. Extreme Manufacturing* 7(6) (2025); *Machines* 11(3):359
  (2023)
