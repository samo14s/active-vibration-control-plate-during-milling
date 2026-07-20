# Foundational text — saturation-aware, position-scheduled AVC for thin-wall milling

**FINAL (v2)** — revised against 40 confirmed findings from a three-persona
Q1-reviewer adversarial pass (manufacturing, control theory, claims audit;
verdicts archived in the session record). Citation keys resolve in
`paper/refs.bib`; `[companion]` = the PS-LPV paper of this repository;
`[fu2013] [altshuller2008]` to be added to refs.bib at drafting time.

---

## 1. Abstract (200 words; X, Y, Z FILLED from this paper's certified
## campaign — provenance and definitions in §5)

> Regenerative chatter, not machine capability, caps the material
> removal rate of thin-walled aerospace parts. Active vibration control
> can raise that cap, yet industrial uptake remains marginal: mainstream
> designs assume unbounded linear actuation and a time-invariant
> workpiece, whereas shop-floor piezoelectric actuators saturate at
> productive depths of cut and cutting-point dynamics vary with tool
> position and removal. This paper develops a saturation-aware,
> position-scheduled strategy treating both nonidealities as design
> objects: a gain-scheduled H-infinity controller, synthesized over the
> NC-known tool-path and removal schedule, is certified regionally on
> the saturated periodic delayed loop by maximal saturation-free
> admissible sets on a sampled-data semi-discretization lifting —
> implementation-exact (control rate, computation delay, amplifier
> bound), matching a nonlinear simulator's clip onset within 2% — and
> swept into a permissible depth-of-cut envelope under a declared
> surface-defect tolerance. On a plate benchmark validated against
> published measurements, certification overturns the linear ranking:
> the scheduled design's 3.2x linear worst-position advantage over the
> best fixed-gain design becomes 2.7x at 1 um tolerance and inverts to
> 0.79x at 20 um, while at the hardest position it cuts vibration by
> 58% at 49% of the fixed design's voltage with fifty-fold less
> clipping — converting saturation from a hidden failure mode into a
> certified planning constraint.

**Fill-in provenance (placeholder discipline honored):** every number
above comes from THIS paper's certified campaign
(`results/satcert_campaign.json`, reproduced by
`scripts/satcert_campaign.py`; definitions and the full record in §5).
None is a companion-paper number: the companion's 4.2× is a
*continuous-model linear Floquet* ratio (here the sampled-loop linear
ratio is 3.2×, computed by this campaign's own machinery), and this
campaign's 58 %/49 % hard-condition values coincide with the
companion's protocol by construction (same seed and metrics) while
adding the saturation census that reframes them (frozen design clips
14.7 % of control updates there; the scheduled design 0.28 %).

## 2. Introduction — research-gap paragraphs

**P1 (critique of idealized assumptions).**
The active-control literature for thin-wall milling has reached
methodological maturity on an idealized plant: a linear actuator of
effectively unbounded authority acting on a workpiece frozen at one
nominal configuration. Within that frame, optimal, robust and
mu-synthesis designs [du2024robust, vandijk2012robust, zhang2019robust]
and adaptive or learning controllers [kleinwort2018adaptive,
nasiri2025chatter] demonstrate substantial — in the best cases
order-of-magnitude [jmp2026spindle] — gains in stable depth of cut, in
simulation or in laboratory cuts conducted inside the actuator's linear
range. Two simplifications underwrite the mainstream synthesis lineage.
First, actuator limits are absent from the design model: the voltage
bound of a piezoelectric patch appears, if at all, as an a-posteriori
check; the exceptions, discussed below, either avoid the constraint or
average the dynamics. Second, the position- and removal-induced variation
of the modal participation at the cutting point is ignored, wrapped into
norm-bounded uncertainty at the price of conservatism [du2024robust], or
tracked heuristically and reactively — position-varying PD gains
[wang2019timespace], adaptive re-tuning [kleinwort2018adaptive],
LPV scheduling on quasi-static setup parameters [brand2025lpv] — without
a stability certificate for the delayed periodic loop.

**P2 (why the two omissions cap industrial productivity).**
Neither simplification survives contact with production economics,
because both bind exactly where productivity is decided. Saturation binds
first at high depth of cut: recent measurements document "saturation
islands" in which chatter erupts *below* the linearly predicted stability
boundary once the actuator clips [ozsoy2025mssp], so a linearly certified
envelope overstates the safe region precisely where MRR is highest — an
unacceptable failure mode for unattended machining. Existing responses
either avoid the constraint through gain maps optimized to remain
unsaturated [jmp2026spindle], or establish restricted-initial-set
invariance for bespoke adaptive laws on Fourier-averaged dynamics
[wu2016adaptive]; the requisite saturated-delay certificate machinery —
generalized-sector conditions and maximized domains of attraction for
delayed saturated systems [fu2013], integral quadratic constraints for
time-periodic delayed feedback [altshuller2008] — exists off the shelf
and has never been applied to machining. What no published work provides
is what a process planner needs: a computable, maximized region of
initial conditions and cutting parameters over which a *given* controller
of standard architecture, behind a *given* amplifier, is certified stable
on the unaveraged time-periodic delayed dynamics. Spatial dependency
binds second: the authority of any fixed design collapses where the
dominant mode's participation at the tool position weakens, so the worst
position along the path — not the average — caps the feed-through depth
of the entire pass [companion]. The two limits compound: where authority
drops, the controller demands more voltage, and saturation arrives
sooner. Productivity in thin-wall milling is therefore governed by a
saturated, position-varying stability boundary that the current design
toolchain neither shapes nor certifies.

**P3 (explicit novelty statement).**
This paper closes that gap with a control strategy in which the actuator
bound and the tool-path variation are first-class design objects. Its
contributions are three. (i) Building on the position- and
removal-scheduled H-infinity synthesis of [companion], the actuator
bound is brought inside the scheduled design flow as a certified
constraint: every operating point handed to process planning carries an
implementation-exact certificate of saturation-free operation with a
quantified physical perturbation tolerance — in contrast to
constraint-avoiding, spindle-speed-mapped gain maps [jmp2026spindle] and
to bespoke adaptive laws with assumed initial-condition sets
[wu2016adaptive]. (ii) Stability of the saturated, time-periodic, delayed
closed loop is certified regionally, for a given fixed controller of
standard architecture, on the unaveraged periodic dynamics: a
sampled-data semi-discretization lifting (finite control rate,
computation delay, DAC-level deadzone) reduces the loop — formulated on
the variational dynamics about the verified-unsaturated forced periodic
response, with the PHASE-RESOLVED voltage headroom as the deadzone
bound — to a finite-dimensional periodic system whose maximal
saturation-free admissible set (Gilbert-Tan lineage, extended to the
periodic voltage constraint) is computed exactly, solver-free, and is
EXACT in the reported perturbation direction — validated to 0.2-2 %
against the independent nonlinear simulator's measured clip onset for
both perturbation signs; the invariant sets of [wu2016adaptive], by
contrast, are neither computed nor maximized and hold for a bespoke law
on averaged dynamics. (The generalized-sector extension certifying
operation BEYOND the clip boundary, where anti-windup becomes active,
is scoped as the saturated-regime extension: the periodic
generalized-sector SDP at lifted dimension is documented infeasible on
commodity memory, and within the saturation-free region delivered here
an anti-windup gain is provably inert.) (iii) The certificates are
swept into a certified permissible-MRR envelope — guaranteed under a
declared perturbation tolerance / linear-only (saturation islands
possible) / unstable — that turns a phenomenon so far predicted only by
uncertified describing-function analysis [ozsoy2025mssp] into an
a-priori certified and avoidable planning constraint; across the
operating grid, under identical certificate machinery, certification is
shown to OVERTURN the linear ranking of scheduled versus fixed-gain
designs as the declared tolerance grows (2.7× at 1 µm inverting to
0.79× at 20 µm) — a tolerance-dependent ranking that linear analysis
cannot see, and the first quantified demonstration that linear
stability margins misrank milling controllers under saturation.

## 3. Methodology outline (section flowchart)

```
 §2  Electromechanical modeling of the varying saturated plant
     FEM (Mindlin) plate + bonded piezo patch → modal LPV model over
     θ = (x_T, ϱ); regenerative milling force (periodic, delayed);
     input nonlinearity u = sat(v), amplifier bound ±V_max;
     validation vs published modal data, measured FRFs, and the
     measured open-loop stability pattern
        │
 §3  Saturation-aware, position-scheduled controller synthesis
     grid LPV H∞ over θ [companion] (weights incl. implementation
     latency & roll-off);
     baselines: best frozen H∞, delayed-PD, unsaturated-ideal
     [anti-windup co-design: deferred to the saturated-regime
      extension — provably inert inside the region-of-linearity
      certificates delivered below, which only certify unclipped
      operation; see §5 record]
        │
 §4  Regional certification of the saturated periodic delayed loop
     • variational formulation: certificates apply about the
       verified-unsaturated forced periodic response; PHASE-RESOLVED
       headroom V_max − |v_forced(t_k)| bounds the deadzone
     • sampled-data semi-discretization lifting at the control rate
       (rank-one delay → scalar histories; ZOH controller, one-period
       computation delay, τ/T_s ticks per period; deadzone at the
       DAC — no intra-interval hold approximation)
     • maximal saturation-free admissible set of the period map
       (Gilbert-Tan O∞, periodic voltage constraint) — closed-form,
       solver-free, EXACT in the reported perturbation direction
       → physical margin map h_max (allowable surface step, both
       signs) [the periodic generalized-sector SDP at lifted
       dimension is documented memory-infeasible; retained only for
       the saturated-regime extension]
     • certificate scope stated plainly: frozen-θ pointwise along the
       schedule; discretization resolution declared with numerical
       convergence check; robustness assessed statistically (§6)
     • companion linear certificates: spillover small-gain, exact
       sampled-loop radius, closed-loop stability lobes
        │
 §5  Certified permissible-MRR envelope and productivity use
     sweep (a_p, Ω, x_T, ϱ) → three-zone envelope for a DECLARED
     required tolerance h_req {certified | linear-only | unstable},
     with sensitivity of the envelope to h_req reported;
     stability-limited MRR maximization s.t. certified-zone
     constraint; cross-check of the optimum against force/torque/
     power and chip-load limits (§7 KPI)
        │
 §6  Numerical campaign and validation
     nonlinear time-domain evaluation (loss of contact, saturation,
     noise, moving tool); numerical basins (IC sweeps) vs certified
     sets (conservatism ratio); QUALITATIVE reproduction of the
     saturation-island phenomenon + re-simulation of the published
     SDOF configuration of [ozsoy2025mssp] under a stated
     comparability protocol (matched saturation-authority ratio);
     robustness Monte-Carlo
        │
 §7  Results → §8 Discussion & limitations → §9 Conclusion
```

## 4. Key Performance Indicators

**A. Productivity / stability capacity (the reviewer's bottom line)**
1. Worst-position depth a_lim,min = min_x a_lim(x_T) at the production
   speed — reported twice: linear-Floquet boundary AND certified depth
   (at h_req) — for open-loop / best-frozen / scheduled / scheduled+AW /
   unsaturated-ideal. *The* numbers that cap a full pass; the
   linear-vs-certified gap IS the saturation story.
2. Certified envelope area (mm·krpm) at declared h_req per strategy;
   guaranteed-zone share vs linear-only zone; band-worst a_lim over
   2–10 krpm; envelope sensitivity to h_req (one curve).
3. Stability-limited MRR gain: ΔMRR % = (a_lim,min ratio − 1) × 100 at
   fixed a_e, f_t; potential machining-time reduction for the benchmark
   pass, ceteris paribus (no per-part economics claimed).
4. Saturation-island census: linearly-stable operating points refused by
   the certificate; islands predicted vs observed in nonlinear
   simulation; qualitative correspondence with [ozsoy2025mssp] under the
   stated comparability protocol, incl. the unsaturated-ideal comparator.

**B. Vibration and surface quality**
5. Milling-point RMS and peak displacement (steady cut, worst position)
   AND chatter-band PSD attenuation (dB) at the dominant chatter
   frequency — reported separately; % vs open loop and vs best-frozen.
6. Simulated machined-surface profile along the pass (surface-generation
   instants) plus the per-revolution mean-deflection SLE proxy, with the
   proxy's limitation stated in §8.

**C. Actuation realism (the industrial-credibility block)**
7. RMS and peak voltage vs ±V_max; saturation duty cycle: steady-state
   ≈ 0 in the certified zone, transient clipping bounded and explicitly
   tolerated by the regional certificate (report the bound).
8. Voltage headroom margin at the envelope boundary; actuation energy per
   pass (J).

**D. Guarantees and robustness (the strict-reviewer block)**
9. Certificate values as numbers: spillover small-gain (< 1), exact
   sampled-loop spectral radius (< 1), certified basin radius in physical
   units (largest tolerated surface defect / force impulse), and the
   conservatism ratio certified-set / numerical-basin volume — the
   honesty metric, with the like-for-like proviso (identical certificate
   machinery across compared controllers).
10. Monte-Carlo capability: median and P(a_lim ≥ a_p,target) under
    cutting-coefficient dispersion (0.3–2.9×), ±10 % modal
    mass/stiffness (mode shapes coherently rescaled), −20 % damping —
    per strategy. State plainly: certificates are nominal-model;
    robustness is assessed statistically (or upgrade: polytopic LMIs
    over the uncertainty box, if solver capacity allows).
11. Mechanism-isolating ablations: stale (fresh-plate) vs
    removal-scheduled gains; frozen-at-x0 vs scheduled along the pass;
    with vs without AW at fixed controller; certified-envelope value vs
    a constraint-avoiding gain map [jmp2026spindle-style] tuned on the
    same benchmark.
12. Process-limit cross-check at the MRR optimum: peak cutting
    force/torque/power vs machine limits and chip-load/wear note —
    guards the "stability-limited" scoping of every MRR claim.

**E. Implementation feasibility**
13. Controller order and max pole vs real-time rate; per-update FLOPs;
    gain-interpolation latency; AW path cost — evidence the strategy runs
    on a PXIe-class target.

## 5. Certified campaign record (2026-07-20)

Machine record for every number in §1; raw values in
`results/satcert_campaign.json` (regenerate:
`python3 scripts/satcert_campaign.py`); certificate code
`avc/satcert.py` (validated by `tests/test_satcert.py`).

**Certificate machinery (as delivered).** Sampled-data period lifting
of the closed loop at the real-time control rate (50 kHz; τ/T_s = 204
ticks per tooth period at 4.9 krpm, 3-tooth cutter): continuous 12-mode
plant under per-tick ZOH, exactly discretized deployed controller
(latency=False), one control period of computation delay as an explicit
pending-voltage state, ±150 V amplifier deadzone acting on the discrete
commanded voltage — the exact place the real DAC clips. Certificate =
maximal saturation-free admissible set (Gilbert–Tan O∞) of the period
map under the periodic voltage constraint, with phase-resolved headroom
V_max − |v*(t_k)| about the verified-unsaturated forced periodic orbit;
its extent along the surface-step direction (a step of height h left in
the stored surface by the previous pass, certified for BOTH signs) is
closed-form and solver-free. Inside the certified set the loop is
provably never clipped, hence exactly linear and decaying (Floquet
radius < 1 checked). a4 sub-averaging convergence: h_max moves < 0.1 %
for n_sub 4→16.

**Validation against the independent nonlinear simulator** (100 kHz,
noise off, loss-of-contact off, PS-LPV at x_T = 50 mm; onset = smallest
injected surface step that clips):

| a_p | certificate (±) | model +h / −h | simulator +h / −h |
|---|---|---|---|
| 0.3 mm | 56.8 µm | 97.7 / 56.8 µm | 97.7 / 57.0 µm |
| 1.0 mm | 6.9 µm | 45.1 / 6.9 µm | 44.5 / 7.0 µm |

Agreement 0.02–1.9 % for both signs; the certificate equals the binding
(negative-step) onset — tight, not merely sound.

**Certified worst-position depths** (4.9 krpm, x_T ∈ {5, 25, 50, 75,
95} mm, a_p ≤ 5 mm search cap, h_req = declared tolerance; "linear" =
sampled-loop Floquet boundary from the same lifting):

| strategy | linear worst | cert. @ 1 µm | @ 5 µm | @ 10 µm | @ 20 µm | @ 50 µm |
|---|---|---|---|---|---|---|
| best frozen H∞ | 0.98 mm (x=95) | 0.98 | 0.98 | 0.92 | 0.645 | 0.35 |
| PS-LPV | 3.09 mm (x=5) | 2.66 | 0.92 | 0.72 | 0.508 | 0.29 |
| ratio (Y) | 3.16× | **2.72×** | 0.94× | 0.79× | **0.79×** | 0.83× |

The headline finding of the campaign: the scheduled design's linear
advantage survives certification only at tight tolerance (≤ ~2 µm);
from ~5 µm upward the ranking INVERTS because the scheduled
controller's higher authority amplifies both the forced tooth-passing
voltage ripple and the voltage response to surface defects, consuming
headroom precisely where the linear analysis promises depth — the
compounding mechanism of §P2, now quantified. At 20 µm both strategies
are certified only to ~0.5–0.65 mm: the certified-vs-linear gap (up to
10× for PS-LPV) IS the saturation-island exposure of KPI 4.

**Hard-condition sims** (companion protocol: x_T = 95 mm, 4.9 krpm,
250 kHz, T = 0.5 s, seed 3, sensor noise and loss-of-contact ON,
±150 V active; metrics on t ≥ 0.1 s):

| a_p = 2 mm | RMS w | peak w | RMS u | peak u | clip duty |
|---|---|---|---|---|---|
| best frozen H∞ | 15.48 µm | 67.3 µm | 84.5 V | 150 V (rail) | 14.7 % |
| PS-LPV | 6.47 µm | 20.0 µm | 41.0 V | 150 V (rail) | 0.28 % |

X = 58 % (1 − 6.47/15.48, RMS milling-point displacement);
Z = 49 % (41.0/84.5, RMS voltage — peak ratio is meaningless here since
both touch the rail); clipping ratio ≈ 52× ("fifty-fold"). Both loops
remain bounded; the frozen design operates 14.7 % clipped — the regime
its own linear certificate silently assumed away, and both points lie
far outside both 20-µm certified depths: production at this condition
is possible but UNCERTIFIED for either controller.

**Census** (a_p = 0.1–3.5 mm × 5 positions, both strategies): certified
cells (h ≥ 20 µm) end by 0.5–1.1 mm everywhere; the wide "linear-only"
zone above them is where saturation islands can live. Forced-orbit
saturation (certificate refuses regardless of tolerance) appears ONLY
for the scheduled design (x = 50 mm: 2.1–2.5 mm; x = 75 mm: ≥ 2.8 mm;
x = 95 mm: 1.7–1.8 mm) — high-authority scheduling, not weak fixed
gains, is what drives the loop into the rail, consistent with the
high-gain clipping onset mechanism of [ozsoy2025mssp].

**Saturation islands: causal demonstration** (WP4;
`scripts/satcert_islands.py`, `results/satcert_islands.json`, figure
`docs/figures/satcert_campaign.png` panel d). Protocol: settle 150
tooth periods, inject a surface step, observe 150 periods; escalate |h|
over a signed ladder; an island is claimed only when the SAME
perturbation decays with the amplifier bound lifted — the causal
control that attributes the instability to saturation and nothing else.
Sign matters through the unilateral chip h_chip = sinφ·(ft + w − w_τ):
a POSITIVE step thins the chip into air cutting (self-limiting, force
clamped at zero — no clipping observed up to +500 µm anywhere), whereas
a NEGATIVE step thickens the chip (unbounded force demand); the
negative sign is also the certificate's binding one (h₋ < h₊). Results
at 4.9 krpm (loss-of-contact ON, noise off, deterministic):

| point | ρ (linear) | decays up to | island at | with bound lifted |
|---|---|---|---|---|
| PS-LPV, x=50 mm, a_p=1.5 mm | 0.955 | ±18.8 µm | −37.6 µm → chatter growth, 99.4 % clip duty | decays |
| PS-LPV, x=50 mm, a_p=2.0 mm | 0.955 | ±20 µm | −50 µm → chatter growth, 99.5 % clip duty | decays |
| frozen, x=95 mm, a_p=0.9 mm | 0.966 | ±500 µm (max tested; ≤ 2.9 % transient clipping) | none | — |
| frozen, x=50 mm, a_p=2.0 mm | 0.698 | ±500 µm (max tested; ≤ 4.5 % transient clipping) | none | — |

Two saturation islands CONFIRMED with causal attribution — both for
the high-authority scheduled controller, none for the moderate-gain
frozen design even at 25× larger perturbations with transient clipping
tolerated. This completes the mechanism chain: the census's
forced-saturated zones, the certified-depth inversion, and the islands
all trace to the same authority-vs-headroom trade, consistent with the
high-gain clipping onset of [ozsoy2025mssp]. Within the model, the
post-onset response grows beyond the model's amplitude validity —
reported as chatter occurrence (as measured in [ozsoy2025mssp]), not
as a characterized limit cycle. The certified h_max at the island
points (1.9 µm / 0.02 µm) sits 20–2500× below the island onset:
the certificate's refusal-to-clip conservatism, quantified.

**Deviations from the WP outline (declared).** WP2's periodic
generalized-sector SDP is memory-infeasible at lifted dimension on
commodity hardware (Clarabel ≈ 3.6 GB & 97 s per solve at N = 61 with
conditioning-driven false infeasibility; larger N intractable); it is
REPLACED by the exact O∞ certificate above — solver-free and tight in
the reported direction, at the declared price that certified
trajectories never clip. Consequently WP5 (anti-windup line search) is
deferred to the saturated-regime extension: within a region-of-linearity
certificate an AW gain never activates and cannot change the certified
set. WP4's island mechanism is demonstrated in-family with causal
attribution (above); the external re-simulation of the published SDOF
configuration of [ozsoy2025mssp] under a matched-authority protocol
remains open for the manuscript stage.
