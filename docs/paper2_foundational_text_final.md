# Foundational text — saturation-aware, position-scheduled AVC for thin-wall milling

**FINAL (v2)** — revised against 40 confirmed findings from a three-persona
Q1-reviewer adversarial pass (manufacturing, control theory, claims audit;
verdicts archived in the session record). Citation keys resolve in
`paper/refs.bib`; `[companion]` = the PS-LPV paper of this repository;
`[fu2013] [altshuller2008]` to be added to refs.bib at drafting time.

---

## 1. Abstract (≤200 words; placeholders X, Y, Z to be filled from THIS
## paper's certified campaign — see note below)

> Thin-walled aerospace parts are milled at a fraction of their attainable
> material removal rate (MRR) because regenerative chatter, rather than
> machine capability, commonly sets the productivity ceiling. Active
> vibration control can raise that ceiling, yet industrial uptake remains
> marginal: mainstream designs assume a linear actuator of unbounded
> authority acting on a time-invariant workpiece, whereas shop-floor
> piezoelectric actuators saturate at aggressive depths of cut and the
> structural dynamics at the tool position vary continuously with feed and
> material removal. This paper develops a saturation-aware,
> position-scheduled control strategy treating both nonidealities as
> design objects: a gain-scheduled H-infinity controller, synthesized over
> the NC-known tool-path and removal schedule, is augmented with an
> anti-windup gain that maximizes a certified stability region, and the
> saturated, time-periodic, delayed closed loop is certified regionally by
> generalized-sector Lyapunov conditions on a semi-discretization lifting.
> Sweeping the certificates yields a permissible depth-of-cut envelope
> maximizing stability-limited MRR under an explicit
> perturbation-tolerance requirement. In a simulation campaign on a plate
> benchmark validated against published measurements, the strategy raises
> the certified worst-position depth of cut by $Y\times$ over the best
> fixed-gain design of the same family, cutting its vibration by $X\%$ at
> $Z\%$ of its voltage at the hardest tool position — converting actuator
> saturation from a hidden failure mode into a certified planning
> constraint.

**Placeholder discipline (do not violate):** Y, X, Z must come from THIS
paper's certified campaign once run. Do NOT reuse the companion paper's
numbers as if they were this paper's: its 4.2× worst-position gain is a
*linear Floquet* result without saturation certificates (the certified
depth is necessarily ≤ the linear limit); its 58 %/49 % figures are *total*
milling-point RMS and *fraction of the frozen design's voltage* at the
hard condition (x_T = 95 mm, a_p = 2 mm, simulation) — cite them only in
that exact framing, attributed to [companion].

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
removal-scheduled H-infinity synthesis of [companion], the actuator bound
is brought inside the scheduled design through an anti-windup gain
synthesized by iterative linear-matrix-inequality optimization to
maximize the certified stability region — in contrast to
constraint-avoiding, spindle-speed-mapped gain maps [jmp2026spindle] and
to bespoke adaptive laws with assumed initial-condition sets
[wu2016adaptive]. (ii) Stability of the saturated, time-periodic, delayed
closed loop is certified regionally, for a given fixed controller of
standard architecture, on the unaveraged periodic dynamics: a
semi-discretization lifting reduces the periodic delay dynamics with one
sector-bounded deadzone — formulated on the variational dynamics about
the verified-unsaturated forced periodic response, with the time-varying
voltage headroom as the deadzone bound — to a finite-dimensional periodic
system on which generalized-sector periodic-Lyapunov conditions
[fu2013-lineage] yield computable invariant sets; this extends the
Tarbouriech-lineage machinery to machining for the first time, whereas
the invariant sets of [wu2016adaptive] are neither computed nor maximized
and hold for a bespoke law on averaged dynamics. (iii) The certificates
are swept into a certified permissible-MRR envelope — guaranteed under a
declared perturbation tolerance / linear-only (saturation islands
possible) / unstable — that turns a phenomenon so far predicted only by
uncertified describing-function analysis [ozsoy2025mssp] into an a-priori
certified and avoidable planning constraint; across the operating grid,
under identical certificate machinery, the scheduled design is shown to
enlarge the certified region relative to the best non-scheduled
controller of the same family.

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
     latency & roll-off) + anti-windup gain by iterative LMI
     maximizing the certified region (V-K scheme; monotone
     non-decreasing certified volume per accepted iterate);
     baselines: best frozen H∞, delayed-PD, unsaturated-ideal
        │
 §4  Regional certification of the saturated periodic delayed loop
     • variational formulation: certificates apply about the
       verified-unsaturated forced periodic response; time-varying
       headroom V_max − |v_forced(t)| bounds the deadzone
     • semi-discretization lifting (rank-one delay → scalar
       histories; sampled-data ZOH with τ, T_s commensurate with the
       lifting step) → discrete periodic system + sector-bounded
       deadzone
     • periodic Lyapunov + generalized-sector LMIs (periodic P_k, G_k
       with step-wise level-set inclusion; structure-exploiting SDP)
       → invariant sets → physical margin map (allowable surface
       defect / force impulse), with the loss-of-contact state
       constraint h(t) > 0 imposed inside the certified set
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
