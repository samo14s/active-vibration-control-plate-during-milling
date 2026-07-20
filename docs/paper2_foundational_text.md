# Foundational text — saturation-aware, position-scheduled AVC for thin-wall milling

Draft v1 (pre-adversarial-review). Basis: verified gap dossiers
(`gap_scouting/`), literature positioning, and the validated numerical
campaign of this repository. Citation keys refer to `paper/refs.bib` plus
the dossier entries (Ozsoy 2025/2026, Wu 2016, Fu 2013, Altshuller 2008).

---

## 1. Abstract (structure: context → industrial gap → strategy → results → significance)

> Thin-walled aerospace components are machined at a fraction of their
> attainable material removal rate (MRR) because regenerative chatter, not
> spindle power, sets the productivity ceiling. Active vibration control
> can raise that ceiling, yet industrial uptake remains marginal: reported
> controllers are designed as if actuators were linear and the workpiece
> time-invariant, whereas on the shop floor piezoelectric actuators
> saturate at aggressive depths of cut and the structural dynamics seen by
> the tool vary continuously with its position and with material removal.
> This paper develops a saturation-aware, position-scheduled control
> strategy that treats both nonidealities as design objects: a
> gain-scheduled H-infinity controller synthesized over the tool-path and
> material-removal schedule is augmented with a certified anti-windup
> loop, and the saturated, time-periodic, delayed closed loop is certified
> by regional (basin) stability conditions built on a semi-discretization
> lifting. The certificates are swept into a permissible depth-of-cut
> envelope that maximizes MRR under a guaranteed-stability constraint. On
> a benchmark thin-walled plate validated against published measurements,
> the strategy increases the worst-position stable depth of cut by
> $Y\times$ over the strongest non-scheduled design, sustains an $X\%$
> reduction in chatter-band vibration at $Z\%$ of the actuator voltage,
> and converts saturation from a hidden failure mode into a certified
> planning constraint — a prerequisite for industrial deployment.

(198 words with placeholders X, Y, Z; verified fill-ins from this
repository's campaign if desired: Y× = 4.2, X% = 58 % at the hard
condition, Z% = 49 %.)

## 2. Introduction — research-gap paragraphs

**P1 (critique of idealized assumptions).**
The active-control literature for thin-wall milling has reached
methodological maturity on an idealized plant: a linear actuator of
unbounded authority acting on a time-invariant structure. Within that
frame, optimal, robust and mu-synthesis designs [du2024robust,
vandijk2012robust, zhang2019robust], model-predictive schemes
[li2019mpc-class], and adaptive or learning controllers
[kleinwort2018adaptive, nasiri2025chatter] all demonstrate order-of-
magnitude gains in stable depth of cut — in simulation or in laboratory
cuts conducted safely inside the actuator's linear range. Two idealizations
underwrite these results. First, actuator limits are absent from the design
model: the voltage bound of a piezoelectric patch (or the force bound of an
inertial actuator) appears, if at all, as an a-posteriori check.
Second, the workpiece is frozen at one nominal configuration: the
position- and removal-induced variation of the modal participation at the
cutting point is either ignored or wrapped into norm-bounded uncertainty,
whose price is conservatism — a single controller detuned everywhere so as
to fail nowhere [du2024robust].

**P2 (why the two omissions cap industrial productivity).**
Neither idealization survives contact with production economics, because
both bind exactly where productivity is decided. Saturation binds first at
high depth of cut: recent measurements document "saturation islands" in
which chatter erupts *below* the linearly predicted stability boundary
once the actuator clips [ozsoy2025mssp], so a linearly certified envelope
overstates the safe operating region precisely where MRR is highest —
an unacceptable failure mode for unattended machining. Existing responses
either avoid the constraint by gain maps tuned to stay unsaturated
[ozsoy2026jmp] or prove signal boundedness for bespoke adaptive laws on
simplified, averaged dynamics [wu2016adaptive]; none certifies what a
process planner needs: the region of initial conditions and cutting
parameters over which a *given* industrial controller, behind a *given*
amplifier, is guaranteed stable on the *true* time-periodic delayed
dynamics. Spatial dependency binds second: the authority of any fixed
design collapses where the dominant mode's participation at the tool
position vanishes, so the worst position along the path — not the average
— caps the feed-through depth of the entire pass. The two limits compound:
where authority drops, the controller demands more voltage, and
saturation arrives sooner. Productivity in thin-wall milling is therefore
governed by a saturated, position-varying stability boundary that the
current design toolchain neither shapes nor certifies.

**P3 (explicit novelty statement).**
This paper closes that gap with a control strategy in which the actuator
bound and the tool-path variation are first-class design objects rather
than nuisances. Its contributions are three. (i) A position- and
removal-scheduled H-infinity controller, synthesized on a validated
finite-element electromechanical model and interpolated over the NC-known
schedule, is co-designed with an anti-windup loop — to the authors'
knowledge the first scheduled AVC synthesis for thin-wall milling in
which saturation enters the design rather than the post-mortem. (ii)
Stability of the resulting saturated, time-periodic, delayed closed loop
is certified regionally: a semi-discretization lifting reduces the
periodic delay dynamics with one sector-bounded deadzone to a
finite-dimensional periodic system, on which generalized-sector
Lyapunov conditions yield computable invariant sets — the first such
certificate for a given (non-bespoke) milling controller on the
unaveraged periodic dynamics, in contrast to the averaged
invariant-set analysis of [wu2016adaptive]. (iii) The certificates are
swept into a certified permissible-MRR envelope — guaranteed /
linear-only / unstable — that turns the measured saturation-island
phenomenon [ozsoy2025mssp] from an unexplained hazard into a predicted
and avoidable planning constraint, and whose guaranteed region the
scheduled design provably enlarges over the best non-scheduled
controller of the same family.

## 3. Methodology outline (section flowchart)

```
 §2  Electromechanical modeling of the varying plant
     FEM (Mindlin) plate + bonded piezo patch → modal LPV model over
     θ = (x_T, ϱ); regenerative milling force (periodic, delayed);
     INPUT NONLINEARITY: u = sat(v) = v − dz(v), amplifier bound ±V_max;
     validation vs published modal data, measured FRFs, and measured
     open-loop stability pattern
        │
 §3  Saturation-aware, position-scheduled controller synthesis
     grid LPV H∞ over θ (weights incl. implementation latency & roll-off)
     + anti-windup co-design (iterative LMI maximizing the certified
     basin); baselines: best frozen H∞, delayed-PD, unsaturated-ideal
        │
 §4  Stability certification of the saturated periodic delayed loop
     semi-discretization lifting (rank-one delay → scalar histories)
     → discrete periodic system + sector-bounded deadzone
     → periodic Lyapunov + generalized-sector LMIs → invariant sets
     → physical margin map (allowable surface-defect / impulse magnitude);
     companion linear certificates: spillover small-gain, exact
     sampled-loop radius, closed-loop stability lobes
        │
 §5  Certified permissible-MRR envelope and productivity optimization
     sweep (a_p, Ω, x_T, ϱ) → three-zone envelope
     {certified | linear-only (islands possible) | unstable};
     MRR maximization s.t. certified-zone constraint; island prediction
        │
 §6  Numerical campaign and validation
     nonlinear time-domain evaluation (loss of contact, saturation,
     noise, moving tool); numerical basins (IC sweeps) vs LMI sets;
     reproduction of the measured saturation islands [ozsoy2025mssp];
     robustness Monte-Carlo
        │
 §7  Results → §8 Discussion & limitations → §9 Conclusion
```

## 4. Key Performance Indicators

**A. Productivity / stability capacity (the reviewer's bottom line)**
1. Worst-position critical depth a_lim,min = min_x a_lim(x_T) at the
   production speed — open-loop / best-frozen / scheduled / scheduled+AW.
   *The* number that caps a full pass.
2. Certified permissible-MRR envelope area (mm·krpm) per strategy, and
   its guaranteed-zone share vs the linear-only zone; band-worst a_lim
   over 2–10 krpm.
3. MRR gain at guaranteed stability: ΔMRR % = (a_lim,min ratio) × 100 at
   fixed a_e, f_t — state alongside cycle-time reduction per part.
4. Saturation-island census: number/extent of linearly-stable but
   certificate-refused operating points; islands predicted vs observed in
   nonlinear simulation (and vs [ozsoy2025mssp] where comparable).

**B. Vibration quality**
5. Milling-point RMS and peak displacement (steady cut, worst position),
   and chatter-band PSD attenuation (dB) at the dominant chatter
   frequency; % reduction vs open loop and vs best-frozen.
6. Surface-location-error proxy: per-revolution mean deflection along the
   pass (links vibration to tolerance, the quality-side argument).

**C. Actuation realism (the industrial-credibility block)**
7. RMS and peak voltage vs ±V_max; saturation duty cycle (% of control
   updates clipped) — must be ≈0 in the certified zone by construction.
8. Voltage headroom margin at the envelope boundary and actuation energy
   per part (J) — the economics of actuator sizing.

**D. Guarantees and robustness (the strict-reviewer block)**
9. Certificate margins reported as numbers: spillover small-gain value
   (<1), exact sampled-loop spectral radius (<1), certified basin radius
   translated to physical units (largest tolerated surface defect h_max /
   force impulse), and the conservatism ratio certified-basin /
   numerical-basin volume (honesty metric).
10. Monte-Carlo capability: median and P(a_lim ≥ a_p,target) under
    cutting-coefficient dispersion (0.3–2.9×), ±10 % modal
    mass/stiffness (mode shapes rescaled), −20 % damping — per strategy.
11. Scheduling-value ablations: performance with stale (fresh-plate)
    gains vs removal-scheduled gains; frozen-at-x0 vs scheduled along the
    pass. Each ablation isolates one claimed mechanism.

**E. Implementation feasibility**
12. Controller order and max pole vs real-time rate; per-update FLOPs;
    gain-interpolation latency — evidence the strategy runs on a PXIe-class
    target.
