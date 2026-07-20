# Gap ③ — Saturation-aware active chatter control with regional certificates

**Verification date:** 2026-07-20 · **Verdict: GAP FRESH** (one flag: distinguish
carefully from Wu et al. 2016) · **SELECTED for the new strategy.**

## Candidate gap

Regional (basin-of-attraction) stability certificates and a certified
permissible depth-of-cut envelope for a SATURATED, TIME-PERIODIC, DELAYED
milling closed loop. Existing treatments are anti-windup add-ons or
constraint-avoiding gain maps without certificates.

## Closest works (agent-verified)

1. **Ozsoy, Sims & Ozturk 2025, MSSP** — "Actuator saturation during active
   vibration control of milling". Describing-function-style saturation model;
   experimentally shows saturation "islands" where chatter occurs BELOW the
   linear stability boundary. Closest on phenomenon and DF tool — explicitly
   no certificate, no basin.
2. **Ozsoy et al. 2026, J. Manuf. Processes** — saturation-aware
   spindle-speed-mapped DVF gain optimization (differential evolution),
   experimental (up to 13x depth). Constraint avoidance, zero certificates.
3. **Wu, Zhang, Huang, Zhao & Ding 2016, IJRNC** — adaptive chatter control
   with input saturation. THE ONE TO DISTINGUISH: contains invariant-set +
   restricted-initial-set language (proto-regional), but: bespoke adaptive
   backstepping law (not a certificate for a given controller), Fourier-
   averaged periodic matrix, basin assumed not computed/maximized, no
   permissible-depth envelope, simulation-only.
4. **IJAMT 2020** — adaptive sliding-mode turning control with input
   constraint; Lyapunov feasibility, no ROA quantification.
5. **Tarbouriech-lineage saturated-delay theory** — Fu/Zhou/Duan 2013 (Asian
   J. Control, maximized domain of attraction for delayed saturated systems),
   Lamrabet 2018/2020 (sampled-data delayed anti-windup). Applied to power
   systems and fast tool servos — NEVER to machining chatter.
6. **Yan/Wiercigroch 2017 PRE, 2021 Physica D** — numerical basins of
   time-delayed cutting dynamics (uncontrolled, turning-type).
7. **Dombovari & Stepan 2015/2019** — bistable/unsafe zones of the periodic
   milling DDE via unstable quasi-periodic orbits (no control, no actuators).
8. **Altshuller 2008, SIAM JCO** — delay-IQCs for time-periodic feedback
   systems: exactly the needed theory, sitting unused; never applied to
   milling or saturation.

## Defensibility requirements

- Claim wording: "no certified, COMPUTABLE, MAXIMIZED regional certificate
  for a GIVEN (linear-authority or anti-windup-augmented) controller on the
  TRUE time-periodic delayed dynamics" — not "no regional analysis of any
  kind" (Wu 2016).
- Certify the genuinely periodic loop (periodic LK functionals or
  Altshuller-type periodic delay-IQCs, generalized sector conditions), not
  Fourier-averaged dynamics.
- Deliver the operational envelope (depth/speed with quantified basin
  margins) — no machining paper provides it.
- Cross-validate LMI/IQC inner estimates against numerically computed basins
  and against the experimentally observed saturation islands of Ozsoy 2025.
