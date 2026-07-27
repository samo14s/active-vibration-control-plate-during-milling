# Retuned controllers: PPF / PD / µ-synthesis / PD + µ — fourth session

Plant: 100 × 80 × 4 mm AL6061 cantilever plate (clamped at *y* = 0), PZT patch
60 × 20 × 0.7 mm at x ∈ [5, 25] mm running up the cantilever from the clamp,
Mindlin/MITC4 FEM, six-mode ROM, ζ = 0.5 %, collocated patch sensor.

**Headline: for the first time, all four laws are feasible on the fresh AND
the machined plate** — every constraint the previous session's robust designs
violated (ζ 0.39 % / 0.43 % against the 0.45 % floor, 151.3 V/N against the
150 V budget) is now met with margin, the machined-side productivity of the
robust laws is up (72–78 % retained vs 66–71 %), and the µ certificates are,
for the first time, computed on a structure that actually covers the replay
and with a bound that is not vacuously conservative.

## The archive was incomplete again — and this time it is pinned down

`patch_fem.py` was missing for the fourth session running and `plate_fem.py`
arrived truncated (mesh and shape functions only — no assembly, no
eigensolve). Both are reconstructed and validated against every number the
previous report documented:

| anchor | reference | reconstruction |
|---|---|---|
| bare-plate modes, 40 × 32 [Hz] | 519.6 / 1054.0 / 2687.5 / 3290.9 / 4047.5 | identical to the digit |
| open loop, corner tool: peak / static [µm/N] | 409.6 / 7.045 | 409.5 / 7.052 |
| PPF g = [0.092, 0.055]: a_lim, effort | ×12.25, 148.1 V/N | ×12.20, 148.1 V/N |
| PPF ζ₁ / ζ₂ / static ratio | 24.33 % / 4.35 % / 0.861 | 24.39 % / 4.34 % / 0.859 |
| stored PD gains: a_lim, effort | ×17.04, 149.9 V/N | ×16.90, 150.3 V/N |
| stored µ controller: a_lim, effort, ζ₁ | ×13.13, 142.3 V/N, 34.87 % | ×13.27, 142.0 V/N, 35.63 % |
| stored combined: a_lim, effort | ×13.19, 151.3 V/N | ×13.19, 150.9 V/N |

Two reconstruction findings worth keeping:

1. **The patch runs up the cantilever, not across it.** A 60-mm spanwise
   strip at the clamp couples to the second bending mode ~3× more strongly
   than to mode 1, which drives the PPF cross-terms to ζ₁ ≈ 33 %, static
   ratio ≈ 0.49 and ~200 V/N — nowhere near the documented 24.33 % / 0.861 /
   148.1. With the 20 × 60 mm footprint (long side along *y*, covering 3/4
   of the free length) every anchor lands within ~1 %: mode-1 curvature
   integrates over the whole patch while the mode-4 curvature partly
   cancels. **Keep `plate_fem.py` and `patch_fem.py` in the archive.**
2. Free parameters are calibrated once and documented: E = 65.85 GPa (as
   before), d₃₁ = −181.3 pm/V (inside the PZT-5A spread, set by the
   148.1 V/N PPF anchor); everything else is standard PZT-5A / AL6061 data.

## What was retuned, and what changed methodologically

The previous session's robust designs failed their own constraints under the
very perturbation they were synthesised against. Five process defects caused
that; all five are fixed rather than re-searched harder:

1. **The machined plate is part of the acceptance test.** Every candidate is
   scored on the fresh plant AND on the frozen-gain machined replay (f₁
   +17 %, f₂ +11 %, modes 3–6 +5 %), plus a mid-excursion stability check.
   Feasible means feasible on both. The old workflow discovered the replay
   violations after selection — and would have crowned several of this
   session's ×13–15 fresh-plate designs that the replay reveals to be
   unstable (ζ as low as −28 %).
2. **The certificate covers the replay.** The old µ upper bound modelled
   only the mode-1/2 excursion; the +5 % shift of modes 3–6 that the replay
   applies was outside the certified set — designs with µ < 1 on that
   structure could and did go unstable on the machined plate through the
   unmodelled high modes. The reported µ is now computed on the full
   structure: six real modal deltas + actuator gain ±15 % + the 2×2
   performance block. (Synthesis still runs D-K on the two-mode structure —
   carrying six fitted D-scales through hinfsyn wrecks its conditioning; a
   bound computed on the full structure is valid no matter how the
   controller was found.)
3. **The µ bound is mixed (G-scales), not complex-D.** The old report
   footnoted that its real parameters were bounded with complex D-scales
   "conservatively". That conservatism is not a rounding concern — it is
   structural: δ₃…₆ are frequency shifts of 0.5 %-damped modes, and a
   complex delta of p = 4.9 % on a mode with 2ζ ≈ 1 % is a phantom damping
   perturbation no voltage-limited controller can cover. Complex-D peaks of
   10–30 collapse to 1.2–1.9 under the Fan–Tits–Doyle mixed bound
   (`mu_ub_mixed`, seeded on the complex-D optimum so it can only tighten;
   validated on exact small cases and Monte-Carlo lower bounds). Without
   G-scales the certificate column would be numerically meaningless.
4. **Every D-K iterate is a candidate.** The iteration is non-monotone (γ
   can jump an order of magnitude between passes, and SB10AD can hang
   outright on a badly D-fitted plant — cells therefore run in forked
   subprocesses with a hard timeout and one retry, wild D-fits fall back to
   the data's log-mean constant, and the D-scaled plant is similarity-
   balanced before synthesis). The best-a_lim controller is frequently not
   the smallest-µ iterate; all iterates of all weight cells enter the same
   two-plant selection.
5. **The control weight no longer has a DC hole.** The legacy Wu evaluates
   to 1/(10 V_target) below its corner — and the resonant voltage peaks of
   modes 1–2 sit exactly there, which is how 300+ V/N designs sailed through
   γ < 1 for three tuning grids in a row. With the flat shape (1/V_target
   through the resonances, 10/V_target above the corner) V_target finally
   means what it says; both surviving winners come from flat-Wu cells.

PD was re-derived on the reconstructed plant and lands where the old report
said it would (k_p = 3.16e6 V/m — weakly identified, the optimiser again
drives the law towards filtered velocity feedback — k_d = 1.61e5 V·s/m,
f_c = 10 kHz, a_lim ×16.91). The PPF baseline is the documented
g = [0.092, 0.055].

## Nominal plate

| | PPF (ref.) | **PD** | µ-synthesis | PD + µ combined |
|---|---|---|---|---|
| a_lim lift (worst tool) | ×12.20 | **×16.91** | ×11.34 | ×10.62 |
| peak receptance [µm/N] | 28.4 | **23.5** | 32.7 | 33.7 |
| peak reduction | ×14.4 | **×17.4** | ×12.5 | ×12.1 |
| ζ mode 1 [%] | **24.39** | 12.10 | 7.76 | 10.04 |
| ζ mode 2 [%] | 4.34 | **6.01** | 4.06 | 3.38 |
| min ζ all modes [%] | 0.50 | **0.62** | 0.50 | 0.50 |
| static stiffness ratio | 0.859 | **1.000** | 0.861 | 0.959 |
| peak effort [V/N] | 148.1 | 148.5 | 133.2 | **127.3** |
| controller order | 4 | **1** | 26 | 16 |
| 5 % settling [ms] | 18.7 | **8.0** | 18.8 | 18.7 |
| peak mixed-µ, full structure* | 1.278 | 1.847 | **1.274** | 1.282 |

Open loop: peak 409.5 µm/N, static 7.052 µm/N, settling 129 ms.

\* one common certificate spec for all four columns: C = 45 µm/N,
V = 150 V/N, flat Wu, F_WU = 2 kHz — six real modal deltas + real actuator
gain ±15 % + full 2×2 performance block, upper bounds.

## Machined plate — gains FROZEN, f1 +17 %, f2 +11 %, modes 3–6 +5 %

| | PPF | **PD** | µ-synthesis | combined |
|---|---|---|---|---|
| a_lim lift | ×8.06 | **×13.57** | ×8.15 | ×8.30 |
| peak receptance [µm/N] | 31.4 | **21.4** | 34.0 | 32.0 |
| peak effort [V/N] | 129.8 | 143.7 | 126.3 | **124.3** |
| min ζ [%] | 0.50 | **0.62** | 0.50 | 0.47 |
| **a_lim retained** | 66 % | **80 %** | 72 % | 78 % |

All four columns satisfy every constraint on both plants. In the previous
session µ and combined sat at 0.39 % / 0.43 % against the 0.45 % floor and
the combined needed a 2 % "tolerance" on the voltage budget; that tolerance
is gone.

## What the numbers say

**1. PD still wins productivity, and now it also wins it honestly.** ×16.91
on the fresh plate, ×13.57 with frozen gains after material removal (80 %
retained), order 1, unit static stiffness. The structural argument is
unchanged: the law is passive on the collocated pair, so the excursion can
detune its performance but not its stability. Nothing in two hundred
retuned robust candidates touches it on the productivity axis.

**2. The old µ and combined results were selection artifacts, and the
retuned campaign proves it the hard way.** 225 candidates were scored across
four weight grids. Under fresh-plate feasibility alone — the previous
session's criterion — dozens of designs with a_lim ×13–15 at ≤ 150 V/N would
have been declared winners; the machined replay shows voltage blow-ups of
2–400× and closed-loop damping to −28 %. Exactly ten candidates survive
two-plant feasibility, and the best of them are the tabled µ (×11.34) and
combined (×10.62). The honest robust numbers are *smaller* on the fresh
plate and *better* where robustness is actually measured: retention 72 % and
78 % against 66–67 % last time, effort margins of 17–23 V/N instead of a
violated budget.

**3. µ-synthesis still does not pay for itself — the negative result
sharpens.** The retuned µ law no longer beats even PPF on the fresh plate
(×11.34 vs ×12.20) and only edges it after machining (72 % vs 66 %
retained). At the common certificate spec their margins are
indistinguishable (1.274 vs 1.278). A 26-state synthesis that lands within
noise of a 1985-vintage two-filter law, on the metric family it was designed
for, is a result — the machinery's real product on this plant is the
certificate, not the performance.

**4. The combined law is rehabilitated from "infeasible and dominated" to
"legitimate, second-best robust".** With the inner PD de-rated to 0.5 and an
honest control weight, the cascade is strictly inside every budget
(127.3 V/N peak — the lowest of all four laws), retains 78 % of its
productivity through the cut (second only to PD), matches the best
certificate (1.282), and does it with 16 states against the µ law's 26. It
still does not beat standalone µ by enough to change the practical
recommendation, and both robust laws remain behind PD — the binding
constraint is still the single patch's authority, and the corollary stands:
the next unit of improvement is a **second patch**, not a better controller.

**5. What the certificates actually say.** No law certifies µ < 1 against
the full flat-Wu spec: the mixed-µ margins are 1.27–1.85, i.e. weighted
robust performance is *guaranteed* for 54–78 % of the modelled uncertainty
box, and the full box is verified feasible by direct replay for all four
laws (the bound is an upper bound; PD's 1.847 is priced by its high-
frequency effort channel, not by any observed fragility). The µ peaks sit
at 1145 Hz — the mode-2 robustness channel — and softening the performance
target from 45 to 375 µm/N moves the peak by < 3 %, so the residual margin
is plant- and budget-limited, not a weight artifact. Two sub-unity numbers
were transiently observed and are worth recording as a caution: they were
valid certificates of the *legacy* effort spec (weak below its corner) that
a parsing slip attached to flat-Wu designs — certificates must certify the
spec the design claims, and the committed numbers do.

## Selected designs

| | weights (C, V, F_WU, noise, Wu) | D-K iterate | order |
|---|---|---|---|
| µ-synthesis | 45 µm/N, 110 V/N, 2 kHz, 1e-3, flat | 2 of 6 | 26 |
| PD + µ combined | 45 µm/N, 150 V/N, 1.4 kHz, 1e-3, flat, inner gain 0.5 | 1 of 4 | 16 |

Committed as `ctrl_mu.npz`, `ctrl_combined.npz`, `pd_gains.npz`; the four
campaign logs (`tune_log.txt` … `tune4_log.txt`) are the full provenance:
225 scored candidates, 10 two-plant feasible.

## Reservations, stated rather than buried

- The reported µ values are upper bounds without a matching lower-bound
  search; margins may still be conservative, never optimistic.
- The D-K iteration remains chaotic: which iterate of which cell survives
  is a draw, amplified by run-to-run BLAS nondeterminism that a
  single-thread pin does not remove. The committed logs record the actual
  draws; cells run in subprocesses with timeout + retry because SB10AD's
  γ iteration can hang outright.
- Robustness is still modelled as a ROM frequency shift with mode shapes,
  modal authorities and damping held fixed; a full FEM re-solve of the
  machined geometry would also move the shapes and the patch/tool coupling.
- Selection maximises fresh-plate a_lim among feasible candidates; the µ
  objective the synthesis optimises is a different (weighted H∞) quantity,
  so the winners are best-of-feasible-draws, not converged optima of a_lim.
- The redesigned PD's measured effort on this plant is 148.5 V/N; the
  archived previous-session gains measure 150.3 V/N here (0.2 % over
  budget) — differences of this size are within the model-reconstruction
  tolerance and do not change any conclusion.

## Files

| file | role |
|---|---|
| `plate_fem.py` | reconstructed Mindlin/MITC4 plate, assembly + eigensolve — **keep this** |
| `patch_fem.py` | reconstructed Crawley–de Luis patch, anchor-calibrated — **keep this** |
| `avc_plant.py` | shared plant, controller container, uniform scoring |
| `avc_pd.py` | PD design |
| `avc_mu.py` | generalised plant (selectable uncertainty channels), D-K, complex-D and mixed-µ bounds |
| `avc_combined.py` | PD inner + µ outer, parallel realisation |
| `avc_tune.py` … `avc_tune4.py` | the four tuning campaigns (two-plant feasibility, subprocess isolation) |
| `avc_recert.py` | full-structure mixed-µ recertification of feasible candidates |
| `avc_compare.py` | four-way comparison, common-spec certificates, robustness replay, figures |
| `pd_gains.npz`, `ctrl_mu.npz`, `ctrl_combined.npz` | selected controllers |
| `figs/cmp_frf.png` | receptance and chatter-relevant real part, both tool positions |
| `figs/cmp_time.png` | impulse response and patch voltage |
| `figs/cmp_robust.png` | mixed-µ certificates and productivity before/after material removal |
| `ref/` | previous session's report, figures and controllers (provenance) |
| `tune*_log.txt` | campaign logs: every candidate, every draw |
