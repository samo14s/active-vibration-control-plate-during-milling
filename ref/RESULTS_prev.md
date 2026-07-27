# PPF → PD, µ-synthesis, and the combined law: four-way comparison

Plant: 100 × 80 × 4 mm AL6061 cantilever plate (clamped at *y* = 0), PZT patch
60 × 20 × 0.7 mm bonded spanwise at *x* = 5 mm, Mindlin/MITC4 FEM, six-mode ROM,
ζ = 0.5 %, collocated patch sensor.

`plate_fem.py` was missing from the archive for the third session running and has
been reconstructed here — *E* calibrated to 65.85 GPa, reproducing the validated
bare-plate basis exactly at the 40 × 32 mesh:
519.6 / 1054.0 / 2687.5 / 3290.9 / 4047.5 Hz. **Keep this file in the archive.**

All four laws are closed on the same plant, tuned against the same objective
(maximise the worst-tool-position stability-limit lift) under the same constraints
(peak effort ≤ 150 V/N, no mode below 0.45 %), and scored by the same metrics.
The PPF baseline reproduces the reference result exactly (×12.25, 148.1 V/N),
which is what makes the comparison meaningful.

## Nominal plate

| | PPF (ref.) | **PD** | µ-synthesis | PD + µ combined |
|---|---|---|---|---|
| a_lim lift (worst tool) | ×12.25 | **×17.04** | ×13.13 | ×13.19 |
| peak receptance [µm/N] | 28.3 | **23.3** | 28.8 | 30.5 |
| peak reduction | ×14.5 | **×17.6** | ×14.2 | ×13.5 |
| ζ mode 1 [%] | 24.33 | 12.05 | **34.87** | 32.12 |
| ζ mode 2 [%] | 4.35 | 6.06 | **10.91** | 6.91 |
| min ζ all modes [%] | 0.50 | **0.62** | 0.53 | 0.53 |
| static stiffness ratio | 0.861 | **1.001** | 0.905 | 0.797 |
| peak effort [V/N] | 148.1 | 149.9 | **142.3** | 151.3 ✗ |
| controller order | 4 | **1** | 26 | 28 |
| 5 % settling [ms] | 18.7 | **8.0** | 12.7 | 14.3 |
| peak µ upper bound | — | — | **1.073** | 1.248 |

Open loop: peak 409.6 µm/N, static 7.045 µm/N, settling 127 ms.

PD gains: k_p = 8.96e6 V/m, k_d = 1.635e5 V·s/m, derivative filter f_c = 10 kHz.

## Machined plate — gains FROZEN, f1 +17 %, f2 +11 %

| | PPF | **PD** | µ-synthesis | combined |
|---|---|---|---|---|
| a_lim lift | ×8.09 | **×13.68** | ×8.81 | ×9.32 |
| peak receptance [µm/N] | 31.2 | **21.1** | 32.2 | 30.2 |
| peak effort [V/N] | 130.0 | 145.0 | 142.5 | 144.7 |
| min ζ [%] | 0.50 | **0.62** | 0.39 ✗ | 0.43 ✗ |
| **a_lim retained** | 66 % | **80 %** | 67 % | 71 % |

## What the numbers say

**1. PD beats PPF outright, and the reason is structural.** The patch sensor is
collocated with the patch actuator (c_S = b_mod), so a negative-feedback PD adds
k_p·b bᵀ in stiffness and k_d·b bᵀ in damping — two rank-1 positive semi-definite
updates. The law is passive on this pair and cannot destabilise the plant at any
gain. PPF is *positive* position feedback: it subtracts stiffness, which is why its
static compliance degrades to 0.861 while PD sits at 1.001. Moving to PD removes a
structural defect rather than merely retuning one.

**2. The optimiser drives k_p towards zero.** It is the derivative term alone that
lifts a_lim; PD degenerates into filtered velocity feedback. Sweep of the derivative
cutoff: ×5.41 at 1 kHz, ×16.04 at 5 kHz, ×17.04 at 10 kHz, ×17.92 at 50 kHz. 10 kHz
was retained — implementable, 5 % below the unconstrained optimum.

**3. µ-synthesis does not pay for itself here — negative result.** It was designed
explicitly against the machining frequency excursion, yet it retains 67 % of its
nominal a_lim, statistically indistinguishable from PPF's 66 % — and PPF was designed
with no robustness consideration at all. The order-1 PD retains 80 %. Worse, µ and the
combined law both fall *below* the 0.45 % spillover floor on the machined plate
(0.39 % and 0.43 %), i.e. the robust designs are the ones that violate their own
constraint under the very perturbation they were synthesised against. The peak µ of
1.073 > 1 was already saying robust performance was not guaranteed; the replay
confirms it.

The mechanism: high modal damping on modes 1–2 (µ reaches ζ₁ = 34.9 %) does not
translate into a_lim, because a_lim is set by the most negative real part of the
receptance, which is a broadband property. Broadband passive damping wins on the
metric that matters for productivity.

**4. The combined law is dominated — hypothesis refuted.** The working hypothesis was
that a cheap PD inner loop would let the outer µ loop reach µ < 1 inside the voltage
budget. It does not: at identical weights the cascade gives µ = 1.248 against 1.073
for standalone µ, at the same a_lim (×13.19 vs ×13.13) and higher effort
(151.3 vs 142.3 V/N). No combined design is strictly admissible under 150 V/N.
De-rating the inner PD to gain 0.6 does push µ to 1.014, but drives effort to
374.7 V/N.

Reason: **both laws drive the same patch.** The PD already consumes most of the
available authority for broadband damping, the µ loop stacks on top, and it is the
*total* voltage that is penalised. There is no residual authority to exploit, so the
cascade cannot beat the better single law. The binding constraint is the actuator,
not the controller structure.

Practical corollary: to make a combined architecture worthwhile, add a **second
patch** (or enlarge the existing one) — not a more sophisticated controller.

## Reservations, stated rather than buried

- The D-K grid is coarse and the iteration is non-monotone. Several designs come out
  with a_lim < 1 (they make chatter *worse*) at absurd voltages — e.g. C = 25 µm/N,
  V_target = 60, inner gain 1.0 gives ×0.60 at 8299 V/N. One case
  (C = 50, V = 110, gain 0.6) is outright unstable, ζ_min = −0.20 %. The µ and
  combined figures are therefore *best-of-a-coarse-grid*, not converged optima,
  unlike PD and PPF which are finely tuned. This asymmetry favours PD in the
  comparison and should be closed before publication.
- The µ upper bound uses complex D-scales without G-scales, while δ₁, δ₂, δ_a are
  real parameters. The reported margins are conservative, not optimistic.
- Robustness is modelled as a ROM frequency shift with mode shapes and modal
  authorities held fixed. A full FEM re-solve with the reduced free length would also
  move the mode shapes and the patch/tool coupling.
- Modal damping is held at ω̄ rather than made a function of δ (error O(ζ·p) ~ 1e-3).
- a_lim and peak metrics are refined by local Brent search after the grid scan; on
  grid alone the PPF baseline read ×11.83 instead of ×12.25 — a 3.4 % error, the same
  order as the differences being measured.

## Files

| file | role |
|---|---|
| `plate_fem.py` | reconstructed Mindlin/MITC4 plate — **keep this** |
| `patch_fem.py` | unchanged (your Crawley–de Luis patch model) |
| `avc_plant.py` | shared plant, controller container, uniform scoring |
| `avc_pd.py` | PD design |
| `avc_mu.py` | generalised plant, µ upper bound, D-scale fitting, D-K iteration |
| `avc_combined.py` | PD inner + µ outer, parallel realisation |
| `avc_compare.py` | four-way comparison, robustness replay, figures |
| `pd_gains.npz`, `ctrl_mu.npz`, `ctrl_combined.npz` | designed controllers |
| `cmp_frf.png` | receptance and chatter-relevant real part, both tool positions |
| `cmp_time.png` | impulse response and patch voltage |
| `cmp_robust.png` | µ curves and productivity before/after material removal |
