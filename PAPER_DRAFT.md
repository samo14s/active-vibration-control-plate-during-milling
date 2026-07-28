# Frontier-certified convex synthesis of active chatter control for flexible-workpiece milling

*Manuscript draft — numbers marked {…} are filled from the committed
artefacts (`_newlaw_*.npz`, `newlaw_log.txt`, `cmp2_*` figures).*

Target: A+ journal (Int. J. Mech. Sci. / MSSP class). Benchmark, plant and
milling model: Du, Liu, Dai & Long, IJMS 274 (2024) 109257 — same plate,
same patch, same cutting-force model, same delay structure, same
material-removal excursion; our independently reconstructed MITC4/Mindlin
FEM replaces their Chebyshev–Ritz basis (their measured plant is reproduced
to ≤2 % on all five modes, closer than their own theoretical model on
modes 2–3).

---

## Abstract (draft)

Active suppression of regenerative chatter in thin-walled milling is
dominated by surrogate-objective designs: the controller optimises an
H∞/µ-weighted norm, and the quantity that pays for machine time — the
guaranteed axial depth of cut — is only inspected afterwards. This paper
reformulates the problem so that the guaranteed depth IS the objective,
and shows that the reformulation is convex. In the single-direction
regenerative model of a flexible workpiece, the speed-independent depth
floor is inversely proportional to max_ω [|G| − Re G] of the operative
cutting-point receptance G — a quantity that vanishes exactly when G is
positive real. Chatter-proofing a plate is therefore a constrained
passivation problem, and with a Youla/IMC parametrisation on the (stable)
piezo-patched plate both the objective and every engineering constraint —
per-frequency voltage budget, spillover caps, robustness to the
frequency drift between re-tunes — are second-order-cone representable in
the parameter Q. The program is solved to global optimality in seconds,
which turns robustness to material removal from a synthesis burden into a
scheduling triviality: the law is re-synthesised as the plant drifts, and
each re-synthesis carries its own certificate. Because the closed-loop
receptance is affine in Q and |·|−Re(·) is 2-Lipschitz, the same algebra
yields a controller-independent *authority frontier*: an upper bound on
the depth floor achievable by ANY internally-stabilising LTI controller
through the given patch under the given voltage budget. On the IJMS-274
benchmark the frontier evaluates to ap* ≈ {0.70–0.82} mm — quantitatively
explaining the 0.8 mm experimental ceiling reported there, and proving
that the next unit of improvement on this plant must come from actuator
authority, not controller sophistication. Against the benchmark's
µ-synthesis-plus-time-delay control and against PD/PPF/µ baselines
re-tuned under identical two-plant feasibility rules, the proposed law
{multiplies the full-discretisation stability floor by ×{…} at equal
voltage / reaches {…} % of the physical frontier}, with a controller of
order {≤24}.

---

## 1. Contributions (the claims the paper stands on)

1. **Exact convex reformulation of the industrially meaningful objective.**
   The guaranteed stability-lobe floor of the single-direction regenerative
   model is maximised directly — not an H∞ surrogate — as an SOCP in a
   Youla/IMC parameter: global optimum, no D–K iterations, no weight
   tuning, reproducible to the solver tolerance. The identity
   floor⁻¹ ∝ max_ω [|G_cl| − Re G_cl] exposes chatter-proofing as
   *budget-constrained passivation* of the cutting point.

2. **The actuator-authority frontier.** A controller-independent upper
   bound on the achievable floor for the architecture (patch position +
   sensor + voltage budget), obtained from the same affine algebra with a
   Lipschitz argument — computable in closed form from open-loop FRFs, no
   optimisation involved. It (i) explains the benchmark's experimental
   0.8 mm ceiling, (ii) gives designers a map of *where* (frequency ×
   position) authority runs out, (iii) turns "add a second patch" from
   folklore into a computed requirement.

3. **Removal-adaptive certified scheduling.** Because a re-synthesis is a
   convex solve (seconds) while material removal is slow (minutes), robust-
   ness over the full excursion is replaced by a scheduled family of
   globally-optimal laws, each certified for the drift window between
   re-solves (IMC small-gain cone |QΔG_su| ≤ ½ — a *constraint of the
   program*, not an afterthought). The "price of robustness" curve — floor
   vs drift window — is reported explicitly.

4. **Speed-scheduled lobe shaping.** At a known spindle speed the same
   machinery maximises the lobe itself (an LP: the delay factor
   (1−e^{−jωτ}) is constant data), lifting the operating point beyond the
   broadband floor.

5. **Honest verification chain.** ZOA is only the design surrogate; every
   reported closed loop is verified by the full-discretisation method on
   the time-periodic delayed system (the benchmark's own instrument), and
   the paper's µ-comparison baselines (PPF / PD / µ / µ+delay) are re-tuned
   under two-plant feasibility so the comparison cannot be won by
   selection bias. (Our re-analysis of the benchmark's method family under
   these rules is itself a contribution to reproducibility.)

## 2. Model matching (Section 2 of the manuscript)

| item | benchmark (Du et al.) | this work |
|---|---|---|
| plate | 100×80×4 mm AL6061 cantilever | same |
| discretisation | Chebyshev–Ritz, penalty BCs | MITC4 Mindlin FEM (independent) |
| modes vs their measured [Hz] | 540/1068/2787/3351/4122 | 543.5/1084/2733/3369/4115 (≤1.9 %) |
| damping | measured 0.31/0.17/0.27/0.56/0.35 % | same values adopted |
| patch | QDA60-20-0.7 at lower-left, vertical | same footprint on our FEM |
| piezo constant | d31·E_pe/(1−ν) = 16.96 | 17.01 (calibrated independently, +0.3 %) |
| force model | Eq (2)–(4), kc=925 MPa, kn=0.26, µc=0.2, helix 35° | same, α₄ sign fixed by their observed chatter branch (f_c > f_n) |
| milling | down, ae=0.1 mm, ft=0.02 mm/t, 3 teeth, Ø10 | same |
| delay | single τ = 60/(Ω·3) | same |
| removal | f1+17 %, f2+9 % (measured) | same (modes 3–6: +5 % assumed) |
| stability instrument | full-discretisation (their [79]) | same (independent implementation) |
| validation | — | uncontrolled floor {0.027} mm, lobes at 3600/5400 rpm, their point S (4900 rpm, 0.3 mm) unstable: reproduced |

## 3. Method (Section 3)

3.1 Regenerative floor and passivation identity — derivation of
    ap_floor = 1/(κ₄ max_ω[|G|−Re G]) and the positive-real limit.
3.2 Youla/IMC parametrisation on the stable patched plate; affinity of
    G_cl and of the per-frequency voltage; the SOCP; the rational basis
    (36 second-order sections + LP tail; solutions reduced to order ≤24 by
    balanced truncation with a ≤2 % floor-change acceptance test).
3.3 The authority frontier (Lipschitz bound); frontier maps.
3.4 Robustness: drift cone |QΔG_su| ≤ ½ (internal stability for any
    mismatch below the sampled envelope — small-gain), scheduled
    re-synthesis, cross-stage FDM verification of the frozen law.
3.5 Speed-scheduled LP variant.

## 4. Results (Section 4) — filled from artefacts

- drift sweep table {…}: floor vs drift window, frontier line, uncontrolled
  line ("the price of robustness").
- per-stage table {…}: floor / frontier / achieved % / order / peak V/N.
- FDM SLDs {…}: uncontrolled vs new law (fresh), stage-matched machined,
  frozen-law cross-stage; lift factors at 3600/4900/5400 rpm and at the
  benchmark's S-point.
- scheduled-LP lifts at the test speeds {…}.
- time-domain showcase at S: displacement/voltage traces {…}.
- comparison table vs PPF / PD / µ / µ+delayed-PD (benchmark's law family,
  re-tuned honestly): floors, retention across removal, voltage, order.

## 5. Reservations to state, not bury

- ZOA is the design surrogate; interrupted cutting (ae/D=1 %) makes it
  conservative in places — all comparative claims rest on the FDM, and the
  FDM-to-ZOA gap is reported per law.
- The frontier bounds LTI feedback through the given sensor/actuator pair
  under a per-frequency voltage spec; nonlinear or non-causal schemes are
  outside its scope (delayed-feedback laws that exploit τ are covered by
  the scheduled variant's algebra, not by the broadband frontier).
- κ₄ carries the benchmark's own 0.3–2.9× coefficient band; it scales
  every law's floor identically and is reported as a band on absolute
  numbers, never on ratios.
- Mode shapes are frozen across removal (as in the benchmark); modes 3–6
  removal shifts are assumed (+5 %) — sensitivity reported.
