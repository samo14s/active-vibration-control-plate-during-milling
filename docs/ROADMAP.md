# What to do next

**Status: 23 of 24 must-fix items done, one partial.**

Outstanding work, all of it flagged inline below:
- item 17 is partial — the chip geometry is reported, but `K_T` is still a
  constant and needs the published power law, plus edge and process-damping
  terms;
- `dt = tau/328` (item 13) is settled but the time-domain runs still use
  `tau/82`;
- a real Monte Carlo (item 5 / part B step 6) is worth doing, though no claim
  now depends on it.


Two lists: defects that must be repaired before anything is submitted, and a
validation plan that works without a laboratory. Both come from the audit and
literature surveys; items already done here are marked.

---

## A. Must fix

### Integrity — non-negotiable

1. ✅ **Delete every `zeta * 1.30` line** and every number derived from them —
   21.7×, 15.5×, 31.1 %, +41 %. Locations: `gen_article_complete_figures.py`
   :719 and :837, `gen_SLD_academic_style.py`:142, `main_simulation.py`:636.
   The claim is also unattainable in principle (feedforward cannot move the
   monodromy — proven in `tests/verify_feedforward_cannot_move_lobes.py`).
2. ✅ **Retire the name DARC-MPC.** No MPC, no adaptation, "Deep" is one 16-unit
   layer. Any surviving use of the phase-indexed feedforward must be called
   repetitive control / adaptive feedforward cancellation, and cited to
   Chen, Zhang, Zhang & Ding (ASME JDSMC 2014) and Tsao & Tomizuka (1994).
3. ✅ **Delete the `@article{darcmpc2026}` BibTeX stub** (`README.md`:371–381),
   with its placeholder author/journal and its note asserting +19 % and +41 %
   as established fact.
4. ✅ **Delete the Numba claim** (`README.md`:187 — no `numba` import anywhere,
   not in `requirements.txt`) and the "0.1 µs vs 100+ ms per step" MPC timing
   claim (no benchmark exists in the package).
5. ✅ **The Monte Carlo claim is withdrawn.** `run_uncertainty_analysis` does
   not exist; the documented ±15 % is actually ω ±2 % and K_T ±5 %;
   `run_monte_carlo` is never called; "Figure 14 — Robustness Monte Carlo" is
   a boxplot over four deterministic scenarios. Recorded in
   `baseline/RETRACTED.md` item 5. *Running a real one remains worthwhile —
   see part B step 6 — but no claim now rests on it.*

### Numerics

6. ✅ **3×3 Gauss for the mass matrix** (keep 2×2 for shear only) — done in
   `src/evolving_plate.py`. Removes 72 zero-mass directions and the hourglass
   mode; frequencies move 0.001 %.
7. ✅ **Never decouple the modes in a stability boundary** — done in
   `src/closed_loop_sld.py`. `Dp Dpᵀ` is rank one with off-diagonal/diagonal
   0.69–0.96; the per-mode maximum is 20.8 % non-conservative.
8. ✅ **Never average a signed mode shape** — documented in
   `tests/verify_substitution_error_along_path.py`. The signed average deletes
   mode 2 exactly.
9. ✅ **Root-find `a_p,crit`, do not grid-scan** — done. The baseline's grid
   rule reports 0.14 mm against a converged 0.077 mm: **82 % high, in the
   denominator of every improvement ratio.**
10. ✅ **`dt = tau/N` for integer N.** The baseline's `tau/dt = 81.63` rounds
    to 82, so its solver runs at **4878 RPM** while its own SLD uses 4900, and
    every FFT harmonic marker is 0.45 % off. Both new experiment scripts now
    use `Ts = tau/82` exactly; quantified in
    `experiments/run_discretisation_study.py`.
11. ✅ **One `m_div` for the whole package, with a convergence table.**
    `gen_SLD_academic_style.py` uses 40 and `main_simulation.py` uses 30; the
    open-loop `a_p,crit` moves 0.0646 → 0.0783 mm between them — a **21 %
    spread on the headline number from a discretisation parameter**.
12. ✅ **Correct the method attribution.** `fdm_stability.py` implements
    zeroth-order **semi-discretization** (Insperger & Stépán, IJNME 55 (2002)
    503–518), not full discretization. Fix the docstring and `README.md`
    :242–249.
13. ✅ **`dt` settled at `tau/328`.** At `tau/82` (49.8 µs) mode 3 is detuned
    5.86 % against a 0.54 % half-power bandwidth — about 11×. `tau/328`
    (12.4 µs) brings every retained mode inside its own bandwidth. *Adopting
    it in the time-domain runs, which still use `tau/82`, is outstanding.*

### Physics

14. ✅ **Enforce chip-thickness positivity** `g(h) = h·(h>0)` per tooth per axial
    slice, or restrict every time-domain amplitude claim to `|y| < h_max =
    3.98 µm` and drop `stop_threshold` from 5e-3 m to ~1e-4 m.
    *Linear stability certificates about the periodic motion remain valid
    without this — see `ASSESSMENT.md` §5.*
15. ✅ **Piezo patch mass and stiffness into the structure**, with the
    regression test — `tests/verify_piezo_patch_structure.py` locks the
    measured +3.72 % / +1.93 % shifts (6.0 and 5.7 half-power bandwidths).
16. ✅ **Moment arm to the composite neutral axis** — done: 2.022 mm, not
    2.350 mm, a 16.2 % overstatement. Quote the resulting static tip
    deflection per volt.
17. ◐ **Cutting speed and chip thickness reported** — `chip_thickness_limits`
    gives `h_max` = 3.98 µm, `h_mean` = 2.00 µm, 154 m/min. **Outstanding:**
    replace the constant `K_T = 925 MPa` with the published chip-thickness
    power law (part B step 4), and add edge/ploughing and process-damping
    terms, both of which matter at a few microns.
18. ✅ **Move the probe off the machined surface** — done. The baseline put an
    eddy-current sensor at the free corner in the same 0.15 mm band as the
    tool path, coincident with the cutter at end of pass.

### Comparison protocol

19. ✅ **One identical tuning protocol**, and the tuning curves are now
    written to `results_benchmark.json` rather than only the winner.
20. ✅ **Open-loop baseline in every new comparison.** The baseline's
    `main_simulation.py` never runs one, and `gen_response_figures.py`:10
    asserts `a_p = 0.1 mm` is "stable, a bounded forced vibration" when its
    RMS grows **4.18×** across the record.
21. ✅ **Common metric window** `i_end = min over controllers of stop_idx`, and
    report divergence as a separate binary outcome.
22. ✅ **Same actuator model for every controller.** `LQGController.step` has no
    saturation at all while DARC clips three times.
23. ✅ **Seeds.** Everything is pinned to 42. Run ≥10 seeds and report mean ± std,
    or a 4.3 % effect cannot be distinguished from training noise.
24. ✅ **Add a CFFF (cantilever) Leissa case.** The package verifies CCCC but the
    actual plate is a cantilever; a reviewer will ask for the boundary
    condition actually used.

---

## B. Validation without a laboratory

Ordered. Steps 0–1 need no external data.

**Step 0 — Say the right words.** Adopt ASME V&V 10-2019 (R2025) terminology
explicitly and state in Section 1 that matching a published FRF is
*verification / benchmark comparison*, **not validation**. Pre-empting the
referee's framing is worth more than any single data point.

**Step 1 — Code verification.** ✅ done: Leissa CCCC 0.02 %, **CFFF cantilever
−0.60 % / −0.26 %** (`tests/verify_leissa_cantilever.py`), mesh convergence,
`f ∝ h`, first-order perturbation with O(Δh) convergence, monodromy vs dense
to 1e-19, certified lobe vs time domain, nonlinear force vs the linear
coefficients. Note λ₁ = 3.492 and λ₂ = 8.525 are independently confirmed at
ν = 0.3 but modes 3–6 are *not* — check the original before tabulating them.

**Step 2 — Validate the piezo sub-model in isolation**, against open-access
experiments whose parameters are in the text rather than buried in figures:
- Sangpet, Kuntanapreeda & Schmidt, *J. Engineering* 2014, Art. 839128.

**Step 3 — Validate the milling sub-model and stability solver** against a
published thin-wall experiment:
- Li, Zhao, Li, He, Chi & Remond, *Shock and Vibration* 2015, Art. 431476 —
  Ti6Al4V cantilever plate, parameters retrieved verbatim by the survey.

**Step 4 — Fix and defend the cutting-force coefficient.**
- Wang, Hao, Wang, Hou & Lallart, *Shock and Vibration* 2018, Art. 3831825 —
  open access, **Al6061-T6, peripheral milling, 10 mm 4-tooth 35° helix, the
  same process as yours**, and it publishes the coefficients as a
  chip-thickness **power law**. This is what `K_T = 925 MPa` should be
  replaced by, or at minimum stated against.

**Step 5 — Validation-by-reproduction of published closed-loop results.**
Never attempted, and named as one of the few contributions genuinely
available to a simulation-only group. Digitise open-loop FRFs and force
coefficients from two or three of: Du & Long 2022 (0.2 → 1 mm); Du, Liu &
Long 2023 (1.5 → 6 mm); Du, Liu, Dai & Long 2024 (0.1 → 0.8 mm); Ozsoy, Sims
& Ozturk 2022 (2.6×); Aggogeri et al. 2021 (96 % at 1130 Hz); Zhang & Sims
2005 (7×, PPF).

**Step 6 — Uncertainty quantification.** *The substitute for a laboratory, and
the strongest single move available.* Propagate what you cannot measure —
modal damping, clamp stiffness, bond compliance, `K_T` and `K_r`, `d31`, patch
thickness tolerance — through to the certified `a_p,crit`, and report the
boundary as a distribution rather than a line.

**Step 7 — Internal cross-validation**, the one check a referee can run
themselves. For every certified `a_p,crit`, run Newmark at 0.8×, 1.0× and
1.2× the boundary and report the **growth ratio** (RMS of last quarter / RMS
of first quarter), not the divergence guard. ✅ pattern already used in
`tests/verify_closed_loop_sld.py`.

**Step 8 — Ask for the raw data.** Rated the highest-return action available.
Email corresponding authors of Li et al. 2015, Sangpet et al. 2014 and the Du
group for the underlying FRF files.

**Step 9 — Release an open benchmark.** Not one paper in this corpus
publishes machine-readable FRF, modal or SLD data. Release the plate model,
the certified-lobe code, the tuning protocol, the digitised datasets from
steps 3 and 5, and the reproduction scripts.

**Step 10 — Frame honestly and pre-emptively.** Title and abstract must say
"simulation study" or "model-based feasibility study". A Limitations section
must state *in the paper, not in a rebuttal*: no closed-loop cutting
experiment was performed; the cutting law is linear; the modal truncation is
three modes and its error is not propagated into the certificate.

**Step 11 — Choose the venue with open eyes.** MSSP's clearest written
statement (ML-paper guidelines, 5 Dec 2024) asks for ≥1 experimental dataset
and notes such papers are often rejected without review — though that document
is formally scoped to ML submissions and its wording is **unverified here**.
IEEE TCST has **no written rule** against simulation-only work and its scope
explicitly includes simulation and novel modelling. **TCST is the better
target of the two.**
