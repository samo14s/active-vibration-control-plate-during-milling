# Retracted claims in this package

This directory is the starting package, kept as the record the audit was
performed against. Its code is otherwise unmodified, so the evidence remains
inspectable. The claims below do not survive reproduction and must not be
carried into any manuscript.

Each entry gives the location, what the code actually does, and the measured
value. Full detail and reproduction scripts are in `../docs/ASSESSMENT.md`.

---

## 1. "Stability domain 21.7× open loop, +41 % vs LQG", "31.1 % effective damping"

**Where.** Four lines, now replaced by a `raise` that quotes the original:

| file | line |
|---|---|
| `05_main/main_simulation.py` | 636 |
| `04_figures/gen_article_complete_figures.py` | 719, 837 |
| `04_figures/gen_SLD_academic_style.py` | 142 |

each of the form `zeta_DARC_eff = zeta_LQG * 1.30`.

**What it does.** Multiplies the LQG closed-loop damping ratios by a
hard-coded constant and feeds the result into the open-loop lobe formula. The
DARC controller object never enters the stability computation. The adjacent
comment says "+20 %" while the code applies 1.30, and a third figure of 1.5
appears in the same comment block.

**Why it is worse than an unsupported number.** It is unattainable in
principle. A feedforward signal is input-additive: it never multiplies the
state, so it cannot appear in the monodromy matrix, and the spectral radius
that decides stability cannot change. **The correct value of that improvement
is exactly zero**, at any amplitude and any harmonic content — demonstrated in
`../tests/verify_feedforward_cannot_move_lobes.py`, where ρ is bit-identical
up to 1000 V across five harmonics.

For a lobe computed from the real sampled loop, use `../src/closed_loop_sld.py`.

---

## 2. "+19.31 % average RMS reduction vs LQG"

**Where.** `04_figures/gen_article_complete_figures.py`:190–201.

**What it does.** Gives the baseline `w_q_list=[1e13]` under a comment that
documents the handicap in its own words —

> `# LQG with SUB-OPTIMAL weights (typical engineer's guess, not full grid search)`
> `# This reflects realistic conditions where LQG is hand-tuned, not optimized.`

— three lines above the proposed method, which gets `base_w_q=1e14` under

> `# DARC-MPC uses OPTIMAL LQG base (w_q=1e14) + NN feedforward`

**Measured.** +4.32 % at matched tuning. The package's own source already
records this: `05_main/main_simulation.py`:634 carries the comment
`y_RMS LQG=0.532, DARC=0.507 → reduction 4.7%`.

**And it reverses.** Retuning the baseline to `w_q = 1e15` — excluded only by
an arbitrary `‖K‖ < 1e8` cap, and needing 18.5 V of the 150 V available —
gives 0.4938 µm against DARC-MPC's 0.5273 µm. The baseline **wins** in S1, S2
and S4.

Reproduce with `../tests/audit_baseline_claims.py`.

---

## 3. "Deep Adaptive Robust Control with MPC"

Three of the four terms are absent from the code.

- **MPC** — no prediction horizon, no finite-horizon cost, no QP anywhere in
  `02_controllers/darc_mpc_v3_controller.py`.
- **Robust** — `lambda_robust` is assigned at line 557 and never read again.
- **Adaptive** — `OnlineRLSAdapter` contains no recursive least squares. Its
  trigger compares a ~3.6e-8 m displacement against a 1e-6 m threshold and
  provably never fires: `omega_hat/omega_nom = 1.000000` after a full run.
- **Deep** — one hidden layer of 16 units, of which 6 of the 8 inputs receive
  exactly zero gradient (training states are appended as `np.zeros`).

What is actually deployed is DC 1.03 V + 2.02 V at the tooth-passing
frequency: a one-harmonic Fourier series, three effective parameters.

That architecture is also long-established prior art under other names —
repetitive control, adaptive feedforward cancellation, FxLMS — and in
machining specifically by Chen, Zhang, Zhang & Ding (ASME JDSMC 2014), who
adapt a Fourier expansion of the regenerative force online. See
`../docs/POSITIONING.md`.

---

## 4. Numba acceleration, and the "0.1 µs vs 100+ ms per step" timing

- `README.md`:187 claimed `precompute_alpha_periodic()` was Numba-compiled.
  There is no `numba` import anywhere in the package and it is not a
  dependency.
- `02_controllers/README.md`:103 claimed 0.1 µs against 100+ ms per step. No
  benchmark exists anywhere in the package, and there is no MPC to time.

Both statements have been removed from those files.

---

## 5. The Monte Carlo robustness study

`03_analysis/uncertainty_analysis.py` defines the machinery, but
`run_uncertainty_analysis` does not exist, `run_monte_carlo` is never called,
and the documented ±15 % is in fact ω ±2 % and K_T ±5 %. "Figure 14 —
Robustness Monte Carlo" is a boxplot over four deterministic scenarios.

Either run it or drop the claim; do not report the figure as stochastic.

---

## 6. The BibTeX entry

`README.md`:371–381 carried `@article{darcmpc2026, ...}` with `[Author names]`,
`[Journal name]`, `year = {2026}`, and a `note` asserting +19 % and +41 % as
established fact. Removed: a year-bearing `@article` for unpublished work
invites mis-citation, and neither number is reproducible.

---

## What in this package *is* sound

Worth stating, because the structural core is why this work was continued
rather than restarted:

- the Q8 Reissner–Mindlin element — shape functions, Jacobian, `Bf`/`Bs`,
  the `h³/12` and `κGh` scalings, rotary inertia;
- CCCC to 0.02 % against Leissa, cantilever f₁ = 519.4 Hz, mesh-converged;
- the closed-form helical force integrals, the regenerative sign convention,
  and the fly-over bookkeeping (12.2 % duty cycle, verified);
- the Newmark update itself.

One defect was found in the element and fixed in `../src/evolving_plate.py`:
the consistent mass matrix is integrated with 2×2 Gauss although its
integrand is quartic, leaving it rank 12 of 24 and the assembled `M_free`
singular. Selective reduced integration restores full rank and moves the
benchmark frequencies by 0.001 %.
