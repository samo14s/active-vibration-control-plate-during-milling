# `research_extensions/` — Deepening the study & closing its research gaps

This folder **extends** the `article_simulation_package` (it does not modify it).
Every module imports the *unmodified* plant, controllers and solver from the base
package via `_pkg_path.py`, so the physics is identical to the article — the new
code only adds rigorous analysis and the baselines/experiments a top-tier
reviewer would demand.

Start with **[`RESEARCH_GAPS_ANALYSIS.md`](RESEARCH_GAPS_ANALYSIS.md)** — the
reviewer-grade audit that motivates everything here (bilingual summary at top).

---

## The five gaps and what closes each

| # | Research gap (verified numerically) | New module | Headline result |
|---|---|---|---|
| **1** ★ | **Unfair comparison.** "Feedback cannot reject a periodic disturbance" contradicts the Internal Model Principle. The 55 % is vs a handicapped feedback-only LQG. | `internal_model_control.py` | A pure-feedback **Repetitive-LQG** gives **0.16 µm vs DARC 0.29 µm** with a good sensor — it *beats* the feedforward. DARC's real edge is **sensor-independence** (crossover at ≈0.35 µm sensor noise). |
| **3** | **Nonlinearity dormant.** LQG on the linear vs nonlinear plant is identical to 0.003 %, so every result is linear-regime. | `bifurcation_analysis.py` | Drives the true nonlinear chatter: a **bounded limit cycle** at a_p = 1 mm where the linear model **diverges** — the physics the NDDE model is *for*. |
| **4** | **Heuristic SLD.** The controlled stability curves use an *assumed* damping boost, not a Floquet result. | `closed_loop_stability.py` | Genuine **closed-loop semi-discretization**: LQG raises a_p,cr ≈ 20–30× (rigorous), and a **feedforward does not move the lobes** — so DARC ≠ 40×. |
| **5** | **Inert "Adaptive-Robust" layer.** `lambda_robust` is computed, never used. | `robust_monte_carlo.py` | Adaptation ON = OFF **bit-for-bit**; all-controller Monte-Carlo with 95 % CIs shows RC-LQG beats DARC on **100 %** of uncertain plants (fine sensor). |
| 6–9 | operating-point coverage, actuator realism, NN falsifiability — see the analysis doc §"Secondary gaps". | — | documented with concrete next steps. |

---

## Run it

```bash
pip install numpy scipy matplotlib          # same deps as the base package
cd research_extensions
python main_gap_study.py            # full study + 4 figures (~6–10 min)
python main_gap_study.py --quick    # coarse sanity run (~2–3 min)
```

Figures land in `research_extensions/figs/`:

| file | shows | gap |
|---|---|---|
| `fig_g1_sensor_crossover.png` | RC-LQG vs DARC vs LQG across sensor noise (the crossover) | #1 |
| `fig_g2_closed_loop_sld.png` | rigorous Floquet stability lobes (OL / LQG / RC-LQG) | #4 |
| `fig_g3_bifurcation.png` | nonlinear limit-cycle branch, NL vs linear, controlled Hopf | #3 |
| `fig_g4_robust_montecarlo.png` | all-controller robustness box-plot + adaptation audit | #5 |

Each new module is also usable on its own (see its docstring and the
`if __name__` / function-level examples).

---

## File map

```
research_extensions/
├── README.md                     ← this file
├── RESEARCH_GAPS_ANALYSIS.md     ← the reviewer-grade audit (read first)
├── _pkg_path.py                  ← imports the base package unmodified
├── internal_model_control.py     ← Gap #1: Repetitive/Internal-Model LQG (fair baseline)
├── closed_loop_stability.py      ← Gap #4: closed-loop Floquet SLD (semi-discretization)
├── bifurcation_analysis.py       ← Gap #3: nonlinear limit-cycle / Hopf study
├── robust_monte_carlo.py         ← Gap #5: adaptation audit + all-controller MC stats
├── main_gap_study.py             ← orchestrator: runs all four, makes the figures
└── figs/                         ← generated figures
```

## Honest bottom line

The base method (plant model + LQG + DARC feedforward) is sound and faithfully
coded. What the deepening changes is the **framing**: the defensible contribution
is not "feedback cannot reject periodic disturbances" (it can, and better with a
good sensor) but **"a spindle-phase feedforward is sensor-independent and
margin-safe, so it dominates a repetitive/internal-model feedback controller
precisely under coarse industrial sensing and at the spindle speeds where the
internal-model gain erodes the chatter margin."** That claim is narrower, true,
novel, and survives review — and it is now backed by rigorous, reproducible code.
