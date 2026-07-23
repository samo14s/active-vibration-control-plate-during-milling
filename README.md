# Active Vibration Control of a Thin-Walled Plate During Milling

Simulation and control study of **chatter mitigation** in peripheral milling of a
cantilever AL6061 plate with a bonded piezoelectric actuator — comparing model-
based controllers (LQG, DARC-MPC) on a nonlinear regenerative (NDDE) plant model
after *Nasiri & Moradi, MSSP 224 (2025) 112198*.

The repository has two parts:

```
active-vibration-control-plate-during-milling/
├── article_simulation_package/   ← the base study (plant, LQG, DARC-MPC, figures)
└── research_extensions/          ← deepening: rigorous baselines, stability & statistics
```

## 1. `article_simulation_package/` — the base study

The original simulation package: analytical Von Kármán / Galerkin plate model,
nonlinear Newmark NDDE solver, an **LQG** controller and the **DARC-MPC**
controller (LQG base + spindle-phase feedforward + neural residual), plus the
figure generators. See [`article_simulation_package/README.md`](article_simulation_package/README.md).
Headline claim: *DARC-MPC reduces RMS vibration ≈ 55 % below LQG.*

## 2. `research_extensions/` — deepening & closing the research gaps

A reviewer-grade audit that runs the base code to **verify** its claims, finds the
gaps a top-tier journal (IEEE TCST / MSSP / Automatica / CIRP) would flag, and
**closes** them with new, self-contained modules. Read
[`research_extensions/RESEARCH_GAPS_ANALYSIS.md`](research_extensions/RESEARCH_GAPS_ANALYSIS.md)
first (bilingual summary at the top), then run:

```bash
pip install numpy scipy matplotlib
cd research_extensions && python main_gap_study.py        # (--quick for a fast pass)
```

### What the deepening found (all measured on the article's own model)

1. **The "55 %" is against a handicapped baseline.** The claim *"feedback cannot
   reject a periodic disturbance"* contradicts the Internal Model Principle. A
   fair **pure-feedback repetitive/internal-model LQG** reaches **0.16 µm vs
   DARC's 0.29 µm** with a good sensor — it *beats* the feedforward. DARC's real,
   defensible edge is **sensor-independence** (it wins once the sensor is coarse,
   ≳ 0.35 µm noise). *(new: `internal_model_control.py`)*
2. **The nonlinear model is never exercised** — LQG on the linear vs nonlinear
   plant is identical to 0.003 %. A new **bifurcation study** wakes it: a bounded
   limit cycle at a_p = 1 mm where the linear model diverges.
   *(new: `bifurcation_analysis.py`)*
3. **The stability-lobe diagram is a heuristic.** A new **closed-loop Floquet /
   semi-discretization** gives the real numbers (LQG ≈ 20–30×, rigorous) and
   shows a **feedforward cannot move the lobes** — so the claimed DARC "40×" is
   both heuristic and conceptually wrong. *(new: `closed_loop_stability.py`)*
4. **The "Adaptive-Robust" layer is dead code** (`lambda_robust` computed, never
   used — audited ON = OFF bit-for-bit). A new **all-controller Monte-Carlo** with
   confidence intervals replaces the LQG-only robustness test.
   *(new: `robust_monte_carlo.py`)*

**Bottom line:** the base method is sound and faithfully coded; the deepening
replaces an over-broad claim with a narrower, true, novel one — *a spindle-phase
feedforward is sensor-independent and stability-margin-safe, so it dominates an
internal-model feedback controller precisely under coarse industrial sensing.*

## Requirements

Python 3.10+, `numpy`, `scipy`, `matplotlib` (see
`article_simulation_package/requirements.txt`).
