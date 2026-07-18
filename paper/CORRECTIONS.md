# Reproducibility audit and corrections

This repository grew out of an earlier simulation package that reproduced the
milling / plate / piezo model of

> J. Du, X. Liu, H. Dai, X. Long, *Robust combined time delay control for
> milling chatter suppression of flexible workpieces*, International Journal of
> Mechanical Sciences **274** (2024) 109257.
> https://doi.org/10.1016/j.ijmecsci.2024.109257

and proposed a learning-augmented controller ("DARC-MPC"). Before building on
that work, every headline result of the original package was re-derived from its
own code. Several reported numbers **could not be reproduced** and were traced to
hard-coded values or ad-hoc scaling rather than computation. This document
records those discrepancies and how the present repository resolves them, so the
contribution rests only on results that are actually computed.

The nominal-case "measured" value below was obtained by running the original
controller (`_incoming/article_simulation_package/02_controllers/darc_mpc_v3_controller.py`)
on the shared model; the reworked, honestly-named controller in `src/` reproduces
the same nominal reduction (`results/scenarios.json`, S1).

---

## 1. RMS vibration reduction was overstated ~4x at the nominal point

The README reports a near-uniform average of **+19.31 %** RMS reduction
(per-scenario +19.20 / +19.51 / +19.22 / +19.17 %, README lines 318-322). That is
not what the code produces. Running the controller at the nominal operating point
(S1) gives about **+4.6 %**, matching the source's own buried comment
(`05_main/main_simulation.py:634`:
`# In S1: y_RMS LQG=0.532, DARC=0.507 -> reduction 4.7%`). A near-identical
~19.2 % across four very different operating points is also implausible for a
nonlinear time-domain simulation.

**What is actually true.** A ~19 % reduction does occur -- but only under
modal-frequency uncertainty, not at the nominal point. This repository's honest
per-scenario results (`results/scenarios.json`) are:

| Scenario | honest RMS reduction (2-DOF vs LQG) |
|---|---:|
| S1 Nominal | +4.7 % |
| S2 Aggressive | +4.6 % |
| S3 Uncertainty (omega -15 %) | **+19.5 %** |
| S4 High K_T | +4.7 % |

The original package's error was to attribute the *uncertainty-case* gain
uniformly to every scenario. The nominal reduction is modest (~4.6 %), with an
additional ~7-8 % peak reduction that the original did not report.

## 2. The controlled stability-lobe diagram was fabricated

The original "DARC" stability boundary was produced by multiplying the LQG
closed-loop damping ratios by a constant factor and re-running the *open-loop*
lobe solver:

```python
# 05_main/main_simulation.py:636
zeta_DARC_eff = (np.array(zeta_LQG_sld) * 1.30).tolist()
# 04_figures/gen_article_complete_figures.py:719, 837
zeta_DARC_eff = zeta_DARC * 1.30
```

The `x1.30` factor has no physical basis. Worse, it contradicts the code's own
(correct) observation that the feedforward term does not move the closed-loop
poles:

```python
# 05_main/main_simulation.py:509
# DARC v3 closed-loop : approximate by LQG closed-loop (DARC uses same K_lqr base)
# In reality, DARC adds NN_FF which doesn't change CL poles directly
```

A feedforward signal cannot change the closed-loop poles, and therefore cannot
change the linear stability boundary. The reported "+41 % stability domain /
21.7x / a_p,crit = 3.05 mm" for DARC over LQG is thus an artefact of the
multiplier, not a property of the controller.

**Resolution.** This repository introduces a **closed-loop full-discretization
method** (`src/cl_fdm.py`) that embeds the actual controller inside the Floquet
monodromy matrix, giving the true controlled stability-lobe diagram. As a
validation, its open-loop critical depth at 4900 rpm (~0.15 mm) matches the
published experimental order (~0.1 mm). Controlled boundaries are then computed,
not scaled:

| Configuration | a_p,crit (computed, observer/ESO in loop) |
|---|---:|
| Open loop | 0.063 mm |
| LQG (Kalman observer in loop) | 1.92 mm |
| ADRC (extended-state observer in loop) | 3.25 mm |

The open-loop value (0.063 mm) is of the same order as the ~0.1 mm measured in
the paper, confirming the assembly. Every controlled boundary is computed with
the controller's observer inside the monodromy matrix -- not scaled. See
`results/sld.npz` and `results/sld_summary.json`.

## 3. Modal-damping figures were hard-coded / scaled

The "modal damping 0.31 % -> 23.9 % -> 31.1 %" values are a mixture of the same
`x1.30` scaling and text hard-coded into the figure generator
(`04_figures/gen_control_strategy_diagram.py:699,736`:
`'• Damping: 0.31% -> 23.9%'`,
`'+19% RMS reduction · +41% stability domain · +100x modal damping'`). These are
annotations, not measurements.

**Resolution.** Damping is read from the actual closed-loop eigenvalues of the
computed feedback design; no figure text is hard-coded.

## 4. Method naming overclaimed the algorithm

The controller was named *"Deep Adaptive Robust Control with MPC"* with a
docstring referring to "differentiable physics". In fact there is no model
predictive control (no receding-horizon optimisation), the network is a single
hidden layer of 16 units, and its "learning" uses a fixed heuristic target
(`darc_mpc_v3_controller.py:412`, `K_correction = 1e6`) rather than a
differentiable-physics gradient.

**Resolution.** The same, genuinely-useful component is retained but named for
what it is: a **phase-aware feedforward** inside an honest two-degree-of-freedom
controller (`src/twodof_control.py`), with its role (forced-vibration and
voltage reduction, *not* stability extension) stated explicitly.

---

## What is kept, because it is real

The physical model is sound and is reused unchanged: the Kirchhoff Q4 plate FEM,
modal reduction (mode 1 at 521 Hz, matching the paper), the helical milling-force
model, the Newmark integrator with the regenerative delay term, the realistic
piezo model, and the LQG controller. The phase-aware feedforward genuinely
reduces forced vibration and peak voltage. The open-loop lobe solver is correct.
The contribution in this repository is built strictly on these verified pieces.
