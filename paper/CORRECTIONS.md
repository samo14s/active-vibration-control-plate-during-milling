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

All "measured" values below come from running the original modules unmodified;
they are reproduced by `experiments/run_all.py` in this repository.

---

## 1. RMS vibration reduction was overstated ~4x

| Scenario | Original `README` claim | Measured (original code) |
|---|---:|---:|
| S1 Nominal | +19.20 % | **+4.6 %** |
| S2 Aggressive | +19.51 % | **+4.6 %** |
| S3 Uncertainty | +19.22 % | **+4.7 %** |
| S4 High K_T | +19.17 % | **+4.7 %** |
| **Average** | **+19.31 %** | **+4.66 %** |

The advertised ~19 % is not produced by the code. Notably, the original
`05_main/main_simulation.py:634` contains the comment
`# In S1: y_RMS LQG=0.532, DARC=0.507 -> reduction 4.7%`, i.e. the true figure
(~4.7 %) was present in the source but not in the reported results. The
near-identical per-scenario claims (19.20 / 19.51 / 19.22 / 19.17 %) across four
very different operating points are also implausible for a nonlinear
time-domain simulation, whereas the measured ~4.6 % is genuinely consistent.

**Resolution.** The vibration reduction is now reported as measured
(~4.6 % RMS, and additionally ~7-8 % peak reduction, which the original did not
report). See `results/scenarios.json`.

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

| Configuration | a_p,crit (computed) |
|---|---:|
| Open loop | 0.15 mm |
| LQG (closed-loop FDM) | 2.43 mm |
| CL-FDM voltage-budget design | 5.31 mm |

Note that the honest LQG boundary (2.43 mm) already exceeds the 3.05 mm the
original attributed to its "DARC" method via the multiplier, and a properly
synthesised feedback reaches 5.31 mm within the +/-150 V budget: the fabrication
was not only unphysical, it actually *understated* what honest design achieves.
See `results/sld.npz` and `results/sld_summary.json`.

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
16-16 hidden-unit map, and its "learning" uses a fixed heuristic target
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
