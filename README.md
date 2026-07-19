# Control-oriented chatter suppression in thin-walled milling

Active vibration / chatter control of a cantilevered thin plate during
peripheral milling, using a piezoelectric patch actuator. This repository builds
on the model of

> J. Du, X. Liu, H. Dai, X. Long, *Robust combined time delay control for milling
> chatter suppression of flexible workpieces*, International Journal of Mechanical
> Sciences **274** (2024) 109257. https://doi.org/10.1016/j.ijmecsci.2024.109257

and contributes an **honest, control-oriented stability framework** plus a
**two-degree-of-freedom controller** whose every reported number is computed and
reproducible.

## Contributions in one paragraph

The standard way to obtain a *controlled* stability-lobe diagram (SLD) is often
faked (see [`paper/CORRECTIONS.md`](paper/CORRECTIONS.md)); a feedforward or a
damping multiplier is used to "improve" a boundary that only feedback can move.
Here the controller — **including its observer** — is placed **inside** the
Floquet monodromy matrix (`src/cl_fdm.py`, *closed-loop semi-discretization*), so
the controlled SLD is genuine. On that basis: (1) we show the Kalman observer
costs the LQG margin ~21 % (1.92 vs 2.43 mm); (2) we design an **ADRC** controller
(`src/adrc_control.py`) with a collocated piezo sensor that raises the linear
critical depth to **3.25 mm**, roughly **halves** tip vibration versus LQG at
25–55 V, and stays stable under ±20 % frequency drift where LQG chatters —
needing only the input gain `b₀`; (3) we introduce the **voltage-feasible
critical depth** (saturated nonlinear time domain) as the honest design metric
and show through a **transducer-placement co-design study** that it ranks
placements in nearly the *opposite* order to the linear boundary (Spearman
ρ = −0.4; a 12 mm linear boundary can mean 0.93 mm feasible) — under it, ADRC
achieves **1.92 mm vs 1.38 mm for LQG (+39 %)** at identical hardware; (4) the Kirchhoff model is refined to **precise** level
(`paper/modeling.md`): the patch's mass/stiffness enter the FEM as a composite
section, dropping the mode-1 error vs the *measured* 540 Hz from −3.5 % to
**+0.12 %** (five-mode mean 1.44 %, better than the source article's own theory
at 1.93 %), and the fidelity-layer analysis (truncation → spillover → sampling)
shows a 10 kHz evaluation *reverses* the verdict spuriously while the converged
endpoint (refined plant, 20 kHz) confirms **ADRC 1.65 mm vs LQG 1.29 mm
(+28 %)**; (5) two natural ADRC augmentations (regeneration-aware delayed
channel, resonant ESO) are honestly documented as **negative results** (< 2 %
gain); (6) a **phase-aware feedforward** (`src/twodof_control.py`) is shown to
reduce forced vibration/voltage but *not* the stability boundary; and (7) on the
**article's original single tip sensor** (a non-minimum-phase channel — the only
sensor used in this comparison), four controller classes meet in the same CL-SD
monodromy and two-stage metric (`experiments/fopid_tip_study.py`,
`experiments/hybrid_tip_study.py`): the model-based LQG works (**1.29 mm**
feasible), a model-free **fractional-order PID** (`src/fopid_control.py`)
survives but is weak (0.20 mm), plain ADRC **fails at every bandwidth** (its
lumped ESO inverts the RHP zeros), and a **HYBRID ADRC-FOPID**
(`src/hybrid_adrc_fopid.py`) — a band-limited ESO with a *searched signed* gain
plus a co-designed FOPID branch — reaches **0.83 mm, 4x the best fixed-structure
law**, making disturbance-rejection control usable at all on this sensor.  Its
costs are stated plainly (saturated ~37 µm limit cycle at low depth, −20% drift
fragility), and the design rule is computed, not asserted: the ESO must be
band-limited and sign-matched to the mode-1 subplant with the companion
controller co-designed around it — a retrofit is provably impossible (stage-A
search finds none). A **v2 iteration** (`experiments/hybrid_v2_study.py`: ESO
leakage + injection roll-off + robust min-max co-design, all LTI, validated in
`experiments/test_hybrid_validation.py`) removes the saturated low-depth limit
cycle (**37 → 0.29 µm** at 0.15 mm, LQG-class there, at 1.6 V), repairs the
refined-plant linear boundary for the robust design (**0.01 → 0.96 mm**), and
proves the added elements jointly load-bearing by ablation — at disclosed
costs (nominal margin 0.65 vs 0.83 mm; +20% drift headroom 0.29 vs 0.74 mm)
and with an honest negative: the −20% drift weakness survives even a
discretization-converged linear ±20% min-max certificate on the design model,
which the fidelity layers revoke. All numbers are computed and reproducible.

## Layout

```
src/                       importable modules
  kirchhoff_q4.py          Kirchhoff Q4 plate finite element (Hermite, 12 dof)
  plate_model.py           FEM assembly, modal reduction, piezo coupling, Dp(x)
  milling_force.py         helical end-mill force coefficients a3(t), a4(t)
  newmark_solver.py        Newmark-beta integrator with regenerative delay
  piezo_actuator.py        realistic piezo (saturation, slew, hysteresis)
  lqg_controller.py        baseline LQG (Kalman + LQR)
  ol_fdm.py                open-loop stability-lobe solver (reference)
  cl_fdm.py                *** closed-loop semi-discretization: controller
                               (+ observer) inside the Floquet monodromy
  adrc_control.py          *** ADRC (extended-state observer + control law)
  afc_adrc.py              *** AFC-ADRC: + spindle-synchronous adaptive
                               feedforward comb on the tip sensor (FxLMS)
  fopid_control.py         *** fractional-order PID (PI^lambda D^mu, Oustaloup)
  hybrid_adrc_fopid.py     *** HYBRID: band-limited ESO + FOPID on the tip sensor
  twodof_control.py        *** feedback + phase-aware feedforward (2-DOF)
  floquet_synthesis.py     feedback-authority design curve (supplementary)
experiments/
  run_all.py               reproduces every core result -> results/
  placement_study.py       transducer-placement co-design (linear + feasible)
  augmentation_study.py    negative results: delayed channel & resonant ESO
  fopid_study.py           FOPID design library (DE tuning, metrics, drift)
  fopid_tip_study.py       tip-sensor baselines: LQG / plain ADRC / plain FOPID
  hybrid_tip_study.py      HYBRID design (2-stage DE) + final tip-only comparison
  hybrid_v2_study.py       hybrid v2: leakage/roll-off/min-max, Pareto, ablation
  test_hybrid_validation.py  auditable module validation (5 checks)
  make_figures.py          builds figures/ from results/
results/                   computed JSON / NPZ (created by run_all.py)
figures/                   publication figures (created by make_figures.py)
paper/
  manuscript.md            the write-up
  CORRECTIONS.md           reproducibility audit of the original package
_incoming/                 the original simulation package (kept for provenance)
```

## Reproduce

```bash
pip install -r requirements.txt
python experiments/run_all.py          # ~6-9 min, writes results/
python experiments/make_figures.py     # writes figures/
```

Use `--quick` for a coarser (faster) SLD grid. Individual stages:
`--stage 1` (authority curve), `2` (SLD: OL/LQG/ADRC, observer in loop),
`3` (2-DOF scenarios), `4` (feedforward role), `5` (ADRC scenarios),
`6` (robustness sweep). Then:

```bash
python experiments/placement_study.py      # ~3 min: co-design + feasible depths
python experiments/augmentation_study.py   # ~1 min: negative results
python experiments/model_refinement.py     # ~3 min: precise-model validation,
                                           #   spillover + sampling endpoint
python experiments/material_removal.py     # ~3 min: in-process removal
python experiments/full_process_sim.py --controller lqg   # ~5 min each:
python experiments/full_process_sim.py --controller adrc  #   continuous 4-pass
python experiments/full_process_sim.py --controller afc   #   end-to-end process
python experiments/fopid_tip_study.py      # ~11 min: tip-sensor baselines
                                           #   (LQG works; plain ADRC fails;
                                           #    plain FOPID weak)
python experiments/hybrid_tip_study.py     # ~10 min: HYBRID ADRC-FOPID design
                                           #   + final tip-only comparison
python experiments/hybrid_v2_study.py      # ~57 min: v2 improvements (leakage,
                                           #   roll-off, robust min-max, Pareto,
                                           #   ablation, delayed-channel test)
```

The full derivation of the (refined) Kirchhoff model is in
[`paper/modeling.md`](paper/modeling.md).

## Physical setup

Cantilever AL6061 plate 100 x 80 x 4 mm; QDA60-200.7 piezo patch (d31, +/-150 V);
3-tooth 10 mm end mill, 35 deg helix; 4900 rpm, f_t = 0.02 mm/tooth,
a_e = 0.1 mm. Mode 1 at 521 Hz. Full parameter list in `experiments/run_all.py`.

## Honesty note

The headline results of the original package could not be reproduced from its own
code and were traced to hard-coded / scaled values. This repository keeps only
what is verifiable and states each effect for what it is. See
[`paper/CORRECTIONS.md`](paper/CORRECTIONS.md).
