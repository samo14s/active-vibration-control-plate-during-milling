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

## Contribution in one paragraph

The standard way to obtain a *controlled* stability-lobe diagram (SLD) is often
faked (see [`paper/CORRECTIONS.md`](paper/CORRECTIONS.md)); a feedforward or a
damping multiplier is used to "improve" a boundary that only feedback can move.
Here the controller is placed **inside** the Floquet monodromy matrix
(`src/cl_fdm.py`, the *closed-loop full-discretization method*), so the
controlled SLD is genuine. On that basis, the feedback gain is chosen by
**directly maximising the critical depth of cut** under a control-authority
constraint (`src/floquet_synthesis.py`) instead of the usual eigenvalue proxy,
and a **phase-aware feedforward** (`src/twodof_control.py`) is added purely to
reduce forced vibration and peak actuator voltage — a role that is quantified
honestly and shown *not* to move the stability boundary.

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
  cl_fdm.py                *** closed-loop FDM: controller inside the monodromy
  floquet_synthesis.py     *** Floquet-direct feedback gain selection
  twodof_control.py        *** feedback + phase-aware feedforward (2-DOF)
experiments/
  run_all.py               reproduces every result -> results/
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
`python experiments/run_all.py --stage 1` (synthesis), `2` (SLD), `3`
(scenarios), `4` (feedforward role).

## Physical setup

Cantilever AL6061 plate 100 x 80 x 4 mm; QDA60-200.7 piezo patch (d31, +/-150 V);
3-tooth 10 mm end mill, 35 deg helix; 4900 rpm, f_t = 0.02 mm/tooth,
a_e = 0.1 mm. Mode 1 at 521 Hz. Full parameter list in `experiments/run_all.py`.

## Honesty note

The headline results of the original package could not be reproduced from its own
code and were traced to hard-coded / scaled values. This repository keeps only
what is verifiable and states each effect for what it is. See
[`paper/CORRECTIONS.md`](paper/CORRECTIONS.md).
