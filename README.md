# Active Vibration Control of a Thin-Walled Plate During Milling

**PS-LPV-DR: Position-Scheduled LPV control with Regenerative-targeted
Delayed feedback for chatter suppression in thin-walled plate milling.**

This repository contains the complete model, controller synthesis, stability
analysis, simulation campaign, and manuscript draft for a journal publication
(target: Q1 — *Mechanical Systems and Signal Processing* / *International
Journal of Mechanical Sciences*).

## The idea in one paragraph

Active chatter control of flexible workpieces must cope with dynamics that
change while the tool moves: the modal participation of the milling point
varies along the feed path, and material removal shifts the modal parameters.
The state of the art (Du *et al.*, IJMS 274:109257, 2024) wraps these *known*
variations into norm-bounded uncertainty and pays for it with conservatism —
lower achievable depth of cut and higher control voltage. This work instead
treats the tool position and removal state — both known in real time from the
NC program — as **measured scheduling parameters** of an LPV plant. A
grid-based gain-scheduled H∞ controller is combined with a
spindle-synchronized delayed feedback term targeted directly at the
regenerative mechanism, whose scheduled gain is tuned offline by maximizing
the *closed-loop* critical depth of cut computed by semi-discretization.
Norm-bounded uncertainty is kept only for what is genuinely uncertain:
truncated high-order modes (spillover), cutting-coefficient dispersion, and
modal tolerances.

## Repository layout

```
docs/control_strategy.md        complete mathematical development
docs/literature_positioning.md  state of the art, gap analysis, novelty claims
avc/                            Python reference implementation
  params.py       physical data (identical to the IJMS-2024 experimental rig)
  fem_plate.py    Mindlin plate FEM (cantilever, material-removal geometry)
  piezo.py        surface-bonded patch actuator coupling
  modal.py        modal reduction, LPV model builder
  milling.py      helical multi-tooth cutting coefficients, regenerative model
  controller.py   controller interface (incl. delayed state tap)
  synthesis.py    H-infinity (DGKF), gain scheduling, baselines
  sld.py          semi-discretization stability lobes (open & closed loop)
  delayed_feedback.py  offline k_r tuning on closed-loop lobes
  simulate.py     nonlinear LTV time-domain engine
scripts/          one script per manuscript figure + campaign pipeline
tests/            validation suite (FEM benchmarks, SLD analytic checks, ...)
paper/            manuscript draft (elsarticle)
results/          cached computation artifacts
```

## Reproducing the results

```bash
pip install numpy scipy matplotlib pytest
python3 -m pytest tests -q          # validation suite
python3 scripts/run_all.py          # full campaign + all manuscript figures
```

## Model basis

The plate FEM follows standard Reissner–Mindlin plate elements with selective
reduced integration. The geometry, material, actuator, sensor, tool, and
cutting data replicate the experimental rig of Du *et al.* (2024) so that
every simulated comparison is anchored to published measurements. The MATLAB
plate-FEM study codebase `Plate-FEM` (N. P. V. Khoa) that inspired the FEM
structure is not redistributed here; the Python implementation is independent.
