# Active vibration control of a thin-walled plate during milling

**Position-scheduled harmonic LQG (PSH-LQG): simulation framework, baselines,
and manuscript.**

This repository contains a complete, reproducible research codebase for
active vibration suppression in thin-walled plate milling with a fixed
surface-bonded piezoelectric patch actuator and a single fixed
accelerometer, together with the draft manuscript
(`paper/main.tex`) targeting a Q1 journal (first-choice venue:
*International Journal of Mechanical Sciences*).

The benchmark configuration reproduces the published experimental setup of
Wang, Song & Liu, *Int J Adv Manuf Technol* (2019),
[doi:10.1007/s00170-019-04493-5](https://doi.org/10.1007/s00170-019-04493-5)
(plate geometry/material, patch and accelerometer placement, drive-chain
gains and saturation, sampling rate, cutting conditions), and the actuator
authority is calibrated to their measured static deflections.

## The proposed control strategy

**PSH-LQG — position-scheduled harmonic LQG** exploits the structural fact
that the plate–actuator–sensor plant is LTI while the *disturbance input
map* and the *performance output map* travel with the cutter:

- **Composite design model** — 6 retained modes + the analog anti-aliasing
  filter discretized jointly (ZOH of the composite system), + 1-sample
  computation delay, + internal-model phasor states of the periodic
  milling force at the tooth-passing frequency (TPF) and 4 harmonics + DC,
  coupled through an **exact sinusoid discretization** (`avcp/discrete.py`).
- **Kalman predictor** with position-scheduled disturbance couplings and
  gains (steady-state Riccati per grid point; Riccati-difference iteration
  where the on-circle internal model defeats standard DARE solvers),
  driven by the *saturated* input (anti-windup), with a leakage factor
  ρ = 0.99995 trading an O(1%) cancellation bias for a contraction budget.
- **Position-scheduled LQR** weighting the deflection at the *moving
  cutting point* (the rigorous form of the VPD intuition), with the
  voltage weight auto-tuned against the **full 20-mode model** (spillover
  bound).
- **Scheduled harmonic cancellation** — regularized model inversion of
  the estimated force phasors at the cutting point, computed on the
  LQR-closed design loop with an absolute per-harmonic regularization
  floor, spindle-synchronized, plus a broadband random-walk force state
  for sub-harmonic regenerative content.
- **Robust spillover masking** — verification-driven per-harmonic gain
  backoff checked against a family of perturbed full-order models
  (±3 % modal frequency, −30 % damping); the LQR weight auto-tuner
  requires family feasibility, so authority is bounded by robust — not
  nominal — stability.

Baselines: an optimally tuned constant PD (CPD) and a faithful,
safeguarded re-derivation of the reference paper's time-space varying PD
(VPD) — including the finding that its quintic gain smoothing can bridge
over local stability dips and must be re-verified after fitting.  Both
baselines are tuned with nonlinear simulation in the selection loop
(frequency-domain surrogates rank candidates incorrectly in the
finite-amplitude regenerative regime).

**Benchmark note.** The corrected force projection
(dFn = dFt sinφ − dFr cosφ) leaves the reference paper's exact feed
nearly quiescent, so the control benchmark uses a declared
semi-finishing condition (fz = 51 µm/tooth, ae = 0.3 mm, runout 2.5 µm)
that reproduces the experimentally reported vibration magnitudes; the
reference condition is retained for model validation.

## Repository layout

```
src/avcp/            the library
  params.py          all physical/process parameters (traceable to the paper)
  fem.py             ACM Kirchhoff plate FEM, piezo coupling, point ops
  modal.py           modal reduction, path scheduling data, FRFs
  milling.py         milling force, multi-delay surface-memory regeneration,
                     ZOA directional factor, Nyquist a_crit
  discrete.py        composite (modes+analog filter) ZOH discretization,
                     exact sinusoid coupling
  simulate.py        50 kHz-substep nonlinear closed-loop simulator
  controllers.py     CPD/VPD tuning, PSH-LQG design (KF/LQR/inversion,
                     spillover mask, r_u autotuning)
  stability.py       frozen closed-loop analysis, closed-loop FRFs
  metrics.py         pass metrics
scripts/
  run_campaign.py    the full simulation campaign (stages, resumable)
  make_figures.py    all manuscript figures from saved results
  fill_numbers.py    writes paper/numbers.tex from the results
tests/               pytest validation suite (FEM vs analytical, piezo
                     statics vs the paper, ZOA vs time domain, ...)
paper/               manuscript (LaTeX) + references
results/, figures/   campaign outputs (regenerable)
```

## Reproducing everything

```bash
pip install -e .[dev]
pytest tests/ -q                    # validation suite
python scripts/run_campaign.py      # ~30-45 min, resumable by stage:
                                    #   run_campaign.py tuning passes sld ...
python scripts/fill_numbers.py
python scripts/make_figures.py
cd paper && latexmk -pdf main.tex   # if TeX is available
```

## Key validation anchors

- Cantilever-plate frequencies within 1% of Leissa's coefficients; mesh
  convergence < 0.5%.
- Static actuator deflection at 1200 V calibrated to the paper's measured
  ≈ 0.56 mm (authority factor 6.8, identified).
- Regenerative implementation vs. direct Nyquist criterion: critical
  depth agreement within ~6% at half immersion.
- Supercritical cutting bounded by the multi-delay surface-memory
  mechanism at physically plausible amplitudes (tens of µm), matching the
  phenomenology reported experimentally in the reference paper.

## Status / roadmap

- [x] Model, controllers, campaign, figures, manuscript draft
- [ ] Complete two bibliography entries flagged `note = {... to be completed}`
- [ ] Experimental validation (hardware campaign) — required by most Q1
      venues; the simulation is designed to transfer (all hardware
      parameters from the reference experiment)
