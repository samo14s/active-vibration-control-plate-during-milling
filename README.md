# Active Vibration Control of a Thin-Walled Plate During Milling

**Position-Scheduled LPV chatter control with certified stability lobes —
and a certified assessment of regenerative-targeted delayed feedback.**

This repository contains the complete model, controller synthesis, stability
analysis, simulation campaign, and manuscript draft for a journal publication
(target: Q1 — *Journal of Sound and Vibration*; the study is
simulation-based but anchored at every link to the published
experimental record of the benchmark rig, including reproduction of its
measured open-loop stability pattern under the published force
calibration — see `paper/cover_letter.md`).

**Headline results** (12-mode evaluation model, 4.9 krpm, all three
deployment certificates passing):

| Metric | no control | delayed PD | best frozen H∞ | **PS-LPV (proposed)** |
|---|---|---|---|---|
| worst-position critical depth | 0.229 mm | 0.161 mm* | 0.703 mm | **2.985 mm** |
| band-worst (2–10 krpm) | 0.157 mm | 0 (destabilizes) | 0.517 mm | **1.971 mm** |
| hard condition (x=95 mm, ap=2 mm) | unstable | unstable | 15.5 µm @ 84.5 V | **6.5 µm @ 41.0 V** |
| Monte-Carlo median (40 samples) | — | — | 0.28 mm | **2.17 mm** |

\* tuned at the reference speed; destabilizes at other speeds in the band.

**FRF-level experimental anchoring**: the measured receptance and piezo
transfer function of the reference rig (digitized, `data/`) are matched
by the model across 5 Hz-5 kHz — resonances within 1.6-3.8%,
antiresonance placements reproduced, and the actuation chain absolute
gain within 35% with handbook piezo constants (fig13).

A certified negative result completes the picture: the lobe-maximizing
tuning of the regenerative-targeted delayed feedback gain returns **zero**
at every scheduling point once the scheduled H∞ loop is active, and gain
staleness in the *removal* axis destabilizes a point-designed loop outright
after 0.5–1 mm of edge recession — both findings are in the manuscript.

## Next paper (in progress): SatCERT

`docs/new_strategy_satcert_outline.md` targets the verified gap
(dossiers in `docs/gap_scouting/`): **certified regional stability under
actuator saturation**. WP1–3 are executed and WP4 is demonstrated
in-family with causal attribution (the external re-simulation of the
published SDOF island configuration is deferred to the manuscript
stage) (`avc/satcert.py`,
`scripts/satcert_campaign.py`, `results/satcert_campaign.json`,
foundational text with filled numbers in
`docs/paper2_foundational_text_final.md`):

- **Implementation-exact certificate**: sampled-data period lifting at
  the 50 kHz control rate (ZOH controller, one-period computation
  delay, ±150 V deadzone at the DAC); the maximal saturation-free
  admissible set (Gilbert–Tan O∞, phase-resolved headroom) gives the
  largest surface-step h_max certified for both signs — closed-form, no
  SDP, and **validated to 0.2–2 % against the nonlinear simulator's
  measured clip onset**.
- **Headline finding**: certification overturns the linear ranking —
  PS-LPV's 3.2× linear worst-position advantage over the best frozen
  design becomes 2.7× at 1 µm certified tolerance and **inverts to
  0.79× at 20 µm**; forced-orbit saturation zones appear only for the
  high-authority scheduled design (the Ozsoy-type island mechanism,
  in-family).
- At the hard condition (x = 95 mm, a_p = 2 mm) the frozen design
  operates 14.7 % clipped vs 0.28 % for PS-LPV, which cuts vibration
  by 58 % at 49 % of the voltage (RMS).
- **Two saturation islands demonstrated with causal attribution**
  (`scripts/satcert_islands.py`): at linearly stable points (ρ ≈ 0.955)
  a −38/−50 µm surface step triggers chatter growth at 99 % clip duty,
  and the SAME step decays once the ±150 V bound is lifted. At the four
  points tested, islands appeared only for the high-authority scheduled
  controller; the frozen design survived ±500 µm steps. Figure:
  `docs/figures/satcert_campaign.png`.

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
