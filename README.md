# Adaptive Control for High-Speed Milling of Thin-Walled Components — Simulation Re-run

**إعادة محاكاة** لدراسة التحكم التكيفي في التفريز عالي السرعة للجدران الرقيقة —
نُفِّذ النموذج الفيزيائي والمتحكم كاملين من معادلات الورقة، وأُعيد إنتاج
المقارنة التجريبية (تقليدي / SSV / تكيفي) بخمس تجارب لكل استراتيجية.

Full re-implementation and re-run of the simulation/control study in:

> Z. Wang, J. Sun, Q. Sun, *"Adaptive Control Strategy for High-Speed
> Milling of Thin-Walled Components"*, ASME Open Journal of Engineering,
> Vol. 5, 051006 (2026). DOI [10.1115/1.4070743](https://doi.org/10.1115/1.4070743)

## Headline results (this re-run vs the paper)

| Quantity | This simulation | Paper |
|---|---|---|
| MRR improvement, adaptive vs conventional | **+46.8 %** | +44.2 % |
| SSV baseline MRR improvement | +32.8 % | +18 % |
| Total machining time | **−32.6 %** | −29.6 % |
| Specific cutting energy | **−28.4 %** | −25.8 % |
| Adaptive RMS vibration (AC, high-passed basis) | < 25 μm (max ≈ 23) | < 25 μm |
| Conventional vibration | ~85 μm RMS (≈120 μm peaks) | peaks > 150 μm |
| Adaptive Ra | 0.80–1.24 μm | 0.9–1.3 μm |
| Conventional Ra | 0.8–3.3 μm | 0.8–4.5 μm |
| Ra σ reduction | −84 % | −87.9 % |
| Wall-thickness deviation | ±0.019 / ±0.068 mm | ±0.02 / ±0.08 mm |
| Modal-model tracking error (median / p90) | 1.2 % / 2.5 % | 3.2 % (identification) |

See `results/REPORT.md` for the full table, figures and the list of known
deviations — most notably: the SSV baseline over-performs on MRR and
under-performs on quality relative to the paper (SSV effectiveness is
plant-specific), the tracking-error headline scores the fused Eq. (6)+RLS
model rather than raw RLS, and the Eq. (23) cross-part learning yields
only ~0–2 % here because the binding constraint in this plant is the
forced-response cap, not a conservative stability boundary.

The implementation was adversarially reviewed (multi-agent verification
pass over force/dynamics, stability, control and metrics); all confirmed
findings — DC contamination of the vibration channel, an Eq. (17)
violation by the trust relaxation, a frozen-speed process-damping term in
the ZOA, lobe-envelope edge-binning, coherence phase-slip, and a circular
identification-accuracy headline — are fixed or honestly documented.

## What is modelled

* **Cutting forces** (Eqs. 1–4): mechanistic tangential/radial model with
  regenerative chip thickness, engagement window, tooth jump-out
  nonlinearity, and per-tooth runout.
* **Structural dynamics** (Eq. 5): modal model — two thin-wall bending
  modes (compliant normal to the wall) + two tool modes; integrated at
  100 kHz with a full delay-history buffer (exact under spindle-speed
  variation).
* **Time-varying dynamics** (Eqs. 6–7, 15): natural frequencies fall and
  damping rises with the removed-volume fraction, so the offline-optimal
  fixed parameters (12 000 rpm, 4 mm) destabilise mid-pass.
* **Stability lobes** (Eqs. 8–13): zero-order Altintas–Budak solution,
  both a reference implementation and a fully vectorised version fast
  enough for the controller's online use; optional process damping.
* **Adaptive controller** (Sec. 3): RLS identification with
  spindle-harmonic notching (Sec. 3.2) fused with Eq. (6) dead-reckoning,
  online lobe computation, hierarchical multiparameter coordination
  (Eq. 22): spindle-speed gradient + strategic lobe relocation
  (Eqs. 18–19, 25), feed-rate MPC (Eq. 20), depth-of-cut vibration
  regulation (Eq. 21) under the stability constraint (Eq. 17), min-max
  robust forced-response cap (Eq. 24), actuator limits (Eq. 26), and
  chatter detection on the nonsynchronous residual with fast retraction.
* **Baselines** (Sec. 4.1): fixed parameters with operator-style manual
  reductions on visible chatter, and ±10 % / 5 Hz SSV (slew-limited).
* **Learning across parts** (Eq. 23): Gaussian-process correction of the
  stability boundary, demonstrated in `run_learning_demo.py`.

The wall-thinning pass is **time-compressed** (15 cm³ of stock in tens of
seconds instead of a full pass): the modal parameters sweep the same
excursion as the physical 10–15 mm → 1–3 mm walls, while the regenerative
dynamics (ms scale) remain fully resolved — the time-scale separation the
paper itself relies on for quasi-static model updating.

## Article package integration (this repository's own study)

`article_package/` contains the simulation package of the repository's
article — **LQG vs DARC-MPC active vibration control of a cantilever
AL6061 plate during peripheral milling** (plant modelling per Nasiri &
Moradi, MSSP 224 (2025) 112198: Von Kármán/Galerkin nonlinear plate,
3-tooth D10 end mill, piezo QDA60 patch, 4900 rpm, aₑ = 0.1 mm).

Re-run results (this environment):

| Quantity | Re-run | Package README claim |
|---|---|---|
| LQG y_RMS, nominal 0.5 s scenario | 0.605 μm | ≈ 0.63 μm |
| DARC-MPC gain over LQG (4 scenarios) | +55.2 % avg | ≈ 55 % |
| Full 20.4 s path: LQG / DARC / DARC-MPC | 0.471 / 0.286 / 0.267 μm | DARC-MPC best |
| ap crit @4900 rpm: open-loop → LQG → DARC | 0.10 → 2.86 → 4.0 mm | idem |

**Model alignment** (`milling_sim/article_plant.py`,
`run_article_alignment.py`): the `milling_sim` framework was mapped onto
the article plant (modal set, force coefficients `Kt_eff = k₂K_T`,
`Kr_eff = k₁/k₂`, scalar regenerative NDDE), giving an independent
frequency-domain (ZOA-style) stability boundary cross-checked against the
package's own time-domain Floquet FDM — two methods, one plant.  They
agree on lobe positions and on the safety-critical valleys (including
`ap_lim ≈ 0.08–0.10 mm` at the nominal point → **the article's operating
point is open-loop unstable, exactly the article's premise**); the
averaged-coefficient method overestimates lobe peaks, as expected for an
ae/D = 1 % interrupted cut.  The adaptive spindle-speed layer of the ASME
re-run composes with the article's active control: speed selection alone
lifts the open-loop limit by an order of magnitude (best lobe ≈ 5200 rpm),
and LQG multiplies it again.  See
`results/article_alignment/alignment_report.md`.

## Run it

```bash
pip install -r requirements.txt

python3 run_simulation.py            # full campaign: 3 strategies x 5 trials (~3 min with numba)
python3 run_simulation.py --quick    # 2 trials, faster
python3 run_learning_demo.py         # Eq. (23) cross-part learning demo
python3 -m pytest tests/ -q          # physics sanity tests
```

Outputs land in `results/`: `fig3_performance.png`, `fig4_quality.png`,
`fig_stability_lobes.png`, `fig_identification.png`, `fig_spectrum.png`,
`results.json`, `REPORT.md`.

## Repository layout

```
milling_sim/
  parameters.py       # all paper parameters (Table 1, Sec. 4.1) + calibrated modal set
  engine.py           # 100 kHz time-domain kernel (numba), Eqs. (1)-(5), (13)
  stability.py        # ZOA lobes, Eqs. (8)-(12) (+ fast vectorised envelope)
  identification.py   # RLS AR(2) + spindle-harmonic notch, Sec. 3.1-3.2
  control.py          # adaptive controller (Sec. 3) + both baselines
  learning.py         # GP stability-boundary learning, Eq. (23)
  metrics.py          # RMS, Ra correlation, coherence, specific energy
  plots.py            # publication-style figures
  runner.py           # campaign orchestration (n = 5 trials per strategy)
run_simulation.py     # main entry point
run_learning_demo.py  # cross-part learning demo
tests/test_physics.py # sanity tests (directional factors, decay, lobes, RLS)
```

An equation-by-equation map from the paper to the code is included at the
end of `results/REPORT.md`.
