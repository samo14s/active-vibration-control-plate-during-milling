# Simulation re-run report

Re-implementation and re-run of the study in Wang, Sun & Sun,
"Adaptive Control Strategy for High-Speed Milling of Thin-Walled
Components", ASME Open J. Eng. 5, 051006 (2026), DOI 10.1115/1.4070743.

Campaign: 3 strategies x 5 trials, time-compressed wall-thinning pass (15 cm3 stock, 100 kHz time-domain integration, 100 Hz control).

## Headline comparison

| Quantity | This simulation | Paper |
|---|---|---|
| MRR improvement (adaptive vs conventional) | +46.8% | +44.2% |
| MRR improvement (SSV vs conventional) | +32.8% | +18% |
| Total machining time | -32.6% | -29.6% |
| Specific cutting energy | -28.4% | -25.8% |
| Modal-model tracking error (median / p90) | 1.2% / 2.5% | 3.2% (identification accuracy) |

## Per-strategy detail (means over trials; paper values in the last two columns)

| Metric | Conventional (sim) | Adaptive (sim) | Conventional (paper) | Adaptive (paper) |
|---|---|---|---|---|
| Average MRR (cm3/min) | 16.3 | 23.9 | 17.2 | 24.8 |
| Peak MRR (cm3/min) | 21.6 | 36.3 | 20.0 | 32.5 |
| Max RMS vibration (um) | 85 | 23 | peaks > 150* | < 25 |
| Time above 25 um (%) | 37.9 | 0.0 | - | ~0 |
| Ra min (um) | 0.80 | 0.80 | 0.8 | 0.9 |
| Ra max (um) | 3.30 | 1.24 | 4.5 | 1.3 |
| Ra sigma (um) | 0.62 | 0.10 | 1.24 | 0.15 |
| Thickness deviation (mm) | 0.068 | 0.019 | ±0.08 | ±0.02 |
| Specific energy (W·min/cm3) | 18.7 | 13.4 | 3.1* | 2.3* |

\* Notes on scale: the paper's Fig. 3(a) reports RMS peaks exceeding 150 um for conventional cutting; our conventional runs saturate near 85 um RMS (~121 um peak amplitude), set by the tooth jump-out limit cycle.  The absolute specific energy in the mechanistic force model (~19 W·min/cm3, i.e. ~0.8 J/mm3) matches aluminium machining physics; the paper's absolute values (3.1 -> 2.3) are on a different accounting basis, so the *relative* reduction is the comparable quantity.

## Simulated machining times (s, compressed pass)

| Strategy | mean | trials |
|---|---|---|
| conventional | 56.0 | 51.6, 52.7, 54.6, 68.2, 52.6 |
| ssv | 41.7 | 41.7, 41.7, 41.7, 41.7, 41.7 |
| adaptive | 37.7 | 36.4, 37.6, 37.1, 38.9, 38.6 |

## Figures

* `fig3_performance.png` - vibration / MRR / spindle speed vs time (paper Fig. 3)
* `fig4_quality.png` - roughness and wall-thickness comparison (paper Fig. 4)
* `fig_stability_lobes.png` - boundary evolution with material removal (Eqs. (6)-(12)) and the adaptive trajectory
* `fig_identification.png` - real-time wall-mode tracking
* `fig_spectrum.png` - harmonic vs nonharmonic (chatter) force content

## Known deviations from the paper

1. **SSV baseline** improves MRR by more than the paper's +18% here (no operator reductions are triggered because its chatter bursts stay short), while its surface/thickness quality is worse than the paper reports.  SSV effectiveness is highly plant-specific; our calibrated wall dynamics sit where 5 Hz modulation only fragments, rather than suppresses, chatter.
2. **Chatter frequency band**: our calibration places the wall mode at 1580 Hz falling to ~1050 Hz across the pass, so chatter appears at 1100-1500 Hz vs the paper's 800-1200 Hz; the paper's physical walls are larger.  The stability mechanics (lobe migration through the fixed operating point) are identical.
3. The economics rows of the paper's Table 2 (cost/part, tooling cost, ROI) and yield/tool-life statistics require shop-floor data and are outside the physics simulation.
4. **Identification**: the tracking-error headline scores the controller's fused modal model - Eq. (6) dead-reckoning from the commanded material removal, refined by the RLS estimate when it is fresh and consistent (the notch and the mode collide near resonant speeds, so raw RLS goes stale there).  The plant's Eq. (6)-(7) sensitivities are scattered per workpiece while the controller knows only the nominal calibration, so the dead-reckoning alone would drift by a few percent over the pass.
5. **Cross-part learning** (`run_learning_demo.py`): the GP mechanism of Eq. (23) is implemented, but the realised gain in our plant is ~0-2% (within trial scatter) versus the paper's +8%: the binding constraint here is usually the forced-response cap, which leaves little boundary conservatism for the GP to reclaim.

## Equation-to-code map

| Paper | Implementation |
|---|---|
| Eqs. (1)-(4) cutting forces | `milling_sim/engine.py` (`_integrate_interval`) |
| Eq. (5) modal dynamics | `milling_sim/engine.py` |
| Eqs. (6)-(7) parameter evolution | `milling_sim/parameters.py` (`Mode.wn`, `Mode.zeta`) |
| Eqs. (8)-(12) stability lobes | `milling_sim/stability.py` |
| Eq. (13) process damping | `engine.py` + `stability.py` |
| Eq. (15) stiffness update concept | modal update per control period (`MillingSimulator._update_modal_arrays`) |
| Eq. (16) objective | `milling_sim/control.py` (`_achievable_q`, weights in `ControllerParams`) |
| Eq. (17) stability constraint | `AdaptiveController.observe` |
| Eqs. (18)-(19) speed adaptation | gradient + relocation loop |
| Eq. (20) feed MPC | `AdaptiveController._solve_feed_mpc` |
| Eq. (21) depth adaptation | slow loop in `observe` |
| Eq. (22) coordination | hierarchical structure of `observe` |
| Eq. (23) GP learning | `milling_sim/learning.py` |
| Eq. (24) min-max robustness | worst-case fn band in `_ap_forced_mm` |
| Eq. (25) gain scheduling | `alpha_n` in `observe` |
| Eq. (26) actuator limits | slew clamps in `observe` |
| RLS identification (Sec. 3.1) | `milling_sim/identification.py` |
