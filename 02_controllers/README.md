# 02_controllers — Control Algorithms

Two strategies are compared: **LQG** (the benchmark baseline) and **ESO-ADRC**
(modal extended-state-observer active disturbance rejection control), developed
into its adaptive form **A-ESO-ADRC** (cost-supervised design ladder). The earlier
PALF-LQG / A-PALF-LQG learned-feedforward family was removed from the package on
request; `docs/AUDIT_FINDINGS.md` keeps the historical audit record.

## Files

| File | Controller | Key class |
|---|---|---|
| `lqg_controller.py` | Linear Quadratic Gaussian (baseline) | `LQGController` |
| `adrc_controller.py` | Modal ESO-ADRC (fixed) | `ESO_ADRC_Controller` |
| `adrc_controller.py` | Adaptive ESO-ADRC (supervised ladder) | `AdaptiveESO_ADRC_Controller` |
| `adrc_controller.py` | Canonical output LADRC — **negative result**, kept reproducible | `CanonicalLADRC_Controller` |
| `adrc_controller.py` | **P8**: adaptive (FxLMS/AFC) HRC — **negative result**, kept reproducible | `ESO_ADRC_AdaptiveHRC_Controller` |
| `adaptive_id.py` | **P6**: real-time-ID-scheduled controller (re-tunes to identified frequencies per finishing pass) | `IDScheduledController` |
| `predictive_removal.py` | **P7**: material-removal-aware preview-predictive controller (feasibility study) | `PreviewPredictiveController`, `PhysicsCuttingModel` |

## LQG Controller (baseline)

Standard LQG = output-weighted LQR (grid-searched weights) + Kalman observer.

```python
from lqg_controller import LQGController

lqg = LQGController(design_view, dt=5e-5, kalman_V=1e-12, u_max=150.0)
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg.discretize_observer()
x_hat, u = lqg.step(x_hat_prev, u_prev, y_meas)   # inside the Newmark loop
```

## ESO-ADRC Controller

The ADRC idea — estimate a lumped "total disturbance" online and let the law use
it — applied in **modal space**, which is what this plant requires:

```
q̈ = -K q - C q̇ + H u + d(t)          d ∈ R³: per-mode TOTAL disturbance
z = [q̂; q̂̇; d̂] ∈ R⁹                  ESO (Riccati-designed gain, knob σ_d)
u = -K_fb [q̂; q̂̇] - γ·Hᵀd̂/(HᵀH)     same output-weighted LQR construction
                                       as the baseline; γ = matched cancellation
```

`d(t)` absorbs the regenerative force, the feed forcing, spillover of unmodelled
modes, and material-removal stiffness drift — no cutting-force model, no delay
model, no identification at run time. The comparison **LQG vs ESO-ADRC isolates
exactly one ingredient**: replace the plain Kalman filter with a
disturbance-estimating ESO.

**Four documented design findings** (full derivations in the module docstring):

1. **Canonical output LADRC destabilizes this plant** for *every* bandwidth pair:
   the piezo→sensor transfer is non-collocated with alternating modal residues
   (D·H = −0.40/+0.65/−0.19), DC and high-frequency gains of opposite sign ⟹ real
   RHP zeros ⟹ the ŷ̈ = f + b₀u premise is wrong. Kept reproducible in
   `CanonicalLADRC_Controller`.
2. **ESO gain via a scaled Riccati equation** — bandwidth-parametrized pole
   placement of a 9-state observer from one output is numerically hopeless
   (|L| ~ 1e17); the disturbance-noise intensity σ_d is the single bandwidth knob.
3. **Matched cancellation γ does not pay** here (actuator direction only ~19 %
   aligned with the tool-force direction) — the grid selects γ = 0; the benefit
   comes from disturbance-aware state estimation.
4. **Closed-loop effectiveness self-identification is biased** (the periodic
   cutting force correlates with u through the feedback), so no κ-adaptation is
   attempted — consistent with this package's earlier identifiability finding
   that persistent excitation is required.

```python
from adrc_controller import ESO_ADRC_Controller

# The fixed design is CERTIFICATION-SELECTED: smallest worst-case Floquet radius
# over a design-time uncertainty ball (see main_simulation.py). No held-out data.
adrc = ESO_ADRC_Controller(design_view, dt=5e-5, w_q=1e14, w_qd=1e8,
                           sigma_d=1e4, kalman_V=1e-12, u_max=150.0)
z, u = adrc.step(z_prev, u_prev, y_meas)
A_con, B_con_y, K_con = adrc.controller_realization()   # for the monodromy SLD
```

## A-ESO-ADRC Controller (adaptive)

No fixed tuning covers the whole uncertainty range: the closed-loop Floquet map
over a frequency-mismatch ball shows **complementary instability holes** for the
aggressive and robust tunings (waterbed effect — see `fig06_certification`).
A-ESO-ADRC therefore supervises a **ladder of two pre-designed rungs** sharing one
physical observer state (bumpless switching):

- **performance rung** — lowest nominal RMS on the design grid;
- **certified rung** — smallest worst-case monodromy radius over the design ball.

The supervisor uses **measured cost only** (no identification, no probe): a slow
EMA of y² against a running-min quiet level (dwell + hysteresis toggling), and a
fast-EMA **panic** jump to the certified rung with an absolute floor and
**escalating post-panic locks**.

```python
from adrc_controller import AdaptiveESO_ADRC_Controller

aadrc = AdaptiveESO_ADRC_Controller(design_view, dt=5e-5,
                                    rungs=((1e16, 1e8, 3e3),    # performance
                                           (1e14, 1e8, 1e4)),   # certified
                                    perf_rung=0, robust_rung=1)
aadrc.reset_adaptation()          # before every independent run
z, u = aadrc.step(z_prev, u_prev, y_meas)
aadrc.history_rung                # rung trace for figures
```

## Verified comparison (committed code — see docs/REPRODUCED_RESULTS.md)

| Scenario (all held-out) | LQG | ESO-ADRC (certified) | A-ESO-ADRC |
|---|---:|---:|---:|
| S1 nominal | **0.777 µm** | 0.826 µm | 0.783 µm |
| S2 a_p = 0.6 mm | **1.558 µm** | 1.824 µm | 3.41 µm |
| S3 ω−8 % | **0.900 µm** | 20.8 µm (bounded) | 1.123 µm |
| S4 K_T+30 % | **1.013 µm** | 1.078 µm | 1.040 µm |
| static ω−12 % | DIVERGES | **1.140 µm** | 1.71 µm |
| ramp to −12 % during pass | DIVERGES | **0.898 µm** | 1.151 µm |
| ramp to +15 % during pass | **0.682 µm** | 1.276 µm | 1.256 µm |
| piezo effectiveness ×0.25 | 1.241 µm | 1.221 µm | **1.184 µm** |

Honest reading: **inside the fixed-design envelope LQG is the best regulator**
(its Kalman model is correct there); the ESO's value is **architectural
robustness** — it survives drift 4 % beyond the LQG margin because the d̂-states
absorb model error — and the **adaptive ladder removes the fixed designs' failure
modes** (LQG's −12 % divergence, the certified rung's −8 % hole, the performance
rung's a_p = 0.6 divergence) at a modest nominal cost. **A-ESO-ADRC is the only
controller that never diverges** across all scenarios tested.

(Plant carries 5 modes, controllers see 3 — spillover; 10 nm measurement noise;
corrected Eq. 3 forces; Eq. 15 piezo coupling; identical ±150 V clipping;
bit-reproducible seeds.)


## IDScheduledController (P6 — real-time identification)

Re-tunes the controller's internal model to the modal frequencies IDENTIFIED by
the active piezo probe at each finishing-pass boundary (see
`03_analysis/realtime_id.py` and `01_core/material_removal.py`). Frequency-only ID
(mode shapes are held at their pristine values — reshaping is second order,
MAC > 0.999), which matches a true-frequency oracle to 3 decimals.

```python
from adaptive_id import IDScheduledController
ctrl = IDScheduledController('lqg', dt, D_obs3, H_Pe3, zeta3, f_nominal3,
                             w_tooth=w_tooth)
for pass in sequence:
    f_hat = transit_probe_identify(pass.plant, ...)   # unbiased, per pass
    ctrl.retune(f_hat)                                 # rebuild gains
    cut(pass.plant, ctrl)
```

Verified finding (`main_realtime_id.py`): fixed pristine LQG **loses control on 7
of 24 passes** as the wall thins to −15 % (worst-pass RMS 10.9 µm), while the
ID-scheduled LQG stays ≤ 0.072 µm (**151× better at the worst pass**) and matches
the oracle. The robust ESO-ADRC survives the whole sequence WITHOUT ID (0
control-loss passes) — identification and disturbance-observer robustness are
complementary, and `'hrc'`-kind scheduling additionally drops a tooth-harmonic
resonator when an identified mode crosses it (a resonant compensator on a
resonance is hazardous regardless of phase).


## PreviewPredictiveController (P7 — material-removal-aware predictive control)

The requested chain: precise cutting/material-removal model -> plate properties per
step -> predicted vibration -> suppression. The investigation established two physical
walls that make a genuine *per-step* material-removal-aware controller unjustified on
this plant (see the module docstring and `docs/REPRODUCED_RESULTS.md` P7):
1. **Timescale** — the tool takes ~13,605 steps (0.68 s) to cross one FEM mesh column,
   so the plant is constant between crossings; per-step update is over-engineering by
   ~10⁴ (correct cadence = event-driven mesh-crossing).
2. **Regime conflict** — within-pass removal grows with depth (≳0.5 % drift by
   a_p=10 mm, >1 % only at a_p=40 mm), but those deep cuts are open-loop uncontrollable
   (ρ up to ~10¹⁷ ≫ piezo authority); the controllable regime (a_p≲2 mm) has ≲0.2 %
   within-pass removal. Mutually exclusive.

The concrete artifact is the **preview-predictive controller**: the regenerative force
uses the already-known delayed state q(k−n_tau) and the feed force is periodic-known,
so the incoming modal disturbance is *previewable* one tooth period ahead; a
receding-horizon quadratic program pre-empts it. Honest result at a_p = 0.3 mm:
stable, modestly beats LQG (0.68–0.72 vs 0.78 µm) but **dominated ~2× by the existing
HRC (0.39 µm)**, and — being model-based — inherently limited by cutting-model accuracy
(weaker than P6's probe, which measures the truth). Anti-inverse-crime: the
controller's `PhysicsCuttingModel` is mismatched (+15 % KT, +15 % kn, −15 % µc) from
the plant. Driver: `python main_predictive_removal.py`.

## Robust HRC + adaptive HRC (P8 — improving the P5 HRC line)

Driver `main_hrc_robustness.py`. The P5 fixed HRC resonator width `lam` is exposed as a
certification-consistent **robustness/performance knob**. Narrow resonators (lam = 5) give
the deepest nominal notch (0.381 µm) but are fragile: under a −12 % frequency ramp the
standalone HRC nearly diverges (33.6 µm). The **design-ball worst-case closed-loop Floquet
radius decreases monotonically with lam** (1.393 → 1.348), so a wider resonator is provably
more robust by the same certification used for the ESO rung. The **robust HRC (lam = 20)**
costs 7 % nominal (0.407 µm) but controls the −12 % ramp (**0.59 µm**) and +30 % K_T
(0.97 µm), staying fully LTI-certifiable (the fixed `ESO_ADRC_HRC_Controller` simply takes a
larger `lam`). Recommended as the standalone HRC (performance/certified duality); inside the
supervised ladder it is a wash (drift handled by rung switching), so A-ESO-ADRC is unchanged.

`ESO_ADRC_AdaptiveHRC_Controller` is a **documented negative result**: adapting a complex
weight per tooth line by filtered-x LMS (AFC) to re-null the drifted residual is worse than a
well-damped fixed resonator here — this plant's modes sit close to the tooth lines, so modest
drift swings the secondary-path phase past the FxLMS ±90° cone and the low-margin integral
action injects rather than cancels (WORSE under every drift; refuted). Kept reproducible.
