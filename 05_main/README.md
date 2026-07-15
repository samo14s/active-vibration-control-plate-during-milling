# 05_main — Main Simulation Scripts

Main entry points for running the simulation. **Authoritative numbers** come from
these drivers and are logged in `docs/REPRODUCED_RESULTS.md`.

## Files

| File | Description | Output |
|---|---|---|
| `main_simulation.py` | LQG vs ESO-ADRC vs A-ESO-ADRC (design grids + certification + 4 held-out scenarios + worst-position closed-loop SLD) | Console + figs |
| `main_robustness_mc.py` | Monte-Carlo robustness (50 samples × 3 controllers, divergence reported) | Console + fig |
| `main_adaptive_removal.py` | Drift / stress benchmark: material-removal drift, static mismatch, actuator-effectiveness loss | Console + fig |
| `main_realtime_id.py` | **P6**: real-time identification over an accurate material-removal finishing sequence | Console + fig |
| `main_predictive_removal.py` | **P7**: material-removal-aware predictive control — honest feasibility study (two walls + controller comparison) | Console + fig |
| `main_hrc_robustness.py` | **P8**: improving the P5 HRC — certification-consistent robust resonator (lam sweep + drift) + adaptive-HRC negative result + benign MC | Console + fig |
| `main_realistic_piezo.py` | LQG with realistic piezo non-linearities | Console + figs |

## main_simulation.py

The **primary script**. Protocol (see `docs/CONTRIBUTION.md` §6):

1. Builds the plate FEM model (Q4 × 30×24) and a **full-order 5-mode plant**; the
   controllers are designed on the **first 3 modes** (spillover — no inverse crime),
   with 10 nm measurement noise and identical ±150 V clipping.
2. **LQG baseline**: output-weighted LQR + Kalman, weight grid search on the
   nominal design model.
3. **ESO-ADRC design grid** over (w_q, w_qd, σ_d) — same LQR machinery — with TWO
   design-time selection criteria (no held-out data enters either):
   - *performance* design: lowest nominal RMS (short sim on the nominal plant);
   - *certified* design: smallest worst-case coupled-monodromy Floquet radius over
     a design ball (mismatch −12…+15 % at a_p = 0.3 mm, plus a_p = 0.6 mm at 0 %;
     tool at x = 0, L/2). The certified design is the fixed "ESO-ADRC" entry.
4. **A-ESO-ADRC** = the two designs as rungs of the supervised ladder.
5. Runs **4 held-out scenarios** with all three controllers:
   - **S1 Nominal**: a_p = 0.3 mm — **S2 Aggressive**: a_p = 0.6 mm —
   - **S3 Uncertainty**: ω − 8 % — **S4 High K_T**: +30 %
6. Draws the **certification map** (Floquet radius vs mismatch for LQG and both
   rungs) and the **worst-of-3-positions closed-loop SLD** (OL / LQG / ESO-ADRC via
   the generic monodromy).

### Verified output (committed code, ~4.5 min — full log in docs/REPRODUCED_RESULTS.md)

```
  Scénario                    LQG        ESO-ADRC    A-ESO-ADRC
  S1 - Nominal article        0.7765     0.8256      0.7826
  S2 - Aggressive ap=0.6mm    1.5580     1.8237      3.4145
  S3 - Uncertainty ω-8%       0.9001     20.7898*    1.1233     (* bounded hole)
  S4 - High K_T +30%          1.0127     1.0784      1.0402

  ESO-ADRC performance : (1e16, 1e8, 3e3)  rms_nom=0.798µm, worst-ρ=1.165
  ESO-ADRC certified   : (1e14, 1e8, 1e4)  rms_nom=0.838µm, worst-ρ=1.046

  SLD @4900 RPM (worst of 3 positions): OL 0.100 mm (= article experiment),
  LQG 1.075 mm (10.8×), ESO-ADRC certified 0.913 mm (9.1×)
```

## main_adaptive_removal.py

Drift / stress benchmark (frequency schedule Kp·s(t)², Cp·s(t) inside the Newmark
solver; static perturbed plants; actuator-effectiveness scaling):

```
  case                    LQG           ESO-ADRC      A-ESO-ADRC
  D0 no drift             0.7765        0.8256        0.7826
  D1 ramp +15%            0.6815        1.2757        1.2559
  D2 ramp -12%            DIV           0.8976        1.1509
  D3 static -12%          DIV           1.1397        1.7076
  D4 effectiveness x0.25  1.2413        1.2213        1.1839
```

The fixed LQG diverges 4 % beyond its ~−9 % margin; the ESO's disturbance states
absorb the drift; A-ESO-ADRC never diverges anywhere and its rung trace shows the
supervisor escalating exactly when the drift crosses the performance rung's
comfort zone.

## main_robustness_mc.py

Monte-Carlo over ±15 % cutting constants, ±3 % modal frequencies, ±20 % damping
(the LQG-safe neighbourhood): all three controllers converge 50/50; medians LQG
0.788 µm, ESO-ADRC 0.850 µm, A-ESO-ADRC 0.886 µm — **LQG wins inside its
envelope**, consistently with the main-table story; the ESO family's advantage
lives beyond that envelope (see the drift benchmark).

## main_realistic_piezo.py

LQG with a realistic piezo model (saturation, slew rate, amplifier bandwidth,
hysteresis, sensor noise/delay). Uses the corrected Eq. (3) constants; it is an
LQG-only actuator study.

## Tips

1. **First run / authoritative numbers**: `main_simulation.py`.
2. **The adaptive story** (the reason A-ESO-ADRC exists): `main_adaptive_removal.py`.
3. **Figures**: authoritative ones are produced by the drivers themselves
   (`figs_lqg_vs_adrc/`); geometry and styled-SLD generators live in `04_figures/`.
