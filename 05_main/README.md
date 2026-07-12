# 05_main — Main Simulation Scripts

Entry points of the study. All scripts follow the **honest evaluation
protocol** (root README): nominal-model controller design (no oracle),
train/validation/evaluation noise seeds disjoint, identical LQG base for
LQG and DARC, closed-loop Floquet SLD.

| File | Study | Runtime |
|---|---|---:|
| `main_simulation.py` | LQG vs DARC : 4 scenarios + SLD + sensor sweep | ~5 min |
| `main_imc_baseline.py` | LQG vs IMC-LQG vs DARC-FF (internal-model baseline) | ~30 s |
| `main_fullpath_comparison.py` | 20.4 s full feed pass — held-out NN evaluation | ~10 min |
| `main_realistic_piezo.py` | LQG under realistic actuator/sensor model | ~1 min |

## main_simulation.py

1. Analytical Galerkin plate (frequencies calibrated to the FEM anchors)
2. Piezo patch (QDA60-200.7, 20×60 mm), corner displacement sensor
3. Nonlinear cutting coefficients — REAL set drives the plant, NOMINAL set
   is all the controllers ever see
4. LQG baseline (w_q=1e14, w_qd=1e8 — the same base as inside DARC)
5. DARC: inverse-model feedforward from the NOMINAL model +
   NN residual (train seeds 100+, validation seed 200, evaluation seed 1)
6. 4 scenarios (S1 nominal / S2 a_p=0.6 mm / S3 ω−15 % / S4 K_T+30 %
   with K_T unknown to the controller)
7. Closed-loop Floquet SLD (observer + discrete feedback in the monodromy)
8. Sensor-noise robustness sweep (0 → 2 µm RMS)

Measured output (bit-exact reproducible, seeds fixed):

```
S1  LQG 0.6052 → DARC 0.2937 µm  (+51.5 %)
S2  LQG 1.2057 → DARC 0.5657 µm  (+53.1 %)
S3  LQG 0.9234 → DARC 0.3340 µm  (+63.8 %)
S4  LQG 0.7881 → DARC 0.4578 µm  (+41.9 %)   ← honest robustness number
AVG LQG 0.8806 → DARC 0.4128 µm  (+53.1 %)
SLD @4900 RPM : OL 0.100 mm | LQG = DARC 2.375 mm (23.8×)
```

## main_imc_baseline.py

The reviewer question, answered in code: with the spindle period known
(same encoder assumption as DARC), an internal-model LQG rejects the
tooth-passing harmonics by FEEDBACK, with **no cutting-force model**:

```
              LQG      IMC-LQG   DARC-FF
S1           0.605     0.223     0.363   µm
S2           1.206     0.475     0.728   µm
S3 (ω−15%)   0.923    15.755 ⚠   0.592   µm
S4 (K_T ?)   0.788     0.293     0.536   µm
```

No architecture dominates: IMC-LQG is strongest nominally and under unknown
K_T, but destabilizes under −15 % structural detuning (mis-phased inversion
near resonance, voltage saturation); DARC-FF degrades gracefully instead.
Includes a sensor-noise sweep showing the structural trade-off (IMC depends
on the displacement sensor; DARC-FF on its cutting model and phase lock).

## main_fullpath_comparison.py

Held-out evaluation: NN trained on one mid-path 0.5 s segment, evaluated on
the whole 20.4 s pass.

```
LQG      0.4710 µm
DARC-FF  0.2855 µm  (+39.4 %)
DARC     0.2662 µm  (+43.5 %)   ← the NN's honest marginal gain is ~+4 pts
```

## main_realistic_piezo.py

LQG with the realistic actuator/sensor pipeline (saturation, slew, 5 kHz
amplifier lag, linear material phase-lag, 0.1 µm noise + 50 µs delay).
After the sensor-delay off-by-one fix:

```
Open loop            : 252 µm
LQG + ideal piezo    : 2.49 µm  (99.01 %)
LQG + realistic piezo: 2.58 µm  (98.98 %, degradation 0.04 %, u_max 13.2 V)
```

⚠ The loop's delay margin is thin: stable at 50 µs sensor delay, divergent
at 100 µs — budget the total loop latency in any deployment discussion.
DARC under this realistic model is an open item (P1).

## Tips

1. Run everything from a flat `workspace/` (`python setup_workspace.py`).
2. Never quote open-loop chatter AMPLITUDES (clamp-limited); only the
   stability boundary is quantitative.
3. If an SLD boundary does not cross ρ=1 inside the grid, report the grid
   top as a lower bound.
