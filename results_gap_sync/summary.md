# Spindle-speed uncertainty experiment — summary

Steady-state window: t > 0.15 s (excludes the mechanical transient AND the v4 lock-in, measured as t_lock below); full-record RMS is reported alongside. Gains are RMS reduction vs the LQG baseline of the SAME scenario.

| Scenario | ap (mm) | δ speed | LQG (µm) | v3 (µm) | v3 gain | v4 (µm) | v4 gain | v4 conf | v4 t_lock (s) | u_rms L/v3/v4 (V) | full-RMS L/v3/v4 (µm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1  ap=0.3  δ=0% | 0.3 | 0.0 | 0.5364 | 0.5116 | +4.6% | 0.5114 | +4.7% | 0.99 | 0.157 | 3.67/3.73/3.72 | 0.5392/0.5142/0.5185 |
| A2  ap=0.3  δ=+1.23% | 0.3 | 1.23 | 0.5485 | 0.5489 | -0.1% | 0.5228 | +4.7% | 0.99 | 0.160 | 3.79/4.04/3.83 | 0.5523/0.5535/0.5311 |
| A3  ap=0.3  δ=+2.50% | 0.3 | 2.5 | 0.5419 | 0.5378 | +0.8% | 0.5116 | +5.6% | 0.98 | 0.167 | 3.76/3.98/3.78 | 0.5452/0.5422/0.5210 |
| A4  ap=0.3  δ=-1.20% | 0.3 | -1.2 | 0.5301 | 0.5312 | -0.2% | 0.5049 | +4.8% | 0.99 | 0.155 | 3.63/3.90/3.68 | 0.5336/0.5311/0.5125 |
| A5  ap=0.6  δ=0% | 0.6 | 0.0 | 1.0568 | 1.0053 | +4.9% | 1.0048 | +4.9% | 0.99 | 0.155 | 7.15/7.28/7.26 | 1.0684/1.0151/1.0254 |
| A6  ap=0.6  δ=+2.50% | 0.6 | 2.5 | 1.0560 | 1.0516 | +0.4% | 0.9938 | +5.9% | 0.98 | 0.168 | 7.27/7.83/7.34 | 1.0675/1.0645/1.0183 |
| B   ap=0.3  SSV ±1% @2Hz | 0.3 | ±1 sin | 0.5271 | 0.5286 | -0.3% | 0.5000 | +5.1% | 0.99 | 0.159 | 3.62/3.76/3.66 | 0.5339/0.5326/0.5094 |
| B2  ap=0.3  SSV ∓1% @2Hz | 0.3 | ∓1 sin | 0.5338 | 0.5217 | +2.3% | 0.5067 | +5.1% | 0.99 | 0.155 | 3.67/3.96/3.70 | 0.5327/0.5185/0.5080 |
| C   ap=0.3  δ=+2.5%  T=4s | 0.3 | 2.5 | 0.4940 | 0.4902 | +0.8% | 0.4605 | +6.8% | 1.00 | 0.167 | 3.48/3.69/3.43 | 0.4964/0.4927/0.4638 |
| D   ap=0.3  δ=+2.5%  noisy | 0.3 | 2.5 | 0.5183 | 0.5133 | +1.0% | 0.4892 | +5.6% | 0.99 | 0.153 | 3.58/3.79/3.62 | 0.5223/0.5181/0.4983 |
| E   ap=0.3  δ=+9.3% | 0.3 | 9.33 | 0.5395 | 0.5305 | +1.7% | 0.5398 | -0.1% | 0.05 | — | 3.89/4.07/3.89 | 0.5431/0.5345/0.5433 |

## NN-seed sensitivity (gain vs LQG, steady window)

| Seed | A1 v3 | A1 v4 | A3 v3 | A3 v4 |
|---|---|---|---|---|
| 42 | +4.6% | +4.7% | +0.8% | +5.6% |
| 7 | +4.5% | +4.5% | +0.3% | +5.5% |
| 123 | +4.4% | +4.5% | +0.5% | +5.4% |

## Disclosures

- A1/A5 are by construction the NN training condition (train-on-test): they serve as the v3 best case, not as a generalisation test.
- Windows are not integer multiples of the beat/modulation period; v3 gains within ±1 % of zero mean 'benefit erased', not a precise below-baseline margin.
- v4 additionally uses the nominal alpha3(φ) profile and the commanded tool position x_p — known process data, no extra sensors.
- Scenario E is outside the PLL pull-in range by design: the confidence gate retracts the feedforward (u_FF → 0) and v4 coincides with the LQG baseline.
- Single deterministic run per scenario (plant and noise seeds fixed); NN-seed sensitivity above bounds the training variability.
