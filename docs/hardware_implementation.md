# Hardware Implementation Path (Experimental Follow-Up)

The simulation campaign replicates the rig of Du et al. (IJMS 274:109257,
2024). This note maps every element of the proposed PS-LPV-DR controller
onto that hardware, for the follow-up experimental validation.

## Rig (identical to the reference paper)

| Element | Device | Notes |
|---|---|---|
| Workpiece | AL6061 cantilever plate 100 × 80 × 4 mm | clamped bottom edge |
| Actuator | QDA60-20-0.7 piezo patch (60 × 20 × 0.7 mm) | bonded lower corner |
| Amplifier | PI E-420 (gain 100, ±150 V effective budget) | sets `v_max` |
| Sensor | eddy-current displacement probe (ZA11-type) | fixed upper corner |
| Real-time target | NI PXIe (PXIe-8861 class) | SISO loop at 50 kHz |
| Spindle | 3-flute Ø10 end mill, down-milling, a_e = 0.1 mm | encoder output required |

## Controller deployment

1. **Gain look-up table.** The offline synthesis (`avc/synthesis.py`)
   produces the raw grid controllers (9 positions × 3 removal states) in a
   common state basis. Export `(Ak, Bk, Ck)` per grid point plus the
   scheduling interpolation weights; the target evaluates the bilinear
   interpolation at the current `(x_T, ϱ)` every 20 ms (scheduling
   time-scale ≫ control period, so gains can be updated in a low-priority
   loop while the 50 kHz loop runs).
2. **Scheduling inputs.** `x_T` from the NC block position / drive
   encoders (RS-274 look-ahead or a position latch); `ϱ` from the process
   plan (pass counter). Neither requires extra sensing.
3. **Roll-off filter.** The 4th-order 4.8 kHz filter is part of the digital
   controller (composed at export time); do not add an extra analog filter
   without re-running the sampled-loop certificate.
4. **Delayed feedback term.** `u_d = k_r(θ) [ŵ(t) − ŵ(t−τ)]` uses the
   observer state (available inside the controller) and a circular buffer
   of length τ = 60/(N_t Ω). Lock τ to the *measured* spindle period from
   the encoder each revolution — do not trust the commanded speed.
5. **Timing budget.** The certificates assume ≤ 1.5 control periods total
   latency at 50 kHz (30 µs). Measure the actual loop latency on the
   target; if it exceeds this, re-run `avc.synthesis.sampled_rho` and the
   SLD certificates with the measured value before cutting.
6. **Safety.** Voltage saturation at ±150 V is in the certified model; add
   a hardware displacement interlock (e.g. 200 µm at the probe) and a
   spindle-load trip as independent layers.

## Suggested experimental matrix (mirrors the simulations)

1. Modal validation: impact + swept-sine with the patch (compare Fig. 1).
2. Fixed-position cuts at 4.9 krpm, a_p sweep 0.1 → 3 mm: OL, DPD,
   frozen H∞, PS-LPV, PS-LPV-DR (stability map vs. Fig. 6).
3. Full passes at a_p = 1 mm (Fig. 8 protocol), logging `x_T`, w, u.
4. Multi-pass campaign (10 passes ≙ 1 mm recession) for the
   removal-scheduling study (Fig. 10 protocol).
