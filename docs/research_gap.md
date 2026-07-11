# Research gap — spindle-speed uncertainty and the learned feedforward

**Contribution implemented in this repository:** `DARC-MPC v4 "PLAD"`
(Phase-Locked Adaptive DARC), `02_controllers/darc_mpc_v4_plad_controller.py`,
demonstrated by `05_main/main_gap_spindle_sync.py` and validated by
`03_analysis/validate_phase_observer.py`.

---

## 1. The gap, as evidenced in this codebase

The baseline DARC-MPC v3 controller (`darc_mpc_v3_controller.py`) computes

```
u(t) = u_LQG(x̂) + α · NN_FF(φ_clock, x̂),      φ_clock = 2π (k mod n_per)/n_per
```

Two structural weaknesses follow directly from the code:

1. **Open-loop synchronisation.** The learned anticipative feedforward is
   indexed by the step counter `k mod n_per` — an open-loop clock that
   assumes the tooth-passing period is *exactly* known and constant
   (`darc_mpc_v3_controller.py`, `step()`). Any deviation of the real
   spindle speed from its nominal value desynchronises the feedforward,
   which then injects a periodic voltage at a slowly drifting, wrong phase.
   The anticipative benefit does not merely disappear — the injection can
   actively degrade the response below the LQG baseline.

2. **Inert adaptation.** v3 instantiates an online adapter
   (`OnlineRLSAdapter`) and computes a robustness factor `lambda_robust`
   at every step, but the factor is never applied to any gain or command:
   the advertised "adaptive robust" layer had no effect on the closed loop.

A third, related observation: the simulator (`newmark_solver.py`) has always
supported controllers that receive the current tool position
(`enable_gs` hook), yet no controller in the package used it — the
position-dependence of the disturbance direction `D_p(x_p)` was ignored.

## 2. The gap, as situated in the literature

Active chatter mitigation for flexible/thin-walled workpieces with
piezoelectric actuation is an established feedback line of work — direct
velocity feedback [1], LQG [2], modal control [3], sliding-mode [4]; the
authoritative review [5] categorises active damping as feedback and treats
spindle-speed-based methods (speed selection, spindle speed variation)
separately, with no learned periodic-feedforward category.

Learned periodic feedforward *is* emerging in machining — an iteratively
learned, pre-scheduled feedforward assisting an inertial damper [6], a
virtual vibration absorber tuned at the (assumed known) tooth-passing
frequency [7], and an angle-indexed learned compensation in the patent
literature, replayed from a spindle **encoder** [8]. None of these analyses
robustness to synchronisation error.

That robustness matters is quantitatively established outside machining:
standard repetitive control tolerates a period mismatch of only ~±0.1 %
before its periodic rejection collapses [9]; angle-domain/spatial repetitive
control fixes this **with an encoder** [10]. Spindle speed in milling
fluctuates measurably under intermittent cutting load [11] (induction-spindle
slip at rated load is 2–3 %), and deliberate spindle-speed variation for
chatter suppression uses amplitudes of 5–30 % [12] — one to two orders of
magnitude beyond what clock-synchronised periodic compensation tolerates.
Tacholess (vibration-only) spindle-speed estimation is mature in condition
monitoring [13], and vibration-referenced synchronous compensation exists
for single-harmonic rotor unbalance in active magnetic bearings [14], but
sensorless "speed" information in machining control has so far only fed
speed-command adaptation, not actuator-side anticipative compensation [15].

**Narrowed gap statement** (defensible against the closest prior art
[6, 8, 9, 10, 14]):

> Learned periodic feedforward compensation in machining is synchronised by
> an open-loop clock or an encoder trigger under the assumption of exactly
> known, constant spindle speed. Speed-robust periodic control exists in the
> general control literature, but the known remedies either require a
> spindle encoder/angle trigger or sacrifice periodic-rejection performance,
> and none has been applied to active chatter control of thin-walled
> milling. Sensorless (vibration-only) phase-locked synchronisation of a
> learned multi-harmonic feedforward waveform — with a confidence-gated
> fallback to the robust feedback baseline — has not been demonstrated for
> milling, where interrupted cutting makes the measured signal impulsive
> and the regenerative dynamics broadband.

## 3. How v4 PLAD addresses the gap

```
u(t) = u_LQG(x̂) + α · c_lock(t) · NN_FF(φ̂(t), x̂)
```

1. **Sensorless spindle-phase observer.** A band-pass filter (Q = 4)
   centred at the nominal tooth-passing frequency followed by a digital
   PLL (product detector + PI loop) tracks the forced fundamental in the
   displacement measurement itself — no encoder signal is used. The PLL
   frequency integrator is clamped to ±7 % of nominal (the pull-in
   range); measured lock time (confidence first crossing 0.9) is
   reported per scenario as `t_lock_s` in `results_gap_sync/metrics.json`.
2. **Model-based, position-scheduled phase referencing.** The PLL tracks the
   phase of the *vibration*; the feedforward needs the phase of the
   *disturbance clock*. The offset between the two is predicted from the
   closed-loop FRF between the modal cutting-force fundamental
   (`ft·C_a3·D_p(x_p)`) and the sensor, evaluated on a (frequency × tool
   position) grid and interpolated online — this finally exploits the
   solver's `enable_gs` position hook. A one-shot offline calibration at
   nominal conditions absorbs the residual bias (neglected regenerative
   coupling, filter effects).
3. **Confidence-gated feedforward — adaptation that is actually wired in.**
   The PLL lock quality (direction of the demodulated (I,Q) vector, which
   equals cos Δθ independently of amplitude normalisation) gates the
   feedforward gain continuously: full authority when locked, smooth
   degradation to **pure LQG** when lock is lost or not yet acquired.
   Because a saturated frequency clamp can hold a *pseudo-lock* (the
   proportional branch sustains a static phase error with cos Δθ still
   high), clamp saturation is detected explicitly and also zeroes the
   confidence — offsets beyond the ±7 % pull-in range therefore retract
   the feedforward deterministically (scenario E, test T4). This replaces
   v3's inert `lambda_robust` with an adaptation signal that provably
   reaches the control law, and it composes with the retained Lyapunov
   safety filter.
4. **Identical learned weights.** v4 reuses the v3-trained network verbatim
   (`copy_feedforward_from`, including the saturation and input-scaling
   constants), so every performance difference measured in the experiments
   is attributable to the synchronisation layer alone. v4 additionally
   consumes two pieces of *known process data* that v3 does not use: the
   nominal `alpha3(φ)` profile (phase-reference model) and the commanded
   tool position `x_p` (position scheduling). Neither requires an extra
   sensor.

## 4. Experimental evidence (this repository)

`05_main/main_gap_spindle_sync.py`; all controllers designed/trained at
nominal speed only; steady-state window t > 0.15 s; full results
(including control effort, full-record RMS, measured lock times and an
NN-seed-sensitivity appendix) in `results_gap_sync/summary.md` /
`metrics.json`. RMS reduction vs the LQG baseline of the same scenario:

| Scenario | DARC v3 (open-loop clock) | DARC v4 (PLAD) | v4 lock conf. |
|---|---|---|---|
| A1 · 0 % (nominal, ap 0.3 mm)  | +4.6 % | +4.7 % | 0.99 |
| A2 · +1.23 %                   | −0.1 % | +4.7 % | 0.99 |
| A3 · +2.50 %                   | +0.8 % | +5.6 % | 0.98 |
| A4 · −1.20 %                   | −0.2 % | +4.8 % | 0.99 |
| A5 · 0 % (ap 0.6 mm)           | +4.9 % | +4.9 % | 0.99 |
| A6 · +2.50 % (ap 0.6 mm)       | +0.4 % | +5.9 % | 0.98 |
| B  · SSV ±1 % @ 2 Hz           | −0.3 % | +5.1 % | 0.99 |
| B2 · SSV ∓1 % @ 2 Hz           | +2.3 % | +5.1 % | 0.99 |
| C  · +2.5 %, long pass 4 s     | +0.8 % | +6.8 % | 1.00 |
| D  · +2.5 %, noisy sensor      | +1.0 % | +5.6 % | 0.99 |
| E  · +9.3 % (beyond pull-in)   | +1.7 % | −0.1 % | 0.05 |

A 1–2.5 % spindle-speed error — inside realistic fluctuation ranges —
erases the learned-feedforward benefit of v3 (−0.3 % … +1.0 % across
A2–A4, A6, B, D; B2 shows the residual benefit is alignment-dependent,
+2.3 % on one modulation side vs −0.3 % on the other), while v4 retains
it consistently (+4.7 % … +6.8 %). Measured lock time (confidence first
crossing 0.9) is 0.15–0.17 s. The desynchronised v3 also *wastes control
effort*: e.g. scenario A6, u_RMS 7.83 V (v3) vs 7.34 V (v4) vs 7.27 V
(LQG) while vibrating more. Scenario D shows the sensorless observer
tolerates 0.1 µm RMS sensor noise + 50 µs delay; scenario E shows the
pseudo-lock guard retracting the feedforward beyond the pull-in range
(u_FF → 0.15 V RMS, v4 within 0.1 % of the LQG baseline).

## 5. Scope and limitations

- **A1/A5 are the training condition.** The nominal scenarios coincide by
  construction with the environment the NN was trained in; they serve as
  the v3 best case, not as a generalisation test. The experiment's object
  is precisely the deviation between training and deployment.
- **Metric windows.** The steady-state window (t > 0.15 s) excludes the
  mechanical transient and the v4 lock-in; full-record RMS is stored
  alongside in `metrics.json`/`summary.md`. During lock-in the v4
  feedforward is disengaged, so v4's *full-record* RMS at nominal sits
  slightly above v3's (+0.8 … +1.0 %) while remaining ≈ 4 % below LQG —
  "no cost at nominal" refers to the locked steady state. Windows are not
  integer multiples of the beat period, so v3 gains within ±1 % of zero
  should be read as "benefit erased" rather than a precise below-baseline
  margin.
- The phase observer needs a detectable forced fundamental near the nominal
  tooth-passing frequency (PLL pull-in clamp ±7 %). Larger deviations —
  e.g. aggressive deliberate SSV — are detected via clamp saturation and
  handled by falling back to LQG (scenario E); retaining the feedforward
  there would require re-centring the observer from the spindle *command*
  (still encoder-free).
- Under developed chatter the fundamental is buried; the confidence gate
  then correctly retracts the feedforward and the system falls back to the
  LQG baseline — graceful, but the anticipative benefit is unavailable
  exactly when margins are smallest. Combining PLAD with spindle-command
  feedforward re-centring is the natural next step.
- The "1–3 % under load" premise is supported qualitatively [11] and by
  motor-slip physics; modern vector-controlled spindles hold smaller
  steady-state error (transient dips at tooth engagement remain). The
  experiments therefore treat the offset as a studied perturbation range.
- Simulation-only evidence, ideal displacement sensor (as in the baseline
  article package); sensor noise/delay are available in
  `piezo_actuator.py` for follow-up studies.

## References

[1] Zhang & Sims, *Smart Mater. Struct.* 14(6):N65, 2005. doi:10.1088/0964-1726/14/6/N01
[2] Parus et al., *J. Vib. Control* 19(7):1103–1120, 2013. doi:10.1177/1077546312442097
[3] Du & Long, *J. Manuf. Processes*, 2022. S1526612522007551
[4] Wan et al., *Mech. Syst. Signal Process.* 136:106528, 2020. doi:10.1016/j.ymssp.2019.106528
[5] Munoa et al., *CIRP Annals* 65(2):785–808, 2016. doi:10.1016/j.cirp.2016.06.004
[6] Bahtiyar, Sencer & Beudaert, *CIRP Annals* 73(1), 2024. S0007850624000210
[7] Franco et al., *CIRP Annals* 72(1), 2023. S0007850623000471
[8] US Patent 9,846,428 (Fanuc), "Controller for spindle motor".
[9] Steinbuch, *Automatica* 38(12):2103–2109, 2002. doi:10.1016/S0005-1098(02)00134-6
[10] e.g. *Control Eng. Practice*, 2013 (S096706611300083X); *Automatica* 158:111282, 2023. doi:10.1016/j.automatica.2023.111282
[11] Soshi, Raymond & Ishii, *Procedia CIRP* 14:159–163, 2014. doi:10.1016/j.procir.2014.03.087
[12] Seguy et al., *Int. J. Adv. Manuf. Technol.* 50:883–895, 2010. doi:10.1007/s00170-009-2336-9
[13] Peeters et al., *Mech. Syst. Signal Process.* 129:407–436, 2019. doi:10.1016/j.ymssp.2019.02.031
[14] Xu, Wu & Guan, *Shock and Vibration* 2020:2606178, 2020. doi:10.1155/2020/2606178
[15] Yamato et al., *Int. J. Precis. Eng. Manuf.* 22:1071, 2021. doi:10.1007/s12541-021-00469-2
