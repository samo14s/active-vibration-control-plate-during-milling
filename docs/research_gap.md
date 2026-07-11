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

1. **Sensorless spindle-phase observer.** A band-pass filter centred at the
   nominal tooth-passing frequency followed by a digital PLL (product
   detector + PI loop) tracks the forced fundamental in the displacement
   measurement itself — no encoder signal is used. Pull-in range ±7 %,
   lock time ≈ 0.1 s.
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
   This replaces v3's inert `lambda_robust` with an adaptation signal that
   provably reaches the control law, and it composes with the retained
   Lyapunov safety filter.
4. **Identical learned weights.** v4 reuses the v3-trained network verbatim
   (`copy_feedforward_from`), so every performance difference measured in
   the experiments is attributable to the synchronisation layer alone.

## 4. Experimental evidence (this repository)

`05_main/main_gap_spindle_sync.py`; all controllers designed/trained at
nominal speed only; steady-state window t > 0.15 s; full results in
`results_gap_sync/`. Headline numbers (ap = 0.3 mm, RMS reduction vs the
LQG baseline of the same scenario):

| Effective speed offset | DARC v3 (open-loop clock) | DARC v4 (PLAD) |
|---|---|---|
| 0 % (nominal)          | +4.6 %                    | +4.6 % |
| +1.23 %                | −0.2 %                    | +5.1 % |
| +2.50 %                | +0.7 %                    | +5.5 % |
| −1.20 %                | −0.2 %                    | +5.3 % |

A 1–2.5 % spindle-speed error — well inside realistic fluctuation ranges —
erases the entire learned-feedforward benefit of v3, while v4 retains it
fully (lock confidence 0.98–0.99) and costs nothing at nominal speed.
Scenario B (sinusoidal ±1 % speed fluctuation at 2 Hz) and scenario C
(4 s pass, sustained lock while the tool advances) are reported in
`results_gap_sync/summary.md`.

## 5. Scope and limitations

- The phase observer needs a detectable forced fundamental near the nominal
  tooth-passing frequency (band-pass pull-in ±7 %). Larger deviations —
  e.g. aggressive deliberate SSV — would require re-centring the band-pass
  from the spindle *command* (still encoder-free).
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
