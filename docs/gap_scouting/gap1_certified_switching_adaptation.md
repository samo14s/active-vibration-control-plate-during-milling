# Gap ① — Certified supervisory/switching adaptation for active chatter control

**Verification date:** 2026-07-20 · **Verdict: GAP FRESH** · Not selected
(highest theoretical risk); kept as a future direction.

## Candidate gap

A supervisor that switches between pre-certified controller gains based on
online modal-shift detection, with switching/dwell-time stability guarantees
for the TIME-PERIODIC DELAY-DIFFERENTIAL closed loop (per-mode
semi-discretization monodromy + dwell-time / joint-spectral-radius bounds).

## Closest works (agent-verified)

1. **Y. Wang, R. Wang & J. Zhao 2013, IET CTA** — ISS of switched delay
   systems with a rotational-cutting worked example. Closest crossing of
   switched-delay theory and machining — but constant-delay autonomous model,
   failure-driven (not supervisory) switching, no monodromy certification.
2. **Insperger 2006 IEEE TCST; Stépán & Insperger 2006** — act-and-wait:
   monodromy-certified periodic on/off gain switching. Strong precedent for
   "switching certified via monodromy" — but fixed open-loop schedule, no
   detection-driven supervisor.
3. **Kleinwort, Platz & Zaeh 2018 (+ CIRP Annals tuning)** — adaptive
   re-tuning under position-dependent dynamics, NO certificate (confirms the
   "adaptive without guarantee" pole).
4. **F. Wu 2026 (arXiv)** — LMI-certified switched-LPV control of an active
   magnetic bearing: certified switching in rotating machinery, no
   regenerative delay, no cutting.
5. **Dumanli & Okwudire 2022; van Dijk 2014; Mizrachi 2020-21** — certified
   single robust controllers: robust-but-not-adaptive.
6. **Wu 2016; Liu 2017/2018** — adaptive chatter control with L-K proofs on
   averaged/constant-delay models; single continuously adapted law, no
   supervisor over pre-certified gains.
7. **arXiv 2511.17894 (Nov 2025)** — ML-based online SLD estimation +
   adaptive spindle-speed control: supervisory in flavor, adapts process
   parameters not the active controller, no switched-system certificate.
   Watch this group.
8. **Theory toolkit (no machining):** Battistelli/Hespanha/Tesi 2012;
   Vu & Liberzon 2012; Yuan 2021 IEEE TAC; Koru 2018 (dwell time for
   switched delay systems).

## Defensibility requirements

Position explicitly against Wang 2013 and act-and-wait; differentiators:
(i) time-periodic coefficients + regenerative delay handled jointly via
lifted monodromy maps; (ii) switching triggered by online modal-shift
detection with a guaranteed fallback mode; high-fidelity demonstration of
modal-shift-triggered switching (fixturing change / tool wear mid-cut).
