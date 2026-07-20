# Positioning Document: LPV Gain-Scheduled H-infinity Control with Scheduled Delayed Feedback for Chatter Suppression in Thin-Walled Workpiece Milling

**Status:** Novelty assessed as intact under precise claim wording (adversarial review, July 2026). This document consolidates the state of the art, the confirmed gap, the defensible novelty statement, differentiation against the closest prior works, the recommended reference base, and target journals.

---

## 1. State of the Art Summary by Theme

### 1.1 Thin-wall milling chatter and the time-varying dynamics problem

Recent reviews (Sun et al. 2025, *Int. J. Extreme Manufacturing*; Yi et al. 2025, *Proc. IMechE Part B*; Qin et al. 2025, *IJAMT*) consistently identify in-process variation of thin-walled workpiece dynamics — driven by tool position along the feed path and progressive material removal — as the core unresolved difficulty for active chatter suppression. They call for adaptive or scheduled control; none cites an existing LPV gain-scheduled H-infinity solution with a scheduled delayed feedback term. Position-dependent *open-loop* stability lobes along the tool path are established practice (Ahmadi & Ismail 2012; Li et al. 2021; Jia et al. 2025), confirming that position/material-removal variation is a physically recognized, exploitable *known* quantity.

### 1.2 Robust / worst-case active control (the dominant paradigm)

The field's standard answer to known variation is robustness: variation is lumped into norm-bounded uncertainty or handled by worst-case design at a single operating point. Representative: Du et al. 2024 (*IJMS*, mu-synthesis + delayed PD on a piezo-patch plate — the direct baseline); Du et al. 2022 (*JMP*, one controller designed at the maximum-vibration position); Zhang et al. 2019 (*MSSP*, perturbation-model robust control); van Dijk et al. 2012/2014 (robust and fixed-structure delayed output feedback over spindle-speed ranges); Mizrachi et al. 2020 (*IJMTM*, delay-embedded H-infinity synthesis, turning); Ruttanatri et al. 2021 (*Automatica*, LKF/LMI H-infinity with the regenerative delay inside the design model); Dumanli & Okwudire 2021/2022 (regeneration-targeted optimal control via feed drives); Huang et al. 2018 (multi-delay LMI, variable-pitch cutters). This paradigm is explicitly conservative: known, predictable variation is treated as if it were unknown.

### 1.3 Scheduled, adaptive, and learning alternatives

Scheduling in active chatter control is sparse but emerging. Wang S. et al. 2019 (*IJAMT*) vary PD gains with tool position on exactly the piezo-patch/thin-walled-plate plant class — heuristically, with no synthesis or guarantees. A 2026 *J. Manufacturing Processes* paper schedules robust delayed (displacement-difference) feedback gains over *spindle speed*, with experiments. Brand et al. 2025 (*Control Engineering Practice*) perform LPV robust H-infinity design for an active tool holder in internal turning, scheduled on a quasi-static setup parameter (overhang). Adaptive routes (Kleinwort & Zaeh 2018/2021, online identification and re-tuning; Wang C. et al. 2022, FFT-tracking FxLMS) react to variation rather than exploiting a-priori NC-program knowledge, and carry no closed-loop stability certificate for the delayed periodic system. Learning-based controllers (Nasiri et al. 2025, *MSSP*, SAC-RL vs type-2 fuzzy on the same plate-piezo plant; Jiang et al. 2026, DQN) explicitly forgo model-based guarantees.

### 1.4 LPV modeling and gain-scheduling precedents (adjacent)

LPV *modeling* of thin-wall milling dynamics over tool position/material removal exists (Maslo et al. 2020, *Procedia CIRP*) — but is used only for passive spindle-speed selection, with no active control synthesis. LPV/gain-scheduled H-infinity control exists in machine-tool *components* (Hanifzadegan & Nagamune 2015, feed drives; Huang et al. 2024, ball screw) and in smart structures (Onat/Sahin/Yaman, piezo beam). Wang D. et al. 2015 attach the literal phrase "time-delay LPV milling control" to a 2-DOF lumped toy model scheduled on cutter angle. Generic LPV-with-delay synthesis machinery is available (de Souza & Palhares 2022).

### 1.5 Delayed feedback and delayed resonator foundations

Regenerative-targeted delayed feedback is well established: the delayed resonator lineage (Olgac & Holm-Hansen 1994; Olgac & Hosek 1998; Vyhlidal et al. 2019; recent adaptive DR work through 2026), spindle-synchronized displacement-difference feedback in milling (Li et al. 2022; Du et al. 2022 *JMPT*), delayed feedback distorting the regenerative effect in turning (Mancisidor et al. 2019), optimal delayed state feedback for thin-walled parts via an active fixture (Dong et al. 2023), and delayed output feedback with SLD shaping (van Dijk et al. 2014). All use fixed or robustly designed gains; none schedules the delayed gain on tool position or material-removal state.

### 1.6 Closed-loop stability analysis of periodic delayed systems

Computing stability lobes *with the controller in the loop* is established: Zhang et al. 2019 (ASME JMSE) coin the closed-loop stability lobe diagram (CLSLD); Lehotzky & Bachrathy 2021 include actuator/sensor/filter dynamics and show that omitting them falsifies lobes; Monnin et al. 2014 validate model-based closed-loop charts experimentally; Kordabad et al. 2023 apply semi-discretization to a controlled plant. Semi-discretization itself (Insperger & Stepan 2002/2004) demonstrably handles time-varying delays and periodic gain modulation. No work applies this machinery to certify a *parameter-varying, scheduled* delayed closed loop along a tool path.

---

## 2. Gap Analysis

**Confirmed gap (all sweep angles concur).** No paper found (2015–2026) that:

- (a) synthesizes a gain-scheduled/LPV H-infinity controller with **tool position along the feed path and material-removal state** — known in real time from the NC program/CAM model — as scheduling parameters, for **thin-walled-workpiece** milling with piezo-patch actuation;
- (b) combines this with a **scheduled, spindle-synchronized, regenerative-targeted delayed feedback term** whose gain is scheduled on the same parameter pair;
- (c) **certifies** the resulting time-periodic, parameter-varying, delayed closed loop via **semi-discretization stability lobes along the tool path**, including full controller/actuator dynamics.

Every individual ingredient exists separately (Sections 1.2–1.6). The contribution is therefore an **integration novelty**, and the paper lives or dies on claim wording and explicit differentiation from the five closest works (Section 4).

**Paradigm framing.** The dominant paradigm treats known variation as uncertainty (Du 2024) or worst case (Du 2022), which is provably conservative; adaptive and learning routes react without certificates; existing scheduling uses either heuristic gain laws (Wang S. 2019), a different scheduling variable (spindle speed, JMP 2026), or a quasi-static setup parameter in turning (Brand 2025). The gap is: *known variation as scheduling, residual variation as uncertainty, with a formal stability certificate for the scheduled delayed periodic loop.*

**Claims that are forbidden** (each falsified by the record):

- "first LPV control in machining" (Hanifzadegan 2015; Huang 2024; Wang D. 2015);
- "first gain scheduling in active chatter control" (JMP 2026 spindle-speed-mapped framework);
- "first position-varying gains for thin-wall milling" (Wang S. 2019);
- "first H-infinity + delayed feedback combination" (Mizrachi 2020; Du 2024);
- "first closed-loop stability lobes / first semi-discretization with controller in the loop" (Zhang 2019 CLSLD; Lehotzky & Bachrathy 2021; Monnin 2014; Kordabad 2023);
- "first LPV description of thin-wall milling dynamics" (Maslo 2020).

**Reviewer attacks to pre-empt (beyond novelty):**

1. *Grid-based LPV synthesis with a-posteriori controller interpolation does not by itself guarantee closed-loop stability under parameter variation.* Do not oversell "LPV guarantees"; present the semi-discretization certification along the tool path (with bounded scheduling rate) as the actual stability certificate, and say so honestly.
2. *Material-removal state is predicted, not measured.* Call the scheduling variables "known in real time (from the NC program/CAM model)" or "measured/known", never "measured" alone; retain a residual uncertainty block (spillover/additive uncertainty) to cover scheduling-map error. This strengthens the story: scheduling handles the known part, robustness handles the residual.
3. *Experimental validation.* Du 2024, Brand 2025, and the JMP 2026 paper are all experimental; a simulation-only submission to IJMS/MSSP will be attacked. At minimum, hardware-in-the-loop or a cutting experiment on one tool pass is strongly advised.
4. *Conservatism must be quantified* against the exact baseline (Du 2024): achievable stable depth of cut and gamma-level vs. position along the pass.

**Contingency pivot.** If a reviewer collapses the distinction with Brand 2025 + JMP 2026, the minimal pivot that restores clear water is to headline the *jointly scheduled delay term*: schedule both the regenerative-targeted delayed gain and the H-infinity controller on the (position, removal-state) pair and prove stability of the resulting time-periodic, parameter-varying DDE, plus experimental validation with in-cut variation. That combination has no precedent even ingredient-wise.

---

## 3. Novelty Statement

This work replaces the prevailing "known variation as uncertainty" treatment with "known variation as scheduling, residual variation as uncertainty," and makes three contributions:

1. **Position- and removal-state-scheduled LPV H-infinity synthesis for thin-walled workpiece milling.** A grid-based LPV H-infinity controller, with controller interpolation, is synthesized for a piezo-patch-actuated thin-walled workpiece using tool position along the feed path and material-removal state — both known in real time from the NC program/CAM model — as scheduling parameters, while a residual additive uncertainty block covers scheduling-map error and truncated-mode spillover. This de-conservatizes the robust baseline (Du et al. 2024), which wraps the same known variation into parametric uncertainty; the conservatism gap is quantified in achievable stable depth of cut and gamma-level as a function of position.

2. **A jointly scheduled, regenerative-targeted delayed feedback term.** The spindle-synchronized delayed (displacement-difference) feedback gain — previously fixed (Du et al. 2022; Dong et al. 2023) or mapped over spindle speed (JMP 2026) — is scheduled on the same (position, removal-state) pair as the H-infinity controller, so that the regeneration-cancelling action tracks the varying workpiece dynamics along the pass.

3. **Certification of the scheduled, time-periodic, delayed closed loop.** Stability of the resulting parameter-varying, time-periodic delay-differential closed loop — including full controller, actuator, and filter dynamics — is certified by semi-discretization stability lobes computed along the tool path under a bounded scheduling rate. This certificate, not the grid LPV synthesis itself, is presented as the formal stability guarantee, extending closed-loop SLD practice (Zhang et al. 2019; Lehotzky & Bachrathy 2021) to scheduled controllers for the first time.

---

## 4. Closest Prior Works and Differentiation

### 4.1 Primary differentiation table

| # | Work | What it does | Shared ingredients | Exact differentiators (ours vs. theirs) |
|---|------|--------------|--------------------|------------------------------------------|
| 1 | **Brand et al. 2025**, *Control Eng. Practice* — active tool holder, robust LPV H-inf, internal turning | LPV H-inf with regenerative-delay awareness and spillover safety; piezo actuation; experimentally validated (>95% RMS reduction) | LPV + H-inf + delay awareness + spillover + piezo | Turning (single constant delay, LTI per operating point) vs. time-periodic milling; scheduling on a quasi-static setup condition (overhang) vs. a real-time trajectory known from the NC program; no scheduled delayed feedback term; no semi-discretization certification of a periodic delayed closed loop; tool-side vs. thin-walled workpiece |
| 2 | **Wang S. et al. 2019**, *IJAMT* — time-space varying PD via piezo patch on thin-walled workpiece | PD gains varied with tool position following the first mode shape | Position-varying gains on exactly our plant class | Heuristic single-mode PD gain law vs. systematic LPV/H-inf synthesis; no robustness/spillover treatment; no regenerative delayed term; no stability certificate. Our paper is the *certified, systematic generalization* — cite in the abstract-level novelty statement |
| 3 | **Spindle-speed-mapped robust active control, 2026**, *J. Manufacturing Processes* (S1526612526004561) | Saturation-aware DVF (delayed displacement-difference) gains optimized per spindle speed and stored as a gain map; experimental; up to 13x stable depth | Gain-scheduled *delayed* feedback in milling chatter control | Scheduling variable is spindle speed vs. tool-path position/material removal; ad-hoc gain map vs. grid LPV H-inf synthesis with interpolation; no LPV stability machinery. **Most dangerous recent neighbor — must be read in full before submission** |
| 4 | **Du et al. 2024**, *IJMS* 274:109257 (reference baseline) | mu-synthesis robust control + delayed PD on the same plant/actuator; known variation treated as uncertainty | Same plant, same actuator, same combined (robust + delayed) architecture | Our entire contribution is the de-conservatization; the paper must quantify the conservatism gap against this exact baseline (stable depth of cut and gamma-level vs. position) |
| 5 | **Maslo et al. 2020**, *Procedia CIRP* — LPV model of thin-wall milling dynamics | LPV modeling over tool position/material removal, used for spindle-speed selection | Exactly our scheduling variables, as a model | No active control, no synthesis. Cite as modeling precedent; never claim first LPV description of thin-wall milling |

### 4.2 Secondary neighbors (one-sentence differentiation each)

- **Wang D. et al. 2015** (*Math. Probl. Eng.*): owns the literal phrase "time-delay LPV milling control" but on a lumped 2-DOF toy model scheduled on cutter angle, with no H-inf channel, no workpiece flexibility, no experiments.
- **van Dijk et al. 2014** (*IJRNC*): delayed output feedback with closed-loop SLD shaping — but robust over a pre-defined spindle-speed range, fixed-gain, spindle-side.
- **Dong et al. 2023** (*J. Vib. Control*): optimal delayed state feedback for a thin-walled part — but a single fixed gain via an active fixture, no scheduling, no LPV.
- **Kleinwort & Zaeh 2018/2021**: the adaptive (identify-then-retune) alternative for position-dependent dynamics — reactive, machine-tool side, no certified stability for the delayed periodic loop.
- **Nasiri et al. 2025** (*MSSP*): model-free RL/fuzzy control on the same plate-piezo plant — explicitly without model-based guarantees; a state-of-the-art comparator, not a competing certified design.
- **Zhang et al. 2019 (CLSLD), Lehotzky & Bachrathy 2021, Monnin et al. 2014, Kordabad et al. 2023**: preclude any "first closed-loop SLD / first SDM-with-controller" claim; our extension is certification of a *scheduled* closed loop along the path.

---

## 5. Reference List by Theme

*(Full citations as far as known; items marked [verify] require full-text confirmation before submission.)*

### A. Thin-wall milling chatter problem and reviews (gap framing)

1. Sun et al. (2025). "A review of chatter suppression in thin-wall milling: strategies, mechanisms, and applications." *International Journal of Extreme Manufacturing*.
2. Yi et al. (2025). "Review of milling chatter in aerospace thin-walled structures." *Proceedings of the IMechE, Part B: Journal of Engineering Manufacture*.
3. Qin et al. (2025). "Chatter stability prediction methods in the machining processes: a review." *International Journal of Advanced Manufacturing Technology*.

### B. Baseline lineage — robust/worst-case treatment of known variation

4. Du, J., Liu, X., Dai, H., Long, X. (2024). "Robust combined time delay control for milling chatter suppression of flexible workpieces." *International Journal of Mechanical Sciences* 274:109257. **[Primary baseline]**
5. Du, J., et al. (2022). "Chatter suppression for milling of thin-walled workpieces based on active modal control." *Journal of Manufacturing Processes*.
6. Du, J., et al. (2022). "Time delay feedback control for milling chatter suppression by reducing the regenerative effect." *Journal of Materials Processing Technology*.
7. Zhang, X., et al. (2019). "Robust active control based milling chatter suppression with perturbation model via piezoelectric stack actuators." *Mechanical Systems and Signal Processing*.
8. van Dijk, N.J., van de Wouw, N., Nijmeijer, H., et al. (2012). "Robust active chatter control in the high-speed milling process." *IEEE Transactions on Control Systems Technology*.
9. van Dijk, N.J., et al. (2014). "Fixed-structure robust controller design for chatter mitigation in high-speed milling." *International Journal of Robust and Nonlinear Control* 25(17):3495–3514.
10. Mizrachi, E., Basovich, S., Arogeti, S. (2020). "Robust time-delayed H-infinity synthesis for active control of chatter in internal turning." *International Journal of Machine Tools and Manufacture* 158:103612.
11. Dumanli, A., Okwudire, C. (2022). "Active chatter mitigation by optimal control of regenerative machining process dynamics." *IEEE/ASME Transactions on Mechatronics* (with 2021 *CIRP Annals* companion: "Active control of high frequency chatter with machine tool feed drives").
12. Huang, T., et al. (2018). "Robust active chatter control in milling processes with variable pitch cutters." *ASME Journal of Manufacturing Science and Engineering*.
13. Ruttanatri, P., Cole, M.O.T., Pongvuthithum, R. (2021). "H-infinity controller design for chatter suppression in machining based on integrated cutting and flexible structure model." *Automatica*.

### C. Scheduled, adaptive, and learning alternatives (must-differentiate set)

14. Wang, S., et al. (2019). "Vibration suppression of thin-walled workpiece milling using a time-space varying PD control method via piezoelectric actuator." *International Journal of Advanced Manufacturing Technology*.
15. Brand, Z., et al. (2025). "An active tool holder and robust LPV control design for practical vibration suppression in internal turning." *Control Engineering Practice*. [verify — read full text]
16. Wang, D., Wu, S., Wan, L., Dulikravich, G.S. (2015). "Time-delay LPV system control and its application in chatter suppression of the milling process." *Mathematical Problems in Engineering*. doi:10.1155/2015/307149.
17. [Authors TBD — resolve before citing] (2026). "Robust active control of milling chatter under actuator constraints: A spindle speed mapped framework with experimental validation." *Journal of Manufacturing Processes* (ScienceDirect S1526612526004561). [verify — read full text]
18. Kleinwort, R., Zaeh, M.F., et al. (2018). "Adaptive active vibration control for machine tools with highly position-dependent dynamics." *International Journal of Automation Technology* (and 2021 companion, *CIRP Journal of Manufacturing Science and Technology*).
19. Wang, C., et al. (2022). "Real time FFT identification based time-varying chatter frequency mitigation in thin-wall workpiece milling." *International Journal of Advanced Manufacturing Technology*.
20. Nasiri, K., et al. (2025). "Chatter suppression in nonlinear milling of a flexible plate-workpiece with attached piezoelectric actuators: SAC-based controller vs optimized type-2 fuzzy controller." *Mechanical Systems and Signal Processing*.
21. Dong, X., et al. (2023). "Suppress chatter in milling of thin-walled parts via fixture with active support." *Journal of Vibration and Control*.
22. Li, X., et al. (2022). "Displacement difference feedback control of chatter in milling processes." *International Journal of Advanced Manufacturing Technology*.
23. Li, X., et al. (2021). "Active control of milling chatter considering the coupling effect of spindle-tool and workpiece systems." *Mechanical Systems and Signal Processing*.

### D. LPV / gain-scheduling methodology precedents (adjacent domains)

24. Maslo, S., et al. (2020). "Improving dynamic process stability in milling of thin-walled workpieces by optimization of spindle speed based on a linear parameter-varying model." *Procedia CIRP* (ScienceDirect S2212827120307575). [verify — read full text]
25. Hanifzadegan, M., Nagamune, R. (2015). "LPV/switching H-infinity control of CNC machine tool feed drives" (journal papers and UBC PhD thesis).
26. Huang, X., et al. (2024). "Polytopic LPV modeling and gain scheduling H-infinity control of ball screw with position- and load-dependent variable dynamics." *Precision Engineering*. [verify]
27. Onat, C., Sahin, M., Yaman, Y. (~2017–2021). "Gain-scheduling H-infinity control of a smart beam based on an LPV model" (SMART 2017 conference; journal version details to be verified). [verify]
28. de Souza, C., Palhares, R.M. (2022). "New gain-scheduling control conditions for time-varying delayed LPV systems." *Journal of the Franklin Institute* 359(2):719–742.

### E. Delayed feedback / delayed resonator foundations

29. Olgac, N., Holm-Hansen, B.T. (1994). "A novel active vibration absorption technique: delayed resonator." *Journal of Sound and Vibration*.
30. Olgac, N., Hosek, M. (1998). "A new perspective and analysis for regenerative machine tool chatter." *International Journal of Machine Tools and Manufacture*.
31. Vyhlidal, T., et al. (2019). "Analysis and design aspects of delayed resonator absorber with position, velocity or acceleration feedback." *Journal of Sound and Vibration*.
32. Mancisidor, I., et al. (2019). "Delayed feedback control for chatter suppression in turning machines." *Mechatronics*.

### F. Stability analysis of periodic delayed systems / closed-loop SLD (verification machinery)

33. Insperger, T., Stepan, G. (2002/2004). "Semi-discretization method for delayed systems." *International Journal for Numerical Methods in Engineering* (and *Semi-Discretization for Time-Delay Systems*, Springer, 2011).
34. Zhang, X., et al. (2019). "Discrete time-delay optimal control method for experimental active chatter suppression and its closed-loop stability analysis." *ASME Journal of Manufacturing Science and Engineering* (introduces the CLSLD).
35. Lehotzky, D., Bachrathy, D., et al. (2021). "Milling processes with active damping: modeling and stability." *Journal of Computational and Nonlinear Dynamics*.
36. Monnin, J., Kuster, F., Wegener, K. (2014). "Optimal control for chatter mitigation in milling — Part 1: Modeling and control design; Part 2: Experimental validation." *Control Engineering Practice* 24:156–166, 167–175.

### Optional additions (if the journal permits more than 36 references)

- Borgioli, F., Hajdu, D., Insperger, T., Stepan, G., Michiels, W. (2020). "Pseudospectral method for assessing stability robustness for linear time-periodic delayed dynamical systems." *International Journal for Numerical Methods in Engineering*.
- Ahmadi, K., Ismail, F. (2012). "Stability maps along the toolpath in thin-wall milling." *CIRP Journal of Manufacturing Science and Technology*.
- Yang, Y., et al. (2022). "A Gaussian process regression-based surrogate model of the varying workpiece dynamics for chatter prediction in milling of thin-walled structures." *International Journal of Mechanical System Dynamics* (cite in modeling section as the scheduling-map source).
- Zhang, Y., Sims, N.D. (2005). "Milling workpiece chatter avoidance using piezoelectric active damping: a feasibility study." *Smart Materials and Structures*.
- Iorga, L., Baruh, H., et al. (2008). "A review of H-infinity robust control of piezoelectric smart structures." *Applied Mechanics Reviews*.
- Selivanov, A., Fridman, E. (2023). "Improved residual mode separation for finite-dimensional control of PDEs: application to the Euler-Bernoulli beam." *Systems & Control Letters* (provably spillover-free truncation).

**Pre-submission action items:** obtain and read full texts of refs 15, 17, and 24 (currently characterized partly from abstracts/snippets); resolve the authorship of ref 17; verify the journal version details of ref 27.

---

## 6. Recommended Target Journals (Q1, in order)

1. **International Journal of Mechanical Sciences (IJMS)** — The baseline paper (Du et al. 2024) is published here, making the quantified de-conservatization a direct, natural successor with maximal reviewer familiarity; requires at least HIL or one-pass cutting validation.
2. **Mechanical Systems and Signal Processing (MSSP)** — Best fit for the control-synthesis and certification depth (LPV H-inf + semi-discretization), with the closest state-of-the-art comparators (Nasiri 2025, Zhang 2019) already in its pages; experimental evidence strongly expected.
3. **Journal of Sound and Vibration (JSV)** — Ideal if the framing emphasizes the delayed-feedback/delayed-resonator theory and stability analysis of the time-periodic parameter-varying DDE, where the field's foundational delay literature resides.
4. **Journal of Manufacturing Processes (JMP)** — Home of the closest competing scheduled-delayed-gain paper (2026), so a head-to-head positioning is natural; process-oriented reviewers will demand cutting experiments but accept a more applied certification narrative.
5. **International Journal of Machine Tools and Manufacture (IJMTM)** — Highest-prestige machining venue (Mizrachi 2020 precedent), but only viable with full cutting experiments demonstrating stable-depth gains on a real thin-walled part; hold as target for the experimentally validated version.