# Gap ② — Position-scheduled multi-actuator (patch array) handover

**Verification date:** 2026-07-20 · **Verdict: GAP FRESH** · Not selected
(natural sequel to the PS-LPV paper; risk of "incremental to own work"
perception); kept as a future direction.

## Candidate gap

A piezo patch ARRAY on the thin-walled workpiece with position-scheduled
actuator weighting/selection/handover (allocation-based or MIMO LPV/H∞),
co-designed placement + schedule, certified against the moving regenerative
loop.

## Closest works (agent-verified)

1. **Du et al. 2022, J. Manuf. Processes** — single patch, placement
   optimized over first three modes, one worst-case controller: the "static
   placement + worst-case controller" baseline the gap argues against.
2. **S. Wang et al. 2019, IJAMT** — time-space varying PD, single patch:
   closest on scheduling; no MIMO, no allocation, uncertified.
3. **Nasiri et al. 2025, MSSP** — plural patches on a plate workpiece with
   RL/fuzzy control: nearest multi-patch work; no position-based
   selection/weighting, no certificate, simulation-only.
4. **Zhang & Sims 2005** — foundational single-patch PPF experiment.
5. **Dong et al. 2023 (active fixture, optimal delayed state feedback);
   Brecher 2010 (two-axis active holder)** — workpiece-side MIMO-ish, fixed
   points, no array handover.
6. **Kleinwort/Zaeh 2018; Ozsoy 2024/2026** — scheduling/adaptation on the
   machine/spindle side or over spindle speed, not a workpiece patch array
   over tool position.
7. **Mirror-milling follow-up support line (IJAMT 2020; JMP 2023/24; MSSP
   2024)** — "authority must follow the tool" solved MECHANICALLY with a
   moving support head + second robot. Strongest conceptual rival;
   differentiator: electronic handover needs no second robot and works on
   closed pockets / single-side access.
8. **Generic smart-structure MIMO placement** (Peng 2005; Kumar & Narayanan
   2007; Bendine 2019) — mature placement machinery; no machining, no moving
   disturbance, no in-process handover.

## Defensibility requirements

(a) Quantify the authority loss of the best single static patch at mode
nodes/weak-coupling positions along the toolpath (the motivating figure that
worst-case designs hide); (b) certify the switched/LPV MIMO loop against the
moving regenerative delay (semi-discretization or IQC over the schedule);
(c) compare stable depth against the best static single patch (Du 2022
protocol) and position against follow-up supports.
