# Active vibration control of thin flexible workpieces during milling

Simulation study of piezo-actuated chatter control for a thin cantilevered
wall, built on a Reissner–Mindlin plate FEM.

The working question is not "does adding a controller reduce vibration" but
the two that decide whether such a controller is usable at all:

- the workpiece is **being consumed** while it is controlled, so the plant
  the controller was designed for stops existing partway through the job;
- the tool **moves**, so the direction in which the disturbance enters the
  structure sweeps, and a fixed actuator can end up unable to oppose it.

Both are measured here on a full machining programme, and the resulting
closed-loop chatter stability is **certified** — the real sampled controller
sits inside the monodromy matrix — rather than inferred from an equivalent
damping ratio.

---

## Layout

```
baseline/     the uploaded starting package, unmodified (see docs/ASSESSMENT.md)
src/          new, verified modules
tests/        verification and audit scripts — every claim below is reproducible
experiments/  the studies that produce the results
docs/         assessment of the starting package and what the paper should be
```

### `src/`

| module | what it does |
|---|---|
| `closed_loop_sld.py` | Monodromy matrix of the **actual sampled closed loop** (plant + ZOH + discrete observer + feedback). Propagates the fully coupled multi-mode system and handles multi-rate controller timing and multi-actuator layouts. |
| `evolving_plate.py` | Cantilever Mindlin plate with a **per-element thickness field**, so material removal changes the modal model. Thickness dependence is factored analytically, so re-assembly is a sparse rebuild rather than 720 element integrations. Also puts the piezo patch into the structure and takes its moment arm to the composite neutral axis. |
| `machining_path.py` | A machining programme (radial layers × axial bands) and the sequence of workpiece models it induces. |
| `actuator_placement.py` | Worst-case-over-path actuator layout: maximise `min_s γ(s)`, the reachable fraction of the disturbance, instead of modal strain energy at one frequency. |
| `baseline_controllers.py` | Established laws for fair comparison — DVF, PPF, and a repetitive controller with a printable convergence certificate — plus a common tuning routine under a shared effort budget. These are *time-domain* laws with a `step()` interface for the Newmark solver. |

Note that `experiments/run_benchmark.py` does **not** use those `step()`
objects: certifying a law through the monodromy requires it in observer +
gain state-space form, so the benchmark builds state-space equivalents
directly. Two consequences are stated rather than glossed:

- its "velocity feedback" reconstructs velocity through the same Kalman
  observer as every other law, so it is *not* true collocated DVF and does
  not inherit DVF's unconditional-stability property — the sensor and
  actuator here are not collocated anyway;
- its "modal position feedback" is the static stiffness-shift limit of PPF,
  not full PPF. Certifying the second-order PPF filter needs the observer
  block of the monodromy extended, which is not yet done.

---

## Main results

All reproducible from `tests/` and `experiments/`.

### The workpiece moves by a factor of ~3, non-monotonically

Over an 8 mm → 3 mm wall (3 radial layers × 10 axial bands):

| mode | blank | final | span |
|---|---|---|---|
| 1 | 1069.5 Hz | 390.1 Hz | 2.90× |
| 2 | 2083.7 Hz | 793.1 Hz | 2.64× |
| 3 | 5110.7 Hz | 2024.3 Hz | 2.52× |

f₁ first *rises* to 1130 Hz — early passes remove inertia near the free edge —
before falling as stiffness loss takes over.

### A single actuator has blind spots along the path

The reachable fraction γ of the disturbance, for the baseline patch:

| tool x [mm] | 0 | 10 | 30 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|
| γ | 0.944 | 0.996 | 0.633 | 0.180 | **0.006** | 0.037 |

Mean over the full pass **0.375**, minimum **0.000**. No single placement
avoids this. With 3 retained modes, three independent actuators span ℝ³ and
lift the worst case to **1.000** — a clean design rule: path-wide
input-matching needs at least as many independent actuators as retained modes.

| layout | min γ | mean γ |
|---|---|---|
| baseline, 1 patch | 0.001 | 0.399 |
| best 1 patch | 0.015 | 0.523 |
| best 2 patches | 0.408 | 0.786 |
| best 3 patches | **1.000** | **1.000** |

### A fixed-gain LQG becomes worse than no control

Checked three independent ways (`tests/verify_fixed_gain_instability.py`):

| layers removed | f₁ | max Re(A−BK) | max Re(**full LQG**) | ρ certified | time domain |
|---|---|---|---|---|---|
| 0 | 1030 Hz | −186.3 | −186.3 | 0.711 | decays |
| 1 | 819 Hz | −163.7 | −82.6 | 0.598 | decays |
| 2 | 605 Hz | −136.1 | **+271.7** | **2.365** | **grows** |
| 3 | 390 Hz | −103.0 | **+257.6** | **3.301** | **grows** |

The **regulator stays stable throughout**; it is the **observer**, built on
the stale blank model, that destabilises the loop. State feedback tolerates
the drift, observer-based output feedback does not.

Over the full programme, the fixed-gain loop is worse than no control at
**10 of 25** stations, while path-scheduled design holds a worst-case
critical depth of **1.48 mm — 7.1× open loop**.

### Minimising vibration is not the same as maximising stability

At matched control effort on the nominal plate (60 V budget, every law tuned
against the same reported objective, both selected gains interior optima):

| law | u peak [V] | y_rms [µm] | certified a_p,crit [mm] |
|---|---|---|---|
| open loop | 0.00 | 28.487 | 0.071 |
| velocity feedback | **8.92** | 0.166 | **2.401** |
| static modal position fb | 0.00 | 28.488 | 0.071 |
| LQG | 28.53 | **0.134** | 1.983 |

LQG wins on RMS by 19 % and **loses on stability by 17 %**, at 3.2x the
voltage. So **an RMS reduction is not evidence of chatter suppression** — a
point that invalidates the headline metric of a large part of this
literature, including the package this work started from.

### Certified lobes differ from the usual shortcut

Closed-loop lobes are commonly drawn by substituting closed-loop damping
ratios into the *open-loop* formula. Against the certified monodromy at
4900 RPM that substitution is **~6 % optimistic** (2.13 vs 2.01 mm), i.e. it
over-promises stability. The certified value is independently corroborated by
time-domain simulation, which places the transition between 1.81 and 2.41 mm.

Treating the modes as independent — as the baseline stability code does —
differs from the coupled system by up to **20.8 %**, because `Dp Dpᵀ` is rank
one and couples every mode through the single contact point.

---

## Reproducing

```bash
pip install numpy scipy matplotlib

python tests/verify_evolving.py                 # element, quadrature, removal
python tests/verify_monodromy_equivalence.py    # fast vs dense monodromy
python tests/verify_closed_loop_sld.py          # certified lobe vs time domain
python tests/verify_fixed_gain_instability.py   # the three-way check
python tests/analyze_actuator_alignment.py      # reachability along the path
python tests/audit_baseline_claims.py           # reproduces the baseline claims

python experiments/run_path_study.py            # the integrated study
python experiments/run_benchmark.py             # matched-effort comparison
```

---

## Status and honest limits

- The stability results are **linearised** statements about growth near the
  nominal periodic motion, which is where the linear cutting law is valid.
  The model does **not** yet enforce chip-thickness positivity, so it must
  not be used for post-instability amplitude or surface finish. See
  `docs/ASSESSMENT.md` §5.
- No process damping or ploughing term, at 2–4 µm chip thickness where both
  matter.
- This is simulation only. Verification against analytical and benchmark
  solutions is done (Leissa CCCC to 0.02 %, mesh convergence, `f ∝ h`,
  first-order eigenvalue perturbation with demonstrated O(Δh) convergence);
  **validation against measured data is not**, and the paper should say so.

`docs/ASSESSMENT.md` records what was checked in the starting package, which
of its published claims did not survive reproduction, and what remains to be
done before submission.
