# 02_controllers — Control Algorithms

This directory contains the controllers compared in the article:
LQG (baseline), DARC-MPC v3 (learned feedforward) and DARC-MPC v4 PLAD
(phase-locked learned feedforward — research-gap contribution).

## Files

| File | Controller | Key class |
|---|---|---|
| `lqg_controller.py` | Linear Quadratic Gaussian | `LQGController` |
| `darc_mpc_v3_controller.py` | DARC-MPC (learned FF, open-loop clock) | `DARC_MPC_v3_Controller` |
| `darc_mpc_v4_plad_controller.py` | DARC-MPC v4 PLAD (sensorless phase-locked FF) | `DARC_MPC_v4_PLAD_Controller` |

## LQG Controller

Standard LQG = LQR (state feedback) + Kalman observer.

```python
from lqg_controller import LQGController

lqg = LQGController(plate, dt=5e-5, verbose=True)

# Sub-optimal weights (typical engineer's first guess)
lqg.optimize_weights(w_q_list=[1e13],
                      w_qd_list=[1e8], w_r=1.0)

# Optimal weights (full grid search)
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                      w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)

lqg.discretize_observer()

# Inside Newmark loop:
u, x_hat = lqg.step(x_hat_prev, u_prev, y_meas)
```

## DARC-MPC Controller

**DARC-MPC** = Deep Adaptive Robust Control with MPC.

**Architecture**:
```
u(t) = u_LQG(x̂) + α · NN_FF(φ, x̂)
       └────┬────┘   └─────┬─────┘
       reactive       anticipative
       baseline       feedforward
```

**Components**:
1. **LQG baseline** (optimal weights)
2. **Phase-aware NN feedforward** (3-input, 1-output)
3. **Iterative Learning Control** (offline pre-training, 30 iter)
4. **Lyapunov safety filter** (rejects unsafe commands)

```python
from darc_mpc_v3_controller import DARC_MPC_v3_Controller

darc = DARC_MPC_v3_Controller(
    plate, dt=5e-5,
    base_w_q=1e14, base_w_qd=1e8, base_w_r=1.0,  # optimal LQG base
    ff_lr=0.005,           # NN learning rate
    ff_max=10.0,           # FF output saturation
    ff_alpha=1.0,          # FF mixing gain
    alpha4_periodic=a4[:n_per],  # one period of cutting force
    n_per=n_per,           # samples per tooth-passing period
    safety_alpha=5.0,      # Lyapunov sensitivity
    enable_adaptation=True,
    u_max=150.0,           # piezo voltage saturation
)

# Pre-train (offline, ~3 min)
darc.pretrain_iterative_simulation(
    simulator, alpha3, alpha4, kp_idx,
    n_iterations=30,
    n_epochs_per_iter=15
)

# Reset history before deployment
darc.history_u_lqg = []
darc.history_u_ff = []
darc.history_u_total = []

# Inside Newmark loop:
u, x_hat = darc.step(x_hat_prev, u_prev, y_meas, k_step)
```

## Performance comparison

| Metric | LQG (sub-opt) | DARC-MPC | Gain |
|---|---:|---:|---:|
| RMS vibration | 0.825 µm | 0.666 µm | **+19%** |
| Modal damping (Mode 1) | 13.2% | 31.1% | **+136%** |
| Stability domain (a_p crit) | 2.17 mm | 3.05 mm | **+41%** |

## Why "Deep Adaptive Robust Control with MPC" (DARC-MPC)?

- **Deep**: uses neural network (multilayer)
- **Adaptive**: NN learns from simulation residuals
- **Robust**: Lyapunov safety filter guarantees stability
- **MPC-inspired**: phase-aware prediction of forcing

The NN replaces the explicit model predictive control optimization,
providing similar anticipative behavior at much lower computational cost
(0.1 µs vs 100+ ms per step).

---

## DARC-MPC v4 — PLAD (Phase-Locked Adaptive DARC)

Addresses the research gap of v3 (see `docs/research_gap.md`): the v3
feedforward is indexed by an open-loop clock `k mod n_per` that assumes an
exactly known, constant spindle speed, and its adaptation factor
(`lambda_robust`) was computed but never applied.

```
u(t) = u_LQG(x̂) + α · c_lock(t) · NN_FF(φ̂(t), x̂)
```

**New components** (`darc_mpc_v4_plad_controller.py`):
1. **`SpindlePhaseObserver`** — band-pass (Q = 4) + digital PLL locks onto
   the tooth-passing fundamental in the displacement signal (sensorless;
   PLL frequency clamp = pull-in range ±7 %; lock time measured per
   scenario as `t_lock_s` in `results_gap_sync/metrics.json`, ≈ 0.1 s).
2. **Model-based phase referencing** — closed-loop FRF from the cutting
   force fundamental to the sensor, scheduled over tool position
   (`enable_gs` solver hook) and frequency; one-shot calibration at
   nominal absorbs residual bias.
3. **Confidence gating** — PLL lock quality (amplitude-independent
   cos Δθ metric) scales the feedforward continuously; falls back to
   pure LQG when lock is lost, and clamp saturation is detected
   explicitly so offsets beyond the pull-in range retract the
   feedforward (no pseudo-lock). Replaces the inert v3 adaptation.

```python
from darc_mpc_v4_plad_controller import DARC_MPC_v4_PLAD_Controller

v4 = DARC_MPC_v4_PLAD_Controller(
    plate, dt=5e-5,
    alpha3_periodic=a3[:n_per],   # for the phase-reference model
    ff_lr=0.005, ff_max=10.0, ff_alpha=1.0,
    alpha4_periodic=a4[:n_per], n_per=n_per,
    safety_alpha=5.0, u_max=150.0,
)
v4.copy_feedforward_from(darc_v3_trained)     # identical NN weights
v4.calibrate_phase_reference(sim, a3, a4, kp_idx)   # one-shot, nominal
v4.reset_runtime()                            # before each deployment run
# NewmarkSimulator passes x_p_now automatically (enable_gs)
```

**Result** (steady state, `main_gap_spindle_sync.py`): a 1–2.5 % spindle
speed error erases the v3 feedforward benefit (+4.6…4.9 % → −0.3…+1.0 %),
while v4 retains it (+4.7…+6.8 %) with no steady-state cost at nominal
speed; beyond the ±7 % pull-in range v4 falls back to the LQG baseline.
