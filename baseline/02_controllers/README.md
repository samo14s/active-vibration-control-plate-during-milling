# 02_controllers — Control Algorithms

This directory contains the **two main controllers** compared in the article:
LQG (baseline) and DARC-MPC (proposed).

## Files

| File | Controller | Key class |
|---|---|---|
| `lqg_controller.py` | Linear Quadratic Gaussian | `LQGController` |
| `darc_mpc_v3_controller.py` | DARC-MPC (proposed novel method) | `DARC_MPC_v3_Controller` |

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
(RETRACTED: no benchmark for this timing claim exists anywhere in
the package, and there is no MPC to time — see ../RETRACTED.md item 4.)
