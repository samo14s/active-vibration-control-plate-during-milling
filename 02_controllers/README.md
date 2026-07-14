# 02_controllers — Control Algorithms

Two controllers are compared: **LQG** (baseline) and **PALF-LQG** (LQG + a phase-locked
learned feedforward). PALF-LQG was renamed from the earlier, over-claiming "DARC-MPC
v3"; see `docs/AUDIT_FINDINGS.md` for why the old name was indefensible.

## Files

| File | Controller | Key class |
|---|---|---|
| `lqg_controller.py` | Linear Quadratic Gaussian | `LQGController` |
| `palf_lqg_controller.py` | LQG + phase-locked learned feedforward | `PALF_LQG_Controller` |

## LQG Controller

Standard LQG = LQR (state feedback) + Kalman observer.

```python
from lqg_controller import LQGController

lqg = LQGController(plate, dt=5e-5, verbose=True)

# Grid-searched weights (use the SAME weights for the baseline and PALF's internal
# feedback so the comparison is apples-to-apples).
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg.discretize_observer()

# Inside Newmark loop:
x_hat, u = lqg.step(x_hat_prev, u_prev, y_meas)
```

## PALF-LQG Controller

**PALF-LQG** = Phase-Aware Learned Feedforward + LQG.

```
u(t) = u_LQG(x̂)  +  α · u_FF(φ)
       └───┬───┘      └───┬────┘
       reactive       phase-locked learned
       LQG feedback   feedforward (periodic map)
```

**What it actually is** (honest description — see the audit):

- **Feedback**: LQG (LQR gain + Kalman observer), the reactive baseline.
- **Feedforward**: a small MLP, `(n_x+2) → 16 → 1` with **tanh** activations
  (~161 parameters), trained by hand-coded SGD inside an iterative-learning loop.
  Its training data is a function of the tooth-passing phase φ, and at deployment the
  state channel is fed zeros, so the learned object is a **periodic map `u_FF(φ)`**
  (repetitive-control-like), not a state-feedback network.
- **Safety**: a heuristic control-Lyapunov-style **voltage governor** evaluated on the
  *nominal, delay-free* model — a soft actuator guard, NOT a stability certificate for
  the true delayed, time-periodic closed loop.

**It is NOT** "Deep" (one hidden layer), NOT "MPC" (no receding-horizon optimization,
no online cost minimization), and there is NO working online adaptation (the earlier
"RLS/adaptive" path was dead code and has been removed).

```python
from palf_lqg_controller import PALF_LQG_Controller

# SAME feedback weights as the standalone LQG baseline (symmetric comparison)
palf = PALF_LQG_Controller(
    plate, dt=5e-5,
    base_w_q=lqg.w_q, base_w_qd=lqg.w_qd, base_w_r=1.0,
    ff_lr=0.005,           # feedforward learning rate
    ff_max=10.0,           # feedforward output saturation (V)
    ff_alpha=1.0,          # feedforward mixing gain
    n_per=n_per,           # samples per tooth-passing period (phase index)
    safety_alpha=5.0,      # governor sensitivity
    u_max=150.0,           # piezo voltage saturation (V)
)

# Pre-train ONCE on the nominal scenario, then FREEZE (do not retrain per scenario —
# retraining on the evaluation scenario would score the controller on its own data).
palf.pretrain_iterative_simulation(
    simulator, alpha3, alpha4, kp_idx,
    n_iterations=30, n_epochs_per_iter=15,
)

# Inside Newmark loop (k_step drives the phase index):
x_hat, u = palf.step(x_hat_prev, u_prev, y_meas, k_step)
```

## Verified comparison (committed code, P0+P1: held-out, spillover + noise — see docs/REPRODUCED_RESULTS.md)

| Metric | LQG | PALF-LQG | Gain |
|---|---:|---:|---:|
| RMS vibration, nominal (S1) | 0.744 µm | 0.706 µm | +5.1 % |
| RMS vibration, model mismatch ω−8 % (S3) | 0.853 µm | 0.744 µm | **+12.7 %** |
| Modal damping (Mode 1) | LQG closed-loop | **identical** | feedforward does not move poles |
| Stability domain (a_p crit @4900 RPM) | 2.05 mm | **2.05 mm** | feedforward does not shift the boundary |

(Plant carries 5 modes, controller sees 3 — spillover; 10 nm measurement noise;
corrected Eq. 3 forces. Kalman `kalman_V` and clipping `u_max` are constructor args.)

The feedforward buys little on the nominal plant but preserves its gain under model
mismatch — that robustness asymmetry is the result worth reporting. It does **not**
increase modal damping or extend the stability lobe (a phase-locked feedforward changes
the periodic forcing, not the closed-loop poles).
