# 02_controllers — Control Algorithms

This directory contains the **three controllers** compared in the study:
LQG (baseline), IMC-LQG (internal-model feedback baseline), and DARC
(proposed).

| File | Controller | Key class |
|---|---|---|
| `lqg_controller.py` | Linear Quadratic Gaussian | `LQGController` |
| `imc_lqg_controller.py` | LQG + harmonic internal model (disturbance-accommodating LQG) | `IMCLQGController` |
| `darc_controller.py` | DARC: Deep Anticipative Residual Control (proposed) | `DARCController` |

## LQG Controller (baseline)

LQR (state feedback) + observer, u = −K·x̂.

```python
from lqg_controller import LQGController

lqg = LQGController(plate, dt=5e-5, verbose=True)
lqg.optimize_weights(w_q_list=[1e14], w_qd_list=[1e8], w_r=1.0,
                     gain_norm_max=1e10)   # the shared base design
lqg.discretize_observer()
# inside the Newmark loop:
x_hat, u = lqg.step(x_hat_prev, u_prev, y_meas)
```

Honest caveats: the observer covariances (W = 1e-6·I, V = 1e-12) are
hard-coded, not derived from the actual disturbance/sensor statistics — no
Kalman-optimality claim should be made; and the observer carries no model of
the deterministic tooth-passing force, so its state estimates degrade under
cutting (this is exactly what IMC-LQG fixes).

## IMC-LQG (internal-model feedback baseline)

The answer to "feedback cannot reject a periodic disturbance" — it can,
whenever the tooth-passing PERIOD is known (spindle encoder, the same
assumption DARC's feedforward makes). The Kalman filter is augmented with
harmonic oscillator states at the tooth-passing harmonics; the estimated
disturbance is cancelled through the exact regulator-equation gains.
**Needs no cutting-force model at all.**

```python
from imc_lqg_controller import IMCLQGController

imc = IMCLQGController(plate, dt=5e-5,
                       f_fund=1/(n_per*dt),   # simulated tooth frequency
                       Dp_dist=Dp_tool,       # disturbance direction
                       n_harm=8, q_dist=1e3)
imc.reset()                                   # REQUIRED before each run
x_hat, u = imc.step(x_hat_prev, u_prev, y_meas)
```

Measured trade-off (see `05_main/main_imc_baseline.py`): best rejection in
S1/S2/S4 (adapts online to unknown K_T), but its nominal-model inversion
mis-phases under −15 % structural detuning (S3) where it degrades severely;
it also relies on the displacement sensor (unlike DARC's feedforward).

## DARC Controller (proposed)

**DARC = Deep Anticipative Residual Control** — three ADDITIVE layers:

```
u(t) = u_LQG(x̂)  +  u_FF(φ)  +  u_NN(x̂, φ)
       └───┬───┘     └──┬───┘     └───┬────┘
       reactive     inverse-model   NN residual
       base (=LQG   feedforward,    (ILC-trained,
       baseline)    phase-locked    validation-
                    to spindle      checkpointed)
```

There is **no MPC** in this controller (no horizon, no online optimization,
no constrained QP) — the former name "DARC-MPC" misrepresented the method
and was dropped, together with the dead "adaptive RLS" and Lyapunov-filter
code.

```python
from darc_controller import DARCController

darc = DARCController(plate, dt=5e-5,
                      base_w_q=1e14, base_w_qd=1e8, base_w_r=1.0,
                      ff_max=20.0, n_per=n_per, u_max=150.0)

# Feedforward from the NOMINAL cutting model (never the true perturbed one)
darc.design_periodic_feedforward(FT, a3_nominal[:n_per], Dp_ff, n_harm=30)

# NN residual: trained in the controller's nominal world, with training and
# VALIDATION noise realizations disjoint from the evaluation seed
darc.train_nn_residual(sim_design, a3_nominal, a4_nominal, kp_idx,
                       alpha4_2_t=a42_n, alpha4_3_t=a43_n,
                       n_iter=20, sensor_noise=1e-7,
                       train_seed=100, val_seed=200)

# inside the Newmark loop:
x_hat, u = darc.step(x_hat_prev, u_prev, y_meas, k_step=k)
```

Honest caveats:
- The feedforward assumes a PERFECT spindle-phase reference (modulo-n_per
  indexing). Measured sensitivity: beyond ~30–40° of tooth-passing phase
  error the mis-phased feedforward becomes WORSE than plain LQG — a phase
  error/jitter study belongs next to any deployment claim.
- The out-of-sample NN gain is a few percentage points (see the full-path
  study), not the in-sample per-scenario figure.

## Performance

See the root README "Key Results" tables — all numbers there are produced
by `05_main/main_simulation.py` and `05_main/main_imc_baseline.py` under the
honest protocol (nominal-model controller design, disjoint train/val/eval
seeds).
