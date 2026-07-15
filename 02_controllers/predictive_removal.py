"""
predictive_removal.py
=====================
P7 — a material-removal-aware PREDICTIVE control strategy and its rigorous
feasibility analysis for thin-wall milling.

THE REQUESTED CHAIN. "A control strategy that accounts for the material removed
during cutting: the milling model removes material by precise physical equations ->
the plate properties are recomputed -> the vibration is predicted -> and suppressed."

WHAT THE INVESTIGATION FOUND (two physical walls — see docs/REPRODUCED_RESULTS.md P7):
  1. TIMESCALE. At the article feed the tool advances 0.245 um per 50-us step and
     takes ~13,605 steps (167 tooth-periods, 0.68 s) to cross ONE FEM mesh column;
     the plant (M, K, C) is piecewise-constant at mesh resolution and cannot change
     between steps. A genuine PER-STEP property update captures ~1e-5 %/step — over-
     engineering by ~1e4. The physically-correct cadence is EVENT-DRIVEN (a re-solve
     when the tool crosses into a new mesh column, ~0.68 s apart); a per-tooth-period
     update is already 167x finer than necessary.
  2. REGIME CONFLICT. Per-step removal is significant only for DEEP cuts (a_p >= 10 mm:
     >0.5 % within-pass drift; it first exceeds 1 % only at a_p = 40 mm), but those are
     catastrophically chatter-unstable open loop (Floquet radius rho ~ 3.5 at a_p=2 mm,
     ~1.1e5 at 10 mm, ~7.9e16 at 40 mm — far beyond +-150 V piezo authority). The
     CONTROLLABLE regime (a_p <= ~2 mm, rho_OL <~ 3.5) is exactly where within-pass
     removal is NEGLIGIBLE (<=0.2 %). CAVEAT: the reported ~0.000 % for a_p <= 1 mm is
     partly a mesh-resolution floor — on the 24-row height mesh (element row ~3.33 mm)
     the thinning band `ez >= hp - a_p` captures no element centroid until a_p ~ 1.67 mm,
     so sub-mm depths quantise to exactly zero removed elements; the true drift there is
     nonetheless bounded above by the resolved 0.2 % at a_p = 2 mm and is physically
     negligible either way. On this plant the two requirements are mutually exclusive.

Hence the honest deliverable is a FEASIBILITY STUDY plus the concrete artifacts:
  * the x-resolved moving-front removal model (01_core/material_removal.remove_moving_front),
  * this PREVIEW-PREDICTIVE controller, and
  * the honest boundary result (the two walls above + the comparison below).

THE PREVIEW-PREDICTIVE CONTROLLER (PreviewPredictiveController). The regenerative
force at step k is a4[k]*Dp*Dp^T*q(k-n_tau) and the feed force is ft*a3[k]*Dp; BOTH
are computable one tooth-period ahead — the feed force is periodic and known, and the
regenerative force uses the DELAYED state q(k-n_tau), which for a horizon H <= n_tau
is entirely PAST (measured/estimated). So the incoming modal disturbance over the
horizon is PREVIEWABLE. A receding-horizon quadratic program
    min_U  sum_j y_pred[j]^2 + r * sum_j u[j]^2
    y_pred = Psi_x x_hat + Psi_u U + Psi_d D_preview
(the plant is LTI over the short horizon; Psi_* are precomputed impulse-response
operators) gives U; the first move is applied (receding horizon). x_hat comes from
the same Kalman observer as LQG.

HONEST RESULT (a_p = 0.3 mm, controllable): the preview law is stable and marginally
beats LQG (mismatched model 0.68 vs 0.78 um, +12 %; exact-model upper bound 0.72 um)
by pre-empting the known disturbance, but it does NOT approach the HRC controller
(0.39 um), which cancels the periodic forcing far better with sharp resonant filters.
Predict-from-physics is also WEAKER than P6's probe
wherever the cutting model is uncertain (a 15 % KT error = 15 % predicted-disturbance
error). So on this plant the preview controller is a sound but non-dominant option;
its documented niche is ZERO-LATENCY action (it computes the compensation from the
calibrated force model from step 0, needing neither a converging probe nor built-up
vibration).

ANTI-INVERSE-CRIME. The controller's internal cutting model (PhysicsCuttingModel)
is MISMATCHED from the plant: KT, k1, k2 are perturbed (+-15 %), the bite is the
nominal (vibration-free) tool projection, and the law is designed on 3 modes while
the plant carries 5 (spillover). NOTE on scope: the mismatch driver here perturbs
the FORCE model and the mode count, NOT the modal frequencies — the controller's
structural (M, K, C) view is the plant's exact 3-mode truncation. This is a partial
(force-and-spillover) mismatch, honestly a weaker inverse-crime guard than also
perturbing the plant frequencies; the reported "Preview (mismatched)" bar therefore
still benefits from exact structural knowledge. The "Preview (exact model)" bar shares
the plant's force model too and is included ONLY as a clearly-labelled upper-bound
reference (an inverse crime), never as the honest result.
"""
import numpy as np
from scipy.linalg import expm


class PhysicsCuttingModel:
    """Controller-internal calibrated cutting model that PREDICTS the modal
    disturbance preview (feed + regenerative) from the physical force equations.
    Kept deliberately separable so it can be MISMATCHED from the plant (wrong KT /
    k1,k2 / nominal bite) to avoid an inverse crime.

    Parameters
    ----------
    a3_model, a4_model : (nstep,) the controller's periodic force coefficients
        (from `milling_force.precompute_alpha_periodic` with the controller's — not
        the plant's — KT/k1/k2). ft : feed per tooth (m). Dp0 : (n,) modal tool
        projection assumed by the controller (nominal position). n_tau : delay steps.
    """

    def __init__(self, a3_model, a4_model, ft, Dp0, n_tau):
        self.a3 = np.asarray(a3_model, float)
        self.a4 = np.asarray(a4_model, float)
        self.ft = float(ft)
        self.Dp0 = np.asarray(Dp0, float)
        self.n = len(self.Dp0)
        self.DpDp = np.outer(self.Dp0, self.Dp0)
        self.n_tau = int(n_tau)

    def preview(self, k, H, qbuf, qbuf_len):
        """Modal disturbance d_j (n,) for j = 0..H-1 at absolute steps k..k+H-1.
        qbuf is the controller's ring buffer of past modal-state estimates q_hat."""
        D = np.zeros(H * self.n)
        N = len(self.a3)
        for j in range(H):
            kj = min(k + j, N - 1)
            dj = self.ft * self.a3[kj] * self.Dp0          # KNOWN periodic feed force
            idx = k + j - self.n_tau
            if idx >= 0:                                   # regenerative preview
                qd = qbuf[idx % qbuf_len]
                dj = dj + self.a4[kj] * (self.DpDp @ qd)
            D[j * self.n:(j + 1) * self.n] = dj
        return D


class PreviewPredictiveController:
    """Receding-horizon preview-predictive suppressor (see module docstring).
    n_x = 2n (Kalman modal state). Uses a Kalman observer (from an LQGController) and
    a PhysicsCuttingModel for the disturbance preview."""

    def __init__(self, design, dt, cutting_model, lqg, horizon=None, r=1e-14,
                 u_max=150.0, verbose=True):
        self.design = design
        self.dt = float(dt)
        self.cm = cutting_model
        self.n = design.n_modes
        self.n_x = 2 * self.n
        self.u_max = float(u_max)
        n = self.n
        # continuous modal SS: x=[q;qd]; xdot=A x + B u + Ed*d_modal, y=C x
        Km = np.linalg.solve(design.Mp, design.Kp)
        Cm = np.linalg.solve(design.Mp, design.Cp)
        H = np.linalg.solve(design.Mp, np.asarray(design.H_Pe_modal, float))
        Dob = np.asarray(design.D_obs, float)
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -Km
        A[n:, n:] = -Cm
        B = np.zeros(2 * n); B[n:] = H
        C = np.zeros(2 * n); C[:n] = Dob
        Ed = np.zeros((2 * n, n)); Ed[n:, :] = np.eye(n)
        # ZOH discretise [A | B Ed]
        m = 2 * n
        Mbig = np.zeros((m + 1 + n, m + 1 + n))
        Mbig[:m, :m] = A; Mbig[:m, m] = B; Mbig[:m, m + 1:] = Ed
        E = expm(Mbig * self.dt)
        self.Ad = E[:m, :m]; self.Bd = E[:m, m]; self.Edd = E[:m, m + 1:]
        self.C = C
        # Kalman observer matrices (reuse the LQG design)
        self.A_obsd = lqg.A_obs_d
        self.Gu = lqg.G_u.flatten()
        self.Gy = lqg.G_y.flatten()
        # horizon operators
        H_h = int(horizon) if horizon is not None else self.cm.n_tau
        self.H = H_h
        self._build_horizon(H_h)
        self.Gain = np.linalg.solve(self.Psi_u.T @ self.Psi_u + r * np.eye(H_h),
                                    self.Psi_u.T)
        self._qbuf = np.zeros((self.cm.n_tau + H_h + 2, n))
        self._k = 0
        if verbose:
            print(f"[Preview-MPC] horizon H={H_h} steps ({H_h*self.dt*1e3:.2f} ms, "
                  f"n_tau={self.cm.n_tau}), r={r:.1e}")

    def _build_horizon(self, H):
        n = self.n
        m = 2 * n
        Psi_x = np.zeros((H, m))
        Psi_u = np.zeros((H, H))
        Psi_d = np.zeros((H, H * n))
        Apow = [np.eye(m)]
        for _ in range(H):
            Apow.append(self.Ad @ Apow[-1])
        for k in range(H):
            Psi_x[k] = self.C @ Apow[k + 1]
            for j in range(k + 1):
                cA = self.C @ Apow[k - j]
                Psi_u[k, j] = cA @ self.Bd
                Psi_d[k, j * n:(j + 1) * n] = cA @ self.Edd
        self.Psi_x, self.Psi_u, self.Psi_d = Psi_x, Psi_u, Psi_d

    def reset(self):
        self._qbuf[:] = 0.0
        self._k = 0

    def step(self, x_prev, u_prev, y_meas):
        x = self.A_obsd @ x_prev + self.Gu * u_prev + self.Gy * y_meas
        self._qbuf[self._k % len(self._qbuf)] = x[:self.n]
        D = self.cm.preview(self._k, self.H, self._qbuf, len(self._qbuf))
        U = -self.Gain @ (self.Psi_x @ x + self.Psi_d @ D)
        u = float(np.clip(U[0], -self.u_max, self.u_max))
        self._k += 1
        return x, u
