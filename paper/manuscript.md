# Control-oriented chatter suppression in thin-walled milling: closed-loop full-discretization synthesis with a two-degree-of-freedom architecture

*Working manuscript. All numerical values are produced by `experiments/run_all.py`
and cross-checked in `paper/CORRECTIONS.md`.*

## Abstract

Chatter limits the achievable depth of cut when milling flexible thin-walled
workpieces. Active control with a piezoelectric actuator can enlarge the stable
region, but the *controlled* stability-lobe diagram (SLD) is frequently reported
without actually placing the controller inside the delayed, time-periodic
stability analysis, which invites unphysical claims. We make three contributions
on the cantilever-plate milling model of Du et al. (2024). First, a **closed-loop
full-discretization method (CL-FDM)** embeds an arbitrary linear controller in
the Floquet monodromy matrix, yielding the true controlled SLD; its open-loop
critical depth (~0.15 mm at 4900 rpm) matches the published experimental order.
Second, using CL-FDM as the objective, we **synthesise the feedback gain by
directly maximising the critical depth of cut** under the true actuator-voltage
budget: this reaches a critical depth of 5.31 mm versus 2.43 mm for a
conventional LQG design at the same speed, while the peak voltage stays at 43 V
(well within the +/-150 V limit) -- the LQG gain-norm cap leaves more than half
the achievable stable region unused. Third, we add a
**phase-aware feedforward** in a two-degree-of-freedom architecture that reduces
forced vibration (~4.6 % RMS, ~7-8 % peak) and peak actuator voltage, and we
show — both analytically and with CL-FDM — that this feedforward does **not**
move the stability boundary. All results are computed and reproducible.

## 1. Introduction

Thin-walled parts (aerospace structures, turbine blades, monolithic frames) are
flexible, so milling them excites regenerative chatter at comparatively small
depths of cut. Passive and semi-active remedies (variable-pitch/variable-helix
tools, tuned dampers, added stiffness) help but are case-specific. Active control
with collocated or nearly-collocated actuators offers a general route to enlarge
the stable region.

Du et al. [1] tackled the full difficulty of this problem — non-smooth
intermittent cutting, the regenerative time delay, multi-mode participation and
dynamics that vary along the tool path — with a robust combined time-delay
controller (mu-synthesis plus active time-delay control) and validated it
experimentally, raising the stable depth from 0.1 mm to 0.8 mm while reducing the
control voltage relative to robust control alone.

A recurring weakness in the broader active-chatter-control literature, and in the
simulation package this work started from, is how the *controlled* SLD is
obtained. Chatter stability is governed by the dominant Floquet multiplier of a
periodic delay-differential equation. A controller changes that multiplier only
through the closed-loop dynamics; yet controlled lobes are sometimes produced by
scaling the open-loop modal damping by an arbitrary factor, or by attributing a
stability gain to a feedforward term that cannot move the closed-loop poles. Such
shortcuts can overstate the benefit several-fold (see `CORRECTIONS.md`).

This paper takes the opposite stance: the stability metric used to *design* and
to *report* the controller is the Floquet multiplier of the genuine closed loop.

## 2. Milling / plate / actuator model

We reuse the model of [1] as implemented in `src/`. A cantilever AL6061 plate
(100 x 80 x 4 mm) is discretised with Kirchhoff Q4 plate elements (Hermite shape
functions, 3 dof/node) and reduced to its first three modes (mode 1 at 521 Hz).
The modal equation of motion during peripheral milling is

    Mp q'' + Cp q' + (Kp + a4(t) Dp Dp^T) q - a4(t) Dp Dp^T q(t - tau)
        = f_t a3(t) Dp + H_pe u(t),                                    (1)

where q are modal coordinates, tau = 60/(N_T * RPM) is the tooth-passing period,
a3(t), a4(t) are the periodic cutting-force coefficients of a 3-tooth helical end
mill, Dp is the modal shape vector at the tool tip, H_pe is the modal actuator
input of the piezo patch (d31 coupling), and u is the actuator voltage. The term
`-a4 Dp Dp^T q(t-tau)` is the regenerative feedback that drives chatter.

In state-space form with x = [q; q'],

    x' = A_p(t) x + A_tau(t) x(t - tau) + B u,                          (2)
    A_p  = [[0, I], [-(Kp + a4 Dp Dp^T), -Cp]],
    A_tau= [[0, 0], [ a4 Dp Dp^T, 0]],   B = [0; H_pe],   y = Dp^T q.

Time-domain response uses a Newmark-beta integrator (`newmark_solver.py`) that
stores the delayed state; a realistic piezo model (saturation +/-150 V, slew
rate, first-order amplifier, hysteresis) is available for actuator-limited runs.

## 3. Method

### 3.1 Closed-loop full-discretization (CL-FDM)

For a linear control law combining static state feedback and optional delayed
feedback,

    u(t) = -K x(t) - K_tau x(t - tau),                                  (3)

the closed loop is again a periodic DDE,

    x' = [A_p(t) - B K] x + [A_tau(t) - B K_tau] x(t - tau).            (4)

We discretise one period tau into m sub-intervals; over each the coefficients are
frozen and the delayed state is the corresponding entry of an augmented history
vector x_aug = [x_k; x_{k-1}; ...; x_{k-m}]. Each sub-step contributes

    x_{k+1} = e^{(A_p - B K) dt} x_k
              + [ (A_p - B K)^{-1} (e^{(A_p-BK)dt} - I) ] (A_tau - B K_tau) x_{k-m},

and the product of the m augmented transition matrices over one period is the
monodromy matrix Phi. Milling is stable iff rho(Phi) = max|eig(Phi)| < 1. Setting
K = K_tau = 0 recovers the classical open-loop FDM [2]. The implementation is
`src/cl_fdm.py`; the critical depth `a_p,crit(RPM)` is found by bracketing and
bisecting rho = 1.

The stability analysis uses the static state feedback u = -K x, i.e. the LQR
core; the deployed controller (§3.3) reconstructs the state with a Kalman
observer, u = -K x_hat. By the separation principle the observer contributes its
own (fast, by design) eigenvalues, so the dominant stability boundary is set by
the state-feedback loop analysed here; a full observer-in-the-loop FDM (doubling
the augmented state) is a straightforward extension left for future work. This is
the one modelling approximation in the stability numbers and is stated again in
§5.

**Validation.** With K = 0 the method gives a_p,crit ≈ 0.15 mm at 4900 rpm, of
the same order as the ~0.1 mm measured in [1], confirming the assembly is
physically correct. (A single actuator can only oppose the regenerative term
within its own input subspace, so delayed feedback K_tau was found to give only
marginal, fragile gains here and is not used in the final controller; the option
is retained in the code for completeness.)

### 3.2 Floquet-direct feedback synthesis

An LQG/LQR gain minimises a quadratic cost, i.e. it pushes the non-delayed
closed-loop eigenvalues left under a gain-norm cap — an *eigenvalue proxy* for
chatter stability. We instead score each candidate gain by the quantity that
actually matters, `a_p,crit` from CL-FDM, and select

    K* = argmax_K  a_p,crit(K)   s.t.  ||K|| <= gain_cap.               (5)

Candidate gains are generated as output-weighted LQR solutions over a grid of
weights (`floquet_synthesis.py`); the eigenvalue-proxy rule and the Floquet rule
are then compared on the *same* candidate set, and each selected design is run in
the time domain to obtain its peak actuator voltage. This exposes the
stability/effort trade-off directly — the metric of practical interest, since [1]
emphasises voltage reduction.

### 3.3 Two-degree-of-freedom controller

The final controller (`twodof_control.py`) is

    u = -K x_hat + alpha * u_ff(phi, x_hat),                            (6)

with x_hat from a Kalman observer, K the synthesised feedback, and u_ff a small
phase-aware feedforward network trained by iterative simulation to anticipate the
periodic cutting force at the tooth-passing phase phi. A Lyapunov inequality
filter keeps the combined command within the feedback's decay region. Crucially,
u_ff is a feedforward signal: it is absent from the characteristic equation of
(4), so it cannot change rho(Phi) and therefore cannot move the stability
boundary. Its purpose is to reduce the *forced* vibration and the peak voltage
inside the stable region — which we quantify separately in §4.

## 4. Results

All numbers below are produced by `experiments/run_all.py` (spindle speed
4900 rpm; CL-FDM with m = 30 sub-intervals; time-domain with dt = 50 us).

### 4.1 Controlled stability-lobe diagram

CL-FDM validates against the physics in open loop and quantifies the feedback
benefit honestly (Fig. 1):

| Configuration | a_p,crit at 4900 rpm | vs open loop |
|---|---:|---:|
| Open loop | 0.15 mm | 1x |
| LQG (closed-loop FDM) | 2.43 mm | ~16x |
| Voltage-budget design (§4.2) | 5.31 mm | ~35x |

The open-loop value is of the same order as the ~0.1 mm measured in [1]. The LQG
extension (~16x) is genuine and large; crucially it is *computed with the
controller in the monodromy matrix*, not scaled. (For reference, the package this
work started from reported a controlled boundary of 3.05 mm obtained by scaling
the modal damping by 1.30 — an operation with no physical basis; see
`CORRECTIONS.md`.)

### 4.2 Feedback authority vs stability, and the voltage budget

Sweeping the damping-authority weight and scoring each gain by CL-FDM gives the
design curve a_p,crit(||K||) (Fig. 2). Two points matter:

| Design | ||K|| | a_p,crit | peak voltage at a_p = 0.6 mm |
|---|---:|---:|---:|
| LQG (its own rule, ||K|| <= 1e8) | 5.8e7 | 2.43 mm | 32 V |
| Voltage-budget pick (this work) | 4.6e8 | **5.31 mm** | **43 V** |

The conventional LQG rule caps the gain norm at ~1e8 and so stops at 2.43 mm,
even though the *actual* actuator constraint (+/-150 V) is nowhere near active:
the peak voltage is only 32 V. Selecting the gain directly on the CL-FDM critical
depth under the true voltage budget reaches 5.31 mm — more than double the stable
depth — while the peak voltage rises only to 43 V, still a factor ~3.5 below
saturation. The gain-norm cap is thus a poor surrogate for the physical limit;
CL-FDM lets the authority be spent where it actually buys stability.

### 4.3 Forced-vibration and voltage reduction (2-DOF vs LQG)

The phase-aware feedforward reduces forced vibration and peak voltage relative to
LQG feedback alone (Fig. 3):

| Scenario | LQG y_RMS | 2-DOF y_RMS | RMS gain | peak gain | u_max change |
|---|---:|---:|---:|---:|---:|
| S1 Nominal | 0.532 um | 0.507 um | +4.7 % | +7.0 % | -1.7 % |
| S2 Aggressive (a_p = 0.6 mm) | 1.058 um | 1.009 um | +4.6 % | +8.3 % | -7.1 % |
| S3 Uncertainty (omega -15 %) | 0.606 um | 0.488 um | **+19.5 %** | +7.2 % | -2.1 % |
| S4 High K_T (+30 %) | 0.692 um | 0.660 um | +4.7 % | +7.7 % | -4.1 % |

Under nominal and matched conditions the feedforward gives a modest, consistent
~4.6 % RMS and ~7-8 % peak reduction at equal-or-lower voltage. The largest
benefit (+19.5 %) appears under a -15 % modal-frequency drift (S3): the LQG
feedback is detuned by the mismatch, whereas the feedforward is locked to the
tooth-passing phase (independent of the plant frequency) and keeps cancelling the
periodic force. This is the honest counterpart of the uniform ~19 % that the
original package claimed for *every* scenario: the ~19 % is real, but only under
frequency uncertainty, not at the nominal operating point.

### 4.4 Role of the feedforward: what it can and cannot do

With the feedback gain fixed, adding the feedforward changes the forced response
but not the stability boundary. At S1: y_RMS 0.532 -> 0.507 um and peak
1.93 -> 1.79 um, while a_p,crit is unchanged at 2.43 mm (the feedforward is not in
the feedback path, so it cannot move the Floquet multipliers). Stability therefore
comes only from the feedback design of §4.2; the feedforward buys forced-vibration
and voltage margin. Conflating the two — attributing a stability gain to the
feedforward — is exactly the error corrected here.

## 5. Discussion

The results should be read as a *simulation* study of achievable performance:
CL-FDM (like all FDM) is a linear stability analysis and does not include
actuator saturation, sensor noise or the amplifier dynamics that the realistic
piezo model captures; simulated stable depths are therefore optimistic relative
to the experiments of [1]. The contribution is methodological — a controlled SLD
that is computed rather than asserted, a feedback design driven by the true
stability metric, and an honest accounting of what feedback versus feedforward
can and cannot do — rather than a claim of a specific machine-tool improvement.

## 6. Conclusion

We presented a closed-loop full-discretization method that embeds the controller
in the Floquet analysis, a feedback synthesis that maximises the critical depth
of cut directly, and a two-degree-of-freedom controller whose feedforward role
(forced-vibration and voltage reduction, not stability) is quantified honestly.
Every reported number is reproducible from the code.

## References

[1] J. Du, X. Liu, H. Dai, X. Long. Robust combined time delay control for
milling chatter suppression of flexible workpieces. *International Journal of
Mechanical Sciences* 274 (2024) 109257.

[2] T. Insperger, G. Stepan. Updated semi-discretization method for periodic
delay-differential equations with discrete delay. *International Journal for
Numerical Methods in Engineering* 61 (2004) 117-141.

[3] Y. Altintas. *Manufacturing Automation*, 2nd ed., Cambridge University Press,
2012.
