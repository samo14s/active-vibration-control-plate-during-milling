# Control-oriented chatter suppression in thin-walled milling: closed-loop semi-discretization and an active-disturbance-rejection controller

*Working manuscript. All numerical values are produced by `experiments/run_all.py`
and cross-checked in `paper/CORRECTIONS.md`.*

## Abstract

Chatter limits the achievable depth of cut when milling flexible thin-walled
workpieces, and the difficulty is compounded by dynamics that vary along the tool
path. Active control with a piezoelectric actuator can enlarge the stable region,
but the *controlled* stability-lobe diagram (SLD) is frequently reported without
placing the controller inside the delayed, time-periodic stability analysis,
which invites unphysical claims. On the cantilever-plate milling model of
Du et al. (2024) we make four contributions, each computed and reproducible.
**(i)** A *closed-loop semi-discretization* (CL-SD) method embeds an arbitrary
linear controller -- including its observer -- inside the Floquet monodromy
matrix, giving the true controlled SLD; its open-loop critical depth
(0.063 mm at 4900 rpm) matches the published experimental order (~0.1 mm).
**(ii)** Using CL-SD we show that the Kalman observer materially reduces the LQG
stability margin (1.92 mm with the observer in the loop, versus 2.43 mm for the
idealised static-feedback bound). **(iii)** We design an *active disturbance
rejection control* (ADRC) with a collocated piezo sensor: it raises the critical
depth to 3.25 mm, reduces tip vibration by ~51 % versus LQG at 25-55 V, and --
because it needs only the input gain b0 -- stays robust when the plant frequency
drifts, whereas LQG loses stability at -20 % drift. **(iv)** A phase-aware
feedforward in a two-degree-of-freedom controller is shown, analytically and with
CL-SD, to reduce forced vibration and peak voltage but *not* the stability
boundary. We also document and correct the fabricated results of the package this
work started from (`CORRECTIONS.md`).

## 1. Introduction

Thin-walled parts (aerospace structures, turbine blades, monolithic frames) are
flexible, so milling them excites regenerative chatter at comparatively small
depths of cut, and the modal properties change as material is removed and the
tool advances. Passive and semi-active remedies (variable-pitch/variable-helix
tools, tuned dampers, added stiffness) help but are case-specific. Active control
offers a general route to enlarge the stable region.

Du et al. [1] addressed the full difficulty of this problem -- non-smooth
intermittent cutting, the regenerative time delay, multi-mode participation and
varying dynamics -- with a robust combined time-delay controller (mu-synthesis
plus active time-delay control) and validated it experimentally, raising the
stable depth from 0.1 mm to 0.8 mm while reducing the control voltage relative to
robust control alone.

Two gaps motivate this work. First, controlled stability lobes are often obtained
by shortcuts (scaling the open-loop modal damping, or crediting a feedforward
term with a stability gain it cannot produce); such shortcuts can overstate the
benefit several-fold (see `CORRECTIONS.md`). We therefore make the Floquet
multiplier of the genuine closed loop the object that is both computed and
designed against. Second, model-based designs (LQG, mu-synthesis) depend on an
accurate, fixed model, which the varying dynamics undermine. Active disturbance
rejection control [4,5] replaces the model with an online estimate of the "total
disturbance", needing only the control gain b0, and is a natural fit for this
regime -- which is the contribution we develop and evaluate here.

## 2. Milling / plate / actuator model

We reuse the model of [1] as implemented in `src/`. A cantilever AL6061 plate
(100 x 80 x 4 mm) is discretised with Kirchhoff Q4 plate elements (Hermite shape
functions, 3 dof/node) and reduced to its first three modes (mode 1 at 521 Hz).
The modal equation of motion during peripheral milling is

    Mp q'' + Cp q' + (Kp + a4(t) Dp Dp^T) q - a4(t) Dp Dp^T q(t - tau)
        = f_t a3(t) Dp + H_pe u(t),                                    (1)

with modal coordinates q, tooth-passing period tau = 60/(N_T*RPM), periodic
cutting coefficients a3(t), a4(t) of a 3-tooth helical end mill, tool-tip modal
shape Dp, piezo modal input H_pe, and voltage u. The term
`-a4 Dp Dp^T q(t-tau)` is the regenerative feedback that drives chatter. In
state-space with x = [q; q']:

    x' = A_p(t) x + A_tau(t) x(t - tau) + B u,                          (2)
    A_p  = [[0, I], [-(Kp + a4 Dp Dp^T), -Cp]],  A_tau = [[0,0],[a4 Dp Dp^T,0]],
    B = [0; H_pe],   y = C x.

Time-domain response uses a Newmark-beta integrator (`newmark_solver.py`) storing
the delayed state; a realistic piezo model (saturation +/-150 V, slew rate,
amplifier, hysteresis) is available for actuator-limited runs.

## 3. Method

### 3.1 Closed-loop semi-discretization (CL-SD)

For any linear controller with realisation z' = Ac z + Bc y, u = Cc z + Dc y
(static state feedback is the special case z = x, Ac,Bc absent), the closed loop
is a periodic delay-differential equation in the augmented state [x; z]; the delay
acts only on x. Following the semi-discretization idea [2], one period tau is
split into m sub-intervals; over each the coefficients are frozen and the delayed
state is the corresponding node of an augmented history vector. The product of the
m augmented transition matrices is the monodromy Phi, and milling is stable iff
the spectral radius rho(Phi) < 1. The implementation is `src/cl_fdm.py`
(`rho`/`ap_crit` for state feedback, `rho_dynamic`/`ap_crit_dynamic` for dynamic
output feedback); the critical depth a_p,crit(RPM) is found by bracketing and
bisecting rho = 1. (The scheme is the 0th-order semi-discretization of Insperger
and Stepan [2]; the module name is retained for brevity.)

**Validation.** With no controller the method gives a_p,crit = 0.063 mm at
4900 rpm, of the same order as the ~0.1 mm measured in [1], confirming the
assembly. Because the method admits *dynamic* controllers, the controller's
observer is placed inside the monodromy -- so the reported margins are those of
the deployed observer-based loop, not an idealised full-state bound.

### 3.2 Active disturbance rejection control (ADRC)

ADRC [4] treats everything except the known control channel as a single total
disturbance f(t), estimates it online, and cancels it. The plant from the piezo
voltage u to a measured displacement y = c^T q has relative degree two:

    y'' = f(t) + b0 u,        b0 = c^T H_pe .                           (3)

A third-order linear extended state observer (ESO) estimates (y, y', f) with a
triple pole at -wo, and the control law imposes a double pole at -wc while
cancelling the estimate:

    u = ( -wc^2 y_hat - 2 wc y_hat' - f_hat ) / b0 .                    (4)

Since f_hat already contains the regenerative and periodic cutting forces, a
single ADRC loop both extends the stability boundary and rejects the forced
vibration, and it needs only b0 -- no modal model. Implementation:
`src/adrc_control.py`; `export_lti` returns (Ac,Bc,Cc,Dc) for CL-SD so the ESO is
included in the stability analysis of Sec. 4.

**Sensor placement.** The performance output is the tool tip, but the tip is
*non-collocated* with the piezo (lower-left patch): its input-output map is
non-minimum-phase with sign-indefinite modal coupling ([-,+,-]), which no
single-b0 ADRC can stabilise. We therefore use a *collocated* control sensor at
the piezo corner, whose map is minimum-phase with sign-definite coupling
([-,-,-]) -- the standard configuration for a piezo actuator-sensor pair -- while
performance is still evaluated at the tip. Bandwidths wc = 1500 rad/s,
wo = 18000 rad/s were selected from the CL-SD stability curve and the
actuator-voltage limit; higher bandwidths destabilise through the delay
interaction and are rejected.

### 3.3 Two-degree-of-freedom controller (feedback + phase-aware feedforward)

For comparison we also retain a 2-DOF controller,
u = -K x_hat + alpha * u_ff(phi, x_hat), where the feedback K (LQR core, Kalman
estimate x_hat) sets stability and u_ff is a small phase-aware network trained by
iterative simulation to anticipate the periodic cutting force. Being a feedforward
signal, u_ff is absent from the characteristic equation of (2) and therefore
cannot change rho(Phi); its role is confined to forced vibration and peak voltage
(quantified in Sec. 4.4). Implementation: `src/twodof_control.py`.

### 3.4 Feedback-authority design curve (supplementary)

CL-SD also exposes the static-feedback trade-off a_p,crit(||K||)
(`src/floquet_synthesis.py`). It confirms that the gain-norm cap of a
conventional LQG search is an arbitrary limit rather than the physical one, but
the static bound is optimistic: the realizable depth is set by the observer
(Sec. 4.1) and by the +/-150 V budget at the actual cutting depth. We therefore
report it as a design aid, not as an achievable operating point.

## 4. Results

All numbers are produced by `experiments/run_all.py` (4900 rpm; CL-SD with m = 30;
time domain with dt = 50 us; performance measured at the tool tip).

### 4.1 Controlled stability-lobe diagram and the cost of the observer

| Configuration | a_p,crit at 4900 rpm | vs open loop |
|---|---:|---:|
| Open loop | 0.063 mm | 1x |
| LQG, static-feedback bound | 2.43 mm | 39x |
| LQG, Kalman observer in loop | 1.92 mm | 31x |
| ADRC, ESO in loop | **3.25 mm** | **52x** |

The open-loop value validates the method against [1]. Putting the observer inside
the monodromy lowers the LQG margin from 2.43 to 1.92 mm -- an honest ~21 %
penalty that the usual static-feedback SLD hides. ADRC, evaluated on the same
footing (its ESO in the loop), reaches 3.25 mm, above LQG across the speed range
(Fig. 1). (For reference, the package this work started from reported a controlled
boundary of 3.05 mm obtained by scaling the modal damping by 1.30, an operation
with no physical basis; see `CORRECTIONS.md`.)

### 4.2 Vibration and voltage: ADRC vs LQG

Measured at the tool tip, ADRC roughly halves the vibration relative to LQG at
modest voltage (Fig. 3):

| Scenario | LQG y_RMS | ADRC y_RMS | RMS reduction | ADRC u_max |
|---|---:|---:|---:|---:|
| S1 Nominal | 0.532 um | 0.262 um | +50.8 % | 25 V |
| S2 Aggressive (a_p = 0.6 mm) | 1.058 um | 0.515 um | +51.3 % | 55 V |
| S3 Uncertainty (omega -15 %) | 0.606 um | 0.294 um | +51.5 % | 27 V |
| S4 High K_T (+30 %) | 0.692 um | 0.341 um | +50.8 % | 34 V |

ADRC actively rejects the periodic cutting force (via f_hat), whereas LQG only
adds damping; hence the large, consistent reduction, achieved well within the
+/-150 V budget.

### 4.3 Robustness to varying dynamics

Both controllers are designed on the nominal plant; the real plant's modal
frequencies are then drifted (Fig. 2). ADRC keeps the tip RMS between 0.21 and
0.32 um across a +/-20 % drift. LQG stays near 0.5 um for small drifts but **loses
stability at -20 % (tip RMS 865 um -- chatter)**, because its fixed model is now
wrong. This is the practical statement of ADRC's model-independence and directly
addresses the varying-dynamics challenge emphasised by [1].

### 4.4 Role of the feedforward (2-DOF)

With the feedback fixed, adding the phase-aware feedforward changes the forced
response but not the boundary: at S1, y_RMS 0.532 -> 0.507 um and peak
1.93 -> 1.79 um, while a_p,crit is unchanged at 2.43 mm (the feedforward is not in
the feedback path). The 2-DOF vibration reduction is therefore modest
(+4.6 % RMS nominal, up to +19.5 % under frequency uncertainty where the phase
lock helps), and much smaller than ADRC's -- confirming that stability and
disturbance rejection must come from feedback, which is where ADRC acts.

### 4.5 Feedback-authority curve (supplementary)

The static a_p,crit(||K||) curve rises from ~2.3 mm to ~5 mm as the gain norm
grows (Fig. 5), showing the LQG gain-norm cap is arbitrary. However, as Sec. 4.1
shows for the observer and as the voltage curve in Fig. 5 shows for the actuator,
the *realizable* depth is well below this static ceiling; the curve is a design
aid, not an operating envelope.

## 5. Discussion

These are *simulation* results. CL-SD, like all Floquet stability analysis, is
linear and does not include actuator saturation, sensor noise, or amplifier
dynamics (the realistic piezo model captures those in time domain); simulated
stable depths are therefore optimistic relative to the experiments of [1], and
should be read as relative comparisons under identical assumptions. The ADRC
comparison uses a collocated control sensor (natural for a piezo transducer)
while the model-based controllers use the tip measurement with a full-state
observer; performance for all is read at the tip. The static feedback-authority
curve (Sec. 4.5) is an idealised bound, not a realizable envelope. The
contribution is a controlled SLD that is computed rather than asserted, an ADRC
design suited to varying dynamics, and an honest accounting of what feedback
versus feedforward, and observer versus static feedback, can and cannot do.

## 6. Conclusion

We presented a closed-loop semi-discretization method that embeds the controller
and its observer in the Floquet analysis, quantified the observer's cost to the
LQG stability margin, and designed an ADRC controller that -- needing only the
input gain -- extends the critical depth (3.25 vs 1.92 mm), roughly halves tip
vibration at low voltage, and remains stable under frequency drift that
destabilises LQG. A two-degree-of-freedom feedforward was shown to help forced
vibration but not stability. Every reported number is reproducible from the code.

## References

[1] J. Du, X. Liu, H. Dai, X. Long. Robust combined time delay control for
milling chatter suppression of flexible workpieces. *International Journal of
Mechanical Sciences* 274 (2024) 109257.

[2] T. Insperger, G. Stepan. Updated semi-discretization method for periodic
delay-differential equations with discrete delay. *International Journal for
Numerical Methods in Engineering* 61 (2004) 117-141.

[3] Y. Altintas. *Manufacturing Automation*, 2nd ed., Cambridge University Press,
2012.

[4] J. Han. From PID to active disturbance rejection control. *IEEE Transactions
on Industrial Electronics* 56 (2009) 900-906.

[5] Z. Gao. Scaling and bandwidth-parameterization based controller tuning.
*Proc. American Control Conference* (2003) 4989-4996.
