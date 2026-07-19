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
rejection control* (ADRC) with a collocated piezo sensor: it raises the linear
critical depth to 3.25 mm, reduces tip vibration by ~51 % versus LQG at 25-55 V,
and -- because it needs only the input gain b0 -- stays robust when the plant
frequency drifts, whereas LQG loses stability at -20 % drift; a full-process
weakness of its single collocated sensor (spatial observability) is then
removed by AFC-ADRC, a spindle-synchronous adaptive feedforward comb on the
existing tip sensor that cuts the end-to-end process vibration ~3x below both
baselines (0.15 vs 0.47-0.53 um) at negligible voltage cost and with the
fast-loop stability untouched. **(iv)** The Kirchhoff model itself is refined to *precise* level: the piezo
patch's mass and bending stiffness are added as a composite section with a
shifted neutral axis, which moves the chatter-dominant mode 1 from -3.5 % error
to **+0.12 %** against the measured 540 Hz and makes the five-mode mean error
(1.44 %) smaller than the source article's own theoretical model (1.93 %); mesh
convergence, density sensitivity, modal-truncation (spillover) and
sampling-resolution effects are quantified, exposing a ~20 kHz controller-rate
requirement and moderating the ADRC advantage to +28 % on the refined plant.
**(v)** We introduce a *two-stage stability metric*: the linear CL-SD boundary
must be paired with the **voltage-feasible critical depth** -- the largest depth
at which the saturated (+/-150 V) nonlinear loop remains chatter-free.  A
transducer-placement co-design study shows the two metrics *rank placements in
nearly opposite order* (Spearman rho = -0.4 on this benchmark): the placement
with a 12 mm linear boundary achieves only 0.93 mm feasibly, while the
strongly-coupled original placement (3.2 mm linear) is the feasible optimum at
1.92 mm -- 39 % above the LQG baseline (1.38 mm).  **(vi)** A phase-aware
feedforward in a two-degree-of-freedom controller is shown, analytically and with
CL-SD, to reduce forced vibration and peak voltage but *not* the stability
boundary; and two natural ADRC augmentations (a regeneration-aware delayed
channel and a resonant ESO) are honestly reported as *negative results* (< 2 %
gain).  **(vii)** Keeping the article's original sensing unchanged -- the single
non-minimum-phase tool-tip measurement -- we compare four controller classes in
the same CL-SD monodromy and two-stage metric: the model-based LQG works
(1.29 mm feasible), a model-free *fractional-order PID* (FOPID, PI^lambda D^mu,
Oustaloup realization) survives but is weak (0.20 mm), and the plain ADRC fails
at every bandwidth (its lumped-disturbance ESO inverts the plant through its
right-half-plane zeros).  We then merge ADRC and FOPID into a **HYBRID**
controller -- a band-limited ESO with a *searched, signed* effective gain plus a
co-designed FOPID branch, one voltage output, same single sensor: joint
co-design (retrofit is provably impossible) locks the ESO onto the mode-1
subplant (b0_eff = -0.55, band ~ 487 Hz) and reaches **0.83 mm -- 4x the best
fixed-structure law** -- making disturbance-rejection control usable at all on
this sensor, at honestly-stated costs (saturated ~37 um limit cycle at low
depth, -20 % drift fragility) that keep LQG the best overall tip-sensor
controller.  We also document and correct the fabricated results of the package
this work started from (`CORRECTIONS.md`).

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

**Model refinement (precise Kirchhoff modelling).**  The source model treated
the piezo patch only as a *force* (the H_pe vector) and ignored its mass and
bending stiffness.  We add the bonded 0.7 mm PZT layer as a composite section
with a shifted neutral axis (derivation and element-level assembly in
`paper/modeling.md`; implementation `plate_model._add_patch_dynamics`).  The
refinement is validated against the *measured* natural frequencies of [1]
(Table 4): it moves the chatter-dominant mode 1 from -3.5 % error to **+0.12 %**,
and the refined model's mean error over five modes (1.44 %) is lower than the
article's own theoretical model (1.93 %).  The mesh is convergence-checked
(30 x 24 indistinguishable from 50 x 40 on f1) and the sensitivity to the
assumed PZT density is < 0.1 % on f1.  The refined model also reveals that
modes 4-5 carry *stronger* actuator coupling than modes 1-2, which motivates
the spillover analysis of Sec. 4.8.

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

### 3.5 Two-stage metric and transducer-placement co-design

Linear Floquet analysis (open- or closed-loop) is the field's standard tool, but
it ignores the actuator's voltage limit, which in practice binds first: a design
can be linearly stable at a depth whose disturbance level demands far more than
+/-150 V, in which case saturation re-opens the loop and chatter returns.  We
therefore evaluate every candidate design with a two-stage metric
(`experiments/placement_study.py`):

1. **Linear stage (fast):** CL-SD boundary with the ESO in the monodromy, used
   to pre-filter placements (sign-viability of the modal coupling
   c_i * H_pe,i, i.e. the minimum-phase condition for a single-b0 ADRC) and to
   tune bandwidths.  ~10 ms per evaluation.
2. **Feasible stage (honest):** the *voltage-feasible critical depth*
   a_p,feas -- the largest depth at which the saturated nonlinear time-domain
   loop stays chatter-free (divergence or steady tip RMS > 50 um counts as
   chatter) -- found by bisection over full Newmark simulations.  The +/-150 V
   saturation is enforced on *every* controller, and each controller's design
   parameters are re-selected under this metric: ADRC over a bandwidth grid,
   and the LQG baseline over a weight grid that includes an aggressive design
   outside the conventional gain-norm cap.

The co-design study slides the 20 x 60 mm patch (with its collocated corner
sensor) over a 5 x 3 position grid on the plate and applies both stages.

### 3.6 Augmentations evaluated (negative results)

Two natural improvements of the plain ESO were tested against the same CL-SD
criterion (`experiments/augmentation_study.py`): a *regeneration-aware delayed
channel* u_tau = g [y(t-tau) - y(t)] exploiting the exactly-known tooth-passing
delay (the monodromy supports controller delayed-output terms), and a *resonant
ESO* embedding an internal model at/near the dominant mode.  Both are reported
in Sec. 4.7.

### 3.7 Fractional-order PID controller (FOPID)

To place the observer-based designs against the cheapest realistic alternative we
add a *fractional-order PID* (`src/fopid_control.py`), a model-free,
observer-free output-feedback law

    C(s) = Kp + Ki s^{-lambda} + Kd s^{mu},     0 < lambda, mu < 1,     u = -sigma C(s) y,

whose two extra knobs (the fractional orders lambda, mu) give independent
low- and high-frequency phase shaping.  The irrational operators are realized by
the Oustaloup recursive approximation (order N = 3 over [5, 8000] Hz, i.e.
2N+1 = 7 biproper first-order sections per branch, built as a well-conditioned
series cascade rather than a companion form; the band top is kept well below the
20 kHz Nyquist so the continuous realization embedded in CL-SD and the deployed
ZOH realization coincide); the individual operators track s^alpha to within ~2 %
magnitude and <=~15 deg phase across the mode band.  The resulting controller is
a strictly-causal LTI system z' = Ac z + Bc y, u = Cc z + Dc y (Dc != 0, the
operators are biproper), so it embeds in the CL-SD monodromy through the *same*
`rho_dynamic` interface as ADRC and LQG -- its stability-lobe diagram is genuine,
not asserted.  Faithful to the system of [1], it reads the article's original
measurement: the tool-tip displacement (a non-minimum-phase channel; Sec. 4.11).

The FOPID is designed against the true chatter metric: for each loop sign the
gains and orders are chosen by differential evolution to *minimize the dominant
Floquet multiplier* of the closed-loop semi-discretization at a reference depth
(the analogue of ADRC's bandwidth grid), then selected by the voltage-feasible
metric of Sec. 3.5 -- the same latitude LQG (weights) and ADRC (bandwidth)
receive.  Results in Sec. 4.11.

### 3.8 Hybrid ADRC-FOPID controller

The tip measurement of [1] is non-minimum phase: the modal coupling
D_tip . H_pe is sign-indefinite (-+--+), i.e. the u -> y_tip transfer carries
right-half-plane zeros.  Sec. 4.11 shows this defeats each parent controller in
a different way: the plain ADRC's single lumped-disturbance ESO effectively
inverts the plant through its RHP zeros (some subset of modes always sees
positive feedback), while the fixed-structure FOPID survives but cannot push
enough phase lead through the RHP zeros to earn a useful margin.  The HYBRID
(`src/hybrid_adrc_fopid.py`) merges them into one voltage output on the single
tip sensor:

    u = u_ESO + u_FOPID,
    u_ESO   = ( -wc^2 z1 - 2 wc z2 - z3 ) / b0_eff,
    u_FOPID = -sigma ( Kp + Ki s^-lambda + Kd s^mu ) y,

with the crucial difference from plain ADRC that the ESO bandwidth wo and the
signed effective input gain b0_eff are *free design parameters*: a band-limited
ESO sees essentially the mode-1 subplant, whose tip coupling has a definite
(negative) sign, so with a matching signed b0_eff its disturbance estimate is
coherent exactly in the band where regenerative chatter lives, while the FOPID
branch shapes the loop elsewhere.  Both branches are LTI, so the hybrid exports
(Ac, Bc, Cc, Dc) -- the ESO sees the total voltage, giving the exact augmented
form -- and drops into the CL-SD monodromy unchanged (implementation validated
three ways: the ESO branch reproduces `ADRCController` to machine precision, the
export equals the analytic (G_y + F)/(1 - G_u) closed form, and the deployed
discrete step matches the continuous export to < 1 % in closed loop).  Design is
two-stage and honest about it: stage A freezes the best tip FOPID and searches
only the ESO add-on (does a retrofit help?); stage B co-designs all eight
parameters jointly by differential evolution on the CL-SD spectral radius; the
winner is then selected by the voltage-feasible metric.  Results in Sec. 4.11.

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
stability at -20 % (tip RMS 379 um at the +/-150 V saturation -- chatter)**,
because its fixed model is now wrong. This is the practical statement of ADRC's model-independence and directly
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

### 4.6 Placement co-design: the two metrics rank in opposite order

Of the 15 candidate patch positions, 10 collapse for the single-b0 ADRC at every
tested bandwidth, and all 10 have mixed-sign modal coupling c_i * H_pe,i
(Fig. 6a).  The sign structure is a useful *screening heuristic* rather than a
sufficient rule: one mixed-sign placement (0, 10) remains viable because the
wrong-signed mode is fast and heavily filtered, and three placements whose first
two modes are correctly signed still collapse; the operative partition is the
CL-SD result itself.  For the five viable placements the two metrics give
(Fig. 6b):

| Patch (x0, z0) mm | linear CL-SD boundary | voltage-feasible depth |
|---|---:|---:|
| (60, 10) | **12.0 mm** (rank 1) | 0.93 mm (**rank 5**) |
| (0, 10)  | 4.75 mm (rank 2) | 1.29 mm (rank 3) |
| (60, 0)  | 4.73 mm (rank 3) | 1.38 mm (rank 2) |
| (0, 0) *(original)* | 3.24 mm (rank 4) | **1.92 mm** (**rank 1**) |
| (80, 0)  | 2.37 mm (rank 5) | 1.11 mm (rank 4) |

The rankings are descriptively anti-correlated (Spearman rho = -0.4; n = 5, so
this is reported as an observation on this benchmark, not a significance
claim).  The mechanism is
plain: weakly-coupled placements (small |b0|) tolerate the delay well in the
linear analysis but need proportionally more voltage to reject the same cutting
disturbance, so the +/-150 V wall arrives much earlier; strongly-coupled
placements spend less voltage per unit of rejection.  **Selecting a transducer
placement -- or reporting a controlled SLD -- from linear analysis alone is
therefore unsafe, even when the analysis honestly includes the controller and
its observer.**  Under the honest metric the original placement of [1] is in
fact near-optimal, and the feasible headline comparison at identical hardware is

| Controller (original placement, both saturated at +/-150 V) | voltage-feasible depth |
|---|---:|
| LQG (tip sensor + Kalman observer; best of a weight search under the feasible metric, incl. an aggressive design outside the gain-norm cap) | 1.38 mm |
| **ADRC (collocated, wc=1800, wo=21600)** | **1.92 mm (+39 %)** |

confirmed by 0.5 s saturated simulations (ADRC stable at 1.8 mm, chatter at
2.1 mm; LQG stable at 1.3 mm at 133 V peak, chatter at 1.5 mm).  With the
conventional (linearly-tuned) LQG weights the saturated feasible depth is
1.29 mm, i.e. the +39 % is measured against the *best* saturated LQG found, not
the default one (+49 % against the default).

### 4.7 Augmentations: honest negative results

Neither augmentation of Sec. 3.6 improves the linear boundary materially over
the plain delay-tuned ESO (baseline 3.25 mm at the nominal placement): the best
delayed-channel gain gives 3.27 mm (+0.7 %) before rapid destabilisation at
larger |g|, and the best resonant-ESO configuration gives 3.31 mm (+1.8 %), with
most configurations worse.  The interpretation is structural: the ESO already
reconstructs the total disturbance with lag ~1/wo << tau, so the delayed channel
mostly injects wrong-phase feedback, and a fixed internal model cannot track the
chatter frequency, which shifts with depth and lobe number.  At fixed hardware
the plain bandwidth-parameterised ESO with delay-aware tuning is close to the
ceiling of single-channel output-feedback laws; the remaining levers are the
transducer placement (Sec. 4.6) and the actuator budget itself.

### 4.8 Model refinement: validation, spillover, and the honest endpoint

The refined 5-mode plant (patch mass/stiffness included; Sec. 2) is used as the
"truth model" and every production controller -- all designed on the 3-mode
bare nominal model -- is re-evaluated on it without modification (Fig. 7).

**Validation.**  Mode 1 error drops from -3.51 % to +0.12 % against the
measured 540 Hz; the mean five-mode error (1.44 %) beats the article's own
theoretical model (1.93 %).  Mode 2 worsens slightly (+2.2 %), which bounds the
neglected glue-layer/membrane effects; we report it rather than tune it away.

**Spillover sensitivity (nominal-tuned designs).**  On the refined plant the
linear boundaries contract to near-parity (LQG 2.32 mm, ADRC 2.34 mm), and in
the saturated regime the nominal-tuned ADRC *loses* its advantage
(voltage-feasible 0.74 mm vs 1.02 mm for LQG).  The mechanism is spillover: the
refined modes 4-5 carry the largest actuator couplings and the collocated
coupling of mode 4 is wrong-signed (`---+-`), so ADRC's aggressive disturbance
cancellation pumps energy into unmodelled dynamics and burns its voltage budget.
The comparison of Sec. 4.6, taken alone, would therefore overstate ADRC's
practical margin -- exactly the kind of conclusion the refined model exists to
test.

**A sampling artifact, diagnosed.**  Before re-tuning, the apparent ADRC
collapse was traced to its root: the refined closed loop carries a marginally
damped spillover pair near 3.4 kHz (continuous max Re(eig) = -35 s^-1, both
with the nominal and refined b0 -- the *continuous* loop is stable).  At the
production sample time dt = 50 us (10 kHz) the one-sample implementation delay
(~60-70 deg of phase at 3.4-3.8 kHz) and the Newmark period distortion of
modes 4-5 (~(w dt)^2/12, up to 14 %) destabilise this pair numerically: the
unsaturated loop diverges at 3.76 kHz, and under saturation this masquerades as
a 150 V limit cycle that the feasibility criterion partially tolerates.  At
dt = 25 us the loop is clean and converged (tip RMS 0.253 um, 29 V; 0.228 um at
12.5 us).  Hence (i) refined-plant time-domain verdicts must be computed at
dt <= 25 us, and (ii) a ~20 kHz controller rate is a hardware *requirement* for
this ADRC on this structure -- a deliverable only the refined model could
produce.

**Honest endpoint (dt = 25 us, both controllers re-tuned on the refined
plant).**

| fidelity layer | LQG best | ADRC best | verdict |
|---|---:|---:|---|
| 3-mode plant, dt = 50 us (Sec. 4.6) | 1.38 mm | 1.92 mm | ADRC +39 % |
| refined 5-mode, dt = 50 us | 1.29 mm | 0.74 mm | sampling artifact |
| **refined 5-mode, dt = 25 us** | **1.29 mm** | **1.47-1.65 mm** | **ADRC +14-28 %** |

The ADRC advantage survives the refined model once the numerical artifact is
removed, at a moderated magnitude.  A tuning nuance must be reported: the
deepest boundary (1.65 mm, wc = 1200 rad/s) tolerates a *saturated limit cycle*
near the operating depth (tip RMS ~12 um at 150 V -- "stable" under the 50 um
chatter criterion but unacceptable for surface finish), whereas the
clean-response tuning (wc = 1800 rad/s: 0.25 um at 29 V at the operating point)
reaches 1.47 mm.  Honest headline: **+14 % with clean actuation, up to +28 % if
saturated limit-cycle operation near the boundary is tolerated**; the
feasibility criterion itself (chatter only) is therefore complemented by the
saturation-time fraction in the in-process study of Sec. 4.9.  The drift-robustness conclusion also survives unchanged on the
refined plant at dt = 25 us: at the nominal point ADRC halves the tip vibration
(0.252 vs 0.508 um, +50 %), and at -20 % drift LQG loses stability (613 um,
saturated) while ADRC holds 0.322 um at 33 V.

The methodological lesson mirrors Sec. 4.6: each modelling-fidelity layer
(patch dynamics -> modal truncation -> sampling/integration resolution) can
*reverse* a comparison if evaluated carelessly; conclusions are only safe when
the layer at which they are computed is stated and converged.

### 4.9 In-process material removal: physically-generated varying dynamics

The FEM now models the removal itself (`plate_model.remove_material`): covered
elements are re-assembled at their reduced thickness (K ~ h^3, M ~ h,
partial-coverage weighted) under the standard piecewise-frozen assumption, and
the modal basis, tool-path shapes, sensor rows and actuator projection are
recomputed per machining state (`experiments/material_removal.py`).

**Single pass (article conditions).**  Removing a_e = 0.1 mm over the top
0.3 mm strip along the full length shifts every natural frequency by less than
0.02 % and b0 by 0.1 %.  This is an honest *bounding* result: within one pass
the constant-dynamics assumption used throughout the production studies is
justified -- the varying-dynamics challenge of [1] comes from the tool
*position* and from pass-to-pass stock removal, not from within-pass removal.

**Thin-walling (pass-to-pass).**  Milling the top 20 mm band from 4.0 mm down
to 3.0 mm in four 0.25 mm passes raises f1 by **+9.5 %** (540.6 -> 592.2 Hz):
at a cantilever's free edge the removed *mass* dominates the removed
*stiffness*, so thinning stiffens the apparent fundamental.  The collocated
coupling strengthens monotonically (b0: -0.744 -> -0.955) and the sign pattern
stays viable throughout (becoming fully sign-definite from h = 3.25 mm).  This
is the physically-generated counterpart of the synthetic +/-20 % drift sweep of
Sec. 4.3 -- with the important difference that the drift is now *upward* in
frequency and accompanied by consistent mode-shape changes.

**Control through the process (designs frozen at h = 4.0 mm).**  Both
controllers are designed once at the initial state (dt = 25 us; ADRC with the
clean-response tuning wc = 1800 and the initial b0) and carried unchanged
through the five machining states; at each state the linear margin is computed
by CL-SD on the *current* plant and the 0.3 mm operating point is simulated in
saturated time domain (Fig. 8):

| band thickness | LQG a_p,crit | ADRC a_p,crit | LQG op. (um / V) | ADRC op. (um / V) |
|---:|---:|---:|---:|---:|
| 4.00 mm | 2.32 mm | 2.33 mm | 0.513 / 13 | 0.252 / 29 |
| 3.75 mm | 2.54 mm | 2.24 mm | 0.514 / 13 | 0.245 / 31 |
| 3.50 mm | 2.26 mm | 2.23 mm | 0.543 / 14 | 0.240 / 31 |
| 3.25 mm | 1.82 mm | 2.29 mm | 0.556 / 14 | 0.235 / 32 |
| 3.00 mm | **1.68 mm** | **2.36 mm** | 0.598 / 14 | 0.228 / 33 |

The short (0.4 s) per-state checks above cover only ~2 mm of travel; the
full-process simulation below removes that limitation.

Honestly stated: *neither* controller fails at the operating point during this
process -- the physically-generated drift is upward in frequency, which is the
benign direction for the frozen LQG (its synthetic failure in Sec. 4.3 was at
-20 %).  The margins, however, tell the practically relevant story: the frozen
LQG's stability margin erodes monotonically once thinning progresses
(2.54 -> 1.68 mm, -28 % over the process, with its operating vibration up
+17 %), while the frozen ADRC's margin stays flat and its vibration *improves*
(0.252 -> 0.228 um, because the collocated coupling b0 strengthens as the wall
thins).  At the thinnest state -- exactly where finishing passes demand the
most predictable margin -- ADRC holds 40 % more margin than LQG (2.36 vs
1.68 mm), without any re-identification or re-tuning.  This is the model-light
property working on physically-generated, not synthetic, dynamics variation.

**Full-process simulation (end-to-end, all passes).**  Finally, the process is
simulated *continuously* (`experiments/full_process_sim.py`): four passes from
edge to edge (4 x 20.41 s at the article feed, dt = 25 us, ~3.3 M Newmark
steps per controller), the band thinned *behind the advancing tool* every
5 mm, and the vibration state plus the regenerative delay history reprojected
onto each updated modal basis with the mass-orthonormal transform
T = V_new^T M V_old (near-identity per segment, diag ~0.999); the frozen
controllers carry their internal states across all updates, as on the real
machine.  Consistency check: the first-segment values reproduce the
piecewise-frozen operating-point results (ADRC 0.251 vs 0.252 um; LQG 0.506 vs
0.513 um) -- the frozen abstraction is *validated where it applies*; its
limitation was coverage (~2 mm of travel), not correctness.

The full path reveals a first-order effect that no short simulation can see
(Fig. 9; the complete 81.6 s tip time response of all controllers, with the
80 ms running-RMS envelope, is shown in Fig. 11).  Per-pass process RMS:

| pass (band ->) | LQG | ADRC |
|---|---:|---:|
| 1 (-> 3.75 mm) | 0.473 um | **0.458 um** |
| 2 (-> 3.50 mm) | 0.474 um | 0.489 um |
| 3 (-> 3.25 mm) | 0.484 um | 0.515 um |
| 4 (-> 3.00 mm) | **0.495 um** | 0.533 um |

The process-mean verdict *flips* by pass 4, and the spatial profiles explain
why: the ADRC's single collocated sensor sits at the patch corner (x = 20 mm),
and its rejection is outstanding near the sensor (pass-4 thirds: 0.28 um for
x = 0-33 mm, twice better than LQG's 0.51) but degrades monotonically with
tool-sensor distance (0.75 um for x = 66-100 mm, ~40 % worse than LQG's 0.53)
-- the ESO cannot reject what its sensor barely observes, and the thinning
amplifies the far-side deficit because the frozen b0 under-represents the
strengthening coupling.  The LQG's tip sensor yields a flatter, U-shaped
profile (0.42-0.59 um) that is nearly pass-independent.  Neither controller
saturates anywhere in the 81.6 s process (ADRC <= 35 V, LQG <= 14 V).

The honest synthesis of Sec. 4.9: ADRC's *stability margin* through the
process is decisively better (+40 % at the thinnest state) and its vibration
is far better over the third of the path nearest its sensor, but a *single*
collocated sensor cannot deliver full-path forced-vibration superiority --
the spatial observability footprint joins stability, voltage and sampling
rate on the list of first-order design constraints, and points directly at
dual-sensor or position-weighted sensing as the natural next design step
(consistent with the placement co-design lens of Sec. 4.6).

### 4.10 Improving the ADRC: tip-driven adaptive feedforward (AFC-ADRC)

Sec. 4.9 identified the single collocated sensor's spatial observability
footprint as the binding weakness, and the far-side residual as *periodic*
(tooth-passing harmonics; the dominant line is the 3rd at ~736 Hz).  Two
structural facts shape the remedy.  First, the tip signal cannot enter the
fast ESO loop: frozen-blend analysis shows even 20 % tip content destabilises
the loop at ~4 kHz (the tip's mode-4/5 couplings are large and wrong-signed),
a low-pass-filtered blend fails at ~700 Hz through the wrong-signed mode-2
channel, and any right-side w-sensor flips mode 2 by antisymmetry -- so the
fast loop's sensor must stay collocated.  Second, we showed in Sec. 4.4 that
feedforward does not move the closed-loop poles.  The improvement therefore
uses the tip sensor -- already present on the rig of [1] -- *outside* the fast
loop (`src/afc_adrc.py`):

    u = u_ADRC(y_col) + sum_k [ a_k cos(k*theta) + b_k sin(k*theta) ],

a spindle-synchronous harmonic comb (K = 5, theta locked to the tooth period)
whose coefficients adapt by normalized filtered-x LMS on the *tip* error
through the closed-loop secondary path G_k, with per-harmonic gains
g_k = 2/(tau_a |G_k|^2), tau_a = 0.3 s, +/-30 V clamp and leaky-LMS decay
(tau_leak = 1 s).  Two implementation details proved essential and are
documented as reproducible failure modes: G_k must include the ESO's response
to the injection (input matrix [B; B_o] -- with [B; 0] the phase is wrong by
up to ~280 deg and the comb pumps), and the comb must be phase-locked to the
implemented tooth period (a 0.4 Hz mismatch from period rounding already
voids the cancellation).

**Fixed-position validation** (x = 90 mm, the worst region, tip RMS):

| condition | plain ADRC | AFC-ADRC | gain |
|---|---:|---:|---:|
| nominal | 0.600 um | **0.177 um** | +70 % |
| drift -20 % | 0.730 um | **0.303 um** | +58 % |
| drift +20 % | 5.281 um | 5.266 um | +0.3 % (neutral) |

Within the FxLMS validity envelope the comb removes ~2/3 of the far-side
residual; at +20 % drift the 736 Hz line approaches the shifted fundamental,
the 90-deg phase condition fails, and the leakage safely *disables* the comb
instead of letting it pump (without leakage this case amplified 9x) -- the
augmentation is never worse than the plain ADRC it wraps.  The machining-state
robustness is also verified: with G_k frozen at h = 4.0 the cancellation still
holds at h = 3.0 (0.126 vs 0.110 um).

**Full-process result** (same 4-pass end-to-end protocol as Sec. 4.9, Fig. 10):

| pass (band ->) | LQG | plain ADRC | **AFC-ADRC** |
|---|---:|---:|---:|
| 1 (-> 3.75 mm) | 0.473 um | 0.458 um | **0.145 um** |
| 2 (-> 3.50 mm) | 0.474 um | 0.489 um | **0.146 um** |
| 3 (-> 3.25 mm) | 0.484 um | 0.515 um | **0.151 um** |
| 4 (-> 3.00 mm) | 0.495 um | 0.533 um | **0.160 um** |

AFC-ADRC delivers a ~3x reduction of the process RMS over BOTH baselines and
resolves the spatial deficit that motivated it: the pass-4 thirds flatten to
0.12 / 0.14 / 0.21 um (plain ADRC: 0.28 / 0.51 / 0.75; LQG: 0.51 / 0.45 /
0.53), with the worst segment at 0.24 um versus 0.81 (ADRC) and 0.59 (LQG).
A full-rate waveform snapshot at the worst region (Fig. 11b) makes the
mechanism explicit: the large tooth-passing peaks of both baselines (LQG
0.55, ADRC 0.72 um at x = 90 mm) are flattened by the comb to 0.20 um.  The
cost is negligible: peak voltage 37.8 V (plain ADRC 35.2 V), zero
saturation anywhere in the 81.6 s process, no new hardware (both sensors exist
on the rig of [1]), and the fast-loop stability boundaries of Secs. 4.1-4.8
untouched.  Honest boundary of validity: the comb targets the *periodic*
component only, its benefit shrinks if the spindle-synchronous assumption
breaks (e.g. heavy chatter onset -- where the ADRC's margin, not the comb, is
the relevant defence), and under frequency drift beyond the FxLMS phase
envelope it degrades to neutrality by design (leakage).

### 4.11 Controllers on the article's tip sensor: FOPID, plain ADRC, and the HYBRID

Faithful to the system of [1], this comparison uses ONE sensor: the tool-tip
displacement.  The tip channel is non-minimum phase -- its modal coupling
D_tip . H_pe is sign-indefinite (-+--+, instantaneous gain b0 = +0.38 but
mode-1 coupling negative), i.e. u -> y_tip has right-half-plane zeros.  All
controllers run on the refined 5-mode plant at dt = 25 us with +/-150 V
saturation; LQG keeps its full-state Kalman observer, FOPID and the hybrid are
output-feedback laws on y_tip
(`experiments/fopid_tip_study.py`, `experiments/hybrid_tip_study.py`).

| controller | needs | voltage-feasible a_p (tip) | tip RMS at 0.15 mm |
|---|---|---:|---:|
| open loop | -- | 0.06 mm (linear) | (chatter) |
| **LQG** | full state model | **1.29 mm** | **0.23 um** (18 V) |
| plain ADRC | b0 only | **0.00 mm -- fails** | (unstable) |
| plain FOPID | nothing | 0.20 mm | 1.30 um (3.6 V) |
| **HYBRID** (Sec. 3.8) | b0_eff, wo (searched) | **0.83 mm** | 37 um (saturated) |

Findings, each computed.

**(i) Plain ADRC fails outright on the tip.**  At every bandwidth tried
(wc = 200-1800 rad/s) the linear CL-SD boundary sits at the floor and the
saturated loop diverges even at 0.2 mm (Fig. 12b): a single lumped-disturbance
ESO cannot invert a plant with RHP zeros.  **Plain FOPID survives but is weak**
(0.20 mm, barely above the 0.06 mm open loop): a fixed structure cannot push
enough phase lead through the RHP zeros.

**(ii) A retrofit ESO is provably useless -- only co-design works.**  Stage A
(best tip FOPID frozen, ESO added on top, 3-parameter search over wc, wo/wc and
the signed b0_eff, both signs) finds NO stabilising ESO at all -- every retrofit
destabilises the loop.  Stage B (joint 8-parameter co-design) does find one, and
only in a physically telling corner: b0_eff = -0.55 (the sign and order of the
*mode-1* coupling, opposite to the instantaneous +0.38), ESO bandwidth
wo = 3058 rad/s ~ 487 Hz (a band around mode 1 at 540 Hz), with the FOPID branch
reshaped to vacate that band (mu drops from 0.64 to 0.30).  The ESO must be
band-limited to, and sign-matched with, the mode-1 subplant -- and the FOPID
must be co-designed to make room for it.

**(iii) The hybrid rescues disturbance-rejection control on the tip.**  The
co-designed hybrid reaches a voltage-feasible depth of **0.83 mm -- 4x the plain
FOPID and infinitely better than the plain ADRC's zero** -- with a nominal-model
linear boundary of 2.02 mm (~2x FOPID's 1.02 mm).  At 0.5 mm, where FOPID and
ADRC both chatter to divergence, the hybrid holds the cut (Fig. 12b).

**(iv) The costs, stated plainly.**  First, finish: at 0.15 mm the hybrid rides
a saturated limit cycle of ~37 um RMS -- stable, but far above LQG's 0.23 um and
plain FOPID's 1.30 um (Fig. 12c).  The hybrid is a chatter *boundary extender*,
not a finish controller; its aggressive mode-1 inversion buys margin with
voltage, not accuracy.  Second, drift: at -20 % plant-frequency drift the
hybrid's mode-1-tuned ESO loses its band and the feasible depth collapses to
zero (as does plain FOPID's), while at +20 % it holds 0.74 mm; LQG degrades but
survives on both sides (0.65 / 0.56 mm) (Fig. 12d).  Third, spillover: like the
aggressive LQG of Sec. 4.8, the hybrid's *linear* boundary on the refined
5-mode plant collapses to the floor (un-modelled 3.4-4.1 kHz modes destabilise
the linear analysis; the saturated nonlinear loop survives) -- one more instance
of Sec. 4.6's lesson that only the feasible metric ranks controllers sensibly.

**(v) Verdict on the article's sensor.**  LQG remains the best tip-sensor
controller on every axis (deepest feasible cut, best finish, only survivor of
-20 % drift): with a single non-collocated NMP measurement, a full-state
model-based observer is not a luxury but the price of admission.  The hybrid's
contribution is to make ESO-based control *usable at all* on this sensor -- a
4x feasible-depth gain over the best fixed-structure law, from two components
that individually fail or underperform -- and to expose the design rule: a
lumped-disturbance observer on an NMP channel must be band-limited and
sign-matched to a single dominant mode, and the companion controller must be
co-designed around it (retrofit is impossible, stage A).  All of it drops into
the same CL-SD monodromy and two-stage metric with no special-casing (Fig. 12).

*Honest limitations.*  The joint search fixed the FOPID loop sign at sigma = +1
(its winning sign as a standalone tip controller) and used a disclosed DE budget
(popsize 8, 10 generations per b0 sign); the gain-scale feasible selection
landed interior (0.35).  The hybrid numbers are therefore lower bounds from a
bounded search, not proven optima.

## 5. Discussion

These are *simulation* results. CL-SD, like all Floquet stability analysis, is
linear; the voltage-feasible stage (Sec. 3.5) closes the largest gap -- actuator
saturation -- but sensor noise and amplifier dynamics are still idealised in the
stability numbers (the realistic piezo model captures them in time domain);
simulated depths therefore remain optimistic relative to the experiments of [1]
and should be read as relative comparisons under identical assumptions. The ADRC
comparison uses a collocated control sensor (natural for a piezo transducer)
while the model-based controllers use the tip measurement with a full-state
observer; performance for all is read at the tip. The controller-class comparison
of Sec. 4.11 deliberately keeps the article's original single tip sensor: there,
the non-minimum-phase channel makes plain ADRC fail outright and cripples FOPID,
and only band-limited, sign-matched, co-designed hybridization (Sec. 3.8)
recovers a useful ESO -- while a full-state model-based observer remains the
strongest option on that sensor. The static feedback-authority
curve (Sec. 4.5) is an idealised bound, not a realizable envelope. The
contribution is a controlled SLD that is computed rather than asserted, an ADRC
design suited to varying dynamics, and an honest accounting of what feedback
versus feedforward, and observer versus static feedback, can and cannot do.

## 6. Conclusion

We presented a closed-loop semi-discretization method that embeds the controller
and its observer in the Floquet analysis, quantified the observer's cost to the
LQG stability margin, and designed an ADRC controller that -- needing only the
input gain -- extends the linear critical depth (3.25 vs 1.92 mm), roughly halves
tip vibration at low voltage, and remains stable under frequency drift that
destabilises LQG.  Beyond the linear analysis, we introduced the voltage-feasible
critical depth as the honest design metric and showed, through a
transducer-placement co-design study, that it ranks placements in nearly the
opposite order to the linear boundary (12 mm linear can mean 0.93 mm feasible);
under this metric ADRC delivers 1.92 mm versus 1.38 mm for LQG (+39 %) at
identical hardware.  Refining the Kirchhoff model to precise level -- patch
mass/stiffness (mode-1 error +0.12 % vs measurement), five retained modes, and
converged sampling resolution -- moderates but confirms the verdict: on the
refined plant at 20 kHz the best ADRC reaches 1.65 mm versus 1.29 mm for the
best LQG (+28 %), halves the nominal tip vibration, and survives the -20 %
drift that destabilises LQG; the intermediate fidelity layers were shown to
reverse comparisons when evaluated carelessly (spillover at 10 kHz sampling).
In-process material removal was modelled and quantified (negligible within a
pass, f1 +9.5 % across the thin-walling passes), and the continuous end-to-end
simulation exposed the spatial observability footprint of single-point
sensing -- which the AFC-ADRC augmentation, a spindle-synchronous adaptive
comb on the rig's existing tip sensor kept outside the fast loop, then
removed: full-process vibration ~3x below both baselines (0.15 vs
0.47-0.53 um) at negligible voltage cost and unchanged fast-loop stability.
Two in-loop ADRC augmentations were honestly reported as negative results, and
a two-degree-of-freedom feedforward was shown to help forced vibration but not
stability.  Finally, keeping the article's original single tip sensor -- a
non-minimum-phase channel -- we compared four controller classes in the same
CL-SD framework and two-stage metric: the model-based LQG works (1.29 mm
feasible), a model-free fractional-order PID survives but is weak (0.20 mm),
plain ADRC fails at every bandwidth, and a HYBRID that merges the two
(band-limited ESO with searched signed gain + co-designed FOPID branch) reaches
0.83 mm -- 4x the best fixed-structure law -- with its costs stated plainly
(saturated low-depth limit cycle, -20 % drift fragility).  The hybrid's design
rule is itself a result: on an NMP channel a lumped-disturbance observer must be
band-limited and sign-matched to one dominant mode and its companion controller
co-designed around it (a retrofit is provably impossible); and the framework
accepted every one of these structures without special-casing.  Every reported
number is reproducible from the code.

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

[6] I. Podlubny. Fractional-order systems and PI^lambda D^mu controllers.
*IEEE Transactions on Automatic Control* 44 (1999) 208-214.

[7] A. Oustaloup, F. Levron, B. Mathieu, F. M. Nanot. Frequency-band complex
noninteger differentiator: characterization and synthesis. *IEEE Transactions on
Circuits and Systems I* 47 (2000) 25-39.

[8] C. A. Monje, Y. Q. Chen, B. M. Vinagre, D. Xue, V. Feliu. *Fractional-order
Systems and Controls: Fundamentals and Applications*. Springer, 2010.
