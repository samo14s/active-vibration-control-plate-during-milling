# Cover Letter — Journal of Sound and Vibration

*(Adapt salutation/author details before submission.)*

Dear Editors,

Please consider the enclosed manuscript, **"Position-scheduled linear
parameter-varying control of thin-walled plate milling: certified
stability lobes and the limits of regenerative-targeted delayed
feedback"**, for publication in the *Journal of Sound and Vibration*.

**What the paper contributes.** Active chatter control of thin-walled
workpieces faces structural dynamics that vary strongly — and
predictably — with the tool position and material removal. The
prevailing practice wraps this *known* variation into norm-bounded
uncertainty and pays with conservatism. We instead schedule a
grid-based LPV H∞ controller on the tool position and removal state
(both known in real time from the NC program), retain uncertainty only
for what is genuinely unknown, and certify the resulting scheduled,
time-periodic, delayed closed loop by semi-discretization stability
lobes along the tool path, a small-gain spillover margin, and an exact
sampled-loop spectral-radius test. On a benchmark plate the scheduling
paradigm multiplies the worst-position stable depth of cut by 4.2 over
the strongest non-scheduled design of the same family. Two certified
negative results complete the study: the lobe-maximizing tuning of the
popular spindle-synchronized delayed feedback add-on returns a *zero*
gain once the scheduled loop is active (while the same delayed action
alone destabilizes the process across the speed band), and freezing the
schedule's removal axis destabilizes a point-designed loop outright
after 0.5–1 mm of edge recession.

**Why JSV.** The paper is a vibration-control study in the direct
lineage of JSV's active chatter-control literature (e.g., Dohner et
al., JSV 269, 2004): its core is stability analysis and certified
synthesis for a time-periodic, delayed, parameter-varying structural
system, with the machining process as the physical carrier.

**On the numerical character of the study.** The work is a simulation
study deliberately anchored at every link to the published experimental
record of a benchmark rig (Du et al., *Int. J. Mech. Sci.* 274, 2024):
identical geometry, actuator, sensor, and amplifier; modal parameters
validated to 1.5–3.8%; and — using that paper's own force
calibration — the model reproduces the rig's measured open-loop
stability pattern across its tested speed band, including the anomaly
at 5500 rpm. The closed loop is evaluated with implementation-true
timing (50 kHz sampling, computation delay, saturation, sensor noise),
and the complete design data and code are released so that any
laboratory with the reference rig's hardware class can execute the
experimental validation directly.

The manuscript is not under consideration elsewhere. We suggest
reviewers from the active chatter control and time-delay systems
communities.

Sincerely,
*The authors*
