# Research-Gap Analysis & Deepening — Active Vibration Control of Thin-Walled Milling

**Subject of review:** the `article_simulation_package` comparing **LQG** and
**DARC-MPC** controllers for chatter mitigation in peripheral milling of a
cantilever AL6061 plate with a bonded piezo actuator.

**Purpose of this document.** A reviewer-grade audit of the study's most
important research gaps, each backed by a *numerical* demonstration produced with
the package's **own** nonlinear NDDE integrator, and each **addressed** by a new,
self-contained module in `research_extensions/`. The goal is to turn claims that
a top-tier reviewer (IEEE TCST / MSSP / Automatica / CIRP) would reject into
statements the data actually support.

> All numbers below were measured on the preserved article model
> (`[521.06, 1069.95, 2733.02] Hz`, ζ = [0.31, 0.17, 0.27] %, 4900 RPM,
> AL6061 plate, QDA60 piezo). Reproduce with `python main_gap_study.py`.

---

## ملخّص تنفيذي (Arabic executive summary)

راجعتُ الحزمة بعمق وشغّلتُ شيفرتها للتحقق. أهم الفجوات البحثية، بالأدلة العددية:

1. **المقارنة غير عادلة (الأهم).** ادعاء المقال أن "التغذية الراجعة لا تستطيع رفض
   اضطراب دوري" مُخالِف لمبدأ النموذج الداخلي (Francis–Wonham). بنيتُ الأساس العادل
   الغائب — **متحكّم تكراري/نموذج داخلي (Repetitive-LQG)** — وهو تغذية راجعة بحتة.
   النتيجة: بمُستشعر جيد يُحقّق **0.16–0.17 ميكرومتر مقابل 0.29 لـDARC و0.60 لـLQG**،
   أي **يتفوّق على DARC**. ادعاء "55% تحسّن" مقيس مقابل أساس مُعاق عمداً.
2. **اللاخطية خاملة.** أثبتُّ أن LQG على النموذج الخطي واللاخطي متطابقان (فرق 0.003%) —
   كل نتائج المقال فعلياً خطّية. بنيتُ **دراسة تشعّب** تُفعّل اللاخطية فعلاً (دورة حدّية
   مقيّدة عند 1mm حيث يتباعد النموذج الخطي لا نهائياً).
3. **مخطط الاستقرار إرشادي.** منحنى DARC يستخدم "دفعة تخميد" مفترضة. بنيتُ **شبه-تقطيع
   للحلقة المغلقة** (Floquet حقيقي): LQG يرفع العتبة الحرجة إلى 1.90mm (لا 2.86)،
   و**التغذية الأمامية لا تُغيّر حدود الاستقرار إطلاقاً** — فادعاء DARC=4.0mm خاطئ مفاهيمياً.
4. **التكيّف تجميلي.** `lambda_robust` يُحسب ولا يُستخدم؛ DARC بالتكيّف ON=OFF تماماً
   (فرق صفري). الـ"Adaptive Robust" في الاسم بلا سند في الشيفرة.
5. **إحصاء ناقص.** Monte Carlo للمقال يختبر LQG فقط. وسّعتُه لكل المتحكّمات مع فواصل ثقة
   ومقارنات مُقترنة: RC-LQG يفوز على DARC في 100% من العيّنات (مُستشعر جيد).

**الخلاصة الصادقة:** ميزة DARC ليست أن التغذية الراجعة تعجز — بل أن تغذيته الأمامية
**مستقلة عن المُستشعر**، فيسيطر في نطاق الاستشعار الخشن الصناعي. هذه إعادة صياغة أقوى
وأصدق للمساهمة، مدعومة بالأرقام أدناه.

---

## Gap #1 — The comparison is not fair: the missing feedback baseline ★ most important

**The claim under review.** The package's headline is *"DARC-MPC reduces RMS
vibration ≈ 55 % below LQG,"* justified by:

> *"a feedback controller cannot fully reject a periodic disturbance (phase lag /
> waterbed limit) — but a feedforward … can."* (`README.md`)

**Why a reviewer rejects it.** This statement is false as written. The
**Internal Model Principle** (Francis & Wonham, 1976) guarantees that a *feedback*
loop achieves **asymptotic rejection** of a disturbance **iff a model of the
disturbance dynamics is embedded in the loop**. For the periodic tooth-passing
excitation (fundamental f_TPF = N_T·Ω/60 = **245 Hz** and harmonics), that model
is a bank of resonators 1/(s²+(hω)²) — precisely the mechanism of **repetitive
control** (Hara et al., 1988), a standard technique in machining. Comparing DARC
only against a *feedback-only* LQG compares against a **deliberately handicapped**
baseline; the honest question is DARC **vs feedback that embeds the internal
model**.

**What I built.** `internal_model_control.py::RepetitiveLQG` — the same LQG DNA
(identical modal plant, base weights w_q=1e14, w_qd=1e8, R=1, same Kalman
observer) **plus** a multi-harmonic internal model of the tooth-passing period,
designed as an LQG-optimal servo-compensator over the augmented plant. It is
*pure feedback*; it adds no feedforward and no encoder signal. (A subtlety worth
publishing on its own: the naive controllable-canonical resonator realization is
so badly scaled that the augmented Riccati leaves the internal model undamped and
the rejection never engages — a *rotational* realization is required. See the
module docstring.)

**Result (measured).** Sweeping the displacement-sensor noise on the nominal
scenario (steady-state y_RMS, µm):

| sensor noise | LQG | **RC-LQG (new)** | DARC | winner |
|---:|---:|---:|---:|:--|
| 0.0 µm (ideal) | 0.599 | **0.162** | 0.290 | **RC-LQG** |
| 0.1 µm (fine)  | 0.600 | **0.173** | 0.293 | **RC-LQG** |
| 0.6 µm         | 0.643 | 0.398 | **0.371** | DARC |
| 1.0 µm         | 0.714 | 0.627 | **0.483** | DARC |
| 2.0 µm         | 0.981 | 1.220 | **0.824** | DARC |
| 4.0 µm         | 1.666 | 2.426 | **1.568** | DARC |

**Consequences for the paper.**
- With a good sensor, a **pure-feedback** repetitive-LQG **beats the DARC
  feedforward** (0.16 vs 0.29 µm) — the "feedback cannot" premise is refuted, and
  the "55 %" is an artifact of the handicapped baseline.
- DARC's **genuine, defensible** advantage is **sensor-independence**: its
  feedforward is spindle-encoder (phase) synchronized, so it is immune to sensor
  degradation, whereas RC-LQG *amplifies* sensor noise at the rejected harmonics
  (the classic repetitive-control **waterbed** penalty — RC-LQG becomes *worse
  than plain LQG* beyond ≈2 µm). **DARC therefore dominates precisely in the
  industrially-relevant regime of coarse / noisy displacement sensing.**

This reframing is strictly stronger: it survives the obvious reviewer objection
and states a real, quantified trade-off with a crossover at ≈0.35 µm sensor noise
(**Fig G1**).

---

## Gap #3 — The nonlinear NDDE model is never exercised (all results are linear-regime)

**Evidence.** The README concedes the Von Kármán cubic and 3rd-order cutting
terms are *"dormant at the nominal µm amplitudes."* I confirmed it: LQG on the
**nonlinear** plant vs the **linearised** plant gives

```
LQG (linear plant)    y_rms = 0.6141 µm
LQG (nonlinear plant) y_rms = 0.6141 µm      Δ = 2.3e-5 µm  (0.003 %)
```

The cubic-to-linear ratio at 1 µm is λη³/(ω²η) ≈ 10⁻⁴. **Every headline result
is effectively a linear-plant result**, so a reviewer asks: why the elaborate
nonlinear model at all?

**What I built.** `bifurcation_analysis.py` drives the plant into the regime
where the nonlinearity *is* the physics — finite-amplitude chatter — at a fixed
operating point, and measures the steady limit-cycle amplitude vs depth of cut.

**Result (measured, open loop, peak-to-peak of steady chatter).**

| a_p | nonlinear plant | linearised plant |
|---:|---:|---:|
| 0.08 mm | 1.3 µm | 1.3 µm  (dormant: identical) |
| 0.10 mm | 2.7 µm | 2.7 µm  (Hopf onset) |
| 0.25 mm | 233 µm | 262 µm |
| 0.60 mm | 6.7 mm | 3.2 mm (linear larger) |
| **1.00 mm** | **17 mm (bounded limit cycle)** | **→ ∞ (diverges)** |

The nonlinearity does exactly what the article asserts but never shows: it
**bounds chatter into a limit cycle** where the linear model blows up. The
controlled bifurcation cross-checks Gap #4: the LQG/RC-LQG Hopf point moves from
≈0.1 mm to ≈1.9–2.0 mm — matching the Floquet critical depth below (**Fig G3**).

**Consequence.** The paper should either (a) present this nonlinear-regime study
to *justify* the model, or (b) drop the nonlinear framing and report an honest
linear H₂ study. Right now it claims (a) while doing (b).

---

## Gap #4 — The stability-lobe diagram for the controlled system is a heuristic

**Evidence.** The README states the controlled SLD curves use a *"heuristic
effective-damping boost … an assumption, not a Floquet result."* So the
"28×/40× stability improvement" is an *assumed* damping multiplier fed to the
open-loop tool — not a stability result. Worse, the article's own open-loop tool
(`fdm_stability.py`) is a **per-mode, decoupled** approximation, while the
rank-1 regenerative term a4·DpDpᵀ couples the modes.

**What I built.** `closed_loop_stability.py` semi-discretises the **full coupled
closed loop** — plant + Kalman observer + controller (+ internal-model
resonators) with the delay τ and periodic a4(t) — and computes genuine Floquet
multipliers. Validated against the article's own tool in the open-loop limit
(ρ agrees to 3 decimals: 0.983 / 1.015 / 1.132).

**Result (measured, rigorous Floquet).**

| controller | critical a_p @ 4900 RPM | across 3000–7000 RPM | article's claim |
|:--|---:|---:|:--|
| Open loop | ≈ 0.06–0.10 mm | 0.06–0.10 mm | 0.10 mm ✓ |
| LQG | **1.90 mm** | 1.75–2.04 mm (≈ 20–30×) | 2.86 mm (28×), *heuristic* |
| Repetitive-LQG | **1.96 mm** | **1.0–2.0 mm** (RPM-dependent) | — |
| DARC | **= its LQG base ≈ 1.9 mm** | = LQG | 4.00 mm (40×) ✗ |

The controlled numbers converge across m_div = 30/50/70 (not a discretization
artifact). Two findings:

**(i) Key theoretical correction.** A **feedforward cannot move the stability
boundary** — it does not alter the characteristic equation of the regenerative
loop; it only cancels the *forced* response. So DARC's chatter limit equals its
LQG base's (~1.9 mm), **not** the claimed 4.0 mm. The article's "40× for DARC" is
wrong both numerically (heuristic) *and* conceptually (it conflates forced-
vibration reduction with stability-margin extension).

**(ii) Repetitive-LQG trades stability margin for forced rejection.** Its
critical a_p equals LQG's near 4900 RPM (1.96 vs 1.90 mm) but drops **below** LQG
at low speed (≈ 1.05 mm at 3000 RPM): the high-gain internal-model resonators
that make it dominate on forced vibration (Gap #1) erode the chatter margin at
some spindle speeds — a genuine, citable repetitive-control trade-off, and a
second axis (beyond sensor noise) on which DARC's feedforward is safer (**Fig
G2**).

---

## Gap #5 — "Deep **Adaptive Robust** Control": the adaptation is dead code

**Evidence.** In `darc_mpc_v3_controller.py::step`, `lambda_robust` is computed
from `OnlineRLSAdapter` and then **never used** — the control law is
`u = u_LQG + u_FF` unconditionally. Audited bit-exactly:

```
DARC adaptation ON  : y_rms = 0.295410 µm
DARC adaptation OFF : y_rms = 0.295410 µm     difference = 0.00e+00 µm
```

The "**A**daptive" and "**R**obust" of DARC are unsupported by the code. `RLS`,
`LyapunovSafetyFilter` (bypassed whenever the feedforward is active), and the
`pretrain_*` variants are similarly inert in the reported pipeline.

**What I built.** `robust_monte_carlo.py`:
- `audit_adaptation` — the on/off test above (a one-line, damning check).
- `monte_carlo_all` — runs **every** controller through the *same* uncertain
  plants (ω±2 %, ζ±20 %, K_T±5 %), unlike the package tool that only tests LQG,
  with bootstrap confidence intervals and **paired** comparisons.

**Result (measured, N = 40, fine sensor).**

| controller | mean y_RMS | 95 % CI | worst-case |
|:--|---:|:--|---:|
| LQG | 0.606 µm | [0.597, 0.616] | 0.641 µm |
| **RC-LQG** | **0.176 µm** | [0.173, 0.179] | 0.187 µm |
| DARC | 0.295 µm | [0.288, 0.303] | 0.325 µm |

Paired advantage (bootstrap 95 % CI): **DARC vs RC-LQG = −68 % [−70, −66],
P(DARC better) = 0.00** — with a fine sensor the fair feedback baseline beats
DARC on **every** uncertain plant. (DARC vs LQG = +51 % confirms the article's
number *against the handicapped baseline*.) **Fig G4.**

**Consequence.** Either implement a real adaptive/robust law (and show it helps
under a mismatch the nominal LQG cannot handle) or remove the "Adaptive Robust"
claim and the dead code; and report robustness with the multi-controller,
CI-backed protocol here.

---

## Secondary gaps (documented, not fully built out)

| # | Gap | Note |
|---|-----|------|
| 6 | **Single operating point.** One RPM (4900), one geometry, one feed. The feedforward is designed at a single tool position. | `closed_loop_stability` already sweeps RPM; extend the feedforward to be position-scheduled along the path. |
| 7 | **No experimental validation.** Pure simulation. | Acceptable for a first paper if framed honestly; the sensor-independence claim (Gap #1) is the most testable and should be the experimental headline. |
| 8 | **Controller realism.** The main comparison uses an ideal control path (only sensor noise); computational delay, DAC quantization and the piezo hysteresis/slew model (`piezo_actuator.py`) are excluded from the headline numbers. | Re-run Fig G1 through `main_realistic_piezo`'s actuator to test whether the crossover moves. |
| 9 | **NN residual marginal & unfalsifiable.** With the nonlinearity dormant, the "Deep nonlinear residual" cannot be learning nonlinear physics; its ~4 % full-path gain is a phase-indexed linear correction obtainable analytically with more harmonics. | Ablate the NN against "+N more feedforward harmonics" to isolate any genuine contribution. |

---

## What the honest, deepened paper should say

1. Keep the plant model, LQG, and DARC feedforward — they are sound and faithful.
2. **Replace** the "feedback cannot reject periodic disturbances" narrative with
   the **quantified sensor-dependence trade-off** (Gap #1, Fig G1): repetitive/
   internal-model feedback is *better* with a good sensor; DARC's feedforward
   wins because it is *sensor-independent* — the real, defensible contribution.
3. **Replace** the heuristic SLD with the rigorous closed-loop Floquet result
   (Gap #4, Fig G2), and correct the conceptual error that a feedforward extends
   the stability margin.
4. **Exercise the nonlinearity** (Gap #3, Fig G3) to justify the NDDE model, or
   drop the nonlinear framing.
5. **Audit or remove** the inert adaptation and report robustness with
   CI-backed, all-controller statistics (Gap #5, Fig G4).

Net effect: the same core method, but positioned so it survives review — a
narrower but **true** and **novel** claim (sensor-independent phase-feedforward
beats internal-model feedback under coarse sensing) instead of a broad claim the
data contradict.

---

### References
- B. A. Francis, W. M. Wonham, *The internal model principle of control theory*, Automatica 12 (1976) 457–465.
- S. Hara, Y. Yamamoto, T. Omata, M. Nakano, *Repetitive control system*, IEEE TAC 33 (1988) 659–668.
- T. Insperger, G. Stépán, *Updated semi-discretization method for periodic DDEs*, Int. J. Numer. Meth. Eng. 61 (2004) 117–141.
- G. Stépán, T. Kalmár-Nagy, *Nonlinear regenerative machine-tool vibrations*, ASME DETC (1997).
- K. Nasiri, H. Moradi, MSSP 224 (2025) 112198  (the adopted plant model).
