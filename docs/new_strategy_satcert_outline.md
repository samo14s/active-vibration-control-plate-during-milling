# SatCERT — المخطط الاستعراضي للاستراتيجية الجديدة
# Certified Regional Stability under Actuator Saturation for Active Chatter Control

**نوع الوثيقة:** مخطط استعراضي مخطَّط له (research outline / proposal) — يسبق التنفيذ.
**الفجوة المستهدفة:** الفجوة ③ (موثقة بالأدلة في `gap_scouting/gap3_*.md`).
**تاريخ:** 2026-07-20.

> **حالة التنفيذ (محدَّثة 2026-07-20، بعد تشغيل الحملة المعتمدة):**
> WP1 ✅ (رفعان: مستمر/Padé للتحقق مقابل `sld`، وعيّني-البيانات الدقيق
> تنفيذيًا عند معدل التحكم 50 kHz مع deadzone عند DAC — `avc/satcert.py`).
> WP2 ✅ **مع انحراف معلَن**: LMI/SDP القطاع المعمم الدوري ثبت أنه غير
> قابل للحل على ذاكرة عادية عند البعد المرفوع (موثق)؛ استُبدل بشهادة
> المجموعة المقبولة العظمى الخالية من الإشباع (Gilbert–Tan O∞، هامش جهد
> محلول طورياً، كلا إشارتي الخطوة) — مغلقة الصيغة، بلا مُحلّ، **ودقيقة في
> اتجاه الاضطراب المبلَّغ**؛ تحققت مقابل عتبة القص المقاسة في المحاكي
> اللاخطي بدقة 0.2–2 %. WP3 ✅ (جدول الأعماق المعتمدة + حساسية h_req +
> الإحصاء المناطقي عند 4.9 krpm). WP4 ✅ (جزيرتا إشباع مؤكدتان
> **بإسناد سببي** — `scripts/satcert_islands.py`: عند نقاط مستقرة
> خطيًا لـPS-LPV تُطلق خطوة سطح سالبة −38/−50 µm ارتجاجًا متناميًا
> بقصّ 99%، ويضمحل الاضطراب نفسه عند رفع حد الجهد؛ لا جزر للمتحكم
> المجمّد حتى ±500 µm — الجزر حكر على المتحكم عالي السلطة، اتساقًا مع
> آلية Ozsoy؛ استنساخ توليفة SDOF المنشورة يبقى لمرحلة المخطوطة).
> WP5 ⏸ مؤجَّل بمبرر: داخل شهادة منطقة-الخطية لا يعمل anti-windup إطلاقًا
> فلا يغيّر المجموعة المعتمدة؛ يُفعَّل فقط في امتداد النظام المشبع.
> WP6 ✅ (النص التأسيسي مملوء بالأرقام المعتمدة —
> `docs/paper2_foundational_text_final.md` §5).
> **النتيجة الرئيسية:** الاعتماد يقلب الترتيب الخطي: أفضلية PS-LPV
> الخطية 3.2× عند أسوأ موضع تصبح 2.7× عند تفاوت 1 µm وتنقلب إلى 0.79×
> عند 20 µm — أول برهان كمّي على أن هوامش الاستقرار الخطية تُسيء ترتيب
> متحكمات التفريز تحت الإشباع.

---

## الملخص التنفيذي (عربي)

كل متحكم ارتجاج نشط عملي يعمل خلف مكبّر جهد محدود (±150V في منصتنا
المرجعية). عند أعماق قطع طموحة يُشبع المشغّل، فيسقط ضمان الاستقرار الخطي —
وقد وثّق Ozsoy وSims وOzturk (MSSP 2025) تجريبيًا «جزر إشباع» يحدث فيها
الارتجاج **تحت** الحد الخطي المتنبأ به، بلا أي إطار تنبؤي معتمد. في المقابل
تملك نظرية التحكم آلات جاهزة (شروط القطاع المعمم لـTarbouriech، دوالّ
Lyapunov–Krasovskii، وIQC الدورية لـAltshuller) لم تُطبَّق على التصنيع قط.

**الاستراتيجية المقترحة SatCERT** تسدّ الفجوة من الجهتين:

1. **نمذجة**: حلقة التفريز المغلقة المشبعة تُرفع — عبر آلة semi-discretization
   المطورة في ورقتنا الأولى — إلى نظام دوري خطي منتهي البعد مع لاخطية
   deadzone محصورة قطاعيًا؛
2. **شهادة**: متباينات مصفوفية خطية (LMI) بدوالّ Lyapunov دورية + شرط
   القطاع المعمم تعطي **قطعًا ناقصًا لامتغيرًا معتمدًا** (تقدير حوض جذب) لكل
   نقطة تشغيل (عمق، سرعة، موضع)؛
3. **مُخرَج عملياتي**: مسح نقاط التشغيل يحوّل الشهادات إلى **مغلف عمق قطع
   مسموح معتمد** بثلاث مناطق: مضمون إقليميًا / مستقر خطيًا فقط (جزر إشباع
   محتملة) / غير مستقر — أول أداة تخطيط عمليات من نوعها؛
4. **تعظيم**: مكسب anti-windup يُصمم بتكرار LMI لتعظيم الحوض المعتمد؛
5. **ترسيخ بلا تجارب**: تقاطع مع الأحواض العددية من محاكينا اللاخطي المتحقق
   منه، واستنساخ ظاهرة جزر الإشباع المنشورة قياساتها في MSSP 2025 —
   نفس استراتيجية الترسيخ الأدبي التي أنجحت ورقتنا الأولى.

**لماذا هذه الفجوة؟** الظاهرة منشورة تجريبيًا وتنتظر نظرية؛ المُخرَج تشغيلي
لا نظري فقط؛ التمايز عن ورقتنا الأولى جذري (سؤال مختلف: ليس «أي مكاسب؟»
بل «إلى أي عمق يبقى الضمان قائمًا رغم الإشباع؟»)؛ وكل البنية الحسابية
اللازمة قائمة عندنا.

---

## 1. Gap statement (submission-ready wording)

> Every practical active chatter controller operates behind a bounded
> actuator. Saturation destroys the linear stability guarantee, and recent
> experiments (Ozsoy et al., MSSP 2025) document "saturation islands" in
> which chatter erupts below the linearly predicted boundary. Yet no
> published work provides a certified, computable, and maximized regional
> stability guarantee for a *given* (linear-authority or anti-windup-
> augmented) controller acting on the *true time-periodic, delayed* milling
> dynamics — nor the operational deliverable such a certificate enables: a
> permissible depth-of-cut/spindle-speed envelope with quantified basin
> margins. The saturated-delay LMI machinery (Tarbouriech lineage) and
> periodic delay-IQC theory (Altshuller 2008) exist off the shelf and have
> never been brought to machining.

Forbidden claims (from the dossier): do not claim "no regional analysis of
any kind" (Wu et al. 2016 contains proto-regional invariant-set language for
a bespoke adaptive law) and do not claim first anti-windup in machining.

## 2. The SatCERT architecture

```
                          ┌─────────────────────────────────────────────┐
                          │              OFFLINE  LAYER                 │
                          │                                             │
 validated plant model ──►│ 1. scalar-history semi-discretization LIFT  │
 (FEM + piezo + regen.    │    periodic DDE + deadzone  →  x_{k+1} =    │
  delay; paper-1 chain)   │    A_k x_k + B_k φ(K x_k),  φ ∈ sector      │
                          │                                             │
                          │ 2. REGIONAL CERTIFICATE (per operating pt)  │
                          │    periodic Lyapunov LMI + generalized      │
                          │    sector  →  invariant ellipsoid  E(P_k)   │
                          │                                             │
                          │ 3. ANTI-WINDUP SYNTHESIS                    │
                          │    iterative LMI maximizing vol E(P_k)      │
                          │                                             │
                          │ 4. ENVELOPE SWEEP over (a_p, Ω, x_T)        │
                          │    → certified permissible-depth envelope   │
                          │      {guaranteed | linear-only | unstable}  │
                          └───────────────┬─────────────────────────────┘
                                          │ gains + envelope
                                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │                     ONLINE  LAYER                        │
       │   y_s ─► [ H∞ controller (paper-1, frozen or PS-LPV) ]   │
       │              +  [ certified anti-windup  E_aw ]          │
       │              ─►  sat(±150 V)  ─►  piezo patch            │
       │   process planning reads the certified envelope          │
       └──────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌────────────────────────────────────────────┐
                  │             VALIDATION  LAYER              │
                  │ A. numerical basins (IC sweeps, nonlinear  │
                  │    simulator with loss-of-contact + sat)   │
                  │ B. reproduce measured saturation islands   │
                  │    of Ozsoy et al. (MSSP 2025)             │
                  │ C. cross-check: certificate ⊆ basin ⊆      │
                  │    island-free region                      │
                  └────────────────────────────────────────────┘
```

## 3. Mathematical skeleton

**3.1 Lifted saturated model.** Paper-1's semi-discretization exploits the
rank-one delay structure to lift the periodic DDE to
`z_{k+1} = Φ_k z_k` with `z` = [modal states; controller states; scalar
histories]. Saturation enters through the single actuation channel:
`u_applied = sat(v) = v − dz(v)` with `v = C_K z`. The lifted system becomes
a **discrete-time periodic linear system with one sector-bounded
nonlinearity**:

    z_{k+1} = A_k z_k + B_k dz(C_K z_k),   k = 0..m−1 (period m),
    dz(·) ∈ sector [0, 1] locally, generalized sector on a level set.

This bridge — periodic-DDE lifting + deadzone sector — is the technical
heart no one has built; everything downstream is (nontrivial but) standard.

**3.2 Regional certificate.** Periodic Lyapunov function
`V_k(z) = zᵀ P_k z`, `P_{k+m} = P_k`, with the Tarbouriech generalized
sector condition (auxiliary gain `G_k`): LMIs in `(P_k, G_k, τ_k)` certify
that the ellipsoid `E = {z : zᵀ P_0 z ≤ 1}` is invariant and contractive
over one period. Feasibility ⇒ every trajectory starting in E converges:
a computable basin estimate for the SATURATED PERIODIC DELAYED loop.

**3.3 Physical margin map.** The ellipsoid is translated into engineering
quantities: the largest surface-defect amplitude h_max / force impulse /
initial vibration amplitude whose lifted state stays in E — the "basin
margin" axis of the envelope.

**3.4 Anti-windup synthesis.** Static AW gain `E_aw` feeding
`(sat(v) − v)` back to the controller state: enters the LMIs bilinearly →
iterative (V-K) LMI scheme maximizing `log det P_0⁻¹` (basin volume).

**3.5 Envelope computation.** For each `(a_p, Ω)` on a grid (and positions
x_T): linear Floquet stable? if yes, largest certified margin from 3.2-3.3.
Three-zone map = the paper's headline figure. Saturation islands =
linearly-stable points whose *numerical* basin (validation layer) is small
— the certificate's conservatism gap quantifies exactly how much guarantee
one gives up in the islands.

## 4. Work packages (all executable on the existing codebase, no experiments)

| WP | المحتوى | يبني على | مخاطرة |
|---|---|---|---|
| WP1 | الرفع المشبع: deadzone في `avc/sld.py` lifting + تحقق مقابل المحاكي | sld.py, simulate.py | منخفضة |
| WP2 | شهادات LMI الدورية (cvxpy) + خريطة الهوامش الفيزيائية | جديد + controller.py | متوسطة (قابلية حل LMI بأبعاد الرفع — خطة تخفيض: اختزال متوازن للرافع) |
| WP3 | مسح المغلف + الأحواض العددية (IC sweeps) | pipeline.py نمط الحملة | منخفضة |
| WP4 | استنساخ جزر Ozsoy 2025 (نموذج SDOF من معاملاتهم المنشورة OA) | milling.py + محاكاة | متوسطة (توفر المعاملات في ورقتهم) |
| WP5 | توليف anti-windup الأعظمي + حملة مقارنة (بدون AW / AW ساكن / AW معتمد) | synthesis.py | متوسطة |
| WP6 | المخطوطة | قالب ورقة 1 | منخفضة |

## 5. Expected contributions (vetted wording)

1. **C1** — First certified, computable regional stability guarantee for a
   given saturated active chatter controller on the true time-periodic
   delayed milling dynamics (periodic Lyapunov + generalized sector on the
   semi-discretization lifting).
2. **C2** — The certified permissible depth-of-cut/speed envelope with
   quantified basin margins: a new operational process-planning deliverable.
3. **C3** — Certified explanation and prediction of the experimentally
   documented saturation-island phenomenon (anchored on the published MSSP
   2025 measurements), plus anti-windup synthesis that provably enlarges
   the certified region.

## 6. Target journals (in order)

1. **MSSP** — direct dialogue with Ozsoy et al. 2025 (the phenomenon paper
   is there); certificates + envelope answer their open question.
2. **JSV** — same rationale as paper 1; theory-friendly.
3. **Nonlinear Dynamics** — basins of periodic DDEs are its home turf.

## 7. Risks and mitigations

- **LMI dimension** (lifted state ~130): balanced truncation of the lifted
  map before certification; or certificate on the reduced design model with
  the truncation covered by the paper-1 uncertainty machinery.
- **Conservatism of the ellipsoidal estimate**: report side-by-side with
  numerical basins; conservatism *is* a result (it quantifies the guarantee
  price), not a failure.
- **Ozsoy configuration mismatch** (their rig is spindle-side): reproduce
  their SDOF configuration from their published parameters for the anchor,
  and demonstrate the envelope on our validated plate benchmark.

## 8. العلاقة بورقة PS-LPV الأولى

مستقلة سؤالًا ومنهجًا (شهادات إقليمية لاخطية مقابل توليف مجدول خطي)، لكنها
تعيد استخدام: النموذج المتحقق منه بالكامل، الرفع، المحاكي اللاخطي (الذي
يملك الإشباع أصلًا)، ونمط الحملة والشهادات. متحكمات ورقة 1 (المجمّد
وPS-LPV) تصبح «المتحكمات المعطاة» التي تُعتمد أحواضها — تكامل لا تكرار.
