# التحكّم النشط في اهتزاز صفيحة رقيقة أثناء الفريزة

نموذج ومقارنة متحكّمات، مبنيّان على:

> J. Du, X. Liu, H. Dai, X. Long, *Robust combined time delay control for milling
> chatter suppression of flexible workpieces*, **International Journal of Mechanical
> Sciences 274 (2024) 109257**.

المستودع يجيب على سؤالين منفصلين، ولكلٍّ منهما مجلّده وتقريره وأشكاله:

| السؤال | التقرير | الشيفرة | الأشكال |
|---|---|---|---|
| ما هو النموذج المستعمل في المقالة، وهل نُقل بأمانة؟ | `MODELE_PAPIER.md` | `paper_model/` + `verification/09..18` | `figures/verification/` |
| أيّهما أفضل بين FOPID و ADRC‑FOPID، بمقارنة عادلة؟ | `COMPARAISON_ADRC_FOPID.md` | `control/` | `figures/comparison/` |
| تدقيق النموذج بعناصر منتهية (عمل سابق) | `VERIFICATION.md` | `simulation/` + `verification/01..08` | — |

## البنية

```
paper_model/     نموذج المقالة نفسه : Chebyshev–Ritz (المعادلات 6–15، الملحق أ)
                 + قوى القطع (المعادلات 2–5) + استقرار فلوكيه + تكامل زمني
control/         FOPID و ADRC-FOPID + PSO + بروتوكول المقارنة العادلة
simulation/      تنفيذ مستقلّ بعناصر منتهية Kirchhoff-Q4 (يُستعمل كتحقّق متقاطع)
verification/    سكربتات التحقّق : 01–08 لنموذج العناصر المنتهية، 09–18 لنموذج المقالة
figures/         مخرجات الأشكال (verification/ و comparison/)
results/         مخرجات رقمية (‎.npz‎) لإعادة إنتاج الأشكال بلا إعادة حساب
```

## التشغيل

```bash
pip install numpy scipy matplotlib

# 1) التحقّق من نموذج المقالة (كل سكربت يطبع جدولًا رقميًا وينتج شكلًا)
python3 verification/09_ritz_model_identification.py
python3 verification/10_cutting_force_coefficients.py
...
python3 verification/18_sign_convention.py

# 2) المقارنة العادلة : بروتوكولان (A تصميم على نمطين، B تصميم على خمسة)
cd control
PROTOCOL=B python3 run_pso.py        # تحسين المتحكّمين بشروط متطابقة
PROTOCOL=B python3 run_compare.py    # التقييم الكامل على نموذج الخمسة أنماط
PROTOCOL=B python3 audit_fairness.py # تدقيق بروتوكول الإنصاف نفسه
PROTOCOL=B python3 run_eso_trace.py  # قياس آليّة ADRC داخل المحاكاة
PROTOCOL=B python3 figures.py        # أشكال البروتوكول
python3 figures.py --cross           # شكل المقارنة بين البروتوكولين
```

نصوص الأشكال بالإنجليزية لأنّ matplotlib لا يشكّل العربية؛ تعليقات الشيفرة بالفرنسية
كما في بقيّة المستودع، والتقارير بالعربية.
