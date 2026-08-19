"""
build_pdf.py — Rapport final PDF (arabe, RTL) : verification + comparaison
==========================================================================
Assemble MODELE_PAPIER.md et COMPARAISON_ADRC_FOPID.md, insere les 12 figures
de verification et les 11 figures de comparaison a leur place, et imprime le
tout en PDF.

Le rendu passe par Chromium (Playwright) et non par ReportLab : c'est la seule
voie disponible ici qui mette en forme l'arabe correctement (HarfBuzz fait la
liaison des lettres et le sens droite-gauche). Les images sont encodees en
base64 dans le HTML, donc le fichier intermediaire est autonome.

    python3 report/build_pdf.py
"""
import base64
import os
import re
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_HTML = os.path.join(HERE, 'rapport_final.html')
OUT_PDF = os.path.join(HERE, 'rapport_final.pdf')
CHROME = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'

# --------------------------------------------------------------------------
# ou va chaque figure : (titre de section, fichier, legende)
FIGS_VERIF = [
    ('### 3.1', '09_ritz_model_identification.png',
     'تقارُب أساس Chebyshev–Ritz وجزاء الحافّة المُنشَبة، والتحقّق المتقاطع مع نموذج '
     'العناصر المنتهية Kirchhoff‑Q4، ونسبة (الجدول 4 «نظري») ÷ (النموذج) لكلّ نمط.'),
    ('### 3.3', '10_cutting_force_coefficients.png',
     'معاملا القطع ‎α₃(t)‎ و‎α₄(t)‎ على دور سنّ واحد عند الحالة S، ونسبة ‎α₄/ᾱ₄‎ مقابل '
     'حدود المعادلة (23)، وتقارُب التربيع العددي، وهندسة التلامس.'),
    ('### 3.4', '17_alpha4_band_stability.png',
     'إعادة إنتاج الشكل 6: فصوص الاستقرار بالمعامل النظري وبالمتوسّط وبـ‎0.3×‎ و‎2.9×‎ '
     'عند ثلاثة مواضع، مع نسبة السرعات التي تصمد فيها الإحاطة.'),
    ('### 3.5', '11_frf_and_antiresonances.png',
     'إعادة إنتاج الشكل 12: استجابة نقطة القيادة عند الزاوية العليا اليمنى وأصفارها '
     'المحسوبة بدقّة، ودالّة «توتر ← إزاحة» للوضعين، وجدول توزيع الأصفار.'),
    ('### 3.6', '19_patch_orientation.png',
     'الأوضاع الأربعة الممكنة للرقعة: بصمة أصفار الشكل 12(b) لا تُطابَق إلّا بالزاوية '
     'السفلى اليمنى، 60 mm على x و20 mm على z.'),
    ('### 3.7', '15_piezo_coupling_eq14_15.png',
     'الاقتران الكهرضغطي: ‎H_Pe‎ لكل نمط وإشارات البواقي للوضعين، ومقارنة المعادلة (15) '
     'بنموذج العزم المكافئ، وأثر الصيغة اللابُعدية مقابل اللابلاسيان الفيزيائي.'),
    ('### 3.8', '12_dtd_along_pass.png',
     'إعادة إنتاج الشكل 7: عناصر ‎D_PrᵀD_Pr‎ على طول الحافّة العليا مع القيم الفعلية '
     'والمتوسّطة والقصوى والدنيا.'),
    ('### 3.9', '16_additive_uncertainty_weights.png',
     'إعادة إنتاج الشكل 5: أقصى استجابة على كل المواضع مقابل الوزنين ‎W_Paf‎ و‎W_Pau‎ '
     'للمعادلتين (18)–(19).'),
    ('### 3.10', '13_open_loop_stability_lobes.png',
     'إعادة إنتاج الشكل 13: سطح الاستقرار على (سرعة × موضع)، وأدنى حدّ على كل المواضع '
     'مقابل السرعة، بالمعايرتين المقيسة والنظرية.'),
    ('### 3.11', '14_uncontrolled_time_response.png',
     'إعادة إنتاج الشكل 14(a): تباعُد الحالة S وطيفه، والحالة المستقرّة عند عمق أصغر — '
     'الخطّ المهيمن هو النمط الأوّل (534 Hz) لا 1135 Hz.'),
    ('### 3.12', '18_sign_convention.png',
     'اختبار إشارة ‎α₃, α₄‎: الفصوص بالإشارتين على المعايرتين، ونقطة الحالة S، وسرعات '
     'الشكل 18 التجريبي.'),
    ('### 3.13', '20_reduction_and_uncertainty.png',
     'تغطية المعادلة (25) لكلّ عنصر، والحالة المضطربة ‎±10 %‎ للشكل 16، واقتطاع النمطين '
     'مقابل خمسة أنماط، والتحقّق المتقاطع بين محرّكَي الاستقرار.'),
]

FIGS_COMP = [
    ('### 2.1', 'fig_lobes_A.png',
     'البروتوكول A: فصوص الاستقرار على الصفيحة الحقيقية. البنيتان المصمَّمتان على النموذج '
     'المُختزل تعطيان صفرًا عند كل سرعة.'),
    ('### 2.1', 'fig_summary_A.png',
     'البروتوكول A: جدول الأرقام الكامل.'),
    ('#### الحالة S للمقالة نفسها', 'fig_time_S_B.png',
     'الحالة S (‎4900 tr/min، ap = 0.30 mm‎): بلا تحكّم تتباعد عند ‎0.085 s‎ برجفان عند '
     '534 Hz؛ FOPID يُثبّت الممرّ كلّه (‎6.6 µm‎) بطيف لا يحوي سوى توافقيات مرور الأسنان؛ '
     'وADRC‑FOPID يتباعد عند ‎0.114 s‎.'),
    ('#### الحالة S للمقالة نفسها', 'fig_voltage_B.png',
     'التوتر عند الحالة S: FOPID يبلغ ‎61.8 V‎ ذروةً بلا أيّ تشبّع، وADRC‑FOPID يصل إلى '
     'سقف المضخّم ‎150 V‎.'),
    ('#### المتانة', 'fig_robust_B.png',
     'المتانة: FOPID أفضل في خمس حالات من ستّ، وينهار إلى الصفر عند ‎−10 %‎ على الكتلة '
     'والصلابة حيث يصمد ADRC‑FOPID عند ‎0.223 mm‎.'),
    ('#### آليّة ADRC', 'fig_eso_B.png',
     'آليّة ADRC مقيسة داخل المحاكاة: ‎z₁‎ يتابع القياس بخطأ ‎2.7 %‎، لكنّ ‎z₃‎ يخطئ '
     'الاضطراب الكلّي بـ‎147 %‎ وبارتباط سالب.'),
    ('## 3. الخلاصة', 'fig_lobes_B.png',
     'البروتوكول B: فصوص الاستقرار الكاملة — FOPID فوق ADRC‑FOPID عند كل السرعات تقريبًا.'),
    ('## 3. الخلاصة', 'fig_positions_B.png',
     'البروتوكول B: الحدّ حسب موضع الأداة عند سرعة التصميم. FOPID أعلى ذروةً، وADRC '
     'أكثر تسطُّحًا.'),
    ('## 3. الخلاصة', 'fig_frequency_B.png',
     'التوقيعات الترددية تحت القيود نفسها: ‎|K|‎ و‎|S|‎ والجهد لكل نيوتن.'),
    ('## 3. الخلاصة', 'fig_pso_B.png',
     'تقارُب PSO: تحرٍّ لكلّ اتّفاقية إشارة ثمّ تكرير على الاتّفاقية الحيّة، بالبذور نفسها.'),
    ('## 3. الخلاصة', 'fig_summary_B.png',
     'البروتوكول B: جدول الأرقام الكامل.'),
    ('## 3. الخلاصة', 'fig_protocols.png',
     'أثر نموذج التصميم: البروتوكول A مقابل B على الصفيحة نفسها.'),
]


FIGS_DIAG = [
    ('## 8. الحكم: ما العنصر الذي يجب تغييره رياضيًّا', 'fig_diagnostic_B.png',
     'التشخيص حلقةً حلقة: (a) الكسب الفعّال يقلب إشارته، (b) صفر في النصف الأيمن، '
     '(c) الـ ESO مضبوط تمامًا، (d) المتبقّي غير المُلغى يبلغ ذُراه عند الرنينات، '
     '(e) التشبّع عَرَض، (f) السقف الأساسي.'),
    ('## 8. الحكم: ما العنصر الذي يجب تغييره رياضيًّا',
     'fig_diagnostic_form_B.png',
     'الإجابة: (g) الـ ESO يضيف مُكامِلًا صحيحًا واحدًا بالضبط، (h) رُتَب `s` القابلة '
     'للتحقيق — المشتقّة الكسرية ‎μ∈(0,1)‎ خارج متناول ADRC‑FOPID، (i) ‎ω_o‎ بواجبَين '
     'متعارضَين، والعلاج كان داخل صندوق البحث ورُفض.'),
]


def b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def fig_html(path, caption, n):
    return (f'<figure><img src="data:image/png;base64,{b64(path)}"/>'
            f'<figcaption><b>الشكل {n}</b> — {caption}</figcaption></figure>')


def insert_figures(md, figs, base_dir, start_n):
    """Insere les figures juste avant le titre qui suit leur section."""
    n = start_n
    out = md
    for anchor, fname, caption in figs:
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            print(f'  ATTENTION : figure absente {path}')
            continue
        i = out.find(anchor)
        if i < 0:
            print(f'  ATTENTION : ancre absente "{anchor}"')
            continue
        # fin de section = prochain titre de niveau <= celui de l'ancre
        level = len(anchor) - len(anchor.lstrip('#'))
        j = len(out)
        for lv in range(1, level + 1):
            k = out.find('\n' + '#' * lv + ' ', i + len(anchor))
            if k >= 0:
                j = min(j, k)
        block = f'\n\n<!--FIG{n}-->\n\n'
        out = out[:j] + block + out[j:]
        out = out.replace(f'<!--FIG{n}-->', fig_html(path, caption, n))
        n += 1
    return out, n


CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
:root { --ink:#16202b; --mut:#5b6b7c; --line:#d7dee6; --acc:#1a3f8f;
        --acc2:#16a085; }
* { box-sizing: border-box; }
body { font-family: "DejaVu Sans", sans-serif; direction: rtl;
       text-align: right; color: var(--ink); font-size: 10.2pt;
       line-height: 1.75; margin: 0; }
h1,h2,h3,h4 { line-height: 1.45; page-break-after: avoid; color: var(--acc); }
h1 { font-size: 17pt; border-bottom: 2px solid var(--acc); padding-bottom: 5px;
     margin-top: 0; }
h2 { font-size: 13.5pt; margin-top: 22px; border-bottom: 1px solid var(--line);
     padding-bottom: 3px; }
h3 { font-size: 11.5pt; margin-top: 16px; color: #24303d; }
h4 { font-size: 10.5pt; margin-top: 12px; color: #35414e; }
p { margin: 7px 0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt;
       background: #f2f5f8; padding: 1px 3px; border-radius: 3px;
       direction: ltr; unicode-bidi: embed; }
pre { background: #f2f5f8; border: 1px solid var(--line); border-radius: 4px;
      padding: 8px 10px; overflow-x: auto; direction: ltr; text-align: left;
      page-break-inside: avoid; }
pre code { background: none; font-size: 8.2pt; line-height: 1.5; }
table { border-collapse: collapse; width: 100%; margin: 10px 0;
        font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid var(--line); padding: 4px 7px; text-align: right; }
th { background: #eef2f7; font-weight: bold; }
tr:nth-child(even) td { background: #fafbfc; }
blockquote { border-right: 3px solid var(--acc2); background: #f4fbf9;
             margin: 10px 0; padding: 7px 12px; color: #26404a;
             page-break-inside: avoid; }
figure { margin: 14px 0 18px; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; border: 1px solid var(--line); border-radius: 3px; }
figcaption { font-size: 8.6pt; color: var(--mut); margin-top: 5px;
             text-align: right; line-height: 1.5; }
hr { border: none; border-top: 1px solid var(--line); margin: 20px 0; }
a { color: var(--acc); text-decoration: none; }
.cover { height: 247mm; display: flex; flex-direction: column;
         justify-content: center; text-align: center; page-break-after: always; }
.cover .t { font-size: 24pt; font-weight: bold; color: var(--acc);
            line-height: 1.5; }
.cover .s { font-size: 12.5pt; color: var(--mut); margin-top: 18px;
            line-height: 1.9; }
.cover .ref { font-size: 10pt; color: var(--ink); margin-top: 30px;
              direction: ltr; text-align: center; line-height: 1.7; }
.cover .meta { font-size: 9.5pt; color: var(--mut); margin-top: 34px; }
.cover .box { border: 1px solid var(--line); border-radius: 6px;
              padding: 14px 18px; margin: 26px auto 0; max-width: 150mm;
              background: #f7f9fb; font-size: 9.6pt; text-align: right;
              line-height: 1.8; }
.part { page-break-before: always; }
.kv { font-size: 9.4pt; }
"""


def main():
    md_v = open(os.path.join(ROOT, 'MODELE_PAPIER.md')).read()
    md_c = open(os.path.join(ROOT, 'COMPARAISON_ADRC_FOPID.md')).read()
    md_d = open(os.path.join(ROOT, 'DIAGNOSTIC_ADRC.md')).read()

    md_v, n = insert_figures(md_v, FIGS_VERIF,
                             os.path.join(ROOT, 'figures', 'verification'), 1)
    md_c, n = insert_figures(md_c, FIGS_COMP,
                             os.path.join(ROOT, 'figures', 'comparison'), n)
    md_d, n = insert_figures(md_d, FIGS_DIAG,
                             os.path.join(ROOT, 'figures', 'comparison'), n)

    conv = markdown.Markdown(extensions=['tables', 'fenced_code', 'md_in_html'])
    html_v = conv.convert(md_v)
    conv.reset()
    html_c = conv.convert(md_c)
    conv.reset()
    html_d = conv.convert(md_d)

    cover = """
<div class="cover">
  <div class="t">التحكّم النشط في اهتزاز صفيحة رقيقة أثناء الفريزة</div>
  <div class="s">تحقيقٌ من نموذج المقالة،<br/>ومقارنة عادلة بين FOPID و ADRC‑FOPID</div>
  <div class="ref">J. Du, X. Liu, H. Dai, X. Long<br/>
    <i>Robust combined time delay control for milling chatter suppression of
    flexible workpieces</i><br/>
    International Journal of Mechanical Sciences <b>274</b> (2024) 109257</div>
  <div class="box">
    <b>ما يحتويه هذا التقرير.</b> الجزء الأوّل يحدّد النموذج الذي تستعمله المقالة
    معادلةً بمعادلة، ثمّ يتحقّق منه بثلاث عشرة تجربة عددية مستقلّة، كلٌّ منها بشكلها.
    الجزء الثاني يقارن بنيتَي تحكّم على النموذج نفسه ببروتوكول إنصاف قابل للتدقيق.
    الجزء الثالث يُشخّص رياضيًّا، حلقةً حلقة، لماذا يخسر ADRC‑FOPID هنا — وأيّ عنصر
    يجب تغييره.
    كلّ رقم في الصفحات التالية يُعاد إنتاجه بسكربت واحد في المستودع.
  </div>
  <div class="meta">مستودع <code>active-vibration-control-plate-during-milling</code>
    — الفرع <code>claude/adrc-fopid-comparison-woigyf</code></div>
</div>
"""
    doc = (f'<!doctype html><html lang="ar" dir="rtl"><head>'
           f'<meta charset="utf-8"><style>{CSS}</style></head><body>'
           f'{cover}'
           f'<div class="part">{html_v}</div>'
           f'<div class="part">{html_c}</div>'
           f'<div class="part">{html_d}</div>'
           f'</body></html>')

    with open(OUT_HTML, 'w') as f:
        f.write(doc)
    print(f'  HTML : {OUT_HTML}  ({len(doc) / 1e6:.1f} Mo)')

    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME,
                              args=['--no-sandbox', '--disable-dev-shm-usage'])
        pg = b.new_page()
        pg.goto('file://' + OUT_HTML, wait_until='networkidle')
        pg.pdf(path=OUT_PDF, format='A4', print_background=True,
               margin={'top': '18mm', 'bottom': '20mm',
                       'left': '16mm', 'right': '16mm'},
               display_header_footer=True,
               header_template='<div></div>',
               footer_template='<div style="width:100%;font-size:8px;'
                               'color:#5b6b7c;text-align:center;'
                               'font-family:DejaVu Sans">'
                               '<span class="pageNumber"></span> / '
                               '<span class="totalPages"></span></div>')
        b.close()
    print(f'  PDF  : {OUT_PDF}  ({os.path.getsize(OUT_PDF) / 1e6:.1f} Mo)')


if __name__ == '__main__':
    main()
