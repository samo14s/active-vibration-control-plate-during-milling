"""
build_bilan_pdf.py — BILAN_SCIENTIFIQUE.md en PDF autonome (arabe, RTL)
=======================================================================
Meme rendu que build_pdf.py — Chromium via Playwright, seule voie disponible
ici qui mette en forme l'arabe correctement (HarfBuzz fait la liaison des
lettres et le sens droite-gauche).

La feuille de style est IMPORTEE de build_pdf.py plutot que recopiee, pour la
meme raison qu'en tete de build_resultats_pdf.py : deux copies divergent.

    python3 report/build_bilan_pdf.py
"""
import os
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_pdf import CSS, CHROME                           # noqa: E402

OUT_HTML = os.path.join(HERE, 'bilan.html')
OUT_PDF = os.path.join(HERE, 'BILAN_SCIENTIFIQUE.pdf')

COVER = """
<div class="cover">
  <div class="t">هل لدينا مساهمة؟</div>
  <div class="s">حصيلةٌ علميّة للعمل الحالي<br/>
    التحكّم النشط في اهتزاز صفيحةٍ رقيقة أثناء الفريزة</div>
  <div class="box">
    <b>لماذا هذا التقرير.</b> أربع أفكارٍ اقتُرحت في المراحل الأولى،
    و<b>سقطت أربعتها أمام الأدبيات</b>. وما بقي هو ثلاثة قياساتٍ لم تكن
    مخطّطة، ظهرت أثناء محاولة إثبات الأفكار الساقطة. ولأنّ الحكم على
    المساهمة لا يصحّ إلّا بمعرفة ما سقط ولماذا، يعرض هذا التقرير السلسلة
    كاملةً — <b>بما فيها الأخطاء المنهجيّة التي وقعتُ فيها وصُحّحت</b>،
    وأهمّها أنّ ثلاث مراحل أنتجت أحكامًا سلبيّة بأداةٍ لم يُتحقّق قطّ من
    أنّها تنجح على شيء.
  </div>
  <div class="meta">مستودع <code>active-vibration-control-plate-during-milling</code>
    — الفرع <code>claude/adrc-fopid-comparison-woigyf</code></div>
</div>
"""


def main():
    md = open(os.path.join(ROOT, 'BILAN_SCIENTIFIQUE.md')).read()
    conv = markdown.Markdown(extensions=['tables', 'fenced_code'])
    doc = (f'<!doctype html><html lang="ar" dir="rtl"><head>'
           f'<meta charset="utf-8"><style>{CSS}</style></head><body>'
           f'{COVER}<div class="part">{conv.convert(md)}</div>'
           f'</body></html>')
    with open(OUT_HTML, 'w') as f:
        f.write(doc)
    print(f'  HTML : {OUT_HTML}')

    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
    os.environ['PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD'] = '1'
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        br = pw.chromium.launch(executable_path=CHROME)
        pg = br.new_page()
        pg.goto('file://' + OUT_HTML, wait_until='networkidle')
        pg.pdf(path=OUT_PDF, format='A4', print_background=True,
               margin=dict(top='16mm', bottom='16mm',
                           left='14mm', right='14mm'),
               display_header_footer=True,
               header_template='<div></div>',
               footer_template='<div style="width:100%;text-align:center;'
                               'font-size:8px;color:#888;">'
                               '<span class="pageNumber"></span> / '
                               '<span class="totalPages"></span></div>')
        br.close()
    print(f'  PDF  : {OUT_PDF}  '
          f'({os.path.getsize(OUT_PDF) / 1e6:.2f} Mo)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
