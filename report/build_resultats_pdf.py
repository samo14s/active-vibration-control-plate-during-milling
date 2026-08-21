"""
build_resultats_pdf.py — RESULTATS.md en PDF autonome (arabe, RTL)
==================================================================
Le meme rendu que `build_pdf.py` — Chromium via Playwright, parce que c'est la
seule voie disponible ici qui mette en forme l'arabe correctement (HarfBuzz
fait la liaison des lettres et le sens droite-gauche) — mais sur le seul
RESULTATS.md, sans les figures ni les quatre autres parties.

La feuille de style est IMPORTEE de build_pdf.py plutot que recopiee : deux
copies divergent, et un resume qui ne ressemble plus au rapport dont il est le
resume est une regression silencieuse.

    python3 report/build_resultats_pdf.py
"""
import os
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_pdf import CSS, CHROME                           # noqa: E402

OUT_HTML = os.path.join(HERE, 'resultats.html')
OUT_PDF = os.path.join(HERE, 'RESULTATS.pdf')

COVER = """
<div class="cover">
  <div class="t">النتائج النهائية</div>
  <div class="s">إحدى عشرة بنية تحكّم على المِحكّ نفسه<br/>
    صفيحة رقيقة تحت الفريزة</div>
  <div class="box">
    <b>ما يحتويه هذا الملخّص.</b> الجواب على السؤال الأصلي، والجدول الكامل
    بشقّيه، والانقسام بين البنى المُولَّفة والمضبوطة بالمكاسب مع جدول الأقطاب
    الذي يسنده، وثلاثة عيوب عدديّة وما أفسده كلٌّ منها، واختبار الرُّتَب،
    والتشتّت، وما لا يراه فلوكيه، والقيود المُعلَنة، وثلاث نتائج سُحبت.
    <b>كلّ رقم فيه فُحص برمجيًّا مقابل ملفّات <code>results/</code> قبل
    الإيداع.</b>
  </div>
  <div class="meta">مستودع <code>active-vibration-control-plate-during-milling</code>
    — الفرع <code>main</code></div>
</div>
"""


def main():
    md = open(os.path.join(ROOT, 'RESULTATS.md')).read()
    # Le titre de niveau 1 et la citation d'entete sont deja portes par la
    # couverture : les garder ferait doublon des la premiere page.
    lignes = md.split('\n')
    i = next((k for k, l in enumerate(lignes) if l.startswith('---')), 0)
    md = '\n'.join(lignes[i + 1:])

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
