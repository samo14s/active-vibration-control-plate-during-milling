const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ImageRun, PageBreak, BorderStyle,
  TabStopType, LevelFormat, ShadingType, convertMillimetersToTwip
} = require('docx');

const ROOT = '/home/user/active-vibration-control-plate-during-milling';
const SCRATCH = __dirname;

function pngDims(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const USABLE_MM = 160;           // A4 21cm - 2x2.5cm margins
const EMU_PER_MM = 36000;

function figParagraphs(rel, caption) {
  const file = path.join(ROOT, rel);
  const { w, h } = pngDims(file);
  const wMM = USABLE_MM;
  const hMM = wMM * h / w;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 80 },
      children: [new ImageRun({
        type: 'png',
        data: fs.readFileSync(file),
        transformation: { width: Math.round(wMM * 3.7795), height: Math.round(hMM * 3.7795) },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: caption, size: 22, italics: false, font: 'Times New Roman' })],
    }),
  ];
}

function tableBlock(caption, rows) {
  const nCols = rows[0].length;
  const total = 9026;                          // DXA usable width
  const colW = Math.floor(total / nCols);
  const widths = Array(nCols).fill(colW);
  widths[0] = total - colW * (nCols - 1);
  const mkCell = (txt, hdr) => new TableCell({
    width: { size: widths[0], type: WidthType.DXA },
    shading: hdr ? { type: ShadingType.CLEAR, fill: 'E8E8E8' } : undefined,
    margins: { top: 60, bottom: 60, left: 80, right: 80 },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: txt, bold: hdr, size: 21, font: 'Times New Roman' })],
    })],
  });
  const table = new Table({
    columnWidths: widths,
    width: { size: total, type: WidthType.DXA },
    rows: rows.map((r, i) => new TableRow({
      children: r.map(c => {
        const cell = mkCell(c, i === 0);
        return cell;
      }),
    })),
  });
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 80 },
      children: [new TextRun({ text: caption, size: 22, font: 'Times New Roman' })],
    }),
    table,
    new Paragraph({ spacing: { after: 160 }, children: [] }),
  ];
}

function parse(file) {
  const lines = fs.readFileSync(file, 'utf-8').split('\n');
  const out = [];
  let i = 0;
  let pendingTabCap = null;
  while (i < lines.length) {
    const l = lines[i];
    const t = l.trim();
    if (!t) { i++; continue; }
    if (t.startsWith('# ')) {
      out.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 240, after: 240 },
        children: [new TextRun({ text: t.slice(2), bold: true, size: 32, font: 'Times New Roman', color: '000000' })],
      }));
    } else if (t.startsWith('## ')) {
      out.push(new Paragraph({
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 240, after: 160 },
        children: [new TextRun({ text: t.slice(3), bold: true, size: 28, font: 'Times New Roman', color: '000000' })],
      }));
    } else if (t.startsWith('### ')) {
      out.push(new Paragraph({
        heading: HeadingLevel.HEADING_3,
        spacing: { before: 200, after: 120 },
        children: [new TextRun({ text: t.slice(4), bold: true, size: 24, font: 'Times New Roman', color: '000000' })],
      }));
    } else if (t.startsWith('[EQ:')) {
      const body = t.slice(4, t.lastIndexOf(']'));
      const bar = body.lastIndexOf('|');
      const eq = body.slice(0, bar);
      const num = body.slice(bar + 1);
      out.push(new Paragraph({
        tabStops: [{ type: TabStopType.RIGHT, position: 9026 }],
        spacing: { before: 120, after: 120 },
        indent: { left: 567 },
        children: [
          new TextRun({ text: eq, italics: true, size: 24, font: 'Times New Roman' }),
          new TextRun({ text: '\t' + num, size: 24, font: 'Times New Roman' }),
        ],
      }));
    } else if (t.startsWith('[FIG:')) {
      const body = t.slice(5, t.lastIndexOf(']'));
      const bar = body.indexOf('|');
      out.push(...figParagraphs(body.slice(0, bar), body.slice(bar + 1)));
    } else if (t.startsWith('[TABCAP:')) {
      pendingTabCap = t.slice(8, t.lastIndexOf(']'));
    } else if (t === '[TAB]') {
      const rows = [];
      i++;
      while (lines[i].trim() !== '[/TAB]') {
        rows.push(lines[i].trim().split('|'));
        i++;
      }
      out.push(...tableBlock(pendingTabCap || '', rows));
      pendingTabCap = null;
    } else if (t.startsWith('- ')) {
      out.push(new Paragraph({
        numbering: { reference: 'bullets', level: 0 },
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 80 },
        children: [new TextRun({ text: t.slice(2), size: 24, font: 'Times New Roman' })],
      }));
    } else {
      out.push(new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 120, line: 300 },
        children: [new TextRun({ text: t, size: 24, font: 'Times New Roman' })],
      }));
    }
    i++;
  }
  return out;
}

const ch3 = parse(path.join(SCRATCH, 'ch3.txt'));
const ch4 = parse(path.join(SCRATCH, 'ch4.txt'));

const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: '•',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  styles: {
    default: {
      document: { run: { font: 'Times New Roman', size: 24 } },
    },
  },
  sections: [{
    properties: {
      page: { margin: { top: 1417, bottom: 1417, left: 1417, right: 1417 } },
    },
    children: [
      ...ch3,
      new Paragraph({ children: [new PageBreak()] }),
      ...ch4,
    ],
  }],
});

Packer.toBuffer(doc).then(b => {
  const out = path.join(SCRATCH, 'Soufi_these_chapitres_3_4.docx');
  fs.writeFileSync(out, b);
  console.log('WROTE', out, b.length, 'bytes');
});
