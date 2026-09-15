// Bangun manuskrip .docx dari Q1-amin/paper-*.md
//
//   node scripts/build_paper_docx.mjs full     -> paper-full.docx
//   node scripts/build_paper_docx.mjs fokus    -> paper-single-focus.docx
//
// Geometri halaman diambil dari Q1-amin/IJRCS-Template.docx (A4, satu kolom,
// margin 1484/1418/1418/1701 twips) supaya naskah terbaca sebagai artikel jurnal,
// BUKAN sebagai skripsi. Tidak ada branding IJRCS di sini: jurnal sasaran belum
// ditetapkan, dan header jurnal yang salah lebih buruk daripada tanpa header.
//
// Berbeda dari scripts/build_docx.mjs (yang mematuhi Panduan Tugas Akhir UHB:
// spasi ganda, penomoran romawi/arab, halaman depan). Naskah jurnal tidak memakai
// satu pun dari itu, sehingga dibuat terpisah alih-alih ditambal dengan flag.

import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  Document, Packer, Paragraph, TextRun, Footer, PageNumber, ImageRun,
  AlignmentType, Table, TableRow, TableCell, WidthType, BorderStyle,
} from "docx";
import { tokenisasi, uraiSebaris } from "./lib_markdown_docx.mjs";

const AKAR = join(dirname(fileURLToPath(import.meta.url)), "..");
const varian = (process.argv[2] || "full").toLowerCase();
const SUMBER = varian === "fokus"
  ? join(AKAR, "Q1-amin", "paper-single-focus.md")
  : join(AKAR, "Q1-amin", "paper-full.md");
const KELUARAN = varian === "fokus"
  ? join(AKAR, "Q1-amin", "paper-single-focus.docx")
  : join(AKAR, "Q1-amin", "paper-full.docx");

const FONT = "Times New Roman";
const SZ = (p) => p * 2;          // docx memakai setengah-poin
const PT = (p) => p * 20;         // twips
const MARGIN = { top: 1484, right: 1418, bottom: 1418, left: 1701 };
const LEBAR_CETAK = 11907 - MARGIN.left - MARGIN.right;

// ── markup sebaris -> TextRun ────────────────────────────────────────────────
function runs(teks, { size = SZ(10), bold = false, italics = false } = {}) {
  return uraiSebaris(teks).map((tk) => {
    if (tk.br) return new TextRun({ break: 1 });
    return new TextRun({
      text: tk.teks,
      bold: bold || tk.tebal,
      italics: italics || tk.miring,
      font: tk.kode ? "Courier New" : FONT,
      size: tk.kode ? size - 2 : size,
    });
  });
}

const para = (teks, o = {}) => new Paragraph({
  alignment: o.align ?? AlignmentType.JUSTIFIED,
  spacing: { line: o.line ?? 240, before: PT(o.before ?? 0), after: PT(o.after ?? 6) },
  indent: o.indent,
  children: runs(teks, o),
});

// ── tabel tiga garis (booktabs), lazim pada naskah ilmiah ───────────────────
const GARIS = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
const KOSONG = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

// Lebar kolom berbobot akar — teks panjang dapat ruang lebih, tetapi tidak sebanding
// lurus, supaya satu kolom prosa tidak menghimpit sisanya.
//
// MIN_KOL ada karena percobaan pertama tanpa itu memampatkan kolom "Ref." sampai
// "[10]" pecah jadi dua baris ("[10" lalu "]"). Bobot akar memberi kolom pendek
// porsi yang terlalu kecil begitu kolom lain memuat kalimat penuh.
const MIN_KOL = 820;

// Lebar minimum per kolom TIDAK boleh flat: tabel 8-kolom (mis. Tabel 7, header
// "Architecture" dan "Difference") pernah pecah header di tengah kata memakai
// MIN_KOL konstan, sebab konstanta itu dicocokkan terhadap tabel sempit ber-5-kolom
// (Tabel 1) dan tidak cukup lebar untuk kata tunggal terpanjang pada tabel yang
// kolomnya lebih banyak. Diganti per-kolom: sekitar lebar kata terpanjang di
// header/isi kolom itu sendiri, bukan angka global.
const LEBAR_PER_KARAKTER = 105; // twips, perkiraan lebar rerata karakter Times 9pt

function lebarKolom(node) {
  const bobot = node.header.map((h, i) => {
    const isi = [h, ...node.rows.map((r) => r[i] ?? "")];
    return Math.sqrt(Math.max(...isi.map((x) => String(x).length), 1));
  });
  const total = bobot.reduce((a, b) => a + b, 0);
  let lebar = bobot.map((b) => Math.round((b / total) * LEBAR_CETAK));

  const minKolom = node.header.map((h, i) => {
    const isi = [h, ...node.rows.map((r) => r[i] ?? "")];
    const kataTerpanjang = Math.max(
      ...isi.flatMap((x) => String(x).split(/\s+/).map((k) => k.length)), 1);
    return kataTerpanjang * LEBAR_PER_KARAKTER;
  });

  // Angkat kolom sempit ke minimumnya sendiri, lalu tarik kembali kelebihannya dari
  // kolom yang masih di atas ambang, proporsional terhadap kelonggarannya.
  const kurang = lebar.reduce((a, w, i) => a + Math.max(0, minKolom[i] - w), 0);
  if (kurang > 0) {
    const longgar = lebar.reduce((a, w, i) => a + Math.max(0, w - minKolom[i]), 0);
    lebar = lebar.map((w, i) => (w < minKolom[i] ? minKolom[i]
      : longgar > 0 ? w - Math.round(((w - minKolom[i]) / longgar) * kurang) : w));
  }
  return lebar;
}

function sel(teks, { lebar, header = false, atas = false, bawah = false }) {
  return new TableCell({
    width: { size: lebar, type: WidthType.DXA },
    margins: { top: PT(3), bottom: PT(3), left: PT(4), right: PT(4) },
    borders: {
      top: atas ? GARIS : KOSONG,
      bottom: bawah || header ? GARIS : KOSONG,
      left: KOSONG, right: KOSONG,
    },
    children: [new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { line: 240, before: 0, after: 0 },
      children: runs(teks, { size: SZ(9), bold: header }),
    })],
  });
}

// ── gambar ───────────────────────────────────────────────────────────────────
// Pola identik scripts/build_docx.mjs:313-325 (dimensiPng/ukuranGambar) — dibaca
// dari IHDR chunk PNG langsung, tanpa pustaka gambar, lalu diskalakan proporsional
// ke lebar cetak. Diulang di sini alih-alih diimpor sebab build_docx.mjs tidak
// mengekspornya dan kedua berkas sengaja tidak saling bergantung (lihat catatan
// berkas di atas: naskah jurnal dan skripsi dijaga terpisah).
function dimensiPng(jalur) {
  const b = readFileSync(jalur);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function ukuranGambar(jalur, lebarMaksTwips = LEBAR_CETAK) {
  const { w, h } = dimensiPng(jalur);
  const rasio = w / h;
  return { width: Math.round(lebarMaksTwips / 20), height: Math.round((lebarMaksTwips / rasio) / 20) };
}

function buatTabel(node) {
  const lebar = lebarKolom(node);
  const baris = [new TableRow({
    cantSplit: true, tableHeader: true,
    children: node.header.map((h, c) => sel(h, { lebar: lebar[c], header: true, atas: true })),
  })];
  node.rows.forEach((r, ri) => {
    const akhir = ri === node.rows.length - 1;
    baris.push(new TableRow({
      cantSplit: true,
      children: r.map((v, c) => sel(v ?? "", { lebar: lebar[c], bawah: akhir })),
    }));
  });
  return new Table({ columnWidths: lebar, rows: baris });
}

// ── rakit dokumen ────────────────────────────────────────────────────────────
const NODE = tokenisasi(readFileSync(SUMBER, "utf8"));
const anak = [];

// Front matter: seluruh node sebelum "## Abstract" diperlakukan posisional —
// node 0 judul, node 1 penulis, sisanya afiliasi/korespondensi.
let i = 0;
const batas = NODE.findIndex((n) => n.tipe === "h2");
const depan = NODE.slice(0, batas);
depan.forEach((n, k) => {
  if (k === 0) {
    anak.push(para(n.teks, { align: AlignmentType.CENTER, size: SZ(17), bold: true, after: 10 }));
  } else if (k === 1) {
    anak.push(para(n.teks, { align: AlignmentType.CENTER, size: SZ(11), after: 8 }));
  } else {
    anak.push(para(n.teks, { align: AlignmentType.CENTER, size: SZ(9), after: 2 }));
  }
});
i = batas;

let dlmPustaka = false;
for (; i < NODE.length; i++) {
  const n = NODE[i];

  if (n.tipe === "h2") {
    dlmPustaka = /^references$/i.test(n.teks.trim());
    anak.push(para(n.teks, {
      align: AlignmentType.LEFT, size: SZ(12), bold: true, before: 12, after: 6,
    }));
    continue;
  }
  if (n.tipe === "h3") {
    anak.push(para(n.teks, {
      align: AlignmentType.LEFT, size: SZ(11), bold: true, before: 8, after: 4,
    }));
    continue;
  }
  if (n.tipe === "tabel") {
    if (n.caption) {
      anak.push(para(n.caption, { align: AlignmentType.LEFT, size: SZ(9), before: 6, after: 3 }));
    }
    anak.push(buatTabel(n));
    anak.push(para("", { after: 6 }));
    continue;
  }
  if (n.tipe === "hr" || n.tipe === "spasi") continue;

  // Gambar: dipusatkan, diskalakan ke lebar cetak, keterangan DI BAWAH gambar
  // (konvensi Elsevier -- kebalikan tabel yang keterangannya di ATAS, sudah benar
  // di atas). n.caption diisi tokenisasi() bila baris berikutnya "**Fig. N.** ...".
  if (n.tipe === "gambar") {
    const jalur = join(AKAR, n.src);
    const ukuran = ukuranGambar(jalur);
    anak.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: PT(6), after: n.caption ? PT(3) : PT(6) },
      children: [new ImageRun({ data: readFileSync(jalur), transformation: ukuran, type: "png" })],
    }));
    if (n.caption) {
      anak.push(para(n.caption, { align: AlignmentType.CENTER, size: SZ(9), after: 8 }));
    }
    continue;
  }

  // Butir daftar. Ditangani eksplisit, bukan dibiarkan jatuh ke cabang paragraf:
  // tanpa ini penanda butirnya hilang diam-diam dan daftar terbaca sebagai prosa.
  if (n.tipe === "bullet") {
    anak.push(new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { line: 240, before: 0, after: PT(3) },
      indent: { left: 340, hanging: 170 },
      children: [new TextRun({ text: "•  ", font: FONT, size: SZ(10) }),
        ...runs(n.teks, { size: SZ(10) })],
    }));
    continue;
  }

  // Daftar pustaka: gantung (hanging indent) sesuai konvensi IEEE.
  if (dlmPustaka) {
    anak.push(para(n.teks, {
      size: SZ(9), after: 4,
      indent: { left: 340, hanging: 340 },
    }));
    continue;
  }

  // Judul tabel yang berdiri sendiri (mis. "**Table 1. ...**") tidak dijorokkan.
  const judulTabel = /^\*\*Table \d+\./.test(n.teks);
  anak.push(para(n.teks, { size: SZ(10), after: 6, indent: judulTabel ? undefined : undefined }));
}

const dokumen = new Document({
  creator: "Agriby Diandra Chaniago",
  title: depan[0]?.teks ?? "Manuscript",
  styles: { default: { document: { run: { font: FONT, size: SZ(10) } } } },
  sections: [{
    properties: { page: { size: { width: 11907, height: 16840 }, margin: MARGIN } },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SZ(9) })],
        })],
      }),
    },
    children: anak,
  }],
});

Packer.toBuffer(dokumen).then((buf) => {
  writeFileSync(KELUARAN, buf);
  console.log(`${NODE.length} node dari ${SUMBER}`);
  console.log(`ditulis: ${KELUARAN} (${Math.round(buf.length / 1024)} KB)`);
});
