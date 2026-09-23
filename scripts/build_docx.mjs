// Bangun proposal.docx dari naskah/sempro-skripsi.md atau naskah/semhas-skripsi.md,
// mengikuti Panduan Tugas Akhir UHB.
//
// Jalankan: node scripts/build_docx.mjs sempro   (atau: semhas)
//
// Kenapa dua langkah baca, bukan satu
// ------------------------------------
// Daftar Isi, Daftar Tabel, dan Daftar Lampiran harus tercetak di depan naskah, tetapi
// isinya (judul + nomor halaman) baru diketahui setelah SELURUH dokumen selesai dibaca.
// Karena itu `tokenisasi()` dipanggil sekali, hasilnya (daftar node datar) dipakai DUA
// KALI: sekali untuk mengumpulkan daftar judul dan penanda (bookmark) yang akan dirujuk,
// sekali lagi untuk benar-benar memancarkan paragraf docx. Nomor halaman sendiri tidak
// pernah dihitung manual — tiap entri daftar memakai `PageReference` ke bookmark yang
// sama, diselesaikan LibreOffice/Word saat dokumen dibuka atau dikonversi ke PDF.
//
// Kenapa bukan field TOC bawaan Word
// ------------------------------------
// docx.js menyediakan `TableOfContents`, tetapi diuji dulu: LibreOffice headless TIDAK
// mengevaluasi field itu saat ekspor PDF (halaman tetap kosong). `PageReference` polos,
// sebaliknya, terbukti diselesaikan dengan benar. Karena itu Daftar Isi/Tabel/Lampiran
// dibangun manual: tab-stop leader titik + `PageReference` per baris, bukan field TOC
// majemuk.
//
// Skema bagian dan penomoran halaman
// ------------------------------------
// - depan   : sampul sampai Daftar Singkatan — romawi kecil, tengah-bawah, spasi 1,5
// - BAB I/II/III (semhas: sampai V) : tiap BAB SATU Section docx.js tersendiri, sebab
//   "halaman pertama beda" (footer tengah-bawah, tanpa header) adalah properti Section,
//   dan harus diaktifkan ulang di tiap BAB. Nomor arab TIDAK diulang dari 1 tiap BAB;
//   hanya Section BAB pertama diberi start:1, sisanya melanjutkan.
// - akhir   : Daftar Pustaka + Lampiran A-D dalam SATU Section, arab lanjutan, SELALU
//   pojok kanan atas walau pada halaman pertama tiap Lampiran — panduan menyatakan itu
//   eksplisit ("nomor halaman pada daftar kepustakaan dan lampiran adalah nomor lanjutan
//   ... dan TIDAK merupakan bab baru").

import { readFileSync, writeFileSync, existsSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import {
  Document, Packer, Paragraph, TextRun, Header, Footer, PageNumber, ImageRun,
  AlignmentType, SectionType, PageBreak, Bookmark, PageReference, SimpleField,
  Table, TableRow, TableCell, WidthType, BorderStyle, VerticalAlign,
  TabStopType, TabStopPosition, LeaderType, ShadingType,
} from "docx";
import { tokenisasi, uraiSebaris } from "./lib_markdown_docx.mjs";

const AKAR = dirname(dirname(fileURLToPath(import.meta.url)));
const DOK = process.argv[2] || "sempro";
const SUMBER = DOK === "semhas"
  ? join(AKAR, "naskah", "semhas-skripsi.md")
  : join(AKAR, "naskah", "sempro-skripsi.md");
const KELUARAN = join(AKAR, "naskah", DOK === "semhas" ? "semhas-skripsi.docx" : "sempro-skripsi.docx");

// ── Ukuran dasar (twips kecuali disebut lain; 1 cm = 566,93 twips) ──────────
const FONT = "Times New Roman";
const PT = (p) => p * 20;                 // pt -> twips (spacing.before/after)
const SZ = (p) => p * 2;                  // pt -> half-point (font size docx.js)
const MARGIN = { top: 2268, bottom: 1701, left: 2268, right: 1701 };   // 4/3/4/3 cm
const LEBAR_CETAK = 11906 - MARGIN.left - MARGIN.right;                // ~7937 twips
const INDENT_ALINEA = 720;                // 0,5 in = 1,27 cm
const SPASI = { tunggal: 240, satuLima: 360, ganda: 480 };

let idBookmark = 0;
const bookmarkBaru = () => `bm${++idBookmark}`;

// ═══════════════════════════════════════════════════════════════════════════
// Markup sebaris -> TextRun[]
// ═══════════════════════════════════════════════════════════════════════════

// Istilah yang SUDAH dicetak miring di markdown (lewat seragamkan_istilah.py) apa adanya
// dipertahankan; fungsi ini hanya menerjemahkan penanda **/*/`  ke properti docx.
function jalankanTeks(teks, opsiDasar = {}) {
  const token = uraiSebaris(teks);
  return token.map((tk) => {
    if (tk.br) return new TextRun({ ...opsiDasar, break: 1 });
    return new TextRun({
      text: tk.teks,
      bold: opsiDasar.bold || tk.tebal,
      italics: opsiDasar.italics || tk.miring,
      font: tk.kode ? "Courier New" : (opsiDasar.font || FONT),
      size: opsiDasar.size || SZ(12),
      ...opsiDasar,
      // opsiDasar di atas boleh menimpa font/size default, tetapi bold/italics gabungan
      // (dasar ATAU markah sebaris) sudah dihitung lebih dulu supaya keduanya tidak
      // saling menimpa satu sama lain.
      bold: opsiDasar.bold || tk.tebal,
      italics: opsiDasar.italics || tk.miring,
    });
  });
}

function paragrafTeks(teks, { align = AlignmentType.JUSTIFIED, spasiBaris = SPASI.ganda,
  indent = undefined, before = 0, after = 0, size = SZ(12), bold = false } = {}) {
  return new Paragraph({
    alignment: align,
    spacing: { line: spasiBaris, before, after },
    indent,
    children: jalankanTeks(teks, { size, bold }),
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Header/Footer
// ═══════════════════════════════════════════════════════════════════════════

const nomorHalaman = (align) => new Paragraph({
  alignment: align,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SZ(12) })],
});
const footerRomawi = () => new Footer({ children: [nomorHalaman(AlignmentType.CENTER)] });
const footerKosong = () => new Footer({ children: [new Paragraph({ text: "" })] });
const headerKosong = () => new Header({ children: [new Paragraph({ text: "" })] });
const headerArab = () => new Header({ children: [nomorHalaman(AlignmentType.RIGHT)] });
const footerArabBawah = () => new Footer({ children: [nomorHalaman(AlignmentType.CENTER)] });

// ═══════════════════════════════════════════════════════════════════════════
// Tabel
// ═══════════════════════════════════════════════════════════════════════════

const GARIS_TIPIS = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const TANPA_GARIS = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

function lebarKolom(header, rows, ncol) {
  const semua = [header, ...rows];
  const rerata = [];
  for (let c = 0; c < ncol; c++) {
    let total = 0, n = 0;
    for (const r of semua) { total += (r[c] || "").length; n++; }
    rerata.push(Math.max(3, total / n));
  }
  const bobot = rerata.map((x) => Math.sqrt(x));
  const jumlah = bobot.reduce((a, b) => a + b, 0);
  return bobot.map((b) => Math.round((b / jumlah) * LEBAR_CETAK));
}

function selTabel(teks, { header = false, lebar, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    width: { size: lebar, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    borders: {
      top: header ? TANPA_GARIS : TANPA_GARIS, bottom: TANPA_GARIS,
      left: TANPA_GARIS, right: TANPA_GARIS,
    },
    children: [new Paragraph({
      alignment: align,
      spacing: { line: SPASI.tunggal },
      children: jalankanTeks(teks, { size: SZ(11), bold: header }),
    })],
  });
}

// Tabel bergaya tiga-garis (APA/booktabs): hanya garis atas, bawah header, dan bawah
// tabel. Panduan: "garis pemisah yang penting hanya tiga, dengan arah mendatar." Border
// per baris tidak diekspos lewat properti Table, sehingga baris header dibangun dengan
// sel yang border bawahnya diset tersendiri (GARIS_TIPIS), baris lain TANPA_GARIS, dan
// border LUAR tabel (atas/bawah) memberi dua garis sisanya.
function buatTabel(node) {
  const ncol = node.header.length;
  const lebar = lebarKolom(node.header, node.rows, ncol);

  const selHeader = (teks, c) => new TableCell({
    width: { size: lebar[c], type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    borders: { top: TANPA_GARIS, bottom: GARIS_TIPIS, left: TANPA_GARIS, right: TANPA_GARIS },
    children: [new Paragraph({
      alignment: AlignmentType.LEFT, spacing: { line: SPASI.tunggal },
      children: jalankanTeks(teks, { size: SZ(11), bold: true }),
    })],
  });

  // Daftar Singkatan (Lampiran 17) tidak berkepala kolom — glosarium dua kolom polos.
  // Panduan: "tabel tidak boleh dipenggal." cantSplit mencegah SATU baris terputus
  // di tengah kata melintasi batas halaman — cacat yang sungguh terjadi pada tabel
  // Jadwal Penelitian (16 baris) sebelum perbaikan ini, sebuah baris "Implementasi
  // baseline BiGRU" terbelah jadi "Implementasi baseline" di halaman lama dan
  // "BiGRU" sendirian di halaman baru. tableHeader mengulang baris kepala pada tiap
  // halaman kelanjutan sebagai jaring pengaman kedua: bila tabel yang panjang tetap
  // terpaksa terbagi, pembaca tidak perlu membalik halaman untuk tahu kolom mana yang
  // mana.
  const baris = node.sembunyikanHeader ? [] :
    [new TableRow({ cantSplit: true, tableHeader: true,
      children: node.header.map((h, c) => selHeader(h, c)) })];
  for (const r of node.rows) {
    baris.push(new TableRow({
      cantSplit: true,
      children: r.map((v, c) => selTabel(v ?? "", { lebar: lebar[c] })),
    }));
  }

  return new Table({
    width: { size: LEBAR_CETAK, type: WidthType.DXA },
    columnWidths: lebar,
    borders: {
      top: GARIS_TIPIS, bottom: GARIS_TIPIS, left: TANPA_GARIS, right: TANPA_GARIS,
      insideHorizontal: TANPA_GARIS, insideVertical: TANPA_GARIS,
    },
    rows: baris,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Baca dan tokenisasi
// ═══════════════════════════════════════════════════════════════════════════

if (!existsSync(SUMBER)) { console.error("tidak ada:", SUMBER); process.exit(1); }
const teksSumber = readFileSync(SUMBER, "utf8");
const NODE = tokenisasi(teksSumber);
console.log(`${NODE.length} node dari ${SUMBER}`);

// ═══════════════════════════════════════════════════════════════════════════
// LANGKAH 1 — kumpulkan bookmark, Daftar Isi, Daftar Tabel, Daftar Lampiran.
//
// Bookmark ditulis LANGSUNG ke objek node (node.bmId), sehingga Langkah 2 memakai id
// yang persis sama tanpa perlu menurunkannya ulang lewat logika terpisah yang bisa
// menyimpang dari Langkah 1.
// ═══════════════════════════════════════════════════════════════════════════

const ROMAWI_SUBBAB = /^[A-Z]\.\s+/;   // "A.  Latar Belakang Masalah"
const BAB = /^BAB\s+([IVX]+)\.\s+(.*)$/;
const LAMPIRAN_H = /^LAMPIRAN\s+([A-Z])\.\s+(.*)$/;

// Label Daftar Isi untuk H1 yang teks tercetak di halamannya BERBEDA dari label di
// Daftar Isi (contoh: sampul tidak mencetak "HALAMAN JUDUL" di halamannya sendiri,
// tetapi Lampiran 13 tetap mencantumkannya sebagai baris pertama Daftar Isi).
const LABEL_TAK_TERCETAK = new Set(["PROPOSAL SKRIPSI", "SKRIPSI"]);
const labelDaftarIsi = (teks) => LABEL_TAK_TERCETAK.has(teks) ? "HALAMAN JUDUL" : teks;

const daftarIsi = [];
const daftarTabel = [];
const daftarLampiran = [];
let dalamDepan = true;   // jadi false persis di H1 "BAB I" pertama

for (let i = 0; i < NODE.length; i++) {
  const n = NODE[i];

  if (n.tipe === "h1") {
    n.bmId = bookmarkBaru();
    const mBab = n.teks.match(BAB);
    if (mBab) dalamDepan = false;
    const label = mBab ? `BAB ${mBab[1]}    ${mBab[2]}` : labelDaftarIsi(n.teks);
    // Entri bagian depan dirujuk lewat SimpleField berswitch "\* roman": PageReference
    // polos hanya mengembalikan indeks halaman MENTAH (arab), tidak mewarisi format
    // romawi milik section tujuannya — teruji langsung pada dokumen ini: footer halaman
    // Daftar Isi sendiri benar menampilkan "x", tetapi PageReference tanpa switch ke
    // bookmark yang sama pernah kembali "10". Word membangun Daftar Isi bawaannya
    // dengan mekanisme switch yang sama, sehingga ini bukan tebakan.
    daftarIsi.push({ level: 1, teks: label, bmId: n.bmId, romawi: dalamDepan });
    const mLamp = n.teks.match(LAMPIRAN_H);
    if (mLamp) daftarLampiran.push({ teks: `Lampiran ${mLamp[1]}  ${mLamp[2]}`, bmId: n.bmId });
    continue;
  }

  if (n.tipe === "h2" && ROMAWI_SUBBAB.test(n.teks)) {
    n.bmId = bookmarkBaru();
    daftarIsi.push({ level: 2, teks: n.teks, bmId: n.bmId });
    continue;
  }

  if (n.tipe === "tabel" && n.caption) {
    n.bmId = bookmarkBaru();
    daftarTabel.push({ teks: n.caption, bmId: n.bmId });
  }
}

console.log(`Daftar Isi: ${daftarIsi.length} entri | Daftar Tabel: ${daftarTabel.length} `
  + `| Daftar Lampiran: ${daftarLampiran.length}`);

// ═══════════════════════════════════════════════════════════════════════════
// Baris berdaun titik + nomor halaman (dipakai Daftar Isi/Tabel/Lampiran)
// ═══════════════════════════════════════════════════════════════════════════

function barisDaftar(teks, bmId, indent = 0, romawi = false) {
  return new Paragraph({
    indent: indent ? { left: indent } : undefined,
    spacing: { line: SPASI.satuLima },
    tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX, leader: LeaderType.DOT }],
    children: [
      ...jalankanTeks(teks, { size: SZ(12) }),
      new TextRun({ text: "\t", size: SZ(12) }),
      romawi
        ? new SimpleField(`PAGEREF ${bmId} \\* roman`, "i")
        : new PageReference(bmId, { font: FONT, size: SZ(12) }),
    ],
  });
}

function bagianDaftarIsi() {
  return daftarIsi.map((e) => barisDaftar(e.teks, e.bmId, e.level === 2 ? 720 : 0, e.romawi));
}
function bagianDaftarTabel() {
  return daftarTabel.map((e) => barisDaftar(e.teks, e.bmId));
}
function bagianDaftarLampiran() {
  return daftarLampiran.map((e) => barisDaftar(e.teks, e.bmId));
}

// Daftar Singkatan tidak butuh nomor halaman (glosarium, bukan indeks lokasi) — dibaca
// langsung dari blok kode "AUC    Area Under..." yang sudah berbentuk final di markdown.
function bagianDaftarSingkatan(barisKode) {
  const baris = barisKode.filter((b) => b.trim());
  const rows = baris.map((b) => {
    const m = b.match(/^(\S+)\s+(.*)$/);
    return m ? [m[1], m[2]] : [b, ""];
  });
  return buatTabel({ header: ["Singkatan", "Kepanjangan"], rows, sembunyikanHeader: true });
}

// ═══════════════════════════════════════════════════════════════════════════
// Gambar: lebar cetak diturunkan dari rasio piksel PNG, bukan ditebak.
// ═══════════════════════════════════════════════════════════════════════════

function dimensiPng(jalur) {
  const b = readFileSync(jalur);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const CM = 566.93;
function ukuranGambar(jalur, lebarMaksTwips = LEBAR_CETAK) {
  const { w, h } = dimensiPng(jalur);
  const rasio = w / h;
  return { width: Math.round(lebarMaksTwips / 20), height: Math.round((lebarMaksTwips / rasio) / 20) };
  // docx.js ImageRun.transformation memakai satuan px-DXA (1/20 pt); dibagi 20 supaya
  // proporsional terhadap twips di atas tanpa memperkenalkan satuan ketiga.
}

// ═══════════════════════════════════════════════════════════════════════════
// LANGKAH 2 — pancarkan seluruh dokumen: potong per Section, terapkan gaya
// kontekstual (sampul vs prosa vs tabel), tempatkan Bookmark di tiap heading.
// ═══════════════════════════════════════════════════════════════════════════

const GAYA_SAMPUL = new Set(["PROPOSAL SKRIPSI", "SKRIPSI", "LEMBAR PERSETUJUAN", "LEMBAR PENGESAHAN"]);
// Baris seremonial (tanggal, label tanda tangan) dipusatkan APAPUN konteksnya, sekalipun
// berada di dalam H1 bergaya prosa (mis. tanda tangan penutup Kata Pengantar).
const POLA_SERTA_MERTA = /^(Oleh:|Disusun oleh:|Menyetujui,|Mengesahkan,|Pada hari|Tanggal|Dewan penguji:|Purwokerto,|NIM\.|NIK\.|Mengetahui,|Ketua Program Studi)/;

const ROMAWI = { I: 1, II: 2, III: 3, IV: 4, V: 5 };

let bagian = [];          // section docx.js yang sudah selesai
let anakSekarang = [];    // paragraf/tabel Section yang sedang dibangun
let propSekarang = null;  // {properties, headers, footers} Section berjalan
let h1Sekarang = "";
let babKe = 0;
let sudahAkhir = false;

function tutup() {
  if (propSekarang) bagian.push({ ...propSekarang, children: anakSekarang });
  anakSekarang = [];
}

function bukaSectionDepan() {
  tutup();
  propSekarang = {
    properties: { page: { size: { width: 11906, height: 16838 }, margin: MARGIN,
      pageNumbers: { start: 1, formatType: "lowerRoman" } } },
    footers: { default: footerRomawi() },
  };
}

function bukaSectionBab(romawi) {
  tutup();
  babKe++;
  const nomor = { start: babKe === 1 ? 1 : undefined, formatType: "decimal" };
  propSekarang = {
    properties: {
      type: SectionType.NEXT_PAGE,
      page: { size: { width: 11906, height: 16838 }, margin: MARGIN, pageNumbers: nomor },
      titlePage: true,
    },
    headers: { default: headerArab(), first: headerKosong() },
    footers: { default: footerKosong(), first: footerArabBawah() },
  };
}

function bukaSectionAkhir() {
  tutup();
  propSekarang = {
    properties: {
      type: SectionType.NEXT_PAGE,
      page: { size: { width: 11906, height: 16838 }, margin: MARGIN,
        pageNumbers: { formatType: "decimal" } },   // tanpa start: lanjutan, bukan bab baru
    },
    headers: { default: headerArab() },
    footers: { default: footerKosong() },
  };
}

bukaSectionDepan();

for (let i = 0; i < NODE.length; i++) {
  const n = NODE[i];

  // ── H1 ──────────────────────────────────────────────────────────────────
  if (n.tipe === "h1") {
    const mBab = n.teks.match(BAB);
    const mLamp = n.teks.match(LAMPIRAN_H);

    if (mBab) {
      bukaSectionBab(mBab[1]);
      h1Sekarang = n.teks;
      anakSekarang.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { line: SPASI.ganda, after: 0 },
        children: [new Bookmark({ id: n.bmId, children: [new TextRun({ text: `BAB ${mBab[1]}`, bold: true, font: FONT, size: SZ(12) })] })],
      }));
      anakSekarang.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { line: SPASI.ganda, after: PT(12) },
        children: [new TextRun({ text: mBab[2].toUpperCase(), bold: true, font: FONT, size: SZ(12) })],
      }));
      continue;
    }

    if (n.teks === "DAFTAR PUSTAKA") {
      bukaSectionAkhir();
      sudahAkhir = true;
      h1Sekarang = n.teks;
      anakSekarang.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { line: SPASI.ganda, after: PT(12) },
        children: [new Bookmark({ id: n.bmId, children: [new TextRun({ text: n.teks, bold: true, font: FONT, size: SZ(12) })] })],
      }));
      continue;
    }

    if (mLamp) {
      h1Sekarang = n.teks;
      anakSekarang.push(new Paragraph({
        pageBreakBefore: true,
        alignment: AlignmentType.CENTER, spacing: { line: SPASI.ganda, after: PT(12) },
        children: [new Bookmark({ id: n.bmId, children: jalankanTeks(`LAMPIRAN ${mLamp[1]}. ${mLamp[2]}`, { bold: true, size: SZ(12) }) })],
      }));
      continue;
    }

    // H1 depan (sampul s.d. Daftar Singkatan)
    h1Sekarang = n.teks;
    const label = labelDaftarIsi(n.teks);
    const cetak = !LABEL_TAK_TERCETAK.has(n.teks);
    anakSekarang.push(new Paragraph({
      pageBreakBefore: i !== 0,
      alignment: AlignmentType.CENTER, spacing: { line: SPASI.satuLima, after: cetak ? PT(12) : 0 },
      children: [new Bookmark({ id: n.bmId, children: cetak
        ? [new TextRun({ text: label, bold: true, font: FONT, size: SZ(12) })]
        : [new TextRun({ text: "", font: FONT, size: SZ(12) })] })],
    }));

    // Sisipkan konten yang dibangkitkan (bukan dari markdown) bagi keempat daftar,
    // lalu lewati blok kode stub yang menyusul di markdown.
    if (n.teks === "DAFTAR ISI") { anakSekarang.push(...bagianDaftarIsi()); if (NODE[i+1]?.tipe === "kode") i++; continue; }
    if (n.teks === "DAFTAR TABEL") { anakSekarang.push(...bagianDaftarTabel()); if (NODE[i+1]?.tipe === "kode") i++; continue; }
    if (n.teks === "DAFTAR LAMPIRAN") { anakSekarang.push(...bagianDaftarLampiran()); if (NODE[i+1]?.tipe === "kode") i++; continue; }
    if (n.teks === "DAFTAR SINGKATAN" && NODE[i+1]?.tipe === "kode") {
      anakSekarang.push(bagianDaftarSingkatan(NODE[i+1].baris)); i++; continue;
    }
    continue;
  }

  // ── H2 ──────────────────────────────────────────────────────────────────
  if (n.tipe === "h2") {
    if (n.bmId) {   // subbab huruf sungguhan (A./B./C. ...), terdaftar di Langkah 1
      anakSekarang.push(new Paragraph({
        alignment: AlignmentType.LEFT, spacing: { line: SPASI.tunggal, before: PT(12), after: PT(6) },
        children: [new Bookmark({ id: n.bmId, children: jalankanTeks(n.teks, { bold: true, size: SZ(12) }) })],
      }));
    } else {        // subjudul seremonial (mis. "PROPOSAL TUGAS AKHIR" di bawah sampul)
      anakSekarang.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { line: SPASI.satuLima, after: PT(6) },
        children: jalankanTeks(n.teks, { bold: true, size: SZ(12) }),
      }));
    }
    continue;
  }

  // ── H3 / H4 (anak dan anak-anak subbab, tidak masuk Daftar Isi) ──────────
  if (n.tipe === "h3" || n.tipe === "h4") {
    anakSekarang.push(new Paragraph({
      alignment: AlignmentType.LEFT, spacing: { line: SPASI.tunggal, before: PT(12), after: PT(4) },
      children: jalankanTeks(n.teks, { bold: true, size: SZ(12) }),
    }));
    continue;
  }

  // ── Paragraf ──────────────────────────────────────────────────────────
  if (n.tipe === "p") {
    const dlmPustaka = sudahAkhir && h1Sekarang === "DAFTAR PUSTAKA";
    const dlmSampul = GAYA_SAMPUL.has(h1Sekarang);
    const serta = POLA_SERTA_MERTA.test(n.teks.trim()) || /^\*\*[^*]+\*\*$/.test(n.teks.trim());
    const pusat = dlmSampul || serta;
    const spasiBaris = dlmPustaka ? SPASI.tunggal
      : (propSekarang?.properties?.page?.pageNumbers?.formatType === "lowerRoman" ? SPASI.satuLima : SPASI.ganda);
    // Entri Daftar Pustaka bergaya APA memakai HANGING indent (baris pertama rata kiri,
    // baris lanjutan menjorok) — kebalikan dari alinea biasa yang baris PERTAMAnya
    // menjorok. Percobaan pertama memakai firstLine untuk semua paragraf tanpa kecuali,
    // menghasilkan Daftar Pustaka terbalik dari konvensi APA.
    const indent = pusat ? undefined
      : dlmPustaka ? { left: INDENT_ALINEA, hanging: INDENT_ALINEA }
      : { firstLine: INDENT_ALINEA };
    anakSekarang.push(new Paragraph({
      alignment: pusat ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
      spacing: { line: spasiBaris, after: dlmPustaka ? PT(6) : 0 },
      indent,
      children: jalankanTeks(n.teks, { size: SZ(12) }),
    }));
    continue;
  }

  // ── Spasi vertikal (<br> tunggal di halaman depan) ───────────────────────
  // Font kecil (6pt) pada baris kosong ini sengaja, BUKAN cacat: <br> di sini murni
  // pemberi jarak dekoratif antarblok sampul, bukan baris teks sungguhan. Ukuran 12pt
  // penuh (dipakai pada percobaan pertama) membuat sampul dan lembar pengesahan
  // meluber ke halaman kedua padahal isinya sendiri muat.
  if (n.tipe === "spasi") {
    for (let k = 0; k < n.n; k++) {
      anakSekarang.push(new Paragraph({
        spacing: { line: SPASI.tunggal },
        children: [new TextRun({ text: "", size: SZ(6) })],
      }));
    }
    continue;
  }

  // ── Blockquote (catatan status dokumen) ──────────────────────────────────
  // Dipaksa halaman baru: markdown menandainya dengan DUA "---" berurutan (pemisah
  // ganda), dan tanpa itu isi sampul (NIM, prodi, tahun) meluber ke halaman berikutnya
  // karena logo 5 cm plus tumpukan spasi sudah nyaris memenuhi satu halaman sendiri.
  if (n.tipe === "quote") {
    anakSekarang.push(new Paragraph({
      pageBreakBefore: true,
      alignment: AlignmentType.JUSTIFIED,
      spacing: { line: SPASI.satuLima, before: PT(12), after: PT(12) },
      indent: { left: 720, right: 720 },
      border: { left: { style: BorderStyle.SINGLE, size: 8, color: "999999", space: 8 } },
      children: jalankanTeks(n.teks, { size: SZ(11) }),
    }));
    continue;
  }

  // ── Gambar ────────────────────────────────────────────────────────────
  if (n.tipe === "gambar") {
    const jalur = join(dirname(SUMBER), n.src);
    // Lambang UHB dicetak 5 cm sesuai instruksi panduan ("Lambang Universitas Harapan
    // Bangsa dengan size: 5 cm") — SATU-SATUNYA gambar dengan lebar tetap; sisanya
    // (kerangka teori/konsep) mengisi lebar cetak penuh. Sebelum perbaikan ini logo
    // memakai lebar penuh yang sama seperti gambar isi, meluap sampai ~14,6 cm dan
    // membuat sampul meluber ke halaman kedua.
    const lebarMaks = n.src.includes("logo_uhb") ? 5 * CM : LEBAR_CETAK;
    const ukuran = ukuranGambar(jalur, lebarMaks);
    anakSekarang.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { line: SPASI.satuLima, before: PT(12) },
      children: [new ImageRun({ data: readFileSync(jalur), transformation: ukuran, type: "png" })],
    }));
    if (n.caption) {
      anakSekarang.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { line: 276, after: PT(12) },  // 1,15 spasi
        children: jalankanTeks(n.caption.replace(/\s{2,}/g, "  "), { size: SZ(12) }),
      }));
    }
    continue;
  }

  // ── Tabel ─────────────────────────────────────────────────────────────
  if (n.tipe === "tabel") {
    if (n.caption) {
      anakSekarang.push(new Paragraph({
        alignment: AlignmentType.LEFT, spacing: { line: SPASI.tunggal, before: PT(12), after: PT(6) },
        children: jalankanTeks(n.caption, { bold: false, size: SZ(12) }),
        ...(n.bmId ? {} : {}),
      }));
      // Bookmark ditempatkan menempel pada baris judul supaya PageReference menunjuk
      // persis ke halaman tempat tabel itu muncul, bukan ke tempat lain.
      if (n.bmId) {
        anakSekarang[anakSekarang.length - 1] = new Paragraph({
          alignment: AlignmentType.LEFT, spacing: { line: SPASI.tunggal, before: PT(12), after: PT(6) },
          children: [new Bookmark({ id: n.bmId, children: jalankanTeks(n.caption, { size: SZ(12) }) })],
        });
      }
    }
    anakSekarang.push(buatTabel(n));
    anakSekarang.push(new Paragraph({ text: "", spacing: { line: SPASI.tunggal, after: PT(6) } }));
    continue;
  }

  // ── Bullet ────────────────────────────────────────────────────────────
  if (n.tipe === "bullet") {
    anakSekarang.push(new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { line: SPASI.ganda },
      indent: { left: 720, hanging: 360 },
      children: [new TextRun({ text: "•  ", font: FONT, size: SZ(12) }), ...jalankanTeks(n.teks, { size: SZ(12) })],
    }));
    continue;
  }

  // ── Blok kode (unduhan python/bash, diagram ASCII) ───────────────────────
  if (n.tipe === "kode") {
    for (const b of n.baris) {
      anakSekarang.push(new Paragraph({
        alignment: AlignmentType.LEFT, spacing: { line: SPASI.tunggal, after: 0 },
        children: [new TextRun({ text: b || " ", font: "Courier New", size: SZ(9) })],
      }));
    }
    anakSekarang.push(new Paragraph({ text: "", spacing: { line: SPASI.tunggal, after: PT(6) } }));
    continue;
  }

  if (n.tipe === "hr") continue;  // murni pemisah penulis, tidak tercetak
}

tutup();
console.log(`\n${bagian.length} Section docx.js siap dipancarkan`);

// ═══════════════════════════════════════════════════════════════════════════
// Rakit dan tulis
// ═══════════════════════════════════════════════════════════════════════════

const dokumen = new Document({
  sections: bagian,
  styles: { default: { document: { run: { font: FONT, size: SZ(12) } } } },
});

Packer.toBuffer(dokumen).then((buf) => {
  writeFileSync(KELUARAN, buf);
  console.log(`ditulis: ${KELUARAN} (${(buf.length / 1024).toFixed(0)} KB)`);
});
