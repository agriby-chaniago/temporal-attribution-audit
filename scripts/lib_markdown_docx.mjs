// Pengurai markdown khusus dialek naskah ini -> node terstruktur.
//
// Bukan pengurai markdown umum. Ia hanya mengenali konstruk yang benar-benar dipakai
// draftSempro/draftSemhas: heading #-####, tabel pipe, gambar, blok kode, blockquote,
// <br>, bullet "- ", dan markup sebaris (**tebal**, *miring*, `kode`).
//
// Dua langkah yang disengaja:
//   1. tokenisasi() memecah teks menjadi node datar, urut sebagaimana naskah.
//   2. Pemanggil (build_docx.mjs) berjalan DUA KALI atas node itu: sekali untuk
//      mengumpulkan judul dan judul tabel/lampiran (bagi Daftar Isi/Tabel/Lampiran
//      yang harus sudah lengkap SEBELUM isi BAB dipancarkan), sekali lagi untuk
//      benar-benar menulis dokumen. Node yang sama dipakai kedua kali sehingga
//      keduanya tidak mungkin berbeda urutan.

export function tokenisasi(teks) {
  const baris = teks.split("\n");
  const node = [];
  let i = 0;

  const kosong = (s) => s.trim() === "";

  while (i < baris.length) {
    const b = baris[i];

    if (kosong(b)) { i++; continue; }

    // heading
    let m = b.match(/^(#{1,4})\s+(.*)$/);
    if (m) {
      node.push({ tipe: "h" + m[1].length, teks: m[2].trim() });
      i++; continue;
    }

    // pembatas horizontal
    if (b.trim() === "---") { node.push({ tipe: "hr" }); i++; continue; }

    // blok kode
    if (b.trim().startsWith("```")) {
      const lang = b.trim().slice(3).trim();
      const isi = [];
      i++;
      while (i < baris.length && !baris[i].trim().startsWith("```")) { isi.push(baris[i]); i++; }
      i++; // lewati penutup ```
      node.push({ tipe: "kode", lang, baris: isi });
      continue;
    }

    // blockquote
    if (b.startsWith(">")) {
      const isi = [];
      while (i < baris.length && baris[i].startsWith(">")) {
        isi.push(baris[i].replace(/^>\s?/, "")); i++;
      }
      node.push({ tipe: "quote", teks: isi.join(" ") });
      continue;
    }

    // gambar, sendirian di baris
    m = b.trim().match(/^!\[([^\]]*)\]\(([^)]*)\)$/);
    if (m) {
      node.push({ tipe: "gambar", alt: m[1], src: m[2] });
      i++; continue;
    }

    // tabel: baris |...| diikuti baris pemisah |---|
    if (b.trim().startsWith("|") && i + 1 < baris.length &&
        /^\s*\|[\s:\-|]+\|\s*$/.test(baris[i + 1])) {
      const pecahBaris = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "")
        .split("|").map((s) => s.trim());
      const header = pecahBaris(b);
      i += 2;
      const rows = [];
      while (i < baris.length && baris[i].trim().startsWith("|")) {
        rows.push(pecahBaris(baris[i])); i++;
      }
      node.push({ tipe: "tabel", header, rows });
      continue;
    }

    // <br> sendirian: spasi vertikal pada halaman depan
    if (b.trim() === "<br>" || b.trim() === "<br><br>" || b.trim() === "<br><br><br>") {
      const n = (b.match(/<br>/g) || []).length;
      node.push({ tipe: "spasi", n });
      i++; continue;
    }

    // bullet "- "
    if (/^\s*-\s+/.test(b)) {
      const kumpul = [b.replace(/^\s*-\s+/, "")];
      i++;
      while (i < baris.length && !kosong(baris[i]) && !/^\s*-\s+/.test(baris[i]) &&
             !baris[i].match(/^#{1,4}\s/) && baris[i].trim() !== "---") {
        kumpul.push(baris[i]); i++;
      }
      node.push({ tipe: "bullet", teks: kumpul.join(" ") });
      continue;
    }

    // DUA ATAU LEBIH baris berurutan yang SELURUHNYA satu span tebal ("**PROGRAM STUDI
    // ...**", tanpa apa pun di luar tanda bintang) tidak pernah digabung satu sama lain.
    // Pola ini eksklusif dipakai blok sampul/tanda tangan ("PROGRAM STUDI ... / FAKULTAS
    // ... / 2026", tiap butir barisnya sendiri).
    //
    // Ambang "dua atau lebih" sengaja, BUKAN "satu atau lebih": percobaan pertama memakai
    // ambang satu baris dan salah menangkap "**3. Merancang evaluasi ... berjauhan.**" —
    // kalimat pembuka kontribusi yang kebetulan pas mengisi satu baris fisik penuh, lalu
    // badan paragrafnya menyusul di baris berikutnya. Baris tunggal semacam itu harus
    // tetap boleh menyambung ke baris sesudahnya (perilaku prosa biasa); hanya RUNTUN
    // sungguhan (>=2 baris berturut-turut) yang menjadi ciri blok sampul.
    const PENUH_TEBAL = /^\*\*[^*]+\*\*$/;
    if (PENUH_TEBAL.test(b.trim()) && PENUH_TEBAL.test((baris[i + 1] || "").trim())) {
      while (i < baris.length && PENUH_TEBAL.test(baris[i].trim())) {
        node.push({ tipe: "p", teks: baris[i].trim() }); i++;
      }
      continue;
    }

    // paragraf: kumpulkan baris sampai baris kosong / konstruk lain
    {
      const kumpul = [b];
      i++;
      while (i < baris.length && !kosong(baris[i]) &&
             !baris[i].match(/^#{1,4}\s/) && baris[i].trim() !== "---" &&
             !baris[i].trim().startsWith("```") && !baris[i].startsWith(">") &&
             !baris[i].trim().startsWith("|") && !/^\s*-\s+/.test(baris[i]) &&
             !baris[i].trim().match(/^!\[/) && !PENUH_TEBAL.test(baris[i].trim())) {
        kumpul.push(baris[i]); i++;
      }
      node.push({ tipe: "p", teks: kumpul.join(" ") });
    }
  }

  return gabungkanKonteks(node);
}

// Gabungkan node yang secara semantik satu kesatuan: judul tabel + tabelnya,
// gambar + keterangannya. Dilakukan sebagai langkah terpisah supaya tokenisasi()
// sendiri tetap sederhana satu-node-per-konstruk.
function gabungkanKonteks(node) {
  const keluar = [];
  for (let k = 0; k < node.length; k++) {
    const n = node[k];

    // "Tabel N.M Judul" diikuti tabel -> lampirkan sebagai caption
    if (n.tipe === "p" && /^Tabel \d+\.\d+ /.test(n.teks) &&
        node[k + 1] && node[k + 1].tipe === "tabel") {
      node[k + 1].caption = n.teks;
      continue;
    }

    // gambar diikuti "Gambar N  Judul" (skripsi) atau "**Fig. N.** ..." (naskah
    // jurnal Inggris, gaya Elsevier) -> lampirkan sebagai caption
    if (n.tipe === "gambar" && node[k + 1] && node[k + 1].tipe === "p" &&
        (/^Gambar \d+\s/.test(node[k + 1].teks) || /^\*\*Fig\. \d+\.\*\*/.test(node[k + 1].teks))) {
      n.caption = node[k + 1].teks;
      keluar.push(n); k++; continue;
    }

    keluar.push(n);
  }
  return keluar;
}

// ── Markup sebaris -> daftar {teks, tebal, miring, kode} ────────────────────
// Urutan penanganan: kode `...` dulu (isinya tidak diuraikan lebih lanjut),
// lalu ***tebal-miring***, **tebal**, *miring*.
// Pengurai **/*  REKURSIF, bukan regex alternasi datar seperti percobaan pertama.
//
// Alasannya konkret, bukan kehati-hatian teoretis: naskah ini memuat 17+ kemunculan
// **tebal berisi *miring* di tengahnya**. Regex alternasi datar `(\*\*[^*]+\*\*)|(\*[^*]+\*)`
// gagal pada pola itu \u2014 pencocok tebal menolak konten yang memuat tanda bintang APA PUN
// (termasuk pasangan miring yang sah di dalamnya), sehingga jatuh ke pencocok miring, yang
// lalu salah memasangkan bintang KEDUA milik pembuka "**" dengan bintang PERTAMA milik
// "*miring*" di tengah \u2014 bintang sisanya lolos sebagai teks harfiah. Diuji langsung dan
// terbukti: "**pada *k* berapa peta berhenti bergerak**" menghasilkan bintang nyasar
// tercetak apa adanya pada dokumen jadi.
//
// Pengurai di bawah memperlakukan "**" sebagai SATU delimiter atom (dicari lewat indexOf
// dua-karakter, bukan per-bintang), dan mengurai ULANG isi di antaranya secara rekursif \u2014
// sehingga *miring* yang bersarang di dalam **tebal** dikenali, bukan direbut.
function uraiRekursif(teks, tebalDasar, miringDasar) {
  const hasil = [];
  let i = 0;
  while (i < teks.length) {
    // `kode` ditangani DI SINI, bukan dipisah sebelum rekursi — sebab naskah memuat
    // "**Mengapa `d_model` tidak ikut disetel...**", span kode bersarang di dalam tebal.
    // Memisahnya lebih dulu (seperti percobaan pertama) memecah span tebal itu jadi tiga
    // potong independen pada batas backtick, sehingga tebalDasar hilang di tengah jalan.
    // Isi kode sendiri tidak diuraikan lebih lanjut untuk markup */**, tetapi tetap
    // mewarisi tebal/miring dari konteks luar (mis. bold+kode sekaligus tetap sah).
    if (teks[i] === "`") {
      const tutup = teks.indexOf("`", i + 1);
      if (tutup !== -1) {
        hasil.push({ teks: teks.slice(i + 1, tutup), tebal: tebalDasar, miring: miringDasar, kode: true });
        i = tutup + 1; continue;
      }
    }
    if (teks.startsWith("***", i)) {
      const tutup = teks.indexOf("***", i + 3);
      if (tutup !== -1) {
        for (const tk of uraiRekursif(teks.slice(i + 3, tutup), true, true)) hasil.push(tk);
        i = tutup + 3; continue;
      }
    }
    if (teks.startsWith("**", i)) {
      const tutup = teks.indexOf("**", i + 2);
      if (tutup !== -1) {
        for (const tk of uraiRekursif(teks.slice(i + 2, tutup), true, miringDasar)) hasil.push(tk);
        i = tutup + 2; continue;
      }
      // "**" tanpa pasangan penutup: bukan markup, dua bintang harfiah.
      hasil.push({ teks: "**", tebal: tebalDasar, miring: miringDasar, kode: false });
      i += 2; continue;
    }
    if (teks[i] === "*") {
      const tutup = teks.indexOf("*", i + 1);
      if (tutup !== -1) {
        for (const tk of uraiRekursif(teks.slice(i + 1, tutup), tebalDasar, true)) hasil.push(tk);
        i = tutup + 1; continue;
      }
      hasil.push({ teks: "*", tebal: tebalDasar, miring: miringDasar, kode: false });
      i += 1; continue;
    }
    let j = teks.length;
    for (const c of ["*", "`"]) {
      const p = teks.indexOf(c, i);
      if (p !== -1 && p < j) j = p;
    }
    if (j > i) hasil.push({ teks: teks.slice(i, j), tebal: tebalDasar, miring: miringDasar, kode: false });
    i = j;
  }
  return hasil;
}

export function uraiSebaris(teks) {
  teks = teks
    .replace(/&nbsp;/g, "\u00A0")
    .replace(/&amp;/g, "&");

  // <br> sebaris ditandai/dipisah LEBIH DAHULU \u2014 token struktural, tidak pernah bersarang
  // di dalam **tebal**/*miring* pada naskah ini (diperiksa: nol baris memuat keduanya
  // sekaligus). `kode` sebaris DITANGANI DI DALAM uraiRekursif, bukan di sini, supaya span
  // kode yang bersarang dalam tebal (mis. "**Mengapa `d_model` ...**") tidak memecah span
  // tebalnya.
  const bagian = teks.split(/(<br>)/);
  const token = [];
  for (const bag of bagian) {
    if (bag === "<br>") { token.push({ br: true }); continue; }
    for (const tk of uraiRekursif(bag, false, false)) token.push(tk);
  }
  return token;
}
