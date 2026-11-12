# Presentasi 16 November 2026

**Beda Peta, Masih Relevan?**

Audit transfer ranking attention pada model tulisan tangan Parkinsonian.
Materi berbahasa Indonesia. Istilah teknis didefinisikan dalam slide, catatan
pembicara, dan glosarium.

## Berkas untuk dipakai

- [PowerPoint yang dapat diedit](presentasi_attention_16nov2026.pptx).
- [PDF dengan isi yang sama](presentasi_attention_16nov2026.pdf).
- [Catatan pembicara per slide](CATATAN_PEMBICARA.md).
- [Jawaban untuk diskusi](notes/TANYA_JAWAB.md).

Slide **1–21** membentuk cerita utama. Slide **22–35** adalah cadangan angka,
metode, sensitivitas, dan sumber. Untuk versi singkat, gunakan urutan
1, 2, 5–7, 9–16, 19, dan 21. Waktu bicara perlu diuji saat latihan; paket ini
tidak mengklaim durasi presentasi sudah diukur.

## Bukti yang digunakan

Dasarnya adalah manuskrip V9 dan hasil tersimpan sampai 25 Oktober 2026.
PDF V9 yang dibaca adalah `english/build/main/main.pdf`. Materi ini disiapkan
pada 12 November untuk presentasi 16 November. Tanggal run eksperimen tetap
Oktober. Tidak ada backdating atau klaim preregistrasi publik.

Bank utama mempunyai dua arsitektur, dua dataset, dua puluh seed dan lima fold.
Terdapat 400 identitas checkpoint. Primer NewHandPD memakai 17 seed baru di kedua
peran. Primer UCI-SST memakai 20 seed baru. Discovery tiga seed dan AUROC historis
sepuluh seed diberi label terpisah.

Angka, CI dan estimand tidak dihitung ulang. Grafik native memakai estimasi
tersimpan. Error bar mengambil endpoint CI yang sama. Persentase coverage hanya
mengubah hitungan numerator/denominator menjadi tampilan persen. Tidak ada
training, inference, intervensi, bootstrap, atau analisis penelitian baru.

## Cerita dan klaim

- [Enam alternatif judul dan strategi scarcity](notes/JUDUL_DAN_SCARCITY.md).
- [Audit klaim, bukti dan batasnya](notes/CLAIM_AUDIT.md).
- [Pemetaan 43 slide lama ke materi baru](notes/PETA_PRESENTASI_LAMA_KE_BARU.md).
- [Tiga pembanding audit dan batas kesepadanan](notes/LITERATURE_COMPARISON.md).
- [Provenance sumber](sources/PROVENANCE.json).
- [Rencana dan catatan seluruh slide](SLIDE_PLAN.json).

Nilai khusus studi dibangun dari pertanyaan, pasangan rekaman/checkpoint,
fixed target, dan kontrol RMS/geometri. Presentasi tidak mengklaim peta rusak,
attention selalu paling baik, atau validitas fisiologis yang sudah teruji.
Judul baru hanya diterapkan ke presentasi. Manuskrip V9 tetap utuh.

## Struktur dan sumber yang dapat diedit

`build_presentation.cjs` menghasilkan teks, diagram, tabel dan delapan chart
PowerPoint native. `tools/finalize_deck.py` menambahkan tiga set CI native dan
kolom error yang sesuai ke workbook chart. Nilai respons asli dipertahankan.
Dua scatter cadangan adalah gambar empiris tersimpan dari V9, bukan data
ilustratif. Seluruh catatan pembicara juga tertanam pada PPTX.

`sources/` memuat salinan CSV/JSON yang diperlukan untuk memeriksa angka.
`review/` memuat ikatan angka tampilan, geometri slide dan hasil pemeriksaan.
Render PNG penuh dan teks ekstraksi dokumen privat disimpan lokal serta tidak
ikut publikasi GitHub.

## Membangun ulang tanpa menjalankan penelitian

Perintah berikut dijalankan dari root repository. Lingkungan ini terpisah dari
lingkungan komputasi penelitian. Node dan LibreOffice diperlukan.

```bash
rtk proxy python3.11 -m venv untuk_presentasi_16nov26/.tools-venv
rtk proxy untuk_presentasi_16nov26/.tools-venv/bin/python -m pip install -r untuk_presentasi_16nov26/requirements-presentation.txt
rtk proxy npm --prefix untuk_presentasi_16nov26 ci
rtk proxy untuk_presentasi_16nov26/.tools-venv/bin/python untuk_presentasi_16nov26/tools/build_all.py --pdf
rtk proxy untuk_presentasi_16nov26/.tools-venv/bin/python untuk_presentasi_16nov26/tools/verify_presentation.py
rtk proxy untuk_presentasi_16nov26/.tools-venv/bin/python untuk_presentasi_16nov26/tools/render_review.py
```

Build standar memakai snapshot bukti yang sudah disertakan. Ia tidak membuka
notebook atau model. `tools/build_evidence.py` hanya diperlukan untuk mengambil
ulang salinan dari workspace penelitian asli; source identity dikunci dan
script tersebut tidak menghitung estimator.

Pemeriksaan XML/XSD juga dilakukan dengan validator Office dari skill `pptx`.
Detail lingkup aktual ada di `review/VALIDATION_REPORT.md`. Render otomatis tidak
digunakan sebagai pengganti inspeksi visual.

## Rujukan dan atribusi data

Presentasi menggunakan 15 artikel jurnal terpilih dari tahun 2021 ke atas.
Daftar lengkap dan DOI tersedia di `sources/references.json`. Bibliografi
manuskrip tidak diganti oleh daftar presentasi ini.

UCI 395: Isenkul, Sakar dan Kursun, *Parkinson Disease Spiral Drawings Using
Digitized Graphics Tablet*, UCI Machine Learning Repository (2013),
[DOI 10.24432/C5Q01S](https://doi.org/10.24432/C5Q01S), CC BY 4.0. Audit replikasi
memakai SST, bukan seluruh task.

NewHandPD: sumber resmi [HandPD / NewHandPD, UNESP](https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/).
Publikasi akuisisi yang diminta pemilik dataset adalah C. R. Pereira, S. A. T.
Weber, C. Hook, G. H. Rosa dan J. P. Papa, *Deep learning-aided Parkinson’s
disease diagnosis from handwritten dynamics*, SIBGRAPI 2016. Ini **prosiding
akuisisi dataset**, bukan artikel jurnal atau pembanding audit yang disamarkan.
Ketentuan dataset tetap mengikuti `LISENSI-DATA.md` pada root repository.

Tidak ada sinyal mentah, nama pasien, nomor identitas, checkpoint, atau cache
sinyal yang dimasukkan. Contoh attention memakai kode subjek dataset yang sama
dengan artefak hasil.

## Alat penyusunan

Codex membantu menyusun materi, kode tampilan dan pemeriksaan sumber. PPTX dibuat
dengan PptxGenJS 4.0.1. PDF diekspor dengan LibreOffice. Grafik hasil tidak
dihasilkan oleh model gambar generatif. Penilaian ilmiah dan penyampaian akhir
tetap perlu ditinjau penulis. Paket ini tidak menyatakan eksperimen atau manuskrip
sudah siap diterima jurnal.
