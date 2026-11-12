# Perubahan dari presentasi September ke 16 November 2026

## Tambahan 12 November — versi tanpa angka dan judul paper

- Tambah `tanpa_angka/` sebagai deck kualitatif terpisah. Deck kuantitatif
  tidak ditimpa. Versi tambahan memakai 21 slide utama dan 5 cadangan.
- Sembunyikan statistik hasil dan sumbu numerik. Tanggal acara serta nama resmi
  BiMamba-2 tetap ada. Kurva contoh menggunakan bobot empiris yang sama.
- Pertahankan hasil magnitude yang menguntungkan pembanding, ketidakpastian
  plasebo, sensitivitas dosis, dan batas cakupan matching.
- Selaraskan judul paper melalui salinan lokal V10. Judul, running title dan
  metadata berubah. Abstrak, isi ilmiah, hasil dan referensi tidak berubah.
- Catat riwayat serta batas istilah relevansi pada `notes/PAPER_TITLE_V10.md`.

## Paket awal

Paket dibuat sebagai folder baru. Deck September, manuskrip V9 dan hasil asli
dipertahankan. Judul manuskrip tidak diubah.

## Isi

- Ganti judul dengan **Beda Peta, Masih Relevan?**. Enam pilihan dan alasan
  ilmiahnya tersedia pada catatan judul.
- Ganti tabel rencana kosong dengan hasil nyata. Materi mempunyai 21 slide utama
  dan 14 cadangan, tanpa penggandaan slide bahasa.
- Pisahkan bank AUROC historis, discovery tiga seed dan replikasi dua dataset
  dengan dua puluh seed.
- Tegaskan target tetap saat ranking sumber digunakan untuk memilih patch input.
- Tampilkan keuntungan own/transfer, kontras berpasangan, coverage, asosiasi JSD,
  placebo, sensitivitas dosis dan pengaruh seed.
- Pertahankan hasil magnitude yang menguntungkan pembanding. Tampilkan baseline
  lengkap dan irisan E2 secara terpisah.
- Pertahankan AUROC, Brier/ECE, usia, baseline klasik dan marker sebagai konteks
  dengan bank serta batas yang sesuai.
- Hilangkan klaim aktif tentang dua ground truth independen, fidelity ceiling,
  pemulihan fidelity, keberhasilan model beku lintas kohort dan preregistrasi
  publik. Riwayatnya tetap dijelaskan, bukan dihapus.

## Penyajian

- Gunakan pertanyaan–desain–hasil–manfaat sebagai alur.
- Tulis satu gagasan utama per kalimat. Pindahkan penjelasan panjang ke catatan
  pembicara dan cadangan.
- Bangun delapan grafik native PowerPoint dan diagram yang dapat diedit.
- Tambahkan CI native dengan endpoint dari CSV tersimpan. Angka tidak dihitung
  ulang; workbook chart diperluas dengan panjang error bar literal.
- Bedakan skala matching 0–100% dari gate stabil 0–20%.
- Pertahankan dua scatter empiris V9 dengan sumbu JSD penuh.
- Perbaiki heading panjang, collision judul/sumbu, serta aksen nama Šimić.

## Pemeriksaan

Pemeriksaan dokumen mencocokkan file sumber, field numerik yang disalin, 53 CI
tampilan dan nilai chart. Pemeriksaan struktur PPTX mencakup XML, relasi dan
chart. PDF diekspor dari PPTX final dan semua 35 halaman dirender. Lingkup
inspeksi visual sebenarnya dicatat dalam `review/VALIDATION_REPORT.md`.

Tidak ada training, inference, intervensi, bootstrap atau analisis ilmiah baru.
Pembaruan GitHub hanya memublikasikan materi presentasi dan dokumentasi yang
telah diperiksa. Zenodo tidak diperbarui pada tahap ini.
