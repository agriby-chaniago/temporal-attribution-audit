# Laporan pemeriksaan aktual

Pemeriksaan selesai pada 12 November 2026 untuk presentasi 16 November.

## Hasil dokumen dan sumber

- PPTX final: 35 slide, dengan 21 slide utama dan 14 cadangan.
- PDF: 35 halaman hasil ekspor dari PPTX final.
- Catatan pembicara: tersedia pada semua 35 slide dan pada Markdown.
- Delapan chart PowerPoint native. Tiga chart menampilkan CI custom native.
- 18,899 field numerik CSV yang disalin cocok dengan sumber.
  Ini bukan jumlah angka yang seluruhnya ditampilkan atau replikasi penelitian.
- 53 ikatan estimasi–CI tampilan cocok dengan row/field CSV.
- 19,317 pemeriksaan dokumen terstruktur, 0 gagal.
- 1845 file presentasi lama dan V9 cocok dengan snapshot awal.
- Source scalar, chart values dan endpoint CI tidak dihitung ulang sebagai estimator.

## Struktur dan kompatibilitas

Validator Office dari skill pptx dijalankan terhadap PPTX final. Output aktual:

```text
All validations PASSED!
```

Pemeriksaan ini mencakup XML/XSD, relasi, content types dan chart. Native CI
menggunakan referensi workbook sehingga muncul pada ekspor LibreOffice.
Nilai seri respons asli dipertahankan; kolom panjang error bar ditambahkan.

PPTX telah diekspor dengan LibreOffice 26.8.1.1. Paket tidak diklaim sudah dibuka
secara manual di Microsoft PowerPoint. Teks memakai Arial dengan fallback
Liberation Sans pada lingkungan render.

## Inspeksi visual yang benar-benar dilakukan

Seluruh 35 halaman dirender menjadi PNG. Enam contact sheet yang mencakup semua
slide ditinjau. Slide 15, 26, 29, 30 dan 35 diperiksa pada render penuh. Grafik CI
NewHandPD juga ditinjau pada render penuh saat perbaikan kompatibilitas.

Perbaikan yang dilakukan setelah render awal:

- Pendekkan heading panjang agar tidak bertabrakan dengan grafik.
- Pisahkan judul sumbu korelasi dari catatan bawah.
- Tampilkan error bar melalui referensi workbook native.
- Perbaiki aksen nama Šimić.
- Beri empat desimal pada bridge agar batas bawah 0,0004 tidak terbaca sebagai nol.
- Bedakan grafik matching 0–100% dan gate stabil 0–20%.
- Tampilkan dua support pembanding magnitude, termasuk hasil median yang berbeda.

Pemeriksaan koordinat menemukan tidak ada textbox/shape di luar kanvas.
Pemeriksaan PDF menemukan nol span teks di luar halaman. Grafik scatter cadangan
mempertahankan sumbu JSD penuh dan memakai aset empiris tersimpan. Pemeriksaan
geometri tidak menggantikan inspeksi visual.

## Lingkup ilmiah

Audit klaim membedakan keluaran, reproduktibilitas peta, relevansi intervensi,
kalibrasi dan validitas fisiologis. Setiap bank diberi label. Delapan p primer
memakai Holm; CI tetap nominal. Status eksploratori dan riwayat spesifikasi lokal
dipertahankan.

Tidak ada training, inference, intervensi, bootstrap, model replay atau analisis
statistik penelitian baru. Seluruh pekerjaan komputasional pada tahap ini adalah
pembuatan dan pemeriksaan dokumen dari hasil tersimpan.
