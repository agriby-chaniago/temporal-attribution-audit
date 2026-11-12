# Presentasi tanpa angka hasil

**Beda Peta, Masih Relevan?**

Versi ini menyampaikan cerita melalui diagram, ringkasan kualitatif, dan contoh
attention nyata. Terdapat 21 slide utama dan 5 cadangan. Tidak ada angka hasil,
CI numerik, p-value, persentase, atau label sumbu numerik pada slide. Tanggal
acara dan nama resmi **BiMamba-2** dipertahankan.

- [PowerPoint yang dapat diedit](presentasi_attention_16nov2026_tanpa_angka.pptx)
- [PDF](presentasi_attention_16nov2026_tanpa_angka.pdf)
- [Catatan pembicara](CATATAN_PEMBICARA.md)
- [Presentasi lengkap dengan angka dan interval](../README.md)

## Cara memilih versi

Gunakan versi ini untuk menjelaskan pertanyaan, rancangan dan takeaway. Gunakan
versi lengkap bila audiens meminta besar efek, interval, denominator, support,
atau konfigurasi. Penyederhanaan tampilan tidak mengubah kesimpulan penelitian.

Ranking transfer tetap melampaui kontrol rata-rata pada support eligible.
Ranking target mempunyai keuntungan rata-rata yang lebih besar. Hubungan JSD
dengan besar efek transfer merupakan hasil sekunder. Perbandingan plasebo dan
sensitivitas dosis tetap dilaporkan sebagai hasil yang bergantung kondisi.
Temuan magnitude yang menguntungkan pembanding tetap ada pada slide cadangan.

## Kurva dan provenance

Contoh pada slide attention berasal dari bobot asli BiGRU di bank discovery.
Pemilihan contoh mengikuti identitas yang sudah dikunci pada presentasi lengkap.
Ia tidak dipilih untuk memaksimalkan JSD atau efek transfer. Sumbu numeriknya
disembunyikan tanpa mengubah nilai atau grid. Kurva berada dalam chart native
PowerPoint yang dapat diedit. Diagram lain bersifat konseptual dan tidak
menggambarkan data empiris buatan.

`../sources/` dan `../notes/CLAIM_AUDIT.md` memuat sumber, identitas, angka, batas
interpretasi, dan rujukan lengkap. `review/CHART_BINDINGS.json` menghubungkan
kurva dengan bobot asli. File JSON dan catatan boleh memuat angka metadata;
ketentuan tanpa angka berlaku pada tampilan hasil presentasi.

## Membangun ulang

Jalankan dari root repository setelah dependensi presentasi pada folder induk
terpasang. Builder tidak membaca checkpoint atau menjalankan penelitian.

```bash
rtk proxy untuk_presentasi_16nov26/.tools-venv/bin/python untuk_presentasi_16nov26/tanpa_angka/tools/build_and_check.py --build
```

Sumber yang dapat diedit adalah `build_presentation.cjs`. Seluruh speaker notes
juga tertanam dalam PPTX. Pemeriksaan numeralia, identitas kurva, struktur
dokumen, dan geometri tersimpan dalam `review/DOCUMENT_CHECKS.json`. Lingkup
inspeksi visual dan validasi XML dicatat terpisah pada `review/VALIDATION_REPORT.md`.

## Judul paper

Revisi judul manuskrip disimpan secara lokal pada
`Q1-amin/manuscript_array_latex_v10/`. Judul Inggrisnya adalah
*Different Attention Maps, Transferable Relevance? A Fixed-Target Audit of
Parkinsonian Handwriting*. Judul Indonesia, running title dan metadata PDF
diselaraskan. V9 serta presentasi kuantitatif tetap dipertahankan.
