# Pembanding audit dan verifikasi sumber

Pemeriksaan ulang dilakukan pada 12 November 2026. Ketiga pembanding adalah
artikel jurnal. Mereka dipilih karena objek evaluasinya berdekatan, bukan karena
mempunyai protokol handwriting yang identik. Tidak ada leaderboard numerik
lintas protokol dalam presentasi.

| Studi jurnal | Objek yang sudah dievaluasi | Perbandingan yang ditawarkan presentasi ini | Batas kesepadanan |
|---|---|---|---|
| Bragança & Souto, *Unified time series classification framework for explainable artificial intelligence*, Scientific Reports, 2026. DOI [10.1038/s41598-026-49467-2](https://doi.org/10.1038/s41598-026-49467-2) | UTS-XAI menggabungkan evaluasi beberapa properti penjelasan temporal. | Ranking source dipakai pada rekaman yang sama dengan checkpoint target tetap. | Dataset dan tugas berbeda. Bukan pembanding AUROC langsung pada UCI-SST atau NewHandPD. |
| Šimić, Veas & Sabol, *A comprehensive analysis of perturbation methods in explainable AI feature attribution validation for neural time series classifiers*, Scientific Reports, 2025. DOI [10.1038/s41598-025-09538-2](https://doi.org/10.1038/s41598-025-09538-2) | Operator perturbasi, ukuran region, dan ukuran evaluasi atribusi. | Setiap mask own/transfer mempunyai kontrol RMS dan geometri run sendiri. | Metrik dan tujuan pengujian tidak sama. AOPC perubahan absolut logit studi ini tidak disamakan dengan metrik paper tersebut. |
| Navakauskas & Dumpis, *Regularizing Temporal Explanations in Dynamic Neural Networks*, Electronics, 2026. DOI [10.3390/electronics15102200](https://doi.org/10.3390/electronics15102200) | Regularisasi penjelasan temporal dan pengujian pada model dinamis. | Hubungan divergence dan relevansi transfer dipisahkan dari perubahan fungsi target. | Model, dataset, dan intervensi berbeda. Tidak dipakai untuk klaim bahwa satu arsitektur handwriting adalah SOTA. |

## Sumber yang benar-benar diperiksa

- Metadata DOI dan jenis publikasi dicocokkan dengan rekaman Crossref yang
  sudah diverifikasi pada V9. Identitas tersebut tersedia dalam
  `sources/references.json` dan bibliografi V9 yang dipertahankan.
- Halaman Nature langsung untuk dua artikel mengalami pengalihan akses pada
  pemeriksaan ini. Hasil terindeks dari halaman primer Šimić menyediakan teks
  metode. Salinan artikel Bragança tersedia pada
  [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369486/).
- Halaman MDPI langsung memberi respons rate-limit. Metadata penerbit dan
  abstrak yang disimpan dalam verifikasi DOI V9 mendukung deskripsi terbatas
  tentang regularisasi. Pemeriksaan ini tidak diklaim sebagai pembacaan ulang
  seluruh suplemen MDPI.
- [Silva Filho et al. (2023)](https://doi.org/10.1007/s10994-023-06336-7)
  diperiksa melalui halaman penerbit. Rujukan ini mendukung pembedaan kalibrasi
  dan proper scoring rules dalam slide probabilitas.

Tidak ada klaim negatif bahwa pembanding “tidak pernah” melakukan setiap langkah
tertentu. Tabel hanya membandingkan objek dan protokol yang diperiksa. Temuan
studi sendiri tetap berasal dari CSV tersimpan, bukan dari literatur eksternal.

## Atribusi dan usia rujukan

Presentasi memakai 15 artikel jurnal terpilih dari tahun 2021 ke atas. Jumlah
tersebut adalah pustaka presentasi, bukan pengganti bibliografi manuskrip.
Bibliografi V9 dengan 53 artikel jurnal dan delapan IJRCS tidak diubah.
Prosiding dan preprint tidak disamarkan sebagai jurnal. Rujukan akuisisi dataset
asli dicatat sebagai atribusi data pada `README.md`, dengan jenis publikasinya.
