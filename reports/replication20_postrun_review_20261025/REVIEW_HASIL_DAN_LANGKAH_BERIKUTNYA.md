# Audit hasil replikasi 20 seed dan dua dataset

Paket komputasi utama 22–24 sudah selesai. Laporan ini membaca artefak tersimpan dan memeriksa integritasnya; tidak menjalankan training, inference, intervensi, bootstrap, atau estimator ilmiah baru. Manuskrip V6 dan seluruh hasil asli tetap utuh.

## Jawaban utama

Ranking attention dapat memindahkan sebagian relevansi intervensinya ke model lain dari arsitektur dan kohort yang sama. Pada model target yang dibekukan, ranking milik target mempunyai keuntungan rata-rata lebih besar daripada ranking dari seed lain pada kedelapan strata. Ranking pinjaman tetap mengungguli kontrol RMS/geometri pada estimasi sekunder rata-rata. Keunggulan terhadap placebo dan cakupan matching bergantung kondisi.

Cerita yang paling kuat: **relevansi ranking bersifat sebagian transferable dan sebagian spesifik terhadap model**. Ukuran perbedaan peta memberi informasi tambahan, tetapi tidak menjadi ukuran kualitas penjelasan yang berdiri sendiri.

## Pemeriksaan integritas yang benar-benar dilakukan

- 3676 pemeriksaan; 0 gagal.
- 400 file checkpoint dicocokkan SHA-256-nya dengan inventory dan metadata.
- Kedua bank: 20 seed × 5 fold × 2 arsitektur = 200 checkpoint per dataset.
- NewHandPD: 264 rekaman, 66 subjek; UCI 395 SST: 77 rekaman, 76 subjek.
- Seluruh 42 CSV yang dideklarasikan manifest diperiksa SHA-256-nya, termasuk tabel besar secara streaming.
- Identitas OOF, pemisahan subjek train/test pada metadata setiap checkpoint, hash source notebook, source eksekusi dan protokol diperiksa.
- 1195 berkas profil pemulihan numerik diperiksa digest, binding dan hash source kernelnya; semuanya pada NewHandPD. Ini pemeriksaan artefak, bukan replay model baru.
- Tabel dua run notebook 24 mempunyai checksum yang sama; laporan memakai run gabungan terakhir.
- Isi seluruh cache intervensi tidak direplay ulang. Pemeriksaan ini tidak mengestimasi ulang statistik atau membuktikan ulang hasil semua forward.

## Bank dan status analisis

NewHandPD mempunyai bank 20 seed total; analisis replikasi primer memakai seed baru 3–19 pada kedua anggota pasangan. Tiga seed discovery tetap dipisahkan, dan bank gabungan 20 seed menjadi sensitivitas. UCI-SST memakai seluruh seed 0–19. Semua model dilatih terpisah per dataset; ranking transfer berlangsung antarseed dalam arsitektur/kohort/fold yang sama.

Spesifikasi perluasan dikunci secara lokal sebelum run perluasan. Manifest menyatakan `public_preregistration=false`; hasil discovery tetap mengikuti status eksploratorinya. Label ini tidak diubah menjadi preregistrasi publik.

## Endpoint primer dan hasil

`T` adalah AOPC trapezoidal respons absolut logit terhadap intervensi ranking. `R` adalah AOPC kontrol acak RMS/geometri yang dicocokkan untuk ranking tersebut. `G=T−R`, dan `D=G_own−G_swap`. Positifnya D menunjukkan keuntungan relatif untuk ranking sendiri pada model target yang sama. Own dan swap masing-masing mempunyai kontrol terpadan sendiri; dosis dan geometri own-versus-swap tidak otomatis identik.

Nilai di bawah adalah estimasi tersimpan dengan CI subjek nominal 95%. Koreksi Holm diterapkan pada delapan p-value primer, bukan mengubah interval menjadi interval simultan. Raw logit-AOPC tidak digunakan sebagai skala kualitas universal lintas model/dataset.

| Dataset | Model | Masker | D [CI 95% nominal] | p Holm | Pasangan eligible | Subjek |
| --- | --- | --- | --- | --- | --- | --- |
| newhandpd | BiGRU | Donor | 1.213 [0.983, 1.449] | 0.001600 | 17733 | 66 |
| newhandpd | BiGRU | Median | 1.266 [0.898, 1.661] | 0.001600 | 16271 | 66 |
| newhandpd | BiMamba-2 | Donor | 1.407 [1.244, 1.579] | 0.001600 | 12654 | 66 |
| newhandpd | BiMamba-2 | Median | 2.190 [1.907, 2.493] | 0.001600 | 8791 | 66 |
| uci395 | BiGRU | Donor | 0.048 [0.035, 0.061] | 0.001600 | 5479 | 74 |
| uci395 | BiGRU | Median | 0.101 [0.064, 0.143] | 0.001600 | 5102 | 67 |
| uci395 | BiMamba-2 | Donor | 0.224 [0.190, 0.260] | 0.001600 | 6000 | 76 |
| uci395 | BiMamba-2 | Median | 0.568 [0.415, 0.713] | 0.001600 | 5280 | 75 |

Delapan estimasi D dan seluruh batas bawah CI bernilai positif. Delapan p-value Holm tersimpan sekitar 0,001600. Nilai p mentah mencapai resolusi bawah bootstrap 5.000 draw; jangan dilaporkan sebagai nol atau dibaca sebagai ukuran besar efek.

## Relevansi ranking pinjaman

| Dataset | Model | Masker | G_swap [CI nominal] |
| --- | --- | --- | --- |
| newhandpd | BiGRU | Donor | 0.698 [0.628, 0.768] |
| newhandpd | BiGRU | Median | 0.732 [0.624, 0.837] |
| newhandpd | BiMamba-2 | Donor | 0.419 [0.317, 0.524] |
| newhandpd | BiMamba-2 | Median | 0.312 [0.194, 0.434] |
| uci395 | BiGRU | Donor | 0.287 [0.221, 0.362] |
| uci395 | BiGRU | Median | 0.616 [0.447, 0.805] |
| uci395 | BiMamba-2 | Donor | 0.189 [0.143, 0.238] |
| uci395 | BiMamba-2 | Median | 0.559 [0.431, 0.698] |

Semua delapan G_swap mempunyai CI nominal positif. Ini hasil sekunder tanpa perluasan otomatis koreksi delapan pengujian primer. Transfer bukan identitas ranking, bukan ranking optimal, dan bukan validitas fisiologis.

## Hubungan perbedaan peta dengan efek swap

JSD adalah Jensen–Shannon divergence base-2 sesuai implementasi. Pada target tetap, D bertanda dan |D| menjawab pertanyaan berbeda. Tabel ini menyalin asosiasi sekunder weighted Spearman dengan |D|; seluruh CI nominal positif, tetapi korelasi bukan identifikasi kausal. Ketika dua ranking identik, mask dan respons ter-reuse identik, sehingga ada kaitan konstruksi menuju D=0; placebo tetap penting.

| Dataset | Model | Masker | Spearman JSD–|D| [CI nominal] |
| --- | --- | --- | --- |
| newhandpd | BiGRU | Donor | 0.217 [0.184, 0.249] |
| newhandpd | BiGRU | Median | 0.153 [0.117, 0.189] |
| newhandpd | BiMamba-2 | Donor | 0.146 [0.094, 0.193] |
| newhandpd | BiMamba-2 | Median | 0.190 [0.133, 0.245] |
| uci395 | BiGRU | Donor | 0.192 [0.118, 0.268] |
| uci395 | BiGRU | Median | 0.275 [0.190, 0.356] |
| uci395 | BiMamba-2 | Donor | 0.255 [0.192, 0.313] |
| uci395 | BiMamba-2 | Median | 0.384 [0.285, 0.470] |

Analisis ini menggunakan fungsi target tetap. Jangan menggantikan atau mencampurnya dengan bridge V5 yang mengubah fungsi model dan attention sekaligus, atau dengan Pearson tersentrik pada estimand historis lain. Pernyataan lama tentang hubungan bridge yang belum konsisten tetap merujuk pada analisis historisnya; hasil swap baru mempunyai objek berbeda.

## Coverage dan subset stabil

Persentase di tabel hanya rasio numerator/denominator tersimpan, bukan estimator efek baru. Pasangan adalah arah target–sumber terjadwal; bukan semua pasangan seed yang mungkin, bukan observasi independen dan bukan subjek tambahan.

| Dataset | Model | Masker | Eligible/planned | Subjek eligible/planned | Gate stabil sebelum matching | Gate stabil + matching | Dose-gap eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| newhandpd | BiGRU | Donor | 17733/18480 (96.0%) | 66/66 | 528/18480 (2.86%) | 504 | 1441 |
| newhandpd | BiGRU | Median | 16271/18480 (88.0%) | 66/66 | 528/18480 (2.86%) | 447 | 1952 |
| newhandpd | BiMamba-2 | Donor | 12654/20064 (63.1%) | 66/66 | 1046/20064 (5.21%) | 591 | 144 |
| newhandpd | BiMamba-2 | Median | 8791/20064 (43.8%) | 66/66 | 1046/20064 (5.21%) | 404 | 212 |
| uci395 | BiGRU | Donor | 5479/7700 (71.2%) | 74/76 | 1404/7700 (18.23%) | 1054 | 139 |
| uci395 | BiGRU | Median | 5102/7700 (66.3%) | 67/76 | 1404/7700 (18.23%) | 1043 | 822 |
| uci395 | BiMamba-2 | Donor | 6000/7700 (77.9%) | 76/76 | 542/7700 (7.04%) | 437 | 71 |
| uci395 | BiMamba-2 | Median | 5280/7700 (68.6%) | 75/76 | 542/7700 (7.04%) | 382 | 514 |

Gate stabil memakai class agreement, |Δp|≤0,02 dan |Δz|≤0,1. Pada NewHandPD, gate terjadwal mencakup sekitar 2,86% pasangan BiGRU dan 5,21% BiMamba-2; pada UCI-SST sekitar 18,23% dan 7,04%. Ini cakupan pada jadwal primer, bukan persentase E1 historis atau seluruh pasangan 20 seed.

Coverage pasangan NewHandPD BiMamba-2/median sekitar 43,8%, walaupun seluruh 66 subjek masih mempunyai setidaknya satu pasangan eligible. Pada UCI-SST BiGRU/median, 67 dari 76 subjek eligible. Perbedaan ini harus terlihat dekat hasil utama.

Diagnostik eligible–excluded yang tersimpan menunjukkan perbedaan pada input-change magnitude: SMD mean requested RMS adalah sekitar +0,399 pada NewHandPD BiMamba-2/median dan −0,558 pada UCI-SST BiGRU/median. Ini alasan membatasi populasi kesimpulan pada support eligible; transparansi coverage tidak menghilangkan selection bias.

## Sensitivitas dan perbedaan yang harus dipertahankan

- Mean D tetap positif dalam seluruh tabel leave-one-seed-out tersimpan. Namun beberapa estimasi target-seed BiGRU negatif: seed 6 (kedua masker) dan seed 8 (median) di NewHandPD; seed 9, 13, 17 dan 19 (median) di UCI-SST. Hindari klaim setiap model atau setiap seed menang.
- D_H memakai H=(T−R)/(T+R) ketika jumlah respons melebihi floor; pada analisis semua support primer, seluruh CI D_H positif. Normalisasi adalah sensitivitas, bukan bukti penyebab variasi skala logit.
- Pada gate output stabil, seluruh delapan D mempunyai CI nominal positif. Gate tetap sensitivitas, bukan landasan klaim bahwa seluruh output stabil.
- Pada own–swap dose gap ≤5%, CI D donor UCI-SST mencakup nol pada kedua arsitektur. Support hanya 139 pasangan/20 subjek (BiGRU) dan 71 pasangan/11 subjek (BiMamba-2). Median UCI dan seluruh strata NewHandPD mempunyai CI D nominal positif pada subset ini, tetapi support subset berubah dan kecil pada beberapa strata.
- Pada common support median–donor, kontras D BiMamba-2 positif pada kedua dataset. BiGRU NewHandPD mencakup nol untuk D, sedangkan D_H negatif; BiGRU UCI mempunyai D positif tetapi D_H mencakup nol. Jangan menyimpulkan replacement memberi satu pola universal dari perbedaan signifikansi terpisah.
- Selisih new-minus-discovery BiGRU di NewHandPD negatif dengan CI yang tidak mencakup nol pada kedua masker; BiMamba-2 donor mencakup nol dan median positif. Ini perbandingan bank/jadwal/support bersyarat, bukan kesetaraan efek atau estimasi perbedaan populasi seed.

## Placebo: ranking lain yang bermakna versus sekadar mask berbeda

Sumber: `placebo_contrasts.csv`, endpoint G_real−G_placebo. Semua interval di bagian ini sekunder dan nominal.

| Dataset | Model | Masker | Placebo | Real−placebo [CI nominal] | Support bersama/planned | Subjek |
| --- | --- | --- | --- | --- | --- | --- |
| newhandpd | BiGRU | Donor | circular_shift | 0.092 [0.018, 0.168] | 11979/18480 | 66 |
| newhandpd | BiGRU | Donor | permutation | 0.200 [0.125, 0.276] | 12684/18480 | 66 |
| newhandpd | BiGRU | Median | circular_shift | 0.129 [0.021, 0.240] | 10842/18480 | 66 |
| newhandpd | BiGRU | Median | permutation | 0.211 [0.103, 0.318] | 11614/18480 | 66 |
| newhandpd | BiMamba-2 | Donor | circular_shift | 0.088 [-0.010, 0.180] | 9277/20064 | 66 |
| newhandpd | BiMamba-2 | Donor | permutation | 0.107 [0.016, 0.198] | 9418/20064 | 66 |
| newhandpd | BiMamba-2 | Median | circular_shift | 0.004 [-0.117, 0.122] | 6376/20064 | 66 |
| newhandpd | BiMamba-2 | Median | permutation | 0.026 [-0.081, 0.127] | 6657/20064 | 66 |
| uci395 | BiGRU | Donor | circular_shift | 0.030 [0.005, 0.059] | 4236/7700 | 74 |
| uci395 | BiGRU | Donor | permutation | 0.064 [0.019, 0.111] | 406/7700 | 44 |
| uci395 | BiGRU | Median | circular_shift | 0.023 [-0.011, 0.053] | 4093/7700 | 65 |
| uci395 | BiGRU | Median | permutation | 0.135 [0.081, 0.194] | 403/7700 | 37 |
| uci395 | BiMamba-2 | Donor | circular_shift | 0.043 [0.023, 0.063] | 3629/7700 | 76 |
| uci395 | BiMamba-2 | Donor | permutation | 0.030 [-0.007, 0.067] | 1270/7700 | 75 |
| uci395 | BiMamba-2 | Median | circular_shift | 0.121 [0.071, 0.177] | 3396/7700 | 74 |
| uci395 | BiMamba-2 | Median | permutation | 0.162 [0.081, 0.248] | 1058/7700 | 71 |

BiGRU NewHandPD mengungguli kedua konstruksi placebo pada CI nominal. Pada BiMamba-2 NewHandPD, circular-shift donor dan kedua placebo median mempunyai interval yang mencakup nol. Pada UCI-SST, circular-shift BiGRU/median dan permutation BiMamba-2/donor mencakup nol. Tidak ada dasar untuk menyamakan interval tersebut dengan equivalence. Matching JSD permutation UCI BiGRU hanya berhasil pada 640/7.700 pasangan sebelum syarat kontrol; hasil permutation-nya mewakili support yang sempit.

## Peta klaim–bukti

Lihat `claim_evidence_limits.csv`. Kontribusi yang dapat diprioritaskan:
1. Audit target tetap yang memisahkan keuntungan ranking sendiri dan pinjaman dari kontrol RMS/geometri pada rekaman yang sama.
2. Replikasi arah keuntungan rata-rata ranking sendiri pada delapan strata, dengan keluarga primer Holm yang dikunci dan pemisahan seed discovery.
3. Bukti transfer parsial beserta batasnya melalui placebo, coverage matching, dose, variasi seed, dan hubungan dengan JSD.
4. Paket provenance eksekusi yang mempertahankan checkpoint, mask tersimpan, hash, serta jejak pemulihan numerik. Ini kontribusi reproduksibilitas komputasional, bukan jaminan klinis.

## Langkah berikutnya untuk naskah

Paket eksperimen utama yang dikunci sudah terpenuhi. Langkah berikutnya adalah copy + modified ke versi manuskrip baru (V7), tanpa menimpa V6 atau hasil:

1. Jadikan target-fixed ranking transfer dan replikasi dua dataset sebagai hasil utama. Judul kerja: **Attention rankings are partly transferable: a two-dataset audit of Parkinsonian handwriting models**.
2. Tampilkan hasil delapan strata dalam visual utama dengan panel/skala yang mempertahankan perbedaan raw-logit antarbank. Letakkan coverage dan placebo dekat hasil primer, lalu seed diagnostics dan dose sensitivity pada bagian yang mudah dilacak.
3. Pertahankan gaya yang disepakati: cerita jelas, abstrak tanpa angka hasil, kontribusi berupa poin, istilah teknis dijelaskan, visual terbaca, dan batas umum cukup dijelaskan sekali pada tempat yang tepat.
4. Tabel performa harus memakai bank yang benar. AUROC historis E0 serta ECE/Brier bank tiga-seed tetap diberi identitas historis; jangan mengatribusikannya ke bank 20 seed. AUROC per seed untuk kedua bank baru tersedia di CSV, tetapi laporan ini tidak mengestimasi kontras performa atau kalibrasi baru.
5. Pertahankan tiga pembanding metodologis dan bibliografi jurnal yang relevan. Tidak perlu mengganti target ke konferensi hanya karena opsi IJCNN telah didiskusikan; format lengkap dapat tetap disiapkan untuk Array, lalu diringkas bila target akhirnya berubah.
6. E3 input perturbation, IG/occlusion, transportabilitas classifier dan validasi fisiologis diperlukan ketika memperluas klaim terkait; tidak menjadi prasyarat otomatis bagi klaim transfer parsial yang sudah dibatasi sekarang.
7. Setelah versi naskah disusun, periksa ulang setiap angka terhadap sumber, konsistensi bank dan denominator, lalu kompilasi serta audit visual. Finalisasi metadata penulis dan arsip publik mengikuti tahap submission, tanpa mengarang informasi yang belum diberikan.

## Lokasi artefak yang harus digunakan

- **newhandpd hasil:** `results/attention_replication20_resume/newhandpd/9f048349d216342f9280825fa3e12dea0fbcb8eeeac3387efccead22f3b540f5/5b4141580bb4d3e8c061bd977968b4c2b8b02576195373a7e75db771f89ab53b/0082ecd5671e689ea2325bb0d45ed2ecfdffd7622ebc487e2edbbc2b0def0726`
- **newhandpd gambar terbaru:** `figures/attention_replication20_fast/newhandpd/9f048349d216342f9280825fa3e12dea0fbcb8eeeac3387efccead22f3b540f5/5b4141580bb4d3e8c061bd977968b4c2b8b02576195373a7e75db771f89ab53b/0082ecd5671e689ea2325bb0d45ed2ecfdffd7622ebc487e2edbbc2b0def0726`
- **uci395 hasil:** `results/attention_replication20_resume/uci395/9f048349d216342f9280825fa3e12dea0fbcb8eeeac3387efccead22f3b540f5/5b4141580bb4d3e8c061bd977968b4c2b8b02576195373a7e75db771f89ab53b/0082ecd5671e689ea2325bb0d45ed2ecfdffd7622ebc487e2edbbc2b0def0726`
- **uci395 gambar terbaru:** `figures/attention_replication20_fast/uci395/9f048349d216342f9280825fa3e12dea0fbcb8eeeac3387efccead22f3b540f5/5b4141580bb4d3e8c061bd977968b4c2b8b02576195373a7e75db771f89ab53b/0082ecd5671e689ea2325bb0d45ed2ecfdffd7622ebc487e2edbbc2b0def0726`
- **Gabungan terakhir:** `results/attention_replication20_resume/combined/9f048349d216342f9280825fa3e12dea0fbcb8eeeac3387efccead22f3b540f5/0082ecd5671e689ea2325bb0d45ed2ecfdffd7622ebc487e2edbbc2b0def0726/20261025T073118047187Z/primary_eight_strata_holm.csv`

Perhatian provenance: gambar hasil resume masih berada di direktori bernama `attention_replication20_fast`, mengikuti path visual builder induk, tetapi memakai execution ID resume `0082ecd…`. Jangan memilih gambar hanya berdasarkan kata FAST; cocokkan execution ID. Berkas tidak dipindah atau diubah pada audit ini.

## Metode penyusunan catatan

Audit dibantu skill lokal `scientific-critical-thinking` untuk memisahkan observasi, interpretasi, status inferensi dan batas klaim. Rujukan perangkat bantu: T. Kassis, V. Agarwal, Y. He, D. Patel, A. M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, preprint, https://doi.org/10.48550/arXiv.2609.00065. Metadata diverifikasi melalui halaman arXiv pada 25 Oktober 2026. Ini atribusi perangkat bantu dalam catatan, bukan artikel jurnal pembanding dan tidak ditambahkan ke bibliografi manuskrip.
