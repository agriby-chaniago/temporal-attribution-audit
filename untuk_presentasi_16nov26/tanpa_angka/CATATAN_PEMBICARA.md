# Beda Peta, Masih Relevan?

Versi tanpa angka hasil. Tanggal acara dan nama resmi model dipertahankan.

## 1. Beda Peta, Masih Relevan?

Pertanyaannya adalah apakah ranking attention dari seed lain masih memilih bagian input yang berpengaruh pada model target. Relevansi di sini berarti respons terhadap intervensi yang ditentukan. Studi ini menemukan transfer parsial pada pasangan dengan kontrol lengkap. Perbandingan berlangsung dalam arsitektur dan dataset yang sama.

Sumber: `../sources/primary_eight_strata.csv; manuscript V10 Introduction and Results`

## 2. AUROC tinggi belum memvalidasi attention

AUROC menilai pemisahan kelompok oleh prediksi. Kesamaan peta menilai reproduktibilitas attention. Relevansi ranking menilai respons saat bagian input terpilih diganti. Ketiganya adalah objek evaluasi berbeda. Studi ini tidak mengestimasi hubungan universal antara AUROC dan mutu penjelasan.

Sumber: `Theissler et al.; Turbé et al.; ../sources/references.json`

## 3. Keluaran antar-seed tidak selalu serupa

Ringkasan seluruh pasangan pada bank discovery menunjukkan variasi probabilitas dan logit. Karena itu, contoh keluaran serupa tidak dapat mewakili seluruh bank. Analisis kontinu dipertahankan sebagai analisis utama. Gate keluaran serupa hanya menjadi sensitivitas yang mencakup bagian terbatas dari pasangan.

Sumber: `../sources/discovery_e1_ci.csv; ../sources/discovery_e1_sensitivity.csv`

## 4. Keluaran serupa dapat disertai peta berbeda

Kurva ini berasal dari pasangan BiGRU pada rekaman discovery yang sama. Keluaran memenuhi gate ketat, tetapi distribusi attention berbeda. Contoh dipilih menurut urutan identitas di antara pasangan eligible, tanpa mengurutkan JSD atau keuntungan ranking. Identitas dan bobot asli disimpan dalam metadata. Grafik menampilkan bobot asli dengan label angka disembunyikan.

Sumber: `../sources/attention_examples.json; ../sources/discovery_e1_sensitivity.csv`

## 5. Apakah peta lain masih memilih bukti yang berguna?

Studi terdekat mengevaluasi dimensi penjelasan temporal, operator perturbasi, dan regularisasi penjelasan. Penelitian ini menguji transfer urutan pemilihan input pada target yang dibekukan. Kekhususannya terletak pada pertanyaan yang dapat dinilai dan kontrol alternatifnya. Pembandingan terbatas ini bukan klaim prioritas global.

Sumber: `../notes/LITERATURE_COMPARISON.md; Bragança and Souto; Šimić et al.; Navakauskas and Dumpis`

## 6. Prosedur audit diuji pada akuisisi spiral berbeda

NewHandPD dan UCI-SST mempunyai instrumen dan representasi kanal berbeda. Model dilatih terpisah pada setiap dataset. UCI-SST adalah tugas spiral statis dan tidak digabung dengan tugas dinamis atau stabilitas. Transfer ranking tetap berlangsung dalam dataset dan arsitektur yang sama.

Sumber: `../sources/newhandpd_record_manifest.csv; ../sources/uci395_record_manifest.csv; ../sources/PROVENANCE.json`

## 7. Bank baru memperluas bukti discovery

Bank historis memberi konteks performa. Bank discovery mendasari audit awal dan contoh attention. Bank perluasan melaporkan evaluasi transfer pada seed baru. Hasil historis dan discovery tidak dipresentasikan seolah berasal dari bank perluasan. Tambahan seed meningkatkan cakupan variasi training, bukan jumlah subjek.

Sumber: `../sources/evidence.json: design and cohort manifests`

## 8. Probabilitas dan attention berasal dari classifier

Sinyal diproses pada fold training, dibagi menjadi patch, lalu diteruskan ke encoder. Attention menimbang representasi temporal untuk menghasilkan prediksi. Encoder dapat menyebarkan informasi antarwaktu. Karena itu, bobot attention bukan anotasi lokasi gejala yang disupervisi.

Sumber: `V10 Methods; ../sources/evidence.json: design`

## 9. Target tetap ketika ranking sumber berubah

Pada rekaman yang sama, target diuji memakai ranking miliknya sendiri dan ranking dari seed lain. Arsitektur, dataset, fold, serta grid temporal harus cocok. Ranking menentukan patch input yang diganti. Bobot attention head tidak ditimpa. Forward target menghitung ulang responsnya sendiri.

Sumber: `../sources/evidence.json: design; V10 fixed-target Methods`

## 10. Intervensi kuat membutuhkan kontrol yang sepadan

Setiap mask attention mempunyai kontrol acak yang dicocokkan sendiri. Jumlah patch, geometri run, dan besar perubahan input diukur dengan RMS. Penggantian menggunakan median train atau donor dari train fold. Pencarian kontrol tidak menggunakan respons model. Mask target dan mask transfer masih dapat berbeda dosis secara langsung.

Sumber: `V10 Methods; ../sources/evidence.json: design and matching coverage`

## 11. Keuntungan ranking dinilai terhadap kontrolnya

Respons intervensi diringkas dari besar perubahan absolut logit di sepanjang budget. Keuntungan ranking adalah respons itu dikurangi respons kontrol RMS-matched. Efek transfer membandingkan keuntungan ranking target dengan keuntungan ranking sumber. Respons ini tidak mengukur validitas gejala dan tidak langsung membentuk leaderboard lintas arsitektur.

Sumber: `V10 AOPC, G and D definitions`

## 12. Relevansi ranking berpindah sebagian pada NewHandPD

Pada rerata pasangan eligible, ranking dari seed lain melampaui kontrol RMS-matched. Ranking target mempunyai keuntungan yang lebih besar. Pola ini terdapat pada kedua arsitektur dan kedua replacement. Interval subjek bersyarat mendukung arah kontras tersebut. Kesimpulan merujuk pada support matching, bukan seluruh kemungkinan pasangan.

Sumber: `../sources/primary_eight_strata.csv; ../sources/newhandpd_secondary_mean_contrasts.csv`

## 13. Relevansi ranking berpindah sebagian pada UCI-SST

Pada rerata pasangan eligible, ranking dari seed lain melampaui kontrol RMS-matched. Ranking target mempunyai keuntungan yang lebih besar. Pola ini terdapat pada kedua arsitektur dan kedua replacement. Interval subjek bersyarat mendukung arah kontras tersebut. Kesimpulan merujuk pada support matching, bukan seluruh kemungkinan pasangan.

Sumber: `../sources/primary_eight_strata.csv; ../sources/uci395_secondary_mean_contrasts.csv`

## 14. Matching menentukan cakupan kesimpulan

Sebagian pasangan tidak memperoleh kontrol lengkap dalam batas pencarian. Cakupan berbeda menurut dataset, arsitektur, replacement, dan perbandingan placebo. Kehadiran banyak subjek tidak berarti semua rekaman atau pasangan masuk. Karakteristik eligible dan excluded dapat berbeda. Batas ini berkaitan dengan populasi yang diwakili hasil.

Sumber: `../sources/newhandpd_matching_coverage.csv; ../sources/uci395_matching_coverage.csv; eligible_excluded_characteristics`

## 15. Peta lebih berbeda berkaitan dengan efek transfer

JSD mengukur perbedaan distribusi attention. Pada analisis target tetap, weighted Spearman menunjukkan asosiasi positif dengan besar selisih keuntungan transfer pada seluruh kondisi utama. Asosiasi ini merupakan hasil sekunder dengan interval nominal. Sebagian kaitannya dapat muncul dari konstruksi mask. Konsentrasi JSD tinggi membatasi penafsiran pada peta yang sangat mirip.

Sumber: `../sources/newhandpd_jsd_associations.csv; ../sources/uci395_jsd_associations.csv`

## 16. Ranking nyata tidak selalu melampaui plasebo

Plasebo diperoleh melalui pergeseran atau permutasi bobot attention. Kandidat dipilih agar JSD mendekati pasangan nyata, tanpa melihat respons target. Ia mendapat kontrol sepadannya sendiri. Keunggulan ranking nyata muncul pada sebagian kondisi. Kondisi lain mempunyai interval yang mencakup nol. Cakupan matching juga berkurang.

Sumber: `../sources/newhandpd_placebo_contrasts.csv; ../sources/uci395_placebo_contrasts.csv`

## 17. Dosis langsung masih perlu diperhatikan

Kontrol sepadan untuk setiap mask tidak otomatis menyamakan dosis ranking target dan transfer. Sensitivitas membatasi pasangan pada dosis yang lebih serupa. Subset ini menyempit. Pada donor UCI, interval kontras mentah mencakup nol pada kedua arsitektur. Temuan tersebut membatasi perluasan klaim dari hasil rata-rata utama.

Sumber: `../sources/newhandpd_secondary_mean_contrasts.csv; ../sources/uci395_secondary_mean_contrasts.csv: dose scenarios`

## 18. Rerata yang bertahan tetap menyisakan variasi

Leave-one-seed-out mempertahankan arah positif rerata efek transfer. Sebagian target BiGRU tetap mempunyai efek yang berlawanan. Ini menunjukkan variasi antar-training di balik rerata. Interval subjek bersyarat tidak mencakup seluruh variasi populasi seed atau pembagian fold baru. Pasangan berbagi model dan subjek.

Sumber: `../sources/newhandpd_leave_one_seed_out.csv; ../sources/uci395_leave_one_seed_out.csv; target_seed_effects`

## 19. Audit membantu menilai penjelasan secara bertahap

Pengguna model dapat memisahkan pemeriksaan prediksi, reproduktibilitas peta, dan respons intervensi. Kontrol amplitudo dan geometri membantu menguji alternatif penjelasan berupa perubahan input yang lebih besar. Transfer parsial memberi alasan untuk menguji ranking lain pada target yang sama saat debugging. Evaluasi klinis tetap memerlukan referensi tersendiri.

Sumber: `V10 Discussion and Conclusion`

## 20. Bukti mempunyai cakupan yang terdefinisi

Audit mencakup spiral, encoder yang dipilih, pembagian subjek tetap, dan replacement yang diuji. Beberapa analisis pendamping berasal dari bank historis atau discovery. Rencana tidak didaftarkan publik. Analisis tambahan dan sensitivitas tetap eksploratori. Replikasi split, kohort lebih besar, atribusi lain, dan rujukan fisiologis menjawab pertanyaan berikutnya.

Sumber: `V10 Discussion; ../notes/CLAIM_AUDIT.md`

## 21. Relevansi berpindah, tetapi hanya sebagian

Ranking dari seed lain mempertahankan keuntungan rata-rata terhadap kontrol pada pasangan eligible. Ranking target lebih unggul rata-rata. Besar perbedaan peta berkaitan dengan besar efek transfer, sedangkan pembanding plasebo memberi hasil yang bergantung kondisi. Kesimpulan ini menyangkut intervensi yang diuji, bukan validitas fisiologis atau mutu penjelasan universal.

Sumber: `../sources/primary_eight_strata.csv; secondary_mean_contrasts; jsd_associations; placebo_contrasts`

## 22. Magnitude juga dapat melampaui attention

Pembanding magnitude menggunakan besar perubahan input untuk mengurutkan patch. Pada discovery, BiGRU lebih menguntungkan attention pada donor dan median. BiMamba lebih menguntungkan magnitude pada donor. Pada median BiMamba, dukungan interval bergantung support penuh atau irisan E2 lengkap. Magnitude belum mendapat dose dan geometri yang disamakan. Ia bukan bukti arsitektur paling faithful.

Sumber: `../sources/discovery_amplitude_ci.csv; V10 magnitude comparator tables`

## 23. Hasil pendamping memberi konteks yang berbeda

Bank historis menunjukkan penurunan AUROC BiMamba yang lebih kecil secara deskriptif setelah retraining terpisah. Interval kontras masih mencakup nol. Bank discovery mendukung Brier lebih rendah untuk BiMamba, sedangkan ECE utama masih tidak pasti. Brier mengukur kualitas probabilitas secara keseluruhan. Kontrol usia dan baseline klasik historis mempunyai batas protokolnya sendiri.

Sumber: `../sources/historical_auroc.csv; historical_auroc_contrast.csv; calibration_contrasts.csv; ../notes/CLAIM_AUDIT.md`

## 24. Istilah yang perlu dibedakan

Reproduktibilitas peta adalah kesamaan attention antar-training. Relevansi intervensi adalah respons model terhadap penggantian input yang ditentukan. Validitas fisiologis membutuhkan rujukan gejala yang tepat. Kalibrasi adalah hubungan probabilitas dengan frekuensi kejadian. Selang yang mencakup nol bukan bukti ekuivalensi atau hubungan nol.

Sumber: `V10 Methods and Discussion; ../sources/references.json`

## 25. Penelitian terdekat memberi pembanding protokol

Daftar ini merupakan pembanding metodologis, bukan leaderboard performa handwriting. Judul, jurnal, tahun dan DOI lengkap terdapat pada sumber presentasi dengan angka. Tidak ada klaim bahwa ketiganya sepenuhnya menyamakan protokol penelitian ini.

Sumber: `../sources/references.json; ../notes/LITERATURE_COMPARISON.md`

## 26. Rujukan dan angka lengkap tetap dapat diperiksa

Presentasi ini adalah pendamping kualitatif. Seluruh angka, confidence interval, denominator, definisi gate, dan rujukan lengkap tetap tersedia dalam versi kuantitatif pada folder induk. Kurva contoh menggunakan bobot attention asli. Diagram lainnya bersifat konseptual dan tidak menyimulasikan data penelitian. Tidak ada eksperimen atau estimator yang dijalankan.

Sumber: `../README.md; ../CATATAN_PEMBICARA.md; ../sources/PROVENANCE.json; Q1-amin/manuscript_array_latex_v10/README.md`
