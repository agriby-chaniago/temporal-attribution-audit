# Keputusan dan batas tahap implementasi

Tanggal: 2026-10-16. Dasar: pengguna mengotorisasi melanjutkan implementasi setelah
diskusi ranking-swap, plasebo dan diagnostik. Belum ada keluaran swap yang diperiksa.

## Dipakai untuk tahap pengembangan sekarang

- Fokus utama tetap audit reproduktibilitas peta dan relevansi ranking terhadap
  intervensi. Tidak ada perubahan menjadi penelitian validitas fisiologis.
- Dua arsitektur, tiga seed, satu bank NewHandPD yang terautentikasi; checkpoint
  tetap, tidak ada training. Seluruh enam pasangan terarah digunakan.
- Endpoint primer swap: rata-rata D=G_own-G_swap; G_swap adalah pendamping utama.
  JSD–|D| dan JSD–D adalah analisis sekunder terencana. Ini memperbarui usulan
  sebelumnya sebelum hasil swap diketahui; bukan mengganti hasil primer lama.
- Circular-shift dan permutation reference mempertahankan nilai bobot attention.
  Kandidat dipilih dengan JSD, tanpa respons masked model. Caliper dan cap tetap.
- Empat analisis disiapkan: ranking-swap beserta plasebo; referensi JSD dan
  konsentrasi; sensitivitas output-change berbobot; kontras replacement berpasangan.
- CI subjek bersyarat dan sensitivitas kontrol dilaporkan terpisah. Tiga seed
  tidak diperlakukan sebagai banyak replikasi hanya karena ada banyak pasangan.
- Notebook dibangun oleh agen melalui `.py`; komputasi ilmiah dijalankan pengguna.
- Semua analisis tambahan tetap eksploratori. Naskah V5 dan artefak lama utuh.

## Yang belum dikunci, dan tidak menghambat tahap sekarang

1. **N replikasi:** skenario 10/20/30, dipilih berdasarkan presisi yang dituju dan
   biaya terukur; tidak berdasarkan apakah efek baru signifikan. Tiga seed lama
   tidak cukup untuk mengklaim estimasi komponen variansi seed yang presisi.
2. **Total versus baru:** 30 total NewHandPD berarti 27 seed baru; 30 seed baru
   berarti 30 seed untuk replikasi di samping bank pengembangan. Keduanya berbeda.
3. **Task UCI:** SST adalah kandidat berdasarkan kesamaan prosedur template spiral,
   bukan karena hasil statistik paling menguntungkan. Eligibility dan kanal
   harus diautentikasi ulang. UCI adalah replikasi prosedur setelah retraining,
   bukan transfer satu model beku dan bukan sumber sensor yang identik.
4. **Jadwal swap bank besar:** kandidat reciprocal ring teracak dan dikunci dengan
   dua sumber per target, dalam setiap arsitektur/bank. Jumlah seed final dan
   seed penjadwalan belum ditetapkan. Biaya swap tidak hanya menggabungkan CSV.
5. **Inferensi replikasi:** keluarga primer lintas dataset/strata, metode interval
   variasi seed dan margin perbedaan penting memerlukan protokol tersendiri.
   Leave-one-seed-out sekarang adalah diagnostik pengaruh, bukan pengganti CI seed.
6. **Registrasi publik:** draf dapat disiapkan; tidak ada unggahan OSF/AsPredicted
   yang diotorisasi atau dilakukan. Hasil dan data awal yang diketahui diungkapkan.
7. **Waktu submit:** rekomendasi kerja adalah menunggu replikasi seed dan UCI
   sebelum finalisasi manuskrip penguatan; bukan klaim bahwa V5 terbatas tidak
   boleh disubmit. Keputusan akhir tetap pada pengguna.

BiMamba-3, atribusi tambahan dan perturbasi input tetap perluasan terpisah.
Tidak ada persyaratan bahwa attention harus menang di semua kondisi. Kriteria
yang dinilai adalah pertanyaan terjawab oleh desain, cakupan dan ketelitian
terukur, serta hasil per strata dapat ditafsirkan tanpa menyembunyikan arah lain.

## Koreksi atas komentar yang diadopsi

- D adalah selisih dua keuntungan, masing-masing terhadap kontrol mask sendiri.
  Karena dosis/geometri own dan swap dapat berbeda, raw T dan RMS tetap ditampilkan.
- JSD=0 dengan tie handling yang sama memberi mask sama. JSD mendekati nol tidak
  menjamin ranking/top-k identik di dekat tie; tidak ada asosiasi positif otomatis.
- D bertanda menunjukkan arah, |D| menunjukkan besar perubahan. Tidak satu pun
  otomatis menunjukkan penurunan kualitas atau hubungan kausal dengan JSD.
- Perbedaan signifikansi median dan donor bukan uji perbedaan efek. Kontras baru
  memakai irisan pasangan dan draw subjek yang sama.
- Reweighting kontrol tersimpan bukan simulasi lengkap pencarian kandidat baru;
  dependensi yang tidak dapat direkonstruksi dinyatakan secara eksplisit.
