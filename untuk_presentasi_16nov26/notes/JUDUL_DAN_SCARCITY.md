# Judul dan alasan pemilihan

Judul yang diterapkan pada presentasi adalah **Beda Peta, Masih Relevan?**

Subjudulnya adalah *Audit transfer ranking attention pada tulisan tangan
Parkinsonian*. Judul ini hanya berlaku untuk presentasi baru. Judul dan isi
manuskrip V9 tidak diubah.

| Alternatif | Daya tarik dan batas bukti |
|---|---|
| **Beda Peta, Masih Relevan?** | Menghadirkan pertanyaan yang langsung dijawab oleh transfer ranking. Kata relevan didefinisikan melalui intervensi pada target tetap. |
| Ketika Ranking Attention Berpindah Model | Menonjolkan tindakan eksperimen. Model sumber dan target tetap berasal dari arsitektur dan dataset yang sama. |
| Seberapa Jauh Relevansi Attention Bisa Berpindah? | Mengarahkan perhatian ke besarnya transfer yang diukur. Tidak mengklaim skala universal lintas model. |
| Prediksi Tinggi Belum Memvalidasi Peta | Menjelaskan motivasi audit. Studi tidak mengestimasi korelasi universal AUROC–kualitas penjelasan. |
| Relevansi Parsial di Balik Peta yang Berbeda | Menyampaikan temuan utama secara langsung. Temuan merupakan rerata pada support yang eligible. |
| Saat Seed Berganti, Apa yang Tetap Relevan? | Mengaitkan reproduktibilitas dengan kegunaan ranking. Tambahan seed tidak dianggap tambahan peserta. |

## Nilai yang ditonjolkan

Peta yang tampak berbeda menimbulkan pertanyaan yang tidak dijawab oleh AUROC.
Ranking yang berbeda dapat tetap memilih bagian input yang memengaruhi satu model.
Desain target tetap memberi cara untuk memeriksa kemungkinan tersebut. Kontrol
RMS dan panjang blok menguji penjelasan alternatif berupa intervensi yang lebih
besar. Placebo pada divergence serupa menambah pembanding yang lebih menuntut.

Cerita presentasi mengikuti urutan itu. Keluaran seluruh pasangan ditampilkan
sebelum contoh subset serupa. Transfer kemudian diuji pada dua bank yang
ditraining terpisah. Ranking sumber mempertahankan keuntungan rata-rata. Ranking
target mempunyai keuntungan lebih besar. Pembandingan placebo dan pembatasan
dosis menunjukkan kondisi yang lebih terbatas.

## Landasan pembandingan

- Bragança dan Souto (2026) menyajikan UTS-XAI dengan beberapa dimensi mutu
  penjelasan temporal. Studi ini memberi konteks bahwa kemiripan peta bukan
  satu-satunya objek evaluasi.
- Šimić dan kolega (2025) mengevaluasi pengaruh operator perturbasi dan region.
  Studi ini mendasari perhatian pada cara input diganti.
- Navakauskas dan Dumpis (2026) mengevaluasi regularisasi penjelasan temporal.
  Objek tersebut berbeda dari transfer ranking pada checkpoint target tetap.

Pembandingan ini mendukung kekhususan pertanyaan. Ia bukan bukti prioritas global.
Tidak ada klaim *first*, *unique*, *rare*, peta rusak, atau kelangkaan universal.
Rincian sumber dan keterbatasan akses tercatat pada `LITERATURE_COMPARISON.md`.

## Bahasa yang dijaga

Relevansi berarti respons terhadap intervensi yang dinyatakan. Ia tidak berarti
lokasi gejala yang tervalidasi. Transfer berarti penggunaan ranking sebagai
urutan pemilihan patch input. Ia tidak berarti menyalin bobot attention head atau
mentransfer checkpoint UCI untuk menguji NewHandPD.
