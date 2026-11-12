# Jawaban singkat untuk diskusi

**Apakah AUROC tinggi berarti peta attention baik?**

AUROC mengukur diskriminasi kelompok. Nilai itu belum memvalidasi penjelasan.
Audit ini menguji ranking pada model yang tetap. Ia tidak menghitung korelasi
universal AUROC versus kualitas penjelasan atau membuktikan semua model dengan
AUROC tinggi mempunyai peta buruk.

**Apa sebenarnya yang ditransfer?**

Urutan patch menurut attention seed sumber. Sumber dan target menggunakan
arsitektur, dataset, rekaman, fold, dan grid temporal yang sama. Ranking sumber
menentukan lokasi input yang diganti. Checkpoint target dan normalizernya tetap.
Bobot attention head tidak ditimpa.

**Apa arti sebagian relevansi berpindah?**

Pada pasangan eligible, ranking transfer mempunyai keuntungan rata-rata di atas
kontrolnya. Ranking sendiri mempunyai keuntungan yang lebih besar. Pernyataan
ini tidak mengatakan bahwa efek setiap pasangan positif.

**Apakah BiMamba-2 lebih baik daripada BiGRU?**

Tidak ada pemenang global yang ditetapkan oleh audit ranking ini. Raw
logit-AOPC memiliki skala model sendiri. AUROC historis menunjukkan BiMamba-2
lebih rendah pada UCI dan lebih tinggi pada NewHandPD setelah retraining.
CI kontras penurunan masih mencakup nol. Brier discovery memberi bukti positif
untuk kualitas probabilitas pada bank itu, dengan estimand berbeda dari ranking.

**Apakah model menerima pasien yang sama pada train dan test?**

Semua rekaman satu subjek berada pada fold yang sama. Normalisasi dan donor
berasal dari training fold. Source dan target diuji pada rekaman held-out yang
sama. Berbagi partisi test antar seed diperlukan untuk pembandingan berpasangan.

**Mengapa memakai RMS dan geometri?**

Respons model dapat membesar ketika intervensi mengubah input lebih kuat.
RMS merangkum besar perubahan. Jumlah patch dan panjang run membatasi geometri.
Kontrol ini tidak menjamin sinyal hasil replacement sepenuhnya berada dalam
distribusi asli.

**Mengapa pasangan matching gagal?**

Prosedur tidak mendapatkan kuota dalam jumlah kandidat yang diizinkan.
Hasil utama mewakili support yang lolos. Kegagalan search tidak membuktikan
bahwa tidak ada match yang mungkin.

**Mengapa tidak hanya memakai subset output stabil?**

Subset tersebut kecil dan mengubah support. Analisis kontinu mempertahankan
variasi output di seluruh pasangan eligible. Gate stabil menjadi sensitivitas.

**Apa yang dinormalisasikan pada bridge?**

Ukuran sensitivitas memakai H = (T − R)/(T + R), dengan T + R > 10⁻⁸.
Perubahan absolut H dibandingkan antar seed. Spearman normalized dan Pearson
tersentrik menjawab estimand berbeda. Normalisasi tidak membuktikan bahwa seluruh
pengaruh skala atau perbedaan fungsi model telah hilang.

**Apakah JSD positif terhadap |D| merupakan sebab-akibat?**

Tidak. Identitas ranking dan mask dapat menyumbang hubungan tersebut secara
konstruksi. Placebo pada JSD serupa memberi referensi tambahan. Hubungan dengan
|D| mengukur besar perubahan tanpa menilai arah penurunan kualitas.

**Mengapa 20 seed belum berarti sampel klinis besar?**

Seed mengulang training pada peserta yang tersedia. NewHandPD tetap 66 subjek.
UCI-SST tetap 76 subjek. Pasangan berbagi model dan peserta. Bootstrap subjek
dan diagnostik pengaruh seed menjawab sumbu ketidakpastian berbeda.

**Apakah protokol ini preregistered?**

Perluasan dikunci lokal sebelum hasil baru, setelah discovery diketahui.
Ia tidak didaftarkan publik. Analisis tambahan dilaporkan eksploratori.

**Apa yang diperlukan untuk klaim fisiologis?**

Rujukan temporal independen yang sesuai dengan gejala atau gerak yang dinilai.
Attention, custom Shapley, dan marker sensor tidak secara otomatis memenuhi
syarat tersebut.

## Glosarium pembicara

- **PD / HC:** kelompok diagnosis Parkinson dan kelompok kontrol pada dataset.
- **Discovery:** tahap tiga seed yang hasilnya sudah diketahui sebelum perluasan.
- **Seed:** nilai yang mengendalikan keacakan pelatihan.
- **Logit:** skor classifier sebelum sigmoid mengubahnya menjadi probabilitas.
- **Attention map:** distribusi bobot pooling sepanjang patch waktu.
- **Ranking:** urutan patch menurut bobot attention.
- **Patch:** kelompok tujuh sampel input; waktu UCI tetap nominal.
- **JSD:** divergence Jensen–Shannon basis dua; nol berarti distribusi identik.
- **RMS:** root mean square dari perubahan input yang dinormalisasi.
- **AOPC:** ringkasan luas kurva respons terhadap budget intervensi.
- **G:** AOPC ranking dikurangi AOPC kontrol terpadan.
- **D:** G ranking sendiri dikurangi G ranking transfer pada target tetap.
- **CI:** selang kepercayaan; di sini bersyarat pada bank dan desain tersedia.
- **Holm:** koreksi beberapa pengujian primer dalam satu keluarga yang ditetapkan.
- **Placebo:** ranking konstruksi pembanding dari pergeseran atau permutasi peta.
- **LOSO:** pemeriksaan pengaruh dengan mengeluarkan satu seed dari kedua peran.
