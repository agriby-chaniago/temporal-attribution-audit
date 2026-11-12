# Perluasan audit: 20 seed dan dua dataset

Ini implementasi baru dari rencana perluasan, bukan revisi manuskrip. Agen hanya
menghasilkan notebook dari builder dan memeriksa implementasinya. Training,
inference, intervensi, bootstrap, dan hasil ilmiah baru belum dijalankan.

## Keputusan yang digunakan

| Komponen | Ketetapan |
|---|---|
| Arsitektur | BiGRU dan BiMamba-2 |
| Dataset | NewHandPD spiral; UCI 395 SST saja |
| Seed | 0–19, total 20 per arsitektur/dataset |
| Fold | 5, kelompok subjek, split seed 42 |
| NewHandPD | 30 checkpoint lama dibaca; 170 baru dilatih |
| UCI-SST | 200 checkpoint baru; model historis tidak dipakai |
| Analisis replikasi NewHandPD | Kedua anggota pasangan berasal dari 17 seed baru (3–19) |
| Sensitivitas NewHandPD | Bank gabungan 20 seed, diberi label terpisah |
| Transfer | Lima sumber per target; ring acak tetap, resiprokal, degree seimbang |
| Primer | Mean own-minus-transferred RMS-control advantage, delapan strata |
| Koreksi | Holm atas delapan kontras primer; CI tetap nominal |
| Inferensi | Bootstrap subjek bersyarat pada bank; variasi seed ditampilkan terpisah |
| Resep | LR 0.0003; batch 8; 35 epoch; final checkpoint; normalizer training fold |

20 adalah **total**, bukan 20 tambahan. Replikasi NewHandPD tidak menganggap tiga
seed yang hasilnya sudah diketahui sebagai replikasi independen. Jadwal pada
`directed_seed_schedule.csv` ditentukan dari seed penjadwalan, tanpa membaca
respons model. Detail dan batas klaim tersedia dalam `PROTOCOL.md`.

Target 30 tidak dijalankan otomatis jika hasil 20 mengecewakan. Perubahan target
harus berupa protokol versi baru sebelum hasil terkait diperiksa. Berkas
`FINALIZED.json` menandai spesifikasi lokal; **bukan bukti registrasi publik**.

## Notebook sudah dibangun

Builder tunggal: `scripts/build_nb_21_24.py`. Jangan mengedit `.ipynb` langsung.

1. `notebooks/21_replication20_design_preflight.ipynb` — jalankan **dahulu**.
   Memeriksa cache, identitas rekaman/subjek, split dan jadwal. Menghasilkan
   denominator SST aktual, anggaran baris evaluasi dan batas atas forward.
   Tidak menjalankan model atau bootstrap.
2. `notebooks/22_replication20_newhandpd.ipynb` — Run All melakukan training
   170 model yang belum tersedia, prediksi bank, intervensi dan analisis.
3. `notebooks/23_replication20_uci_sst.ipynb` — pipeline sama untuk SST;
   200 model dilatih bila belum tersedia. Tidak menggabungkan task lain.
4. `notebooks/24_replication20_joint_report.ipynb` — sesudah kedua dataset
   lengkap. Memeriksa completion status/hash/alignment dan menyusun delapan
   kontras dengan Holm. Tidak melakukan model inference.

Gunakan kernel **Skripsi (.venv)**. Notebook 22/23 membutuhkan CUDA serta versi
lingkungan training yang sama dengan bank discovery. Source, cache, normalizer,
checkpoint dan konfigurasi diverifikasi; ketidakcocokan menghentikan run dan
tidak memicu training pengganti checkpoint lama.

Untuk regenerasi identik, dari root proyek:

```bash
.venv/bin/python scripts/build_nb_21_24.py
```

Builder tidak menjalankan sel. Notebook yang sudah berisi hasil dipertahankan.
Perubahan scientific code setelah design seal membutuhkan versi rencana baru.
Flag `--refresh-unexecuted` hanya untuk draft implementasi yang belum disegel,
belum dijalankan dan belum mempunyai hasil; tidak digunakan untuk run penelitian.

## Input yang harus tersedia

- Kedua cache `data/cache/*_fs100.npz` yang dipatok.
- Manifest/record/model inventory E2 `20261012T050124548170Z` dan seluruh
  checkpoint seed 0–2 NewHandPD yang dirujuknya.
- Tabel discovery swap `20261017T002514477657Z` untuk perbandingan bersyarat.
- Source training/model/normalizer serta builder 16–20 dengan hash yang dipatok.

Manifest menautkan input melalui `design_lock.json`; checkpoint lama juga
diperiksa terhadap payload metadata dan SHA-256 bobot. Umur tidak diisi atau
digabungkan berdasarkan urutan baris. Dataset mentah tidak diunduh atau dibangun ulang.

## Anggaran dan runtime

NewHandPD mempunyai 100 arah terjadwal per arsitektur: 105.600 baris
rekaman×target×source×arsitektur×replacement untuk seluruh bank 20 seed.
Primer seed baru menggunakan subgraf tersendiri. UCI memakai jumlah SST aktual.
Notebook 21 menghitung denominator tepat dari manifest, sebelum training.

Satu rank curve dapat membutuhkan paling banyak 184 forward batch-1 pada empat
budget apabila seluruh kuota berhasil. Own + lima transfer + dua placebo untuk
setiap transfer memberi paling banyak 16 rank curves per target/rekaman sebelum
reuse mask identik. Ini batas desain konservatif, **bukan perkiraan waktu**.
Matching gagal, mask identik dan cache memengaruhi jumlah kerja aktual.

Belum ada benchmark runtime bank baru. Jangan menafsirkan jumlah baris sebagai
jam. `timings.jsonl` memisahkan training, prediksi asli, candidate search,
ranking forward dan control forward. Run lama dapat memberi konteks, tetapi
durasi setiap fase berbeda menurut model/panjang rekaman/cache.

## Checkpoint, resume dan kegagalan

- Checkpoint baru: `results/attention_replication20_models/<cohort>/checkpoints/e2/<model_id>/`.
  Direktori lengkap dipublikasikan atomik; staging terputus tidak dianggap model valid.
- Hasil: `results/attention_replication20/<cohort>/<design_id>/<runtime_id>/`.
- Visual: `figures/attention_replication20/<cohort>/<design_id>/<runtime_id>/`.
- Model selesai, prediksi per rekaman, evaluasi rekaman selesai, serta komponen
  mask pada ranking yang selesai dapat dipakai ulang. Payload cache dikompresi
  dan mempunyai checksum. File penelitian lama tidak ditimpa.
- Training **belum** mendukung resume tengah epoch. Fold yang belum selesai
  mengulang training 35 epoch; fold selesai tetap tersedia.
- Dua kernel pada stage yang sama ditolak oleh lock PID. Lock proses yang sudah
  mati pada host yang sama diarsipkan sebagai stale saat resume. Jangan menghapus
  lock proses yang masih hidup; hentikan kernel lama terlebih dahulu.
- Runtime berbeda memakai namespace evaluasi berbeda. Toleransi anchor tetap;
  kegagalan tidak diperbaiki dengan melonggarkan toleransi secara diam-diam.

## Tabel dan grafik yang akan dihasilkan setelah Anda menjalankan

- Inventory 200 checkpoint, manifest rekaman/fold, prediksi OOF dan probabilitas
  subjek, serta AUROC **per seed**.
- Seluruh planned transfer rows, termasuk own/transferred matching failures;
  dosis/run geometry/overlap setiap budget dan background; self-swap checks.
- Mean D primer, own/transferred G, D_H/raw/stable/dose sensitivities;
  JSD associations; coverage pasangan/rekaman/subjek/target.
- Eligible-versus-excluded characteristics dan standardized differences;
  tahap kegagalan JSD/control placebo dan real-minus-placebo contrast.
- Distribusi target-seed dan leave-one-seed-out pada kedua peran;
  perbandingan subjek new-minus-discovery untuk NewHandPD.
- Forest plot transfer, scatter JSD–|D| pada skala penuh, dan coverage dengan
  panel subset stabil berskala terpisah. PDF/SVG/PNG disimpan.
- Notebook 24: `primary_eight_strata_holm.csv` dan manifest dua bank lengkap.

CI subjek tidak mencakup seluruh variasi training. Dosis yang dibatasi bukan
identifikasi kausal. Hasil raw logit antararsitektur/dataset tidak digunakan sebagai
skala kualitas penjelasan universal. Naskah dan abstrak belum diperbarui; abstrak
tetap tanpa angka sesuai preferensi penulis.

## Yang diperiksa sebelum diserahkan

Schema notebook, kompilasi semua sel sebagai source Python (tanpa eksekusi),
dependensi nama global, keseimbangan/resiprositas jadwal dan pemisahan seed lama.
Pengujian perangkat lunak sintetis mencakup geometri/RMS/kuota gagal, bobot
hierarkis, simetri dosis, checksum dan lock. Pengujian memakai data buatan kecil;
tidak membaca data studi, melatih model, menjalankan network forward atau bootstrap.

Lihat `BUILD_CHECKS.json` dan `validation/VALIDATION_REPORT.md`. Pipeline ilmiah
end-to-end pada CUDA **belum dijalankan oleh agen**; kelulusan pemeriksaan software
tidak diganti menjadi klaim bahwa run penelitian sudah berhasil.
