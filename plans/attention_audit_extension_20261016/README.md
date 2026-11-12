# Pelaksanaan penguatan audit attention setelah V5

Paket ini menyiapkan analisis pada **bank NewHandPD tiga seed yang sudah ada**.
Tidak ada training, inference model nyata, intervensi, bootstrap subjek, atau
analisis ilmiah baru yang dijalankan agen. Notebook dibangkitkan melalui builder;
Anda menjalankan sel penelitiannya. V5, notebook 15–18, checkpoint, CSV dan
manifest historis dipertahankan.

## Berkas utama

| Berkas | Fungsi |
|---|---|
| `PROTOCOL.md` | Rencana operasional tambahan eksploratori, sebelum keluaran swap baru diketahui |
| `DECISIONS.md` | Keputusan tahap ini dan keputusan replikasi yang masih terbuka |
| `source_pins.json` | Hash sumber yang dipatok; berisi identitas/provenance, bukan hasil penelitian baru |
| `protected_files_before.json` | Snapshot hash 2.203 berkas lama sebelum implementasi |
| `validation/` | Pemeriksaan perangkat lunak; bukan hasil eksperimen |
| `../../scripts/build_nb_19.py` | Builder diagnostik dan sensitivitas dari artefak tersimpan |
| `../../scripts/build_nb_20.py` | Builder ranking-swap model tetap, plasebo dan kontrol RMS |
| `../../notebooks/19_attention_audit_diagnostics.ipynb` | Notebook CPU, sudah dibuat tanpa eksekusi |
| `../../notebooks/20_fixed_model_ranking_swap.ipynb` | Notebook GPU, sudah dibuat tanpa eksekusi |

Jalur relatif pada tabel di atas dibaca dari root proyek melalui nama lengkap
`scripts/...` dan `notebooks/...`; kedua folder itu berada di root Skripsi.

## Urutan menjalankan

1. Buka `notebooks/19_attention_audit_diagnostics.ipynb` di VS Code/Jupyter.
   Pilih kernel **Skripsi (.venv)**, kemudian **Restart Kernel → Run All**.
2. Periksa `analysis_manifest.json`: status harus `complete` dan seluruh
   `alignment_checks.csv` harus lolos. Notebook 19 tidak memerlukan CUDA.
3. Buka `notebooks/20_fixed_model_ranking_swap.ipynb` pada kernel proyek dengan
   CUDA asli, kemudian **Restart Kernel → Run All**. Tutup run swap lain yang
   masih aktif. Penguncian cache mencegah dua loop swap berjalan bersamaan.
4. Notebook 20 memakai konfigurasi yang sudah tertulis, bukan memilih estimator
   berdasarkan hasil notebook 19. Ia tidak memerlukan hasil notebook 19 sebagai
   input komputasi, sehingga keduanya dapat ditinjau secara terpisah.
5. Setelah selesai, berikan nama direktori run baru. Jika akses workspace tetap
   tersedia, seluruh keluaran dapat diperiksa langsung; Anda tidak perlu memilih
   angka untuk disalin ke chat.

**Jangan Run All notebook 16 untuk memenuhi dependency**: notebook tersebut dapat
melatih model. Semua input tahap ini berasal dari run lengkap yang sudah dipatok.

Notebook sudah tersedia. Regenerasi hanya diperlukan jika builder diperbarui:

```bash
.venv/bin/python -B scripts/build_nb_19.py
.venv/bin/python -B scripts/build_nb_20.py
```

Builder hanya membuat notebook, tidak menjalankan sel. Jika notebook dengan nama
sama sudah ada, builder mengarsipkannya beserta output ke `notebooks/archive/`
sebelum menulis versi baru. Jangan edit `.ipynb` langsung.

## Input yang diperlukan

- Bank E2: `results/e2/20261012T050124548170Z/`.
- Kontrol RMS: `results/e2_amplitude/20261014T024232464643Z/`.
- Penghubung: `results/e1_e2_rms_bridge/20261014T041914953496Z/`.
- Checkpoint dan metadata `checkpoints/e2/<model_id>/` yang dirujuk inventory.
- Cache fold `results/e2_cache/<evaluation_id>/<model_id>/` beserta donor plans.
- Cache rekaman `data/cache/newhandpd_spiral_fs100.npz`.
- Profil runtime tersimpan `c994f20cfc2ce7b14f283f4c4da90562876e05aa9830c3758f0eb178561ad29a`.
- Versi paket CUDA/model yang sama dengan bank asli untuk notebook 20.
- Kode lama yang sesuai hash, termasuk builder 17/18. Builder baru menyalin
  definisi pemeriksaan yang relevan sebagai teks; tidak menjalankan notebook lama.

Tidak ada input dari bank AUROC historis yang disubstitusikan ke bank ini.

## Keluaran notebook 19

Direktori: `results/attention_audit_diagnostics/<run_id>/` dan
`figures/attention_audit_diagnostics/<run_id>/`.

| Tabel | Yang diperiksa |
|---|---|
| `map_concentration.csv` | Entropy, effective support, massa maksimum, W, gap/tie batas top-k |
| `map_reference_draws.csv`, `map_reference_pairs.csv` | JSD aktual versus circular-shift/permutation reference; jumlah draw dan map periodik |
| `map_reference_subject_ci.csv` | Selisih JSD aktual–rerata referensi, dengan CI subjek bersyarat |
| `output_change_sensitivity.csv` | Spearman asli dan partial-rank berbobot, raw/H, termasuk raw pada support normalized yang sama |
| `replacement_common_pair_support.csv` | Irisan identitas dan matching kedua replacement |
| `replacement_contrast_ci.csv`, `replacement_coverage.csv` | Kontras asosiasi median–donor dan denominator pada support bersama |
| `alignment_checks.csv`, `input_inventory.csv`, `analysis_manifest.json` | Integritas dan completion |

Grafik: JSD aktual versus rerata referensi (skala penuh 0–1), konsentrasi attention,
sensitivitas perubahan output, dan kontras replacement. Draw referensi adalah
konstruksi pembanding, bukan physiological ground truth atau ratusan seed baru.

## Keluaran notebook 20

Direktori: `results/attention_ranking_swap/<run_id>/` dan
`figures/attention_ranking_swap/<run_id>/`.

| Tabel | Yang diperiksa |
|---|---|
| `directed_seed_schedule.csv` | Enam pasangan terarah per arsitektur; tanpa self-pair ilmiah |
| `runtime_profile_check.csv`, `runtime_sentinel_points.csv`, `target_validation.csv` | Profil lama, anchor, alpha dan replay attention |
| `self_swap_checks.csv` | D tepat nol ketika terdefinisi; ketidaklayakan matching tetap dibedakan dari nol |
| `control_components.csv`, `control_responses.csv` | Mask, geometri, RMS, kuota, respons kontrol dan sebarannya |
| `rank_scores.csv`, `rank_points.csv`, `rank_component_links.csv` | AOPC dan keterkaitan setiap ranking dengan komponen kontrol yang sebenarnya |
| `swap_pair_curves.csv`, `swap_pair_budgets.csv` | D, G sendiri/tukar, ukuran normalized, perubahan output dan eligible flags |
| `paired_background_dose.csv` | RMS sendiri–tukar dan geometri per budget/background |
| `swap_primary_ci.csv` | Empat mean-D primer, CI nominal, p bootstrap berpusat aproksimatif dan penyesuaian Holm |
| `swap_secondary_ci.csv`, `swap_association_ci.csv`, `swap_normalization_ci.csv`, `swap_budget_ci.csv` | Keuntungan ranking sumber, besar/arah perubahan, hubungan dengan JSD dan sensitivitas |
| `swap_coverage.csv`, `swap_subject_coverage.csv`, `swap_seed_pair_coverage.csv` | Pasangan, rekaman, subjek, target/source seed dan subset stabil |
| `placebo_matching.csv`, `placebo_candidates.csv` | Kedekatan JSD, aturan seleksi, kandidat dan kegagalan |
| `swap_placebo_pairs.csv`, `swap_placebo_ci.csv`, `swap_placebo_coverage.csv` | Perbandingan ranking sumber–plasebo pada support bersama |
| `control_reweighting_sensitivity.csv`, `pair_control_reweighting_sensitivity.csv` | Sensitivitas terhadap kontrol tersimpan; kuantil ini **bukan CI populasi** |
| `swap_replacement_common_support.csv`, `swap_replacement_contrast_ci.csv` | Kontras D/H median–donor pada irisan pasangan |
| `leave_one_seed_out.csv` | Pengaruh mengeluarkan setiap seed, bukan interval populasi seed |
| `record_cache_inventory.csv`, `computation_timings.csv`, `timing_summary.csv` | Resume dan biaya fase yang terukur; tidak mencakup benchmark training baru |

Grafik: mean D dan keuntungan ranking sumber, JSD versus D/|D|, coverage dengan
panel terpisah untuk subset stabil, serta ranking sumber versus plasebo.
Label dan legenda menggunakan bahasa Inggris. Grafik baru belum dapat ditinjau
secara empiris/visual sampai notebook Anda menghasilkan angka sebenarnya.

## Cara membaca hasil

1. Pastikan identitas dan replay lolos sebelum menafsirkan efek.
2. Periksa cakupan matching, termasuk plasebo. Gagal mencari match tidak berarti
   match secara matematis mustahil. Tidak ada fallback ke nearest candidate.
3. Baca mean D bersama G sendiri/tukar dan |D|. Rerata yang sama dapat menutupi
   perubahan arah antarrekaman; D sekitar nol tidak membuktikan equivalence.
4. Baca raw/H pada support yang sama dan periksa dosis sendiri–tukar.
5. Bandingkan ranking sumber dan plasebo pada irisan eligibility, bukan dua
   populasi yang berbeda. JSD matching tidak menyamakan seluruh sifat mask.
6. Pisahkan CI subjek, sensitivitas kontrol, dan leave-one-seed-out. Semuanya
   menjawab sumbu ketidakpastian yang berbeda.
7. Status tetap eksploratori. P Holm hanya untuk empat mean-D primer; CI tetap
   nominal dan hasil sekunder/sensitivitas tidak otomatis terkoreksi.

## Jika run terputus atau gagal

Run All membuat folder run baru. Notebook 20 dapat membaca cache rekaman lengkap
di namespace baru `results/attention_ranking_swap_cache/<analysis_id>/`, setelah
hash, konteks input/model dan seluruh respons direplay. Ia tidak mengulang search
untuk cache yang cocok. Crash setelah atomic payload write dapat dipulihkan dengan
membuat checksum sidecar baru **setelah replay penuh**, tanpa mengubah payload.

Jika mismatch terjadi, pertahankan exception, manifest dan CSV diagnostik. Jangan
memperlebar toleransi, mengganti seed/checkpoint atau menghapus rekaman agar run
lolos. Perbedaan environment menghentikan notebook sebelum loop intervensi baru.

## Pemeriksaan perangkat lunak

```bash
.venv/bin/python -B tests/test_attention_extension_builders.py
```

Tes memakai objek sintetis kecil dan pemeriksaan sintaks/struktur. Ia tidak
membaca sinyal penelitian, menjalankan model, atau membuat bootstrap subjek.
Lolos tes tidak berarti runtime CUDA atau hasil ilmiah telah tervalidasi.

## Sesudah hasil tersedia

Periksa paket hasil lengkap dan profil biaya, kemudian tetapkan N seed, apakah N
berarti total atau seed baru, task primer UCI, jadwal swap replikasi, inferensi dan
aturan replikasi. Siapkan protokol prospektif dengan pengungkapan hasil awal yang
sudah diketahui. Pendaftaran publik memerlukan instruksi terpisah. Baru setelah
hasil nyata diperiksa, susun versi manuskrip berikutnya dari V5.
