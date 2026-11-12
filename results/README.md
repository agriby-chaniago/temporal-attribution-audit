# Peta `results/`

Hasil historis tetap berada pada nama aslinya. Keluaran audit Oktober disimpan
dalam folder keluarga analisis dan run UTC. Ringkasan hasil lengkap yang dipilih
untuk GitHub dipetakan di [cakupan artefak publik](../docs/PUBLIC_ARTIFACTS.md).

## Audit terkini

| Tahap | Run lengkap / lokasi | Cakupan |
|---|---|---|
| E0/E1 | `e0_e1/20261012T035813806939Z/` | AUROC historis dan audit pasangan awal |
| E1/E2 bank discovery | `e2/20261012T050124548170Z/` | NewHandPD, tiga seed, 30 checkpoint |
| Kontrol amplitudo | `e2_amplitude/20261014T024232464643Z/` | Magnitude dan RMS/geometri |
| Penghubung | `e1_e2_rms_bridge/20261014T041914953496Z/` | Fungsi model berubah antar-seed |
| Kalibrasi | `calibration_oof/20261014T135037828430Z/` | Bank historis/discovery, bukan bank dua puluh seed |
| Diagnostik | `attention_audit_diagnostics/20261016T045207091905Z/` | Referensi JSD dan sensitivitas berbobot |
| Swap discovery | `attention_ranking_swap/20261017T002514477657Z/` | Model target tetap, tiga seed |
| Perluasan final | `attention_replication20_resume/` | Dua dataset, dua arsitektur, dua puluh seed; pointer per kohort |

Semua run di atas mempunyai `completion_status: complete` pada manifest yang
diperiksa. Daftar keluaran manifest mencakup hasil lokal yang tidak seluruhnya
diunggah. Ledger publik membedakan file yang disertakan dari tabel rinci,
checkpoint, cache, dan binary yang tetap lokal. Nilai dan timestamp sumber tidak
diubah. Pembaruan GitHub dilakukan 12 November 2026 tanpa komputasi ilmiah baru.

## Keluarga hasil historis

Peta berikut mempertahankan pengelompokan September. Narasi historis tidak
menggantikan [batas klaim terkini](../docs/RESEARCH_STATUS.md).

**Wewenang tetap ada di naskah.** Peta subbab-hasil terhadap berkas penghasilnya ada di
`naskah/NASKAH-SUMBER.md`, bagian "Peta provenans". Berkas ini hanya menjawab pertanyaan yang
berlawanan arah — *berkas ini datang dari mana* — dan sengaja tidak memuat satu pun angka hasil,
supaya tidak ada lapis ketiga yang dapat basi tanpa ketahuan.

Nama berkas tidak diubah: naskah, dosier, dan notebook induk mengutipnya apa adanya.

## Keluaran skenario terprotokol

Dihasilkan notebook, satu skenario satu notebook.

| Awalan | Berkas | Penghasil |
|---|---|---|
| `s1_*` | 3 | `notebooks/04_kontrol_positif.ipynb` — kontrol positif |
| `s2_*` | 4 | `notebooks/06_pilot_resolusi.ipynb` — pilot resolusi |
| `s3_*` | 5 | `notebooks/07_eksperimen_utama.ipynb` — performa klasifikasi |
| `s4_*` | 4 | `notebooks/11_lintas_tugas.ipynb` — lintas tugas |
| `s5_*` | 5 | `notebooks/12_replikasi_newhandpd.ipynb` — replikasi NewHandPD |
| `s6_*` | 10 | `notebooks/08_kesetiaan_atensi.ipynb` — kesetiaan atensi |
| `s7_*` | 9 | `notebooks/09_keselarasan_klinis.ipynb` — keselarasan penanda, analisis primer |
| `s8_*` | 4 | `notebooks/10_uji_kewarasan.ipynb` — uji kewarasan |
| `c1_*` | 3 | `notebooks/13_fold_rekaman.ipynb` — *fold* tingkat rekaman |

## Keluaran analisis berdiri sendiri

Dihasilkan skrip di `scripts/`; lihat `scripts/README.md` bagian 3.

| Awalan | Berkas | Penghasil |
|---|---|---|
| `a0_*`, `a1_*` | 3 + 1 | `analisis_a0_perancu_stcp.py` — audit perancu STCP, massa atensi |
| `a3_*` | 3 | `analisis_a3_penanda_lambat.py` — penanda lambat pada DST |
| `a_baseline_*` | 4 | `analisis_a_baseline_fitur.py` — *baseline* fitur kinematik agregat |
| `b1_*` | 3 | `analisis_b1_bootstrap.py` — bootstrap dan uji permutasi alpha vs phi |
| `e7_*` | 2 | `analisis_e7_spektrum.py` — spektrum rotasi state BiMamba-3 |
| `interaksi_*` | 2 | `analisis_interaksi_arsitektur_kohort.py` — interaksi arsitektur x kohort |
| `p0_*` | 4 | `analisis_p0_ensemble_peta.py` — reproduktibilitas peta ensemble (RM6) |
| `p1a_*` | 2 | `analisis_p1_welch_sepuluh_seed.py` — retensi pada sepuluh *seed* |
| `p1b_*` | 3 | `analisis_p1_s8_sepuluh_seed.py` — kekokohan peta pada sepuluh *seed* |
| `p2_*` | 4 | `analisis_p2_sapuan_panjang.py` — biaya terhadap panjang urutan |
| `p3a_*` | 1 | `analisis_p1_welch_sepuluh_seed.py` — retensi bebas plafon |
| `p3b_*` | 4 | `analisis_p3_garis_dasar.py` dan `analisis_p3_strata_cocok.py` — garis dasar disamakan, dan strata yang dicocokkan |
| `p4_*` | 5 | `analisis_p4_geser_tugas.py` — sumbu geser tugas NewHandPD |
| `p5_*` | 1 | `analisis_p5_perancu_usia.py` — audit perancu usia |
| `retensi_*` | 5 | `analisis_definisi_retensi.py` dan `analisis_bootstrap_retensi.py` — tiga definisi retensi, bootstrap tingkat subjek |
| `rm2_*` | 5 | `analisis_rm2_tuning.py` — *tuning* berimbang bagi ketiga arm |
| `rm5_*` | 3 | `analisis_rm5_newhandpd.py` — RM5 pada 35 kontrol |

## Keluaran analisis tambahan E0/E1

`notebooks/15_audit_e0_e1.ipynb` (dibangun melalui `scripts/build_nb_15.py`)
menyiapkan E0/E1 sebagai analisis tambahan eksploratori setelah hasil lama diketahui.
Status preregistrasi lama tidak diubah. Run lengkap pengguna tercatat pada tabel di atas.
Keluaran masuk `results/e0_e1/<UTC-run-id>/` dan
`figures/e0_e1/<UTC-run-id>/`, tanpa menimpa artefak lama. Input, batas provenans,
estimator, CI bersyarat dan daftar keluarannya dijelaskan di
[petunjuk E0/E1](../scripts/E0-E1.md).

## Keluaran analisis tambahan E2 pada model baru

`notebooks/16_audit_e2.ipynb`, dibangun melalui `scripts/build_nb_16.py`, menyiapkan
E2 dan penghitungan ulang E1 pada checkpoint identik. Run lengkap pengguna tercatat
pada tabel di atas. Bank discovery NewHandPD memakai dua arsitektur, tiga seed, lima fold:
Run All oleh pengguna dapat melatih 30 model jika checkpoint belum ada. Keluaran
baru berada di `results/e2/<UTC-run-id>/`, cache per-fold di `results/e2_cache/`,
dan checkpoint berprovenans di `checkpoints/e2/`. Hasil/preregistrasi lama tidak
ditimpa. Lihat [petunjuk E2](../scripts/E2.md).

## Tanpa penghasil di dalam repositori

Dua berkas dibaca oleh kode di repositori ini, tetapi tidak ada berkas di `scripts/` maupun
`notebooks/` yang menulisnya — juga tidak pada cadangan mana pun.

| Berkas | Keterangan |
|---|---|
| `d4_durasi_postur_lintas_tugas.csv` | Dibaca `scripts/build_nb_master.py` dan `scripts/build_figur_paper.py`. Berasal dari langkah yang tidak terkodekan, dan hanya dapat diperbarui dengan tangan. |
| `kohort_kedua_peringkat.csv` | **Usang.** Memakai lima *seed*; digantikan `a_baseline_lintas_kohort.csv` yang memakai sepuluh. Angkanya tidak lagi dipakai naskah. Status itu tercatat pada daftar `TERGANTIKAN` di `scripts/audit_angka_naskah.py`, yang menyatakannya sebagai catatan alih-alih peringatan. |

## Bentuk berkas

`.csv` memuat angka yang dikutip naskah. `.pkl` adalah artefak antara — tensor, peta atensi, dan
keadaan model — yang dimuat ulang oleh analisis berikutnya supaya tidak perlu melatih ulang. Yang
terbesar `s5_artefak.pkl` (39,8 MB), `s7_artefak.pkl` (10,5 MB), dan `s6_artefak_pasien.pkl`
(2,4 MB).
