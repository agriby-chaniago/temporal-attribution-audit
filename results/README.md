# Peta `results/`

Seratus empat berkas dalam satu direktori datar, dikelompokkan di sini menurut awalan namanya.
Tiap awalan adalah satu keluarga keluaran dari satu penghasil.

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
