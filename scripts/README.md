# Peta `scripts/`

Enam puluh berkas dalam satu direktori datar. Berkas ini mengelompokkannya per peran,
sehingga urutan pemakaian dan ketergantungan antar berkas terbaca tanpa membuka satu per satu.

Nama berkas **tidak** diubah dengan sengaja: jalur seperti `scripts/analisis_p4_geser_tugas.py`
dikutip apa adanya pada peta provenans di `naskah/NASKAH-SUMBER.md` dan pada `naskah/DOSIER-PROYEK.md`.
Memindahkannya ke subdirektori akan membuat kedua tabel itu menunjuk ke tempat yang tidak ada.

Peta ini menyebut **peran**, bukan angka hasil. Angka hidup di `results/` dan ditafsirkan di naskah;
menyalinnya ke sini hanya menambah satu lapis lagi yang dapat basi tanpa ketahuan.

---

## 1. Penyiapan data dan lingkungan

| Berkas | Peran |
|---|---|
| `unduh_data.py` | Unduh UCI 395 dan NewHandPD dari sumber aslinya ke `data/raw/`. Idempoten, dan mencocokkan sha256 tiap arsip terhadap rilis yang dipakai penelitian ini. |
| `patch_mamba_ssm.py` | Tambal `mamba_ssm` sesudah `pip install -r requirements.txt`. Idempoten. Lihat `SETUP.md`. |

## 2. Pembangun notebook

Notebook **tidak pernah disunting langsung**; ia dibangun dari berkas-berkas ini, sehingga
perubahannya terbaca lewat diff kode, bukan lewat JSON notebook.

| Berkas | Keluaran |
|---|---|
| `build_nb_02.py` | `notebooks/02_prapemrosesan.ipynb` |
| `build_nb_04.py` | `notebooks/04_kontrol_positif.ipynb` (S1) |
| `build_nb_05.py` | `notebooks/05_atribusi_shapley.ipynb` |
| `build_nb_06.py` | `notebooks/06_pilot_resolusi.ipynb` (S2) |
| `build_nb_07.py` | `notebooks/07_eksperimen_utama.ipynb` (S3) |
| `build_nb_08.py` | `notebooks/08_kesetiaan_atensi.ipynb` (S6) |
| `build_nb_09.py` | `notebooks/09_keselarasan_klinis.ipynb` (S7, analisis primer) |
| `build_nb_10.py` | `notebooks/10_uji_kewarasan.ipynb` (S8) |
| `build_nb_11.py` | `notebooks/11_lintas_tugas.ipynb` (S4) |
| `build_nb_12.py` | `notebooks/12_replikasi_newhandpd.ipynb` (S5) |
| `build_nb_13.py` | `notebooks/13_fold_rekaman.ipynb` (C1) |
| `build_nb_14.py` | `notebooks/14_visualisasi_data.ipynb` |
| `build_nb_master.py` | `notebooks/00_induk.ipynb` — satu alur yang dapat dijalankan dari atas ke bawah, memuat artefak `results/` tanpa melatih ulang apa pun |
| `gabung_notebook.py` | `notebooks/00_gabungan_beku.ipynb` — arsip tempelan seluruh notebook tahapan. Arsip, bukan notebook yang dijalankan |

**`build_nb_master.py` menulis ulang notebook tanpa keluaran sel.** Notebook induk yang ada
sekarang memuat keluaran hasil eksekusi. Menjalankan pembangunnya membuang keluaran itu, dan
memulihkannya menuntut eksekusi ulang:

```bash
python3 scripts/build_nb_master.py
jupyter nbconvert --to notebook --execute --inplace notebooks/00_induk.ipynb
```

## 3. Analisis berdiri sendiri

Tiap berkas menulis satu keluarga berkas di `results/`; awalan keluarannya disebut di kolom kanan.
Peta lengkap subbab-hasil terhadap berkas penghasilnya ada di `naskah/NASKAH-SUMBER.md` (Peta provenans).

| Berkas | Awalan keluaran |
|---|---|
| `analisis_a0_perancu_stcp.py` | `a0_*`, `a1_*` |
| `analisis_a3_penanda_lambat.py` | `a3_*` |
| `analisis_a_baseline_fitur.py` | `a_baseline_*` |
| `analisis_b1_bootstrap.py` | `b1_*` |
| `analisis_bootstrap_retensi.py` | `retensi_bootstrap_*` |
| `analisis_definisi_retensi.py` | `retensi_tiga_definisi`, `retensi_penyesuaian_garis_dasar`, `retensi_per_seed` |
| `analisis_interaksi_arsitektur_kohort.py` | `interaksi_*` |
| `analisis_p0_ensemble_peta.py` | `p0_*` |
| `analisis_p1_s8_sepuluh_seed.py` | `p1b_*` |
| `analisis_p1_welch_sepuluh_seed.py` | `p1a_*`, `p3a_*` |
| `analisis_p2_sapuan_panjang.py` | `p2_*` |
| `analisis_p3_garis_dasar.py` | `p3b_garis_dasar_disamakan`, `p3b_vonis`, `p3b_artefak` |
| `analisis_p3_strata_cocok.py` | `p3b_strata_cocok` |
| `analisis_p4_geser_tugas.py` | `p4_*` |
| `analisis_p5_perancu_usia.py` | `p5_*` |
| `analisis_rm2_tuning.py` | `rm2_*` |
| `analisis_rm5_newhandpd.py` | `rm5_*` |
| `analisis_e7_spektrum.py` | `e7_*` |
| `analisis_e8_konstanta_waktu.py` | **tidak dijalankan**, dan angkanya tidak dipakai naskah. Disertakan agar keputusan itu dapat diperiksa |
| `verifikasi_encoder.py` | tidak menulis berkas; tujuh pemeriksaan per arm, lolos/gagal dilaporkan di layar |
| `plafon_resolusi_kasar.py` | plafon kesetiaan kontrol positif pada resolusi kasar, tanpa melatih ulang |

## 4. Naskah: pembangun dan penyunting

Sumber tunggalnya `naskah/NASKAH-SUMBER.md`. Kedua dokumen turunan — `naskah/sempro-skripsi.md`
dan `naskah/semhas-skripsi.md` — dibangun darinya, tidak disunting sendiri. Keduanya bersanding
dengan `.docx` hasil bangunannya di direktori yang sama.

| Berkas | Peran |
|---|---|
| `pisah_sempro_semhas.py` | **Titik masuk.** Memisah naskah menjadi `naskah/sempro-skripsi.md` dan `naskah/semhas-skripsi.md`; memanggil sendiri penomor tabel dan ketiga pembangun daftar |
| `bangun_daftar_isi.py` | Daftar isi, dibangkitkan dari heading naskah |
| `bangun_daftar_pelengkap.py` | Daftar tabel, daftar lampiran, daftar singkatan |
| `nomori_tabel.py` | Judul dan nomor tabel menurut Panduan Tugas Akhir UHB |
| `petakan_penomoran.py` | Peta penomoran desimal naskah ke penomoran huruf panduan |
| `restruktur_bab.py` | Pindah blok antar bab; `--uji` menuntut bongkar-pasang identik bita per bita |
| `seragamkan_istilah.py` | Seragamkan istilah teknis, cetak miring istilah asing |
| `desimal_dua_angka.py` | Lengkapi desimal menjadi sekurang-kurangnya dua angka di belakang koma |
| `ke_apa.py` | Ubah sitasi IEEE bernomor menjadi APA nama-tahun |
| `tulis_daftar_pustaka.py` | Tulis ulang Daftar Pustaka gaya APA 6th, urut abjad |
| `bangun_pustaka.py`, `tulis_bib.py` | Bangun `references.bib` dari Daftar Pustaka naskah lewat CrossRef dan OpenAlex; `_pustaka_mentah.json` adalah metadata mentah hasil pengambilan itu |

> **Jangan jalankan `bangun_daftar_pelengkap.py` sendirian pada dokumen yang sudah dibangun.**
> Skrip itu **menyisipkan**, bukan mengganti: satu jalannya lagi di atas dokumen yang sudah
> lengkap menggandakan isi daftar tabel (22 menjadi 44 pada sempro, 86 menjadi 172 pada semhas).
> Bangun ulang lewat `pisah_sempro_semhas.py`, yang memanggilnya pada dokumen yang masih bersih.
> `bangun_daftar_isi.py` tidak punya sifat itu, tetapi memakai jalur yang sama tetap lebih aman.

## 5. Dokumen dan gambar

| Berkas | Keluaran |
|---|---|
| `build_docx.mjs` | `naskah/sempro-skripsi.docx` atau `naskah/semhas-skripsi.docx`. Argumen: `sempro` (bawaan) atau `semhas` |
| `build_paper_docx.mjs` | `Q1-amin/paper-full.docx` |
| `lib_markdown_docx.mjs` | Pengurai markdown dialek naskah ini; dipakai kedua pembangun `.docx` di atas, bukan dijalankan sendiri |
| `build_figur_paper.py` | Gambar berbahasa Inggris untuk manuskrip Q1, ke `figures/paper/` |
| `build_gambar_kerangka.py` | Gambar kerangka teori dan kerangka konsep untuk Bab II |
| `perbaiki_benang_merah.py` | Tulis ulang `figures/benang_merah.{png,pdf}` dengan sumber data yang benar |

## 6. Audit dan arsip

| Berkas | Peran |
|---|---|
| `audit_angka_naskah.py` | Cocokkan angka naskah terhadap berkas `results/` dan terhadap pemisahan sempro-semhas |
| `audit_sempro.py` | Audit menyeluruh naskah sempro pada delapan sumbu |
| `periksa_sempro_bersih.py` | Pastikan sempro benar-benar bebas hasil empiris |
| `arsip_prapelaksanaan.py` | Jaring pengaman sebelum pekerjaan besar: salin `results/`, `notebooks/`, `figures/`, `src/`, `scripts/`, dan `naskah/`, beserta manifes sha256 |
| `arsipkan_usang.py` | Pindahkan artefak yang sudah tergantikan ke arsip bertanda, tanpa menghapus apa pun |

Status tiap cadangan tercatat di `cadangan/INDEKS.md`. Cadangan historis disimpan terkompres
sebagai `.tar.zst`; buka dengan `tar --use-compress-program=unzstd -xf <berkas>`.
