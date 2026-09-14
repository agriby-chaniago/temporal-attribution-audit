# Atribusi dan Lisensi Data

Menutup Lampiran C butir G8. Kedua basis data yang dipakai penelitian ini bersifat publik dan
mensyaratkan atribusi. Syarat itu dipenuhi di sini, pada naskah proposal, dan pada setiap subbab yang
memakainya.

---

## UCI 395 — Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet

**Sitasi.** M. Isenkul, B. Sakar, dan O. Kursun, *Parkinson Disease Spiral Drawings Using Digitized
Graphics Tablet*, UCI Machine Learning Repository, 2013. DOI: 10.24432/C5Q01S.

**Publikasi asal.** M. E. Isenkul, B. E. Sakar, dan O. Kursun, "Improved spiral test using digitized
graphics tablet for monitoring Parkinson's disease," dalam *Proc. 2nd International Conference on
e-Health and Telemedicine (ICEHTM-2014)*, Istanbul, Turki, hlm. 171–175, 2014.

**Lisensi.** Creative Commons Attribution 4.0 International (**CC BY 4.0**).
<https://creativecommons.org/licenses/by/4.0/>

Lisensi ini mengizinkan penyalinan, penyebaran, dan modifikasi untuk keperluan apa pun termasuk
komersial, dengan dua syarat: atribusi diberikan, dan perubahan dinyatakan.

**Perubahan yang dilakukan penelitian ini.** Dinyatakan sebagaimana disyaratkan lisensi:

- Pemecahan berkas menjadi segmen pada lompatan timestamp mundur, yang menandai batas antar sesi
  perekaman yang tersambung dalam satu berkas. Potongan tidak disambung agar tidak menciptakan
  transien palsu
- Segmen berdurasi di bawah 2,86 detik dibuang, ambang yang diturunkan dari pita tremor terendah
- Interpolasi ke grid waktu seragam pada frekuensi asli 127,52 Hz, lalu penurunan ke 100 Hz melalui
  `resample_poly` yang menyertakan filter anti-alias
- Penurunan enam kanal masukan model dari koordinat dan tekanan (Subbab 3.4 naskah)

Data mentah tidak diubah dan tetap disimpan apa adanya di `data/raw/uci395/`.

---

## NewHandPD

**Sitasi yang disyaratkan.** C. R. Pereira, S. A. T. Weber, C. Hook, G. H. Rosa, dan J. P. Papa,
"Deep learning-aided Parkinson's disease diagnosis from handwritten dynamics," dalam *Proc. 29th
SIBGRAPI Conference on Graphics, Patterns and Images*, 2016.

**Sumber.** Halaman resmi basis data HandPD, Universidade Estadual Paulista.
<https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/>

**Berkas yang diunduh.**

```
NewHealthy/HealthySignal.zip    145 MB    35 subjek kontrol
NewPatients/PatientSignal.zip   240 MB    31 subjek penderita
```

**Ketentuan penggunaan.** Basis data ini mensyaratkan pengutipan publikasi SIBGRAPI 2016 di atas.
Syarat itu dipenuhi pada Daftar Pustaka naskah sebagai rujukan [3], dan dikutip pada setiap subbab
yang memakai basis data ini.

**Perubahan yang dilakukan penelitian ini.**

- Hanya empat tugas spiral (`sigSp1` sampai `sigSp4`) yang dipakai, dari dua belas tugas yang tersedia
- Penurunan frekuensi dari 1000 Hz ke 100 Hz melalui `resample_poly(up=1, down=10)`
- Pemisahan kanal: kanal 1 sampai 3 diberikan kepada model setelah direkayasa menjadi enam kanal
  turunan; kanal 4 sampai 6 **ditahan sepenuhnya** dan hanya dipakai menghitung penanda motorik
  (Subbab 3.3.2 naskah)

Data mentah tidak diubah dan tetap disimpan apa adanya di `data/raw/newhandpd/`.

---

## Penanganan data pribadi

Header meta tiap berkas sinyal **NewHandPD** memuat kolom identitas — `Forename`, `Surename`,
`Person_ID_Number`, dan `Notice` — di samping kovariat demografis (`Age`, `Gender`, `Writing_Hand`,
`Weight`, `Height`, `Smoker`). Pada rilis yang diunduh, sekurangnya satu subjek memiliki nama lengkap
asli beserta nomor yang menyerupai nomor rekam medis pada kolom tersebut.

Ketentuan yang berlaku pada repositori ini:

1. Ekstraksi metadata dibatasi pada daftar kolom non-identitas `META_AMAN` di `src/newhandpd.py`.
   Kolom identitas tidak pernah ditulis ke berkas hasil.
2. **Metadata mentah tidak disertakan** dalam repositori. Pembaca yang hendak mereproduksi hasil
   mengunduh data langsung dari sumber aslinya.
3. Berkas hasil yang memuat kovariat (`results/p4_meta_subjek.csv`) hanya berisi kolom non-identitas.

---

## Perangkat lunak pihak ketiga

**mamba-ssm.** Modul Mamba-3 (`mamba_ssm/modules/mamba3.py` dan `mamba_ssm/ops/triton/mamba3/`)
disalin dari rilis `v2.3.2.post1` repositori `state-spaces/mamba` ke dalam lingkungan virtual proyek,
tanpa modifikasi. Sitasi: A. Lahoti dkk., "Mamba-3: Improved Sequence Modeling using State Space
Principles," ICLR 2026, arXiv:2603.15569.
