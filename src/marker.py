"""
Penanda klinis eksternal, dihitung sepenuhnya tanpa melibatkan model.

Mengikuti Subbab 3.6.4 naskah proposal. Seluruh besaran di modul ini diturunkan
dari sinyal mentah, termasuk koordinat absolut yang sengaja tidak diberikan
kepada model. Konsekuensinya, apabila atribusi model tetap selaras dengan
penanda ini, model menemukannya lewat dinamika saja — hasil yang lebih kuat
daripada bila koordinat ikut diberikan.

Dua keputusan yang menentukan kesahihan penanda
-----------------------------------------------
1. **Filter wajib zero-phase.** Filter kausal menggeser kejadian dalam waktu,
   dan pergeseran itu membuat penanda tidak sejajar secara sistematis dengan peta
   atribusi. Karena yang diukur adalah keselarasan temporal, pergeseran sistematis
   akan merusak justru besaran yang hendak diuji. Dipakai filtfilt.

2. **Pada STCP, penanda dihitung hanya pada segmen pena melayang.** Verifikasi
   Tahap 1 menunjukkan 68 persen baris Test ID 2 bertekanan nol, sesuai definisi
   tugasnya pada Isenkul dkk. (2014): menahan pena di atas titik tanpa menyentuh
   layar. Sisa 32 persen merupakan sentuhan yang tidak diinstruksikan, sehingga
   dikeluarkan dari perhitungan penanda.

Metode filter dipilih menggantikan transformasi Fourier jangka pendek karena
resolusi frekuensi sebuah jendela adalah frekuensi sampling dibagi panjang
jendela. Pada patch pendek, seluruh pita tremor akan berada di dalam satu bin
frekuensi dan tidak terdeteksi.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt, hilbert

__all__ = [
    "PITA_TREMOR", "PITA_VOLUNTER", "BATAS_LAMBAT", "TUGAS_SPIRAL",
    "selubung_pita", "selubung_pita_vektor", "penanda_cepat", "penanda_cepat_bisp",
    "penanda_cadangan",
    "pusat_spiral", "bentangkan_spiral", "residual_spiral", "penanda_lambat",
    "energi_total", "ke_grid_patch", "mask_pena_melayang",
]

PITA_TREMOR = (3.5, 7.5)      # Subbab 2.2.1
PITA_VOLUNTER = (0.1, 3.5)    # gerakan yang disengaja
BATAS_LAMBAT = PITA_TREMOR[0]  # penanda lambat dibatasi di bawah pita tremor
TUGAS_SPIRAL = (0, 1)          # SST, DST. STCP tidak menggambar spiral.


def _bandpass(x: np.ndarray, pita: tuple[float, float], fs: int, orde: int = 4) -> np.ndarray:
    """Bandpass zero-phase. Lihat catatan rancangan nomor 1 pada docstring modul."""
    nyq = fs / 2
    lo, hi = pita[0] / nyq, min(pita[1] / nyq, 0.99)
    b, a = butter(orde, [lo, hi], btype="band")
    # padlen default filtfilt bisa melebihi panjang sinyal pendek
    padlen = min(3 * max(len(a), len(b)), len(x) - 1)
    return filtfilt(b, a, x, padlen=max(padlen, 0))


def selubung_pita(x: np.ndarray, pita: tuple[float, float], fs: int) -> np.ndarray:
    """Amplitudo sesaat pada satu pita, lewat selubung analitik."""
    return np.abs(hilbert(_bandpass(x, pita, fs)))


def kecepatan_bertanda(xy: np.ndarray, fs: int) -> np.ndarray:
    """Komponen kecepatan bertanda (vx, vy). Panjangnya disamakan dengan xy."""
    return np.diff(xy, axis=0, prepend=xy[:1]) * fs


def laju_gerak(xy: np.ndarray, fs: int) -> np.ndarray:
    """Besar kecepatan per langkah, dipakai untuk energi total."""
    return np.linalg.norm(kecepatan_bertanda(xy, fs), axis=1)


def selubung_pita_vektor(xy: np.ndarray, pita: tuple[float, float], fs: int) -> np.ndarray:
    """Amplitudo pita, dihitung dari komponen BERTANDA lalu digabungkan.

    Kritis, dan sempat salah pada versi awal modul ini. Menghitung pita tremor
    dari besar kecepatan yang sudah tersearahkan akan menghancurkan komponen yang
    hendak diukur: rektifikasi sinus 5 Hz menghasilkan komponen searah dan 10 Hz,
    sehingga energi pada 5 Hz justru lenyap sebelum difilter. Uji kewarasan dengan
    tremor buatan memperlihatkan penanda versi lama tidak naik, bahkan turun.

    Filter karena itu diterapkan pada vx dan vy secara terpisah, baru selubungnya
    digabungkan sebagai norma.
    """
    v = kecepatan_bertanda(xy, fs)
    env = np.stack([selubung_pita(v[:, c], pita, fs) for c in range(v.shape[1])], axis=1)
    return np.linalg.norm(env, axis=1)


def penanda_cepat(xy: np.ndarray, fs: int, pakai_rasio: bool) -> np.ndarray:
    """Amplitudo pita tremor, tingkat sampel.

    Parameters
    ----------
    pakai_rasio : True untuk tugas yang mengandung gerakan volunter (SST, DST),
        sehingga daya tremor absolut tidak dapat dipakai langsung dan perlu
        dirasiokan terhadap pita gerakan volunter (Subbab 2.2.1). False untuk
        STCP, yang tidak melibatkan gerakan volunter sehingga amplitudo pita
        tremor dipakai apa adanya.
    """
    tremor = selubung_pita_vektor(xy, PITA_TREMOR, fs)
    if not pakai_rasio:
        return tremor
    volunter = selubung_pita_vektor(xy, PITA_VOLUNTER, fs)
    return tremor / np.maximum(volunter, 1e-9)


def _lowpass(x: np.ndarray, batas: float, fs: int, orde: int = 4) -> np.ndarray:
    """Lolos-rendah zero-phase. Alasan zero-phase sama seperti pada _bandpass."""
    b, a = butter(orde, min(batas / (fs / 2), 0.99), btype="low")
    padlen = min(3 * max(len(a), len(b)), len(x) - 1)
    return filtfilt(b, a, x, padlen=max(padlen, 0))


def pusat_spiral(xy: np.ndarray) -> np.ndarray:
    """Pusat spiral, dicari sebagai titik yang membuat r paling linear terhadap theta.

    Pusat tidak dapat diambil begitu saja dari rerata koordinat. Spiral tidak
    tersebar merata di sekitar pusatnya: putaran luar memuat lebih banyak sampel
    daripada putaran dalam, sehingga rerata tertarik keluar dan residualnya menjadi
    lengkungan sistematis, bukan penyimpangan gerak. Karena definisi spiral
    Archimedes justru adalah r yang linear terhadap theta, pusat yang benar
    ditetapkan sebagai pusat yang meminimalkan sisa kuadrat regresi tersebut.
    """
    def rugi(c):
        r, th = bentangkan_spiral(xy, c)
        if len(r) < 3:
            return np.inf
        A = np.c_[th, np.ones_like(th)]
        sisa = r - A @ np.linalg.lstsq(A, r, rcond=None)[0]
        return float(sisa @ sisa)

    awal = xy.mean(axis=0)
    hasil = minimize(rugi, awal, method="Nelder-Mead",
                     options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 400})
    return hasil.x if hasil.success else awal


def bentangkan_spiral(xy: np.ndarray, pusat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bentangkan spiral dua dimensi menjadi radius terhadap sudut kumulatif [24].

    Sudut dibuka lilitannya (unwrap) supaya terus bertambah sepanjang putaran,
    sehingga spiral ideal menjadi garis lurus pada bidang (theta, r).
    """
    d = xy - np.asarray(pusat, dtype=float)
    r = np.linalg.norm(d, axis=1)
    theta = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return r, theta


def residual_spiral(xy: np.ndarray) -> tuple[np.ndarray, dict]:
    """Simpangan radius tiap sampel terhadap spiral Archimedes ideal.

    Kuantifikasi spiral yang mapan mengukur seberapa jauh gambar menyimpang dari
    hubungan linear antara radius dan sudut [24]. Ukuran yang dilaporkan di sana
    berupa satu skalar per gambar. Yang dipakai di sini adalah residual regresinya
    sendiri, yang sudah terdefinisi per sampel tanpa tambahan asumsi — skalar
    tersebut merupakan ringkasan dari deret residual ini.
    """
    c = pusat_spiral(xy)
    r, th = bentangkan_spiral(xy, c)
    A = np.c_[th, np.ones_like(th)]
    koef = np.linalg.lstsq(A, r, rcond=None)[0]
    sisa = r - A @ koef
    info = {"pusat": c, "pitch": float(koef[0]), "r_median": float(np.median(r)),
            "r2": float(1 - sisa.var() / max(r.var(), 1e-12))}
    return sisa, info


def penanda_lambat(xy: np.ndarray, fs: int, tugas: int) -> np.ndarray:
    """Penyimpangan geometri bertempo lambat, tingkat sampel. Subbab 3.6.4.

    Residual spiral disaring lolos-rendah pada batas bawah pita tremor, sehingga
    yang tersisa hanya simpangan yang berkembang lebih lambat daripada tremor.
    Pemisahan ini bukan sekadar kerapian: tanpa penyaringan, penanda lambat akan
    ikut menangkap tremor yang berosilasi di sekitar garis spiral, dan
    keselarasannya terhadap peta atribusi tidak dapat lagi dibedakan dari
    keselarasan penanda cepat. Dengan penyaringan, kedua penanda menempati pita
    yang saling lepas menurut konstruksi.

    Nilai dikembalikan sebagai besar simpangan tanpa normalisasi. Analisis primer
    Skenario S7 memakai korelasi peringkat, yang kebal terhadap penskalaan monoton
    per rekaman, sehingga normalisasi tidak memengaruhinya. Normalisasi baru
    diperlukan bila nilai dibandingkan antar rekaman, misalnya pada uji pemisahan
    kelompok, dan di sana pembaginya dinyatakan eksplisit.

    Tugas di luar TUGAS_SPIRAL mengembalikan NaN, bukan nol. STCP tidak memuat
    gambar spiral sama sekali, sehingga penanda ini tidak terdefinisi di sana —
    mengembalikan nol akan menyamarkan hal itu menjadi seolah tidak ada simpangan.
    """
    if tugas not in TUGAS_SPIRAL:
        return np.full(len(xy), np.nan)
    sisa, _ = residual_spiral(xy)
    return np.abs(_lowpass(sisa, BATAS_LAMBAT, fs))


def penanda_cepat_bisp(accel: np.ndarray, fs: int) -> np.ndarray:
    """Penanda tremor NewHandPD, dihitung dari kanal tilt/akselerasi yang DITAHAN.

    Sepadan dengan `penanda_cepat` pada basis data utama, dan mempertahankan
    sifat yang membuat keselarasan bermakna: penanda dihitung dari kanal yang
    **tidak pernah diberikan kepada model** (Subbab 3.3.2). Pada UCI 395 peran
    itu dipegang koordinat; pada NewHandPD dipegang kanal 4 sampai 6.

    Filter diterapkan pada komponen **bertanda** tiap sumbu, baru selubungnya
    digabung sebagai norma — alasannya sama persis dengan `selubung_pita_vektor`,
    dan mengabaikannya pernah menghancurkan justru komponen yang hendak diukur.

    Rasio terhadap pita gerakan volunter dipakai karena tugas spiral melibatkan
    gerakan yang disengaja, sebagaimana SST dan DST pada basis data utama.
    """
    tremor = np.linalg.norm(
        np.stack([selubung_pita(accel[:, c], PITA_TREMOR, fs)
                  for c in range(accel.shape[1])], axis=1), axis=1)
    volunter = np.linalg.norm(
        np.stack([selubung_pita(accel[:, c], PITA_VOLUNTER, fs)
                  for c in range(accel.shape[1])], axis=1), axis=1)
    return tremor / np.maximum(volunter, 1e-9)


def penanda_cadangan(xy: np.ndarray, fs: int) -> np.ndarray:
    """Jerk ternormalisasi, tingkat sampel. Penanda cadangan Subbab 3.6.4."""
    v = np.diff(xy, axis=0, prepend=xy[:1]) * fs
    a = np.diff(v, axis=0, prepend=v[:1]) * fs
    j = np.diff(a, axis=0, prepend=a[:1]) * fs
    besar = np.linalg.norm(j, axis=1)
    return besar / np.maximum(np.median(besar), 1e-9)


def energi_total(xy: np.ndarray, fs: int) -> np.ndarray:
    """Energi gerak total per sampel, sebagai perancu yang harus dikendalikan.

    Subbab 3.6.6 menuntut korelasi terhadap energi total dilaporkan, karena
    atribusi dapat sekadar mengikuti besarnya gerakan alih-alih tremornya.
    """
    return laju_gerak(xy, fs) ** 2


def mask_pena_melayang(tekanan: np.ndarray) -> np.ndarray:
    """True pada sampel ketika pena tidak menyentuh layar."""
    return tekanan <= 0


def ke_grid_patch(nilai: np.ndarray, patch_size: int, n_patch: int | None = None,
                  bobot: np.ndarray | None = None) -> np.ndarray:
    """Ringkas besaran tingkat sampel ke grid patch model.

    Parameters
    ----------
    bobot : bila diberikan, hanya sampel berbobot yang dirata-ratakan. Dipakai
        untuk membatasi penanda STCP pada segmen pena melayang. Patch yang tidak
        memuat satu pun sampel berbobot diberi nilai NaN, bukan nol, agar dapat
        dikeluarkan dari korelasi alih-alih ikut sebagai nilai palsu.
    """
    n = len(nilai) // patch_size if n_patch is None else n_patch
    n = min(n, len(nilai) // patch_size)
    if n == 0:
        return np.zeros(0)
    v = nilai[: n * patch_size].reshape(n, patch_size)
    if bobot is None:
        return v.mean(axis=1)
    w = bobot[: n * patch_size].reshape(n, patch_size).astype(float)
    jumlah = w.sum(axis=1)
    keluar = np.full(n, np.nan)
    sah = jumlah > 0
    keluar[sah] = (v[sah] * w[sah]).sum(axis=1) / jumlah[sah]
    return keluar
