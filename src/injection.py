"""
Injeksi sinyal tremor sintetis untuk kontrol positif (Subbab 3.6.2).

Prosedurnya: ambil rekaman kontrol sehat, suntikkan osilasi pada pita tremor di
lokasi waktu yang ditentukan peneliti, lalu periksa apakah pipeline menemukan
lokasi itu kembali. Kalau tidak, pipeline mengandung kesalahan — dan itu
diketahui sebelum data pasien diproses.

Dua keputusan rancangan yang menentukan kesahihan kontrol ini
-------------------------------------------------------------
1. Injeksi dilakukan pada **koordinat**, bukan pada kanal turunan. Tremor
   sungguhan adalah osilasi posisi; kecepatan, percepatan, dan jerk mewarisi
   osilasi itu secara konsisten lewat turunan. Menyuntik langsung ke kanal
   turunan akan menghasilkan kanal yang tidak saling konsisten, sehingga model
   dapat mengenalinya lewat ketidakkonsistenan itu, bukan lewat tremornya.

2. Amplop injeksi **melandai halus** di kedua ujung, bukan hidup-mati mendadak.
   Tepi mendadak menciptakan transien berpita lebar yang jauh lebih mudah
   dideteksi daripada tremor itu sendiri, sehingga kontrol positifnya akan
   lulus karena alasan yang salah.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from preprocessing import Rekaman, rekayasa_kanal

__all__ = ["HasilInjeksi", "amplop_landai", "suntik_tremor"]


@dataclass
class HasilInjeksi:
    """Rekaman tersuntik beserta kebenaran acuan lokasi suntikan."""

    rekaman: Rekaman
    mask: np.ndarray        # (T,) bobot amplop 0..1, kebenaran acuan tingkat sampel
    amplitudo: float        # piksel
    frekuensi: float        # Hz
    awal: int               # indeks sampel awal jendela injeksi
    akhir: int              # indeks sampel akhir jendela injeksi

    def mask_patch(self, patch_size: int) -> np.ndarray:
        """Kebenaran acuan pada grid patch, sejajar dengan peta alpha dan phi.

        Nilai tiap patch adalah rata-rata amplop di dalamnya, sehingga patch yang
        hanya sebagian tertutup jendela injeksi mendapat bobot menengah, bukan
        nol atau satu.
        """
        n = len(self.mask) // patch_size
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        return self.mask[: n * patch_size].reshape(n, patch_size).mean(axis=1).astype(np.float32)


def amplop_landai(panjang: int, awal: int, akhir: int, landai: float = 0.25) -> np.ndarray:
    """Amplop persegi dengan tepi kosinus (jendela Tukey) pada rentang [awal, akhir).

    Pelandaian mencegah tepi mendadak yang akan menciptakan transien berpita
    lebar. Lihat catatan rancangan nomor 2 pada docstring modul.
    """
    amplop = np.zeros(panjang, dtype=np.float64)
    lebar = akhir - awal
    if lebar <= 0:
        return amplop

    n_landai = max(int(lebar * landai / 2), 1)
    inti = np.ones(lebar)
    tepi = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_landai)))
    inti[:n_landai] = tepi
    inti[-n_landai:] = tepi[::-1]
    amplop[awal:akhir] = inti
    return amplop


def suntik_tremor(
    rek: Rekaman,
    amplitudo: float,
    frekuensi: float = 5.0,
    porsi: float = 0.3,
    rng: np.random.Generator | None = None,
) -> HasilInjeksi:
    """Suntikkan osilasi tremor sintetis pada satu jendela waktu.

    Parameters
    ----------
    rek : rekaman sumber, biasanya dari subjek kontrol sehat
    amplitudo : amplitudo osilasi dalam piksel
    frekuensi : Hz, standarnya 5 Hz yaitu tengah pita tremor 3,5-7,5 Hz (Subbab 2.2.1)
    porsi : proporsi durasi rekaman yang disuntik
    rng : generator acak; posisi jendela, arah osilasi, dan fase diambil darinya

    Returns
    -------
    HasilInjeksi berisi rekaman baru dan kebenaran acuan lokasinya.
    """
    rng = rng or np.random.default_rng()
    T = len(rek.xy)
    lebar = max(int(T * porsi), 2)
    awal = int(rng.integers(0, max(T - lebar, 1)))
    akhir = min(awal + lebar, T)

    amplop = amplop_landai(T, awal, akhir)
    t = np.arange(T) / rek.fs
    fase = rng.uniform(0, 2 * np.pi)
    sudut = rng.uniform(0, 2 * np.pi)  # arah osilasi, tetap sepanjang jendela
    arah = np.array([np.cos(sudut), np.sin(sudut)])

    osilasi = amplitudo * np.sin(2 * np.pi * frekuensi * t + fase) * amplop
    xy_baru = rek.xy + osilasi[:, None] * arah[None, :]

    baru = replace(
        rek,
        xy=xy_baru.astype(np.float32),
        kanal=rekayasa_kanal(xy_baru, rek.tekanan_mentah, int(rek.fs)),
    )
    return HasilInjeksi(
        rekaman=baru, mask=amplop.astype(np.float32), amplitudo=amplitudo,
        frekuensi=frekuensi, awal=awal, akhir=akhir,
    )
