"""
Prapemrosesan dan rekayasa kanal untuk UCI 395.

Mengikuti Subbab 3.4 naskah proposal. Keputusan utamanya: koordinat absolut
TIDAK dipakai sebagai masukan model, karena posisi absolut mengkodekan tata
letak template dan posisi duduk subjek, bukan manifestasi penyakit. Model hanya
menerima kanal dinamika.

Koordinat absolut tetap dipertahankan di berkas cache karena penanda klinis pada
Subbab 3.6.4 dihitung dari sinyal mentah termasuk koordinat. Yang dibatasi hanya
akses model, bukan akses analisis.

Temuan data yang menentukan rancangan modul ini
-----------------------------------------------
1. Frekuensi sampling sesungguhnya ~127,5 Hz, bukan 142,86 Hz. Angka 142,86
   berasal dari median delta timestamp, yang menyesatkan karena distribusinya
   bimodal: 57,5 persen bernilai 7 ms dan 39,9 persen bernilai 9 ms. Rata-rata
   delta 7,842 ms adalah ukuran yang benar.

2. Timestamp tidak seragam, sehingga resampling ke grid seragam wajib dilakukan
   sebelum turunan apa pun dihitung. Tanpa itu, kecepatan dan percepatan
   mengandung derau yang berasal dari jitter pencatatan waktu, bukan dari gerak.

3. Empat belas dari 202 rekaman mengandung lompatan timestamp mundur, sebagian
   besar pada Test ID 2. Pemeriksaan menunjukkan lompatan ini adalah batas antara
   dua sesi perekaman yang tersambung dalam satu berkas. Tiap potongan kontinu
   diperlakukan sebagai rekaman tersendiri, bukan disambung, karena penyambungan
   akan menciptakan transien palsu tepat pada besaran yang diukur.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

__all__ = [
    "FS_ASLI", "FS_TARGET", "DURASI_MIN_DETIK", "TEKANAN_MAKS",
    "NAMA_TUGAS", "KANAL_MODEL",
    "Rekaman", "muat_berkas", "pecah_segmen", "ke_grid_seragam",
    "rekayasa_kanal", "muat_dataset", "Normalisasi",
    "simpan_cache", "muat_cache",
]

# Frekuensi asli, dihitung dari rata-rata delta timestamp seluruh berkas.
FS_ASLI = 127.52

# Frekuensi target. Dipilih 100 Hz karena: satu sampel tepat 10 ms sehingga
# pemetaan ukuran patch ke durasi menjadi lugas; Nyquist 50 Hz jauh di atas
# batas atas pita tremor 7,5 Hz (Subbab 2.2.1); dan merupakan penurunan ringan
# dari 127,5 Hz sehingga tidak ada interpolasi ke atas yang mengarang data.
FS_TARGET = 100

# Durasi minimum segmen yang dipertahankan. Diturunkan dari pita tremor, bukan
# ditetapkan sembarang: tremor terendah 3,5 Hz berarti satu siklus 286 ms, dan
# estimasi amplitudo pita yang stabil membutuhkan sekitar sepuluh siklus.
DURASI_MIN_DETIK = 2.86

# Skala tekanan perangkat, dari dokumentasi resmi UCI 395 (0 sampai 1023).
# Konstanta perangkat, bukan statistik data, sehingga aman dipakai tanpa
# menimbulkan kebocoran antar fold.
TEKANAN_MAKS = 1023.0

NAMA_TUGAS = {0: "SST", 1: "DST", 2: "STCP"}

# Kanal yang diberikan kepada model. Koordinat absolut sengaja tidak termasuk.
KANAL_MODEL = ["dx", "dy", "kecepatan", "percepatan", "jerk", "tekanan"]


@dataclass
class Rekaman:
    """Satu segmen kontinu dari satu subjek pada satu tugas."""

    subjek: str
    kelompok: str          # "PD" atau "HC"
    tugas: int             # 0=SST, 1=DST, 2=STCP
    segmen: int            # indeks segmen dalam rekaman asli
    fs: float
    kanal: np.ndarray      # (T, 6) sesuai KANAL_MODEL, belum dinormalisasi
    xy: np.ndarray | None = None            # (T, 2) koordinat absolut, penanda klinis saja
    tekanan_mentah: np.ndarray | None = None  # (T,) skala 0-1023, deteksi pena melayang
    # Kanal yang SENGAJA ditahan dari model dan hanya dipakai menghitung penanda.
    # Pada UCI 395 perannya dipegang xy; pada NewHandPD, kanal tilt/akselerasi
    # (Subbab 3.3.2). Medan ini ditaruh terakhir agar urutan dataclass tetap sah
    # dan jalur UCI tidak berubah sama sekali.
    kanal_ditahan: np.ndarray | None = None

    @property
    def label(self) -> int:
        return 1 if self.kelompok == "PD" else 0

    @property
    def durasi(self) -> float:
        return len(self.kanal) / self.fs

    def meta(self) -> dict:
        d = asdict(self)
        for k in ("kanal", "xy", "tekanan_mentah", "kanal_ditahan"):
            d.pop(k, None)
        d["durasi"] = self.durasi
        d["label"] = self.label
        d["n_sampel"] = len(self.kanal)
        d["nama_tugas"] = NAMA_TUGAS[self.tugas]
        return d


def muat_berkas(path: Path) -> np.ndarray:
    """Baca satu berkas UCI 395 menjadi array (N, 7).

    Kolom: x, y, z, tekanan, sudut_grip, timestamp, test_id.
    """
    baris = [b.strip().split(";") for b in open(path) if b.strip()]
    return np.array([[float(v) for v in r] for r in baris], dtype=np.float64)


def pecah_segmen(t: np.ndarray) -> list[np.ndarray]:
    """Pecah indeks menjadi potongan-potongan dengan timestamp menaik.

    Lompatan mundur menandai batas antar sesi perekaman yang tersambung dalam
    satu berkas. Potongan tidak disambung karena penyambungan akan menciptakan
    transien palsu.
    """
    batas = np.where(np.diff(t) <= 0)[0]
    return [p for p in np.split(np.arange(len(t)), batas + 1) if len(p) > 0]


def ke_grid_seragam(t_ms: np.ndarray, nilai: np.ndarray, fs_target: int = FS_TARGET) -> np.ndarray:
    """Interpolasi ke grid waktu seragam, lalu turunkan ke frekuensi target.

    Dua tahap dan bukan satu, agar penurunan frekuensi melewati filter anti-alias
    yang benar. Tahap pertama meregularkan jitter timestamp pada frekuensi asli;
    tahap kedua menurunkannya ke frekuensi target lewat resample_poly yang sudah
    menyertakan filter anti-alias.

    Parameters
    ----------
    t_ms : (N,) timestamp dalam milidetik, harus menaik
    nilai : (N, C) nilai yang diinterpolasi

    Returns
    -------
    (M, C) pada frekuensi target
    """
    durasi = (t_ms[-1] - t_ms[0]) / 1000.0
    n_seragam = max(int(round(durasi * FS_ASLI)), 2)
    grid = np.linspace(t_ms[0], t_ms[-1], n_seragam)
    seragam = np.column_stack([np.interp(grid, t_ms, nilai[:, c]) for c in range(nilai.shape[1])])

    # 127,52 -> 100 Hz. Pembulatan rasio ke 25/32 (setara 128 -> 100) menjaga
    # kesalahan frekuensi di bawah 0,4 persen, jauh lebih kecil daripada lebar
    # pita tremor yang dianalisis.
    return resample_poly(seragam, up=25, down=32, axis=0)


def rekayasa_kanal(xy: np.ndarray, tekanan: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """Bentuk enam kanal masukan model dari koordinat dan tekanan.

    Seluruhnya invarian terhadap translasi: tidak ada satu pun kanal yang
    membawa posisi absolut, sesuai keputusan utama Subbab 3.4.

    Turunan dihitung setelah resampling, saat selang waktu sudah konstan,
    sehingga tidak mencampurkan jitter pencatatan waktu ke dalam kecepatan.
    """
    dt = 1.0 / fs
    dxy = np.diff(xy, axis=0, prepend=xy[:1])
    v_vec = dxy / dt
    a_vec = np.diff(v_vec, axis=0, prepend=v_vec[:1]) / dt
    j_vec = np.diff(a_vec, axis=0, prepend=a_vec[:1]) / dt

    return np.column_stack([
        dxy[:, 0],
        dxy[:, 1],
        np.linalg.norm(v_vec, axis=1),
        np.linalg.norm(a_vec, axis=1),
        np.linalg.norm(j_vec, axis=1),
        tekanan / TEKANAN_MAKS,
    ]).astype(np.float32)


def _daftar_berkas(akar: Path) -> list[tuple[Path, str, str]]:
    """Kumpulkan (path, id_subjek, kelompok) dari ketiga folder UCI 395."""
    hasil = []
    for sub, kelompok in [("hw_dataset/control", "HC"),
                          ("hw_dataset/parkinson", "PD"),
                          ("new_dataset/parkinson", "PD")]:
        for f in sorted((akar / sub).glob("*.txt")):
            # Nama berkas unik lintas folder, dipakai langsung sebagai id subjek.
            hasil.append((f, f.stem, kelompok))
    return hasil


def muat_dataset(akar: Path, fs_target: int = FS_TARGET,
                 durasi_min: float = DURASI_MIN_DETIK) -> list[Rekaman]:
    """Muat seluruh UCI 395 menjadi daftar Rekaman siap pakai.

    Urutan pemrosesan: pisah per tugas, pecah di lompatan waktu mundur,
    resampling ke grid seragam, baru rekayasa kanal.
    """
    rekaman = []
    for path, subjek, kelompok in _daftar_berkas(akar):
        data = muat_berkas(path)
        for tugas in sorted(np.unique(data[:, 6]).astype(int)):
            blok = data[data[:, 6] == tugas]
            for i, potongan in enumerate(pecah_segmen(blok[:, 5])):
                s = blok[potongan]
                if len(s) < 4:  # tidak cukup untuk turunan ketiga
                    continue
                if (s[-1, 5] - s[0, 5]) / 1000.0 < durasi_min:
                    continue

                grid = ke_grid_seragam(s[:, 5], np.column_stack([s[:, 0], s[:, 1], s[:, 3]]), fs_target)
                if len(grid) < 4:
                    continue

                xy, tekanan = grid[:, :2], np.clip(grid[:, 2], 0, TEKANAN_MAKS)
                rekaman.append(Rekaman(
                    subjek=subjek, kelompok=kelompok, tugas=tugas, segmen=i,
                    fs=float(fs_target),
                    kanal=rekayasa_kanal(xy, tekanan, fs_target),
                    xy=xy.astype(np.float32),
                    tekanan_mentah=tekanan.astype(np.float32),
                ))
    return rekaman


class Normalisasi:
    """Penskalaan robust per kanal, dipasang HANYA pada rekaman fold latih.

    Memakai median dan rentang antar-kuartil, bukan rata-rata dan simpangan
    baku, karena kanal turunan mengandung pencilan ekstrem: 0,08 persen langkah
    melampaui 50 piksel per 10 milidetik, dengan puncak 721 piksel. Kecepatan
    setara 72.000 piksel per detik itu tidak mungkin merupakan gerak tangan.
    Pemeriksaan menunjukkan lompatan tersebut 3,4 kali lebih sering terjadi saat
    pena melayang, sehingga berasal dari pena keluar-masuk jangkauan sensor.

    Pencilan tetap dipertahankan di berkas cache agar cache setia pada sumber.
    Pengekangan hanya dilakukan di sini, sebagai keputusan pemodelan yang
    diterapkan identik pada kedua arsitektur.

    Statistik dipasang hanya pada fold latih. Memasangnya pada seluruh data akan
    membocorkan informasi fold uji ke dalam penskalaan, persis jenis kebocoran
    yang hendak dicegah protokol validasi pada Subbab 3.6.1.
    """

    def __init__(self, batas_kuantil: float = 0.999):
        self.batas_kuantil = batas_kuantil
        self.tengah_: np.ndarray | None = None
        self.skala_: np.ndarray | None = None
        self.batas_: np.ndarray | None = None

    def fit(self, rekaman_latih: list[Rekaman]) -> "Normalisasi":
        X = np.concatenate([r.kanal for r in rekaman_latih])
        self.tengah_ = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        # Kanal konstan (IQR nol) diberi skala satu agar tidak membagi nol.
        self.skala_ = np.where(iqr > 1e-8, iqr, 1.0)
        self.batas_ = np.percentile(np.abs(X - self.tengah_), self.batas_kuantil * 100, axis=0)
        return self

    def transform(self, kanal: np.ndarray) -> np.ndarray:
        if self.tengah_ is None:
            raise RuntimeError("Normalisasi belum dipasang; panggil fit() pada rekaman fold latih.")
        selisih = np.clip(kanal - self.tengah_, -self.batas_, self.batas_)
        return (selisih / self.skala_).astype(np.float32)


def simpan_cache(rekaman: list[Rekaman], path: Path) -> None:
    """Simpan hasil prapemrosesan ke satu berkas npz.

    Kanal disimpan tanpa normalisasi, karena statistik penskalaan harus dipasang
    per fold latih saat pelatihan. Menyimpan versi ternormalisasi akan membekukan
    kebocoran antar fold ke dalam artefak.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    simpanan: dict[str, np.ndarray] = {}
    meta = []
    for i, r in enumerate(rekaman):
        simpanan[f"kanal_{i}"] = r.kanal
        simpanan[f"xy_{i}"] = r.xy
        simpanan[f"tekanan_{i}"] = r.tekanan_mentah
        meta.append((r.subjek, r.kelompok, r.tugas, r.segmen, r.fs))
    simpanan["meta"] = np.array(meta, dtype=object)
    np.savez_compressed(path, **simpanan)


def muat_cache(path: Path) -> list[Rekaman]:
    """Baca kembali berkas cache menjadi daftar Rekaman."""
    with np.load(path, allow_pickle=True) as z:
        meta = z["meta"]
        return [
            Rekaman(
                subjek=str(m[0]), kelompok=str(m[1]), tugas=int(m[2]),
                segmen=int(m[3]), fs=float(m[4]),
                kanal=z[f"kanal_{i}"], xy=z[f"xy_{i}"], tekanan_mentah=z[f"tekanan_{i}"],
            )
            for i, m in enumerate(meta)
        ]
