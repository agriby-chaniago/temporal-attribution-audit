"""Pemuat NewHandPD untuk Skenario S5, dengan pemisahan kanal yang menjaga penanda tetap independen.

Mengikuti Subbab 3.3.2. Enam kanal sensor smart pen BiSP adalah mikrofon,
fingergrip, tekanan aksial isi pena, serta tilt dan akselerasi arah X, Y, Z.

Keputusan rancangan yang menentukan kesahihan seluruh skenario
--------------------------------------------------------------
Pada basis data utama, penanda klinis dihitung dari koordinat yang **sengaja
tidak diberikan kepada model**. Itulah yang membuat keselarasan yang terukur
tidak dapat dijelaskan sebagai model membaca ulang masukannya sendiri.

Rancangan awal replikasi merusak sifat itu tanpa disadari, karena penandanya
hendak dihitung dari kanal tilt/akselerasi yang juga menjadi masukan model.
Pemisahan berikut memulihkannya:

- **Kanal 1 sampai 3 diberikan kepada model** (mikrofon, fingergrip, tekanan
  aksial), direkayasa menjadi enam kanal lewat beda pertama dan kedua.
- **Kanal 4 sampai 6 ditahan sepenuhnya** dan hanya dipakai menghitung penanda.

Sah karena diperiksa lebih dahulu: kanal 1 sampai 3 membawa sinyal setara atau
lebih kuat (AUC 0,831, 0,819, dan 0,742) dibanding kanal yang ditahan (0,758,
0,795, 0,719), sehingga penahanan itu tidak melumpuhkan model.

Resampling
----------
NewHandPD sudah berada pada grid seragam 1000 Hz dan tidak memiliki kolom
timestamp, sehingga cukup `resample_poly(up=1, down=10)`. `ke_grid_seragam` pada
`preprocessing.py` **tidak dipakai dan tidak boleh diubah**: ia mengunci
FS_ASLI dan rasio 25/32 milik UCI 395, dan seluruh jalur beku bergantung padanya.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from preprocessing import Rekaman

__all__ = ["FS_ASLI_BISP", "FS_TARGET_BISP", "KANAL_MODEL_BISP", "TUGAS_SPIRAL_BISP",
           "TUGAS_MEANDER_BISP", "TUGAS_DIA_BISP", "KELUARGA_BISP", "META_AMAN",
           "rekayasa_kanal_bisp", "muat_berkas_bisp", "baca_meta_bisp", "muat_newhandpd",
           "simpan_cache_bisp", "muat_cache_bisp"]

FS_ASLI_BISP = 1000
FS_TARGET_BISP = 100                 # sama dengan basis data utama, agar patch sepadan
KANAL_MODEL_BISP = ["d_mikrofon", "d_fingergrip", "d_tekanan",
                    "dd_mikrofon", "dd_fingergrip", "dd_tekanan"]
TUGAS_SPIRAL_BISP = {0: "sigSp1", 1: "sigSp2", 2: "sigSp3", 3: "sigSp4"}

# Delapan tugas lain sudah ikut terunduh tetapi tidak pernah dipakai Skenario S5.
# Mereka membuka sumbu geser TUGAS dengan perangkat, negara, dan subjek terkunci
# konstan — sesuatu yang tidak dapat diberikan kohort baru mana pun. Subjeknya
# sama, sehingga mereka BUKAN kohort ketiga dan tidak boleh dipakai menambah
# pasangan retensi lintas kohort.
TUGAS_MEANDER_BISP = {0: "sigMea1", 1: "sigMea2", 2: "sigMea3", 3: "sigMea4"}
TUGAS_DIA_BISP = {0: "sigDiaA", 1: "sigDiaB"}

# Satu-satunya kolom header yang boleh keluar ke berkas hasil. Sisanya
# (Surename, Forename, Notice, Person_ID_Number, ...) mengidentifikasi orang.
META_AMAN = ["Age", "Gender", "Writing_Hand", "Weight", "Height", "Smoker",
             "Pen", "Samplerate"]

KELUARGA_BISP = {"spiral": ("sigSp*.txt", TUGAS_SPIRAL_BISP),
                 "meander": ("sigMea*.txt", TUGAS_MEANDER_BISP),
                 "dia": ("sigDia*.txt", TUGAS_DIA_BISP)}

_POLA_NAMA = re.compile(r"(sigSp\d|sigMea\d|sigDia[AB])-([HP]\d+)\.txt$")

# Satu berkas pada rilis NewHandPD kehilangan awalan kelompok pada namanya:
# `sigMea1-28.txt`, sehingga sigMea1 hanya punya 65 berkas sementara sigMea2
# sampai sigMea4 punya 66. Identitasnya dipulihkan lewat `Person_ID_Number` di
# header meta, yang cocok persis dan hanya dengan H28. Alias ini dicatat di sini
# alih-alih dibiarkan dijatuhkan diam-diam oleh regex.
_ALIAS_BERKAS = {"sigMea1-28.txt": "sigMea1-H28.txt"}
_POLA_META = re.compile(r"^#<(\w+)>(.*?)</\1>$")


def rekayasa_kanal_bisp(sensor: np.ndarray) -> np.ndarray:
    """Bentuk enam kanal model dari tiga sensor yang boleh dilihat model.

    Memakai beda pertama dan kedua, mengikuti prinsip `rekayasa_kanal` pada
    basis data utama: seluruhnya invarian terhadap taraf searah, sehingga tidak
    ada kanal yang membawa nilai mutlak sensor. Ini penting karena taraf searah
    sensor BiSP berbeda antar sesi perekaman dan bukan besaran fisiologis.
    """
    d1 = np.diff(sensor, axis=0, prepend=sensor[:1])
    d2 = np.diff(d1, axis=0, prepend=d1[:1])
    return np.column_stack([d1, d2]).astype(np.float32)


def muat_berkas_bisp(path: Path) -> np.ndarray:
    """Baca satu berkas sinyal NewHandPD menjadi array (N, 6), header dilewati."""
    return np.loadtxt(path, comments="#")


def baca_meta_bisp(path: Path) -> dict[str, str]:
    """Baca header meta yang selama ini dibuang oleh `comments="#"`.

    Tiap berkas memuat `Person_ID_Number`, `Age`, `Gender`, `Writing_Hand`, dan
    `Weight`. `Person_ID_Number` memungkinkan identitas subjek diperiksa lintas
    tugas; sisanya membuka audit perancu demografis pada NewHandPD, yang sampai
    kini hanya dikerjakan pada basis data utama.
    """
    PERINGATAN = None  # lihat catatan di bawah
    meta = {}
    with open(path) as f:
        for baris in f:
            if not baris.startswith("#"):
                break
            m = _POLA_META.match(baris.strip())
            if m:
                meta[m.group(1)] = m.group(2)
    return meta


def muat_newhandpd(akar: Path, durasi_min: float = 2.86,
                   keluarga: str = "spiral") -> list[Rekaman]:
    """Muat rekaman NewHandPD satu keluarga tugas sebagai daftar Rekaman.

    `keluarga` bawaannya "spiral" sehingga seluruh angka Skenario S5 yang sudah
    ada tidak berubah satu digit pun. Pilihan lain: "meander" dan "dia".

    `kanal` berisi enam kanal turunan sensor 1 sampai 3; `kanal_ditahan` berisi
    tilt/akselerasi 4 sampai 6 yang tidak pernah diberikan kepada model.
    `xy` dan `tekanan_mentah` tetap None, sebab NewHandPD tidak memuat keduanya.
    """
    if keluarga not in KELUARGA_BISP:
        raise ValueError(f"keluarga tak dikenal: {keluarga!r}; pilih {sorted(KELUARGA_BISP)}")
    pola_glob, peta_tugas = KELUARGA_BISP[keluarga]

    akar = Path(akar)
    berkas = []
    for sub, kelompok in [("extracted/Signal", "HC"), ("extracted_patient/Signal", "PD")]:
        for f in sorted((akar / sub).glob(pola_glob)):
            if f.name.startswith("._"):          # resource fork macOS
                continue
            m = _POLA_NAMA.search(_ALIAS_BERKAS.get(f.name, f.name))
            if m:
                berkas.append((f, m.group(1), m.group(2), kelompok))

    nama_ke_tugas = {v: k for k, v in peta_tugas.items()}
    rekaman = []
    for f, nama_tugas, subjek, kelompok in berkas:
        d = muat_berkas_bisp(f)
        if d.ndim != 2 or d.shape[1] != 6:
            raise ValueError(f"{f.name}: diharapkan 6 kolom, ditemukan {d.shape}")
        turun = resample_poly(d, up=1, down=FS_ASLI_BISP // FS_TARGET_BISP, axis=0)
        if len(turun) / FS_TARGET_BISP < durasi_min:
            continue
        rekaman.append(Rekaman(
            subjek=subjek, kelompok=kelompok,
            tugas=nama_ke_tugas[nama_tugas], segmen=0, fs=FS_TARGET_BISP,
            kanal=rekayasa_kanal_bisp(turun[:, :3]),
            kanal_ditahan=turun[:, 3:6].astype(np.float32),
        ))
    return rekaman


def simpan_cache_bisp(rekaman: list[Rekaman], path: Path) -> None:
    """Cache tersendiri, tidak menumpang format cache UCI 395."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    simpanan, meta = {}, []
    for i, r in enumerate(rekaman):
        simpanan[f"kanal_{i}"] = r.kanal
        simpanan[f"ditahan_{i}"] = r.kanal_ditahan
        meta.append((r.subjek, r.kelompok, r.tugas, r.fs))
    np.savez_compressed(path, meta=np.array(meta, dtype=object), **simpanan)


def muat_cache_bisp(path: Path) -> list[Rekaman]:
    z = np.load(path, allow_pickle=True)
    return [Rekaman(subjek=str(m[0]), kelompok=str(m[1]), tugas=int(m[2]), segmen=0,
                    fs=float(m[3]), kanal=z[f"kanal_{i}"], kanal_ditahan=z[f"ditahan_{i}"])
            for i, m in enumerate(z["meta"])]
