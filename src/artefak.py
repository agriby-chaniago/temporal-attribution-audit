"""Pemuatan artefak yang dapat ditambah arm baru tanpa melatih ulang yang lama.

Sebelum modul ini, setiap builder memuat artefak utuh lalu melewati pelatihan
sepenuhnya. Menambah arsitektur ketiga karena itu menuntut dua arm lama dilatih
ulang tanpa guna, dan berisiko menimpa hasil yang sudah dibayar berjam-jam
komputasi.

Dua pengaman yang wajib dipakai bersama:

1. **Tulis atomik.** Artefak ditulis ke berkas sementara lalu diganti namanya.
   Proses yang terhenti di tengah karena itu tidak dapat meninggalkan artefak
   separuh jadi di tempat artefak yang baik.

2. **Gabung, bukan timpa.** Yang dihitung hanya arm yang belum ada. Arm lama
   dikembalikan apa adanya dari berkas, sehingga angkanya dijamin tidak berubah
   satu digit pun.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = ["muat_artefak", "simpan_atomik", "arm_ada", "arm_hilang", "lapor_arm"]


def muat_artefak(path: Path) -> Any | None:
    """Muat artefak bila ada, None bila belum pernah dibuat."""
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def simpan_atomik(path: Path, obj: Any) -> None:
    """Simpan lewat berkas sementara lalu ganti nama, agar tidak pernah separuh jadi."""
    path = Path(path)
    sementara = path.with_name(path.name + ".tmp")
    with open(sementara, "wb") as f:
        pickle.dump(obj, f)
    os.replace(sementara, path)


def arm_ada(art: Any, ambil: Callable[[Any], str] | None = None) -> set[str]:
    """Himpunan arsitektur yang sudah ada di dalam artefak.

    Menangani ketiga bentuk artefak yang dipakai penelitian ini:

    - daftar berisi dict dengan kunci ``arsitektur`` (S2, S3)
    - dict berkunci tuple yang unsur pertamanya arsitektur (S6, S7)
    - dict berisi daftar-daftar yang unsurnya punya kunci ``arsitektur`` (S4, S8)
    """
    if art is None:
        return set()
    if ambil is not None:
        return {ambil(x) for x in art}

    ditemukan: set[str] = set()
    if isinstance(art, dict):
        for kunci, nilai in art.items():
            if isinstance(kunci, tuple) and kunci and isinstance(kunci[0], str):
                ditemukan.add(kunci[0])
            elif isinstance(nilai, list):
                ditemukan |= {d["arsitektur"] for d in nilai
                              if isinstance(d, dict) and "arsitektur" in d}
    elif isinstance(art, list):
        ditemukan |= {d["arsitektur"] for d in art
                      if isinstance(d, dict) and "arsitektur" in d}
    return ditemukan


def arm_hilang(art: Any, diminta: Iterable[str],
               ambil: Callable[[Any], str] | None = None) -> list[str]:
    """Arm yang diminta tetapi belum ada di artefak, urutannya dipertahankan."""
    ada = arm_ada(art, ambil)
    return [a for a in diminta if a not in ada]


def lapor_arm(nama_berkas: str, ada: Iterable[str], hilang: Iterable[str]) -> None:
    """Cetak apa yang dipakai ulang dan apa yang akan dihitung, sebelum menghitung."""
    ada, hilang = sorted(ada), list(hilang)
    if ada:
        print(f"{nama_berkas}: memakai ulang {', '.join(ada)} — tidak dilatih ulang")
    if hilang:
        print(f"{nama_berkas}: menghitung {', '.join(hilang)}")
    else:
        print(f"{nama_berkas}: lengkap, tidak ada yang perlu dihitung")
