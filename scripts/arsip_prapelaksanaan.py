"""Arsip prapelaksanaan beserta manifes sidik jari.

Jalankan: python3 scripts/arsip_prapelaksanaan.py

Kenapa perlu, dan kenapa tidak cukup menyalin `results/`
---------------------------------------------------------
Proyek ini **bukan repositori git**. Tidak ada `git checkout`, tidak ada riwayat,
tidak ada diff. Rangkaian analisis yang akan dijalankan mengubah `src/newhandpd.py`
dan menulis ke `results/` dari tujuh skrip baru. Tanpa arsip, satu kesalahan menimpa
hasil yang dibayar berjam-jam komputasi GPU dan tidak dapat dipulihkan.

Cadangan sebelumnya hanya menyalin `results/` — kini berada di
`cadangan/usang_20260906_1208/cadangan/hasil_20260902_2118/`.
Cakupan itu tidak cukup kali ini sebab `src/` ikut berubah.

Manifes, dan kenapa dia yang paling berharga
--------------------------------------------
Ketentuan verifikasi berbunyi "tidak ada angka lama yang berubah". Dengan 80 berkas
di `results/`, ketentuan itu tidak akan benar-benar diperiksa bila caranya
membandingkan angka dengan mata.

`MANIFES.sha256` membuat pemeriksaan itu mekanis: setelah tiap tahap, satu perintah
memperlihatkan persis berkas mana yang berubah. Berkas yang berubah padahal tahap itu
tidak menulisnya adalah bug, dan ketahuan seketika.

Cara memakainya sesudah sebuah tahap:

    cd results && sha256sum -c ../cadangan/<arsip>/MANIFES.sha256 2>&1 | grep -v ': OK$'

Baris yang tersisa adalah berkas yang berubah atau hilang.

Aturan penamaan direktori di dalam `cadangan/`
----------------------------------------------
Ditetapkan 6 September 2026 setelah dua cadangan sempat bersanding tanpa pembeda,
dan salah satunya memuat sepuluh berkas hasil yang sudah tidak berlaku:

- `prapelaksanaan_<stempel>/` — jaring pengaman **berlaku**, dibuat berkas ini
- `usang_<stempel>/` — sudah **tergantikan**, dibuat `scripts/arsipkan_usang.py`

Cadangan tanpa salah satu awalan itu dianggap belum diklasifikasi. Status tiap
cadangan tercatat di `cadangan/INDEKS.md`.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
CADANGAN = AKAR / "cadangan"

# Direktori yang disalin utuh. `data/` (1,5 G) dan `.venv/` (5,7 G) sengaja
# dikecualikan: keduanya dapat dibangun ulang dan tidak pernah ditulis oleh analisis.
DIREKTORI = ["results", "notebooks", "figures", "src", "scripts",
             "naskah", "laporan"]

# Berkas akar yang ikut. Naskah sumber dan dosier sendiri tinggal di `naskah/`,
# yang sudah disalin utuh lewat DIREKTORI di atas.
POLA_BERKAS = ["*.md", "*.txt"]

# Manifes dihitung hanya untuk direktori ini, sebab hanya di sinilah angka tinggal.
DIR_MANIFES = "results"


def sidik(path: Path, blok: int = 1 << 20) -> str:
    """SHA-256 sebuah berkas, dibaca per blok agar artefak 40 MB tidak dimuat utuh."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while potongan := f.read(blok):
            h.update(potongan)
    return h.hexdigest()


def ukuran_dir(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    stempel = datetime.now().strftime("%Y%m%d_%H%M")
    tujuan = CADANGAN / f"prapelaksanaan_{stempel}"
    if tujuan.exists():
        print(f"GAGAL: {tujuan} sudah ada — arsip tidak ditimpa.", file=sys.stderr)
        return 1
    tujuan.mkdir(parents=True)

    baris_ringkas, total = [], 0
    for nama in DIREKTORI:
        asal = AKAR / nama
        if not asal.is_dir():
            baris_ringkas.append(f"{nama:14s} LEWAT (tidak ada)")
            continue
        shutil.copytree(asal, tujuan / nama)
        n = sum(1 for f in (tujuan / nama).rglob("*") if f.is_file())
        b = ukuran_dir(tujuan / nama)
        total += b
        baris_ringkas.append(f"{nama:14s} {n:5d} berkas  {b/1e6:8.2f} MB")

    akar_tujuan = tujuan / "_akar"
    akar_tujuan.mkdir()
    n_akar = 0
    for pola in POLA_BERKAS:
        for f in sorted(AKAR.glob(pola)):
            shutil.copy2(f, akar_tujuan / f.name)
            total += f.stat().st_size
            n_akar += 1
    baris_ringkas.append(f"{'_akar':14s} {n_akar:5d} berkas")

    # Manifes: dihitung dari SUMBER, bukan dari salinan, sehingga ia mencatat keadaan
    # results/ yang sesungguhnya pada saat arsip dibuat.
    sumber_manifes = AKAR / DIR_MANIFES
    berkas = sorted(f for f in sumber_manifes.rglob("*") if f.is_file())
    manifes = tujuan / "MANIFES.sha256"
    with open(manifes, "w") as f:
        for b in berkas:
            f.write(f"{sidik(b)}  {b.relative_to(sumber_manifes)}\n")

    ringkas = tujuan / "RINGKAS.txt"
    with open(ringkas, "w") as f:
        f.write(f"Arsip prapelaksanaan\nDibuat  : {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Sumber  : {AKAR}\nTotal   : {total/1e6:.2f} MB\n")
        f.write(f"Manifes : {len(berkas)} berkas di {DIR_MANIFES}/\n\n")
        f.write("\n".join(baris_ringkas))
        f.write("\n\nPeriksa perubahan sesudah sebuah tahap:\n")
        f.write(f"  cd {DIR_MANIFES} && sha256sum -c {manifes} 2>&1 | grep -v ': OK$'\n")

    print(f"arsip  : {tujuan}")
    print(f"total  : {total/1e6:.2f} MB")
    print(f"manifes: {len(berkas)} berkas di {DIR_MANIFES}/")
    for baris in baris_ringkas:
        print("  " + baris)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
