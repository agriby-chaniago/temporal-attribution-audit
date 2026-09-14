"""Bangkitkan blok DAFTAR ISI dari heading naskah itu sendiri.

Jalankan:
    python3 scripts/bangun_daftar_isi.py            # cetak, tidak menulis
    python3 scripts/bangun_daftar_isi.py --terapkan # tulis ke NASKAH-SUMBER.md

Kenapa dibangkitkan, bukan ditulis tangan
------------------------------------------
Daftar isi yang ditulis tangan adalah salinan kedua dari struktur naskah, dan dua salinan
yang sama adalah dua salinan yang akan berbeda suatu hari. `audit_sempro.py` sumbu [5]
memang membandingkan keduanya, tetapi mencegah lebih murah daripada menangkap.

Dibangkitkan dari heading, daftar isi tidak mungkin melenceng menurut konstruksi.

Dibangkitkan per dokumen
------------------------
Sempro dan semhas memuat halaman depan yang berbeda: `PERNYATAAN KEASLIAN` dan
`KATA PENGANTAR` ditandai khusus naskah Tugas Akhir oleh panduan, sehingga keduanya
dibuang dari sempro. Daftar isi yang dibangkitkan sekali di naskah sumber karena itu akan
menyebut halaman yang tidak ada pada salah satu dokumen — dan itulah yang terjadi pada
percobaan pertama.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
DOKUMEN = [AKAR / "draftSempro" / "sempro-skripsi.md",
           AKAR / "draftSemhas" / "semhas-skripsi.md"]

# Lebar kolom judul bab sebelum daftar subbabnya menjorok, mengikuti Lampiran 13.
JOROK = " " * 9


def bangun(teks: str) -> str:
    baris: list[str] = []
    for l in teks.split("\n"):
        m1 = re.match(r"^# (.+)$", l)
        m2 = re.match(r"^## ([A-Z])\. (.+)$", l)
        if m1:
            judul = m1.group(1).strip()
            if judul in ("PROPOSAL SKRIPSI", "SKRIPSI"):
                baris += ["", "HALAMAN JUDUL"]     # sampul, mengikuti Lampiran 13
                continue
            if judul == "DAFTAR ISI":
                # Lampiran 13 mencantumkan DAFTAR ISI di dalam daftar isi itu sendiri,
                # meski BAB III C.h berbunyi seolah halaman sebelumnya tidak perlu dimuat.
                # Contoh nyata dipakai sebagai penengah ketika panduan berbunyi dua arah.
                baris += ["", "DAFTAR ISI"]
                continue
            if judul.startswith("BAB "):
                nomor, _, nama = judul.partition(". ")
                baris += ["", f"{nomor:<8s} {nama}"]
            else:
                # LAMPIRAN berderet tanpa baris kosong di antaranya, seperti Lampiran 13.
                rapat = judul.startswith("LAMPIRAN") and baris and baris[-1].startswith("LAMPIRAN")
                baris += ([] if rapat else [""]) + [judul.replace(". ", "  ", 1)]
        elif m2:
            baris.append(f"{JOROK}{m2.group(1)}.  {m2.group(2)}")
    return "\n".join(x for x in "\n".join(baris).strip().split("\n"))


def main() -> int:
    for jalur in DOKUMEN:
        if not jalur.exists():
            continue
        teks = jalur.read_text()
        isi = bangun(teks)
        pola = re.compile(r"(# DAFTAR ISI\n\n```\n)(.*?)(\n```)", re.S)
        if not pola.search(teks):
            print(f"  {jalur.name}: blok DAFTAR ISI tidak ditemukan")
            continue
        jalur.write_text(pola.sub(lambda m: m.group(1) + isi + m.group(3), teks, count=1))
        print(f"  {jalur.name}: daftar isi {len(isi.splitlines())} baris")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
