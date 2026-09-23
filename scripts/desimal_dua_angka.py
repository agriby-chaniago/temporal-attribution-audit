"""Lengkapi angka desimal menjadi sekurang-kurangnya dua angka di belakang koma.

Jalankan:
    python3 scripts/desimal_dua_angka.py --pratinjau
    python3 scripts/desimal_dua_angka.py --terapkan

Aturan panduan dan cara membacanya
-----------------------------------
Panduan BAB III E butir 9: "Penulisan angka desimal menggunakan tanda koma ( , ) dan dua
angka dibelakang koma, misal: 2,01". Kalimat itu tidak memuat kata "minimal" maupun
"tepat", sehingga terbuka dua bacaan.

Yang dipakai naskah ini adalah **minimal dua angka**. Bacaan "tepat dua angka" tidak dapat
dipakai: ia mengubah `p = 0,0001` menjadi `0,00` dan `0,0062` menjadi `0,01`, yakni
menghapus seluruh pelaporan statistik. Bacaan "minimal" hanya menuntut angka berdesimal
satu dilengkapi, dan itulah yang dikerjakan berkas ini.

Satu akibat yang dinyatakan terus terang
-----------------------------------------
Pita tremor 3,5 sampai 7,5 Hz berasal dari literatur klinis dengan presisi satu desimal.
Menuliskannya 3,50 sampai 7,50 Hz memenuhi aturan format tetapi memberi kesan presisi
seperseratus yang tidak dimiliki definisinya. Hal itu dicatat pada Lampiran D agar
pembaca tidak salah menyimpulkan ketelitian sumbernya.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
SUMBER = AKAR / "naskah" / "NASKAH-SUMBER.md"
# Prosa Bab IV dan V tinggal di dalam skrip pemisah, bukan di naskah sumber, sehingga ia
# harus ikut diproses — kalau tidak, semhas berdesimal campur sementara sempro seragam.
PEMISAH = AKAR / "scripts" / "pisah_sempro_semhas.py"

# Angka berdesimal tepat satu, di luar rangkaian angka lain.
POLA = re.compile(r"(?<![\d,.])(\d+),(\d)(?![\d.])")

# Jangan sentuh kode, nama berkas, tautan, maupun penanda komentar.
LINDUNG = re.compile(
    r"```.*?```"
    r"|`[^`]*`"
    r"|<!--.*?-->"
    r"|https?://\S+"
    r"|\b[\w./-]+\.(?:py|md|csv|pkl|json|ipynb|png|bib)\b",
    re.S)


def olah(teks: str, terapkan: bool) -> tuple[str, Counter]:
    rekap: Counter = Counter()

    def satu(bagian: str) -> str:
        def ganti(m: re.Match) -> str:
            rekap[f"{m.group(1)},{m.group(2)}"] += 1
            return f"{m.group(1)},{m.group(2)}0" if terapkan else m.group(0)
        return POLA.sub(ganti, bagian)

    keluar, i = [], 0
    for m in LINDUNG.finditer(teks):
        keluar.append(satu(teks[i:m.start()]))
        keluar.append(m.group(0))
        i = m.end()
    keluar.append(satu(teks[i:]))
    return "".join(keluar), rekap


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pratinjau", action="store_true")
    p.add_argument("--terapkan", action="store_true")
    a = p.parse_args()

    teks = SUMBER.read_text()
    i = teks.rfind("# DAFTAR PUSTAKA")
    j = teks.find("\n# ", i + 1)
    j = j if j > 0 else len(teks)
    badan, pustaka, ekor = teks[:i], teks[i:j], teks[j:]

    b, r1 = olah(badan, a.terapkan)
    e, r2 = olah(ekor, a.terapkan)
    rekap = r1 + r2
    print(f"angka berdesimal satu: {sum(rekap.values())}")
    print("paling sering:", rekap.most_common(8))

    if a.terapkan:
        SUMBER.write_text(b + pustaka + e)
        print(f"ditulis: {SUMBER.name}")

    s, r3 = olah(PEMISAH.read_text(), a.terapkan)
    print(f"prosa Bab IV-V di skrip pemisah: {sum(r3.values())} angka")
    if a.terapkan and sum(r3.values()):
        PEMISAH.write_text(s)
        print(f"ditulis: {PEMISAH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
