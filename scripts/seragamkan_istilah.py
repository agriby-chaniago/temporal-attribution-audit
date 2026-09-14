"""Seragamkan istilah teknis dan cetak miring istilah asing, sesuai Panduan UHB.

Jalankan:
    python3 scripts/seragamkan_istilah.py --pratinjau  # cetak rencana, tanpa menulis
    python3 scripts/seragamkan_istilah.py --terapkan   # ubah NASKAH-SUMBER.md
    python3 scripts/seragamkan_istilah.py --balik      # buang cetak miring saja

Dua pekerjaan, satu berkas
--------------------------
**Pertama, istilah teknis dikembalikan ke bentuk aslinya.** Naskah sempat memakai padanan
Indonesia yang justru mengaburkan: `lipatan` untuk *fold*, `derau` untuk *noise*. Garis
pemisahnya: yang merupakan **nama** sesuatu dikembalikan ke bahasa Inggris; kata kerja dan
kata sifat Indonesia biasa dibiarkan. `bergerbang` pada "rekurensi bergerbang" karena itu
tetap, sebab "rekurensi gated" membaca lebih buruk, bukan lebih jelas.

**Kedua, istilah asing dicetak miring.** Panduan BAB III D.3.b: "Penulisan istilah asing
tersebut harus dicetak miring." Tidak ada pengecualian kemunculan pertama di dalam teks
panduan; pengecualian itu konvensi jurnal, bukan aturan di sini. Penguji yang keberatan
pada terlalu banyak miring sedang berselisih selera; penguji yang menemukan istilah asing
tidak miring sedang menunjuk pelanggaran. Karena itu seluruh kemunculan dimiringkan, dan
`--balik` disediakan agar keputusan itu dapat dicabut dalam satu perintah.

Kata yang sengaja TIDAK disentuh
---------------------------------
`merugikan`, `dirugikan` — bukan *loss*, melainkan kata Indonesia biasa.
`kelipatan` — bukan *fold*, melainkan "berapa kali lipat".
`gerbang konservatif` — gerbang keputusan, bukan gate arsitektur.
Ketiganya pernah menjadi calon penggantian pada rancangan datar, dan ketiganya akan rusak
bila penggantian dilakukan tanpa melihat konteks.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
SUMBER = AKAR / "NASKAH-SUMBER.md"

# ── Bagian satu: padanan yang dikembalikan ke bentuk aslinya ────────────────
# (pola, pengganti, keterangan). Pola memakai batas kata agar turunan seperti
# `kelipatan` dan `merugikan` tidak ikut tertangkap.
GANTI: list[tuple[str, str, str]] = [
    (r"\blipatan\b", "fold", "fold"),
    (r"\bLipatan\b", "Fold", "fold"),
    (r"\blapisan\b", "layer", "layer"),
    (r"\bderau\b", "noise", "noise"),
    (r"\bDerau\b", "Noise", "noise"),
    (r"\bhanyutan\b", "drift", "drift"),
    (r"\bpenskalaan\b", "scaling", "scaling"),
    (r"\bPenskalaan\b", "Scaling", "scaling"),
    (r"\bpenyetelan\b", "tuning", "tuning"),
    (r"\bPenyetelan\b", "Tuning", "tuning"),
    (r"\brugi\b", "loss", "loss"),
    (r"\bRugi\b", "Loss", "loss"),
    # `latih` dan `uji` hanya bila bermakna bagian data. `uji permutasi`,
    # `uji Welch`, dan `daya uji` adalah kosakata statistik Indonesia dan tetap.
    (r"\b(fold|data|himpunan|sisi|bagian)\s+latih\b", r"\1 train", "train"),
    (r"\b(fold|data|himpunan|sisi|bagian)\s+uji\b", r"\1 test", "test"),
    # Gate arsitektur, bukan gerbang keputusan. Ketiga konteksnya disebut eksplisit.
    (r"gerbang yang mengatur pembaruan hidden state", "gate yang mengatur pembaruan hidden state", "gate"),
    (r"Gerbang pembaruan GRU", "Gate pembaruan GRU", "gate"),
    (r"dari gerbang ke peluruhan", "dari gate ke peluruhan", "gate"),
]

# ── Bagian dua: istilah asing yang dicetak miring ───────────────────────────
MIRING = [
    "attention pooling", "patch embedding", "state space", "hidden state",
    "encoder", "baseline", "patch", "seed", "fold", "epoch", "batch",
    "dropout", "logit", "state", "train", "test", "tuning", "scaling",
    "noise", "drift", "loss", "layer", "gate",
]

# Konteks yang tidak boleh disentuh: blok kode, kode sebaris, jalur berkas,
# penanda komentar, dan teks yang sudah miring atau tebal.
LINDUNG = re.compile(
    r"```.*?```"                     # blok kode
    r"|`[^`]*`"                      # kode sebaris
    r"|<!--.*?-->"                   # komentar penanda
    r"|\*\*[^*]*\*\*"                # tebal
    r"|\*[^*\n]*\*"                  # sudah miring
    r"|!\[[^\]]*\]\([^)]*\)"         # gambar
    r"|\b[\w./-]+\.(?:py|md|csv|pkl|json|ipynb|png|bib)\b"   # nama berkas
    # Heading: judul miring salah secara tipografi, dan judul blok hasil dipakai skrip
    # pemisah sebagai jangkar persis. `[^\n]*` bukan `.*` — dengan re.S titik menelan
    # baris baru, sehingga seluruh naskah sesudah heading pertama ikut terlindungi.
    r"|^#{1,6} [^\n]*",
    re.S | re.M)


def peta_miring(teks: str) -> tuple[str, int]:
    """Miringkan tiap istilah asing di luar konteks terlindung."""
    pola = re.compile(r"(?<![\w*])(" + "|".join(
        sorted((re.escape(x) for x in MIRING), key=len, reverse=True)) + r")(?![\w*])", re.I)
    n = 0

    def olah(bagian: str) -> str:
        nonlocal n

        def ganti(m: re.Match) -> str:
            nonlocal n
            n += 1
            return f"*{m.group(1)}*"

        return pola.sub(ganti, bagian)

    keluar, i = [], 0
    for m in LINDUNG.finditer(teks):
        keluar.append(olah(teks[i:m.start()]))
        keluar.append(m.group(0))
        i = m.end()
    keluar.append(olah(teks[i:]))
    return "".join(keluar), n


def buang_miring(teks: str) -> tuple[str, int]:
    pola = re.compile(r"\*(" + "|".join(
        sorted((re.escape(x) for x in MIRING), key=len, reverse=True)) + r")\*", re.I)
    n = len(pola.findall(teks))
    return pola.sub(r"\1", teks), n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pratinjau", action="store_true")
    p.add_argument("--terapkan", action="store_true")
    p.add_argument("--balik", action="store_true")
    a = p.parse_args()

    teks = SUMBER.read_text()
    i = teks.rfind("# DAFTAR PUSTAKA")
    j = teks.find("\n# ", i + 1)
    j = j if j > 0 else len(teks)
    badan, pustaka, ekor = teks[:i], teks[i:j], teks[j:]

    if a.balik:
        badan, n1 = buang_miring(badan)
        ekor, n2 = buang_miring(ekor)
        SUMBER.write_text(badan + pustaka + ekor)
        print(f"cetak miring dibuang: {n1 + n2}")
        return 0

    rekap: dict[str, int] = {}
    for pola, pengganti, nama in GANTI:
        c = len(re.findall(pola, badan)) + len(re.findall(pola, ekor))
        if c:
            rekap[nama] = rekap.get(nama, 0) + c
        if a.terapkan:
            badan = re.sub(pola, pengganti, badan)
            ekor = re.sub(pola, pengganti, ekor)

    print("padanan yang dikembalikan:")
    for k, v in sorted(rekap.items(), key=lambda x: -x[1]):
        print(f"  {k:10s} {v:3d}")
    print(f"  TOTAL      {sum(rekap.values()):3d}")

    if a.terapkan:
        badan, m1 = peta_miring(badan)
        ekor, m2 = peta_miring(ekor)
        SUMBER.write_text(badan + pustaka + ekor)
        print(f"\ncetak miring: {m1 + m2} kemunculan")
        print(f"ditulis: {SUMBER.name}")
    else:
        _, m1 = peta_miring(badan)
        _, m2 = peta_miring(ekor)
        print(f"\ncetak miring (perkiraan): {m1 + m2} kemunculan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
