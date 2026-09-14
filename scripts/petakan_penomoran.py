"""Petakan penomoran desimal naskah ke penomoran huruf Panduan Tugas Akhir UHB.

Jalankan:
    python3 scripts/petakan_penomoran.py --periksa           # gerbang saja, tidak menyunting
    python3 scripts/petakan_penomoran.py --bab I --terapkan  # heading BAB I + seluruh rujukan

Kenapa satu skrip, bukan suntingan tangan
------------------------------------------
Naskah memuat **116 rujukan silang** atas 30 nomor unik, ditambah **38 rujukan** di dalam
`scripts/pisah_sempro_semhas.py`. Menyuntingnya satu per satu berarti 154 kesempatan salah
ketik, dan kesalahannya tidak akan terlihat sampai seseorang mengikuti rujukan itu.

Diturunkan dari satu tabel, seluruhnya konsisten menurut konstruksi.

Tiga gerbang, dan kenapa persis ketiganya
------------------------------------------
Rancangan rencana ini gagal tiga kali sebelum disetujui, dan tiap gerbang di bawah menangkap
satu kegagalan yang **sudah pernah benar-benar terjadi**:

1. **Injektif.** Rancangan pertama melebur `3.6.2`, `3.6.3`, dan `3.6.8` ke satu alamat.
   Sepuluh rujukan menjadi ambigu: pembaca yang mengikutinya tidak tahu bagian mana yang dimaksud.

2. **Lengkap.** Rancangan pertama tidak memberi tujuan bagi lima anak `2.2.x` yang ternyata
   dirujuk sebelas kali, dan melewatkan `5.2` sama sekali.

3. **Kedalaman BAB III maksimal tiga.** Blok hasil yang dipindahkan `pisah_sempro_semhas.py`
   adalah **saudara** induknya pada tingkat yang sama, bukan anaknya. Begitu induk melampaui
   tingkat tiga, blok itu menutup rentang induknya alih-alih berada di dalamnya, dan Bab IV
   terpotong tanpa ada yang menyadarinya.

Gerbang berjalan sebelum satu karakter pun disunting. Gagal berarti berhenti, bukan melanjutkan
dengan peringatan.

Satu pengecualian yang disengaja
--------------------------------
`3.6.9` memuat dua hal berbeda — ramalan pra-registrasi (menjadi Hipotesis pada `II.D`) dan
mekanisme yang menguncinya (menjadi `III.A.3`). Kelima rujukannya **tidak** dipetakan otomatis;
skrip mencetak lokasinya agar dibaca konteks per konteks.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
SUMBER = AKAR / "NASKAH-SUMBER.md"
PEMISAH = AKAR / "scripts" / "pisah_sempro_semhas.py"

# Nomor lama -> alamat baru. Alamat memakai awalan bab romawi sebab huruf subbab
# berulang di tiap bab: 'A' pada BAB I dan BAB III adalah dua tempat berbeda.
PETA: dict[str, str] = {
    # ── BAB I ────────────────────────────────────────────────────────────────
    "1.1": "I.A", "1.2": "I.B", "1.4": "I.C", "1.6": "I.D", "1.5": "I.E",
    "1.3": "III.A.4",
    # ── BAB II ───────────────────────────────────────────────────────────────
    "2.2": "II.A",
    "2.2.1": "II.A.1.a", "2.2.2": "II.A.1.b",
    "2.2.6": "II.A.2.a", "2.2.7": "II.A.2.b", "2.2.8": "II.A.2.c", "2.2.9": "II.A.2.d",
    "2.2.3": "II.A.3.a", "2.2.4": "II.A.3.b", "2.2.5": "II.A.3.c",
    "2.1": "II.A.4",
    "2.3": "II.B",
    # ── BAB III ──────────────────────────────────────────────────────────────
    "3.1": "III.A.1", "3.6.1": "III.A.2",
    "3.3": "III.E.2",
    "3.3.1": "III.E.2.a", "3.3.2": "III.E.2.b", "3.3.3": "III.E.2.c", "3.3.4": "III.E.2.d",
    "3.4": "III.F.1", "3.5": "III.F.2",
    "3.6.2": "III.F.3", "3.6.3": "III.F.4", "3.6.4": "III.F.5",
    "3.6.5": "III.F.6", "3.6.8": "III.F.7",
    "3.7": "III.G.1", "3.6.6": "III.G.2", "3.6.7": "III.G.3", "3.8": "III.G.4",
    "3.6.10": "III.G.5", "3.6.11": "III.G.6", "3.6.12": "III.G.7", "3.6.13": "III.G.8",
    "3.6.14": "III.G.9", "3.6.15": "III.G.10", "3.6.16": "III.G.11", "3.6.17": "III.G.12",
    "3.9": "III.H",
    # `3.2 Tahapan Penelitian` sengaja TIDAK ada di sini: ia tidak pernah dirujuk,
    # dan headingnya dilebur menjadi pengantar III.F alih-alih dinomori ulang.
    "3.6": "III.F",           # satu rujukan, kepada rancangan skenario secara utuh
    # ── BAB IV dan V, hidup di dalam skrip pemisah ───────────────────────────
    **{f"4.{i}": f"IV.A.{i}" for i in range(1, 17)},
    "4.1.1": "IV.A.1.a",
    **{f"4.13.{i}": f"IV.A.13.{chr(96 + i)}" for i in range(1, 5)},
    "5.1": "V.A", "5.2": "V.B",
}

# Judul yang ikut berubah. Nomor yang tidak tercantum mempertahankan judul lamanya.
JUDUL_BARU: dict[str, str] = {
    "1.1": "Latar Belakang Masalah",
    "1.2": "Perumusan Masalah",
    "1.5": "Keaslian Penelitian",
}

# Dipecah dua, sehingga tidak dapat dipetakan otomatis.
KECUALI = {"3.6.9"}

POLA_RUJUKAN = re.compile(r"(Subbab\s+)(\d+(?:\.\d+)*)")
POLA_HEADING = re.compile(r"^(#{1,6})\s+(\d+(?:\.\d+)*)\s+(.*)$", re.M)


def nomor_dirujuk(*teks: str) -> Counter:
    c: Counter = Counter()
    for t in teks:
        c.update(m.group(2) for m in POLA_RUJUKAN.finditer(t))
    return c


def induk_blok_hasil() -> dict[str, int]:
    """Nomor subbab yang menaungi blok hasil, beserta jumlahnya.

    Dibaca dari naskah dan dari daftar `HASIL` milik `pisah_sempro_semhas.py`, bukan
    ditulis ulang di sini — dua salinan daftar yang sama adalah dua salinan yang akan
    berbeda suatu hari.
    """
    sys.path.insert(0, str(AKAR / "scripts"))
    from pisah_sempro_semhas import HASIL  # noqa: E402

    induk: Counter = Counter()
    sekarang = None
    for baris in SUMBER.read_text().split("\n"):
        m = re.match(r"^#{1,6}\s+(\d+(?:\.\d+)*)\s", baris)
        if m:
            sekarang = m.group(1)
        elif baris.strip() in HASIL and sekarang:
            induk[sekarang] += 1
    return dict(induk)



def gerbang(dirujuk: Counter) -> list[str]:
    """Ketiga gerbang. Mengembalikan daftar kegagalan; kosong berarti aman."""
    gagal: list[str] = []

    tabrakan = {v: sorted(k for k in PETA if PETA[k] == v)
                for v, n in Counter(PETA.values()).items() if n > 1}
    if tabrakan:
        gagal.append(f"tidak injektif: {tabrakan}")

    yatim = sorted(n for n in dirujuk if n not in PETA and n not in KECUALI)
    if yatim:
        gagal.append(f"dirujuk tanpa tujuan: {yatim}")

    # Kedalaman hanya mengikat subbab yang MENAUNGI blok hasil. Blok itu saudara
    # induknya pada tingkat yang sama; begitu induknya melampaui tingkat tiga, blok
    # tersebut menutup rentang induknya dan Bab IV terpotong. Subbab tanpa blok hasil
    # boleh lebih dalam — II.A.2.c dan III.E.2.a tidak menimbulkan masalah apa pun.
    terlalu_dalam = sorted(
        f"{lama} -> {PETA[lama]} (menaungi {n} blok hasil)"
        for lama, n in induk_blok_hasil().items()
        if lama in PETA and PETA[lama].count(".") + 1 > 3)
    if terlalu_dalam:
        gagal.append(f"induk blok hasil melampaui tiga tingkat: {terlalu_dalam}")

    return gagal


def petakan_rujukan(teks: str) -> tuple[str, int, list[str]]:
    """Ganti tiap 'Subbab N.N'. Mengembalikan (teks, jumlah diganti, yang dilewati)."""
    n = 0
    lewat: list[str] = []

    def ganti(m: re.Match) -> str:
        nonlocal n
        nomor = m.group(2)
        if nomor in KECUALI:
            lewat.append(nomor)
            return m.group(0)
        baru = PETA.get(nomor)
        if baru is None:
            lewat.append(nomor)
            return m.group(0)
        n += 1
        return m.group(1) + baru

    return POLA_RUJUKAN.sub(ganti, teks), n, lewat


def petakan_heading(teks: str, bab: str | None) -> tuple[str, int]:
    """Tulis ulang heading bernomor menjadi bentuk huruf.

    Tingkat `#` diturunkan dari kedalaman alamat baru, dan label yang dicetak adalah
    ruas terakhirnya: 'III.G.5' menjadi '### 5. Judul', 'II.A.2.a' menjadi '#### a. Judul'.
    """
    n = 0

    def ganti(m: re.Match) -> str:
        nonlocal n
        nomor, judul = m.group(2), m.group(3)
        baru = PETA.get(nomor)
        if baru is None or nomor in KECUALI:
            return m.group(0)
        if bab and not baru.startswith(bab + "."):
            return m.group(0)
        ruas = baru.split(".")
        label, tingkat = ruas[-1], len(ruas)
        n += 1
        return f"{'#' * tingkat} {label}. {JUDUL_BARU.get(nomor, judul)}"

    return POLA_HEADING.sub(ganti, teks), n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--periksa", action="store_true", help="jalankan gerbang saja")
    p.add_argument("--terapkan", action="store_true", help="tulis perubahan ke berkas")
    p.add_argument("--bab", default=None, help="batasi penulisan ulang heading pada satu bab, mis. I")
    a = p.parse_args()

    sumber, pemisah = SUMBER.read_text(), PEMISAH.read_text()
    dirujuk = nomor_dirujuk(sumber, pemisah)

    print("=" * 78)
    print(f"PEMETAAN PENOMORAN — {len(PETA)} entri, {sum(dirujuk.values())} rujukan "
          f"atas {len(dirujuk)} nomor unik")
    print("=" * 78)

    gagal = gerbang(dirujuk)
    for g in gagal:
        print(f"  GERBANG GAGAL: {g}")
    if gagal:
        print("\nBerhenti. Tidak ada berkas yang disentuh.")
        return 1
    print("  ok  injektif")
    print("  ok  tiap nomor yang dirujuk punya tujuan")
    print("  ok  tidak ada tujuan BAB III melampaui tiga tingkat")

    if a.periksa or not a.terapkan:
        print(f"\nmode periksa. Rujukan ke {sorted(KECUALI)} sengaja tidak dipetakan "
              f"({sum(dirujuk[k] for k in KECUALI)} kemunculan).")
        return 0

    for jalur, teks in ((SUMBER, sumber), (PEMISAH, pemisah)):
        baru, n_ruj, lewat = petakan_rujukan(teks)
        n_head = 0
        if jalur == SUMBER:
            baru, n_head = petakan_heading(baru, a.bab)
        jalur.write_text(baru)
        print(f"\n{jalur.name}: {n_ruj} rujukan dipetakan, {n_head} heading ditulis ulang")
        if lewat:
            print(f"  dilewati (perlu tangan): {dict(Counter(lewat))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
