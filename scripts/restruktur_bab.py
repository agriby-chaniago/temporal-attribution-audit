"""Bongkar `NASKAH-SUMBER.md` menjadi blok bernomor, lalu susun ulang pada urutan panduan.

Jalankan:
    python3 scripts/restruktur_bab.py --uji        # bongkar-pasang identik? wajib lolos dulu
    python3 scripts/restruktur_bab.py --pratinjau  # kerangka hasil, tanpa menulis
    python3 scripts/restruktur_bab.py --terapkan   # tulis NASKAH-SUMBER.md

Kenapa skrip, bukan suntingan tangan
------------------------------------
Restrukturisasi memindahkan blok besar: `3.3 Data Penelitian` sendirian 160 baris, dan
`1.3 Batasan Masalah` harus berpindah bab. Memindahkannya dengan tangan berarti belasan
operasi potong-tempel yang kegagalannya senyap — satu baris tertinggal di tempat lama tidak
akan terlihat sampai seseorang membaca ulang seluruh naskah.

Syarat keamanan yang membuat sisanya dapat dipercaya
-----------------------------------------------------
`--uji` membongkar naskah menjadi blok lalu menyusunnya kembali **pada urutan aslinya**, dan
menuntut hasilnya identik bita per bita dengan berkas asli. Bila bongkar-pasang saja sudah
mengubah sesuatu, penyusunan ulang pada urutan berbeda tidak dapat dipercaya sama sekali.

Uji itu wajib lolos sebelum `--terapkan` boleh dijalankan, dan skrip menegakkannya sendiri.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
SUMBER = AKAR / "NASKAH-SUMBER.md"

POLA_HEADING = re.compile(r"^(#{1,6})\s+(\d+(?:\.\d+)*)\s+(.*)$")


class Potongan:
    """Satu heading beserta isinya sampai heading berikutnya, tingkat apa pun.

    Model sengaja **datar**, bukan bersarang. Percobaan pertama memakai model
    "pembuka + blok bernomor + penutup" dan langsung gugur pada uji bongkar-pasang:
    heading `# BAB II. TINJAUAN PUSTAKA` tidak bernomor, sehingga ia tertelan sebagai
    batas blok sebelumnya lalu tidak pernah dikeluarkan lagi. Dengan model datar tidak
    ada celah yang mungkin hilang — tiap baris naskah milik tepat satu potongan.
    """

    def __init__(self, tingkat: int, nomor: str | None, judul: str, baris: list[str]):
        self.tingkat, self.nomor, self.judul, self.baris = tingkat, nomor, judul, baris

    def __repr__(self) -> str:
        n = self.nomor or "-"
        return f"<{n} {self.judul[:34]!r} {len(self.baris)} baris>"


POLA_APA_PUN = re.compile(r"^(#{1,6})\s+(.*)$")


def bongkar(teks: str) -> tuple[list[str], list[Potongan]]:
    """Pisahkan naskah menjadi (pembuka, potongan). Pembuka = sebelum heading pertama."""
    baris = teks.split("\n")
    awal = [i for i, b in enumerate(baris) if POLA_APA_PUN.match(b)]
    if not awal:
        return baris, []

    potongan: list[Potongan] = []
    for k, i in enumerate(awal):
        m = POLA_APA_PUN.match(baris[i])
        assert m
        batas = awal[k + 1] if k + 1 < len(awal) else len(baris)
        h = POLA_HEADING.match(baris[i])
        potongan.append(Potongan(
            tingkat=len(m.group(1)),
            nomor=h.group(2) if h else None,
            judul=h.group(3) if h else m.group(2),
            baris=baris[i:batas],
        ))
    return baris[:awal[0]], potongan


def pasang(pembuka: list[str], potongan: list[Potongan]) -> str:
    keluar = list(pembuka)
    for p in potongan:
        keluar += p.baris
    return "\n".join(keluar)


def uji_bongkar_pasang(teks: str) -> tuple[bool, str]:
    pembuka, potongan = bongkar(teks)
    ulang = pasang(pembuka, potongan)
    if ulang == teks:
        bernomor = sum(1 for p in potongan if p.nomor)
        return True, (f"{len(potongan)} potongan ({bernomor} bernomor), "
                      f"{len(pembuka)} baris pembuka")
    a, b = teks.split("\n"), ulang.split("\n")
    for i, (x, y) in enumerate(zip(a, b), 1):
        if x != y:
            return False, f"beda pada baris {i}:\n  asli  : {x[:90]!r}\n  ulang : {y[:90]!r}"
    return False, f"panjang berbeda: asli {len(a)} baris, ulang {len(b)} baris"


# ── Susunan tujuan ───────────────────────────────────────────────────────────
# Tiap entri salah satu dari:
#   ("bab",    judul)                    heading tingkat 1, mis. "BAB I. PENDAHULUAN"
#   ("ambil",  nomor, alamat, judul)     pindahkan potongan yang ada, tulis ulang headingnya
#   ("gabung", nomor)                    ambil ISInya saja, headingnya dibuang
#   ("baru",   alamat, judul, isi)       subbab yang belum ada; isi diisi tahap berikutnya
#
# Urutan daftar ini ADALAH urutan naskah hasil. Tidak ada penyortiran di belakang layar:
# apa yang tertulis di sini persis apa yang keluar, sehingga dapat diperiksa dengan mata.
RINTIS = "> *Ditulis pada langkah berikutnya.*"

SUSUNAN: list[tuple] = [
    ("bab", "BAB I. PENDAHULUAN"),
    ("ambil", "1.1", "I.A", "Latar Belakang Masalah"),
    ("ambil", "1.2", "I.B", "Perumusan Masalah"),
    ("ambil", "1.4", "I.C", "Tujuan Penelitian"),
    ("ambil", "1.6", "I.D", "Manfaat Penelitian"),
    ("ambil", "1.5", "I.E", "Keaslian Penelitian"),

    ("bab", "BAB II. TINJAUAN PUSTAKA"),
    ("ambil", "2.2", "II.A", "Tinjauan Teori"),
    ("baru", "II.A.1", "Penyakit Parkinson dan tulisan tangan daring", RINTIS),
    ("ambil", "2.2.1", "II.A.1.a", None),
    ("ambil", "2.2.2", "II.A.1.b", None),
    ("baru", "II.A.2", "Variabel terikat: lokalisasi temporal dan kualitasnya", RINTIS),
    ("ambil", "2.2.6", "II.A.2.a", None),
    ("ambil", "2.2.7", "II.A.2.b", None),
    ("ambil", "2.2.8", "II.A.2.c", None),
    ("ambil", "2.2.9", "II.A.2.d", None),
    ("baru", "II.A.3", "Variabel bebas: encoder yang dipertukarkan", RINTIS),
    ("ambil", "2.2.3", "II.A.3.a", None),
    ("ambil", "2.2.4", "II.A.3.b", None),
    ("ambil", "2.2.5", "II.A.3.c", None),
    ("ambil", "2.1", "II.A.4", "Keterkaitan antarvariabel pada penelitian terdahulu"),
    ("ambil", "2.3", "II.B", "Kerangka Teori"),
    ("baru", "II.C", "Kerangka Konsep", RINTIS),
    ("baru", "II.D", "Hipotesis", RINTIS),

    ("bab", "BAB III. METODE PENELITIAN"),
    ("baru", "III.A", "Jenis dan Rancangan Penelitian", RINTIS),
    ("ambil", "3.1", "III.A.1", "Jenis Penelitian"),
    ("ambil", "3.6.1", "III.A.2", "Protokol validasi"),
    ("ambil", "3.6.9", "III.A.3", "Pra-registrasi dan pemisahan konfirmatori dan eksploratori"),
    ("ambil", "1.3", "III.A.4", "Ruang lingkup dan batasan rancangan"),
    ("baru", "III.B", "Lokasi dan Waktu Penelitian", RINTIS),
    ("baru", "III.C", "Variabel Penelitian", RINTIS),
    ("baru", "III.D", "Definisi Operasional Variabel", RINTIS),
    ("baru", "III.E", "Alat dan Bahan", RINTIS),
    ("baru", "III.E.1", "Alat", RINTIS),
    ("ambil", "3.3", "III.E.2", "Bahan"),
    ("ambil", "3.3.1", "III.E.2.a", None),
    ("ambil", "3.3.2", "III.E.2.b", None),
    ("ambil", "3.3.3", "III.E.2.c", None),
    ("ambil", "3.3.4", "III.E.2.d", None),
    ("ambil", "3.2", "III.F", "Prosedur Penelitian"),
    ("gabung", "3.6"),
    ("ambil", "3.4", "III.F.1", None),
    ("ambil", "3.5", "III.F.2", None),
    ("ambil", "3.6.2", "III.F.3", None),
    ("ambil", "3.6.3", "III.F.4", None),
    ("ambil", "3.6.4", "III.F.5", None),
    ("ambil", "3.6.5", "III.F.6", None),
    ("ambil", "3.6.8", "III.F.7", None),
    ("baru", "III.G", "Analisis Data", RINTIS),
    ("ambil", "3.7", "III.G.1", "Metrik Evaluasi"),
    ("ambil", "3.6.6", "III.G.2", None),
    ("ambil", "3.6.7", "III.G.3", None),
    ("ambil", "3.8", "III.G.4", "Skenario Hasil"),
    ("ambil", "3.6.10", "III.G.5", None),
    ("ambil", "3.6.11", "III.G.6", None),
    ("ambil", "3.6.12", "III.G.7", None),
    ("ambil", "3.6.13", "III.G.8", None),
    ("ambil", "3.6.14", "III.G.9", None),
    ("ambil", "3.6.15", "III.G.10", None),
    ("ambil", "3.6.16", "III.G.11", None),
    ("ambil", "3.6.17", "III.G.12", None),
    ("ambil", "3.9", "III.H", "Jadwal Penelitian"),
]


def unit(potongan: list[Potongan]) -> dict[str, list[Potongan]]:
    """Kelompokkan tiap potongan bernomor dengan potongan tak bernomor yang mengekornya.

    Blok hasil (`### Hasil ...`) tidak bernomor dan selalu mengikuti subbab pemiliknya.
    Memindahkan subbab tanpa membawa ekornya akan menceraikan hasil dari metodenya.
    Heading tingkat 1 tidak pernah ikut: ia penanda bab, bukan ekor siapa pun.
    """
    keluar: dict[str, list[Potongan]] = {}
    for i, x in enumerate(potongan):
        if not x.nomor:
            continue
        rombongan = [x]
        for y in potongan[i + 1:]:
            if y.nomor or y.tingkat == 1 or y.tingkat < x.tingkat:
                break
            rombongan.append(y)
        keluar[x.nomor] = rombongan
    return keluar


def rakit(potongan: list[Potongan]) -> tuple[list[str], list[str]]:
    """Susun naskah baru menurut SUSUNAN. Mengembalikan (baris, keluhan)."""
    u = unit(potongan)
    baris: list[str] = []
    keluhan: list[str] = []
    terpakai: set[str] = set()

    # Halaman depan dan bagian akhir dilewatkan apa adanya, tanpa dienumerasi satu per
    # satu. Mengenumerasinya berarti daftar itu harus dijaga selaras dengan naskah
    # selamanya; membiarkannya lewat berarti apa pun yang ada di sana tidak mungkin
    # hilang tanpa disadari. Percobaan pertama melewatkan keduanya dan kehilangan 490
    # baris tanpa satu pun keluhan tercetak.
    i_bab = next((i for i, x in enumerate(potongan)
                  if x.tingkat == 1 and x.judul.startswith("BAB ")), len(potongan))
    i_akhir_nomor = max((i for i, x in enumerate(potongan) if x.nomor), default=-1)
    i_akhir = next((i for i in range(i_akhir_nomor + 1, len(potongan))
                    if potongan[i].tingkat == 1), len(potongan))
    for x in potongan[:i_bab]:
        baris += x.baris

    for entri in SUSUNAN:
        jenis = entri[0]
        if jenis == "bab":
            baris += ["", f"# {entri[1]}", ""]
        elif jenis == "baru":
            _, alamat, judul, isi = entri
            baris += [f"{'#' * len(alamat.split('.'))} {alamat.split('.')[-1]}. {judul}", "", isi, ""]
        elif jenis in ("ambil", "gabung"):
            nomor = entri[1]
            if nomor not in u:
                keluhan.append(f"potongan {nomor} tidak ditemukan di naskah")
                continue
            terpakai.add(nomor)
            rombongan = u[nomor]
            if jenis == "gabung":
                baris += rombongan[0].baris[1:]
            else:
                _, _, alamat, judul = entri
                ruas = alamat.split(".")
                judul = judul or rombongan[0].judul
                baris += [f"{'#' * len(ruas)} {ruas[-1]}. {judul}"] + rombongan[0].baris[1:]
            for ekor in rombongan[1:]:
                baris += ekor.baris

    for x in potongan[i_akhir:]:
        baris += x.baris

    tertinggal = sorted(n for n in u if n not in terpakai)
    if tertinggal:
        keluhan.append(f"potongan bernomor tidak dipakai SUSUNAN: {tertinggal}")

    # Jaring pengaman terakhir: tak satu pun potongan boleh menguap. Yang tidak masuk
    # halaman depan, SUSUNAN, maupun bagian akhir dilaporkan sebagai keluhan.
    tercakup = set()
    for x in potongan[:i_bab] + potongan[i_akhir:]:
        tercakup.add(id(x))
    for n in terpakai:
        for x in u[n]:
            tercakup.add(id(x))
    hilang = [x for x in potongan
              if id(x) not in tercakup and not (x.tingkat == 1 and x.judul.startswith("BAB "))]
    if hilang:
        keluhan.append(f"{len(hilang)} potongan menguap: "
                       f"{[x.judul[:40] for x in hilang[:6]]}")
    return baris, keluhan


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uji", action="store_true", help="bongkar-pasang identik?")
    p.add_argument("--pratinjau", action="store_true", help="cetak kerangka blok")
    p.add_argument("--terapkan", action="store_true", help="tulis NASKAH-SUMBER.md")
    a = p.parse_args()

    teks = SUMBER.read_text()
    ok, ket = uji_bongkar_pasang(teks)
    print(f"uji bongkar-pasang: {'LOLOS' if ok else 'GAGAL'} — {ket}")
    if not ok:
        return 1

    pembuka, potongan = bongkar(teks)
    baris, keluhan = rakit(potongan)
    for k in keluhan:
        print(f"  KELUHAN: {k}")

    n_lama = len(teks.split("\n"))
    n_baru = len(pembuka) + len(baris)
    print(f"baris: {n_lama} -> {n_baru}  (selisih {n_baru - n_lama:+d})")

    if a.pratinjau:
        print()
        for b in baris:
            m = re.match(r"^(#{1,4})\s+(.*)$", b)
            if m:
                print(f"  {'  ' * (len(m.group(1)) - 1)}{m.group(2)[:70]}")
    if keluhan:
        print("\nBerhenti: SUSUNAN belum lengkap. Tidak ada berkas yang disentuh.")
        return 1

    if a.terapkan:
        hasil = "\n".join(baris)
        # Jumlah penanda wajib kekal: restrukturisasi memindahkan teks, tidak membuangnya.
        for penanda, n_lama in (("<!-- HASIL-ONLY -->", teks.count("<!-- HASIL-ONLY -->")),
                                ("<!-- /HASIL-ONLY -->", teks.count("<!-- /HASIL-ONLY -->"))):
            if hasil.count(penanda) != n_lama:
                print(f"BATAL: {penanda} {n_lama} -> {hasil.count(penanda)}")
                return 1
        SUMBER.write_text(hasil)
        print(f"\nditulis: {SUMBER.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
