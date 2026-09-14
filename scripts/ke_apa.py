"""Ubah sitasi naskah dari gaya IEEE bernomor menjadi gaya APA nama-tahun.

Jalankan:
    python3 scripts/ke_apa.py --pratinjau   # cetak peta nomor -> sitasi, tanpa menulis
    python3 scripts/ke_apa.py --terapkan    # ubah NASKAH-SUMBER.md dan skrip pemisah

Kenapa perlu
------------
Panduan Tugas Akhir UHB menetapkan gaya APA 6th, daftar pustaka diurutkan abjad, dan
penulisannya lewat reference manager. Naskah ini semula memakai gaya IEEE bernomor.

Nama dan tahun diambil dari `scripts/_pustaka_mentah.json`, yaitu metadata yang sudah
ditarik dari CrossRef, arXiv, OpenAlex, dan DataCite — bukan dibaca ulang dari prosa
naskah. Dengan begitu tidak ada kesempatan salah eja yang lolos diam-diam.

Penggabungan sitasi berurutan
-----------------------------
`[1], [4], [5]` menjadi satu tanda kurung `(A, 2016; B, 2016; C, 2021)`, bukan tiga
tanda kurung berturut-turut. Itu bentuk yang dituntut APA dan yang dicontohkan panduan.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
MENTAH = AKAR / "scripts" / "_pustaka_mentah.json"
BERKAS = [AKAR / "NASKAH-SUMBER.md", AKAR / "scripts" / "pisah_sempro_semhas.py"]


def nama_tahun(rec: dict) -> tuple[list[str], str]:
    """Kembalikan (daftar nama belakang, tahun) dari bentuk rekaman mana pun."""
    if rec.get("_manual") or rec.get("_datacite"):
        return [a.split(",")[0].strip() for a in rec.get("authors", [])], str(rec.get("year", ""))
    if rec.get("_arxiv"):
        return [a.split()[-1] for a in rec.get("authors", []) if a], str(rec.get("year", ""))
    if "authorships" in rec:
        nama = [(a.get("author") or {}).get("display_name", "") for a in rec["authorships"]]
        return [x.split()[-1] for x in nama if x], str(rec.get("publication_year", ""))
    nama = [a.get("family", "") for a in rec.get("author", []) if a.get("family")]
    tahun = ""
    for k in ("published-print", "published-online", "issued", "created"):
        bag = (rec.get(k) or {}).get("date-parts") or []
        if bag and bag[0] and bag[0][0]:
            tahun = str(bag[0][0]); break
    return nama, tahun


def sitasi(nama: list[str], tahun: str) -> str:
    if not nama:
        return f"Anon., {tahun}"
    if len(nama) == 1:
        return f"{nama[0]}, {tahun}"
    if len(nama) == 2:
        return f"{nama[0]} & {nama[1]}, {tahun}"
    return f"{nama[0]} dkk., {tahun}"


def tahun_berimbuhan() -> dict[int, str]:
    """Nomor entri -> tahun, dengan imbuhan a/b bila bertabrakan.

    APA menuntut dua karya penulis pertama yang sama pada tahun yang sama dibedakan
    `2016a` dan `2016b`. Tanpa itu sitasi `(Pereira dkk., 2016)` menunjuk dua sumber
    sekaligus dan pembaca tidak dapat tahu yang mana. Kasus itu benar-benar ada pada
    naskah ini, dan baru terlihat ketika pemeriksa sitasi menghitung kunci uniknya.

    Imbuhan diberikan menurut urutan nomor entri, sehingga hasilnya tetap sama pada
    tiap kali dijalankan.
    """
    mentah = json.loads(MENTAH.read_text())
    kelompok: dict[tuple[str, str], list[int]] = {}
    for k, (_, _, rec) in mentah.items():
        if not rec:
            continue
        nama, tahun = nama_tahun(rec)
        kelompok.setdefault((nama[0] if nama else "Anon", tahun), []).append(int(k))

    hasil: dict[int, str] = {}
    for (_, tahun), nomor in kelompok.items():
        # Entri kembar sungguhan (sumber yang sama disitasi dua kali) tidak diberi imbuhan.
        judul = {}
        for n in sorted(nomor):
            rec = mentah[str(n)][2]
            judul.setdefault(_judul(rec), []).append(n)
        if len(judul) == 1:
            for n in nomor:
                hasil[n] = tahun
            continue
        for huruf, (_, kel) in zip("abcdefgh", sorted(judul.items(), key=lambda x: min(x[1]))):
            for n in kel:
                hasil[n] = f"{tahun}{huruf}"
    return hasil


def _judul(rec: dict) -> str:
    if rec.get("_manual") or rec.get("_datacite") or rec.get("_arxiv"):
        return (rec.get("title") or "").lower()
    if "authorships" in rec:
        return (rec.get("display_name") or "").lower()
    return ((rec.get("title") or [""])[0]).lower()


def peta() -> dict[int, str]:
    mentah = json.loads(MENTAH.read_text())
    tahun = tahun_berimbuhan()
    hasil = {}
    for k, (_, _, rec) in mentah.items():
        if rec:
            nama, _ = nama_tahun(rec)
            hasil[int(k)] = sitasi(nama, tahun[int(k)])
    return hasil


# Deretan sitasi berurutan: '[1], [4], dan [5]' atau '[8], [9], [10]'
DERET = re.compile(r"\[(\d+)\](?:\s*(?:,|dan|serta)\s*\[(\d+)\])*")


def ubah(teks: str, m: dict[int, str]) -> tuple[str, int, set[int]]:
    n = 0
    tak_dikenal: set[int] = set()

    def ganti(mm: re.Match) -> str:
        nonlocal n
        nomor = [int(x) for x in re.findall(r"\[(\d+)\]", mm.group(0))]
        if any(x not in m for x in nomor):
            tak_dikenal.update(x for x in nomor if x not in m)
            return mm.group(0)
        n += 1
        return "(" + "; ".join(dict.fromkeys(m[x] for x in nomor)) + ")"

    return DERET.sub(ganti, teks), n, tak_dikenal


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pratinjau", action="store_true")
    p.add_argument("--terapkan", action="store_true")
    a = p.parse_args()

    m = peta()
    if a.pratinjau or not a.terapkan:
        for k in sorted(m):
            print(f"  [{k:2d}] -> ({m[k]})")
        print(f"\n{len(m)} entri terpetakan")
        return 0

    for jalur in BERKAS:
        teks = jalur.read_text()
        # Entri Daftar Pustaka sendiri disisihkan agar penomorannya tidak ikut diubah, TAPI
        # Lampiran yang menyusul (A, B, C, D) tetap diproses. Versi pertama berkas ini memotong
        # di "# DAFTAR PUSTAKA" dan berhenti sampai akhir dokumen — akibatnya delapan sitasi
        # gaya lama di Lampiran A dan B lolos, baru ketahuan saat naskah dibaca ulang untuk
        # membangun .docx. Bibliografi sendiri dipastikan tidak memuat pola `[angka]`, sehingga
        # aman memproses sisa dokumen sebagai satu bagian.
        i = teks.rfind("# DAFTAR PUSTAKA")
        j = teks.find("\n# LAMPIRAN", i) if i > 0 else -1
        if i > 0 and j > 0:
            kepala, pustaka, lampiran = teks[:i], teks[i:j], teks[j:]
        else:
            kepala, pustaka, lampiran = teks, "", ""
        kepala, n1, asing1 = ubah(kepala, m)
        lampiran, n2, asing2 = ubah(lampiran, m)
        jalur.write_text(kepala + pustaka + lampiran)
        n, asing = n1 + n2, asing1 | asing2
        print(f"  {jalur.name}: {n} kelompok sitasi diubah"
              + (f"; nomor tak dikenal {sorted(asing)}" if asing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
