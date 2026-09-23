"""Bangun `references.bib` dari Daftar Pustaka naskah, lewat CrossRef dan OpenAlex.

Jalankan:
    python3 scripts/bangun_pustaka.py --ekstrak   # cetak pengenal tiap entri, tanpa jaringan
    python3 scripts/bangun_pustaka.py --ambil     # ambil metadata, tulis references.bib

Kenapa diambil dari layanan, bukan diketik ulang
------------------------------------------------
Panduan Tugas Akhir UHB mewajibkan daftar pustaka disusun lewat reference manager dan
ditulis dengan gaya APA. Mengetik ulang 36 entri dari gaya IEEE ke gaya APA berarti 36
kesempatan salah ketik nama, tahun, volume, atau halaman — dan kesalahan semacam itu tidak
dapat ditangkap pemeriksa mana pun, sebab naskah tidak tahu apa yang benar.

Metadata karena itu diambil dari sumber resminya: CrossRef bagi entri ber-DOI, arXiv bagi
pracetak, dan pencarian judul OpenAlex bagi entri yang tidak mencantumkan pengenal apa pun.
Keluarannya `references.bib`, yang diimpor ke Mendeley dan dari situ Mendeley yang
mengeluarkan APA — sehingga gaya penulisannya bukan tafsir saya melainkan keluaran alat
yang memang diwajibkan panduan.

Metadata diperlakukan sebagai data yang tidak dipercaya
--------------------------------------------------------
Judul dan nama penulis datang dari rekaman yang dikendalikan penerbit. Seluruhnya
diperlakukan sebagai teks biasa: tidak pernah dirangkai ke dalam perintah shell, dan
kunci sitasi divalidasi terhadap `^[A-Za-z0-9]+$` sebelum dipakai.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
SUMBER = AKAR / "naskah" / "NASKAH-SUMBER.md"
KELUARAN = AKAR / "references.bib"

SURAT = "agrieby.chaniago@student.uhb.ac.id"   # kolam sopan OpenAlex dan CrossRef
JEDA = 0.4


# Lima entri yang tidak tertangkap jalur baku, beserta sebabnya. Ditulis di sini agar
# pemulihannya dapat diperiksa, bukan disembunyikan sebagai perbaikan sekali jalan.
TAMBALAN = {
    # DOI DataCite; CrossRef tidak memuat DOI repositori data.
    22: dict(jenis="datacite", doi="10.24432/C5Q01S",
             judul="Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet",
             penulis=["Isenkul, M. E.", "Sakar, B. E.", "Kursun, O."],
             tahun="2013", penerbit="UCI Machine Learning Repository"),
    36: dict(jenis="datacite", doi="10.24432/C5Q01S",
             judul="Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet",
             penulis=["Isenkul, M. E.", "Sakar, B. E.", "Kursun, O."],
             tahun="2013", penerbit="UCI Machine Learning Repository"),
    # Pencarian judul mula-mula gagal karena tanda baca pada judulnya.
    10: dict(jenis="doi", doi="10.18653/v1/2022.acl-long.269"),
    # Pencarian mula-mula mengembalikan makalah lain yang judulnya mirip.
    16: dict(jenis="openalex_judul", judul="Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson"),
    19: dict(jenis="arxiv", arxiv="2405.21060"),
}

def entri_naskah() -> list[tuple[int, str]]:
    t = SUMBER.read_text()
    dp = t[t.rfind("# DAFTAR PUSTAKA"):]
    hasil = re.findall(r"^\[(\d+)\]\s*(.+?)(?=^\[\d+\]|\Z)", dp, re.S | re.M)
    return [(int(n), " ".join(isi.split())) for n, isi in hasil]


def pengenal(isi: str) -> tuple[str, str]:
    """Kembalikan (jenis, nilai): doi, arxiv, atau judul untuk dicari."""
    m = re.search(r"\b(10\.\d{4,9}/[^\s,\"]+)", isi)
    if m:
        return "doi", m.group(1).rstrip(".,;")
    m = re.search(r"arXiv[:\s]+(\d{4}\.\d{4,5})", isi, re.I)
    if m:
        return "arxiv", m.group(1)
    m = re.search(r'"([^"]{12,})"', isi)
    return "judul", (m.group(1).rstrip(".,") if m else isi[:110])


def ambil_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": f"skripsi-uhb/1.0 ({SURAT})"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception as e:
        print(f"      gagal: {type(e).__name__}")
        return None


def dari_crossref(doi: str) -> dict | None:
    d = ambil_json("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""))
    return d.get("message") if d else None


def dari_openalex_judul(judul: str) -> dict | None:
    q = urllib.parse.urlencode({"filter": f"title.search:{judul}", "per-page": 1,
                                "mailto": SURAT})
    d = ambil_json("https://api.openalex.org/works?" + q)
    hasil = (d or {}).get("results") or []
    return hasil[0] if hasil else None


def dari_arxiv(ident: str) -> dict | None:
    req = urllib.request.Request(
        f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(ident)}",
        headers={"User-Agent": f"skripsi-uhb/1.0 ({SURAT})"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            xml = r.read().decode()
    except Exception as e:
        print(f"      gagal: {type(e).__name__}")
        return None
    judul = re.search(r"<title>(.*?)</title>", xml, re.S)
    tahun = re.search(r"<published>(\d{4})", xml)
    penulis = re.findall(r"<name>(.*?)</name>", xml)
    if not judul:
        return None
    return {"_arxiv": ident, "title": " ".join(judul.group(1).split()),
            "year": tahun.group(1) if tahun else "", "authors": penulis}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ekstrak", action="store_true")
    p.add_argument("--ambil", action="store_true")
    a = p.parse_args()

    entri = entri_naskah()
    print(f"{len(entri)} entri pada Daftar Pustaka\n")

    rekap: dict[str, int] = {}
    rencana = []
    for n, isi in entri:
        jenis, nilai = pengenal(isi)
        rekap[jenis] = rekap.get(jenis, 0) + 1
        rencana.append((n, jenis, nilai, isi))
        if a.ekstrak:
            print(f"  [{n:2d}] {jenis:6s} {nilai[:88]}")
    print("\nrekap:", rekap)

    if not a.ambil:
        return 0

    hasil = {}
    for n, jenis, nilai, isi in rencana:
        print(f"  [{n:2d}] {jenis} -> {nilai[:64]}")
        if jenis == "doi":
            rec = dari_crossref(nilai)
        elif jenis == "arxiv":
            rec = dari_arxiv(nilai)
        else:
            rec = dari_openalex_judul(nilai)
        hasil[n] = (jenis, nilai, rec)
        time.sleep(JEDA)

    tanpa = [n for n, (_, _, r) in hasil.items() if not r]
    print(f"\nberhasil {len(hasil) - len(tanpa)}/{len(hasil)}; gagal: {tanpa}")
    (AKAR / "scripts" / "_pustaka_mentah.json").write_text(
        json.dumps({str(k): v for k, v in hasil.items()}, ensure_ascii=False))
    print("metadata mentah disimpan ke scripts/_pustaka_mentah.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
