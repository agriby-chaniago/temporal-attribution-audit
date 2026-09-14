"""Ubah metadata mentah menjadi `references.bib` untuk diimpor ke Mendeley.

Jalankan: python3 scripts/tulis_bib.py

Masukannya `scripts/_pustaka_mentah.json`, keluaran `scripts/bangun_pustaka.py --ambil`.
Rekaman datang dari empat bentuk berbeda — CrossRef, arXiv, OpenAlex, dan tambalan
terverifikasi — sehingga berkas ini yang menyeragamkannya.

Kenapa berhenti di `.bib`, bukan langsung menulis APA
------------------------------------------------------
Panduan Tugas Akhir UHB **mewajibkan** daftar pustaka disusun lewat reference manager
(Mendeley atau EndNote). Menulis APA sendiri dari naskah ini akan memenuhi bentuknya tetapi
melanggar ketentuannya. Karena itu keluarannya berhenti pada `.bib`: Mendeley yang
mengimpornya, dan Mendeley yang mengeluarkan APA 6th.

Kunci sitasi
------------
`<nama belakang penulis pertama><tahun><kata pertama judul>`, seluruhnya alfanumerik.
Divalidasi terhadap `^[A-Za-z0-9]+$` sebelum ditulis, sebab judul dan nama penulis berasal
dari rekaman yang dikendalikan penerbit dan tidak dipercaya begitu saja.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
MENTAH = AKAR / "scripts" / "_pustaka_mentah.json"
KELUARAN = AKAR / "references.bib"


def bersih(s: str) -> str:
    """Alfanumerik ASCII saja — dipakai membentuk kunci sitasi."""
    s = unicodedata.normalize("NFKD", s or "")
    return re.sub(r"[^A-Za-z0-9]", "", s.encode("ascii", "ignore").decode())


def dari_crossref(r: dict) -> dict:
    penulis = [f"{a.get('family','')}, {a.get('given','')}".strip(", ")
               for a in r.get("author", []) if a.get("family")]
    tahun = ""
    for k in ("published-print", "published-online", "issued", "created"):
        bagian = (r.get(k) or {}).get("date-parts") or []
        if bagian and bagian[0] and bagian[0][0]:
            tahun = str(bagian[0][0]); break
    wadah = (r.get("container-title") or [""])[0]
    jenis = "inproceedings" if r.get("type") == "proceedings-article" else "article"
    return dict(jenis=jenis, penulis=penulis, judul=(r.get("title") or [""])[0],
                wadah=wadah, volume=r.get("volume", ""), nomor=r.get("issue", ""),
                halaman=r.get("page", ""), tahun=tahun, doi=r.get("DOI", ""),
                penerbit=r.get("publisher", ""))


def dari_openalex(r: dict) -> dict:
    penulis = []
    for a in r.get("authorships", []):
        nama = (a.get("author") or {}).get("display_name") or ""
        bagian = nama.split()
        if bagian:
            penulis.append(f"{bagian[-1]}, {' '.join(x[0] + '.' for x in bagian[:-1])}".strip(", "))
    lok = (r.get("primary_location") or {}).get("source") or {}
    bib = r.get("biblio") or {}
    hal = "-".join(x for x in (bib.get("first_page"), bib.get("last_page")) if x)
    return dict(jenis="article", penulis=penulis, judul=r.get("display_name", ""),
                wadah=lok.get("display_name", ""), volume=bib.get("volume") or "",
                nomor=bib.get("issue") or "", halaman=hal,
                tahun=str(r.get("publication_year") or ""),
                doi=(r.get("doi") or "").replace("https://doi.org/", ""), penerbit="")


def dari_arxiv(r: dict) -> dict:
    penulis = []
    for nama in r.get("authors", []):
        b = nama.split()
        if b:
            penulis.append(f"{b[-1]}, {' '.join(x[0] + '.' for x in b[:-1])}".strip(", "))
    return dict(jenis="misc", penulis=penulis, judul=r.get("title", ""),
                wadah=f"arXiv:{r.get('_arxiv','')}", volume="", nomor="", halaman="",
                tahun=r.get("year", ""), doi="", penerbit="arXiv")


def dari_manual(r: dict) -> dict:
    jenis = "misc" if r.get("arxiv") else ("inproceedings" if "Proceedings" in r.get("venue", "") else "article")
    return dict(jenis=jenis, penulis=r.get("authors", []), judul=r.get("title", ""),
                wadah=r.get("venue", "") or f"arXiv:{r.get('arxiv','')}",
                volume="", nomor="", halaman="", tahun=r.get("year", ""),
                doi=r.get("doi", ""), penerbit=r.get("publisher", ""))


def dari_datacite(r: dict) -> dict:
    return dict(jenis="misc", penulis=r.get("authors", []), judul=r.get("title", ""),
                wadah="", volume="", nomor="", halaman="", tahun=r.get("year", ""),
                doi=r.get("doi", ""), penerbit=r.get("publisher", ""))


def normalkan(jenis_sumber: str, r: dict) -> dict:
    if r.get("_manual"):
        return dari_manual(r)
    if r.get("_datacite"):
        return dari_datacite(r)
    if r.get("_arxiv"):
        return dari_arxiv(r)
    if "authorships" in r:
        return dari_openalex(r)
    return dari_crossref(r)


def main() -> int:
    mentah = json.loads(MENTAH.read_text())
    entri, kunci_dipakai, kurang = [], {}, []

    for n in sorted(mentah, key=int):
        jenis_sumber, _, r = mentah[n]
        if not r:
            kurang.append(int(n)); continue
        d = normalkan(jenis_sumber, r)

        pertama = (d["penulis"][0].split(",")[0] if d["penulis"] else "Anon")
        dasar = bersih(pertama) + bersih(d["tahun"]) + bersih(d["judul"].split()[0] if d["judul"] else "")
        dasar = dasar or f"entri{n}"
        assert re.fullmatch(r"[A-Za-z0-9]+", dasar), dasar
        kunci = dasar
        if kunci in kunci_dipakai:
            continue                      # entri kembar, mis. basis data yang disitasi dua kali
        kunci_dipakai[kunci] = int(n)

        baris = [f"@{d['jenis']}{{{kunci},"]
        baris.append(f"  author = {{{' and '.join(d['penulis'])}}},")
        baris.append(f"  title = {{{{{d['judul']}}}}},")
        for medan, nilai in (("journal" if d["jenis"] == "article" else "booktitle", d["wadah"]),
                             ("volume", d["volume"]), ("number", d["nomor"]),
                             ("pages", d["halaman"]), ("year", d["tahun"]),
                             ("publisher", d["penerbit"]), ("doi", d["doi"])):
            if nilai:
                baris.append(f"  {medan} = {{{nilai}}},")
        baris.append("}")
        entri.append((kunci, "\n".join(baris)))

    KELUARAN.write_text("\n\n".join(t for _, t in sorted(entri)) + "\n")
    print(f"{len(entri)} entri ditulis ke {KELUARAN.name}")
    if kurang:
        print(f"  tanpa metadata, TIDAK ditulis: {kurang}")
    kembar = len(mentah) - len(entri) - len(kurang)
    if kembar:
        print(f"  {kembar} entri kembar digabung (sitasi ganda ke sumber yang sama)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
