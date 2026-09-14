"""Tulis ulang Daftar Pustaka naskah dengan gaya APA 6th, diurutkan abjad.

Jalankan:
    python3 scripts/tulis_daftar_pustaka.py             # cetak, tidak menulis
    python3 scripts/tulis_daftar_pustaka.py --terapkan  # tulis ke NASKAH-SUMBER.md

Kedudukan berkas ini terhadap Mendeley
---------------------------------------
Panduan Tugas Akhir UHB mewajibkan daftar pustaka disusun lewat reference manager. Yang
mengikat kewajiban itu adalah `references.bib` — berkas itulah yang diimpor ke Mendeley,
dan Mendeley yang menjadi sumber resmi bentuk APA pada naskah `.docx` nanti.

Berkas ini menghasilkan daftar pustaka bagi naskah Markdown, agar sempro lengkap dan dapat
dibaca sebelum tahap `.docx` dikerjakan. Keduanya berangkat dari metadata yang sama
(`scripts/_pustaka_mentah.json`), sehingga keluarannya tidak dapat berbeda isi — hanya
berbeda alat yang merangkainya.

Entri kembar
------------
Basis data UCI 395 disitasi dua kali pada naskah lama, dengan nomor berbeda. Keduanya
menunjuk DOI yang sama dan karena itu digabung menjadi satu entri, sebagaimana dituntut
APA.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
MENTAH = AKAR / "scripts" / "_pustaka_mentah.json"
SUMBER = AKAR / "NASKAH-SUMBER.md"


def inisial(depan: str) -> str:
    return " ".join(x[0].upper() + "." for x in re.split(r"[\s.]+", depan or "") if x)


def bagian(rec: dict) -> dict:
    """Seragamkan keempat bentuk rekaman menjadi medan yang sama."""
    if rec.get("_manual") or rec.get("_datacite"):
        return dict(penulis=list(rec.get("authors", [])), tahun=str(rec.get("year", "")),
                    judul=rec.get("title", ""), wadah=rec.get("venue", ""),
                    volume="", nomor="", halaman="", doi=rec.get("doi", ""),
                    penerbit=rec.get("publisher", ""), arxiv=rec.get("arxiv", ""))
    if rec.get("_arxiv"):
        pen = []
        for x in rec.get("authors", []):
            b = x.split()
            if b:
                pen.append(f"{b[-1]}, {inisial(' '.join(b[:-1]))}".strip().rstrip(","))
        return dict(penulis=pen, tahun=str(rec.get("year", "")), judul=rec.get("title", ""),
                    wadah="", volume="", nomor="", halaman="", doi="", penerbit="",
                    arxiv=rec.get("_arxiv", ""))
    if "authorships" in rec:
        pen = []
        for a in rec["authorships"]:
            x = (a.get("author") or {}).get("display_name") or ""
            b = x.split()
            if b:
                pen.append(f"{b[-1]}, {inisial(' '.join(b[:-1]))}".strip().rstrip(","))
        lok = (rec.get("primary_location") or {}).get("source") or {}
        bib = rec.get("biblio") or {}
        hal = "-".join(x for x in (bib.get("first_page"), bib.get("last_page")) if x)
        return dict(penulis=pen, tahun=str(rec.get("publication_year") or ""),
                    judul=rec.get("display_name", ""), wadah=lok.get("display_name", ""),
                    volume=bib.get("volume") or "", nomor=bib.get("issue") or "",
                    halaman=hal, doi=(rec.get("doi") or "").replace("https://doi.org/", ""),
                    penerbit="", arxiv="")
    pen = []
    for a in rec.get("author", []):
        if a.get("family"):
            pen.append(f"{a['family']}, {inisial(a.get('given', ''))}".strip().rstrip(","))
    tahun = ""
    for k in ("published-print", "published-online", "issued", "created"):
        bag = (rec.get(k) or {}).get("date-parts") or []
        if bag and bag[0] and bag[0][0]:
            tahun = str(bag[0][0]); break
    return dict(penulis=pen, tahun=tahun, judul=(rec.get("title") or [""])[0],
                wadah=(rec.get("container-title") or [""])[0], volume=rec.get("volume", ""),
                nomor=rec.get("issue", ""), halaman=rec.get("page", ""),
                doi=rec.get("DOI", ""), penerbit=rec.get("publisher", ""), arxiv="")


def rangkai(d: dict) -> str:
    pen = d["penulis"]
    if not pen:
        nama = "Anon."
    elif len(pen) == 1:
        nama = pen[0]
    else:
        nama = ", ".join(pen[:-1]) + ", & " + pen[-1]

    bagianku = [f"{nama} ({d['tahun'] or 't.t.'}). {d['judul'].rstrip('.')}."]
    if d["wadah"]:
        w = f" *{d['wadah']}*"
        if d["volume"]:
            w += f", *{d['volume']}*"
            if d["nomor"]:
                w += f"({d['nomor']})"
        if d["halaman"]:
            w += f", {d['halaman']}"
        bagianku.append(w + ".")
    elif d["arxiv"]:
        bagianku.append(f" arXiv:{d['arxiv']}.")
    if d["penerbit"] and not d["wadah"]:
        bagianku.append(f" {d['penerbit']}.")
    if d["doi"]:
        bagianku.append(f" https://doi.org/{d['doi']}")
    return "".join(bagianku)


def bangun() -> tuple[str, int, int]:
    mentah = json.loads(MENTAH.read_text())
    # Tahun berimbuhan dihitung oleh `ke_apa.py` dan dipakai bersama, sehingga imbuhan pada
    # daftar pustaka tidak mungkin berbeda dari imbuhan pada sitasi di dalam teks.
    import sys
    sys.path.insert(0, str(AKAR / "scripts"))
    from ke_apa import tahun_berimbuhan
    imbuhan = tahun_berimbuhan()

    terlihat, entri = set(), []
    for k in sorted(mentah, key=int):
        _, _, rec = mentah[k]
        if not rec:
            continue
        d = bagian(rec)
        d["tahun"] = imbuhan.get(int(k), d["tahun"])
        kunci = ((d["doi"] or d["arxiv"] or d["judul"]).lower(), d["tahun"])
        if kunci in terlihat:
            continue
        terlihat.add(kunci)
        entri.append(rangkai(d))
    entri.sort(key=lambda s: s.lower())
    return "\n\n".join(entri), len(entri), len(mentah) - len(entri)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--terapkan", action="store_true")
    a = p.parse_args()

    isi, n, kembar = bangun()
    print(isi if not a.terapkan else "")
    print(f"\n{n} entri, {kembar} kembar digabung")

    if a.terapkan:
        teks = SUMBER.read_text()
        i = teks.rfind("# DAFTAR PUSTAKA")
        if i < 0:
            print("GAGAL: bagian DAFTAR PUSTAKA tidak ditemukan.")
            return 1
        # Bagian berakhir pada heading tingkat satu berikutnya (LAMPIRAN).
        j = teks.find("\n# ", i + 1)
        j = j if j > 0 else len(teks)
        SUMBER.write_text(teks[:i] + "# DAFTAR PUSTAKA\n\n" + isi + "\n" + teks[j:])
        print(f"ditulis: {SUMBER.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
