"""Isi DAFTAR TABEL, DAFTAR LAMPIRAN, dan DAFTAR SINGKATAN pada dokumen turunan.

Jalankan: python3 scripts/bangun_daftar_pelengkap.py
Dipanggil otomatis dari `scripts/pisah_sempro_semhas.py` sesudah tabel dinomori.

Kenapa per dokumen, bukan di naskah sumber
-------------------------------------------
Alasan yang sama dengan penomoran tabel: kedua dokumen memuat himpunan tabel yang berbeda.
Sempro memuat 22 tabel bernomor, semhas 86 — selisihnya tabel di dalam blok berpenanda
hasil. Daftar tabel yang dibakukan di naskah sumber karena itu pasti salah pada salah satu
dokumen.

Daftar dibangkitkan dari dokumen yang sudah jadi, sehingga tidak mungkin menyimpang dari
isinya. Nomor halaman sengaja tidak dicantumkan: naskah Markdown belum berhalaman, dan
angkanya baru bermakna sesudah tahap `.docx`. Titik-titik pengisi disediakan agar
penomoran halaman tinggal ditempelkan di sana.

Daftar singkatan
----------------
Singkatan tidak dibaca dari prosa — mencarinya dengan pola akan menangkap nama arsitektur,
nama berkas, dan potongan kode. Ia diambil dari daftar yang ditulis tangan di bawah, yang
memang berperan sebagai satu-satunya sumber kebenaran bagi singkatan naskah ini.
"""

from __future__ import annotations

import re
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
DOKUMEN = [AKAR / "naskah" / "sempro-skripsi.md",
           AKAR / "naskah" / "semhas-skripsi.md"]

RINTIS = "(dibangkitkan scripts/bangun_daftar_pelengkap.py)"

# Singkatan yang benar-benar dipakai naskah, beserta kepanjangannya. Urut abjad.
SINGKATAN = [
    ("AUC", "Area Under the Curve, luas di bawah kurva ROC"),
    ("BiGRU", "Bidirectional Gated Recurrent Unit"),
    ("BiSP", "Biometric Smart Pen"),
    ("CI", "Confidence Interval, selang kepercayaan"),
    ("DST", "Dynamic Spiral Test"),
    ("GRU", "Gated Recurrent Unit"),
    ("HC", "Healthy Control, kelompok kontrol sehat"),
    ("MDE", "Minimum Detectable Effect, ukuran efek terkecil yang dapat dideteksi"),
    ("MIL", "Multiple Instance Learning"),
    ("PD", "Parkinson's Disease, kelompok penderita Parkinson"),
    ("ROC", "Receiver Operating Characteristic"),
    ("SHAP", "SHapley Additive exPlanations"),
    ("SSD", "Structured State Space Duality"),
    ("SSM", "State Space Model"),
    ("SST", "Static Spiral Test"),
    ("STCP", "Stability Test on Certain Point"),
    ("UCI", "University of California, Irvine Machine Learning Repository"),
    ("XAI", "eXplainable Artificial Intelligence"),
]

TITIK = " " + "." * 6


def daftar_tabel(teks: str) -> str:
    baris = [f"Tabel {m.group(1)}  {m.group(2).rstrip()}{TITIK}"
             for m in re.finditer(r"^Tabel (\d+\.\d+) (.+)$", teks, re.M)]
    return "\n".join(baris) if baris else "(tidak ada tabel)"


def daftar_lampiran(teks: str) -> str:
    baris = [f"Lampiran {m.group(1)}  {m.group(2).rstrip()}{TITIK}"
             for m in re.finditer(r"^# LAMPIRAN ([A-Z])\. (.+)$", teks, re.M)]
    return "\n".join(baris) if baris else "(tidak ada lampiran)"


def daftar_singkatan(teks: str) -> str:
    # Hanya singkatan yang benar-benar muncul di dokumen ini yang didaftarkan.
    lebar = max(len(k) for k, _ in SINGKATAN)
    dipakai = [(k, v) for k, v in SINGKATAN
               if re.search(r"\b" + re.escape(k) + r"\b", teks)]
    return "\n".join(f"{k:<{lebar}}  {v}" for k, v in dipakai)


def isi(jalur: Path) -> tuple[int, int, int]:
    teks = jalur.read_text()
    hasil = {"DAFTAR TABEL": daftar_tabel(teks),
             "DAFTAR LAMPIRAN": daftar_lampiran(teks),
             "DAFTAR SINGKATAN": daftar_singkatan(teks)}
    for judul, badan in hasil.items():
        pola = re.compile(r"(# " + judul + r"\n\n```\n)" + re.escape(RINTIS) + r"(\n```)")
        teks = pola.sub(lambda m: m.group(1) + badan + m.group(2), teks, count=1)
    jalur.write_text(teks)
    return tuple(len(x.split("\n")) for x in hasil.values())


def main() -> int:
    for jalur in DOKUMEN:
        if not jalur.exists():
            continue
        tab, lam, sing = isi(jalur)
        print(f"  {jalur.name}: daftar tabel {tab}, lampiran {lam}, singkatan {sing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
