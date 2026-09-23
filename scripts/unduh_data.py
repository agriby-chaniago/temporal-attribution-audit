"""Unduh kedua basis data mentah dari sumber aslinya ke `data/raw/`.

Jalankan: python3 scripts/unduh_data.py

Tidak satu pun basis data disebarkan ulang lewat repositori ini; keduanya publik
dan diambil langsung dari penerbitnya. Ketentuan atribusi dan daftar perubahan
yang dilakukan penelitian ini ada di `LISENSI-DATA.md`.

Skrip ini idempoten: berkas yang sudah ada dilewati, bukan diunduh ulang. Tiap
arsip dicocokkan sha256-nya terhadap rilis yang dipakai penelitian ini, sehingga
angka yang dihasilkan pipeline dapat ditelusuri ke bita yang sama. Bila penerbit
mengganti isi arsip tanpa mengganti namanya, pencocokan itu akan menyatakannya
sebagai PERINGATAN, bukan diam-diam meneruskan.

Setelah selesai, struktur yang diharapkan pemuat di `src/`:

    data/raw/uci395/extracted/hw_dataset/control
    data/raw/uci395/extracted/hw_dataset/parkinson
    data/raw/uci395/extracted/new_dataset/parkinson
    data/raw/newhandpd/extracted/Signal            (35 subjek kontrol)
    data/raw/newhandpd/extracted_patient/Signal    (31 subjek penderita)

Unduhan berjumlah kira-kira 390 MB dan memakan beberapa menit pada sambungan
rumahan. Ekstraksi menambah kira-kira 1,1 GB.

PERINGATAN DATA PRIBADI. Header metadata NewHandPD mentah memuat ruas
pengenal (nama, sebuah pengenal yang menyerupai nomor rekam medis). Berkas
mentah hasil skrip ini **tidak boleh** dipublikasikan ulang atau dimasukkan ke
repositori; `data/` sudah dikecualikan lewat `.gitignore`. Ekstraksi metadata di
`src/newhandpd.py` dibatasi daftar-izin `META_AMAN`.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
MENTAH = AKAR / "data" / "raw"

# sha256 rilis yang dipakai penelitian ini, dihitung dari arsip yang benar-benar
# diunduh saat Tahap 1. Dipakai sebagai pemeriksaan provenans, bukan sebagai
# gerbang: arsip yang berbeda tetap diekstrak, tetapi dinyatakan berbeda.
SUMBER = [
    {
        "nama": "UCI 395",
        "url": "https://archive.ics.uci.edu/static/public/395/"
               "parkinson+disease+spiral+drawings+using+digitized+graphics+tablet.zip",
        "arsip": MENTAH / "uci395" / "uci395.zip",
        "tujuan": MENTAH / "uci395" / "extracted",
        "sha256": "4181243d64c29382a0035a9bca3ebcf446079b65c70a0b343fe7975372dc92e6",
        "penanda": "hw_dataset",
    },
    {
        "nama": "NewHandPD kontrol",
        "url": "https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewHealthy/HealthySignal.zip",
        "arsip": MENTAH / "newhandpd" / "HealthySignal.zip",
        "tujuan": MENTAH / "newhandpd" / "extracted",
        "sha256": "f278f12f0eae53ab031824bdd884a9804c1fa3f0b1512ae9e10c389deff44289",
        "penanda": "Signal",
    },
    {
        "nama": "NewHandPD penderita",
        "url": "https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewPatients/PatientSignal.zip",
        "arsip": MENTAH / "newhandpd" / "PatientSignal.zip",
        "tujuan": MENTAH / "newhandpd" / "extracted_patient",
        "sha256": "51dad3fb56e0a72342091454f5ab80863a3733474ada43073f678ec24bf03517",
        "penanda": "Signal",
    },
]


def sidik(path: Path, blok: int = 1 << 20) -> str:
    """SHA-256 sebuah berkas, dibaca per blok agar arsip 240 MB tidak dimuat utuh."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while potongan := f.read(blok):
            h.update(potongan)
    return h.hexdigest()


def kemajuan(blok: int, ukuran_blok: int, total: int) -> None:
    if total <= 0:
        return
    terunduh = min(blok * ukuran_blok, total)
    persen = 100 * terunduh / total
    print(f"\r    {terunduh/1e6:7.1f} / {total/1e6:.1f} MB  ({persen:5.1f}%)",
          end="", flush=True)


def sudah_terekstrak(tujuan: Path, penanda: str) -> bool:
    """Ekstraksi dianggap lengkap hanya bila direktori penandanya benar-benar ada.

    Memeriksa keberadaan `tujuan` saja tidak cukup: unduhan yang terputus di
    tengah ekstraksi meninggalkan direktori kosong yang akan dilewati diam-diam.
    """
    return tujuan.is_dir() and any(tujuan.rglob(penanda))


def kerjakan(s: dict) -> bool:
    print(f"\n{s['nama']}")
    arsip: Path = s["arsip"]
    tujuan: Path = s["tujuan"]

    if sudah_terekstrak(tujuan, s["penanda"]):
        print(f"  sudah ada   : {tujuan.relative_to(AKAR)}")
        return True

    if not arsip.exists():
        arsip.parent.mkdir(parents=True, exist_ok=True)
        print(f"  mengunduh   : {s['url']}")
        sementara = arsip.with_suffix(arsip.suffix + ".sedang-unduh")
        try:
            urllib.request.urlretrieve(s["url"], sementara, kemajuan)
        except Exception as e:                                   # noqa: BLE001
            sementara.unlink(missing_ok=True)
            print(f"\n  GAGAL unduh : {e}", file=sys.stderr)
            return False
        # Ganti nama hanya setelah unduhan utuh, sehingga arsip separuh tidak
        # pernah tampil sebagai arsip yang sah pada jalannya yang berikut.
        sementara.replace(arsip)
        print()
    else:
        print(f"  arsip ada   : {arsip.relative_to(AKAR)}")

    diperoleh = sidik(arsip)
    if diperoleh == s["sha256"]:
        print("  sha256      : cocok dengan rilis yang dipakai penelitian ini")
    else:
        print("  PERINGATAN  : sha256 berbeda dari rilis penelitian ini.")
        print(f"                diharap {s['sha256']}")
        print(f"                didapat {diperoleh}")
        print("                Penerbit mungkin mengganti isi arsip. Angka yang")
        print("                dihasilkan dapat berbeda dari yang dilaporkan naskah.")

    print(f"  mengekstrak : {tujuan.relative_to(AKAR)}")
    tujuan.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arsip) as z:
        z.extractall(tujuan)

    if not sudah_terekstrak(tujuan, s["penanda"]):
        print(f"  GAGAL       : '{s['penanda']}' tidak ditemukan sesudah ekstraksi.",
              file=sys.stderr)
        return False
    return True


def main() -> int:
    print("Mengunduh basis data mentah ke data/raw/.")
    print("Atribusi dan ketentuan pemakaian: LISENSI-DATA.md")

    gagal = [s["nama"] for s in SUMBER if not kerjakan(s)]

    print()
    if gagal:
        print("GAGAL untuk: " + ", ".join(gagal), file=sys.stderr)
        print("Jalankan ulang skrip ini; yang sudah berhasil akan dilewati.",
              file=sys.stderr)
        return 1

    print("Selesai. Seluruh basis data siap di data/raw/.")
    print("Langkah berikut: notebooks/01_verifikasi_struktur_data.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
