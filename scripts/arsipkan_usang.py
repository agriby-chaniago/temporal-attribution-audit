"""Pindahkan artefak yang sudah tergantikan ke arsip bertanda, tanpa menghapus apa pun.

Jalankan: python3 scripts/arsipkan_usang.py

Masalah yang diselesaikan
-------------------------
Setelah proyek berjalan berbulan-bulan, sebagian berkas menjadi salinan lama dari
sesuatu yang sudah berubah. Risikonya bukan ruang disk melainkan **tertukar**:
seseorang membuka salinan lama, membaca angka yang sudah tidak berlaku, lalu
memakainya. Yang paling berbahaya adalah cadangan hasil 2 September, sebab
sepuluh berkasnya merupakan hasil Skenario S3 dan S5 dari sebelum pelaporan
diperluas ke sepuluh seed — persis jenis angka yang paling merusak bila terbaca
sebagai mutakhir.

Prinsip: **memindahkan, bukan menghapus.** Tiap butir tetap dapat dikembalikan,
dan tiap pemindahan disertai alasan tertulis beserta apa yang menggantikannya.

Tiga hal yang membuat skrip ini bukan sekadar `mv`
---------------------------------------------------
**Gerbang menguji keadaan, bukan keberadaan.** Salah satu butir baru layak
diarsipkan setelah sebuah sitasi di naskah diperbaiki. Mencari nama berkas
penggantinya saja tidak cukup, sebab nama itu bisa muncul di tempat lain
sementara sitasi lamanya tetap tertinggal. Karena itu gerbangnya menuntut dua
syarat sekaligus: frasa lama **hilang** dan frasa baru **ada**.

**Seluruh pemeriksaan tuntas sebelum berkas pertama bergerak.** Tanpa itu, butir
pertama bisa sudah berpindah ketika butir keenam ternyata tidak ditemukan, dan
proyek tertinggal pada keadaan yang bukan sebelum maupun sesudah.

**Gagal-tertutup.** Bila satu pemindahan gagal di tengah, skrip berhenti seketika
dan mencetak apa yang sudah berpindah beserta jalur asalnya, bukan melanjutkan
diam-diam ke butir berikutnya.

Aturan penamaan cadangan, ditetapkan di sini
--------------------------------------------
- `prapelaksanaan_<stempel>/` — jaring pengaman **berlaku**, dibuat sebelum
  pekerjaan besar, manifesnya dipakai memverifikasi perubahan
- `usang_<stempel>/` — sudah **tergantikan**, disimpan hanya untuk penelusuran

Cadangan tanpa salah satu awalan itu dianggap belum diklasifikasi dan harus
diperiksa. Aturan sesederhana ini yang membuat penumpukannya tidak berulang.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
CADANGAN = AKAR / "cadangan"

# Gerbang: dua syarat pada naskah, keduanya wajib terpenuhi.
NASKAH = AKAR / "NASKAH-SUMBER.md"
SITASI_LAMA = "Angka pada `results/kohort_kedua_peringkat.csv`"
SITASI_BARU = "Angka pada `results/a_baseline_lintas_kohort.csv`"

# Tujuh butir, beserta alasan dan penggantinya. Jalur asal dipertahankan di dalam
# arsip supaya pengembalian cukup dengan memindahkan balik ke jalur yang sama.
BUTIR = [
    ("cadangan/hasil_20260902_2118",
     "Tergantikan arsip 4 September. Sepuluh berkasnya merupakan hasil S3 dan S5 dari sebelum "
     "pelaporan diperluas ke sepuluh seed; lima puluh sisanya identik dengan results/ sekarang.",
     "cadangan/prapelaksanaan_20260904_2343/"),
    ("notebooks/00_gabungan_beku.ipynb",
     "Versi 4 September, menggabungkan tiga belas notebook. Notebook keempat belas lahir sesudahnya.",
     "notebooks/00_gabungan_beku.ipynb versi baru berisi empat belas notebook"),
    ("contoh_format_notebook",
     "Contoh format notebook dari luar proyek. Tidak dirujuk naskah maupun skrip mana pun.",
     "tidak ada — memang tidak dipakai"),
    ("draftSempro/arsip",
     "Draf sempro 19 Agustus, sebelum pemisahan sempro dan semhas ditegakkan skrip.",
     "draftSempro/sempro-skripsi.md, dibangun dari NASKAH-SUMBER.md"),
    ("laporan/audit.html",
     "Laporan audit 27 Agustus dalam bentuk HTML statis, tidak dapat dijalankan ulang.",
     "scripts/periksa_sempro_bersih.py, scripts/audit_sempro.py, scripts/audit_angka_naskah.py"),
    ("draftSempro/outline-sempro-skripsi.docx",
     "Outline awal sempro. Tidak dirujuk naskah maupun skrip.",
     "struktur bab pada NASKAH-SUMBER.md"),
    ("bahan_pahaw",
     "Perjanjian lisensi PaHaW. Basis data itu dipertimbangkan lalu tidak dipakai, dan jalur "
     "aksesnya sudah dinyatakan tertutup.",
     "tidak ada — Subbab 3.3.4 menjelaskan alasannya"),
]


def sidik(path: Path, blok: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while potongan := f.read(blok):
            h.update(potongan)
    return h.hexdigest()


def ukuran(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def jumlah_berkas(path: Path) -> int:
    return 1 if path.is_file() else sum(1 for f in path.rglob("*") if f.is_file())


def gerbang() -> None:
    """Menolak berjalan bila prasyarat sitasi belum dikerjakan.

    Diuji sebagai keadaan, bukan keberadaan: frasa lama wajib hilang DAN frasa
    baru wajib ada. Salah satu saja tidak cukup.
    """
    t = NASKAH.read_text()
    masih_lama = SITASI_LAMA in t
    sudah_baru = SITASI_BARU in t
    if masih_lama or not sudah_baru:
        print("GERBANG GAGAL — prasyarat belum dikerjakan.", file=sys.stderr)
        print(f"  sitasi lama masih ada : {masih_lama}  (harus False)", file=sys.stderr)
        print(f"  sitasi baru sudah ada : {sudah_baru}  (harus True)", file=sys.stderr)
        print("\nPerbaiki sitasi pada NASKAH-SUMBER.md lebih dahulu. Tidak ada berkas "
              "yang dipindahkan.", file=sys.stderr)
        raise SystemExit(1)
    print("gerbang  : sitasi sudah diperbaiki, boleh lanjut")


def preflight(tujuan: Path) -> list[tuple[Path, str, str]]:
    """Periksa seluruh sumber dan tujuan sebelum satu berkas pun bergerak."""
    siap = []
    hilang = []
    for jalur, alasan, pengganti in BUTIR:
        p = AKAR / jalur
        if not p.exists():
            hilang.append(jalur)
        else:
            siap.append((p, alasan, pengganti))
    if hilang:
        print("PREFLIGHT GAGAL — sumber berikut tidak ditemukan:", file=sys.stderr)
        for h in hilang:
            print(f"  {h}", file=sys.stderr)
        print("\nTidak ada berkas yang dipindahkan.", file=sys.stderr)
        raise SystemExit(1)

    if tujuan.exists():
        print(f"PREFLIGHT GAGAL — tujuan sudah ada: {tujuan}", file=sys.stderr)
        raise SystemExit(1)

    butuh = sum(ukuran(p) for p, _, _ in siap)
    luang = shutil.disk_usage(AKAR).free
    if luang < butuh * 2:
        print(f"PREFLIGHT GAGAL — ruang disk kurang: butuh ~{butuh/1e6:.0f} MB, "
              f"luang {luang/1e6:.0f} MB", file=sys.stderr)
        raise SystemExit(1)

    print(f"preflight: {len(siap)} sumber ada, tujuan bebas, ruang cukup "
          f"({butuh/1e6:.1f} MB akan berpindah)")
    return siap


def pindahkan(siap, tujuan: Path) -> list[dict]:
    """Pindahkan satu per satu. Gagal-tertutup: berhenti seketika bila ada yang gagal."""
    sudah = []
    for p, alasan, pengganti in siap:
        relatif = p.relative_to(AKAR)
        sasaran = tujuan / relatif
        sasaran.parent.mkdir(parents=True, exist_ok=True)
        catat = {"asal": relatif.as_posix(), "tujuan": sasaran.relative_to(tujuan).as_posix(),
                 "alasan": alasan, "pengganti": pengganti,
                 "n_berkas": jumlah_berkas(p), "ukuran": ukuran(p)}
        try:
            shutil.move(str(p), str(sasaran))
        except Exception as e:  # noqa: BLE001 — sengaja menangkap apa pun lalu berhenti
            print(f"\nPEMINDAHAN GAGAL pada {relatif}: {e}", file=sys.stderr)
            print("Berhenti. Yang sudah berpindah dan jalur asalnya:", file=sys.stderr)
            for s in sudah:
                print(f"  {tujuan/s['tujuan']}  ->  {AKAR/s['asal']}", file=sys.stderr)
            raise SystemExit(1) from e
        sudah.append(catat)
        print(f"  pindah  {relatif.as_posix():44s} {catat['n_berkas']:4d} berkas "
              f"{catat['ukuran']/1e6:8.2f} MB")
    return sudah


def tulis_indeks(tujuan: Path, sudah: list[dict], stempel: str) -> None:
    """Delapan medan per butir, supaya arsipnya menjelaskan dirinya sendiri."""
    manifes = tujuan / "MANIFES.sha256"
    berkas = sorted(f for f in tujuan.rglob("*") if f.is_file() and f.name != "MANIFES.sha256")
    with open(manifes, "w") as f:
        for b in berkas:
            f.write(f"{sidik(b)}  {b.relative_to(tujuan).as_posix()}\n")

    baris = ["# Arsip artefak usang", "",
             f"Dibuat **{datetime.now():%d %B %Y, %H:%M}**. Seluruh isinya **dipindahkan, bukan "
             "disalin** — tidak ada duplikat yang tertinggal di tempat asal, dan tidak ada yang "
             "dihapus.", "",
             "Pengembalian cukup dengan memindahkan balik ke kolom **asal**; jalur di dalam arsip "
             "sengaja mencerminkan jalur asalnya persis.", "",
             f"Keutuhan isi dapat diperiksa terhadap `MANIFES.sha256` ({len(berkas)} berkas), yang "
             "memakai jalur **relatif terhadap akar arsip ini** sehingga tetap sah bila arsipnya "
             "dipindahkan ke disk lain:", "",
             "```", "cd <akar arsip ini> && sha256sum -c MANIFES.sha256", "```", "",
             "---", ""]
    for s in sudah:
        baris += [
            f"## `{s['asal']}`", "",
            f"- **Asal** — `{s['asal']}`",
            f"- **Tujuan** — `cadangan/usang_{stempel}/{s['tujuan']}`",
            f"- **Alasan** — {s['alasan']}",
            f"- **Pengganti** — {s['pengganti']}",
            f"- **Tanggal arsip** — {datetime.now():%d %B %Y}",
            f"- **Jumlah berkas** — {s['n_berkas']}",
            f"- **Ukuran** — {s['ukuran']/1e6:.2f} MB",
            f"- **Rujukan checksum** — `MANIFES.sha256`, awalan `{s['tujuan']}`", "",
        ]
    (tujuan / "INDEKS.md").write_text("\n".join(baris))


def tulis_indeks_cadangan(stempel: str) -> None:
    """Indeks di akar cadangan/ — menandai yang berlaku maupun yang usang.

    Menandai yang usang saja tidak cukup: orang yang membuka cadangan/ perlu tahu
    mana yang masih menjadi jaring pengaman, bukan hanya mana yang bukan.
    """
    baris = ["# Isi `cadangan/`", "",
             "Berkas ini menyatakan status tiap cadangan. Bacalah lebih dahulu sebelum memakai "
             "isi salah satunya.", "",
             "## Aturan penamaan", "",
             "| Awalan | Arti |", "|---|---|",
             "| `prapelaksanaan_<stempel>/` | **Berlaku.** Jaring pengaman yang dibuat sebelum "
             "pekerjaan besar; manifesnya dipakai memverifikasi tiap perubahan. |",
             "| `usang_<stempel>/` | **Tergantikan.** Disimpan hanya untuk penelusuran. Jangan "
             "dipakai sebagai sumber angka. |", "",
             "Cadangan tanpa salah satu awalan itu belum diklasifikasi dan harus diperiksa.", "",
             "## Isi sekarang", ""]
    # Hanya arsip prapelaksanaan TERBARU yang menjadi acuan verifikasi berjalan.
    # Yang lebih tua tetap sah sebagai jaring pengaman, tetapi manifesnya sudah
    # tertinggal — memakainya sebagai acuan akan memunculkan "hilang" palsu bagi
    # berkas yang lahir sesudahnya.
    pra = sorted(d.name for d in CADANGAN.iterdir()
                 if d.is_dir() and d.name.startswith("prapelaksanaan_"))
    terbaru = pra[-1] if pra else None

    for d in sorted(CADANGAN.iterdir()):
        if not d.is_dir():
            continue
        n = sum(1 for f in d.rglob("*") if f.is_file())
        mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
        if d.name == terbaru:
            man = d / "MANIFES.sha256"
            n_man = sum(1 for _ in open(man)) if man.exists() else 0
            status = (f"**ACUAN BERJALAN** — jaring pengaman terbaru. Manifesnya mencakup "
                      f"{n_man} berkas `results/` dan inilah yang dipakai memverifikasi tiap "
                      f"perubahan.")
        elif d.name.startswith("prapelaksanaan_"):
            man = d / "MANIFES.sha256"
            n_man = sum(1 for _ in open(man)) if man.exists() else 0
            status = (f"**HISTORIS** — jaring pengaman lama, isinya tetap sah. Manifesnya hanya "
                      f"mencakup {n_man} berkas dan **tidak lagi dipakai sebagai acuan**: berkas "
                      f"yang lahir sesudah arsip ini akan tampil sebagai hilang secara keliru.")
        elif d.name.startswith("usang_"):
            status = ("**USANG** — artefak tergantikan. Rincian per butir ada pada `INDEKS.md` "
                      "di dalamnya.")
        else:
            status = "**BELUM DIKLASIFIKASI** — periksa isinya lalu beri awalan yang sesuai."
        baris += [f"### `{d.name}/`", "", f"{status}", "",
                  f"- {n} berkas, {mb:.1f} MB", ""]
    (CADANGAN / "INDEKS.md").write_text("\n".join(baris))


def main() -> int:
    print("Mengarsipkan artefak usang. Memindahkan, bukan menghapus.\n")
    gerbang()
    stempel = datetime.now().strftime("%Y%m%d_%H%M")
    tujuan = CADANGAN / f"usang_{stempel}"
    siap = preflight(tujuan)

    tujuan.mkdir(parents=True)
    print(f"\ntujuan   : {tujuan.relative_to(AKAR)}\n")
    sudah = pindahkan(siap, tujuan)

    tulis_indeks(tujuan, sudah, stempel)
    tulis_indeks_cadangan(stempel)

    total = sum(s["ukuran"] for s in sudah)
    n = sum(s["n_berkas"] for s in sudah)
    print(f"\nselesai  : {len(sudah)} butir, {n} berkas, {total/1e6:.2f} MB berpindah")
    print(f"indeks   : {(tujuan/'INDEKS.md').relative_to(AKAR)}")
    print(f"           {(CADANGAN/'INDEKS.md').relative_to(AKAR)}")
    print(f"manifes  : {(tujuan/'MANIFES.sha256').relative_to(AKAR)}")
    print("\nTidak ada berkas yang dihapus. Pengembalian: pindahkan balik sesuai kolom asal "
          "pada INDEKS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
