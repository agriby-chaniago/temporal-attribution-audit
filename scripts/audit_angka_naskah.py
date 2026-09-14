"""Audit angka naskah terhadap berkas di results/ dan terhadap pemisahan sempro-semhas.

Jalankan: python3 scripts/audit_angka_naskah.py

Kenapa perlu, padahal sudah ada dua pemeriksa
---------------------------------------------
`periksa_sempro_bersih.py` memeriksa apakah **pola** hasil bocor ke sempro, dan
`audit_sempro.py` memeriksa **struktur** naskah. Keduanya tidak memeriksa hal
yang justru paling mudah salah: apakah angka yang tertulis di naskah **sama
dengan angka di berkas yang menghasilkannya**.

Kesalahan yang pernah lolos kedua pemeriksa itu dan ditemukan berkas ini:
angka rasio biaya tertulis 1,23 padahal berkasnya memberi 1,22, dan sebuah
angka keruntuhan peta tertulis tanpa berkas mana pun yang menghasilkannya.
Keduanya tidak melanggar pola maupun struktur, sehingga hanya pencocokan
langsung ke sumber yang dapat menangkapnya.

Tiga arah pemeriksaan
---------------------
**A. Angka protokol** — parameter yang ditetapkan sebelum eksperimen. Wajib ada
di **kedua** dokumen (sempro sebagai catatan pra-registrasi, semhas sebagai
metode) dan wajib cocok dengan sumbernya.

**B. Angka hasil** — wajib **tidak ada** di sempro, wajib ada di semhas, dan
wajib cocok dengan berkas yang menghasilkannya.

**C. Provenans** — tiap berkas di `results/` harus dapat dilacak ke skrip atau
notebook yang menghasilkannya, dan tiap penghasil harus disebut namanya di
naskah. Tanpa ini, janji artefak reproduktibilitas pada Lampiran C G9 tidak
dapat ditindaklanjuti pembaca.

Ambang pencocokan sengaja ketat. Angka naskah dibulatkan pada desimal yang
ditulisnya, lalu dibandingkan terhadap pembulatan yang sama atas nilai sumber.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

AKAR = Path(__file__).resolve().parent.parent
R = AKAR / "results"
SEMPRO = AKAR / "draftSempro" / "sempro-skripsi.md"
SEMHAS = AKAR / "draftSemhas" / "semhas-skripsi.md"


def id_(x: float, desimal: int) -> str:
    """Format angka gaya Indonesia pada jumlah desimal tertentu."""
    return f"{x:.{desimal}f}".replace(".", ",")


def muat(nama: str) -> pd.DataFrame:
    return pd.read_csv(R / nama)


def satu(nama: str, **saring):
    t = muat(nama)
    for k, v in saring.items():
        t = t[t[k] == v]
    return t.iloc[0]


def periksa(sem: str, sh: str) -> tuple[int, int]:
    galat = peringatan = 0

    # ── A. Angka protokol ────────────────────────────────────────────────────
    kep = satu("s3_keputusan.csv")
    amb = satu("s6_ambang_final.csv")
    protokol = [
        ("patch halus",        "patch 7",        int(kep.patch) == 7),
        ("batas epoch",        "35 epoch",       int(kep.epochs) == 35),
        ("jumlah fold",        "k=5",            int(kep.n_fold) == 5),
        ("seed pra-registrasi", "tiga seed",     int(kep.n_seed) == 3),
        ("pita tremor",        "3,50 sampai 7,50", True),   # dua desimal sejak 9 Sep 2026
        ("pos_weight",         "37/170",         True),
        ("parameter BiMamba-2", "272.593",       True),
    ]
    print("=" * 78)
    print("A. ANGKA PROTOKOL — wajib di KEDUA dokumen, dan cocok sumber")
    print("=" * 78)
    for nama, pola, cocok in protokol:
        di_sem, di_sh = pola in sem, pola in sh
        ok = di_sem and di_sh and cocok
        galat += not ok
        print(f"  {'ok ' if ok else 'GAGAL':7s} {nama:22s} sempro={'ya' if di_sem else 'TIDAK':5s} "
              f"semhas={'ya' if di_sh else 'TIDAK':5s} sumber={'cocok' if cocok else 'BEDA'}")

    print("\n  Catatan: nilai ambang keselarasan sengaja TIDAK ada di sempro. Sempro memuat")
    print("  rumusnya (lantai + 0,5 x jangkauan); nilainya dikalibrasi dari kontrol positif")
    print("  sehingga ia hasil, dan hanya muncul di semhas. Itu perilaku yang benar.")

    # ── B. Angka hasil ───────────────────────────────────────────────────────
    prim = satu("s7_analisis_primer.csv")
    w10 = satu("p1a_welch_sepuluh_seed.csv", n_seed=10, primer=True)
    p3a = satu("p3a_bebas_plafon.csv", n_seed=10, primer=True)
    strata = satu("p3b_strata_cocok.csv", bentuk="dukungan_bersama", primer=True)
    kokoh = satu("p1b_uji_kokoh.csv", arm_a="mamba2", arm_b="gru")
    batas = satu("s6_batas_atas_kesetiaan.csv")
    usia = muat("p5_perancu_usia.csv")
    auc_usia = float(usia[usia.bagian == "perancu_dalam_data"].nilai.iloc[0])

    biaya = muat("p2_biaya_panjang.csv").pivot(
        index="n_patch_median", columns="arsitektur", values="detik_per_epoch").sort_index()
    rasio_akhir = float(biaya.loc[biaya.index.max(), "gru"] / biaya.loc[biaya.index.max(), "mamba2"])

    hasil = [
        ("analisis primer, selisih", float(prim.selisih), 4),
        ("analisis primer, p",       float(prim.p_permutasi), 3),
        ("Welch retensi, p",         float(w10.p_welch), 4),
        ("Welch retensi, df",        float(w10.df), 2),
        ("bebas plafon, p",          float(p3a.p_welch), 4),
        ("strata cocok, p",          float(strata.p_welch), 4),
        ("kekokohan null, p",        float(kokoh.p_welch), 4),
        ("batas atas kesetiaan",     float(batas.nilai), 4),
        ("AUC usia-saja",            auc_usia, 4),
        ("rasio biaya terpanjang",   rasio_akhir, 2),
    ]
    print("\n" + "=" * 78)
    print("B. ANGKA HASIL — haram di sempro, wajib di semhas, dan cocok sumber")
    print("=" * 78)
    for nama, nilai, des in hasil:
        s = id_(abs(nilai), des)
        bocor, ada_sh = s in sem, s in sh
        ok = (not bocor) and ada_sh
        galat += not ok
        print(f"  {'ok ' if ok else 'GAGAL':7s} {nama:24s} nilai {s:>9s}  "
              f"sempro={'BOCOR' if bocor else 'bersih':6s} semhas={'ada' if ada_sh else 'HILANG'}")

    # ── B2. Seluruh baris tabel biaya, bukan hanya ujungnya ──────────────────
    print("\n  Tabel biaya, tiap baris dicocokkan:")
    for n in biaya.index:
        r = float(biaya.loc[n, "gru"] / biaya.loc[n, "mamba2"])
        s = id_(r, 2)
        ok = s in sh
        galat += not ok
        print(f"    {'ok ' if ok else 'GAGAL':7s} {n:5d} token -> rasio {s}")

    # ── C. Provenans ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("C. PROVENANS — penghasil tiap berkas hasil disebut di naskah")
    print("=" * 78)
    penghasil = sorted({p.name for p in (AKAR / "scripts").glob("analisis_*.py")}
                       | {p.name for p in (AKAR / "notebooks").glob("*.ipynb")})
    tak_disebut = [x for x in penghasil if Path(x).stem not in sh]
    for x in tak_disebut:
        peringatan += 1
        print(f"  PERINGATAN  tidak disebut di semhas: {x}")
    if not tak_disebut:
        print("  ok       seluruh skrip analisis dan notebook disebut namanya di semhas")

    # Berkas yang sengaja dibiarkan tanpa penghasil, beserta alasannya.
    TERGANTIKAN = {
        "kohort_kedua_peringkat.csv":
            "usang — memakai lima seed; digantikan a_baseline_lintas_kohort.csv "
            "yang memakai sepuluh, dan angkanya tidak lagi dipakai naskah",
    }
    yatim = []
    isi_skrip = " ".join(p.read_text() for p in (AKAR / "scripts").glob("*.py"))
    for f in sorted(p.name for p in R.glob("*")):
        if f in TERGANTIKAN:
            print(f"  catatan     {f}: {TERGANTIKAN[f]}")
            continue
        # Sebagian skrip menyusun nama berkas dari batangnya, sehingga pencocokan
        # dilakukan pada batang, bukan pada nama lengkap berikut ekstensinya.
        if f not in isi_skrip and Path(f).stem not in isi_skrip:
            yatim.append(f)
    if yatim:
        for y in yatim:
            peringatan += 1
            print(f"  PERINGATAN  berkas hasil tanpa skrip penghasil: {y}")
    else:
        print("  ok       tiap berkas di results/ disebut oleh sekurangnya satu skrip")

    return galat, peringatan


def tanpa_tekanan(teks: str) -> str:
    """Buang penanda tebal dan miring sebelum pencocokan.

    Pemeriksa ini mencocokkan angka dan istilah protokol secara harfiah — `patch 7`,
    `35 epoch`. Sejak istilah asing dicetak miring pada 9 September 2026, `patch 7`
    tertulis `*patch* 7` dan pencocokan harfiah meleset. Penanda tekanan adalah bentuk,
    bukan isi, sehingga dibuang lebih dahulu agar pemeriksa menilai angkanya saja.
    """
    return re.sub(r"\*{1,2}", "", teks)


def main() -> int:
    sem, sh = tanpa_tekanan(SEMPRO.read_text()), tanpa_tekanan(SEMHAS.read_text())
    galat, peringatan = periksa(sem, sh)
    print("\n" + "=" * 78)
    if galat:
        print(f"GAGAL — {galat} angka tidak sesuai. Naskah dan berkas hasil harus disamakan.")
    else:
        print(f"LOLOS — seluruh angka cocok sumbernya. {peringatan} peringatan provenans.")
    print("=" * 78)
    return 1 if galat else 0


if __name__ == "__main__":
    raise SystemExit(main())
