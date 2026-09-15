"""Tulis ulang figures/benang_merah.{png,pdf} dengan sumber data yang benar.

Jalankan: python3 scripts/perbaiki_benang_merah.py

Kenapa skrip terpisah, bukan menjalankan ulang notebooks/00_gabungan_beku.ipynb
-------------------------------------------------------------------------------
Perbaikan sumber data sudah ditempelkan ke scripts/build_nb_master.py (baris "Kohort" pada
blok BM), sehingga notebook itu akan benar bila DIBANGUN ULANG lalu DIEKSEKUSI ULANG. Tapi
notebook itu "arsip beku" berisi jauh lebih banyak sel daripada satu figure ini, dan
menjalankannya penuh bukan wewenang skrip ini untuk memutuskan sendiri (lihat memo proyek:
build lalu serahkan dengan estimasi runtime, jangan jalankan notebook penuh atas inisiatif
sendiri). Skrip ini HANYA membaca lima CSV yang sudah ada di results/ dan memplot ulang satu
figure — bukan pelatihan, instan, aman dijalankan langsung.

Bug yang diperbaiki
--------------------
Baris "Kohort" pada tabel BM di build_nb_master.py memakai `A["s5_kls"]`
(results/s5_klasifikasi.csv), skema yang sudah digantikan. Sumber kanonis untuk kehilangan
akurasi lintas kohort adalah `A["base_lk"]` (results/a_baseline_lintas_kohort.csv) — CSV yang
sama dipakai figure baseline_lintas_kohort dan retensi_lintas_kohort untuk klaim setara.
Akibat bug: rasio yang dicetak figure lama "7 sampai 43 kali", berbeda dari klaim headline
naskah "4 sampai 95 kali". Skrip ini memverifikasi rasio baru persis 4 dan 95 sebelum menulis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR / "src"))
from viz import WARNA, TINTA, rapikan, pasang_gaya  # noqa: E402

HASIL = AKAR / "results"
GAMBAR = AKAR / "figures"


def muat(nama: str) -> pd.DataFrame:
    return pd.read_csv(HASIL / f"{nama}.csv")


def main() -> int:
    pasang_gaya()

    s7p = muat("s7_berpasangan_alpha_phi")
    s5s = muat("s5_keselarasan")
    s7s = muat("s7_kestabilan_seed")
    s3p = muat("s3_per_seed")
    base_lk = muat("a_baseline_lintas_kohort")

    bm2_p = float(s7p[s7p.arsitektur == "mamba2"].selisih_berpasangan.iloc[0])
    bm3_p = float(s7p[s7p.arsitektur == "mamba3"].selisih_berpasangan.iloc[0])
    uci_a = float(s7p[s7p.arsitektur == "mamba2"].rho_alpha_median.iloc[0])
    nhp_a = float(s5s[s5s.arsitektur == "mamba2"].rho_alpha.iloc[0])
    gru_auc = s3p[s3p.arsitektur == "gru"].auc.values
    gru_rho = s7s[s7s.arsitektur == "gru"][
        ["seed_0", "seed_1", "seed_2", "seed_3", "seed_4"]
    ].values.ravel()

    BM = [
        ("Arsitektur\nBiMamba-2 → BiMamba-3",
         abs(float(s3p[s3p.arsitektur == "mamba2"].auc.mean()
                   - s3p[s3p.arsitektur == "mamba3"].auc.mean())),
         abs(bm2_p - bm3_p)),
        ("Kohort\nUCI 395 → NewHandPD",
         abs(float(base_lk[base_lk.model == "mamba2"].kehilangan.iloc[0])),
         abs(uci_a - nhp_a)),
        ("Seed saja\n(BiGRU, data & model sama)",
         float(gru_auc.max() - gru_auc.min()),
         float(gru_rho.max() - gru_rho.min())),
        ("Resolusi patch\nP=7 → P=56", 0.0226, 0.3646),
    ]
    bm = pd.DataFrame(BM, columns=["yang diubah", "pergeseran akurasi", "pergeseran peta"])
    bm["rasio"] = bm["pergeseran peta"] / bm["pergeseran akurasi"]
    print(bm.round(4).to_string(index=False))
    print(f"\nrasio peta terhadap akurasi: {bm.rasio.min():.0f}x sampai {bm.rasio.max():.0f}x")

    lo, hi = round(bm.rasio.min()), round(bm.rasio.max())
    if (lo, hi) != (4, 95):
        raise SystemExit(
            f"rasio terhitung ({lo}x-{hi}x) tidak cocok klaim naskah (4x-95x) -- "
            "periksa ulang sumber data sebelum menulis figure"
        )

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    yy = np.arange(len(bm))[::-1]
    h = 0.34
    ax.barh(yy + h / 2, bm["pergeseran akurasi"], h, color=WARNA["biru"], label="akurasi (poin AUC)")
    ax.barh(yy - h / 2, bm["pergeseran peta"], h, color=WARNA["jingga"], label="peta (poin rho)")
    for i, (_, r) in enumerate(bm.iterrows()):
        Y = yy[i]
        ax.text(r["pergeseran akurasi"] + 0.006, Y + h / 2, f"{r['pergeseran akurasi']:.3f}",
                va="center", fontsize=8.5, color=TINTA["sekunder"])
        ax.text(r["pergeseran peta"] + 0.006, Y - h / 2,
                f"{r['pergeseran peta']:.3f}   ({r['rasio']:.0f}x)",
                va="center", fontsize=8.5, color=TINTA["utama"])
    ax.set_yticks(yy)
    ax.set_yticklabels(bm["yang diubah"], fontsize=9)
    ax.set_xlabel("besar pergeseran, satuan absolut")
    ax.set_title(f"Peta temporal runtuh {bm.rasio.min():.0f} sampai {bm.rasio.max():.0f} kali "
                 "lebih cepat daripada akurasi", loc="left", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    rapikan(ax)
    plt.tight_layout()
    for ext, kw in [("png", dict(dpi=200)), ("pdf", {})]:
        fig.savefig(GAMBAR / f"benang_merah.{ext}", bbox_inches="tight", **kw)
    print(f"\nditulis ulang: {GAMBAR / 'benang_merah.png'} dan .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
