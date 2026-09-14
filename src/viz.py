"""
Gaya dan pembantu visualisasi, dipakai bersama seluruh notebook.

Palet mengikuti urutan tetap dan sudah divalidasi terhadap keterbacaan bagi
pembaca dengan defisiensi penglihatan warna: pemisahan CVD terburuk pada seluruh
pasangan adalah delta-E 9,2 (deuteranopia) dan 24,0 pada penglihatan normal.

Aturan yang dipatuhi di seluruh gambar:
- Identitas tidak pernah bersandar pada warna saja; selalu ada legenda, penanda
  bentuk, atau label langsung.
- Teks memakai warna tinta, bukan warna seri.
- Kisi dan sumbu bersifat resesif terhadap data.
- Tidak ada sumbu ganda. Dua besaran berskala beda digambar terpisah.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ["WARNA", "TINTA", "pasang_gaya", "rapikan", "label_langsung", "pita_tremor"]

# Slot kategorikal dengan urutan tetap, tidak pernah didaur ulang.
WARNA = {
    "biru": "#2a78d6",      # slot 1 — besaran utama yang diamati
    "jingga": "#eb6834",    # slot 2 — kebenaran acuan / lokasi injeksi
    "toska": "#1baf7a",     # slot 3 — pembanding ketiga
    "netral": "#8a8a85",    # distribusi nol, garis acuan
    "redup": "#c9c9c4",     # kisi, elemen resesif
}

TINTA = {
    "utama": "#0b0b0b",
    "sekunder": "#52514e",
    "redup": "#8a8a85",
}

# Pita tremor dari Subbab 2.2.1 naskah.
pita_tremor = (3.5, 7.5)


def pasang_gaya() -> None:
    """Terapkan gaya dasar: kisi resesif, bingkai minimal, teks berwarna tinta."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": TINTA["redup"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": TINTA["sekunder"],
        "axes.titlecolor": TINTA["utama"],
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": WARNA["redup"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "xtick.color": TINTA["redup"],
        "ytick.color": TINTA["redup"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "font.size": 10,
    })


def rapikan(ax, atas: bool = False, kanan: bool = False) -> None:
    """Buang bingkai yang tidak membawa informasi, kisi hanya pada sumbu y."""
    ax.spines["top"].set_visible(atas)
    ax.spines["right"].set_visible(kanan)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)


def label_langsung(ax, x, y, teks: str, warna: str, dx: float = 0.0, dy: float = 0.0,
                   ha: str = "left", va: str = "center") -> None:
    """Label seri di ujung garis, memakai warna tinta bukan warna seri.

    Warna seri dibawa oleh penanda di sebelahnya, bukan oleh teksnya, agar teks
    tetap terbaca pada latar apa pun dan identitas tidak bersandar pada warna saja.
    """
    ax.annotate(teks, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=9, color=TINTA["sekunder"])
    ax.plot([x], [y], "o", color=warna, markersize=6, zorder=5,
            markeredgecolor="white", markeredgewidth=1.2)
