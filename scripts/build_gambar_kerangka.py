"""Bangun dua gambar kerangka untuk Bab II: kerangka teori dan kerangka konsep.

Jalankan: python3 scripts/build_gambar_kerangka.py

Kenapa matplotlib, bukan pembangkit gambar
-------------------------------------------
Skill `scientific-schematics` dipertimbangkan dan ditolak. Ia mengirim prompt beserta
gambarnya ke layanan luar, tidak memberi kendali DPI, bersifat non-deterministik, dan
dokumentasinya sendiri memperingatkan bahwa model gambar salah mengeja label. Kerangka
teori di bawah memuat **sitasi di dalam kotaknya**; nama penulis yang salah eja pada
gambar skripsi adalah cacat yang tidak dapat dipertanggungjawabkan.

Seluruh gambar proyek ini dibangun dari skrip dan dapat dihasilkan ulang bit demi bit.
Kedua gambar ini mengikuti aturan itu.

Bentuk mengikuti Lampiran 18 dan 19 Panduan Tugas Akhir UHB
-----------------------------------------------------------
Lampiran 18: bagan kotak berpanah, tiap kotak memuat konsep beserta sitasinya, tiap panah
berlabel relasi. Lampiran 19: dua kotak, variabel bebas di kiri menunjuk variabel terikat
di kanan, disertai keterangan arti kotaknya.

Judul gambar tidak dibakar ke dalam berkas gambar. Panduan menuntut judul diketik **di
bawah** gambar dengan spasi 1,15 dan tanpa cetak tebal, sehingga ia menjadi teks naskah,
bukan piksel.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

AKAR = Path(__file__).resolve().parent.parent
GAMBAR = AKAR / "figures"

# Times New Roman jarang terpasang di Linux; DejaVu Serif adalah serif baku matplotlib dan
# tetap serif, sehingga gambar tidak bertabrakan gaya dengan naskah yang berhuruf serif.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

GARIS = "#222222"
LATAR = "#ffffff"


# Kotak digambar matplotlib sendiri lewat `bbox` milik teksnya, sehingga ia tidak mungkin
# lebih kecil daripada isinya. Tiga percobaan sebelumnya menghitung tinggi kotak secara
# terpisah dari teks — pada percobaan pertama kotak lolos ke luar kanvas, pada kedua baris
# terakhir meluber ke bawah bingkai, pada ketiga judul menimpa badan. Ketiganya kesalahan
# yang sama: dua sumber kebenaran bagi satu tinggi.
#
# Sumbu-y bersatuan INCI, sehingga tinggi kotak dapat dihitung tepat dari ukuran huruf.
UKURAN, SPASI, PAD = 7.8, 1.45, 0.7


def tinggi_inci(n_baris: int) -> float:
    return (n_baris * SPASI + 2 * PAD) * UKURAN / 72


def kerangka_teori() -> Path:
    tahap = [
        ("MASALAH",
         "Model deteksi Parkinson dari tulisan tangan menghasilkan satu label per rekaman "
         "(Drotár dkk., 2016; Pereira dkk., 2016; Diaz dkk., 2021), sehingga keterangan "
         "mengenai kapan gangguan motorik terjadi hilang.", "menuntut"),
        ("GAGASAN",
         "Attention pooling menghasilkan bobot per segmen waktu (Ilse dkk., 2018), sehingga "
         "lokalisasi temporal dapat diperoleh tanpa anotasi per satuan waktu.", "menimbulkan"),
        ("KERAGUAN",
         "Bobot atensi belum tentu menunjuk tempat yang benar (Jain & Wallace, 2019; "
         "Wiegreffe & Pinter, 2019; Serrano & Smith, 2019).", "dijawab lewat"),
        ("PENGUJIAN DUA LAPIS",
         "Lapis pertama menguji kesetiaan atensi terhadap atribusi Shapley (Lundberg & Lee, "
         "2017); lapis kedua menguji keselarasan atribusi terhadap penanda motorik yang "
         "diturunkan dari literatur klinis.", "disahkan oleh"),
        ("KONTROL POSITIF",
         "Sinyal buatan disuntikkan pada lokasi yang diketahui. Pipeline yang tidak "
         "menemukannya dinyatakan rusak sebelum data pasien disentuh.", "menghasilkan"),
        ("EMPAT KEMUNGKINAN HASIL",
         "Keempatnya ditetapkan sebelum data dilihat dan seluruhnya dilaporkan apa adanya "
         "(Subbab III.G.4).", None),
    ]

    LEBAR, JARAK, MARGIN = 64, 0.34, 0.05
    isi = [j + "\n" + "\n".join(textwrap.wrap(b, LEBAR)) for j, b, _ in tahap]
    tinggi = [tinggi_inci(x.count("\n") + 1) for x in isi]
    H = sum(tinggi) + JARAK * (len(tahap) - 1) + 2 * MARGIN

    fig, ax = plt.subplots(figsize=(6.2, H))
    ax.set_xlim(0, 1); ax.set_ylim(0, H); ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    y = H - MARGIN
    for teks, h, (_, _, relasi) in zip(isi, tinggi, tahap):
        ax.text(0.5, y - h / 2, teks, ha="center", va="center", fontsize=UKURAN,
                linespacing=SPASI, zorder=3,
                bbox=dict(boxstyle=f"round,pad={PAD}", facecolor=LATAR,
                          edgecolor=GARIS, linewidth=0.9))
        y -= h
        if relasi:
            ax.annotate("", xy=(0.5, y - JARAK), xytext=(0.5, y),
                        arrowprops=dict(arrowstyle="-|>", linewidth=0.9, color=GARIS,
                                        shrinkA=0, shrinkB=0), zorder=1)
            ax.text(0.515, y - JARAK / 2, relasi, ha="left", va="center",
                    fontsize=UKURAN - 0.6, style="italic", color=GARIS, zorder=3)
            y -= JARAK

    jalur = GAMBAR / "f_kerangka_teori.png"
    fig.savefig(jalur, facecolor="white")
    plt.close(fig)
    return jalur


def kerangka_konsep() -> Path:
    fig, ax = plt.subplots(figsize=(6.1, 3.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    bebas = ("Variabel bebas\n\nArsitektur encoder, tiga taraf:\n"
             "1.  Bidirectional GRU (baseline)\n"
             "2.  Bidirectional Mamba-2 (diusulkan)\n"
             "3.  Bidirectional Mamba-3 (eksploratori)")
    terikat = ("Variabel terikat\n\n"
               "1.  Performa klasifikasi\n"
               "2.  Kesetiaan atensi\n"
               "3.  Keselarasan terhadap penanda motorik\n"
               "4.  Retensi lintas kohort\n"
               "5.  Reproduktibilitas peta antar seed")

    for x, teks in ((0.255, bebas), (0.745, terikat)):
        ax.add_patch(FancyBboxPatch(
            (x - 0.215, 0.34), 0.43, 0.50,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            linewidth=0.9, edgecolor=GARIS, facecolor=LATAR, zorder=2))
        ax.text(x - 0.195, 0.80, teks, ha="left", va="top", fontsize=7.8,
                linespacing=1.5, zorder=3)

    ax.annotate("", xy=(0.528, 0.59), xytext=(0.472, 0.59),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.0, color=GARIS))

    ax.add_patch(FancyBboxPatch(
        (0.055, 0.115), 0.055, 0.058,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=0.9, edgecolor=GARIS, facecolor=LATAR))
    ax.text(0.045, 0.205, "Keterangan:", ha="left", va="center", fontsize=7.8, fontweight="bold")
    ax.text(0.130, 0.144, ":  Diteliti", ha="left", va="center", fontsize=7.8)

    fig.tight_layout(pad=0.2)
    jalur = GAMBAR / "f_kerangka_konsep.png"
    fig.savefig(jalur, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return jalur


def main() -> int:
    GAMBAR.mkdir(exist_ok=True)
    for bangun in (kerangka_teori, kerangka_konsep):
        j = bangun()
        print(f"  {j.relative_to(AKAR)}  {j.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
