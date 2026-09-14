"""
Metrik lokalisasi atribusi baku, untuk menilai seberapa tepat peta atribusi
menunjuk wilayah yang menjadi kebenaran acuan.

Seluruh metrik di sini berasal dari literatur evaluasi XAI dan bukan turunan
sendiri. Definisinya mengikuti Arras, Osman, dan Samek, "CLEVR-XAI: A benchmark
dataset for the ground truth evaluation of neural network explanations",
Information Fusion 81 (2022) 14-40, serta kategori localisation pada Quantus
(Hedstrom dkk., JMLR 24 (2023) 1-11).

Latar penambahan
----------------
Naskah proposal Subbab 3.6.6 mempra-registrasi korelasi peringkat Spearman
sebagai statistik keselarasan. Kontrol positif Skenario S1 menunjukkan metrik itu
tidak sensitif terhadap konsentrasi magnitudo: pada amplitudo di atas batas
sensitivitas, 99,6 persen massa atensi jatuh di dalam jendela injeksi, namun
korelasi peringkatnya hanya 0,27. Sebabnya, atribusi bersifat runcing sehingga
mayoritas segmen di dalam jendela bernilai kecil dan berbaur peringkatnya dengan
segmen di luar jendela.

Korelasi Spearman tetap dilaporkan karena ia yang dipra-registrasi. Metrik di
modul ini dilaporkan mendampinginya sebagai perangkat baku, sehingga pembaca
tidak menyimpulkan kualitas lokalisasi hanya dari satu ukuran yang diketahui
memiliki titik buta.

Catatan tanda
-------------
Bobot atensi selalu tak negatif karena keluaran softmax. Atribusi Shapley
bertanda, sehingga metrik berbasis massa memakai bagian positifnya, mengikuti
definisi Relevance Mass Accuracy pada Quantus. Perilaku ini dikendalikan
parameter `hanya_positif` dan wajib dinyatakan saat melaporkan hasil.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = [
    "relevance_mass_accuracy",
    "relevance_rank_accuracy",
    "pointing_game",
    "auc_lokalisasi",
    "gini",
    "ringkas_lokalisasi",
]


def _siapkan(atribusi: np.ndarray, acuan: np.ndarray, hanya_positif: bool) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(atribusi, dtype=float).ravel()
    m = np.asarray(acuan, dtype=float).ravel() > 0.5
    if len(a) != len(m):
        n = min(len(a), len(m))
        a, m = a[:n], m[:n]
    if hanya_positif:
        a = np.clip(a, 0, None)
    return a, m


def relevance_mass_accuracy(atribusi: np.ndarray, acuan: np.ndarray,
                            hanya_positif: bool = True) -> float:
    """Proporsi massa atribusi yang jatuh di dalam wilayah acuan.

    Nilainya berkisar 0 sampai 1. Pembanding netralnya adalah lebar wilayah acuan
    relatif terhadap panjang rekaman: nilai yang sama dengan pembanding itu berarti
    atribusi tersebar rata dan tidak melokalisasi apa pun.

    Arras dkk. (2022), Information Fusion 81, 14-40.
    """
    a, m = _siapkan(atribusi, acuan, hanya_positif)
    total = a.sum()
    if total <= 0 or m.sum() == 0:
        return float("nan")
    return float(a[m].sum() / total)


def relevance_rank_accuracy(atribusi: np.ndarray, acuan: np.ndarray,
                            hanya_positif: bool = True) -> float:
    """Proporsi K atribusi tertinggi yang berada di dalam wilayah acuan, dengan
    K sama dengan ukuran wilayah acuan.

    Berbeda dari korelasi peringkat Spearman, metrik ini hanya memeriksa segmen
    bernilai tinggi, sehingga tidak terganggu oleh banyaknya segmen bernilai kecil
    yang peringkatnya berbaur.

    Arras dkk. (2022), Information Fusion 81, 14-40.
    """
    a, m = _siapkan(atribusi, acuan, hanya_positif)
    k = int(m.sum())
    if k == 0 or k >= len(a):
        return float("nan")
    teratas = np.argsort(a)[::-1][:k]
    return float(m[teratas].sum() / k)


def pointing_game(atribusi: np.ndarray, acuan: np.ndarray,
                  hanya_positif: bool = True) -> float:
    """Bernilai 1 bila atribusi tertinggi jatuh di dalam wilayah acuan, 0 bila tidak.

    Ukuran paling longgar di antara metrik lokalisasi: ia hanya menuntut satu
    titik puncak berada di tempat yang benar. Dilaporkan sebagai proporsi rekaman
    yang lolos.

    Zhang dkk. (2018); tersedia pada kategori localisation Quantus.
    """
    a, m = _siapkan(atribusi, acuan, hanya_positif)
    if m.sum() == 0:
        return float("nan")
    return float(m[int(np.argmax(a))])


def auc_lokalisasi(atribusi: np.ndarray, acuan: np.ndarray,
                   hanya_positif: bool = False) -> float:
    """AUC ketika atribusi diperlakukan sebagai skor detektor bagi wilayah acuan.

    Mengukur seberapa baik pemeringkatan atribusi memisahkan segmen di dalam dan
    di luar wilayah acuan, tanpa memilih ambang. Nilai 0,5 berarti setara tebakan.
    """
    a, m = _siapkan(atribusi, acuan, hanya_positif)
    if m.sum() == 0 or m.all():
        return float("nan")
    return float(roc_auc_score(m.astype(int), a))


def gini(atribusi: np.ndarray, hanya_positif: bool = True) -> float:
    """Koefisien Gini sebagai ukuran keruncingan peta atribusi.

    Tidak memerlukan kebenaran acuan. Nilai mendekati 1 berarti massa terkonsentrasi
    pada sedikit segmen, yaitu kondisi yang membuat korelasi peringkat meremehkan
    keselarasan. Dilaporkan agar keruncingan menjadi besaran terukur, bukan kesan.
    """
    a = np.asarray(atribusi, dtype=float).ravel()
    if hanya_positif:
        a = np.clip(a, 0, None)
    if a.sum() <= 0:
        return float("nan")
    a = np.sort(a)
    n = len(a)
    indeks = np.arange(1, n + 1)
    return float((2 * (indeks * a).sum()) / (n * a.sum()) - (n + 1) / n)


def ringkas_lokalisasi(atribusi: np.ndarray, acuan: np.ndarray,
                       hanya_positif: bool = True) -> dict[str, float]:
    """Seluruh metrik lokalisasi untuk satu rekaman, beserta pembanding netralnya.

    `massa_bila_rata` adalah nilai yang akan dicapai Relevance Mass Accuracy bila
    atribusi tersebar rata. Melaporkan massa tanpa pembanding ini menyesatkan,
    karena wilayah acuan yang lebar akan menghasilkan massa tinggi dengan
    sendirinya.
    """
    a, m = _siapkan(atribusi, acuan, hanya_positif)
    return {
        "massa": relevance_mass_accuracy(a, m, hanya_positif=False),
        "massa_bila_rata": float(m.mean()) if len(m) else float("nan"),
        "rank_accuracy": relevance_rank_accuracy(a, m, hanya_positif=False),
        "pointing_game": pointing_game(a, m, hanya_positif=False),
        "auc": auc_lokalisasi(a, m, hanya_positif=False),
        "gini": gini(a, hanya_positif=False),
    }
