"""Plafon kesetiaan kontrol positif pada resolusi kasar, tanpa melatih ulang apa pun.

Jalankan: python3 scripts/plafon_resolusi_kasar.py

Pertanyaan yang dijawab
-----------------------
Naskah melaporkan plafon kesetiaan alpha-phi sebesar 0,1438, diukur pada P=7 atas 37 rekaman
kontrol positif (results/s6_batas_atas_kesetiaan.csv). Subbab kesetiaan juga melaporkan kesetiaan
model P=56 sebesar 0,4509 pada data nyata, yaitu 3,1 kali plafon itu. Angka yang melampaui
plafonnya sendiri terbaca seperti kontradiksi, padahal plafon itu sendiri terikat resolusi:
diukur pada resolusi yang sama, plafonnya ikut naik.

Cara mengujinya tanpa pelatihan
-------------------------------
Memakai uji yang sama dengan dekomposisi kenaikan kesetiaan pada data nyata: peta alpha dan phi
dari model P=7 yang sudah tersimpan dibin post-hoc menjadi kelompok delapan patch (7 x 8 = 56
sampel, setara grid P=56), lalu korelasinya dihitung ulang pada rekaman yang sama. Tidak ada model
yang dilatih; skrip ini hanya membaca results/s6_kalibrasi_p7.pkl.

Batasnya dinyatakan terus terang: ini mengukur komponen MEKANIS dari kenaikan plafon, yaitu bagian
yang muncul semata karena binning lebih kasar. Model yang benar-benar dilatih pada P=56 dapat
memberi angka berbeda; memisahkan keduanya menuntut pelatihan ulang kontrol positif pada P=56.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

AKAR = Path(__file__).resolve().parent.parent
HASIL = AKAR / "results"

FAKTOR = 8  # 7 x 8 = 56 sampel per bin, setara grid P=56


def bin_rerata(v: np.ndarray, faktor: int) -> np.ndarray:
    """Rata-ratakan tiap `faktor` patch berurutan. Sisa di ekor dibuang."""
    n = len(v) // faktor
    return v[: n * faktor].reshape(n, faktor).mean(1)


def rho_median(peta: list[dict], faktor: int | None) -> tuple[float, int]:
    v = []
    for p in peta:
        a, f = np.asarray(p["alpha"]), np.asarray(p["phi"])
        if faktor is not None:
            a, f = bin_rerata(a, faktor), bin_rerata(f, faktor)
        if len(a) < 3 or np.std(a) == 0 or np.std(f) == 0:
            continue
        v.append(spearmanr(a, f).statistic)
    return float(np.median(v)), len(v)


def main() -> int:
    peta = pickle.load(open(HASIL / "s6_kalibrasi_p7.pkl", "rb"))

    halus, n_halus = rho_median(peta, None)
    kasar, n_kasar = rho_median(peta, FAKTOR)
    n_bin = [len(p["alpha"]) // FAKTOR for p in peta]

    print(f"kontrol positif, {len(peta)} rekaman sintetis")
    print(f"  plafon P=7  (apa adanya)     : {halus:.4f}   n={n_halus}")
    print(f"  plafon P=7 dibin {FAKTOR} (~P=56) : {kasar:.4f}   n={n_kasar}")
    print(f"  kenaikan                     : {kasar - halus:+.4f}  ({kasar / halus:.1f}x)")
    print(f"  jumlah bin per rekaman       : median {int(np.median(n_bin))}, "
          f"min {min(n_bin)}, maks {max(n_bin)}")

    acuan = 0.14381368289023594
    if abs(halus - acuan) > 1e-9:
        raise SystemExit(
            f"plafon P=7 terhitung {halus!r} tidak cocok nilai tercatat {acuan!r} -- "
            "artefak atau metodenya berubah, periksa sebelum memakai hasil di atas"
        )
    print("\nplafon P=7 cocok persis dengan results/s6_batas_atas_kesetiaan.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
