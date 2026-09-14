"""Estimasi dan pengujian interaksi arsitektur x kohort.

Jalankan: python3 scripts/analisis_interaksi_arsitektur_kohort.py

Pertanyaan primer penelitian ini: **apakah perubahan performa akibat perpindahan
kohort berbeda menurut arsitektur?** Membandingkan dua tabel lalu menyebutnya
interaksi tidak cukup; suku interaksinya diestimasi dan diuji.

Hierarki, dikunci sebelum analisis dijalankan
---------------------------------------------
| Uji | Status |
|---|---|
| Omnibus arsitektur x kohort | primer |
| Contrast BiMamba-2 x BiGRU | **konfirmatori apriori, satu-satunya** |
| Contrast BiMamba-2 x BiMamba-3 | eksploratori |
| Contrast BiMamba-3 x BiGRU | eksploratori |
| Uji permutasi | sekunder |
| ANOVA dua arah | sensitivitas |

Aturannya: contrast berpasangan hanya ditafsirkan setelah omnibus memenuhi
kriteria; contrast BiMamba-2 lawan BiGRU berstatus apriori sehingga berdiri
sendiri; dua contrast yang melibatkan BiMamba-3 membentuk keluarga eksploratori
dan dikoreksi Bonferroni pada taraf keluarga 0,05 dengan dua contrast, yaitu
ambang per-contrast 0,025.

Batas inferensi, ditulis bukan disembunyikan
--------------------------------------------
1. **Seed bukan unit eksperimen independen.** Ia replikasi stokastik pelatihan
   pada subjek yang sama. Inferensinya berbunyi "apakah efek melampaui derau
   pelatihan", bukan "apakah berlaku pada subjek baru".
2. **Kohort adalah faktor tetap berlevel dua**, bukan cuplikan acak dari populasi
   kohort. Klaim interaksi berlaku bagi kedua kohort ini, bukan bagi kohort pada
   umumnya. Batas ini tidak dapat diperbaiki prosedur statistik mana pun, hanya
   oleh kohort ketiga.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as fdist

HASIL = Path("results")
ARM = ["gru", "mamba2", "mamba3"]
NAMA = {"gru": "BiGRU", "mamba2": "BiMamba-2", "mamba3": "BiMamba-3"}
N_BOOT, N_PERM = 10000, 10000
ALFA_KELUARGA_EKSPLORATORI = 0.05 / 2   # dua contrast BiMamba-3


def muat() -> pd.DataFrame:
    s3 = pd.read_csv(HASIL / "s3_per_seed.csv")[["arsitektur", "seed", "auc"]]
    s5 = pd.read_csv(HASIL / "s5_per_seed.csv")[["arsitektur", "seed", "auc"]]
    a = s3.assign(kohort="asal")
    b = s5.assign(kohort="lintas")
    return pd.concat([a, b], ignore_index=True)


def delta(df: pd.DataFrame, arm: str) -> np.ndarray:
    """Delta per seed = AUC asal - AUC lintas."""
    a = df[(df.arsitektur == arm) & (df.kohort == "asal")].sort_values("seed").auc.values
    l = df[(df.arsitektur == arm) & (df.kohort == "lintas")].sort_values("seed").auc.values
    n = min(len(a), len(l))
    return a[:n] - l[:n]


def omnibus(D: dict[str, np.ndarray]) -> tuple[float, float, float, int, int]:
    """Uji F satu arah atas Delta antar arsitektur.

    Suku interaksi arsitektur x kohort pada rancangan dua faktor setara dengan
    efek arsitektur atas Delta, sebab Delta sudah merupakan selisih antar kohort.
    """
    grup = [D[a] for a in ARM]
    semua = np.concatenate(grup)
    k, N = len(grup), len(semua)
    ss_antar = sum(len(g) * (g.mean() - semua.mean()) ** 2 for g in grup)
    ss_dalam = sum(((g - g.mean()) ** 2).sum() for g in grup)
    df1, df2 = k - 1, N - k
    F = (ss_antar / df1) / (ss_dalam / df2)
    p = 1 - fdist.cdf(F, df1, df2)
    eta2 = ss_antar / (ss_antar + ss_dalam)      # ukuran efek
    return F, p, eta2, df1, df2


def boot_ci(x: np.ndarray, y: np.ndarray, rs: int = 0) -> tuple[float, float, float]:
    """Selang bootstrap bagi selisih rerata Delta. Unit: SEED."""
    r = np.random.default_rng(rs)
    d = x.mean() - y.mean()
    b = [r.choice(x, len(x), True).mean() - r.choice(y, len(y), True).mean()
         for _ in range(N_BOOT)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return d, lo, hi


def perm_p(x: np.ndarray, y: np.ndarray, rs: int = 1) -> float:
    r = np.random.default_rng(rs)
    obs = abs(x.mean() - y.mean())
    s = np.concatenate([x, y]); k = len(x)
    c = sum(abs(p[:k].mean() - p[k:].mean()) >= obs - 1e-12
            for p in (r.permutation(s) for _ in range(N_PERM)))
    return (c + 1) / (N_PERM + 1)


def main() -> int:
    df = muat()
    D = {a: delta(df, a) for a in ARM}
    n = len(D["gru"])
    print("=" * 78)
    print("INTERAKSI ARSITEKTUR x KOHORT")
    print("=" * 78)
    print(f"\nRancangan {len(ARM)} arsitektur x 2 kohort x {n} seed = {len(ARM)*2*n} pengamatan")
    print("Ukuran luaran: AUC tingkat subjek. Estimand: efek interaksi terhadap Delta.\n")

    print("--- Delta per arsitektur (asal dikurangi lintas) ---")
    for a in ARM:
        print(f"  {NAMA[a]:11s} {D[a].mean():+.4f} (sb {D[a].std(ddof=1):.4f}, n={len(D[a])})")

    print("\n--- [PRIMER] Omnibus arsitektur x kohort ---")
    F, p, eta2, df1, df2 = omnibus(D)
    print(f"  F({df1},{df2}) = {F:.3f}   p = {p:.4f}   eta kuadrat = {eta2:.4f}")
    lolos = p < 0.05
    print(f"  -> omnibus {'MEMENUHI' if lolos else 'TIDAK memenuhi'} kriteria; "
          f"contrast {'ditafsirkan' if lolos else 'TIDAK ditafsirkan'} di bawah")

    print("\n--- Contrast interaksi berpasangan ---")
    baris = []
    for a, b, status, ambang in [
            ("mamba2", "gru", "konfirmatori apriori", 0.05),
            ("mamba2", "mamba3", "eksploratori", ALFA_KELUARGA_EKSPLORATORI),
            ("mamba3", "gru", "eksploratori", ALFA_KELUARGA_EKSPLORATORI)]:
        d_, lo, hi = boot_ci(D[a], D[b])
        pp = perm_p(D[a], D[b])
        nyata = pp < ambang and lolos
        baris.append(dict(arm_a=a, arm_b=b, status=status, selisih_delta=d_,
                          ci_bawah=lo, ci_atas=hi, p_permutasi=pp, ambang=ambang,
                          nyata=bool(nyata), n_seed=n))
        print(f"  {NAMA[a]:11s} - {NAMA[b]:11s} {d_:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"p_perm={pp:.4f} ambang={ambang:.3f}  [{status}]  "
              f"{'nyata' if nyata else 'tidak nyata'}")

    print("\n--- Batas inferensi ---")
    print("  1. Seed = replikasi stokastik pelatihan pada subjek yang sama, bukan")
    print("     unit independen tingkat subjek. Inferensi berlaku terhadap derau")
    print("     pelatihan, bukan terhadap subjek baru.")
    print("  2. Kohort = faktor tetap berlevel dua, bukan cuplikan acak. Klaim")
    print("     interaksi berlaku bagi KEDUA KOHORT INI, bukan kohort pada umumnya.")

    pd.DataFrame(baris).to_csv(HASIL / "interaksi_contrast.csv", index=False)
    pd.DataFrame([{"uji": "omnibus", "F": F, "df1": df1, "df2": df2, "p": p,
                   "eta_kuadrat": eta2, "n_seed": n,
                   "catatan": "estimand: efek interaksi arsitektur x kohort terhadap Delta"}
                  ]).to_csv(HASIL / "interaksi_omnibus.csv", index=False)
    print(f"\ndisimpan: {HASIL}/interaksi_omnibus.csv, interaksi_contrast.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
