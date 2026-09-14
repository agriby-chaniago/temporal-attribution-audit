"""P1A + P3A — klaim retensi dinaikkan ke sepuluh seed. Tanpa GPU.

Jalankan: python3 scripts/analisis_p1_welch_sepuluh_seed.py

Kenapa ini analisis ulang murni, bukan eksperimen baru
------------------------------------------------------
`results/s3_keputusan.csv` mencatat `n_seed = 3, n_seed_total = 10`. Sepuluh seed
sudah dijalankan sejak `notebooks/07_eksperimen_utama.ipynb` dan tersimpan lengkap
di `s3_artefak.pkl` (30 entri) serta `retensi_per_seed.csv` (30 baris). Yang masih
berhenti di lima seed hanyalah uji Welch B1 di `notebooks/00_induk.ipynb`.

Berkas ini karena itu tidak menambah data. Ia melengkapi pelaporan atas data yang
sudah ada, dan **melaporkan angka lima seed berdampingan** supaya perpindahannya
dapat diperiksa pembaca, bukan disembunyikan.

Uji dinamai sebelum dijalankan
------------------------------
    Arah tidak diharapkan berubah. Yang diuji adalah apakah taraf nyatanya
    bertahan pada derajat bebas yang lebih besar.

Aturan keputusan konfirmatori TIDAK disentuh: ia tetap dihitung dari
SEEDS_PRAREG = [0, 1, 2] dan `s3_keputusan.csv` tidak ditulis ulang.

Tiga keluaran
-------------
P1A-1  uji Welch atas kehilangan AUC lintas kohort, 5 vs 10 seed
P1A-2  regresi perancu plafon, 15 vs 30 titik
P3A    bentuk bebas-plafon: AUC NewHandPD saja, tanpa aritmetika kehilangan.
       Perancu plafon lenyap sebab tidak ada selisih kehilangan yang dihitung.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AKAR = Path(__file__).resolve().parent.parent
HASIL = AKAR / "results"

# Pasangan uji dan taraf, dibekukan pada B1 sebelum lima seed dijalankan.
PASANGAN = [("mamba2", "gru", 0.05, True),
            ("mamba3", "gru", 0.025, False),
            ("mamba2", "mamba3", 0.025, False)]


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch t dua-sampel beserta selang kepercayaan 95 %. Persis prosedur B1."""
    t, p = stats.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(va / na + vb / nb)
    df = se**4 / (va**2 / (na**2 * (na - 1)) + vb**2 / (nb**2 * (nb - 1)))
    selisih = float(np.mean(a) - np.mean(b))
    kritis = stats.t.ppf(0.975, df)
    return {"selisih": selisih, "ci_bawah": selisih - kritis * se,
            "ci_atas": selisih + kritis * se, "t": float(t), "df": float(df),
            "p_welch": float(p)}


def ols(X: np.ndarray, y: np.ndarray, nama: list[str]) -> pd.DataFrame:
    """OLS dengan galat baku dan p dua arah. X sudah memuat kolom konstanta."""
    koef, *_ = np.linalg.lstsq(X, y, rcond=None)
    sisa = y - X @ koef
    n, k = X.shape
    dof = n - k
    s2 = float(sisa @ sisa) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    tt = koef / se
    r2 = 1 - float(sisa @ sisa) / float(((y - y.mean()) ** 2).sum())
    return pd.DataFrame({"suku": nama, "koef": koef, "se": se, "t": tt,
                         "p": 2 * stats.t.sf(np.abs(tt), dof),
                         "r2": r2, "df": dof, "n": n})


def main() -> int:
    r = pd.read_csv(HASIL / "retensi_per_seed.csv")
    print("P1A + P3A — retensi pada sepuluh seed. Tanpa GPU.\n")
    print(f"retensi_per_seed.csv: {len(r)} baris, "
          f"seed {r.seed.min()}–{r.seed.max()}, arm {sorted(r.arsitektur.unique())}\n")

    # ---------------------------------------------------------------- P1A-1
    baris = []
    for n_seed in (5, 10):
        sub = r[r.seed < n_seed]
        for a, b, alfa, primer in PASANGAN:
            w = welch(sub[sub.arsitektur == a].delta.values,
                      sub[sub.arsitektur == b].delta.values)
            baris.append({"arm_a": a, "arm_b": b, **w, "alfa": alfa,
                          "primer": primer, "nyata": w["p_welch"] < alfa,
                          "n_seed": n_seed})
    p1a = pd.DataFrame(baris)
    p1a.to_csv(HASIL / "p1a_welch_sepuluh_seed.csv", index=False)
    print("p1a_welch_sepuluh_seed.csv — uji Welch atas kehilangan AUC")
    print(p1a[["arm_a", "arm_b", "n_seed", "selisih", "ci_bawah", "ci_atas",
               "df", "p_welch", "alfa", "nyata"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"), "\n")

    # ---------------------------------------------------------------- P1A-2
    baris = []
    for n_seed in (5, 10):
        sub = r[r.seed < n_seed]
        X = np.column_stack([np.ones(len(sub)), sub.m_asal.values,
                             (sub.arsitektur == "mamba2").astype(float),
                             (sub.arsitektur == "mamba3").astype(float)])
        t = ols(X, sub.delta.values, ["konstanta", "auc_awal_uci", "mamba2", "mamba3"])
        t["n_seed"] = n_seed
        baris.append(t)
    p1a2 = pd.concat(baris, ignore_index=True)
    p1a2.to_csv(HASIL / "p1a_kontrol_plafon_30titik.csv", index=False)
    print("p1a_kontrol_plafon_30titik.csv — kehilangan ~ AUC awal + arm")
    print(p1a2[["n_seed", "suku", "koef", "se", "p", "r2", "n"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"), "\n")

    # ------------------------------------------------------------------ P3A
    # Bentuk bebas-plafon: AUC NewHandPD saja. Tidak ada aritmetika kehilangan,
    # sehingga tidak ada tempat bagi perancu plafon untuk masuk.
    baris = []
    for n_seed in (5, 10):
        sub = r[r.seed < n_seed]
        for a, b, alfa, primer in PASANGAN:
            w = welch(sub[sub.arsitektur == a].m_lintas.values,
                      sub[sub.arsitektur == b].m_lintas.values)
            baris.append({"arm_a": a, "arm_b": b, **w, "alfa": alfa,
                          "primer": primer, "nyata": w["p_welch"] < alfa,
                          "n_seed": n_seed,
                          "auc_a": float(sub[sub.arsitektur == a].m_lintas.mean()),
                          "sb_a": float(sub[sub.arsitektur == a].m_lintas.std(ddof=1)),
                          "auc_b": float(sub[sub.arsitektur == b].m_lintas.mean()),
                          "sb_b": float(sub[sub.arsitektur == b].m_lintas.std(ddof=1))})
    p3a = pd.DataFrame(baris)
    p3a.to_csv(HASIL / "p3a_bebas_plafon.csv", index=False)
    print("p3a_bebas_plafon.csv — AUC NewHandPD saja, tanpa aritmetika kehilangan")
    print(p3a[["arm_a", "arm_b", "n_seed", "auc_a", "sb_a", "auc_b", "sb_b",
               "selisih", "df", "p_welch", "nyata"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"), "\n")

    # ------------------------------------------------- ringkas untuk naskah
    print("RINGKAS — angka lima seed dan sepuluh seed berdampingan")
    pr = p1a[p1a.primer]
    for _, x in pr.iterrows():
        print(f"  Welch kehilangan mamba2−gru, {x.n_seed:2.0f} seed: "
              f"selisih {x.selisih:+.4f}  df {x.df:.2f}  p {x.p_welch:.4f}"
              f"  {'nyata' if x.nyata else 'TIDAK nyata'}")
    pr = p3a[p3a.primer]
    for _, x in pr.iterrows():
        print(f"  NewHandPD saja mamba2 vs gru, {x.n_seed:2.0f} seed: "
              f"{x.auc_a:.4f} vs {x.auc_b:.4f}  selisih {x.selisih:+.4f}"
              f"  p {x.p_welch:.4f}  {'nyata' if x.nyata else 'TIDAK nyata'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
