"""Tiga definisi retensi berdampingan, beserta penyesuaian garis dasar yang benar.

Jalankan: python3 scripts/analisis_definisi_retensi.py

Menutup dua hal sekaligus.

**Pertama, definisi operasional retensi.** Rumusan Masalah 3 menanyakan arsitektur
mana yang paling mempertahankan performanya. Frasa itu kosong tanpa definisi
matematis, dan peringkat dapat berubah semata karena titik awal tiap arm berbeda.
Tiga definisi dihitung berdampingan, dengan **hierarki yang tidak setara**:

| Definisi | Rumus | Kedudukan |
|---|---|---|
| Selisih absolut | D = M_asal - M_lintas | **estimand primer, satu-satunya** |
| Rasio retensi | R = M_lintas / M_asal | analisis sensitivitas |
| Degradasi relatif | G = (M_asal - M_lintas) / M_asal | analisis sensitivitas |

Nilai D yang lebih kecil menandakan retensi lebih tinggi; D negatif berarti
performa pada kohort lintas justru lebih tinggi. Bila ketiganya memberi peringkat
berbeda, kesimpulan primer tetap berdasarkan D dan perbedaannya dilaporkan sebagai
temuan sensitivitas — bukan sebagai kesempatan memilih definisi yang menguntungkan.

**Kedua, koreksi terhadap kendali perancu plafon.** Perhitungan sebelumnya
(`results/b1_kontrol_plafon.csv`) meregresikan D terhadap M_asal, padahal
D = M_asal - M_lintas sehingga prediktor termuat di dalam luaran. Itu
*mathematical coupling*, bias yang dikenal sejak Oldham (1962), dan menghasilkan
hubungan negatif semu bahkan ketika tidak ada efek. Diganti penyesuaian garis
dasar dengan luaran M_lintas:

    M_lintas = b0 + b1 * M_asal + b2 * Arsitektur + e

Prosedur ini berkedudukan **analisis sensitivitas**, bukan bukti utama: unit
observasinya pasangan (arsitektur, seed), dan seed bukan unit independen tingkat
subjek. Ia memeriksa besar pengaruh titik awal, bukan menghilangkannya.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

HASIL = Path("results")
ARM = ["gru", "mamba2", "mamba3"]
NAMA = {"gru": "BiGRU", "mamba2": "BiMamba-2", "mamba3": "BiMamba-3"}


def muat() -> pd.DataFrame:
    """Gabungkan AUC kohort asal dan kohort lintas per (arsitektur, seed)."""
    s3 = pd.read_csv(HASIL / "s3_per_seed.csv")[["arsitektur", "seed", "auc"]]
    s5 = pd.read_csv(HASIL / "s5_per_seed.csv")[["arsitektur", "seed", "auc"]]
    d = s3.rename(columns={"auc": "m_asal"}).merge(
        s5.rename(columns={"auc": "m_lintas"}), on=["arsitektur", "seed"])
    d["delta"] = d.m_asal - d.m_lintas
    d["rasio"] = d.m_lintas / d.m_asal
    d["degradasi"] = (d.m_asal - d.m_lintas) / d.m_asal
    return d


def ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    sisa = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = sisa @ sisa / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    tt = beta / se
    p = 2 * (1 - tdist.cdf(np.abs(tt), dof))
    r2 = 1 - sisa @ sisa / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, p, r2, dof


def main() -> int:
    d = muat()
    n_seed = d.seed.nunique()
    print("=" * 78)
    print("DEFINISI RETENSI DAN PENYESUAIAN GARIS DASAR")
    print("=" * 78)
    print(f"\n{len(d)} pasangan (arsitektur, seed) — {d.arsitektur.nunique()} arm "
          f"x {n_seed} seed\n")

    # ---------- tiga definisi berdampingan ----------
    print("--- Tiga definisi, dihitung berdampingan ---")
    print(f"{'arm':11s} {'D (primer)':>18s} {'R (sensitivitas)':>18s} {'G (sensitivitas)':>18s}")
    ring = []
    for a in ARM:
        v = d[d.arsitektur == a]
        ring.append(dict(arsitektur=a, delta=v.delta.mean(), sb_delta=v.delta.std(ddof=1),
                         rasio=v.rasio.mean(), degradasi=v.degradasi.mean(), n_seed=len(v)))
        print(f"{NAMA[a]:11s} {v.delta.mean():+9.4f} (sb {v.delta.std(ddof=1):.4f}) "
              f"{v.rasio.mean():>18.4f} {v.degradasi.mean():>+18.4f}")

    r = pd.DataFrame(ring)
    print("\n--- Peringkat menurut tiap definisi (terbaik lebih dahulu) ---")
    pering = {
        "D primer  (kecil lebih baik)": list(r.sort_values("delta").arsitektur),
        "R sensitiv (besar lebih baik)": list(r.sort_values("rasio", ascending=False).arsitektur),
        "G sensitiv (kecil lebih baik)": list(r.sort_values("degradasi").arsitektur),
    }
    for k, v in pering.items():
        print(f"  {k}: " + " > ".join(NAMA[x] for x in v))
    stabil = len({tuple(v) for v in pering.values()}) == 1
    print(f"\n  -> peringkat {'STABIL' if stabil else 'BERBEDA'} di ketiga definisi")
    if not stabil:
        print("     Kesimpulan primer tetap berdasarkan D; perbedaannya dilaporkan")
        print("     sebagai temuan sensitivitas, bukan dasar memilih definisi.")

    # ---------- penyesuaian garis dasar ----------
    print("\n--- Penyesuaian garis dasar (menggantikan regresi D yang cacat) ---")
    print("    Luaran M_lintas, bukan D, agar prediktor tidak termuat di dalam luaran.")
    X = np.column_stack([
        np.ones(len(d)), d.m_asal.values,
        (d.arsitektur == "mamba2").astype(float), (d.arsitektur == "mamba3").astype(float)])
    beta, se, p, r2, dof = ols(X, d.m_lintas.values)
    label = ["konstanta", "M_asal", "BiMamba-2", "BiMamba-3"]
    for nm, b_, s_, p_ in zip(label, beta, se, p):
        print(f"    {nm:11s} koef {b_:+.4f}  se {s_:.4f}  p {p_:.4f}")
    print(f"    R2 {r2:.3f}, df {dof}, n {len(d)}")
    print("\n    Kedudukan: ANALISIS SENSITIVITAS terhadap perancu titik awal.")
    print("    Bukan bukti utama, dan tidak menghilangkan perancunya — seed bukan")
    print("    unit independen tingkat subjek, dan n-nya kecil.")

    # ---------- pembanding: model cacat, hanya untuk didokumentasikan ----------
    bc, _, pc, r2c, _ = ols(X, d.delta.values)
    print("\n--- Pembanding terhadap model lama, dan apa yang sebenarnya berubah ---")
    print(f"    D ~ M_asal + arsitektur memberi koef BiMamba-2 {bc[2]:+.4f} (p {pc[2]:.4f}), R2 {r2c:.3f}.")
    print("    Karena D = M_asal - M_lintas, koefisien ARSITEKTUR pada kedua model identik")
    print("    besarannya dan hanya berlawanan tanda, dengan p yang sama persis. Yang benar-benar")
    print("    tercemar mathematical coupling adalah koefisien M_asal — tergeser tepat 1,0 —")
    print("    beserta R kuadrat yang menggelembung. Kesimpulan mengenai arsitektur karena itu")
    print("    TIDAK berubah; yang dicabut adalah tafsir atas koefisien M_asal dan nilai R2.")

    r.to_csv(HASIL / "retensi_tiga_definisi.csv", index=False)
    pd.DataFrame([{"suku": nm, "koef": b_, "se": s_, "p": p_}
                  for nm, b_, s_, p_ in zip(label, beta, se, p)]
                 ).assign(r2=r2, df=dof, n=len(d), luaran="M_lintas",
                          kedudukan="sensitivitas").to_csv(
        HASIL / "retensi_penyesuaian_garis_dasar.csv", index=False)
    d.to_csv(HASIL / "retensi_per_seed.csv", index=False)
    print(f"\ndisimpan: {HASIL}/retensi_tiga_definisi.csv, "
          f"retensi_penyesuaian_garis_dasar.csv, retensi_per_seed.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
