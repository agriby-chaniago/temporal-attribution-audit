"""P3B' — retensi pada garis dasar yang dicocokkan lewat strata. Tanpa GPU.

Jalankan: python3 scripts/analisis_p3_strata_cocok.py

Kenapa berkas ini menggantikan rancangan handicap epoch
-------------------------------------------------------
Rancangan P3B semula menyamakan titik berangkat dengan melatih BiGRU lebih
pendek, dan memilih jumlah epoch lewat AUC fold latih. **Kriteria itu gagal
secara mekanis**: AUC fold latih BiMamba-2 pada 35 epoch sudah jenuh di 1,0000,
dan BiGRU mendekatinya secara monoton (0,9867 pada E=15 sampai 0,9999 pada
E=35), sehingga E yang "paling cocok" adalah 35 — yaitu tanpa handicap sama
sekali. Rancangan itu merosot menjadi perbandingan biasa dan tidak menjawab apa
pun. Kegagalannya dicatat, bukan disembunyikan, dan hasilnya tetap disimpan di
`p3b_garis_dasar_disamakan.csv` sebagai jejak jalur yang ditempuh.

Rancangan pengganti lebih baik pada dua hal sekaligus:

1. **Tidak ada intervensi.** Handicap lewat undertraining bukan intervensi
   netral: model yang dilatih lebih pendek bisa lebih atau kurang tahan karena
   regularisasi implisit, yang tidak ada hubungannya dengan plafon. Pencocokan
   strata tidak mengubah pelatihan sama sekali.
2. **Dicocokkan pada besaran yang dipersoalkan.** Perancunya adalah AUC awal
   UCI, dan di sinilah pencocokan dilakukan langsung, bukan lewat proksi.

Sepuluh seed per arm memberi sebaran AUC awal yang bertumpang tindih lebar,
sehingga perbandingan dapat dibatasi pada dukungan bersama tanpa kehilangan
banyak data.

Dua bentuk dilaporkan berdampingan
----------------------------------
`dukungan_bersama` membuang seed yang AUC awalnya di luar rentang yang dimiliki
kedua arm. `pasangan_terdekat` lebih ketat: tiap seed mamba2 dipasangkan dengan
seed gru berjarak AUC awal terkecil, tanpa pengembalian, dan pasangan yang
jaraknya melampaui batas dibuang.

Uji dinamai sebelum dijalankan
------------------------------
    Welch t dua-sampel atas kehilangan AUC per seed di dalam dukungan bersama.
    Arah diramalkan tetap. Yang diuji adalah apakah keunggulan retensi bertahan
    ketika kedua arm berangkat dari AUC UCI yang secara statistik tidak dapat
    dibedakan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AKAR = Path(__file__).resolve().parent.parent
HASIL = AKAR / "results"
PASANGAN = [("mamba2", "gru", 0.05, True),
            ("mamba3", "gru", 0.025, False),
            ("mamba2", "mamba3", 0.025, False)]
BATAS_JARAK = 0.02          # jarak AUC awal maksimum bagi pasangan terdekat


def welch(a, b) -> dict:
    t, p = stats.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(va / na + vb / nb)
    df = se**4 / (va**2 / (na**2 * (na - 1)) + vb**2 / (nb**2 * (nb - 1)))
    d = float(np.mean(a) - np.mean(b))
    k = stats.t.ppf(0.975, df)
    return {"selisih": d, "ci_bawah": d - k * se, "ci_atas": d + k * se,
            "t": float(t), "df": float(df), "p_welch": float(p)}


def pasangkan(ra: pd.DataFrame, rb: pd.DataFrame) -> tuple[list, list, list]:
    """Pasangkan tiap baris ra dengan baris rb terdekat pada m_asal, tanpa pengembalian."""
    sisa = rb.copy()
    pa, pb, jarak = [], [], []
    for _, x in ra.sort_values("m_asal").iterrows():
        if sisa.empty:
            break
        j = (sisa.m_asal - x.m_asal).abs().idxmin()
        d = abs(sisa.loc[j].m_asal - x.m_asal)
        if d <= BATAS_JARAK:
            pa.append(x); pb.append(sisa.loc[j]); jarak.append(d)
            sisa = sisa.drop(j)
    return pa, pb, jarak


def main() -> int:
    r = pd.read_csv(HASIL / "retensi_per_seed.csv")
    print("P3B' — retensi pada garis dasar dicocokkan. Tanpa GPU.\n")

    baris = []
    for a, b, alfa, primer in PASANGAN:
        ra, rb = r[r.arsitektur == a], r[r.arsitektur == b]

        # --- bentuk 1: dukungan bersama
        lo = max(ra.m_asal.min(), rb.m_asal.min())
        hi = min(ra.m_asal.max(), rb.m_asal.max())
        sa = ra[(ra.m_asal >= lo) & (ra.m_asal <= hi)]
        sb = rb[(rb.m_asal >= lo) & (rb.m_asal <= hi)]
        w = welch(sa.delta.values, sb.delta.values)
        # Uji keseimbangan: apakah garis dasarnya memang sudah tak terbedakan?
        _, p_bal = stats.ttest_ind(sa.m_asal.values, sb.m_asal.values, equal_var=False)
        baris.append({"bentuk": "dukungan_bersama", "arm_a": a, "arm_b": b, **w,
                      "n_a": len(sa), "n_b": len(sb),
                      "dasar_a": float(sa.m_asal.mean()), "dasar_b": float(sb.m_asal.mean()),
                      "selisih_dasar": float(sa.m_asal.mean() - sb.m_asal.mean()),
                      "p_keseimbangan_dasar": float(p_bal),
                      "alfa": alfa, "primer": primer, "nyata": w["p_welch"] < alfa})

        # --- bentuk 2: pasangan terdekat
        pa, pb, jd = pasangkan(ra, rb)
        if len(pa) >= 3:
            da = np.array([x.delta for x in pa]); db = np.array([x.delta for x in pb])
            w = welch(da, db)
            ma = np.array([x.m_asal for x in pa]); mb = np.array([x.m_asal for x in pb])
            _, p_bal = stats.ttest_ind(ma, mb, equal_var=False)
            baris.append({"bentuk": "pasangan_terdekat", "arm_a": a, "arm_b": b, **w,
                          "n_a": len(pa), "n_b": len(pb),
                          "dasar_a": float(ma.mean()), "dasar_b": float(mb.mean()),
                          "selisih_dasar": float(ma.mean() - mb.mean()),
                          "p_keseimbangan_dasar": float(p_bal),
                          "jarak_dasar_median": float(np.median(jd)),
                          "alfa": alfa, "primer": primer, "nyata": w["p_welch"] < alfa})

    t = pd.DataFrame(baris)
    t.to_csv(HASIL / "p3b_strata_cocok.csv", index=False)

    print("Acuan tanpa pencocokan (sepuluh seed penuh):")
    for a, b, _, _ in PASANGAN[:1]:
        ra, rb = r[r.arsitektur == a], r[r.arsitektur == b]
        print(f"  dasar UCI {a} {ra.m_asal.mean():.4f} vs {b} {rb.m_asal.mean():.4f}"
              f"  (selisih {ra.m_asal.mean()-rb.m_asal.mean():+.4f})")
        w = welch(ra.delta.values, rb.delta.values)
        print(f"  kehilangan selisih {w['selisih']:+.4f}  p {w['p_welch']:.4f}\n")

    print("p3b_strata_cocok.csv")
    print(t[["bentuk", "arm_a", "arm_b", "n_a", "n_b", "dasar_a", "dasar_b",
             "selisih_dasar", "p_keseimbangan_dasar", "selisih", "df",
             "p_welch", "nyata"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    pr = t[t.primer]
    print("\nVONIS — BiMamba-2 lawan BiGRU")
    for _, x in pr.iterrows():
        seimbang = "seimbang" if x.p_keseimbangan_dasar > 0.05 else "MASIH TIMPANG"
        print(f"  {x.bentuk:18s} n {x.n_a:.0f}v{x.n_b:.0f}  "
              f"dasar {x.dasar_a:.4f} vs {x.dasar_b:.4f} ({seimbang}, p {x.p_keseimbangan_dasar:.3f})"
              f"  kehilangan {x.selisih:+.4f}  p {x.p_welch:.4f}  "
              f"{'nyata' if x.nyata else 'TIDAK nyata'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
