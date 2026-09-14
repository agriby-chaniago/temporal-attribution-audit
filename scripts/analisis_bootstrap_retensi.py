"""Bootstrap tingkat subjek bagi retensi lintas kohort.

Jalankan: python3 scripts/analisis_bootstrap_retensi.py

Mengapa perlu, padahal selang antar seed sudah ada
--------------------------------------------------
Kedua selang menjawab pertanyaan yang berbeda, dan satu selang tidak mewakili
keduanya:

| Sumber ketidakpastian | Unit penganggitan ulang | Menjawab |
|---|---|---|
| Stokastisitas pelatihan | seed | seberapa hasil bergoyang bila pelatihan diulang |
| Ketidakpastian populasi subjek | **subjek** | seberapa hasil bergoyang bila subjeknya lain |

Seluruh selang yang sudah dilaporkan sejauh ini memakai seed sebagai unit,
sehingga hanya menjawab pertanyaan pertama. Berkas ini menjawab yang kedua, dan
memberi jalur bukti yang **tidak bergantung pada jumlah seed**.

Penganggitan ulang dilakukan **di dalam tiap kohort secara terpisah**, sebab
himpunan subjek kedua basis data saling lepas — tidak ada subjek yang muncul di
keduanya. Menganggit gabungannya akan mencampur dua populasi yang berbeda.

Prosedur
--------
Untuk tiap ulangan: subjek dianggit ulang dengan pengembalian di dalam tiap
kohort, AUC tingkat subjek dihitung ulang dari peluang out-of-fold yang sudah
ada, lalu Delta dan selisih antar arm dihitung. Tidak ada pelatihan ulang.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HASIL = Path("results")
ARM = ["gru", "mamba2", "mamba3"]
NAMA = {"gru": "BiGRU", "mamba2": "BiMamba-2", "mamba3": "BiMamba-3"}
N_BOOT = 5000


def peluang_subjek(logit: np.ndarray, grup: np.ndarray, y: np.ndarray):
    """Agregasi ke tingkat subjek: rata-rata logit, label pertama."""
    d = pd.DataFrame({"s": grup, "p": logit, "y": y})
    a = d.groupby("s").agg(p=("p", "mean"), y=("y", "first"))
    return a.index.values, a.p.values, a.y.values


def muat_kohort(nama_artefak: str, cache: str, modul: str):
    import sys
    sys.path.insert(0, "src")
    art = pickle.load(open(HASIL / nama_artefak, "rb"))
    if modul == "uci":
        from preprocessing import muat_cache
        rek = muat_cache(Path(cache))
    else:
        from newhandpd import muat_cache_bisp
        rek = muat_cache_bisp(Path(cache))
    y = np.array([r.label for r in rek])
    grup = np.array([str(r.subjek) for r in rek])
    return art, y, grup


def auc_boot(subjek, p, yl, idx):
    ys = yl[idx]
    return roc_auc_score(ys, p[idx]) if len(set(ys)) > 1 else np.nan


def main() -> int:
    print("=" * 78)
    print("BOOTSTRAP TINGKAT SUBJEK BAGI RETENSI")
    print("=" * 78)

    art3, y3, g3 = muat_kohort("s3_artefak.pkl", "data/cache/uci395_fs100.npz", "uci")
    art5, y5, g5 = muat_kohort("s5_artefak.pkl", "data/cache/newhandpd_spiral_fs100.npz", "bisp")

    def logit_rata(art, arm):
        v = [a["logit_oof"] for a in art if a["arsitektur"] == arm]
        return np.mean(v, axis=0), len(v)

    print(f"\nkohort asal   : {len(set(g3))} subjek")
    print(f"kohort lintas : {len(set(g5))} subjek  (himpunan saling lepas)")

    sub = {}
    for arm in ARM:
        l3, n3 = logit_rata(art3, arm)
        l5, n5 = logit_rata(art5, arm)
        sub[arm] = (peluang_subjek(l3, g3, y3), peluang_subjek(l5, g5, y5))
    print(f"seed dirata-ratakan lebih dahulu: {n3} seed asal, {n5} seed lintas\n")

    r = np.random.default_rng(0)
    n_asal = len(sub["gru"][0][0]); n_lintas = len(sub["gru"][1][0])
    delta_b = {a: [] for a in ARM}
    for _ in range(N_BOOT):
        i3 = r.integers(0, n_asal, n_asal)
        i5 = r.integers(0, n_lintas, n_lintas)
        for a in ARM:
            (s3_, p3_, y3_), (s5_, p5_, y5_) = sub[a]
            A = auc_boot(s3_, p3_, y3_, i3)
            L = auc_boot(s5_, p5_, y5_, i5)
            delta_b[a].append(A - L if np.isfinite(A) and np.isfinite(L) else np.nan)

    print("--- Delta beserta selang bootstrap TINGKAT SUBJEK ---")
    baris = []
    for a in ARM:
        v = np.array(delta_b[a], float); v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        (s3_, p3_, y3_), (s5_, p5_, y5_) = sub[a]
        titik = roc_auc_score(y3_, p3_) - roc_auc_score(y5_, p5_)
        baris.append(dict(arsitektur=a, delta=titik, ci_bawah=lo, ci_atas=hi,
                          unit="subjek", n_boot=len(v)))
        print(f"  {NAMA[a]:11s} {titik:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    print("\n--- Selisih antar arm, selang bootstrap tingkat subjek ---")
    pas = []
    for a, b in [("mamba2", "gru"), ("mamba3", "gru"), ("mamba2", "mamba3")]:
        x = np.array(delta_b[a], float); y = np.array(delta_b[b], float)
        m = np.isfinite(x) & np.isfinite(y)
        d = x[m] - y[m]
        lo, hi = np.percentile(d, [2.5, 97.5])
        lewat_nol = lo <= 0 <= hi
        pas.append(dict(arm_a=a, arm_b=b, selisih=float(d.mean()),
                        ci_bawah=lo, ci_atas=hi, memuat_nol=bool(lewat_nol), unit="subjek"))
        print(f"  {NAMA[a]:11s} - {NAMA[b]:11s} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"{'memuat nol' if lewat_nol else 'tidak memuat nol'}")

    print("\n--- Yang dijawab dan yang tidak ---")
    print("  Dijawab      : ketidakpastian akibat subjek mana yang kebetulan terpilih.")
    print("  TIDAK dijawab: stokastisitas pelatihan — itu dijawab selang antar seed,")
    print("                 dan kedua selang tidak boleh disatukan atau saling menggantikan.")

    pd.DataFrame(baris).to_csv(HASIL / "retensi_bootstrap_subjek.csv", index=False)
    pd.DataFrame(pas).to_csv(HASIL / "retensi_bootstrap_subjek_pasangan.csv", index=False)
    print(f"\ndisimpan: {HASIL}/retensi_bootstrap_subjek.csv dan _pasangan.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
