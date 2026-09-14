"""P3B — retensi lintas kohort dengan garis dasar disamakan secara rancangan.

Jalankan: python3 scripts/analisis_p3_garis_dasar.py     (GPU, ~1,3 jam)

Masalah yang diserang
---------------------
Klaim retensi lintas kohort bertahan, tetapi separuh kekuatannya hilang begitu
perancu plafon dikendalikan: BiGRU memulai lebih tinggi (AUC UCI 0,9503 berbanding
0,9338 pada sepuluh seed) sehingga punya lebih banyak untuk hilang. Koefisien AUC
awal +1,0892 dengan p 0,0000 pada regresi 30 titik — perancunya nyata dan besar.

Sampai kini koreksinya dilakukan lewat **regresi post-hoc**. Berkas ini menguji
klaim yang sama lewat **rancangan**: samakan titik berangkatnya, lalu bandingkan.

Aturan penyamaan, dibekukan sebelum AUC uji dilihat
---------------------------------------------------
BiGRU dilatih pada grid epoch {15, 20, 25, 30, 35}. Dipilih E yang membuat AUC
**fold latih** BiGRU paling dekat dengan AUC fold latih BiMamba-2 pada 35 epoch.
Pemilihan memakai data latih saja; AUC uji tidak menyentuh pemilihan sama sekali.
`src/training.py` tidak diubah — penyamaan dikerjakan lewat panggilan `latih`
terpisah, bukan lewat kait early-stop.

Batasan yang harus dinyatakan terus terang
------------------------------------------
Menghandikap BiGRU dengan melatihnya lebih pendek **bukan intervensi netral**.
Model yang dilatih lebih pendek bisa jadi lebih tahan atau kurang tahan karena
sebab yang tidak ada hubungannya dengan plafon — regularisasi implisit, misalnya.
P3B karena itu **menukar satu perancu dengan perancu lain**, dan dilaporkan
sebagai jalur bukti kedua yang kelemahannya berbeda, bukan sebagai koreksi yang
lebih murni. Jalur primer tetap P3A, yang tidak memakai aritmetika kehilangan
sama sekali.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR / "src"))

from artefak import muat_artefak, simpan_atomik  # noqa: E402
from model import PDClassifier  # noqa: E402
from newhandpd import muat_cache_bisp  # noqa: E402
from preprocessing import Normalisasi, muat_cache  # noqa: E402
from training import latih, prediksi  # noqa: E402

HASIL = AKAR / "results"
P = 7
GRID_EPOCH = [15, 20, 25, 30, 35]
E_ACUAN = 35
SEEDS = [0, 1, 2, 3, 4]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def auc_subjek(logit, grup, y):
    subj = sorted(set(grup))
    ps = np.array([logit[grup == s].mean() for s in subj])
    ys = np.array([y[grup == s][0] for s in subj])
    return float(roc_auc_score(ys, ps))


def jalankan(rek, arsitektur, seed, epochs) -> dict:
    """Kembalikan AUC fold latih (untuk memilih E) dan AUC uji OOF (untuk vonis)."""
    y = np.array([r.label for r in rek])
    grup = np.array([r.subjek for r in rek])
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    logit = np.zeros(len(rek))
    auc_latih = []
    for i_latih, i_uji in skf.split(np.zeros(len(rek)), y, grup):
        norm = Normalisasi().fit([rek[i] for i in i_latih])
        torch.manual_seed(seed)
        m = PDClassifier(encoder=arsitektur, patch_size=P).to(DEV)
        latih(m, rek, list(i_latih), norm, epochs=epochs, device=DEV, seed=seed)
        lo_l, _ = prediksi(m, rek, list(i_latih), norm, device=DEV)
        auc_latih.append(auc_subjek(lo_l, grup[i_latih], y[i_latih]))
        lo_u, _ = prediksi(m, rek, list(i_uji), norm, device=DEV)
        logit[i_uji] = lo_u
    return {"arsitektur": arsitektur, "seed": seed, "epochs": epochs,
            "auc_latih": float(np.mean(auc_latih)),
            "auc_uji": auc_subjek(logit, grup, y)}


def main() -> int:
    print(f"P3B — penyamaan garis dasar. {DEV}, patch {P}.\n")
    uci = muat_cache(AKAR / "data" / "cache" / "uci395_fs100.npz")
    nhp = muat_cache_bisp(AKAR / "data" / "cache" / "newhandpd_spiral_fs100.npz")

    art_path = HASIL / "p3b_artefak.pkl"
    art = muat_artefak(art_path) or []

    def perlu(kohort, arsitektur, seed, epochs):
        return not any(a["kohort"] == kohort and a["arsitektur"] == arsitektur
                       and a["seed"] == seed and a["epochs"] == epochs for a in art)

    def tambah(kohort, rek, arsitektur, seed, epochs):
        if not perlu(kohort, arsitektur, seed, epochs):
            return
        t0 = time.time()
        h = jalankan(rek, arsitektur, seed, epochs)
        art.append({"kohort": kohort, **h})
        simpan_atomik(art_path, art)
        print(f"  {kohort:9s} {arsitektur:7s} E={epochs:2d} seed={seed}  "
              f"{time.time()-t0:6.1f}s  latih={h['auc_latih']:.4f} "
              f"uji={h['auc_uji']:.4f}", flush=True)

    # Tahap 1 — acuan BiMamba-2 pada 35 epoch, UCI.
    print("tahap 1: acuan BiMamba-2 pada UCI, 35 epoch")
    for s in SEEDS:
        tambah("uci395", uci, "mamba2", s, E_ACUAN)

    # Tahap 2 — grid epoch BiGRU pada UCI.
    print("\ntahap 2: grid epoch BiGRU pada UCI")
    for e in GRID_EPOCH:
        for s in SEEDS:
            tambah("uci395", uci, "gru", s, e)

    t = pd.DataFrame(art)
    acuan = t.query("kohort == 'uci395' and arsitektur == 'mamba2' "
                    f"and epochs == {E_ACUAN}").auc_latih.mean()
    kandidat = (t.query("kohort == 'uci395' and arsitektur == 'gru'")
                .groupby("epochs").auc_latih.mean())
    E_gru = int((kandidat - acuan).abs().idxmin())
    print(f"\nPEMILIHAN E (hanya dari AUC fold latih, AUC uji tidak dilihat)")
    print(f"  acuan mamba2 E=35 : auc_latih {acuan:.4f}")
    for e, v in kandidat.items():
        tanda = " <- dipilih" if e == E_gru else ""
        print(f"  gru E={e:2d}          : auc_latih {v:.4f}  selisih {v-acuan:+.4f}{tanda}")

    # Tahap 3 — kedua arm pada NewHandPD dengan E yang sudah dibekukan.
    print(f"\ntahap 3: NewHandPD, mamba2 E={E_ACUAN} dan gru E={E_gru}")
    for s in SEEDS:
        tambah("newhandpd", nhp, "mamba2", s, E_ACUAN)
    for s in SEEDS:
        tambah("newhandpd", nhp, "gru", s, E_gru)

    # ------------------------------------------------------------- vonis
    t = pd.DataFrame(art)
    baris = []
    for arm, e in [("mamba2", E_ACUAN), ("gru", E_gru)]:
        for s in SEEDS:
            a = t.query("kohort=='uci395' and arsitektur==@arm and epochs==@e and seed==@s")
            b = t.query("kohort=='newhandpd' and arsitektur==@arm and epochs==@e and seed==@s")
            if len(a) and len(b):
                baris.append({"arsitektur": arm, "epochs": e, "seed": s,
                              "auc_uci": float(a.auc_uji.iloc[0]),
                              "auc_newhandpd": float(b.auc_uji.iloc[0]),
                              "hilang": float(a.auc_uji.iloc[0] - b.auc_uji.iloc[0])})
    v = pd.DataFrame(baris)
    v.to_csv(HASIL / "p3b_garis_dasar_disamakan.csv", index=False)
    print("\np3b_garis_dasar_disamakan.csv")
    print(v.groupby("arsitektur")[["auc_uci", "auc_newhandpd", "hilang"]]
          .agg(["mean", "std"]).round(4).to_string())

    a = v.query("arsitektur=='mamba2'").hilang.values
    b = v.query("arsitektur=='gru'").hilang.values
    tt, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(va/len(a) + vb/len(b))
    df = se**4 / (va**2/(len(a)**2*(len(a)-1)) + vb**2/(len(b)**2*(len(b)-1)))
    print(f"\nWelch atas kehilangan, garis dasar disamakan:")
    print(f"  mamba2 {a.mean():+.4f}  gru {b.mean():+.4f}  selisih {a.mean()-b.mean():+.4f}")
    print(f"  t {tt:+.3f}  df {df:.2f}  p {p:.4f}  {'nyata' if p < 0.05 else 'TIDAK nyata'}")
    dasar_m = float(v[v.arsitektur == "mamba2"].auc_uci.mean())
    dasar_g = float(v[v.arsitektur == "gru"].auc_uci.mean())
    print(f"  garis dasar UCI: mamba2 {dasar_m:.4f}  gru {dasar_g:.4f}"
          f"  (selisih {dasar_m - dasar_g:+.4f})")
    pd.DataFrame([{"E_mamba2": E_ACUAN, "E_gru": E_gru, "auc_latih_acuan": acuan,
                   "selisih": float(a.mean()-b.mean()), "t": float(tt), "df": float(df),
                   "p_welch": float(p), "n_seed": len(SEEDS)}]
                 ).to_csv(HASIL / "p3b_vonis.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
