"""Baseline fitur kinematik agregat, pembanding wajib bagi klaim deep learning.

Latar
-----
Seluruh perbandingan pada penelitian ini bersifat antar arsitektur sekuens —
BiGRU lawan BiMamba-2 lawan BiMamba-3. Tidak satu pun menjawab pertanyaan yang
paling lazim diajukan pada literatur handwriting Parkinson: **apakah pemodelan
sekuens memang diperlukan, atau statistik agregat sudah cukup?**

Rancangan
---------
Baseline dihitung dari **kanal yang sama persis** dengan yang diberikan kepada
model (`KANAL_MODEL` = dx, dy, kecepatan, percepatan, jerk, tekanan), ditambah
besaran agregat baku pada literatur (NCV, NCA, fraksi melayang, durasi). Karena
informasinya identik dan hanya representasinya yang berbeda, perbandingan ini
mengisolasi satu hal: nilai dari mempertahankan sumbu waktu.

Protokolnya disamakan baris demi baris dengan Skenario S3:
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, agregasi ke
tingkat subjek lewat rata-rata peluang, AUC tingkat subjek, lima seed.

Yang dilaporkan apa adanya
--------------------------
Bila baseline setara atau menang, itu **bukan kegagalan penelitian**. Ia justru
sejalan dengan benang merah: akurasi klasifikasi tidak membedakan apa pun,
sedangkan yang membedakan adalah peta temporal — dan peta temporal hanya dapat
dihasilkan model sekuens, tidak oleh vektor fitur agregat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, "src")
from preprocessing import muat_cache, KANAL_MODEL  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
SEEDS_PRAREG = [0, 1, 2]
HASIL = Path("results")

rek = muat_cache(Path("data/cache/uci395_fs100.npz"))
y = np.array([r.label for r in rek])
grup = np.array([r.subjek for r in rek])


def ncv(v: np.ndarray) -> float:
    """Cacah pergantian arah, dinormalisasi panjang. Baku pada Drotar dkk."""
    d = np.diff(v)
    if len(d) < 2:
        return 0.0
    return float((np.diff(np.sign(d)) != 0).sum() / len(d))


def fitur(r) -> dict[str, float]:
    """Statistik agregat atas kanal yang sama dengan masukan model."""
    K = r.kanal                      # (T, 6) sesuai KANAL_MODEL
    f: dict[str, float] = {}
    for j, nama in enumerate(KANAL_MODEL):
        c = K[:, j]
        c = c[np.isfinite(c)]
        if len(c) == 0:
            c = np.zeros(1)
        a = np.abs(c)
        f[f"{nama}_rata"] = float(c.mean())
        f[f"{nama}_sb"] = float(c.std())
        f[f"{nama}_med"] = float(np.median(c))
        f[f"{nama}_p95"] = float(np.percentile(a, 95))
        f[f"{nama}_maks"] = float(a.max())
        f[f"{nama}_ncv"] = ncv(c)
    f["durasi_detik"] = len(K) / 100.0
    if r.tekanan_mentah is not None:
        t = r.tekanan_mentah
        f["fraksi_sentuh"] = float(np.mean(t > 0))
        b = (t > 0).astype(int)
        f["n_episode"] = float(np.abs(np.diff(b)).sum() / 2 + b[0])
    else:
        f["fraksi_sentuh"] = np.nan
        f["n_episode"] = np.nan
    return f


X = pd.DataFrame([fitur(r) for r in rek])
X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], 0.0)

MODEL = {
    "regresi_logistik": lambda s: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced",
                                             random_state=s)),
    "svm_rbf": lambda s: make_pipeline(
        StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=s)),
    "random_forest": lambda s: RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=s, n_jobs=-1),
}


def jalankan(nama: str, seed: int) -> float:
    """Satu (model, seed): AUC tingkat subjek atas prediksi out-of-fold."""
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    prob = np.zeros(len(rek))
    for i_latih, i_uji in skf.split(np.zeros(len(rek)), y, grup):
        m = MODEL[nama](seed)
        m.fit(X.iloc[i_latih], y[i_latih])
        prob[i_uji] = m.predict_proba(X.iloc[i_uji])[:, 1]
    agg = (pd.DataFrame({"subjek": grup, "prob": prob, "label": y})
             .groupby("subjek").agg(prob=("prob", "mean"), label=("label", "first")))
    return float(roc_auc_score(agg.label, agg.prob))


print("=" * 74)
print("BASELINE FITUR KINEMATIK AGREGAT")
print("=" * 74)
print(f"\n{len(rek)} rekaman, {len(set(grup))} subjek, {X.shape[1]} fitur")
print(f"kanal sumber identik dengan masukan model: {', '.join(KANAL_MODEL)}")
print(f"protokol: StratifiedGroupKFold k=5 random_state=42, AUC tingkat subjek, "
      f"{len(SEEDS)} seed\n")

baris = []
for nama in MODEL:
    a = np.array([jalankan(nama, s) for s in SEEDS])
    a3 = a[:len(SEEDS_PRAREG)]
    baris.append(dict(model=nama, auc_5seed=a.mean(), sb_5seed=a.std(ddof=1),
                      auc_3seed=a3.mean(), n_fitur=X.shape[1],
                      sumber="scripts/analisis_a_baseline_fitur.py"))
    print(f"  {nama:18s} AUC {a.mean():.4f} (sb {a.std(ddof=1):.4f})   "
          f"per seed: {', '.join(f'{v:.4f}' for v in a)}")

b = pd.DataFrame(baris)
s3 = pd.read_csv(HASIL / "s3_per_seed.csv")
print("\n--- Pembanding: model sekuens, protokol identik ---")
seq = []
for a_ in ["gru", "mamba2", "mamba3"]:
    v = s3[s3.arsitektur == a_].sort_values("seed").auc.values
    seq.append(dict(model=a_, auc_5seed=v.mean(), sb_5seed=v.std(ddof=1),
                    auc_3seed=v[:3].mean(), n_fitur=np.nan, sumber="results/s3_per_seed.csv"))
    print(f"  {a_:18s} AUC {v.mean():.4f} (sb {v.std(ddof=1):.4f})")

semua = pd.concat([b, pd.DataFrame(seq)], ignore_index=True)
terbaik_fitur = b.auc_5seed.max()
terbaik_seq = max(s["auc_5seed"] for s in seq)
print(f"\n  fitur agregat terbaik : {terbaik_fitur:.4f}")
print(f"  model sekuens terbaik : {terbaik_seq:.4f}")
print(f"  selisih               : {terbaik_seq - terbaik_fitur:+.4f} AUC")
print(f"  ambang tak-terbedakan dari aturan keputusan S3 (sb antar fold) : 0,0613")
print(f"  -> {'TERBEDAKAN' if abs(terbaik_seq-terbaik_fitur) > 0.0613 else 'TIDAK TERBEDAKAN'}")

semua.to_csv(HASIL / "a_baseline_fitur.csv", index=False)
X.assign(subjek=grup, label=y).to_csv(HASIL / "a_baseline_matriks_fitur.csv", index=False)
print(f"\ndisimpan ke {HASIL}/a_baseline_*.csv")


# ===================================================================== lintas kohort
# Bagian ini yang menentukan. Pada kohort asal, fitur agregat sudah cukup — bahkan
# lebih baik. Pertanyaannya apakah kecukupan itu ikut berpindah.

import warnings  # noqa: E402
warnings.filterwarnings("ignore", category=FutureWarning)
from scipy.stats import ttest_ind  # noqa: E402
from newhandpd import muat_cache_bisp, KANAL_MODEL_BISP  # noqa: E402

rek_n = muat_cache_bisp(Path("data/cache/newhandpd_spiral_fs100.npz"))
y_n = np.array([r.label for r in rek_n])
grup_n = np.array([r.subjek for r in rek_n])


def fitur_bisp(r) -> dict[str, float]:
    """Fitur yang sama, atas kanal BiSP. Kanal 4-6 tetap ditahan dari keduanya."""
    K = r.kanal
    f: dict[str, float] = {}
    for j, nama in enumerate(KANAL_MODEL_BISP):
        c = K[:, j]
        c = c[np.isfinite(c)]
        if len(c) == 0:
            c = np.zeros(1)
        a = np.abs(c)
        f[f"{nama}_rata"] = float(c.mean())
        f[f"{nama}_sb"] = float(c.std())
        f[f"{nama}_med"] = float(np.median(c))
        f[f"{nama}_p95"] = float(np.percentile(a, 95))
        f[f"{nama}_maks"] = float(a.max())
        f[f"{nama}_ncv"] = ncv(c)
    f["durasi_detik"] = len(K) / 100.0
    return f


Xn = pd.DataFrame([fitur_bisp(r) for r in rek_n]).replace([np.inf, -np.inf], 0.0)
Xn = Xn.fillna(Xn.median(numeric_only=True))


def jalankan_n(nama: str, seed: int) -> float:
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    prob = np.zeros(len(rek_n))
    for i_latih, i_uji in skf.split(np.zeros(len(rek_n)), y_n, grup_n):
        m = MODEL[nama](seed)
        m.fit(Xn.iloc[i_latih], y_n[i_latih])
        prob[i_uji] = m.predict_proba(Xn.iloc[i_uji])[:, 1]
    agg = (pd.DataFrame({"subjek": grup_n, "prob": prob, "label": y_n})
             .groupby("subjek").agg(prob=("prob", "mean"), label=("label", "first")))
    return float(roc_auc_score(agg.label, agg.prob))


print("\n" + "=" * 74)
print("LINTAS KOHORT — apakah kecukupan fitur agregat ikut berpindah?")
print("=" * 74)
print(f"\nNewHandPD: {len(rek_n)} rekaman, {len(set(grup_n))} subjek "
      f"(PD {len(set(grup_n[y_n==1]))}, HC {len(set(grup_n[y_n==0]))})")
print("Rancangannya **replikasi**, bukan transfer: metode yang sama dipasang ulang pada tiap")
print("kohort, persis sebagaimana Skenario S5. Perbandingannya karena itu sepadan.\n")

hilang_fitur, lintas = {}, []
for nama in MODEL:
    au = np.array([jalankan(nama, s) for s in SEEDS])
    an = np.array([jalankan_n(nama, s) for s in SEEDS])
    hilang_fitur[nama] = au - an
    lintas.append(dict(model=nama, jenis="fitur agregat", auc_uci=au.mean(),
                       auc_newhandpd=an.mean(), sb_newhandpd=an.std(ddof=1),
                       kehilangan=(au - an).mean()))
    print(f"  {nama:18s} UCI {au.mean():.4f} -> NewHandPD {an.mean():.4f}   "
          f"kehilangan {(au-an).mean():+.4f}")

s5p = pd.read_csv(HASIL / "s5_per_seed.csv")
s3p = pd.read_csv(HASIL / "s3_per_seed.csv")[["arsitektur", "seed", "auc"]].rename(
    columns={"auc": "auc_uci"})
gab = s5p.merge(s3p, on=["arsitektur", "seed"])
gab["hilang"] = gab.auc_uci - gab.auc
print()
hilang_seq = {}
for a_ in ["gru", "mamba2", "mamba3"]:
    v = gab[gab.arsitektur == a_]
    hilang_seq[a_] = v.hilang.values
    lintas.append(dict(model=a_, jenis="model sekuens", auc_uci=v.auc_uci.mean(),
                       auc_newhandpd=v.auc.mean(), sb_newhandpd=v.auc.std(ddof=1),
                       kehilangan=v.hilang.mean()))
    print(f"  {a_:18s} UCI {v.auc_uci.mean():.4f} -> NewHandPD {v.auc.mean():.4f}   "
          f"kehilangan {v.hilang.mean():+.4f}")

print("\n--- Uji beda kehilangan, Welch t dua-sampel atas lima seed ---")
uji = []
for nama, hf in hilang_fitur.items():
    for a_, hs in hilang_seq.items():
        t_, p_ = ttest_ind(hs, hf, equal_var=False)
        uji.append(dict(sekuens=a_, fitur=nama, selisih=hs.mean() - hf.mean(), p_welch=p_))
best = min(hilang_fitur, key=lambda k: hilang_fitur[k].mean())
print(f"  baseline fitur paling tahan: {best} (kehilangan {hilang_fitur[best].mean():+.4f})")
for a_ in ["mamba2", "mamba3", "gru"]:
    t_, p_ = ttest_ind(hilang_seq[a_], hilang_fitur[best], equal_var=False)
    print(f"  {a_:7s} lawan {best:18s} selisih kehilangan "
          f"{hilang_seq[a_].mean()-hilang_fitur[best].mean():+.4f}  p={p_:.4f}")

print("\n--- AUC pada kohort kedua saja, tanpa aritmetika kehilangan ---")
for nama, r in sorted([(l["model"], l) for l in lintas],
                      key=lambda x: -x[1]["auc_newhandpd"]):
    print(f"  {nama:18s} {r['auc_newhandpd']:.4f}   ({r['jenis']})")

pd.DataFrame(lintas).to_csv(HASIL / "a_baseline_lintas_kohort.csv", index=False)
pd.DataFrame(uji).to_csv(HASIL / "a_baseline_uji_lintas.csv", index=False)
print(f"\ndisimpan ke {HASIL}/a_baseline_lintas_kohort.csv dan a_baseline_uji_lintas.csv")
