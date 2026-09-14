"""RM2 diulang dengan penyetelan hyperparameter yang adil bagi ketiga arm.

Masalah yang diperbaiki
-----------------------
Seluruh perbandingan arsitektur pada penelitian ini berjalan pada satu
konfigurasi yang dibekukan Skenario S2 — patch 7, 35 epoch, d_model 128,
tiga lapis, dropout 0,3, lr 3e-4 — dan konfigurasi itu **dipilih ketika hanya
BiGRU dan BiMamba-2 yang ada**. BiMamba-3 dijalankan pada konfigurasi yang
dioptimalkan untuk dua arsitektur lain. Naskah sudah menyatakan hal itu sebagai
batasan; skrip ini mengujinya alih-alih membiarkannya sebagai catatan kaki.

Mengapa d_model TIDAK ikut disetel
----------------------------------
Seluruh perbandingan arsitektur pada penelitian ini bersandar pada pencocokan
jumlah parameter dalam rentang lima persen: BiGRU 224.256, BiMamba-2 233.808,
BiMamba-3 244.512. Mengubah d_model merusak pencocokan itu, dan pada Mamba ia
hanya sah pada nilai tertentu — d_dir 48 memberi d_in_proj 326 yang bukan
kelipatan delapan sehingga forward-nya gagal, dan satu-satunya nilai sah
berikutnya melipatgandakan parameter empat kali. Menyetel d_model karena itu
akan mengganti perbandingan arsitektur dengan perbandingan ukuran. Grid dibatasi
pada laju belajar, dropout, dan kedalaman, yang berlaku setara bagi ketiganya.

Apa yang membuat penyetelan ini sah, dan bukan pencarian angka yang enak
------------------------------------------------------------------------
1. **Grid identik bagi ketiga arm.** Tidak ada arm yang memperoleh kandidat
   lebih banyak atau ruang pencarian yang lebih menguntungkan.
2. **Seleksi tidak pernah menyentuh lipatan uji.** Validasi silang bersarang:
   lipatan luar hanya dipakai menilai, pemilihan konfigurasi berlangsung
   sepenuhnya di dalam lipatan latih lewat pemisahan validasi bertingkat subjek.
3. **Grid dan aturan pemilihan ditulis sebelum dijalankan** dan tidak diubah
   sesudah hasil terlihat.
4. **Vonis pra-registrasi tidak tersentuh.** Aturan keputusan Subbab 3.7 tetap
   dihitung dari konfigurasi beku dan dua arm pra-registrasi. Analisis ini
   **eksploratori** dan dilaporkan sebagai tambahan, bukan pengganti.

Yang dilaporkan apa adanya
--------------------------
Bila setelah penyetelan yang adil BiMamba-2 tetap tidak unggul, itu dilaporkan
sebagai jawaban, bukan sebagai alasan mencari ruang pencarian yang lain.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "src")
from artefak import muat_artefak, simpan_atomik  # noqa: E402
from preprocessing import muat_cache, Normalisasi  # noqa: E402
from model import PDClassifier  # noqa: E402
from training import latih, prediksi  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P = 7
EPOCH_SELEKSI = 20        # lebih murah; dipakai hanya untuk MEMBANDINGKAN kandidat
EPOCH_FINAL = 35          # sama dengan konfigurasi beku, agar sepadan dengan S3
SEEDS_FINAL = list(range(16))   # 3 -> 16: kuasa 80% bagi selisih teramati BiMamba-2 vs BiMamba-3
ARSITEKTUR = ["gru", "mamba2", "mamba3"]
HASIL = Path("results")
ARTEFAK = HASIL / "rm2_tuning_artefak.pkl"

# Grid ditulis sebelum dijalankan. Baris pertama adalah konfigurasi beku S2,
# sehingga penyetelan tidak dapat merugikan arm mana pun dibanding keadaan awal.
GRID = [
    dict(nama="beku_s2",        n_layers=3, dropout=0.30, lr=3e-4),
    dict(nama="lr_tinggi",      n_layers=3, dropout=0.30, lr=1e-3),
    dict(nama="lr_rendah",      n_layers=3, dropout=0.30, lr=1e-4),
    dict(nama="dropout_rendah", n_layers=3, dropout=0.15, lr=3e-4),
    dict(nama="dropout_tinggi", n_layers=3, dropout=0.45, lr=3e-4),
    dict(nama="dangkal",        n_layers=2, dropout=0.30, lr=3e-4),
    dict(nama="dalam",          n_layers=4, dropout=0.30, lr=3e-4),
]

rek = muat_cache(Path("data/cache/uci395_fs100.npz"))
y = np.array([r.label for r in rek])
grup = np.array([r.subjek for r in rek])


def satu_run(arsitektur, cfg, i_latih, i_uji, seed, epochs):
    """Latih pada i_latih, kembalikan logit untuk i_uji."""
    norm = Normalisasi().fit([rek[i] for i in i_latih])
    torch.manual_seed(seed)
    m = PDClassifier(encoder=arsitektur, patch_size=P,
                     n_layers=cfg["n_layers"], dropout=cfg["dropout"]).to(DEV)
    latih(m, rek, list(i_latih), norm, epochs=epochs, device=DEV, seed=seed,
          lr=cfg["lr"])
    logit, _ = prediksi(m, rek, list(i_uji), norm, device=DEV)
    del m
    torch.cuda.empty_cache()
    return logit


def auc_subjek(logit, idx):
    df = pd.DataFrame({"s": grup[idx], "p": logit, "y": y[idx]})
    a = df.groupby("s").agg(p=("p", "mean"), y=("y", "first"))
    return float(roc_auc_score(a.y, a.p)) if a.y.nunique() > 1 else np.nan


def pilih_konfigurasi(arsitektur, i_latih, seed=0):
    """Pemilihan konfigurasi SEPENUHNYA di dalam lipatan latih.

    Lipatan latih dibagi lagi bertingkat subjek menjadi latih-dalam dan validasi.
    Lipatan uji luar tidak pernah dilihat pada tahap ini.
    """
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=7)
    i_dalam, i_val = next(iter(inner.split(np.zeros(len(i_latih)),
                                           y[i_latih], grup[i_latih])))
    i_dalam = i_latih[i_dalam]; i_val = i_latih[i_val]
    skor = []
    for cfg in GRID:
        lg = satu_run(arsitektur, cfg, i_dalam, i_val, seed, EPOCH_SELEKSI)
        skor.append(auc_subjek(lg, i_val))
    terbaik = int(np.nanargmax(skor))
    return GRID[terbaik], skor


def main() -> int:
    art = muat_artefak(ARTEFAK) or {}
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    lipatan = list(skf.split(np.zeros(len(rek)), y, grup))

    print("=" * 78)
    print("RM2 — PENYETELAN HYPERPARAMETER YANG ADIL BAGI KETIGA ARM")
    print("=" * 78)
    print(f"\nperangkat {DEV} | {len(rek)} rekaman, {len(set(grup))} subjek")
    print(f"grid {len(GRID)} konfigurasi, identik bagi ketiga arm:")
    for c in GRID:
        print(f"  {c['nama']:16s} lapis={c['n_layers']} "
              f"dropout={c['dropout']:.2f} lr={c['lr']:.0e}")
    print(f"\nvalidasi silang bersarang: seleksi pada pemisahan validasi di DALAM lipatan latih")
    print(f"seleksi {EPOCH_SELEKSI} epoch, penilaian akhir {EPOCH_FINAL} epoch, "
          f"{len(SEEDS_FINAL)} seed")
    t0 = time.time()

    # ---------- tahap 1: seleksi per lipatan luar ----------
    print("\n--- Tahap 1: pemilihan konfigurasi per lipatan luar ---")
    for a_ in ARSITEKTUR:
        for k, (i_latih, _) in enumerate(lipatan):
            kunci = f"sel|{a_}|{k}"
            if kunci in art:
                continue
            cfg, skor = pilih_konfigurasi(a_, i_latih)
            art[kunci] = dict(arsitektur=a_, fold=k, terpilih=cfg["nama"],
                              skor={g["nama"]: s for g, s in zip(GRID, skor)})
            simpan_atomik(ARTEFAK, art)
            print(f"  {a_:7s} fold {k}: {cfg['nama']:16s} "
                  f"(val AUC {max(s for s in skor if np.isfinite(s)):.4f})  "
                  f"[{time.time()-t0:.0f}s]")

    print("\n  konfigurasi terpilih per arm:")
    for a_ in ARSITEKTUR:
        pilih = [art[f"sel|{a_}|{k}"]["terpilih"] for k in range(5)]
        print(f"    {a_:7s} {pilih}")

    # ---------- tahap 2: penilaian akhir ----------
    print("\n--- Tahap 2: penilaian akhir pada lipatan luar ---")
    for a_ in ARSITEKTUR:
        for seed in SEEDS_FINAL:
            kunci = f"eval|{a_}|{seed}"
            if kunci in art:
                continue
            logit = np.zeros(len(rek))
            for k, (i_latih, i_uji) in enumerate(lipatan):
                nama = art[f"sel|{a_}|{k}"]["terpilih"]
                cfg = next(c for c in GRID if c["nama"] == nama)
                logit[i_uji] = satu_run(a_, cfg, i_latih, i_uji, seed, EPOCH_FINAL)
            art[kunci] = dict(arsitektur=a_, seed=seed, logit_oof=logit,
                              auc=auc_subjek(logit, np.arange(len(rek))))
            simpan_atomik(ARTEFAK, art)
            print(f"  {a_:7s} seed {seed}: AUC {art[kunci]['auc']:.4f}  "
                  f"[{time.time()-t0:.0f}s]")

    # ---------- ringkas ----------
    print("\n--- Hasil ---")
    baris = []
    s3 = pd.read_csv(HASIL / "s3_per_seed.csv")
    for a_ in ARSITEKTUR:
        v = np.array([art[f"eval|{a_}|{s}"]["auc"] for s in SEEDS_FINAL])
        b = s3[(s3.arsitektur == a_) & (s3.seed.isin(SEEDS_FINAL))].auc.values
        baris.append(dict(arsitektur=a_, auc_disetel=v.mean(), sb_disetel=v.std(ddof=1),
                          auc_beku=b.mean(), selisih=v.mean() - b.mean(),
                          konfigurasi=json.dumps(
                              [art[f"sel|{a_}|{k}"]["terpilih"] for k in range(5)]),
                          n_seed=len(SEEDS_FINAL)))
        print(f"  {a_:7s} beku {b.mean():.4f} -> disetel {v.mean():.4f}  "
              f"selisih {v.mean()-b.mean():+.4f}")

    d = pd.DataFrame(baris)
    m2 = float(d[d.arsitektur == "mamba2"].auc_disetel.iloc[0])
    gr = float(d[d.arsitektur == "gru"].auc_disetel.iloc[0])
    print(f"\n  BiMamba-2 dikurangi BiGRU setelah penyetelan : {m2-gr:+.4f}")
    print(f"  ambang tak-terbedakan aturan keputusan S3     : 0,0613")
    print(f"  -> {'BiMamba-2 UNGGUL' if m2-gr > 0.0613 else 'tetap TIDAK TERBEDAKAN'}")
    d.to_csv(HASIL / "rm2_tuning.csv", index=False)
    pd.DataFrame([{"arsitektur": art[k]["arsitektur"], "fold": art[k]["fold"],
                   "terpilih": art[k]["terpilih"], **art[k]["skor"]}
                  for k in art if k.startswith("sel|")]).to_csv(
        HASIL / "rm2_tuning_seleksi.csv", index=False)
    print(f"\ndisimpan ke {HASIL}/rm2_tuning*.csv   [{time.time()-t0:.0f}s total]")
    # Uji berpasangan antar arm pada konfigurasi tersetel. Sebelumnya berkas ini
    # ada di results/ tanpa skrip yang menghasilkannya, sehingga angkanya tidak
    # dapat diregenerasi; perhitungannya dikembalikan ke sini.
    per = {}
    for k, v in art.items():
        if k.startswith("eval|"):
            _, arm, seed = k.split("|")
            per.setdefault(arm, {})[int(seed)] = float(v["auc"])
    auc = {a: np.array([s[i] for i in sorted(s)]) for a, s in per.items()}
    baris = []
    for a, b, alfa in [("mamba2", "mamba3", 0.05 / 3), ("mamba2", "gru", 0.05 / 3),
                       ("mamba3", "gru", 0.05 / 3)]:
        if a not in auc or b not in auc:
            continue
        va, vb = auc[a], auc[b]          # nama sengaja bukan x/y: `y` global
        tt, pv = stats.ttest_ind(va, vb, equal_var=False)
        baris.append({"arm_a": a, "arm_b": b, "selisih": float(va.mean() - vb.mean()),
                      "p_welch": float(pv), "lolos_bonferroni": bool(pv < alfa),
                      "n_seed": len(va)})
    pd.DataFrame(baris).to_csv(HASIL / "rm2_tuning_berpasangan.csv", index=False)
    print("rm2_tuning_berpasangan.csv ditulis ulang dari artefak")


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
