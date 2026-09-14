"""P2 — sapuan panjang urutan sampai BiGRU tidak muat anggaran.

Jalankan: python3 scripts/analisis_p2_sapuan_panjang.py     (GPU, ~3,3 jam)

Yang diuji, dan kenapa ia berbeda dari seluruh sumbu lain
---------------------------------------------------------
Subbab 2.2.4 butir 3 menuliskan **paralelisasi saat pelatihan** sebagai keunggulan
Mamba-2, dengan konsekuensi bahwa resolusi temporal lebih halus dapat dipakai
dalam anggaran komputasi yang sama. Butir itu **belum pernah diuji**; butir 1 dan
2 sudah diuji lewat AUC pada Skenario S2 dan tidak didukung.

Sumbu ini istimewa pada penelitian ini sebab ia **tidak dapat membalik antar
kohort**. Biaya per epoch adalah sifat kompleksitas arsitektur, bukan sifat data:
rekurensi bergerbang menuntut langkah berurutan sepanjang urutan, sementara scan
terstruktur Mamba-2 dapat diparalelkan. Seluruh sumbu perbandingan lain pada
penelitian ini membalik sekurangnya sekali; yang ini secara struktural tidak bisa.

Uji dinamai sebelum dijalankan
------------------------------
    Rasio biaya gru terhadap mamba2 naik monoton terhadap jumlah token, dan pada
    P=1 melampaui 2,15 kali yang tercatat pada P=4.

Ramalan yang dapat meleset, ikut ditulis di depan:

    AUC diperkirakan datar pada resolusi halus, mengikuti sapuan S2 yang bergerak
    antara 0,883 dan 0,913. Bila AUC justru runtuh pada P=2 atau P=1, klaim biaya
    kehilangan artinya sebab resolusi itu tidak lagi dapat dipakai, dan itu
    dilaporkan sebagai kegagalan ramalan.

Bingkainya anggaran, BUKAN Nyquist. Naskah sudah menarik argumen Nyquist sebagai
motivasi resolusi halus: yang dilokalisasi bobot atensi adalah selubung analitik,
bukan gelombang pembawanya, dan selubung itu terwakili pada grid 14,29 Hz.

Dua kendala kesahihan
---------------------
1. Protokol pengukuran waktu ditiru persis dari `build_nb_06.py`: pemanasan dua
   epoch pada 16 rekaman, `torch.cuda.synchronize()` mengapit, sepuluh epoch
   diukur lalu dibagi sepuluh, satu fold. Tanpa itu titik baru tidak sebanding
   dengan empat titik yang sudah ada.
2. `batch_size` wajib sama untuk ketiga arm pada tiap resolusi. Bila memori
   memaksa penurunan, ia diturunkan untuk SEMUA arm sekaligus dan angkanya
   dilaporkan. Batch yang berbeda antar arm membuat perbandingan tidak sah.

Catatan yang harus masuk naskah: pada P=1 `PatchEmbedding` menjadi
`Conv1d(kernel=1, stride=1)`, yaitu proyeksi murni tanpa agregasi temporal,
sehingga "patch" merosot menjadi satu sampel 10 ms. Sah sebagai titik ekstrem
kurva biaya, tetapi bukan konfigurasi yang bermakna bagi peta.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR / "src"))

from artefak import muat_artefak, simpan_atomik  # noqa: E402
from model import PDClassifier  # noqa: E402
from preprocessing import Normalisasi, muat_cache  # noqa: E402
from training import latih, prediksi  # noqa: E402

HASIL = AKAR / "results"
EPOCHS, BATCH = 35, 8
ARSITEKTUR = ["mamba2", "gru", "mamba3"]
PATCH_TIMING_BARU = [2, 1]          # titik baru
PATCH_TIMING_LAMA = [4, 7, 14, 56]  # backfill mamba3 saja; mamba2 dan gru sudah ada
PATCH_AUC = [2, 1]
SEEDS_AUC = [0, 1, 2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

rek = muat_cache(AKAR / "data" / "cache" / "uci395_fs100.npz")
y = np.array([r.label for r in rek])
grup = np.array([r.subjek for r in rek])


def ukur_biaya(arsitektur: int, P: int, batch: int) -> dict:
    """Detik per epoch dan puncak memori. Protokol persis build_nb_06."""
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    i_latih, _ = next(skf.split(np.zeros(len(rek)), y, grup))
    norm = Normalisasi().fit([rek[i] for i in i_latih])

    torch.manual_seed(0)
    m = PDClassifier(encoder=arsitektur, patch_size=P).to(DEV)
    latih(m, rek, list(i_latih[:16]), norm, epochs=2, device=DEV,
          batch_size=batch, seed=0)                      # pemanasan
    if DEV == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    latih(m, rek, list(i_latih), norm, epochs=10, device=DEV, batch_size=batch, seed=0)
    if DEV == "cuda":
        torch.cuda.synchronize()
    puncak = torch.cuda.max_memory_allocated() / 2**20 if DEV == "cuda" else np.nan
    return {"patch": P, "arsitektur": arsitektur, "batch": batch,
            "detik_per_epoch": (time.time() - t0) / 10,
            "puncak_mib": float(puncak),
            "n_patch_median": int(np.median([len(r.kanal) // P for r in rek]))}


def ukur_auc(arsitektur: str, P: int, seed: int, batch: int) -> dict:
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    logit = np.zeros(len(rek))
    for i_latih, i_uji in skf.split(np.zeros(len(rek)), y, grup):
        norm = Normalisasi().fit([rek[i] for i in i_latih])
        torch.manual_seed(seed)
        m = PDClassifier(encoder=arsitektur, patch_size=P).to(DEV)
        latih(m, rek, list(i_latih), norm, epochs=EPOCHS, device=DEV,
              batch_size=batch, seed=seed)
        lo, _ = prediksi(m, rek, list(i_uji), norm, device=DEV, batch_size=batch)
        logit[i_uji] = lo
    subj = sorted(set(grup))
    ps = np.array([logit[grup == s].mean() for s in subj])
    ys = np.array([y[grup == s][0] for s in subj])
    return {"patch": P, "arsitektur": arsitektur, "seed": seed, "batch": batch,
            "auc": float(roc_auc_score(ys, ps))}


def cari_batch(P: int) -> int:
    """Batch terbesar dari {8, 4, 2} yang muat untuk KETIGA arm pada resolusi ini.

    Diturunkan bersama-sama, tidak per arm: batch yang berbeda antar arm membuat
    perbandingan waktu tidak sah.
    """
    for batch in (BATCH, 4, 2):
        try:
            for a in ARSITEKTUR:
                torch.manual_seed(0)
                m = PDClassifier(encoder=a, patch_size=P).to(DEV)
                skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                i_latih, _ = next(skf.split(np.zeros(len(rek)), y, grup))
                norm = Normalisasi().fit([rek[i] for i in i_latih])
                latih(m, rek, list(i_latih[:batch * 2]), norm, epochs=1,
                      device=DEV, batch_size=batch, seed=0)
                del m
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            return batch
        except torch.cuda.OutOfMemoryError:
            if DEV == "cuda":
                torch.cuda.empty_cache()
            print(f"    P={P} batch={batch} kehabisan memori, turun", flush=True)
    raise RuntimeError(f"P={P} tidak muat bahkan pada batch 2")


def main() -> int:
    print(f"P2 — sapuan panjang urutan. {DEV}, {EPOCHS} epoch.\n")

    batch_per_patch = {}
    for P in sorted(set(PATCH_TIMING_BARU + PATCH_AUC)):
        batch_per_patch[P] = cari_batch(P)
        print(f"  P={P}: batch dipakai {batch_per_patch[P]} untuk ketiga arm", flush=True)
    for P in PATCH_TIMING_LAMA:
        batch_per_patch.setdefault(P, BATCH)
    print()

    # -------------------------------------------------------------- biaya
    art_path = HASIL / "p2_artefak_biaya.pkl"
    biaya = muat_artefak(art_path) or []
    ada = {(b["arsitektur"], b["patch"]) for b in biaya}
    perlu = ([("mamba3", P) for P in PATCH_TIMING_LAMA]
             + [(a, P) for P in PATCH_TIMING_BARU for a in ARSITEKTUR])
    perlu = [x for x in perlu if x not in ada]
    print(f"biaya: memakai ulang {len(ada)}, menghitung {len(perlu)}")
    for a, P in perlu:
        h = ukur_biaya(a, P, batch_per_patch[P])
        biaya.append(h)
        simpan_atomik(art_path, biaya)
        print(f"  P={P:3d} {a:7s} {h['detik_per_epoch']:6.2f} s/epoch  "
              f"{h['puncak_mib']:7.1f} MiB  {h['n_patch_median']:4d} patch", flush=True)

    # Gabungkan dengan empat titik lama untuk mamba2 dan gru.
    import pickle
    lama = pickle.load(open(HASIL / "s2_biaya_latih.pkl", "rb"))
    for d in lama:
        if (d["arsitektur"], d["patch"]) not in {(b["arsitektur"], b["patch"]) for b in biaya}:
            biaya.append({**d, "batch": BATCH, "puncak_mib": np.nan})
    tb = pd.DataFrame(biaya).sort_values(["patch", "arsitektur"], ascending=[False, True])
    tb.to_csv(HASIL / "p2_biaya_panjang.csv", index=False)
    print("\np2_biaya_panjang.csv")
    piv = tb.pivot(index="n_patch_median", columns="arsitektur", values="detik_per_epoch")
    piv["rasio_gru_mamba2"] = piv["gru"] / piv["mamba2"]
    print(piv.sort_index().to_string(float_format=lambda v: f"{v:.3f}"), "\n")

    # ---------------------------------------------------------------- AUC
    art_auc = HASIL / "p2_artefak_auc.pkl"
    hasil = muat_artefak(art_auc) or []
    ada = {(h["arsitektur"], h["patch"], h["seed"]) for h in hasil}
    perlu = [(a, P, s) for P in PATCH_AUC for a in ARSITEKTUR for s in SEEDS_AUC
             if (a, P, s) not in ada]
    print(f"AUC: memakai ulang {len(ada)}, menghitung {len(perlu)}")
    for a, P, s in perlu:
        t0 = time.time()
        h = ukur_auc(a, P, s, batch_per_patch[P])
        hasil.append(h)
        simpan_atomik(art_auc, hasil)
        print(f"  P={P} {a:7s} seed={s}  {time.time()-t0:6.1f}s  AUC={h['auc']:.4f}",
              flush=True)

    ta = pd.DataFrame(hasil)
    ta.to_csv(HASIL / "p2_auc_panjang.csv", index=False)
    print("\np2_auc_panjang.csv")
    print(ta.groupby(["patch", "arsitektur"]).auc.agg(["mean", "std", "count"])
          .round(4).to_string())
    print("\nAcuan S2 (satu seed): AUC bergerak 0,883-0,913 pada patch 4 sampai 56.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
