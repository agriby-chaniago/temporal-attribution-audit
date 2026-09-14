"""P1B — uji permutasi bobot dan ablasi mean pooling pada sepuluh seed.

Jalankan: python3 scripts/analisis_p1_s8_sepuluh_seed.py     (GPU, ~2,3 jam)

Kenapa perlu
------------
Salah satu dari tiga alasan yang membenarkan BiMamba-2 sebagai objek audit adalah
**kekokohan**: petanya menanggung beban prediksinya sendiri jauh lebih besar
daripada BiGRU. Angkanya kuat — mengacak bobot atensi menurunkan AUC BiMamba-2
sebesar 0,0699 berbanding 0,0151 pada BiGRU, 4,6 kali lipat — tetapi
`results/s8_putusan.csv` mencatat `seed = 0`.

Satu seed. Pada penelitian ini itu berstatus **petunjuk, bukan temuan**, dan
sudah tiga kali angka satu-seed memaksa penarikan klaim. Berkas ini menaikkannya
ke sepuluh seed, jumlah yang sama dengan analisis pembanding.

Uji dinamai sebelum dijalankan
------------------------------
    Welch t dua-sampel atas penurunan-akibat-pengacakan per seed, mamba2 lawan
    gru, equal_var=False, alfa 0,05. Arah diramalkan tetap: BiMamba-2 turun lebih
    banyak. Yang diuji adalah apakah selisihnya bertahan ketika seed bukan lagi
    satu.

Keamanan artefak
----------------
`s8_artefak.pkl` menyimpan `{"label_acak": [...], "permutasi": [...]}` sebagai
daftar TANPA dimensi seed, dan `arm_hilang` bekerja di tingkat arm saja. Struktur
itu tidak diubah. Berkas ini menulis ke artefaknya sendiri, `p1b_s8_artefak.pkl`,
berkunci (arm, seed), sehingga seluruh `s8_*.csv` tetap identik bit-per-bit dan
manifes SHA-256 membuktikannya.

Kontrol label acak sengaja di luar lingkup: ia memeriksa pipeline, bukan
arsitektur, dan pada nb 10 pun dikunci ke satu arm dan dihitung sekali.
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
from preprocessing import Normalisasi, muat_cache  # noqa: E402
from training import latih, prediksi, susun_batch  # noqa: E402

HASIL = AKAR / "results"
FS, P, EPOCHS = 100, 7, 35
SEEDS = list(range(10))
ARSITEKTUR = ["mamba2", "gru", "mamba3"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

rek = muat_cache(AKAR / "data" / "cache" / "uci395_fs100.npz")
y = np.array([r.label for r in rek])
grup = np.array([r.subjek for r in rek])


def ke_subjek(logit, indeks, label):
    """Agregasi ke tingkat subjek. Persis nb 10."""
    prob = 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({"s": grup[indeks], "p": prob, "l": label[indeks]})
    agg = df.groupby("s").agg(p=("p", "mean"), l=("l", "first"))
    return agg.p.values, agg.l.values


@torch.no_grad()
def prediksi_dua_cara(m, indeks, norm, seed):
    """Prediksi normal dan prediksi dengan bobot atensi diacak, model sama.

    `torch.no_grad()` wajib: pemanggilan langsung `m(...)` di luar `prediksi()`
    mengembalikan tensor yang masih menyimpan graf autograd, sehingga `.numpy()`
    menolak. Pada nb 10 konteksnya disediakan sel notebook; di sini harus eksplisit.
    """
    m.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    lo_asli, lo_acak, geser = [], [], []
    for b in range(0, len(indeks), 8):
        pilih = list(indeks[b:b + 8])
        batch = susun_batch(rek, pilih, norm, DEV)
        logit, alpha = m(batch.x, batch.lengths)
        n_patch = (batch.lengths // P).tolist()
        acak = alpha.clone()
        for j, n in enumerate(n_patch):
            if n > 1:
                urut = torch.randperm(n, generator=g).to(alpha.device)
                acak[j, :n] = alpha[j, :n][urut]
        logit_acak, _ = m(batch.x, batch.lengths, alpha_pengganti=acak)
        lo_asli.append(logit.cpu().numpy()); lo_acak.append(logit_acak.cpu().numpy())
        geser.extend((logit_acak - logit).abs().cpu().numpy().tolist())
    return np.concatenate(lo_asli), np.concatenate(lo_acak), np.array(geser)


def jalankan(arsitektur, seed) -> dict:
    """Dua model per fold: attention pooling dan mean pooling. Persis nb 10."""
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    lo_asli = np.zeros(len(rek)); lo_acak = np.zeros(len(rek))
    lo_mean = np.zeros(len(rek)); geser = []
    for i_latih, i_uji in skf.split(np.zeros(len(rek)), y, grup):
        norm = Normalisasi().fit([rek[i] for i in i_latih])

        torch.manual_seed(seed)
        m = PDClassifier(encoder=arsitektur, patch_size=P).to(DEV)
        latih(m, rek, list(i_latih), norm, epochs=EPOCHS, device=DEV, seed=seed)
        a, b, g = prediksi_dua_cara(m, i_uji, norm, seed)
        lo_asli[i_uji] = a; lo_acak[i_uji] = b; geser.extend(g.tolist())

        torch.manual_seed(seed)
        mm = PDClassifier(encoder=arsitektur, patch_size=P, pooling="mean").to(DEV)
        latih(mm, rek, list(i_latih), norm, epochs=EPOCHS, device=DEV, seed=seed)
        lo, _ = prediksi(mm, rek, list(i_uji), norm, device=DEV)
        lo_mean[i_uji] = lo

    auc = lambda lo: float(roc_auc_score(*reversed(ke_subjek(lo, np.arange(len(rek)), y))))
    a_asli, a_acak, a_mean = auc(lo_asli), auc(lo_acak), auc(lo_mean)
    return {"arsitektur": arsitektur, "seed": seed,
            "auc_asli": a_asli, "auc_bobot_diacak": a_acak, "auc_mean_pooling": a_mean,
            "turun_saat_diacak": a_asli - a_acak,
            "atensi_minus_mean": a_asli - a_mean,
            "geser_logit_median": float(np.median(geser))}


def main() -> int:
    print(f"P1B — S8 pada {len(SEEDS)} seed. {DEV}, patch {P}, {EPOCHS} epoch.")
    print(f"{len(ARSITEKTUR)} arm x {len(SEEDS)} seed x 5 fold x 2 model = "
          f"{len(ARSITEKTUR)*len(SEEDS)*10} pelatihan\n")

    art_path = HASIL / "p1b_s8_artefak.pkl"
    art = muat_artefak(art_path) or []
    ada = {(a["arsitektur"], a["seed"]) for a in art}
    perlu = [(a, s) for a in ARSITEKTUR for s in SEEDS if (a, s) not in ada]
    print(f"memakai ulang {len(ada)} kombinasi, menghitung {len(perlu)}\n")

    for arsitektur, seed in perlu:
        t0 = time.time()
        h = jalankan(arsitektur, seed)
        art.append(h)
        simpan_atomik(art_path, art)
        print(f"  {arsitektur:7s} seed={seed}  {time.time()-t0:6.1f}s  "
              f"turun={h['turun_saat_diacak']:+.4f}  "
              f"atensi−mean={h['atensi_minus_mean']:+.4f}", flush=True)

    t = pd.DataFrame(art).sort_values(["arsitektur", "seed"])
    t.to_csv(HASIL / "p1b_s8_sepuluh_seed.csv", index=False)
    print("\np1b_s8_sepuluh_seed.csv")
    print(t.groupby("arsitektur")[["auc_asli", "turun_saat_diacak", "atensi_minus_mean"]]
          .agg(["mean", "std"]).round(4).to_string(), "\n")

    baris = []
    for a, b, alfa in [("mamba2", "gru", 0.05), ("mamba3", "gru", 0.025),
                       ("mamba2", "mamba3", 0.025)]:
        x = t[t.arsitektur == a].turun_saat_diacak.values
        z = t[t.arsitektur == b].turun_saat_diacak.values
        tt, p = stats.ttest_ind(x, z, equal_var=False)
        va, vb = np.var(x, ddof=1), np.var(z, ddof=1)
        se = np.sqrt(va / len(x) + vb / len(z))
        df = se**4 / (va**2/(len(x)**2*(len(x)-1)) + vb**2/(len(z)**2*(len(z)-1)))
        baris.append({"arm_a": a, "arm_b": b, "turun_a": float(x.mean()),
                      "turun_b": float(z.mean()), "selisih": float(x.mean()-z.mean()),
                      "t": float(tt), "df": float(df), "p_welch": float(p),
                      "alfa": alfa, "nyata": p < alfa, "n_seed": len(SEEDS)})
    u = pd.DataFrame(baris)
    u.to_csv(HASIL / "p1b_uji_kokoh.csv", index=False)
    print("p1b_uji_kokoh.csv — Welch atas penurunan-akibat-pengacakan")
    print(u.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nAcuan satu seed (s8_permutasi_ablasi.csv): "
          f"mamba2 0,0699  gru 0,0151  rasio 4,6x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
