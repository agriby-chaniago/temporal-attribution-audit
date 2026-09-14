"""A3 — keselarasan peta terhadap penanda LAMBAT pada tugas DST.

Latar
-----
Analisis primer memakai penanda cepat pada STCP, dan gagal. Penanda lambat sudah
tervalidasi empat uji (nol pada spiral ideal, naik pada simpangan lambat, R^2
median 0,972 pada 150 spiral nyata, memisahkan kelompok pada DST dengan AUC
0,942) tetapi belum pernah dipakai satu analisis pun.

DST memiliki lima belas subjek kontrol, dua kali lipat STCP, dan penandanya
memisahkan kelompok jauh lebih kuat (AUC 0,942 berbanding 0,809). Ini analisis
paling berdaya yang tersedia tanpa mengumpulkan data baru.

Dua batasan yang ditetapkan sebelum dijalankan
----------------------------------------------
1. **Ambang 0,0694 DIPINJAM, tidak dikalibrasi ulang.** Ambang itu dikalibrasi
   untuk penanda cepat pada STCP lewat injeksi tremor. Mengkalibrasi ulang dengan
   cara yang sama tidak koheren di sini: penanda lambat justru dirancang menolak
   tremor, dengan redaman sembilan belas kali pada lima hertz, sehingga lantai
   dan atapnya akan nyaris berimpit. Kalibrasi yang benar menuntut perturbasi
   lambat berlokasi diketahui, yang fungsinya belum ada.

   Konsekuensinya, A3 **diturunkan dari uji ambang menjadi estimasi besaran efek
   beserta selang kepercayaan**. Ambang tetap ditampilkan sebagai acuan pinjaman,
   dan diberi label demikian.

2. **A3 tetap eksploratori apa pun hasilnya.** Menaikkannya menjadi konfirmatori
   setelah berhasil persis melanggar prinsip tanpa fallback yang dipegang seluruh
   protokol.

Besaran efek minimum terdeteksi (MDE) dihitung lebih dahulu, sebab klaim "bila
gagal di sini maka penyebabnya bukan daya uji" hanya sahih bila MDE-nya berada di
bawah efek yang masuk akal secara fisiologis.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, "src")
from preprocessing import muat_cache  # noqa: E402
from marker import penanda_lambat, ke_grid_patch  # noqa: E402

FS, P, N_PERM, N_BOOT = 100, 7, 5000, 10000
AMBANG_PINJAMAN = 0.0694
HASIL = Path("results")

rek = muat_cache(Path("data/cache/uci395_fs100.npz"))
y = np.array([r.label for r in rek])
grup = np.array([r.subjek for r in rek])
tugas = np.array([r.tugas for r in rek])
DST = np.flatnonzero(tugas == 1)

art = pickle.load(open(HASIL / "s7_artefak.pkl", "rb"))
SEEDS = sorted({k[1] for k in art})
ARSITEKTUR = sorted({k[0] for k in art})

PENANDA = {int(i): ke_grid_patch(penanda_lambat(rek[i].xy, FS, tugas=1), P) for i in DST}


def korelasi(nilai, penanda):
    n = min(len(nilai), len(penanda))
    a, b = np.asarray(nilai[:n], float), np.asarray(penanda[:n], float)
    sah = ~np.isnan(b)
    if sah.sum() < 4 or np.std(a[sah]) == 0 or np.std(b[sah]) == 0:
        return np.nan
    return spearmanr(a[sah], b[sah]).statistic


def rho_subjek(arsitektur, besaran):
    """Rata-rata atas rekaman lalu atas seed, sepadan `rho_subjek_umum` nb09."""
    kumpul = {}
    for s_ in SEEDS:
        per = {}
        for idx in DST:
            d = art.get((arsitektur, s_), {}).get(int(idx))
            if d is None:
                continue
            r = korelasi(d[besaran], PENANDA[int(idx)])
            if np.isfinite(r):
                per.setdefault(grup[idx], []).append(r)
        for k, v in per.items():
            kumpul.setdefault(k, []).append(float(np.mean(v)))
    return {k: float(np.mean(v)) for k, v in kumpul.items()}


def pisah(rho):
    lab = {s: int(y[grup == s][0]) for s in rho}
    return (np.array([v for s, v in rho.items() if lab[s] == 1]),
            np.array([v for s, v in rho.items() if lab[s] == 0]))


def selisih(a, b):
    return float(np.median(a) - np.median(b))


def perm_dua_arah(a, b, rng, n=N_PERM):
    """Uji permutasi DUA ARAH, sepadan `uji_permutasi` nb09."""
    semua = np.concatenate([a, b]); k = len(a)
    obs = selisih(a, b)
    c = sum(abs(selisih(p[:k], p[k:])) >= abs(obs) - 1e-12
            for p in (rng.permutation(semua) for _ in range(n)))
    return (c + 1) / (n + 1)


print("=" * 78)
print("A3 — PENANDA LAMBAT PADA DST")
print("=" * 78)
n_pd = len({grup[i] for i in DST if y[i] == 1})
n_hc = len({grup[i] for i in DST if y[i] == 0})
print(f"\n{len(DST)} rekaman, {n_pd} subjek PD / {n_hc} HC  "
      f"(STCP sebagai pembanding: 45 / 7)")

# ---------- MDE sebelum melihat hasil ----------
print("\n--- Besaran efek minimum terdeteksi, dihitung sebelum hasil dilihat ---")
rng = np.random.default_rng(0)
ra = rho_subjek("mamba2", "alpha")
pd_, hc = pisah(ra)
sd_gab = float(np.std(np.concatenate([pd_, hc]), ddof=1))
mde = None
for d in np.arange(0.0, 0.60, 0.005):
    kuasa = np.mean([perm_dua_arah(rng.normal(d, sd_gab, len(pd_)),
                                   rng.normal(0, sd_gab, len(hc)), rng, n=400) < 0.05
                     for _ in range(40)])
    if kuasa >= 0.80:
        mde = float(d); break
print(f"  simpangan baku gabungan rho antar subjek : {sd_gab:.4f}")
print(f"  MDE pada kuasa 80 persen, {n_pd} lawan {n_hc} : "
      f"{mde:.4f}" if mde else "  MDE > 0,60 — daya sangat terbatas")
print(f"  Sebagai pembanding, ambang pinjaman        : {AMBANG_PINJAMAN:.4f}")
if mde and mde > AMBANG_PINJAMAN:
    print(f"  -> MDE DI ATAS ambang. Kalimat 'penyebabnya bukan daya uji' TIDAK sahih")
else:
    print(f"  -> MDE di bawah ambang. Kegagalan di sini bukan sekadar soal daya uji")

# ---------- hasil ----------
print("\n--- Selisih kelompok, penanda lambat, DST ---")
print(f"{'arm':8s} {'peta':6s} {'selisih':>9s} {'p permutasi':>12s} {'lolos pinjaman':>15s}")
baris = []
for a_ in ARSITEKTUR:
    for b_ in ["alpha", "phi"]:
        pd_, hc = pisah(rho_subjek(a_, b_))
        s = selisih(pd_, hc)
        p = perm_dua_arah(pd_, hc, np.random.default_rng(2))
        baris.append(dict(arsitektur=a_, peta=b_, selisih=s, p_permutasi=p,
                          lolos_ambang_pinjaman=bool(s > AMBANG_PINJAMAN),
                          n_pd=len(pd_), n_hc=len(hc)))
        print(f"{a_:8s} {b_:6s} {s:>+9.4f} {p:>12.4f} {str(s > AMBANG_PINJAMAN):>15s}")

print("\n--- Uji berpasangan alpha vs phi atas seluruh subjek DST ---")
pas = []
for a_ in ARSITEKTUR:
    ra, rp = rho_subjek(a_, "alpha"), rho_subjek(a_, "phi")
    s_ = sorted(set(ra) & set(rp))
    va = np.array([ra[x] for x in s_]); vp = np.array([rp[x] for x in s_])
    d = va - vp
    _, pw = wilcoxon(va, vp)
    r4 = np.random.default_rng(4)
    bs = [np.median(r4.choice(d, len(d), replace=True)) for _ in range(N_BOOT)]
    ci = np.percentile(bs, [2.5, 97.5])
    pas.append(dict(arsitektur=a_, n_subjek=len(s_), rho_alpha=float(np.median(va)),
                    rho_phi=float(np.median(vp)), selisih=float(np.median(d)),
                    ci_bawah=ci[0], ci_atas=ci[1], p_wilcoxon=pw,
                    alpha_lebih_besar=int((d > 0).sum())))
    print(f"{a_:8s} n={len(s_)} | rho alpha {np.median(va):+.4f} vs phi {np.median(vp):+.4f} | "
          f"selisih {np.median(d):+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] p={pw:.2e} | "
          f"{int((d>0).sum())}/{len(d)}")

pd.DataFrame(baris).to_csv(HASIL / "a3_selisih_kelompok_dst.csv", index=False)
pd.DataFrame(pas).to_csv(HASIL / "a3_berpasangan_dst.csv", index=False)
pd.DataFrame([{"mde_kuasa80": mde, "sd_gabungan": sd_gab, "n_pd": n_pd, "n_hc": n_hc,
               "ambang_pinjaman": AMBANG_PINJAMAN, "status": "eksploratori",
               "sumber": "scripts/analisis_a3_penanda_lambat.py"}]).to_csv(
    HASIL / "a3_mde.csv", index=False)
print(f"\ndisimpan ke {HASIL}/a3_*.csv")
