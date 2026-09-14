"""RM5 diulang pada kohort dengan kontrol lima kali lebih banyak.

Latar
-----
Analisis primer Rumusan Masalah 5 menguji apakah keselarasan peta terhadap
penanda motorik **lebih besar pada penderita daripada kontrol**. Ia gagal pada
STCP dengan selisih +0,0112 terhadap ambang 0,0694 dan p = 0,585. Audit perancu
memperlihatkan sebab kegagalannya tidak teridentifikasi, dan hitungan daya pada
tugas dengan kontrol terbanyak memberi MDE 0,1450 — dua kali ambangnya sendiri.

Batas itu berasal dari **tujuh subjek kontrol**. NewHandPD memiliki **tiga puluh
lima**, yaitu lima kali lipat, dan artefak Skenario S5 sudah memuat alpha, phi,
serta penanda per rekaman untuk seluruh 66 subjek. Pertanyaan RM5 karena itu
dapat diajukan ulang pada kohort itu **tanpa pelatihan baru sama sekali**.

Batasan yang ditetapkan sebelum dijalankan
-------------------------------------------
1. **Ambang 0,0694 DIPINJAM, tidak dikalibrasi ulang.** Ia dikalibrasi lewat
   injeksi tremor pada UCI 395 dengan penanda dari koordinat pena. Penanda
   NewHandPD diturunkan dari akselerometer badan pena BiSP, besaran fisis yang
   berbeda, sehingga ambangnya tidak otomatis berpindah. Ia ditampilkan sebagai
   acuan pinjaman dan diberi label demikian.

2. **Status eksploratori.** Analisis primer pra-registrasi tetap yang gagal di
   STCP. Analisis ini tidak menggantikannya dan tidak dapat menyelamatkannya;
   menaikkannya menjadi konfirmatori setelah berhasil persis melanggar prinsip
   tanpa fallback yang dipegang seluruh protokol.

3. **MDE dihitung lebih dahulu**, sebelum hasil dilihat, sebab seluruh gunanya
   analisis ini terletak pada dayanya.

Prosedur disamakan baris demi baris dengan analisis primer S7: rho per rekaman,
dirata-ratakan ke subjek lalu ke seed, selisih median antar kelompok, uji
permutasi dua arah 5000 iterasi.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

N_PERM, N_BOOT = 5000, 10000
AMBANG_PINJAMAN = 0.0694
HASIL = Path("results")

art = pickle.load(open(HASIL / "s5_artefak.pkl", "rb"))
SEEDS = sorted({a["seed"] for a in art})
ARSITEKTUR = sorted({a["arsitektur"] for a in art})


def kor(a: np.ndarray, b: np.ndarray) -> float:
    sah = np.isfinite(a) & np.isfinite(b)
    if sah.sum() < 4 or np.std(a[sah]) == 0 or np.std(b[sah]) == 0:
        return np.nan
    return float(spearmanr(a[sah], b[sah]).statistic)


def rho_subjek(arsitektur: str, besaran: str) -> tuple[dict, dict]:
    """Rata-rata atas rekaman lalu atas seed. Mengembalikan (rho, label)."""
    kumpul, lab = {}, {}
    for a in art:
        if a["arsitektur"] != arsitektur:
            continue
        per = {}
        for p in a["peta"]:
            r = kor(p[besaran], p["penanda"])
            if np.isfinite(r):
                per.setdefault(str(p["subjek"]), []).append(r)
                lab[str(p["subjek"])] = int(p["label"])
        for s, v in per.items():
            kumpul.setdefault(s, []).append(float(np.mean(v)))
    return {s: float(np.mean(v)) for s, v in kumpul.items()}, lab


def pisah(rho: dict, lab: dict) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([v for s, v in rho.items() if lab[s] == 1]),
            np.array([v for s, v in rho.items() if lab[s] == 0]))


def selisih(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.median(a) - np.median(b))


def perm_dua_arah(a, b, rng, n=N_PERM) -> float:
    semua = np.concatenate([a, b]); k = len(a)
    obs = abs(selisih(a, b))
    c = sum(abs(selisih(p[:k], p[k:])) >= obs - 1e-12
            for p in (rng.permutation(semua) for _ in range(n)))
    return (c + 1) / (n + 1)


print("=" * 78)
print("RM5 DIULANG PADA NEWHANDPD — 35 KONTROL, BUKAN 7")
print("=" * 78)

ra, lab = rho_subjek("mamba2", "alpha")
pd_, hc = pisah(ra, lab)
print(f"\n{len(pd_)} subjek penderita / {len(hc)} kontrol   "
      f"(STCP sebagai pembanding: 45 / 7)")
print(f"{len(SEEDS)} seed, {len(ARSITEKTUR)} arsitektur, "
      f"{len(art[0]['peta'])} rekaman per (arsitektur, seed)")

# ---------------- MDE sebelum hasil dilihat ----------------
print("\n--- Besaran efek minimum terdeteksi, dihitung sebelum hasil dilihat ---")
rng = np.random.default_rng(0)
sd_gab = float(np.std(np.concatenate([pd_, hc]), ddof=1))
mde = None
for d in np.arange(0.0, 0.60, 0.005):
    kuasa = np.mean([perm_dua_arah(rng.normal(d, sd_gab, len(pd_)),
                                   rng.normal(0, sd_gab, len(hc)), rng, n=400) < 0.05
                     for _ in range(40)])
    if kuasa >= 0.80:
        mde = float(d); break
print(f"  simpangan baku gabungan rho antar subjek : {sd_gab:.4f}")
print(f"  MDE pada kuasa 80 persen, {len(pd_)} lawan {len(hc)} : "
      + (f"{mde:.4f}" if mde else "> 0,60"))
print(f"  Ambang pinjaman sebagai pembanding       : {AMBANG_PINJAMAN:.4f}")
print(f"  MDE pada STCP (7 kontrol) sebagai pembanding : 0,1450 (tugas DST, 15 kontrol)")
if mde and mde <= AMBANG_PINJAMAN:
    print("  -> MDE DI BAWAH ambang. Rancangan ini berdaya menguji hipotesisnya sendiri")
else:
    print("  -> MDE masih di atas ambang. Dayanya membaik namun belum memadai")

# ---------------- hasil ----------------
print("\n--- Selisih keselarasan antar kelompok ---")
print(f"{'arm':8s} {'peta':6s} {'PD':>9s} {'HC':>9s} {'selisih':>9s} {'CI 95%':>22s} "
      f"{'p perm':>8s} {'>ambang':>8s}")
baris = []
for a_ in ARSITEKTUR:
    for b_ in ["alpha", "phi"]:
        rho, lb = rho_subjek(a_, b_)
        p_, h_ = pisah(rho, lb)
        s = selisih(p_, h_)
        pv = perm_dua_arah(p_, h_, np.random.default_rng(2))
        r4 = np.random.default_rng(4)
        bs = [selisih(r4.choice(p_, len(p_), replace=True),
                      r4.choice(h_, len(h_), replace=True)) for _ in range(N_BOOT)]
        ci = np.percentile(bs, [2.5, 97.5])
        baris.append(dict(arsitektur=a_, peta=b_, median_pd=float(np.median(p_)),
                          median_hc=float(np.median(h_)), selisih=s,
                          ci_bawah=ci[0], ci_atas=ci[1], p_permutasi=pv,
                          lolos_ambang_pinjaman=bool(s > AMBANG_PINJAMAN),
                          n_pd=len(p_), n_hc=len(h_)))
        print(f"{a_:8s} {b_:6s} {np.median(p_):>+9.4f} {np.median(h_):>+9.4f} {s:>+9.4f} "
              f"[{ci[0]:>+8.4f},{ci[1]:>+8.4f}] {pv:>8.4f} {str(s > AMBANG_PINJAMAN):>8s}")

print("\n--- Uji berpasangan alpha lawan phi, seluruh subjek ---")
pas = []
for a_ in ARSITEKTUR:
    rA, lA = rho_subjek(a_, "alpha")
    rP, _ = rho_subjek(a_, "phi")
    s_ = sorted(set(rA) & set(rP))
    va = np.array([rA[x] for x in s_]); vp = np.array([rP[x] for x in s_])
    d = va - vp
    _, pw = wilcoxon(va, vp)
    r5 = np.random.default_rng(5)
    bs = [float(np.median(r5.choice(d, len(d), replace=True))) for _ in range(N_BOOT)]
    ci = np.percentile(bs, [2.5, 97.5])
    pas.append(dict(arsitektur=a_, n_subjek=len(s_), rho_alpha=float(np.median(va)),
                    rho_phi=float(np.median(vp)), selisih=float(np.median(d)),
                    ci_bawah=ci[0], ci_atas=ci[1], p_wilcoxon=pw,
                    alpha_lebih_besar=int((d > 0).sum())))
    print(f"{a_:8s} n={len(s_)} | alpha {np.median(va):+.4f} vs phi {np.median(vp):+.4f} | "
          f"selisih {np.median(d):+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] p={pw:.3e} | "
          f"{int((d>0).sum())}/{len(d)}")

pd.DataFrame(baris).to_csv(HASIL / "rm5_selisih_kelompok_newhandpd.csv", index=False)
pd.DataFrame(pas).to_csv(HASIL / "rm5_berpasangan_newhandpd.csv", index=False)
pd.DataFrame([{"mde_kuasa80": mde, "sd_gabungan": sd_gab, "n_pd": len(pd_), "n_hc": len(hc),
               "ambang_pinjaman": AMBANG_PINJAMAN, "status": "eksploratori",
               "sumber": "scripts/analisis_rm5_newhandpd.py"}]).to_csv(
    HASIL / "rm5_mde.csv", index=False)
print(f"\ndisimpan ke {HASIL}/rm5_*.csv")
