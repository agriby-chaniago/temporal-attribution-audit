"""P5 — audit perancu usia dan kovariat demografis pada NewHandPD. Tanpa GPU.

Jalankan: python3 scripts/analisis_p5_perancu_usia.py

Kenapa audit ini wajib, dan kenapa baru sekarang
------------------------------------------------
Header meta tiap berkas sinyal NewHandPD memuat usia, jenis kelamin, tangan
menulis, berat, tinggi, dan status merokok. Seluruhnya selama ini dibuang oleh
`np.loadtxt(comments="#")`, sehingga tidak pernah diperiksa.

Ketika akhirnya dibaca, angkanya besar: kelompok penderita rata-rata **14,7
tahun lebih tua** daripada kelompok kontrol, dan **usia sendirian** memisahkan
kedua kelompok dengan AUC sekitar 0,81. Model mencapai 0,89 sampai 0,93 pada
kohort yang sama. Selisih itu cukup dekat untuk menuntut pemeriksaan: berapa
bagian dari performa lintas kohort yang sebenarnya usia, bukan tanda motorik?

Bentuk auditnya sama dengan Subbab 4.8
--------------------------------------
Audit perancu STCP tidak mencari sebab tunggal melainkan memeriksa apakah sebab
yang berbeda dapat dibedakan pada n yang tersedia. Audit ini mengikuti bentuk
yang sama, dan pertanyaannya dipecah dua:

1. **Apakah perancunya ada di dalam data?** Ya — dan besarannya dilaporkan tanpa
   dikurangi.
2. **Apakah model membacanya?** Diuji lewat korelasi logit terhadap usia **di
   dalam tiap kelompok**. Korelasi antar kelompok tidak menjawab apa pun, sebab
   penderita memang lebih tua dan memang mendapat logit lebih tinggi; yang
   membedakan model-membaca-usia dari model-membaca-penyakit hanyalah apakah
   urutan di dalam satu kelompok ikut mengikuti usia.

Batas yang wajib dinyatakan bersama hasilnya
--------------------------------------------
**UCI 395 tidak memuat usia.** Berkas mentahnya hanya kolom numerik tanpa header
meta, sehingga audit ini hanya mungkin pada kohort lintas. Asimetri itu bukan
pilihan melainkan batas data.

**Null di dalam kelompok bukan bukti ketiadaan.** Dengan 31 dan 35 subjek, hanya
korelasi yang cukup besar yang dapat terdeteksi andal. Korelasi terdeteksi
minimum karena itu ikut dilaporkan, supaya null-nya dibaca sebagai "tidak
terdeteksi pada n ini", bukan "tidak ada".
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

AKAR = Path(__file__).resolve().parent.parent
HASIL = AKAR / "results"
ARSITEKTUR = ["mamba2", "gru", "mamba3"]
ALFA, DAYA = 0.05, 0.80


def rho_terdeteksi_minimum(n: int, alfa: float = ALFA, daya: float = DAYA) -> float:
    """Korelasi Spearman terkecil yang terdeteksi andal pada n ini.

    Lewat transformasi z Fisher: |atanh(r)| * sqrt(n-3) harus melampaui jumlah
    nilai kritis dua arah dan nilai daya. Dilaporkan agar hasil null tidak
    terbaca sebagai bukti ketiadaan.
    """
    if n <= 3:
        return np.nan
    z = stats.norm.ppf(1 - alfa / 2) + stats.norm.ppf(daya)
    return float(np.tanh(z / np.sqrt(n - 3)))


def muat_meta() -> pd.DataFrame:
    m = pd.read_csv(HASIL / "p4_meta_subjek.csv", dtype=str).set_index("subjek")
    for k in ["Age", "Weight", "Height"]:
        if k in m:
            m[k] = pd.to_numeric(m[k], errors="coerce")
    return m


def logit_per_subjek(entri, arm: str) -> dict[str, float]:
    """Logit OOF tingkat subjek, dirata-ratakan atas rekaman lalu atas seed."""
    per: dict[str, list[float]] = {}
    for e in [x for x in entri if x["arsitektur"] == arm]:
        for p in e["peta"]:
            per.setdefault(str(p["subjek"]), []).append(float(e["logit_oof"][p["idx"]]))
    return {k: float(np.mean(v)) for k, v in per.items()}


def main() -> int:
    m = muat_meta()
    entri = pickle.load(open(HASIL / "s5_artefak.pkl", "rb"))
    print("P5 — audit perancu usia pada NewHandPD. Tanpa GPU.\n")

    baris: list[dict] = []

    # ── 1. Seberapa besar perancunya di dalam data ────────────────────────────
    usia = m.dropna(subset=["Age"])
    pd_, hc = usia[usia.kelompok == "PD"].Age, usia[usia.kelompok == "HC"].Age
    t, p = stats.ttest_ind(pd_, hc, equal_var=False)
    y = (usia.kelompok == "PD").astype(int).values
    auc_usia = float(roc_auc_score(y, usia.Age.values))
    print("1. BESARAN PERANCU DI DALAM DATA")
    print(f"   usia PD {pd_.mean():.1f} th (sb {pd_.std():.1f}, n {len(pd_)})  "
          f"vs HC {hc.mean():.1f} th (sb {hc.std():.1f}, n {len(hc)})")
    print(f"   selisih {pd_.mean()-hc.mean():+.1f} th  Welch t {t:.2f}  p {p:.3g}")
    print(f"   AUC usia-saja sebagai prediktor label: {auc_usia:.4f}")
    print(f"   rentang: PD {pd_.min():.0f}-{pd_.max():.0f}  HC {hc.min():.0f}-{hc.max():.0f}\n")
    baris.append({"bagian": "perancu_dalam_data", "besaran": "usia", "arsitektur": "-",
                  "kelompok": "-", "n": len(usia), "nilai": auc_usia,
                  "p": float(p), "keterangan": f"AUC usia-saja; selisih {pd_.mean()-hc.mean():+.1f} th"})

    # Kovariat lain, diperiksa agar usia tidak dilaporkan sendirian.
    print("   kovariat lain:")
    for k in ["Gender", "Writing_Hand", "Smoker"]:
        if k not in m:
            continue
        tab = pd.crosstab(m[k].fillna("?"), m.kelompok)
        if tab.shape[0] > 1:
            chi2, pk, _, _ = stats.chi2_contingency(tab)
            baris.append({"bagian": "kovariat_lain", "besaran": k, "arsitektur": "-",
                          "kelompok": "-", "n": int(tab.values.sum()), "nilai": float(chi2),
                          "p": float(pk), "keterangan": "chi-kuadrat kemandirian thd kelompok"})
            print(f"     {k:14s} chi2 {chi2:6.2f}  p {pk:.3f}"
                  f"  {'TIMPANG' if pk < 0.05 else 'seimbang'}")
    for k in ["Weight", "Height"]:
        if k in m and m[k].notna().sum() > 5:
            a = m[m.kelompok == "PD"][k].dropna(); b = m[m.kelompok == "HC"][k].dropna()
            tk, pk = stats.ttest_ind(a, b, equal_var=False)
            baris.append({"bagian": "kovariat_lain", "besaran": k, "arsitektur": "-",
                          "kelompok": "-", "n": len(a) + len(b), "nilai": float(a.mean() - b.mean()),
                          "p": float(pk), "keterangan": "Welch selisih kelompok"})
            print(f"     {k:14s} selisih {a.mean()-b.mean():+6.1f}  p {pk:.3f}"
                  f"  {'TIMPANG' if pk < 0.05 else 'seimbang'}")

    # ── 2. Apakah model membacanya ────────────────────────────────────────────
    print("\n2. APAKAH MODEL MEMBACA USIA — korelasi logit terhadap usia DI DALAM kelompok")
    print(f"   (korelasi terdeteksi minimum pada alfa {ALFA}, daya {DAYA}: "
          f"n=31 -> |rho| {rho_terdeteksi_minimum(31):.3f}, "
          f"n=35 -> |rho| {rho_terdeteksi_minimum(35):.3f})")
    for arm in ARSITEKTUR:
        lo = logit_per_subjek(entri, arm)
        s = pd.DataFrame({"logit": lo}).join(m[["Age", "kelompok"]]).dropna()
        pesan = []
        for g in ["PD", "HC"]:
            sub = s[s.kelompok == g]
            r, pv = stats.spearmanr(sub.Age, sub.logit)
            rmin = rho_terdeteksi_minimum(len(sub))
            baris.append({"bagian": "logit_vs_usia_dalam_kelompok", "besaran": "rho_spearman",
                          "arsitektur": arm, "kelompok": g, "n": len(sub), "nilai": float(r),
                          "p": float(pv), "keterangan": f"rho terdeteksi minimum {rmin:.3f}"})
            pesan.append(f"{g} n={len(sub)} rho={r:+.3f} p={pv:.3f}")
        r_all, p_all = stats.spearmanr(s.Age, s.logit)
        baris.append({"bagian": "logit_vs_usia_gabungan", "besaran": "rho_spearman",
                      "arsitektur": arm, "kelompok": "gabungan", "n": len(s), "nilai": float(r_all),
                      "p": float(p_all), "keterangan": "bayangan antar kelompok, BUKAN bukti"})
        print(f"   {arm:7s} {' | '.join(pesan)}   gabungan rho {r_all:+.3f}")
    print("   Korelasi gabungan yang positif adalah bayangan antar kelompok dan tidak")
    print("   menjawab apa pun: penderita memang lebih tua DAN memang berlogit lebih tinggi.")

    # ── 3. Apakah keunggulan bertahan pada subjek yang usianya bertumpang tindih ─
    lo_min, hi_min = max(pd_.min(), hc.min()), min(pd_.max(), hc.max())
    cocok = usia[(usia.Age >= lo_min) & (usia.Age <= hi_min)]
    print(f"\n3. AUC PADA SUBJEK YANG USIANYA BERTUMPANG TINDIH [{lo_min:.0f}, {hi_min:.0f}] th")
    print(f"   {len(cocok)} dari {len(usia)} subjek bertahan  "
          f"(PD {(cocok.kelompok=='PD').sum()}, HC {(cocok.kelompok=='HC').sum()})")
    a2 = cocok[cocok.kelompok == "PD"].Age; b2 = cocok[cocok.kelompok == "HC"].Age
    t2, p2 = stats.ttest_ind(a2, b2, equal_var=False)
    print(f"   usia setelah dicocokkan: PD {a2.mean():.1f} vs HC {b2.mean():.1f}  "
          f"selisih {a2.mean()-b2.mean():+.1f} th  p {p2:.3f}"
          f"  {'seimbang' if p2 > 0.05 else 'MASIH TIMPANG'}")
    auc_usia_cocok = roc_auc_score((cocok.kelompok == "PD").astype(int), cocok.Age)
    print(f"   AUC usia-saja setelah dicocokkan: {auc_usia_cocok:.4f} (semula {auc_usia:.4f})")
    baris.append({"bagian": "usia_dicocokkan", "besaran": "auc_usia_saja", "arsitektur": "-",
                  "kelompok": "-", "n": len(cocok), "nilai": float(auc_usia_cocok),
                  "p": float(p2), "keterangan": f"rentang [{lo_min:.0f},{hi_min:.0f}] th"})

    hasil_auc = {}
    for arm in ARSITEKTUR:
        per_seed = []
        for e in [x for x in entri if x["arsitektur"] == arm]:
            lo: dict[str, list[float]] = {}
            for p_ in e["peta"]:
                lo.setdefault(str(p_["subjek"]), []).append(float(e["logit_oof"][p_["idx"]]))
            s = pd.Series({k: np.mean(v) for k, v in lo.items()})
            s = s[s.index.isin(cocok.index)]
            lab = (cocok.loc[s.index].kelompok == "PD").astype(int)
            per_seed.append(roc_auc_score(lab, s.values))
        hasil_auc[arm] = np.array(per_seed)
        print(f"   {arm:7s} AUC {np.mean(per_seed):.4f} (sb {np.std(per_seed, ddof=1):.4f}, "
              f"n {len(per_seed)} seed)")
        baris.append({"bagian": "auc_usia_dicocokkan", "besaran": "auc_model", "arsitektur": arm,
                      "kelompok": "-", "n": len(per_seed), "nilai": float(np.mean(per_seed)),
                      "p": np.nan, "keterangan": f"sb {np.std(per_seed, ddof=1):.4f}"})

    print("\n   Perbandingan arm pada subjek yang usianya dicocokkan:")
    for a, b in [("mamba2", "gru"), ("mamba3", "gru"), ("mamba2", "mamba3")]:
        x, z = hasil_auc[a], hasil_auc[b]
        tt, pv = stats.ttest_ind(x, z, equal_var=False)
        print(f"     {a} - {b}: {x.mean()-z.mean():+.4f}  p {pv:.4f}"
              f"  {'nyata' if pv < 0.05 else 'tidak nyata'}")
        baris.append({"bagian": "banding_arm_usia_dicocokkan", "besaran": f"{a}_minus_{b}",
                      "arsitektur": a, "kelompok": b, "n": len(x),
                      "nilai": float(x.mean() - z.mean()), "p": float(pv),
                      "keterangan": "Welch atas AUC per seed, subjek usia dicocokkan"})

    pd.DataFrame(baris).to_csv(HASIL / "p5_perancu_usia.csv", index=False)
    print("\np5_perancu_usia.csv ditulis.")
    print("\nBATAS: UCI 395 tidak memuat usia, sehingga audit ini hanya mungkin pada")
    print("NewHandPD. Perancu antar kelompok tetap ada dan tidak dihapus oleh audit ini.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
