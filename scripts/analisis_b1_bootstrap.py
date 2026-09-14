"""B1 — bootstrap dan uji permutasi untuk perbandingan alpha vs phi terhadap penanda klinis.

Mereproduksi persis prosedur `rho_subjek_umum` dan `selisih_kelompok` pada
scripts/build_nb_09.py, lalu menambahkan selang kepercayaan yang sebelumnya
tidak ada.
"""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from preprocessing import muat_cache
from marker import penanda_cepat, ke_grid_patch, mask_pena_melayang

FS, P, N_BOOT, N_PERM = 100, 7, 10000, 20000
SEEDS = sorted({k[1] for k in __import__("pickle").load(open("results/s7_artefak.pkl","rb"))})
HASIL = Path("results")

rek = muat_cache(Path("data/cache/uci395_fs100.npz"))
y = np.array([r.label for r in rek])
grup = np.array([r.subjek for r in rek])
tugas = np.array([r.tugas for r in rek])
art = pickle.load(open(HASIL / "s7_artefak.pkl", "rb"))


def korelasi_rekaman(nilai, penanda):
    n = min(len(nilai), len(penanda))
    a, b = np.asarray(nilai[:n], float), np.asarray(penanda[:n], float)
    sah = ~np.isnan(b)
    if sah.sum() < 4 or np.std(a[sah]) == 0 or np.std(b[sah]) == 0:
        return np.nan
    return spearmanr(a[sah], b[sah]).statistic


def ambil_penanda(r):
    m = penanda_cepat(r.xy, FS, pakai_rasio=(r.tugas != 2))
    w = mask_pena_melayang(r.tekanan_mentah) if r.tugas == 2 else None
    return ke_grid_patch(m, P, bobot=w)


PENANDA = {i: ambil_penanda(r) for i, r in enumerate(rek) if r.tugas == 2}


def rho_subjek(arsitektur, besaran):
    """Rata-rata atas rekaman lalu atas seed, terbatas pada STCP. Persis nb 09."""
    kumpul = {}
    for s_ in SEEDS:
        per = {}
        for idx, d in art[(arsitektur, s_)].items():
            if tugas[idx] != 2:
                continue
            rho = korelasi_rekaman(d[besaran], PENANDA[idx])
            if np.isfinite(rho):
                per.setdefault(grup[idx], []).append(rho)
        for k, v in per.items():
            kumpul.setdefault(k, []).append(float(np.mean(v)))
    return {k: float(np.mean(v)) for k, v in kumpul.items()}


def pisah(rho):
    lab = {s: int(y[grup == s][0]) for s in rho}
    pd_ = np.array([v for s, v in rho.items() if lab[s] == 1])
    hc = np.array([v for s, v in rho.items() if lab[s] == 0])
    return pd_, hc


def selisih(pd_, hc):
    return float(np.median(pd_) - np.median(hc))


def boot_ci(pd_, hc, rng, n=N_BOOT):
    d = [selisih(rng.choice(pd_, len(pd_), replace=True),
                 rng.choice(hc, len(hc), replace=True)) for _ in range(n)]
    return np.percentile(d, [2.5, 97.5]), np.array(d)


def perm_p(pd_, hc, rng, n=N_PERM):
    """DUA ARAH, persis `uji_permutasi` pada nb 09 yang sudah dipra-registrasi.

    Memakai satu arah setelah hasil terlihat akan memperkecil p tanpa dasar.
    """
    semua = np.concatenate([pd_, hc]); k = len(pd_)
    obs = selisih(pd_, hc)
    c = sum(abs(selisih(p[:k], p[k:])) >= abs(obs) - 1e-12
            for p in (rng.permutation(semua) for _ in range(n)))
    return (c + 1) / (n + 1), obs


ARM = sorted({k[0] for k in art})
rng = np.random.default_rng(0)
baris, simpan = [], {}
print("Selisih kelompok (median PD - median HC) korelasi peringkat terhadap penanda tremor")
print("Tugas STCP, tingkat subjek, rata-rata 3 seed. Ambang pra-registrasi = 0,0694.\n")
print(f"{'arsitektur':<9} {'peta':<6} {'selisih':>9} {'CI 95% bootstrap':>22} {'p permutasi':>13} {'n PD/HC':>9}")
for a in ARM:
    for b in ["alpha", "phi"]:
        pd_, hc = pisah(rho_subjek(a, b))
        s = selisih(pd_, hc)
        ci, dist = boot_ci(pd_, hc, np.random.default_rng(1))
        p, _ = perm_p(pd_, hc, np.random.default_rng(2))
        simpan[(a, b)] = (pd_, hc, dist)
        baris.append(dict(arsitektur=a, peta=b, selisih=s, ci_bawah=ci[0], ci_atas=ci[1],
                          p_permutasi=p, lolos_ambang=s > 0.0694, n_pd=len(pd_), n_hc=len(hc)))
        print(f"{a:<9} {b:<6} {s:>9.4f} {'[' + f'{ci[0]:+.4f}, {ci[1]:+.4f}' + ']':>22} "
              f"{p:>13.4f} {len(pd_):>5}/{len(hc):<3}")

print("\nKontras alpha - phi (apakah atensi LEBIH selaras klinis daripada Shapley):")
kontras = []
for a in ARM:
    pa, ha, _ = simpan[(a, "alpha")]
    pp, hp, _ = simpan[(a, "phi")]
    obs = selisih(pa, ha) - selisih(pp, hp)
    r2 = np.random.default_rng(3)
    idx_pd = [r2.integers(0, len(pa), len(pa)) for _ in range(N_BOOT)]
    idx_hc = [r2.integers(0, len(ha), len(ha)) for _ in range(N_BOOT)]
    d = [(selisih(pa[i], ha[j]) - selisih(pp[i], hp[j])) for i, j in zip(idx_pd, idx_hc)]
    ci = np.percentile(d, [2.5, 97.5])
    d = np.array(d)
    # dua arah, sepadan dengan uji permutasi pra-registrasi
    p_boot = min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (N_BOOT + 1))
    kontras.append(dict(arsitektur=a, kontras=obs, ci_bawah=ci[0], ci_atas=ci[1], p_bootstrap=p_boot))
    print(f"  {a:<9} kontras {obs:+.4f}  CI 95% [{ci[0]:+.4f}, {ci[1]:+.4f}]  p={p_boot:.4f}")
print("\n  Bootstrap berpasangan pada subjek yang sama, sehingga alpha dan phi")
print("  diresample bersamaan dan korelasi antar keduanya ikut diperhitungkan.")

print("\n" + "=" * 78)
print("UJI BERPASANGAN — besaran yang sebenarnya diklaim")
print("=" * 78)
print("Analisis primer menguji selisih PD-HC, yang terikat pada 7 subjek kontrol saja.")
print("Klaim 'atensi lebih selaras klinis daripada Shapley' menanyakan hal berbeda:")
print("apakah rho alpha > rho phi pada subjek yang SAMA. Itu uji berpasangan atas")
print("seluruh 52 subjek STCP, dan dayanya jauh lebih besar.\n")
from scipy.stats import wilcoxon
berpasangan = []
for a in ARM:
    ra, rp = rho_subjek(a, "alpha"), rho_subjek(a, "phi")
    s_ = sorted(set(ra) & set(rp))
    va = np.array([ra[x] for x in s_]); vp = np.array([rp[x] for x in s_])
    d = va - vp
    _, p_w = wilcoxon(va, vp)
    r4 = np.random.default_rng(4)
    bs = [np.median(r4.choice(d, len(d), replace=True)) for _ in range(N_BOOT)]
    ci = np.percentile(bs, [2.5, 97.5])
    berpasangan.append(dict(arsitektur=a, n_subjek=len(s_),
                            rho_alpha_median=float(np.median(va)),
                            rho_phi_median=float(np.median(vp)),
                            selisih_berpasangan=float(np.median(d)),
                            ci_bawah=ci[0], ci_atas=ci[1], p_wilcoxon=p_w,
                            n_alpha_lebih_besar=int((d > 0).sum())))
    print(f"{a:<8} n={len(s_)} | rho alpha {np.median(va):+.4f} vs phi {np.median(vp):+.4f}")
    print(f"{'':8} selisih {np.median(d):+.4f} CI [{ci[0]:+.4f}, {ci[1]:+.4f}] "
          f"Wilcoxon p={p_w:.2e} | alpha>phi pada {int((d>0).sum())}/{len(d)} subjek")

pd.DataFrame(baris).to_csv(HASIL / "s7_bootstrap_peta.csv", index=False)
pd.DataFrame(kontras).to_csv(HASIL / "s7_kontras_alpha_phi.csv", index=False)
pd.DataFrame(berpasangan).to_csv(HASIL / "s7_berpasangan_alpha_phi.csv", index=False)
print(f"\ndisimpan ke {HASIL}/s7_bootstrap_peta.csv, s7_kontras_alpha_phi.csv, "
      f"s7_berpasangan_alpha_phi.csv")
