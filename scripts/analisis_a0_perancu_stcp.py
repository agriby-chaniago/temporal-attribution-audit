"""A0 dan A1 — audit perancu STCP dan ke mana massa atensi jatuh.

Seluruhnya memakai artefak yang sudah ada (`results/s7_artefak.pkl`,
`results/s3_artefak.pkl`), tanpa pelatihan ulang dan tanpa GPU.

Latar
-----
Analisis primer memakai tugas STCP karena penandanya paling bersih di sana: tidak
ada gerakan volunter yang perlu dipisahkan. Analisis itu gagal, dan penyebabnya
selama ini dinyatakan sebagai tujuh subjek kontrol. Skrip ini memeriksa apakah
penyebabnya lebih dalam.

Yang diperiksa
--------------
A0. Empat perancu tingkat subjek pada STCP: dukungan penanda, durasi, fraksi
    sentuh, dan apakah logit model melacak ketiganya di dalam kelompok penderita
    (di mana labelnya konstan, sehingga tujuh kontrol tidak ikut campur).

    Kontrol dukungan memakai **transplantasi mask sentuh penderita yang nyata**
    ke rekaman kontrol, bukan masking acak. Patch tak-sah pada penderita bukan
    tersebar melainkan berblok kontigu; masking acak menyepadankan jumlah tetapi
    bukan struktur, dan karena alpha maupun penanda berautokorelasi, derajat
    bebas efektifnya berbeda.

A1. Ke mana massa atensi jatuh, diuji terhadap **null geseran siklik**, bukan
    terhadap satu. Null satu hanya sah bila alpha seragam; alpha tidak seragam
    dan patch melayang berblok, sehingga rasio dapat bergeser dari satu tanpa
    hubungan apa pun dengan gejala.

Ramalan, ditulis sebelum dijalankan
-----------------------------------
- Selisih penderita dikurangi kontrol **bergerak berarti** pada kontrol
  tersepadan -> penyebab kegagalan adalah perancu dukungan.
- Selisih **tidak bergerak** -> dukungan bukan penyebabnya.
- Regresi bertanda sedemikian sehingga dukungan rendah memberi rho tinggi ->
  rho penderita tergelembung, perancu mendorong **ke arah hipotesis**, dan
  analisis primer tetap gagal meski dibantu. Kegagalannya menjadi lebih kuat.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from preprocessing import muat_cache  # noqa: E402
from marker import penanda_cepat, ke_grid_patch, mask_pena_melayang  # noqa: E402

FS, P = 100, 7
N_ULANG, N_ROTASI = 200, 1000
HASIL = Path("results")

rek = muat_cache(Path("data/cache/uci395_fs100.npz"))
y = np.array([r.label for r in rek])
grup = np.array([r.subjek for r in rek])
tugas = np.array([r.tugas for r in rek])
STCP = np.flatnonzero(tugas == 2)

art7 = pickle.load(open(HASIL / "s7_artefak.pkl", "rb"))
SEEDS = sorted({k[1] for k in art7})
art3 = pickle.load(open(HASIL / "s3_artefak.pkl", "rb"))
LOGIT = [e for e in art3 if e["arsitektur"] == "mamba2" and e["seed"] == 0][0]["logit_oof"]


# ---------------------------------------------------------------- pembantu
def korelasi(nilai, penanda):
    """Korelasi peringkat satu rekaman, mengabaikan patch tanpa penanda sah."""
    n = min(len(nilai), len(penanda))
    a, b = np.asarray(nilai[:n], float), np.asarray(penanda[:n], float)
    sah = ~np.isnan(b)
    if sah.sum() < 4 or np.std(a[sah]) == 0 or np.std(b[sah]) == 0:
        return np.nan
    return spearmanr(a[sah], b[sah]).statistic


def penanda_stcp(r, mask=None):
    """Penanda cepat pada grid patch. mask=None memakai mask melayang aslinya."""
    m = mask_pena_melayang(r.tekanan_mentah) if mask is None else mask
    return ke_grid_patch(penanda_cepat(r.xy, FS, pakai_rasio=False), P, bobot=m)


def jumlah_blok(m):
    """Jumlah episode melayang kontigu."""
    return int(np.sum(np.diff(np.r_[0, m.astype(int), 0]) == 1))


def selisih_kelompok(rho_subjek):
    lab = {s: int(y[grup == s][0]) for s in rho_subjek}
    pd_ = [v for s, v in rho_subjek.items() if lab[s] == 1]
    hc = [v for s, v in rho_subjek.items() if lab[s] == 0]
    return float(np.median(pd_) - np.median(hc)), pd_, hc


def rho_per_subjek(mask_pengganti=None):
    """Rata-rata atas rekaman lalu atas seed, sepadan `rho_subjek_umum` nb09."""
    kumpul = {}
    for s_ in SEEDS:
        per = {}
        for idx in STCP:
            d = art7.get(("mamba2", s_), {}).get(idx)
            if d is None:
                continue
            m = None if mask_pengganti is None else mask_pengganti.get(idx)
            r = korelasi(d["alpha"], penanda_stcp(rek[idx], m))
            if np.isfinite(r):
                per.setdefault(grup[idx], []).append(r)
        for k, v in per.items():
            kumpul.setdefault(k, []).append(float(np.mean(v)))
    return {k: float(np.mean(v)) for k, v in kumpul.items()}


# ---------------------------------------------------------------- A0 tabel perancu
print("=" * 78)
print("A0 — PERANCU TINGKAT SUBJEK PADA STCP")
print("=" * 78)

baris = []
for idx in STCP:
    m = mask_pena_melayang(rek[idx].tekanan_mentah)
    pen = penanda_stcp(rek[idx])
    baris.append(dict(subjek=grup[idx], label=int(y[idx]), logit=float(LOGIT[idx]),
                      durasi=len(rek[idx].kanal), frac_melayang=float(m.mean()),
                      frac_sentuh=float(1 - m.mean()), blok=jumlah_blok(m),
                      n_patch=len(pen), n_valid=int(np.isfinite(pen).sum())))
subj = pd.DataFrame(baris).groupby(["subjek", "label"]).mean(numeric_only=True).reset_index()

print(f"\n{'besaran':26s} {'PD median':>11s} {'HC median':>11s} {'AUC':>8s} {'p':>9s}")
for kol, nm in [("durasi", "durasi (sampel)"), ("frac_sentuh", "fraksi sentuh"),
                ("n_valid", "patch penanda sah"), ("blok", "episode melayang")]:
    a = subj[subj.label == 1][kol].values
    b = subj[subj.label == 0][kol].values
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    print(f"{nm:26s} {np.median(a):>11.4f} {np.median(b):>11.4f} "
          f"{u/(len(a)*len(b)):>8.4f} {p:>9.4f}")

print(f"\nAUC prediktor tunggal terhadap label ({int(subj.label.sum())} PD / "
      f"{int((1-subj.label).sum())} HC):")
for kol, nm in [("durasi", "durasi saja"), ("frac_sentuh", "fraksi sentuh saja"),
                ("logit", "logit model")]:
    print(f"  {nm:22s} AUC = {roc_auc_score(subj.label, subj[kol]):.4f}")

print("\nDi dalam PD saja (label konstan, kontrol tidak ikut campur):")
pdv = subj[subj.label == 1]
kor_pd = []
for kol, nm in [("durasi", "durasi"), ("frac_sentuh", "fraksi sentuh"),
                ("blok", "episode melayang"), ("n_valid", "patch penanda sah")]:
    r, p = spearmanr(pdv.logit, pdv[kol])
    kor_pd.append(dict(pembanding=nm, rho=r, p=p, n=len(pdv)))
    print(f"  Spearman(logit, {nm:20s}) = {r:+.4f}  p = {p:.4f}")

# struktur blok: terukur vs tersebar vs lantai
f = float(pdv.frac_melayang.median())
sah_ukur = float((pdv.n_valid / pdv.n_patch).median())
print(f"\nStruktur melayang pada PD:")
print(f"  fraksi melayang               {f:.4f}")
print(f"  patch sah bila TERSEBAR acak  {1-(1-f)**P:.4f}")
print(f"  patch sah bila BERBLOK penuh  {f:.4f}")
print(f"  patch sah TERUKUR             {sah_ukur:.4f}  -> "
      f"{'BERBLOK' if abs(sah_ukur-f) < abs(sah_ukur-(1-(1-f)**P)) else 'tersebar'}")

# ---------------------------------------------------------------- A0 kontrol dukungan
print("\n" + "=" * 78)
print("A0 — KONTROL DUKUNGAN: transplantasi mask sentuh PD nyata ke rekaman HC")
print("=" * 78)

idx_pd = [i for i in STCP if y[i] == 1]
idx_hc = [i for i in STCP if y[i] == 0]
mask_pd = [mask_pena_melayang(rek[i].tekanan_mentah) for i in idx_pd]

asli, _, _ = selisih_kelompok(rho_per_subjek())
print(f"\nselisih PD - HC asli: {asli:+.4f}")

rng = np.random.default_rng(0)
sebaran, cek_frac, cek_blok = [], [], []
for _ in range(N_ULANG):
    peta = {}
    for i in idx_hc:
        m_sumber = mask_pd[rng.integers(len(mask_pd))]
        n = len(rek[i].tekanan_mentah)
        # regangkan/potong mask sumber ke panjang rekaman HC
        pos = np.linspace(0, len(m_sumber) - 1, n)
        peta[i] = m_sumber[np.round(pos).astype(int)]
        cek_frac.append(float(peta[i].mean())); cek_blok.append(jumlah_blok(peta[i]))
    s, _, _ = selisih_kelompok(rho_per_subjek(peta))
    sebaran.append(s)
sebaran = np.array(sebaran)

print(f"selisih setelah transplantasi: median {np.median(sebaran):+.4f}  "
      f"selang 95% [{np.percentile(sebaran,2.5):+.4f}, {np.percentile(sebaran,97.5):+.4f}]")
print(f"pergeseran dari asli         : {np.median(sebaran)-asli:+.4f}")
print(f"\nverifikasi transplantasi (harus cocok dengan PD, bukan hanya fraksinya):")
print(f"  fraksi melayang  HC tertransplantasi {np.median(cek_frac):.4f} | PD asli {f:.4f}")
print(f"  jumlah blok      HC tertransplantasi {np.median(cek_blok):.1f} | PD asli "
      f"{pdv.blok.median():.1f}")

# ---------------------------------------------------------------- A1 massa atensi
print("\n" + "=" * 78)
print("A1 — MASSA ATENSI PADA PATCH MELAYANG, null geseran siklik")
print("=" * 78)

rng = np.random.default_rng(1)
hasil_a1 = []
for idx in idx_pd:
    m = mask_pena_melayang(rek[idx].tekanan_mentah).astype(float)
    frac = ke_grid_patch(m, P)
    a_kum = []
    for s_ in SEEDS:
        d = art7.get(("mamba2", s_), {}).get(idx)
        if d is not None:
            a_kum.append(np.asarray(d["alpha"], float))
    if not a_kum:
        continue
    a = np.mean([x[:min(map(len, a_kum))] for x in a_kum], axis=0)
    n = min(len(a), len(frac)); a, fr = a[:n], frac[:n]
    if a.sum() <= 0 or fr.mean() <= 0:
        continue
    obs = float((a * fr).sum() / a.sum() / fr.mean())
    nul = [float((a * np.roll(fr, rng.integers(n))).sum() / a.sum() / fr.mean())
           for _ in range(N_ROTASI)]
    nul = np.array(nul)
    hasil_a1.append(dict(subjek=grup[idx], rasio=obs, nul_median=float(np.median(nul)),
                         z=float((obs - nul.mean()) / max(nul.std(), 1e-9)),
                         p=float((np.abs(nul - nul.mean()) >= abs(obs - nul.mean())).mean())))
a1 = pd.DataFrame(hasil_a1).groupby("subjek").mean(numeric_only=True).reset_index()

print(f"\n{len(a1)} subjek PD, {N_ROTASI} rotasi per rekaman")
print(f"  rasio teramati            median {a1.rasio.median():.4f}")
print(f"  rasio null geseran siklik median {a1.nul_median.median():.4f}")
print(f"  z terhadap null           median {a1.z.median():+.4f}")
print(f"  subjek dengan p < 0,05    {int((a1.p < 0.05).sum())} dari {len(a1)}")
u, p_a1 = mannwhitneyu(a1.rasio, a1.nul_median, alternative="two-sided")
print(f"\n  uji berpasangan rasio terhadap null-nya sendiri: p = {p_a1:.4f}")
print("  Dibandingkan terhadap 1,0 (null yang SALAH), rasio 0,9363 tampak menyimpang;")
print("  terhadap null geseran siklik, penyimpangannya dinilai dengan benar.")

# ---------------------------------------------------------------- simpan
subj.to_csv(HASIL / "a0_perancu_subjek.csv", index=False)
pd.DataFrame(kor_pd).to_csv(HASIL / "a0_korelasi_dalam_pd.csv", index=False)
pd.DataFrame([{
    "selisih_asli": asli, "selisih_transplantasi_median": float(np.median(sebaran)),
    "ci_bawah": float(np.percentile(sebaran, 2.5)), "ci_atas": float(np.percentile(sebaran, 97.5)),
    "pergeseran": float(np.median(sebaran) - asli), "n_ulang": N_ULANG,
    "auc_durasi": float(roc_auc_score(subj.label, subj.durasi)),
    "auc_frac_sentuh": float(roc_auc_score(subj.label, subj.frac_sentuh)),
    "auc_logit": float(roc_auc_score(subj.label, subj.logit)),
    "patch_sah_terukur": sah_ukur, "patch_sah_bila_tersebar": float(1 - (1 - f) ** P),
    "sumber": "scripts/analisis_a0_perancu_stcp.py",
}]).to_csv(HASIL / "a0_ringkas.csv", index=False)
a1.to_csv(HASIL / "a1_massa_atensi.csv", index=False)
print(f"\ndisimpan ke {HASIL}/a0_*.csv dan a1_massa_atensi.csv")
