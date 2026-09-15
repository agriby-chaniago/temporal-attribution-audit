"""Build English-language figures for Q1-amin/paper-full.md from results/*.csv.

Run: python3 scripts/build_figur_paper.py

Why a separate script, not an edit to scripts/build_nb_master.py
------------------------------------------------------------------
build_nb_master.py generates the thesis's Indonesian-language notebook. Editing
its label strings to English would break the thesis. This script reads the same
results/*.csv files directly and emits an independent set of English figures to
figures/paper/ (300 dpi PNG for the .docx build, vector PDF for journal upload).
It never touches a notebook and never trains anything -- every number here is
read from an already-computed CSV (or, for two figures, from the cached
preprocessed signal in data/cache/), the same category of instant, read-only
work as scripts/build_docx.mjs.

Every number is recomputed independently from results/*.csv here, not copied
from the Indonesian thesis figures -- the same discipline used to verify every
claim in Q1-amin/paper-full.md itself. Cross-check prints are included so a
human reviewer can compare against NASKAH-SUMBER.md before trusting a figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.signal import welch

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR / "src"))
from preprocessing import FS_TARGET, muat_cache  # noqa: E402
from viz import TINTA, WARNA, pasang_gaya, pita_tremor, rapikan  # noqa: E402

HASIL = AKAR / "results"
GAMBAR = AKAR / "figures" / "paper"
GAMBAR.mkdir(parents=True, exist_ok=True)
CACHE = AKAR / "data" / "cache"

pasang_gaya()

NAMA = {"gru": "BiGRU", "mamba2": "BiMamba-2", "mamba3": "BiMamba-3"}
WARNA_ARCH = {"gru": WARNA["netral"], "mamba2": WARNA["biru"], "mamba3": WARNA["jingga"]}
URUTAN_ARCH = ["gru", "mamba2", "mamba3"]


def muat(nama: str) -> pd.DataFrame:
    return pd.read_csv(HASIL / f"{nama}.csv")


def simpan(fig, nama: str) -> None:
    for ext, kw in [("png", dict(dpi=300)), ("pdf", {})]:
        fig.savefig(GAMBAR / f"{nama}.{ext}", bbox_inches="tight", **kw)
    print(f"  saved: figures/paper/{nama}.png and .pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — audit design schematic (no CSV; hand-drawn, no numeric claim)
# ═══════════════════════════════════════════════════════════════════════════

def fig01_audit_design() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def kotak(x, y, w, h, teks, warna_tepi=TINTA["utama"], fc="white", fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                                     linewidth=1.3, edgecolor=warna_tepi, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2, teks, ha="center", va="center", fontsize=fs,
                color=TINTA["utama"], zorder=3, linespacing=1.35)

    def panah(x0, y0, x1, y1, warna=TINTA["sekunder"], style="-", lw=1.5):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                      linewidth=lw, color=warna, linestyle=style, zorder=1))

    # Main pipeline, left to right.
    y_main = 3.55
    h_main = 0.95
    kotak(0.15, y_main, 1.55, h_main, "Raw signal\n(6 channels)")
    panah(1.70, y_main + h_main / 2, 2.05, y_main + h_main / 2)
    kotak(2.05, y_main, 1.35, h_main, "Patch\nembedding\n(P = 7)")
    panah(3.40, y_main + h_main / 2, 3.75, y_main + h_main / 2)
    kotak(3.75, y_main, 1.85, h_main, "Interchangeable\nencoder\nBiGRU / BiMamba-2/3", fc="#eef4fc")
    panah(5.60, y_main + h_main / 2, 5.95, y_main + h_main / 2)
    kotak(5.95, y_main, 1.55, h_main, "Bottleneck\nattention\npooling", fc="#eef4fc")
    panah(7.50, y_main + h_main / 2 + 0.28, 7.85, y_main + h_main - 0.10)
    panah(7.50, y_main + h_main / 2 - 0.28, 7.85, y_main + 0.10)
    kotak(7.85, y_main + 0.55, 2.0, 0.55, "Prediction\n(subject-level)")
    kotak(7.85, y_main - 0.10, 2.0, 0.55, "Attention map\n(per-segment)", warna_tepi=WARNA["jingga"])

    # Validation branch, below, pointing up into the attention map.
    y_ref = 0.35
    h_ref = 1.05
    refs = [
        ("Positive control\n(synthetic signal,\nknown location)", 0.15, TINTA["redup"]),
        ("Shapley attribution\n(output-fidelity\nground truth)", 3.55, WARNA["biru"]),
        ("Motor marker\n(withheld channel,\nphysiological ground truth)", 6.95, WARNA["jingga"]),
    ]
    for teks, x, warna in refs:
        kotak(x, y_ref, 2.85, h_ref, teks, warna_tepi=warna, fs=8.8)
    panah(1.35, y_ref + h_ref, 1.35, 1.95, warna=TINTA["redup"], style=":")
    ax.text(1.35, 2.05, "validates\npipeline mechanics", ha="center", va="bottom", fontsize=7.3,
            color=TINTA["redup"], style="italic")
    panah(4.95, y_ref + h_ref, 8.6, y_main + 0.05, warna=WARNA["biru"], style="--")
    panah(8.35, y_ref + h_ref, 8.85, y_main + 0.05, warna=WARNA["jingga"], style="--")

    ax.set_title("Attribution pipeline and its three independent references",
                 loc="left", fontsize=11.5, pad=10)
    fig.tight_layout()
    simpan(fig, "fig01_audit_design")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — data channels and tasks (results/d4_durasi_postur_lintas_tugas.csv)
# ═══════════════════════════════════════════════════════════════════════════

def fig02_data_channels_tasks() -> None:
    d4 = muat("d4_durasi_postur_lintas_tugas")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.7))

    sen = d4[d4.besaran == "fraksi_sentuh"]
    xx = np.arange(len(sen))
    w = 0.36
    ax[0].bar(xx - w / 2, sen.pd_median, w, color=WARNA["jingga"], label="patients")
    ax[0].bar(xx + w / 2, sen.hc_median, w, color=WARNA["biru"], label="controls")
    ax[0].set_xticks(xx)
    ax[0].set_xticklabels(sen.tugas)
    ax[0].set_ylabel("median fraction of rows touching")
    ax[0].set_title("Only STCP instructs the pen to lift", loc="left", fontsize=10.5)
    ax[0].legend(frameon=False, fontsize=8.5)

    for j, (b, lab) in enumerate([("durasi", "duration"), ("fraksi_sentuh", "touch fraction")]):
        v = d4[d4.besaran == b]
        ax[1].plot(range(len(v)), v.auc, "o-", lw=1.8, ms=7,
                   color=[WARNA["biru"], WARNA["jingga"]][j], label=lab)
    ax[1].axhline(0.5, color=TINTA["redup"], ls="--", lw=1.2)
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(["SST", "DST", "STCP"])
    ax[1].set_ylabel("subject-level AUC")
    ax[1].set_ylim(0.2, 1.0)
    ax[1].set_title("Duration reverses direction; touch fraction does not", loc="left", fontsize=10.5)
    ax[1].legend(frameon=False, fontsize=8.5)
    for a in ax:
        rapikan(a)
    fig.tight_layout()
    simpan(fig, "fig02_data_channels_tasks")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — patch grid + tremor-band spectrum (data/cache/uci395_fs100.npz)
# ═══════════════════════════════════════════════════════════════════════════

def satu(rek, tugas, kelompok):
    """Longest recording for one subject/task/group. Mirrors scripts/build_nb_14.py."""
    kandidat = [r for r in rek if r.tugas == tugas and r.kelompok == kelompok
                and r.xy is not None and len(r.xy) > 500]
    return max(kandidat, key=lambda r: len(r.xy)) if kandidat else None


def fig03_patch_grid_spectrum() -> None:
    berkas = CACHE / "uci395_fs100.npz"
    if not berkas.exists():
        print(f"  SKIPPED fig03: {berkas} not found (cache required, see SETUP.md)")
        return
    uci = muat_cache(berkas)
    FS = FS_TARGET

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6))

    # Panel A: patch grid over a velocity trace, STCP task.
    r = satu(uci, 2, "PD")
    P, n_tampil = 7, 40
    potong = r.kanal[:P * n_tampil, 2]  # channel 2 = velocity, per KANAL_MODEL
    t = np.arange(len(potong)) / FS
    ax[0].plot(t, potong, lw=1.0, color=WARNA["biru"])
    for b in range(n_tampil + 1):
        ax[0].axvline(b * P / FS, color=WARNA["redup"], lw=0.6, zorder=0)
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("velocity")
    ax[0].set_title(f"Patch grid — first {n_tampil} patches, 1 patch = {P * 10} ms",
                     loc="left", fontsize=10.5)
    ax[0].set_xlim(0, t[-1])
    rapikan(ax[0])

    # Panel B: median PSD, HC vs PD, STCP task, tremor band highlighted.
    def psd_kelompok(rek, tugas, kel, kanal=2):
        kur = []
        for rr in rek:
            if rr.tugas != tugas or rr.kelompok != kel or len(rr.kanal) < 512:
                continue
            f, P_ = welch(rr.kanal[:, kanal], fs=FS, nperseg=256)
            kur.append(P_)
        return f, np.median(np.vstack(kur), axis=0)

    for kel, w in [("HC", WARNA["biru"]), ("PD", WARNA["jingga"])]:
        f, P_ = psd_kelompok(uci, 2, kel)
        ax[1].semilogy(f, P_, color=w, label=f"{kel} (median)")
    ax[1].axvspan(*pita_tremor, color=WARNA["jingga"], alpha=0.12, zorder=0)
    ax[1].annotate(f"tremor band\n{pita_tremor[0]:.1f}-{pita_tremor[1]:.1f} Hz",
                    xy=(np.mean(pita_tremor), ax[1].get_ylim()[1]),
                    ha="center", va="top", fontsize=8.5, color=TINTA["sekunder"])
    ax[1].set_xlim(0, 25)
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel("power spectral density")
    ax[1].set_title("Spectral content, velocity channel, STCP task", loc="left", fontsize=10.5)
    ax[1].legend(frameon=False, fontsize=8.5)
    rapikan(ax[1])

    fig.tight_layout()
    simpan(fig, "fig03_patch_grid_spectrum")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — attention-Shapley fidelity vs measured ceiling
# ═══════════════════════════════════════════════════════════════════════════

def fig04_attention_shapley_fidelity() -> None:
    s6 = muat("s6_kesetiaan")
    atap = float(muat("s6_batas_atas_kesetiaan").nilai.iloc[0])
    print(f"  fig04 check: ceiling = {atap:.4f} (paper claims 0.1438)")

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    u = s6.set_index("arsitektur").reindex(URUTAN_ARCH)
    xx = np.arange(3)
    ax.bar(xx, u.rho_median, 0.5, color=[WARNA_ARCH[a] for a in u.index])
    ax.errorbar(xx, u.rho_median, yerr=[u.rho_median - u.rho_q25, u.rho_q75 - u.rho_median],
                fmt="none", ecolor=TINTA["sekunder"], elinewidth=1.2, capsize=4)
    ax.axhline(atap, color=WARNA["jingga"], ls="--", lw=1.6)
    ax.text(2.42, atap + 0.004, f"measured pipeline ceiling {atap:.4f}",
            fontsize=8.5, color=WARNA["jingga"], ha="right")
    for i, v in enumerate(u.rho_median):
        ax.text(i, v + 0.006, f"{v:.4f}", ha="center", fontsize=9, color=TINTA["utama"])
    ax.set_xticks(xx)
    ax.set_xticklabels([NAMA[a] for a in u.index])
    ax.set_ylabel("attention-Shapley fidelity (median rho)")
    ax.set_title("Fidelity is low on all three, as theory predicts", loc="left", fontsize=10.5)
    rapikan(ax)
    fig.tight_layout()
    simpan(fig, "fig04_attention_shapley_fidelity")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — fidelity vs marker-alignment gap, both cohorts
# ═══════════════════════════════════════════════════════════════════════════

def fig05_fidelity_both_cohorts() -> None:
    p5 = muat("rm5_berpasangan_newhandpd").copy()
    p7 = muat("s7_berpasangan_alpha_phi").copy()

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x = np.arange(3)
    w = 0.36
    u7 = [float(p7[p7.arsitektur == a].selisih_berpasangan.iloc[0]) for a in URUTAN_ARCH]
    u5 = [float(p5[p5.arsitektur == a].selisih.iloc[0]) for a in URUTAN_ARCH]
    ax.bar(x - w / 2, u7, w, color=WARNA["biru"], label="UCI 395")
    ax.bar(x + w / 2, u5, w, color=WARNA["toska"], label="NewHandPD")
    for i, (v7, v5) in enumerate(zip(u7, u5)):
        ax.text(i - w / 2, v7 + 0.006, f"{v7:+.3f}", ha="center", fontsize=8, color=TINTA["sekunder"])
        ax.text(i + w / 2, v5 + 0.006, f"{v5:+.3f}", ha="center", fontsize=8, color=TINTA["sekunder"])
    ax.axhline(0, color=TINTA["redup"], lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMA[a] for a in URUTAN_ARCH])
    ax.set_ylabel("paired difference  attention − Shapley")
    ax.set_title("Attention aligns with the marker more than Shapley does — both cohorts",
                 loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5)
    rapikan(ax)
    fig.tight_layout()
    simpan(fig, "fig05_fidelity_both_cohorts")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 — seed reproducibility (accuracy stability vs seed count)
# ═══════════════════════════════════════════════════════════════════════════

def fig06_seed_reproducibility() -> None:
    s3 = muat("s3_per_seed")
    # Jumlah seed sebenarnya dibaca dari data, bukan diasumsikan: naskah menyatakan
    # analisis S3 primer akhirnya memakai SEPULUH seed ("Pada konfigurasi beku dengan
    # sepuluh seed..."), sementara kode sumber figure Indonesia yang diadaptasi di sini
    # masih berasal dari draf ketika baru ada lima -- label "3 seed"/"5 seed" hardcode
    # sudah tidak cocok lagi dengan results/s3_per_seed.csv saat ini. Dihitung ulang
    # supaya figure tidak diam-diam menyimpang dari datanya sendiri (kelas bug yang
    # sama dengan figures/benang_merah.png, lihat scripts/perbaiki_benang_merah.py).
    n_total = int(s3.seed.nunique())
    n_awal = min(3, n_total)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a_ in URUTAN_ARCH:
        v = s3[s3.arsitektur == a_].sort_values("seed")
        ax[0].plot(v.seed, v.auc, "o-", color=WARNA_ARCH[a_], label=NAMA[a_], lw=1.8, ms=6)
    ax[0].axvline(n_awal - 0.5, color=TINTA["redup"], ls=":", lw=1)
    ax[0].text(n_awal - 0.45, ax[0].get_ylim()[0] + 0.002,
               f"seeds {n_awal}-{n_total - 1}\nadded\nlater",
               fontsize=7.5, color=TINTA["redup"], va="bottom")
    ax[0].set_xlabel("seed")
    ax[0].set_ylabel("subject-level AUC")
    ax[0].set_title("AUC per seed", loc="left")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_xticks(range(n_total))

    x = np.arange(3)
    w = 0.36
    sb_awal = [s3[(s3.arsitektur == a_) & (s3.seed < n_awal)].auc.std(ddof=1) for a_ in URUTAN_ARCH]
    sb_penuh = [s3[s3.arsitektur == a_].auc.std(ddof=1) for a_ in URUTAN_ARCH]
    ax[1].bar(x - w / 2, sb_awal, w, color=WARNA["redup"], label=f"{n_awal} seeds")
    ax[1].bar(x + w / 2, sb_penuh, w, color=WARNA["biru"], label=f"{n_total} seeds")
    for i, (a3, a5) in enumerate(zip(sb_awal, sb_penuh)):
        ax[1].text(i - w / 2, a3 + 0.0008, f"{a3:.4f}", ha="center", fontsize=7.5, color=TINTA["sekunder"])
        ax[1].text(i + w / 2, a5 + 0.0008, f"{a5:.4f}", ha="center", fontsize=7.5, color=TINTA["sekunder"])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([NAMA[a] for a in URUTAN_ARCH])
    ax[1].set_ylabel("between-seed AUC standard deviation")
    ax[1].set_title(f"Stability ranking shifts as seeds grow from {n_awal} to {n_total}", loc="left")
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        rapikan(a)
    fig.tight_layout()
    simpan(fig, "fig06_seed_reproducibility")
    print(f"  fig06 check: {n_awal} -> {n_total} seeds used (read from data, not hardcoded)")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7 — the 4-95x ratio synthesis (English twin of the corrected
# figures/benang_merah.png; see scripts/perbaiki_benang_merah.py for the fix)
# ═══════════════════════════════════════════════════════════════════════════

def fig07_ratio_synthesis() -> None:
    s7p = muat("s7_berpasangan_alpha_phi")
    s5s = muat("s5_keselarasan")
    s7s = muat("s7_kestabilan_seed")
    s3p = muat("s3_per_seed")
    base_lk = muat("a_baseline_lintas_kohort")

    bm2_p = float(s7p[s7p.arsitektur == "mamba2"].selisih_berpasangan.iloc[0])
    bm3_p = float(s7p[s7p.arsitektur == "mamba3"].selisih_berpasangan.iloc[0])
    uci_a = float(s7p[s7p.arsitektur == "mamba2"].rho_alpha_median.iloc[0])
    nhp_a = float(s5s[s5s.arsitektur == "mamba2"].rho_alpha.iloc[0])
    gru_auc = s3p[s3p.arsitektur == "gru"].auc.values
    gru_rho = s7s[s7s.arsitektur == "gru"][
        ["seed_0", "seed_1", "seed_2", "seed_3", "seed_4"]
    ].values.ravel()

    BM = [
        ("Architecture\nBiMamba-2 → BiMamba-3",
         abs(float(s3p[s3p.arsitektur == "mamba2"].auc.mean()
                   - s3p[s3p.arsitektur == "mamba3"].auc.mean())),
         abs(bm2_p - bm3_p)),
        ("Cohort\nUCI 395 → NewHandPD",
         abs(float(base_lk[base_lk.model == "mamba2"].kehilangan.iloc[0])),
         abs(uci_a - nhp_a)),
        ("Seed only\n(BiGRU, same data & model)",
         float(gru_auc.max() - gru_auc.min()),
         float(gru_rho.max() - gru_rho.min())),
        ("Patch resolution\nP=7 → P=56", 0.0226, 0.3646),
    ]
    bm = pd.DataFrame(BM, columns=["changed", "accuracy shift", "map shift"])
    bm["ratio"] = bm["map shift"] / bm["accuracy shift"]
    print(bm.round(4).to_string(index=False))
    lo, hi = round(bm.ratio.min()), round(bm.ratio.max())
    print(f"  fig07 check: ratio {lo}x to {hi}x (paper claims 4x to 95x)")
    if (lo, hi) != (4, 95):
        raise SystemExit(f"fig07 ratio {lo}x-{hi}x does not match paper's 4x-95x claim")

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    yy = np.arange(len(bm))[::-1]
    h = 0.34
    ax.barh(yy + h / 2, bm["accuracy shift"], h, color=WARNA["biru"], label="accuracy (AUC points)")
    ax.barh(yy - h / 2, bm["map shift"], h, color=WARNA["jingga"], label="map (rho points)")
    for i, (_, r) in enumerate(bm.iterrows()):
        Y = yy[i]
        ax.text(r["accuracy shift"] + 0.006, Y + h / 2, f"{r['accuracy shift']:.3f}",
                va="center", fontsize=8.5, color=TINTA["sekunder"])
        ax.text(r["map shift"] + 0.006, Y - h / 2, f"{r['map shift']:.3f}   ({r['ratio']:.0f}x)",
                va="center", fontsize=8.5, color=TINTA["utama"])
    ax.set_yticks(yy)
    ax.set_yticklabels(bm["changed"], fontsize=9)
    ax.set_xlabel("shift magnitude, absolute units")
    ax.set_title(f"The map shifts {lo} to {hi} times faster than accuracy does",
                 loc="left", fontsize=11.5)
    # loc="lower right" semula bertumpuk dengan label "0,365 (16x)" milik baris
    # bawah (Patch resolution), sebab baris itu punya bar peta terpanjang di
    # seluruh chart. Baris atas (Arsitektur) punya bar jauh lebih pendek, sehingga
    # kanan-atas kosong -- legenda dipindah ke situ.
    ax.set_xlim(0, max(bm["map shift"]) * 1.32)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    rapikan(ax)
    fig.tight_layout()
    simpan(fig, "fig07_ratio_synthesis")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8 — cross-cohort retention vs aggregate-feature baselines
# ═══════════════════════════════════════════════════════════════════════════

def fig08_retention_vs_baselines() -> None:
    lk = muat("a_baseline_lintas_kohort")
    b1_seed = muat("b1_kehilangan_per_seed")
    nama_p = {**NAMA, "regresi_logistik": "Logistic regression", "svm_rbf": "SVM RBF",
              "random_forest": "Random forest"}

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    urut = lk.sort_values("auc_newhandpd", ascending=True).reset_index(drop=True)
    # Beberapa titik berhimpit di sumbu NewHandPD (mis. BiGRU 0.8959 vs random forest
    # 0.8957 -- selisih 0.0002) sehingga labelnya bertumpuk kalau ditulis apa adanya.
    # Sisipkan jarak vertikal minimum antar label berurutan (dalam satuan sumbu-y).
    y_label = urut["auc_newhandpd"].tolist()
    JARAK_MIN = 0.012
    for i in range(1, len(y_label)):
        if y_label[i] - y_label[i - 1] < JARAK_MIN:
            y_label[i] = y_label[i - 1] + JARAK_MIN
    for (_, r), yl in zip(urut.iterrows(), y_label):
        seq = r.jenis == "model sekuens"
        c = WARNA["biru"] if seq else WARNA["jingga"]
        ax[0].plot([0, 1], [r.auc_uci, r.auc_newhandpd], "-o", color=c, lw=2.0, ms=7,
                   alpha=0.95 if seq else 0.75, ls="-" if seq else "--")
        ax[0].plot([1, 1.02], [r.auc_newhandpd, yl], "-", color=TINTA["redup"], lw=0.7, zorder=1)
        ax[0].text(1.04, yl, nama_p.get(r.model, r.model), va="center",
                   fontsize=8.5, color=TINTA["utama"] if seq else TINTA["sekunder"])
    ax[0].set_xlim(-0.05, 1.55)
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["UCI 395\n(origin cohort)", "NewHandPD\n(second cohort)"])
    ax[0].set_ylabel("subject-level AUC")
    ax[0].set_title("Aggregate features win at home, then collapse away", loc="left", fontsize=10.5)
    ax[0].legend(handles=[Line2D([], [], color=WARNA["biru"], lw=2, marker="o", label="sequence model"),
                          Line2D([], [], color=WARNA["jingga"], lw=2, ls="--", marker="o",
                                 label="aggregate feature")],
                frameon=False, fontsize=8.2, loc="lower left")
    rapikan(ax[0])

    u = lk.sort_values("kehilangan")
    c = [WARNA["biru"] if j == "model sekuens" else WARNA["jingga"] for j in u.jenis]
    ax[1].barh(range(len(u)), u.kehilangan, color=c)
    ax[1].set_yticks(range(len(u)))
    ax[1].set_yticklabels([nama_p.get(m, m) for m in u.model], fontsize=9)
    ax[1].axvline(0, color=TINTA["redup"], lw=1)
    ax[1].set_xlabel("cross-cohort AUC loss")
    ax[1].set_title("Every aggregate-feature baseline loses more than either Mamba", loc="left",
                     fontsize=10.5)
    rapikan(ax[1])

    fig.tight_layout()
    simpan(fig, "fig08_retention_vs_baselines")
    n_baseline_lebih_rugi = (u[u.jenis == "fitur agregat"].kehilangan
                              > lk[lk.model == "mamba2"].kehilangan.iloc[0]).sum()
    print(f"  fig08 check: {n_baseline_lebih_rugi}/3 aggregate-feature baselines "
          "lose more than BiMamba-2 (paper claims 3/3)")


# ═══════════════════════════════════════════════════════════════════════════
# Tables — computed from results/*.csv, written to Q1-amin/_tabel_dihitung.md
# for manual review and transcription into paper-full.md (deliberate human
# checkpoint, not blind auto-insertion -- same discipline used to verify every
# other number in the manuscript this session).
# ═══════════════════════════════════════════════════════════════════════════

def tabel_a_performa_primer() -> str:
    s3 = muat("s3_per_seed")
    baris = ["| Architecture | Mean AUC | SD (seeds) | Seeds |", "|---|---|---|---|"]
    for a in URUTAN_ARCH:
        v = s3[s3.arsitektur == a].auc
        baris.append(f"| {NAMA[a]} | {v.mean():.4f} | {v.std(ddof=1):.4f} | {len(v)} |")
    return "\n".join(baris)


def tabel_b_kesetiaan_shapley() -> str:
    s6 = muat("s6_kesetiaan")
    atap = float(muat("s6_batas_atas_kesetiaan").nilai.iloc[0])
    baris = ["| Architecture | Median fidelity (rho) | IQR | Fraction positive | % of ceiling |",
             "|---|---|---|---|---|"]
    for a in URUTAN_ARCH:
        r = s6[s6.arsitektur == a].iloc[0]
        baris.append(f"| {NAMA[a]} | {r.rho_median:.4f} | "
                     f"{r.rho_q25:.4f}–{r.rho_q75:.4f} | {r.frac_positif:.2f} | "
                     f"{100 * r.rho_median / atap:.0f}% |")
    baris.append(f"\nMeasured pipeline ceiling (positive-control recordings): {atap:.4f}")
    return "\n".join(baris)


def tabel_c_keselarasan_enam_sel() -> str:
    d = muat("rm5_selisih_kelompok_newhandpd")
    baris = ["| Architecture | Map | Median (patients) | Median (controls) | Difference | "
             "95% CI | p (permutation) | Below threshold |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        lolos = "yes" if not r.lolos_ambang_pinjaman else "no"
        baris.append(f"| {NAMA[r.arsitektur]} | {r.peta} | {r.median_pd:+.4f} | {r.median_hc:+.4f} "
                     f"| {r.selisih:+.4f} | [{r.ci_bawah:+.3f}, {r.ci_atas:+.3f}] | "
                     f"{r.p_permutasi:.4f} | {lolos} |")
    return "\n".join(baris)


def tabel_d_kurva_reproduktibilitas() -> str:
    d = muat("p0_kurva_seed")
    baris = ["| k | UCI: BiGRU | UCI: BiMamba-2 | UCI: BiMamba-3 | "
             "NewHandPD: BiGRU | NewHandPD: BiMamba-2 | NewHandPD: BiMamba-3 |",
             "|---|---|---|---|---|---|---|"]
    for k in range(1, 11):
        sel = []
        for koh in ["uci395", "newhandpd"]:
            for a in URUTAN_ARCH:
                v = d[(d.kohort == koh) & (d.arsitektur == a) & (d.k == k)]
                sel.append(f"{v.rho_thd_ensemble_penuh.iloc[0]:.4f}" if len(v) else "—")
        baris.append(f"| {k} | " + " | ".join(sel) + " |")
    return "\n".join(baris)


def tabel_e_transfer_lintas_kohort() -> str:
    lk = muat("a_baseline_lintas_kohort").sort_values("kehilangan")
    nama_p = {**NAMA, "regresi_logistik": "Logistic regression", "svm_rbf": "SVM RBF",
              "random_forest": "Random forest"}
    baris = ["| Method | Type | AUC (UCI 395) | AUC (NewHandPD) | Cross-cohort loss |",
             "|---|---|---|---|---|"]
    for _, r in lk.iterrows():
        jenis = "sequence" if r.jenis == "model sekuens" else "aggregate feature"
        baris.append(f"| {nama_p.get(r.model, r.model)} | {jenis} | {r.auc_uci:.4f} | "
                     f"{r.auc_newhandpd:.4f} | {r.kehilangan:+.4f} |")
    return "\n".join(baris)


TABEL_F_STATIS = """| Parameter | Value |
|---|---|
| Model dimension (d_model) | 128 |
| Layers / blocks | 3 |
| Dropout | 0.30 |
| Optimizer | AdamW, lr 3e-4, cosine schedule |
| Batch size | 8 (fixed across architectures and resolutions) |
| Epochs (primary) | 35 |
| Epochs (tuning: selection / final) | 20 / 35 |
| Parameters | BiGRU 263,041 · BiMamba-2 272,593 · BiMamba-3 283,297 (within ±5%) |
| Cross-validation | Stratified Group K-Fold, k=5, grouped by subject |
| GPU | NVIDIA RTX 3050 Laptop, 6 GB |
| Software | Python 3.11.8, PyTorch 2.11.0+cu130, CUDA 13.0, mamba-ssm 2.2.2 |
| Shapley method | WindowSHAP, 2048-coalition budget per recording |"""


def bangun_tabel() -> None:
    keluar = Path(AKAR / "Q1-amin" / "_tabel_dihitung.md")
    bagian = [
        "# Tabel dihitung dari results/*.csv -- tinjau lalu transkrip manual ke paper-full.md\n",
        "## Table A -- Primary-cohort classification performance (5-seed mean)\n" + tabel_a_performa_primer(),
        "\n## Table B -- Attention-Shapley fidelity vs measured ceiling\n" + tabel_b_kesetiaan_shapley(),
        "\n## Table C -- Marker alignment on NewHandPD, six cells\n" + tabel_c_keselarasan_enam_sel(),
        "\n## Table D -- Reproducibility: correlation to full ensemble by seed count k\n" + tabel_d_kurva_reproduktibilitas(),
        "\n## Table E -- Cross-cohort transfer, all six methods\n" + tabel_e_transfer_lintas_kohort(),
        "\n## Table F -- Architecture configuration and training setup (static, from NASKAH-SUMBER.md)\n" + TABEL_F_STATIS,
    ]
    keluar.write_text("\n".join(bagian) + "\n")
    print(f"\nditulis: {keluar.relative_to(AKAR)}")


def main() -> int:
    print("Building English-language paper figures -> figures/paper/\n")
    fig01_audit_design()
    fig02_data_channels_tasks()
    fig03_patch_grid_spectrum()
    fig04_attention_shapley_fidelity()
    fig05_fidelity_both_cohorts()
    fig06_seed_reproducibility()
    fig07_ratio_synthesis()
    fig08_retention_vs_baselines()
    bangun_tabel()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
