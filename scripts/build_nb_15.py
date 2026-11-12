"""Bangun notebook tambahan E0/E1 tanpa menjalankan satu pun sel analisis.

Jalankan dari root proyek: .venv/bin/python scripts/build_nb_15.py
Notebook lama, hasil lama, dan status preregistrasi tidak diubah.
"""

from pathlib import Path

import nbformat as nbf


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text))

    md("""# Tahap 15 — Analisis tambahan E0/E1: prediksi dan attention map

**Status: tambahan eksploratori, dirancang setelah hasil S3/S5 diketahui.**
Notebook ini tidak mengubah preregistrasi atau keputusan analisis lama. Ia membaca
artefak yang sudah ada, berjalan di CPU, dan tidak memuat model, melakukan tuning,
training, inferensi baru, SHAP baru, atau perubahan manuskrip.

- **E0:** probabilitas per rekaman → mean probabilitas per subjek → AUROC per
  seed → mean AUROC antar-seed. Bootstrap subjek mempertahankan seluruh keluaran
  model/seed yang berpasangan dalam setiap kohort; kedua kohort di-bootstrap independen.
- **E1:** perubahan probabilitas, logit, dan attention pada rekaman yang sama antar-seed,
  memakai logit dan peta yang tersimpan bersama pada artefak NewHandPD.

CI bersyarat pada model OOF, split, konfigurasi, dan seed yang tersedia. Resampling
subjek tidak melatih ulang fold yang saling bergantung. Hasil tidak menguji ketahanan
satu model beku pada kohort baru. **Perubahan peta bukan bukti hilangnya faithfulness;
faithfulness langsung tetap pertanyaan E2.**

Pilih kernel **Skripsi (.venv)**, lalu **Restart Kernel and Run All** agar modul
yang dimuat sesuai dengan snapshot sumber. Detail tersedia di
`scripts/E0-E1.md`. Keluaran baru masuk subdirektori bertimestamp; hasil lama tidak ditimpa.
""")

    code("""from pathlib import Path
import sys, json, platform
from datetime import datetime, timezone
from importlib.metadata import version

def cari_root():
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "src/audit_e0_e1.py").exists() and (path / "results").is_dir():
            return path
    raise FileNotFoundError("jalankan notebook dari root proyek atau notebooks/")

ROOT = cari_root()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from audit_e0_e1 import (
    STATUS_ANALISIS, BATAS_INFERENSI, E1_METRICS, file_sha256, cache_manifest,
    prepare_cohort, subject_scores, reference_check, e0_bootstrap, e1_pairs,
    e1_subject_summary, e1_subject_relations, e1_sensitivity, bootstrap_subject_table,
)
from viz import WARNA, pasang_gaya, rapikan
pasang_gaya()
print("Root:", ROOT)
print("Status:", STATUS_ANALISIS)
print("Batas:", BATAS_INFERENSI)""")

    md("""## 1. Pilihan analisis tambahan

BiGRU dan BiMamba-2 adalah perbandingan utama tambahan. Daftar seed **eksplisit**,
bukan irisan seed yang kebetulan tersedia. Run atau peta yang hilang menyebabkan
analisis berhenti. Tidak ada imputasi seed lama menjadi 0.

Ambang probabilitas/logit hanya analisis sensitivitas. Semua pasangan tetap dipakai
untuk ringkasan kontinu. Ambang kelas memakai logit 0 (probabilitas 0,5), sebagai
kesepakatan keputusan model, bukan ambang diagnostik yang telah divalidasi.
""")

    code("""ARMS = ("gru", "mamba2")
SEEDS = tuple(range(10))
PATCH = 7
FS = 100.0
EXPECTED_CONFIG = {
    "patch_size": PATCH, "epochs": 35, "in_channels": 6,
    "d_model": 128, "n_layers": 3, "dropout": 0.30, "pooling": "attention",
}
N_BOOT = 5000
BOOT_SEED_E0 = 1500
BOOT_SEED_E1 = 1501
CONFIDENCE = 0.95
TOP_FRACTION = 0.10
EPS_PROB = (0.01, 0.02, 0.05)
EPS_LOGIT = (None, 0.10, 0.25, 0.50)

# Referensi 4 desimal yang telah dilaporkan, bukan hasil baru notebook ini.
HEADLINES = {
    "uci395": {"gru": 0.9503, "mamba2": 0.9338},
    "newhandpd": {"gru": 0.8959, "mamba2": 0.9307},
}
INPUTS = {
    "uci_artifact": ROOT / "results/s3_artefak.pkl",
    "new_artifact": ROOT / "results/s5_artefak.pkl",
    "uci_cache": ROOT / "data/cache/uci395_fs100.npz",
    "new_cache": ROOT / "data/cache/newhandpd_spiral_fs100.npz",
    "uci_reference": ROOT / "results/s3_per_seed.csv",
    "new_reference": ROOT / "results/s5_per_seed.csv",
    "frozen_parameters": ROOT / "results/s2_parameter_dibekukan.csv",
}
SOURCE_FILES = [ROOT / p for p in (
    "src/audit_e0_e1.py", "scripts/build_nb_15.py",
    "scripts/build_nb_07.py", "scripts/build_nb_12.py",
    "notebooks/07_eksperimen_utama.ipynb", "notebooks/12_replikasi_newhandpd.ipynb",
    "src/preprocessing.py", "src/newhandpd.py", "src/model.py", "src/training.py",
)]
for path in [*INPUTS.values(), *SOURCE_FILES]:
    if not path.is_file():
        raise FileNotFoundError(f"input wajib tidak tersedia: {path}; tidak dibuat/dilatih otomatis")
frozen = pd.read_csv(INPUTS["frozen_parameters"])
if int(frozen.patch_halus.iloc[0]) != PATCH or int(frozen.epoch_dibekukan.iloc[0]) != EXPECTED_CONFIG["epochs"]:
    raise ValueError("PATCH/epochs audit berbeda dari metadata parameter beku")
inventory = pd.DataFrame([
    {"path": str(p.relative_to(ROOT)), "sha256_before": file_sha256(p), "bytes": p.stat().st_size}
    for p in [*INPUTS.values(), *SOURCE_FILES]
])
display(inventory[["path", "bytes", "sha256_before"]])""")

    md("""## 2. Pemeriksaan identitas dan batas provenans

Pemeriksaan keras: seed/run unik dan lengkap; panjang logit sesuai cache; identitas
rekaman `(subjek, task, segmen)` unik; label konsisten; setiap peta S5 cocok dengan
`idx`, subjek dan label; panjang peta tepat `floor(T/7)`; α finite/nonnegatif dengan
massa 1. Tidak ada crop, interpolasi, pergeseran peta, atau irisan rekaman diam-diam.
Patch j dimulai pada `j*0,07` detik dan berakhir pada `(j+1)*0,07` detik. Sisa sampel
di luar patch penuh dicatat dalam `tail_samples`.

**Batas artefak lama:** fold historis, konfigurasi lengkap, hash cache saat training,
dan model/checkpoint ID tidak tersedia. Fold direkonstruksi dengan SGKF5,
`shuffle=True, random_state=42` pada urutan cache saat audit. Ini hanya rekonstruksi.
Konfigurasi di atas adalah ekspektasi dari kode, bukan metadata training yang ditemukan.
Hash baru mencatat keadaan saat audit; ia tidak membuktikan identitas berkas lama.
Logit/peta S5 berada dalam entri run yang sama, tetapi bobot historis tidak dapat
diautentikasi. Laporan mempertahankan perbedaan status tersebut.
""")

    code("""records_uci = cache_manifest(INPUTS["uci_cache"], "uci395", PATCH, FS)
records_new = cache_manifest(INPUTS["new_cache"], "newhandpd", PATCH, FS)
uci = prepare_cohort(INPUTS["uci_artifact"], records_uci, ARMS, SEEDS, EXPECTED_CONFIG)
new = prepare_cohort(INPUTS["new_artifact"], records_new, ARMS, SEEDS,
                     EXPECTED_CONFIG, require_maps=True)
checks = pd.concat([uci.checks, new.checks], ignore_index=True)
display(checks.groupby(["cohort", "field", "status"]).size().rename("n_run").reset_index())
display(new.records[["idx", "record_id", "subject", "label", "task", "fold_reconstructed",
                     "n_samples", "n_patches", "tail_samples", "patch_seconds"]].head(12))
print("Pemeriksaan keselarasan lulus. Metadata berstatus tidak_tersimpan/rekonstruksi tetap belum terverifikasi.")""")

    md("""## 3. E0 — cocokkan estimator dengan angka headline sebelum bootstrap

Probabilitas diambil dari sigmoid logit setiap rekaman. Probabilitas **dirata-ratakan
per subjek untuk setiap seed**, lalu AUROC dihitung per seed. Tidak ada mean logit,
dan tidak ada penghitungan AUROC pada probabilitas ensemble.

Referensi per-seed S3/S5 harus cocok pada toleransi numerik 1e-8. Mean AUROC per-seed
juga harus cocok dengan headline yang dibulatkan empat desimal. Ketidakcocokan
menghentikan analisis agar estimator/data yang berbeda tidak dianggap mengonfirmasi headline.
""")

    code("""scores = {"uci395": subject_scores(uci), "newhandpd": subject_scores(new)}
references = pd.concat([
    reference_check(scores["uci395"], INPUTS["uci_reference"], HEADLINES["uci395"]),
    reference_check(scores["newhandpd"], INPUTS["new_reference"], HEADLINES["newhandpd"]),
], ignore_index=True)
per_seed = pd.concat([scores[c].per_seed for c in scores], ignore_index=True)
per_subject_seed = pd.concat([scores[c].per_subject_seed for c in scores], ignore_index=True)
display(references)
display(per_seed.pivot(index=["cohort", "seed"], columns="architecture", values="auroc"))

# Direktori baru hanya dibuat setelah pemeriksaan wajib dan referensi berhasil.
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "results/e0_e1" / RUN_ID
FIG = ROOT / "figures/e0_e1" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
FIG.mkdir(parents=True, exist_ok=False)

def save_table(frame, filename):
    frame.to_csv(OUT / filename, index=False)

def save_figure(figure, filename):
    figure.savefig(FIG / f"{filename}.png", dpi=180, bbox_inches="tight")
    figure.savefig(FIG / f"{filename}.pdf", bbox_inches="tight")

save_table(inventory, "input_inventory.csv")
save_table(checks, "alignment_provenance.csv")
save_table(pd.concat([uci.records, new.records], ignore_index=True), "record_manifest.csv")
save_table(references, "e0_reference_check.csv")
save_table(per_seed, "e0_auroc_per_seed.csv")
save_table(per_subject_seed, "e0_probability_subject_seed.csv")

manifest = {
    "run_id": RUN_ID, "status": STATUS_ANALISIS,
    "analysis_design_record": "2026-10-12; tambahan setelah hasil lama diketahui",
    "old_preregistration_changed": False, "training_performed": False,
    "e0_estimator": "sigmoid per rekaman -> mean probabilitas per subjek -> AUROC per seed -> mean AUROC seed",
    "e0_contrast": "(AUROC_UCI_mamba2-AUROC_New_mamba2)-(AUROC_UCI_gru-AUROC_New_gru)",
    "bootstrap": "subjek berstrata kelas; multiplicity sama semua model/seed; kohort independen; seed tetap",
    "e1_aggregation": "mean pasangan seed eligible per rekaman -> mean rekaman eligible per subjek -> mean subjek",
    "e1_relation": "Spearman perubahan output vs JSD per rekaman -> mean rekaman per subjek -> mean subjek",
    "e1_ci": "pool subjek penuh, termasuk subjek tanpa pasangan eligible; mean conditional mengabaikan NaN",
    "inference_limit": BATAS_INFERENSI,
    "historical_provenance_limit": "fold/config/hash cache training/model_id tidak tersimpan; sumber saat ini dan rekonstruksi dicatat",
    "arms": ARMS, "seeds": SEEDS, "expected_config_not_historical_metadata": EXPECTED_CONFIG,
    "fs": FS, "top_fraction": TOP_FRACTION, "n_boot": N_BOOT, "confidence": CONFIDENCE,
    "bootstrap_seeds": {"e0": BOOT_SEED_E0, "e1": BOOT_SEED_E1},
    "sensitivity_eps_prob": EPS_PROB, "sensitivity_eps_logit": EPS_LOGIT,
    "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")},
    "python": platform.python_version(), "completion_status": "started",
}
(OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Output baru:", OUT)
print("Grafik baru:", FIG)""")

    md("""## 4. E0 — bootstrap subjek dan selisih perubahan AUROC

Di setiap draw, subjek PD dan HC diambil dengan pengembalian dalam masing-masing
kohort, mempertahankan jumlah per kelas. Seluruh skor semua arm/seed milik subjek
tersebut ikut diambil bersama. Kohort diambil independen; seed tidak di-resample.

`Delta_m = mean_AUROC_UCI,m - mean_AUROC_NewHandPD,m`.
Contrast primer tambahan adalah `Delta_mamba2 - Delta_gru`; nilai negatif berarti
penurunan AUROC BiMamba-2 lebih kecil. Point estimate dihitung pada data asli,
**bukan** mean bootstrap. CI percentile 95% bersyarat pada artefak OOF tetap.

Untuk efisiensi CPU, implementasi memakai kernel pasangan PD–HC dengan bobot 0,5
untuk ties, dirata-ratakan antar-seed. Ini secara aljabar sama dengan mean AUROC
per seed dan diperiksa terhadap sklearn pada skor subjek asli.
""")

    code("""e0 = e0_bootstrap(scores, n_boot=N_BOOT, random_seed=BOOT_SEED_E0, confidence=CONFIDENCE)
for name in ("cohort", "delta", "contrast", "draws"):
    save_table(e0[name], f"e0_{name}.csv")
display(e0["cohort"])
display(e0["delta"])
display(e0["contrast"])
print("CI memuat nol: belum menentukan arah contrast; bukan bukti kesetaraan.")
print("CI sepenuhnya negatif: mendukung retensi relatif pada dua pipeline yang dilatih ulang, dengan batas inferensi di atas.")""")

    code("""colors = {"gru": WARNA["biru"], "mamba2": WARNA["jingga"]}
labels = {"gru": "BiGRU", "mamba2": "BiMamba-2"}
fig, ax = plt.subplots(figsize=(7.4, 4.4))
for offset, arm in zip((-0.035, 0.035), ARMS):
    table = e0["cohort"].set_index(["architecture", "cohort"])
    rows = table.loc[arm].reindex(["uci395", "newhandpd"])
    x = np.arange(2) + offset
    ax.plot(x, rows.mean_auroc, marker="o" if arm == "gru" else "s",
            color=colors[arm], label=labels[arm])
    for xx, row in zip(x, rows.itertuples()):
        ax.vlines(xx, row.ci_low, row.ci_high, color=colors[arm], lw=2)
    for xx, cohort in zip(x, ("uci395", "newhandpd")):
        seed_values = per_seed.loc[(per_seed.architecture == arm) & (per_seed.cohort == cohort), "auroc"]
        ax.scatter(xx + np.linspace(-.015, .015, len(seed_values)), seed_values,
                   s=12, alpha=.45, color=colors[arm])
ax.set_xticks([0, 1], ["UCI 395", "NewHandPD"])
ax.set_ylim(0, 1.02)
ax.set_ylabel("Mean AUROC per seed; CI bootstrap subjek")
ax.set_title("E0 — pipeline dilatih ulang terpisah pada setiap dataset", loc="left")
ax.legend(); rapikan(ax); fig.tight_layout()
save_figure(fig, "e0_auroc_cohort"); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7))
for i, row in enumerate(e0["delta"].itertuples()):
    axes[0].hlines(i, row.ci_low, row.ci_high, color=colors[row.architecture], lw=2)
    axes[0].plot(row.delta_uci_minus_new, i, "o", color=colors[row.architecture])
axes[0].set_yticks(range(len(ARMS)), [labels[a] for a in ARMS])
axes[0].set_xlabel("Delta AUROC: UCI − NewHandPD")
row = e0["contrast"].iloc[0]
axes[1].hlines(0, row.ci_low, row.ci_high, color=WARNA["jingga"], lw=2)
axes[1].plot(row.contrast, 0, "s", color=WARNA["jingga"])
axes[1].set_yticks([0], ["Delta BiMamba-2 − Delta BiGRU"])
axes[1].set_xlabel("Selisih perubahan AUROC")
for ax in axes:
    ax.axvline(0, color=WARNA["netral"], linestyle=":"); rapikan(ax)
fig.suptitle("E0 — CI 95% bersyarat pada model dan split tersedia")
fig.tight_layout(); save_figure(fig, "e0_delta_contrast"); plt.show()""")

    md("""## 5. E1 — perubahan kontinu pada rekaman yang sama

Seluruh pasangan seed dipakai (10 seed menghasilkan 45 pasangan per rekaman).
Pasangan seed bukan 45 subjek baru. Tidak ada uji p-value dengan pasangan sebagai
observasi independen.

- `abs_delta_prob` dan `abs_delta_logit`: perubahan output individual utama.
- `js_divergence`: JSD base-2 antara distribusi α, dalam [0,1], bukan akar JSD.
- `rho_alpha` dan `(1-rho)/2`: kesamaan ranking dan perubahan ranking.
- `top_jaccard_fractional`: Jaccard top-10%, dengan massa fraksional bila tie
  melintasi batas k. Tanpa tie, identik dengan Jaccard himpunan top-k biasa.
  Ties dan peta uniform dicatat. Spearman peta uniform tetap NaN, tidak diganti 1.

Ringkasan utama: mean pasangan per rekaman → mean rekaman per subjek → mean subjek.
CI mengambil subjek, bukan patch/rekaman/pasangan. Asosiasi kontinu output–JSD
dihitung per rekaman, lalu diringkas per subjek; ia tidak menyatakan sebab-akibat.
""")

    code("""pairs = e1_pairs(new, top_fraction=TOP_FRACTION)
record_continuous, subject_continuous = e1_subject_summary(pairs)
continuous_ci = bootstrap_subject_table(subject_continuous, E1_METRICS,
    n_boot=N_BOOT, random_seed=BOOT_SEED_E1, confidence=CONFIDENCE)
subject_relations = e1_subject_relations(pairs)
relation_ci = bootstrap_subject_table(subject_relations,
    ("rho_delta_prob_vs_jsd", "rho_delta_logit_vs_jsd"),
    n_boot=N_BOOT, random_seed=BOOT_SEED_E1, confidence=CONFIDENCE)
for filename, table in (
    ("e1_pairs.csv", pairs), ("e1_record_continuous.csv", record_continuous),
    ("e1_subject_continuous.csv", subject_continuous), ("e1_continuous_ci.csv", continuous_ci),
    ("e1_subject_relations.csv", subject_relations), ("e1_relation_ci.csv", relation_ci),
):
    save_table(table, filename)
display(continuous_ci)
display(relation_ci)
display(pairs.groupby("architecture").agg(n_pair_record=("idx", "size"),
    n_subjects=("subject", "nunique"), n_records=("idx", "nunique"),
    n_undefined_rho=("rho_alpha", lambda x: x.isna().sum()),
    n_uniform_a=("uniform_a", "sum"), n_uniform_b=("uniform_b", "sum")))""")

    code("""fig, axes = plt.subplots(len(ARMS), 2, figsize=(10.4, 7.0), squeeze=False)
for i, arm in enumerate(ARMS):
    d = pairs[pairs.architecture == arm].copy()
    # Massa total setiap subjek sama; bukan bobot independensi statistik.
    n_records = d.groupby("subject").idx.transform("nunique")
    n_pairs_record = d.groupby(["subject", "idx"]).idx.transform("size")
    weight = 1 / (d.subject.nunique() * n_records * n_pairs_record)
    for j, field in enumerate(("abs_delta_prob", "abs_delta_logit")):
        x = d[field].to_numpy()
        xmax = max(float(x.max()), 1e-8)
        histogram, xe, ye = np.histogram2d(x, d.js_divergence.to_numpy(), bins=35,
                                          range=[[0, xmax], [0, 1]], weights=weight)
        artist = axes[i, j].pcolormesh(xe, ye, np.ma.masked_where(histogram.T == 0, histogram.T),
                                      cmap="viridis", shading="auto", rasterized=True)
        fig.colorbar(artist, ax=axes[i, j], label="massa berbobot sama per subjek")
        axes[i, j].set_xlabel("|Delta probabilitas|" if j == 0 else "|Delta logit|")
        axes[i, j].set_ylabel("JSD attention (base 2)")
        axes[i, j].set_title(labels[arm], loc="left"); rapikan(axes[i, j])
fig.suptitle("E1 — perubahan output dan peta pada rekaman yang sama; seluruh pasangan seed")
fig.tight_layout(); save_figure(fig, "e1_continuous_output_map"); plt.show()""")

    md("""## 6. E1 — sensitivitas definisi output relatif stabil

Semua ambang probabilitas dikombinasikan dengan tanpa ambang logit dan tiga ambang
logit. Skenario logit saja serta semua pasangan juga disertakan. Skenario stabil
selalu mensyaratkan keputusan kelas sama. **Tidak ada ambang yang dijadikan keputusan
konfirmatori atau batas klinis.**

Filter diterapkan pada pasangan dalam rekaman yang sama. Mean metrik dihitung
atas pasangan eligible per rekaman, lalu rekaman eligible per subjek. Setiap
subjek berbobot sama dalam mean conditional. Subjek tanpa pasangan eligible tetap
ada dalam tabel/pool bootstrap dengan metrik NaN; cakupannya tidak disembunyikan.
`fraction_pairs_eligible` dan `fraction_records_eligible` diringkas atas seluruh
subjek. CI conditional dengan sedikit subjek atau sedikit draw valid harus dibaca
bersama kolom cakupan, bukan sebagai kesimpulan untuk seluruh kohort.
""")

    code("""sensitivity_subject = e1_sensitivity(pairs, eps_prob=EPS_PROB, eps_logit=EPS_LOGIT)
sensitivity_ci = bootstrap_subject_table(sensitivity_subject,
    E1_METRICS + ("fraction_pairs_eligible", "fraction_records_eligible"),
    group_columns=("architecture", "scenario", "eps_prob", "eps_logit", "require_same_class"),
    n_boot=N_BOOT, random_seed=BOOT_SEED_E1, confidence=CONFIDENCE)
save_table(sensitivity_subject, "e1_sensitivity_subject.csv")
save_table(sensitivity_ci, "e1_sensitivity_ci.csv")
display(sensitivity_ci[sensitivity_ci.metric.isin(
    ["js_divergence", "top_jaccard_fractional", "fraction_pairs_eligible"])])
print("Periksa n_subjects_valid, n_pd_valid, n_hc_valid dan n_boot_valid untuk setiap ambang.")""")

    code("""fig, axes = plt.subplots(len(ARMS), 2, figsize=(11.3, 7.2), squeeze=False)
for i, arm in enumerate(ARMS):
    for j, metric in enumerate(("js_divergence", "fraction_pairs_eligible")):
        values = np.full((len(EPS_LOGIT), len(EPS_PROB)), np.nan)
        coverage = np.zeros_like(values, dtype=int)
        for r, eps_l in enumerate(EPS_LOGIT):
            for c, eps_p in enumerate(EPS_PROB):
                d = sensitivity_ci[(sensitivity_ci.architecture == arm) &
                    (sensitivity_ci.metric == metric) & sensitivity_ci.require_same_class &
                    np.isclose(sensitivity_ci.eps_prob, eps_p)]
                d = d[d.eps_logit.isna()] if eps_l is None else d[np.isclose(d.eps_logit, eps_l)]
                if len(d) != 1:
                    raise ValueError("sel sensitivitas tidak unik")
                values[r, c] = d.estimate.iloc[0]
                coverage[r, c] = int(d.n_subjects_valid.iloc[0])
        artist = axes[i, j].imshow(np.ma.masked_invalid(values), vmin=0, vmax=1,
                                   cmap="viridis", aspect="auto")
        fig.colorbar(artist, ax=axes[i, j])
        for r in range(len(EPS_LOGIT)):
            for c in range(len(EPS_PROB)):
                text = f"{values[r,c]:.3f}\\nn={coverage[r,c]}" if np.isfinite(values[r,c]) else "NA\\nn=0"
                axes[i, j].text(c, r, text, ha="center", va="center", color="white", fontsize=8)
        axes[i, j].set_xticks(range(len(EPS_PROB)), [str(p) for p in EPS_PROB])
        axes[i, j].set_yticks(range(len(EPS_LOGIT)),
            ["tanpa batas" if v is None else str(v) for v in EPS_LOGIT])
        axes[i, j].set_xlabel("ambang |Delta probabilitas|")
        axes[i, j].set_ylabel("ambang |Delta logit|")
        title = "mean JSD conditional" if j == 0 else "fraksi pasangan eligible (semua subjek)"
        axes[i, j].set_title(f"{labels[arm]} — {title}", loc="left")
        axes[i, j].grid(False)
fig.suptitle("E1 — sensitivitas; kelas tetap; n adalah jumlah subjek untuk metrik pada sel")
fig.tight_layout(); save_figure(fig, "e1_sensitivity_coverage"); plt.show()""")

    md("""## 7. Batas interpretasi dan catatan penyelesaian

E0 menilai konsistensi performa **arsitektur/pipeline yang dilatih ulang** pada dua
dataset. CI bukan ketidakpastian atas semua kemungkinan training, split, atau kohort.
CI contrast yang mencakup nol bukan bukti ekuivalensi.

E1 mengukur perubahan attention, bukan kualitas penjelasan. Output stabil dan peta
berbeda dapat mencerminkan penjelasan yang tidak unik. Output stabil dan peta stabil
juga belum membuktikan faithfulness. Keduanya memerlukan intervensi input langsung E2.
E1 tidak menggunakan marker atau SHAP sebagai ground truth dan tidak membuat rasio
perubahan attention/AUROC. Semua ambang di sini merupakan sensitivitas tambahan.

Sebelum memilih E2/E3, bagikan manifest, laporan keselarasan, tabel E0 contrast,
ringkasan kontinu/asosiasi E1, dan tabel sensitivitas beserta cakupannya. Tidak ada
checkpoint yang dibuat atau dipulihkan oleh notebook ini.
""")

    code("""# Pastikan input/sumber tidak berganti selama notebook berjalan.
inventory["sha256_after"] = [file_sha256(ROOT / p) for p in inventory.path]
inventory["unchanged"] = inventory.sha256_before == inventory.sha256_after
save_table(inventory, "input_inventory.csv")
if not inventory.unchanged.all():
    manifest["completion_status"] = "invalid_inputs_changed"
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    raise RuntimeError("input/sumber berubah selama analisis; jangan gunakan hasil run ini")
manifest["completion_status"] = "complete"
manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
manifest["tables"] = sorted(p.name for p in OUT.glob("*.csv"))
manifest["figures"] = sorted(p.name for p in FIG.iterdir())
(OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
display(pd.DataFrame({"table": manifest["tables"]}))
print("Selesai:", OUT)
print("Grafik:", FIG)
print("Analisis tambahan; preregistrasi lama tetap. E1 tidak menyimpulkan faithfulness.")""")

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Skripsi (.venv)", "language": "python", "name": "skripsi"},
        "language_info": {"name": "python", "version": "3.11.8"},
        "analysis_status": STATUS_TEXT,
    }
    return nb


STATUS_TEXT = "Analisis tambahan eksploratori; tidak mengubah preregistrasi lama"


def main():
    notebook = build_notebook()
    nbf.validate(notebook)
    output = Path(__file__).resolve().parent.parent / "notebooks/15_audit_e0_e1.ipynb"
    nbf.write(notebook, output)
    print(f"ditulis: {output} ({len(notebook.cells)} sel, tanpa eksekusi)")


if __name__ == "__main__":
    main()
