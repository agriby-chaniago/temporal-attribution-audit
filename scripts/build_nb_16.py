"""Bangun notebook E2 + E1 pada checkpoint baru yang sama. Tidak mengeksekusi sel.

Jalankan: .venv/bin/python scripts/build_nb_16.py
Run All oleh pengguna dapat melatih 30 model jika checkpoint belum tersedia.
"""
from pathlib import Path

import nbformat as nbf


def build_notebook():
    cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))

    md("""# Tahap 16 — E2: intervensi input dan E1 pada model baru yang sama

**Analisis tambahan eksploratori**, disiapkan setelah E0/E1 lama diketahui.
Status preregistrasi, hasil lama, dan manuskrip tidak diubah.

**Run All notebook ini oleh pengguna dapat melakukan training ulang:** NewHandPD
spiral, BiGRU/BiMamba-2, seed 0/1/2, lima fold subjek = **30 model** bila tidak ada
checkpoint yang cocok. Agen hanya membangun notebook ini, tidak menjalankannya.
Tiga seed merupakan pilihan minimum praktis, bukan jaminan power atau reproduksi
seluruh hasil E1 sepuluh seed. Tidak ada pencarian hyperparameter.

Checkpoint baru menyimpan bobot, normalizer, split aktual, konfigurasi, hash data
dan kode. E1 dan E2 menggunakan logit, attention dan model yang sama. Data S3/S5
tidak ditimpa. Checkpoint dan hasil per-fold dapat dipakai ulang bila spec cocok.

Perbaikan teknis setelah run awal terhenti pada pemeriksaan anchor: baseline,
anchor dan setiap input bermasker sekarang memakai **forward batch 1**. Toleransi
anchor tetap, dan pemeriksaan dilakukan sebelum/sesudah intervensi. Checkpoint
training tetap kompatibel; cache evaluasi protokol lama tidak dipakai ulang.

**Primary E2:** apakah penghapusan input yang ditonjolkan attention mengubah logit
lebih besar daripada ranking acak? **Secondary:** apakah penghapusan tersebut
mengurangi dukungan terhadap kelas asli? Attention tidak bertanda; pengaruh input
dan arah dukungan kelas tidak disamakan. Hasil mengukur respons model terhadap
masker, bukan kausalitas fisiologis. Petunjuk: `scripts/E2.md`.

Pilih kernel Skripsi (.venv), baca sel konfigurasi, lalu Restart Kernel and Run All.
""")
    code("""from pathlib import Path
import sys, json, pickle, platform
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "src/audit_e2.py").is_file()), None)
if ROOT is None: raise FileNotFoundError("jalankan dari root proyek/notebooks")
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch
from sklearn.model_selection import StratifiedGroupKFold
from IPython.display import display
from newhandpd import muat_cache_bisp
from artefak import simpan_atomik
from audit_e0_e1 import (
    AlignmentError, PreparedCohort, STATUS_ANALISIS, E1_METRICS,
    cache_manifest, file_sha256, subject_scores, e1_pairs, e1_subject_summary,
    e1_subject_relations, e1_sensitivity, bootstrap_subject_table,
)
from e2_checkpoints import fold_spec, spec_digest, get_fold_model, FrozenPredictor
from audit_e2 import (evaluate_record, e2_record_scores, e2_subject_scores,
                      E2_METRICS, join_e1_e2, JOINT_METRICS, joint_subject_sensitivity)
from viz import pasang_gaya, WARNA, rapikan
pasang_gaya()""")
    md("""## Pemeriksaan sintetis sebelum training

Ketika pengguna menjalankan sel ini, pemeriksaan CPU memakai model linear buatan
untuk masker, donor, AOPC, konteks batch/anchor dan identitas E1/E2. Ia tidak membaca data penelitian
atau melatih model. Kegagalan menghentikan notebook sebelum training yang mahal.
Pemeriksaan numerik ini belum dijalankan oleh agen saat menyiapkan notebook.
""")
    code("""import unittest
suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_audit_e2.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.testsRun == 0 or not result.wasSuccessful():
    raise RuntimeError("pemeriksaan sintetis E2 gagal; training belum dimulai")""")
    md("""## 1. Konfigurasi yang ditetapkan sebelum E2 dijalankan

Ubah parameter melalui builder `.py`, lalu bangun ulang notebook. Pilihan masker,
budget, jumlah seed dan kontrol berikut dicatat sebelum hasil E2 diketahui.
Konfigurasi training mengikuti pipeline lama. Jangan memperlakukan AUROC tiga
seed baru sebagai reproduksi headline sepuluh seed, atau menambah seed/ambang
berdasarkan apakah hasil yang diinginkan telah muncul.
""")
    code("""ARMS = ("gru", "mamba2")
SEEDS = (0, 1, 2)
PATCH, FS = 7, 100.
MODEL_CONFIG = dict(in_channels=6, d_model=128, patch_size=PATCH, n_layers=3,
                    dropout=.30, d_dir=64, pooling="attention")
TRAINING_CONFIG = dict(epochs=35, batch_size=8, lr=3e-4)
ALLOW_TRAINING_IF_MISSING = True  # Run All dapat melatih 30 model baru.
FRACTIONS = (.05, .10, .20, .30)
RANDOM_REPEATS, DONOR_REPEATS = 20, 5
DONOR_BLOCK_PATCHES = 32  # blok kontigu 2,24 detik; semua kanal dipindahkan bersama
PERTURBATION_BATCH = 1  # wajib: baseline/anchor/intervensi memakai forward batch yang sama
INTERVENTION_SEED = 1600
N_BOOT, BOOT_SEED, CONFIDENCE = 5000, 1601, .95
EPS_PROB, EPS_LOGIT = (.01, .02, .05), (None, .1, .25, .5)
if PERTURBATION_BATCH != 1:
    raise ValueError("E2 memerlukan PERTURBATION_BATCH=1; training belum dimulai")
CACHE = ROOT / "data/cache/newhandpd_spiral_fs100.npz"
if not CACHE.is_file(): raise FileNotFoundError(CACHE)
if not torch.cuda.is_available():
    raise RuntimeError("Notebook model BiMamba-2 ini memerlukan lingkungan CUDA proyek. Training belum dimulai.")
DEVICE = "cuda"
VERSIONS = {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "torch")}
for package in ("mamba-ssm", "causal-conv1d", "triton", "einops"):
    try: VERSIONS[package] = version(package)
    except PackageNotFoundError: VERSIONS[package] = "metadata_version_unavailable"
VERSIONS["cuda_runtime"] = str(torch.version.cuda)
records = cache_manifest(CACHE, "newhandpd", PATCH, FS)
recordings = muat_cache_bisp(CACHE)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(splitter.split(np.zeros(len(records)), records.label, records.subject))
fold_actual = np.full(len(records), -1, int)
for fold, (train, test) in enumerate(splits):
    if set(records.subject.iloc[train]) & set(records.subject.iloc[test]):
        raise AlignmentError("subjek muncul di train dan test")
    fold_actual[test] = fold
records["fold_actual"] = fold_actual
if (fold_actual < 0).any(): raise AlignmentError("split tidak lengkap")
source_paths = [ROOT / p for p in (
    "src/audit_e2.py", "src/e2_checkpoints.py", "src/audit_e0_e1.py", "scripts/build_nb_16.py",
    "src/model.py", "src/training.py", "src/preprocessing.py", "src/newhandpd.py")]
inventory = pd.DataFrame([dict(path=str(p.relative_to(ROOT)), sha256_before=file_sha256(p))
                           for p in [CACHE, *source_paths]])
cache_hash = file_sha256(CACHE)
EVAL_SPEC = dict(evaluation_protocol_version=2, cohort="newhandpd", patch=PATCH, fs=FS, arms=ARMS, seeds=SEEDS,
    model_config=MODEL_CONFIG, training_config=TRAINING_CONFIG,
    fractions=FRACTIONS, random_repeats=RANDOM_REPEATS, donor_repeats=DONOR_REPEATS,
    donor_block_patches=DONOR_BLOCK_PATCHES, forward_batch_size=PERTURBATION_BATCH,
    unmasked_check=dict(phases=["before", "after"], atol=1e-4, rtol=1e-4),
    intervention_seed=INTERVENTION_SEED, cache_sha256=cache_hash,
    analysis_sources={p: file_sha256(ROOT / p) for p in ("src/audit_e2.py", "src/audit_e0_e1.py")},
    jsd=dict(kind="divergence", log_base=2, input="nonnegative alpha normalized to sum 1", range=[0,1]),
    primary="normalized deletion AOPC abs_delta_logit: alpha_top - random",
    secondary="signed_margin_drop and predicted_class_prob_drop")
evaluation_id = spec_digest(EVAL_SPEC)
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT, FIG = ROOT / "results/e2" / RUN_ID, ROOT / "figures/e2" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False); FIG.mkdir(parents=True, exist_ok=False)
inventory.to_csv(OUT / "input_inventory.csv", index=False)
records.to_csv(OUT / "record_manifest.csv", index=False)
manifest = dict(run_id=RUN_ID, evaluation_id=evaluation_id, evaluation_spec=EVAL_SPEC,
    status=STATUS_ANALISIS, old_preregistration_changed=False, versions=VERSIONS,
    python=platform.python_version(), device=torch.cuda.get_device_name(),
    completion_status="started", analysis_design_date="2026-10-12",
    technical_revision="batch 1 dan anchor sebelum/sesudah; perbaikan setelah run awal terhenti, toleransi tetap",
    inference="CI subjek bersyarat pada checkpoint, split, seed dan library intervensi; tidak mencakup training/split baru",
    masker_limit="intervensi pada representasi model; zero/donor tidak dijamin in-distribution")
(OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Maksimum model yang perlu dilatih:", len(ARMS)*len(SEEDS)*len(splits))
print("Checkpoint baru: checkpoints/e2/<training-spec-id>/model.pt")
print("Inferensi: forward batch 1, anchor sebelum/sesudah; checkpoint cocok dipakai ulang.")
print("Output:", OUT)
display(records[["idx", "record_id", "subject", "task", "fold_actual", "n_patches"]].head())""")
    md("""## 2. Model beku, masker dan kontrol

Untuk setiap fold, normalizer hanya fit pada train. Model/checkpoint disimpan
sebelum intervensi. Tidak ada training ulang untuk setiap budget/masker.

- **train_median:** patch normalized diganti nol, yang merepresentasikan median
  kanal training (bukan koordinat/sinyal mentah nol).
- **train_donor:** library lima background dari blok same-task 32 patch, dipilih
  hanya dari subjek training tanpa memakai label test. Semua enam kanal disalin
  bersama. Blok kontigu mempertahankan dependensi lokal, tetapi sambungan blok
  tetap dapat membuat artefak; rencana donor dan perubahan input dicatat.
- Ranking α tertinggi, α terendah, serta 20 ranking acak nested. Semua memakai
  jumlah patch dan library background yang sama. Tie diputus acak secara seeded.
  Kontrol acak dibagi seimbang atas lima background donor.

Intervensi mengganti input, lalu seluruh encoder/head dihitung ulang. T, grid dan
tail tidak berubah. Jumlah run mask, RMS perubahan input dan hash mask dilaporkan;
geometri mask tidak selalu sama antara ranking. Hasil tetap bersyarat pada masker.

Baseline/attention, anchor unmasked dan semua intervensi memakai batch 1 untuk
menghindari perbedaan konteks batching. Anchor diperiksa sebelum/sesudah; toleransi
tetap atol=rtol=1e-4. Inferensi bisa lebih lambat daripada batching. Perubahan ini
mengubah evaluation ID, bukan training ID: checkpoint cocok dimuat, sedangkan
cache intervensi protokol lama dihitung ulang. Semua artefak lama dipertahankan.
""")
    code("""runs, all_curves, model_rows = {}, [], []
for arm in ARMS:
    for seed in SEEDS:
        run = dict(arsitektur=arm, seed=seed, logit_oof=np.full(len(records), np.nan),
                   peta=[], fold_oof=fold_actual.copy(), config={**MODEL_CONFIG, **TRAINING_CONFIG})
        for fold, (train, test) in enumerate(splits):
            spec = fold_spec(ROOT, cache_hash, records, train, test, arm, seed, fold,
                             MODEL_CONFIG, TRAINING_CONFIG, VERSIONS)
            model_id = spec_digest(spec)
            cache_dir = ROOT / "results/e2_cache" / evaluation_id / model_id
            fold_cache = cache_dir / "fold.pkl"
            cache_metadata = cache_dir / "fold_manifest.json"
            checkpoint = ROOT / "checkpoints/e2" / model_id / "model.pt"
            metadata = checkpoint.parent / "metadata.json"
            if fold_cache.exists():
                if not cache_metadata.is_file():
                    raise AlignmentError("cache evaluasi parsial tanpa manifest; periksa sebelum melanjutkan")
                cache_info = json.loads(cache_metadata.read_text())
                if (cache_info["evaluation_id"] != evaluation_id or cache_info["model_id"] != model_id
                        or cache_info["fold_cache_sha256"] != file_sha256(fold_cache)):
                    raise AlignmentError("identitas/hash cache evaluasi tidak cocok")
                payload = pickle.loads(fold_cache.read_bytes())
                if payload["evaluation_spec_json"] != json.dumps(EVAL_SPEC, sort_keys=True):
                    raise AlignmentError("spec cache evaluasi tidak cocok")
                if not checkpoint.is_file() or not metadata.is_file():
                    raise AlignmentError("cache evaluasi ada tetapi checkpoint/metadata hilang")
                meta = json.loads(metadata.read_text())
                checksum = file_sha256(checkpoint)
                if meta["spec"] != spec or meta["checkpoint_sha256"] != checksum or payload["checkpoint_sha256"] != checksum:
                    raise AlignmentError("checkpoint cache evaluasi tidak cocok")
                trained_now = False
                print(arm, seed, fold, "memakai cache evaluasi dan checkpoint terverifikasi")
            else:
                print(arm, seed, fold, "memuat/melatih checkpoint lalu mengevaluasi input")
                m, norm, info = get_fold_model(ROOT, spec, recordings, DEVICE, ALLOW_TRAINING_IF_MISSING)
                trained_now = info["trained_now"]
                donor_pool = {}
                for idx in train:
                    row = records.iloc[idx]
                    donor_pool.setdefault(int(row.task), []).append(
                        (int(idx), row.subject, norm.transform(recordings[idx].kanal)))
                maps, logits, fold_curves, plans = [], [], [], []
                for idx in test:
                    row = records.iloc[idx]
                    x = norm.transform(recordings[idx].kanal)
                    predictor = FrozenPredictor(m, DEVICE, len(x))
                    logit, alpha = predictor.original(x)
                    try:
                        curves, donor_plan = evaluate_record(predictor, x, alpha, logit,
                            donor_pool[int(row.task)], record_id=row.record_id,
                            random_seed=INTERVENTION_SEED, patch=PATCH, fractions=FRACTIONS,
                            random_repeats=RANDOM_REPEATS, donor_repeats=DONOR_REPEATS,
                            block_patches=DONOR_BLOCK_PATCHES, batch_size=PERTURBATION_BATCH)
                    except AlignmentError as error:
                        raise AlignmentError(
                            f"architecture={arm}, seed={seed}, fold={fold}, idx={int(idx)}, "
                            f"model_id={model_id}: {error}") from error
                    for key, value in dict(architecture=arm, seed=seed, fold=fold, model_id=model_id,
                        idx=int(idx), subject=row.subject, label=int(row.label), record_id=row.record_id).items():
                        curves[key] = value
                    fold_curves.append(curves)
                    maps.append(dict(idx=int(idx), subjek=row.subject, label=int(row.label), alpha=alpha,
                        logit=logit, model_id=model_id, seed=seed, arsitektur=arm,
                        fold=fold, tugas=int(row.task), fs=FS, patch=PATCH))
                    logits.append(logit)
                    plans.append(dict(idx=int(idx), record_id=row.record_id, donor_library=donor_plan))
                payload = dict(evaluation_spec_json=json.dumps(EVAL_SPEC, sort_keys=True),
                    checkpoint_sha256=info["checkpoint_sha256"], test_indices=list(map(int,test)),
                    logits=logits, maps=maps, curves=pd.concat(fold_curves, ignore_index=True),
                    donor_plans=plans)
                cache_dir.mkdir(parents=True, exist_ok=True)
                simpan_atomik(fold_cache, payload)
                temporary_manifest = cache_dir / "fold_manifest.json.tmp"
                temporary_manifest.write_text(json.dumps(dict(evaluation_id=evaluation_id,model_id=model_id,
                    fold_cache_sha256=file_sha256(fold_cache)),indent=2),encoding="utf-8")
                temporary_manifest.replace(cache_metadata)
                del predictor, m, norm, donor_pool
                torch.cuda.empty_cache()
            if payload["test_indices"] != list(map(int,test)):
                raise AlignmentError("test indices cache berbeda")
            run["logit_oof"][test] = payload["logits"]
            run["peta"].extend(payload["maps"])
            all_curves.append(payload["curves"])
            model_rows.append(dict(architecture=arm, seed=seed, fold=fold, model_id=model_id,
                checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=payload["checkpoint_sha256"],
                trained_now=trained_now, evaluation_cache=str(fold_cache.relative_to(ROOT))))
        if not np.isfinite(run["logit_oof"]).all() or {p["idx"] for p in run["peta"]} != set(records.idx):
            raise AlignmentError("OOF/peta model baru tidak lengkap")
        runs[(arm,seed)] = run
new_artifact = list(runs.values())
simpan_atomik(OUT / "joint_oof_maps.pkl", new_artifact)
model_inventory = pd.DataFrame(model_rows)
model_inventory.to_csv(OUT / "model_inventory.csv", index=False)
curves = pd.concat(all_curves, ignore_index=True)
curves.to_csv(OUT / "e2_curves_raw.csv", index=False)
display(model_inventory)
print("Model baru dilatih pada run ini:", int(model_inventory.trained_now.sum()))""")
    md("""## 3. E1 pada model baru, identitas fold/checkpoint aktual

E1 dihitung ulang, tidak diwarisi dari sepuluh seed lama. AUROC baru hanya diagnostik
OOF model ini. Model ID E1/E2 dan logit asli diperiksa sebelum keduanya digabung.

JSD adalah divergence base-2: `0.5*KL2(alpha_a || m) + 0.5*KL2(alpha_b || m)`.
Alpha nonnegatif memiliki massa 1; normalisasi hanya mengoreksi pembulatan floating
point. SciPy mengembalikan distance sehingga kode menguadratkannya. Rentang JSD
[0,1]; nilai 0,72 bukan persentase kegagalan atau ukuran faithfulness.
""")
    code("""prepared_records = records.copy()
# Nama medan kompatibilitas helper E1; di ekspor diganti fold_actual.
prepared_records["fold_reconstructed"] = fold_actual
prepared_runs = {(arm,seed): dict(logits=run["logit_oof"],
    maps={p["idx"]:p["alpha"] for p in run["peta"]}) for (arm,seed),run in runs.items()}
prepared = PreparedCohort(prepared_records, prepared_runs, ARMS, SEEDS, pd.DataFrame())
scores_new = subject_scores(prepared)
scores_new.per_seed.to_csv(OUT / "new_auroc_per_seed.csv", index=False)
scores_new.per_subject_seed.to_csv(OUT / "new_probability_subject_seed.csv", index=False)
pairs = e1_pairs(prepared).rename(columns={"fold_reconstructed":"fold_actual"})
lookup = pd.DataFrame([dict(architecture=arm, seed=seed, idx=p["idx"], model_id=p["model_id"])
    for (arm,seed),run in runs.items() for p in run["peta"]])
for suffix in ("a","b"):
    pairs = pairs.merge(lookup.rename(columns={"seed":f"seed_{suffix}","model_id":f"e1_model_id_{suffix}"}),
                        on=["architecture",f"seed_{suffix}","idx"], how="left", validate="many_to_one")
record_e1, subject_e1 = e1_subject_summary(pairs)
e1_ci = bootstrap_subject_table(subject_e1,E1_METRICS,n_boot=N_BOOT,random_seed=BOOT_SEED,confidence=CONFIDENCE)
sensitivity_e1 = e1_sensitivity(pairs, EPS_PROB, EPS_LOGIT)
sensitivity_ci = bootstrap_subject_table(sensitivity_e1,
    E1_METRICS+("fraction_pairs_eligible","fraction_records_eligible"),
    group_columns=("architecture","scenario","eps_prob","eps_logit","require_same_class"),
    n_boot=N_BOOT,random_seed=BOOT_SEED,confidence=CONFIDENCE)
for name,data in (("e1_pairs_new.csv",pairs),("e1_subject_new.csv",subject_e1),
                  ("e1_ci_new.csv",e1_ci),("e1_sensitivity_new.csv",sensitivity_ci)):
    data.to_csv(OUT / name,index=False)
display(scores_new.per_seed)
display(e1_ci)
display(sensitivity_ci[sensitivity_ci.metric.isin(["js_divergence","fraction_pairs_eligible"])])""")
    md("""## 4. E2: contrast dalam rekaman/model yang sama dan CI subjek

AOPC adalah integral trapezoidal perubahan output atas fraksi patch yang dihapus,
dibagi fraksi maksimum, dengan anchor (0,0). Ini bukan AUROC klasifikasi.
Repeat kontrol/background dirata-ratakan dahulu, lalu score per seed/rekaman,
mean seed per rekaman, dan mean rekaman per subjek. Bootstrap 5.000 subjek berstrata
kelas memakai draw yang sama untuk semua model/masker. Seed dan random repeats
bukan subjek independen.

Primary: `adv_abs_delta_logit = AOPC_top - AOPC_random`. CI sepenuhnya positif
mendukung ranking attention sebagai penunjuk input yang memengaruhi model relatif
terhadap acak pada masker tersebut. CI mencakup nol belum menentukan perbedaan.
Signed margin memakai kelas prediksi asli q: `(2q-1)*(logit_asli-logit_masked)`;
negatif berarti penghapusan memperkuat kelas asli, bukan tidak ada pengaruh input.
Konsistensi antara kedua masker memperkuat interpretasi; masker yang berbeda arah
memerlukan batas klaim, bukan pemilihan masker yang paling menguntungkan.
""")
    code("""curve_mean, record_e2 = e2_record_scores(curves)
subject_e2 = e2_subject_scores(record_e2)
e2_ci = bootstrap_subject_table(subject_e2,E2_METRICS,group_columns=("architecture","masker"),
                                n_boot=N_BOOT,random_seed=BOOT_SEED,confidence=CONFIDENCE)
E2_INFERENCE = "CI subjek bersyarat pada checkpoint/split/seed/library intervensi; respons terhadap masker, bukan validasi fisiologis"
e2_ci["inference"] = E2_INFERENCE
for name,data in (("e2_curve_mean.csv",curve_mean),("e2_record_scores.csv",record_e2),
                  ("e2_subject_scores.csv",subject_e2),("e2_ci.csv",e2_ci)):
    data.to_csv(OUT / name,index=False)
display(e2_ci[e2_ci.metric.isin(["adv_abs_delta_logit","adv_signed_margin_drop",
                               "top_minus_bottom_abs_delta_logit"])])
# Diagnostik geometri dan perubahan input: deskriptif, tidak membuktikan masker in-distribution.
diagnostics = curves.groupby(["architecture","masker","ranking","requested_fraction"])[
    ["n_mask_runs","input_rms_change","boundary_ties"]].mean().reset_index()
diagnostics.to_csv(OUT / "mask_diagnostics.csv",index=False)
display(diagnostics)""")
    md("""## 5. Penghubung E1–E2 pada checkpoint identik

Tabel gabungan menyimpan model ID dan logit yang cocok untuk kedua seed dalam
setiap pasangan. Besarnya perbedaan peta tidak dijadikan label faithful/tidak
faithful. Tampilkan score E2 setiap model, selisih score, dan mean contrast pada
subset output stabil beserta cakupan subjek. Score positif pada satu rekaman
bukan keberhasilan inferensial yang telah dibuktikan untuk rekaman tersebut.
""")
    code("""joint = join_e1_e2(pairs,record_e2)
joint_subject = joint_subject_sensitivity(joint,EPS_PROB,EPS_LOGIT)
joint_ci = bootstrap_subject_table(joint_subject,JOINT_METRICS+("fraction_pairs_eligible",),
    group_columns=("architecture","masker","scenario","eps_prob","eps_logit","require_same_class"),
    n_boot=N_BOOT,random_seed=BOOT_SEED,confidence=CONFIDENCE)
joint_ci["inference"] = E2_INFERENCE
joint.to_csv(OUT / "joint_e1_e2_pairs.csv",index=False)
joint_subject.to_csv(OUT / "joint_e1_e2_subject.csv",index=False)
joint_ci.to_csv(OUT / "joint_e1_e2_ci.csv",index=False)
display(joint_ci[joint_ci.metric.isin(["mean_influence_adv","mean_support_adv","fraction_pairs_eligible"]) &
    ((joint_ci.scenario=="all_pairs") | ((joint_ci.eps_prob==.02) & (joint_ci.eps_logit==.1)))])""")
    md("""## 6. Grafik yang dihasilkan

Kurva: random/background repeats → seed → rekaman → subjek, lalu mean subjek.
Grafik contrast menampilkan CI subjek. Grafik gabungan tidak mengubah perbedaan
attention menjadi penilaian kualitas klinis.
""")
    code("""def savefig(fig,name):
    fig.savefig(FIG / (name+".png"),dpi=180,bbox_inches="tight")
    fig.savefig(FIG / (name+".pdf"),bbox_inches="tight")
labels={"gru":"BiGRU","mamba2":"BiMamba-2"}
styles={"alpha_top":("#eb6834","o"),"alpha_bottom":("#8a8a85","s"),"random":("#2a78d6","^")}
subject_curve = curve_mean.groupby(["architecture","masker","ranking","subject","label","idx","requested_fraction"])[
    ["abs_delta_logit","signed_margin_drop"]].mean().reset_index()
subject_curve = subject_curve.groupby(["architecture","masker","ranking","subject","label","requested_fraction"])[
    ["abs_delta_logit","signed_margin_drop"]].mean().reset_index()
subject_curve.to_csv(OUT / "e2_subject_curves.csv",index=False)
population_curve = subject_curve.groupby(["architecture","masker","ranking","requested_fraction"])[
    ["abs_delta_logit","signed_margin_drop"]].mean().reset_index()
population_curve.to_csv(OUT / "e2_population_curves.csv",index=False)
for metric in ("abs_delta_logit","signed_margin_drop"):
    fig,axes=plt.subplots(2,2,figsize=(10.6,7),squeeze=False)
    for i,arm in enumerate(ARMS):
        for j,masker in enumerate(("train_median","train_donor")):
            ax=axes[i,j]
            for ranking,(color,marker) in styles.items():
                d=population_curve[(population_curve.architecture==arm)&(population_curve.masker==masker)&
                                   (population_curve.ranking==ranking)].sort_values("requested_fraction")
                ax.plot(np.r_[0,d.requested_fraction],np.r_[0,d[metric]],color=color,marker=marker,label=ranking)
            ax.set_title(labels[arm]+" — "+masker,loc="left")
            ax.set_xlabel("budget fraksi patch (aktual dibulatkan ceil)"); ax.set_ylabel(metric)
            ax.axhline(0,color="#8a8a85",ls=":"); ax.legend(fontsize=8); rapikan(ax)
    fig.tight_layout(); savefig(fig,"e2_curves_"+metric); plt.show()
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,metric in zip(axes,("adv_abs_delta_logit","adv_signed_margin_drop")):
    selected=e2_ci[e2_ci.metric==metric].reset_index(drop=True)
    for i,row in selected.iterrows():
        ax.hlines(i,row.ci_low,row.ci_high,color="#2a78d6",lw=2)
        ax.plot(row.estimate,i,"o",color="#2a78d6")
    ax.set_yticks(range(len(selected)),[labels[r.architecture]+" / "+r.masker for r in selected.itertuples()])
    ax.axvline(0,color="#8a8a85",ls=":");ax.set_xlabel(metric);rapikan(ax)
fig.suptitle("E2 — top attention minus random, CI subjek bersyarat")
fig.tight_layout();savefig(fig,"e2_contrast_ci");plt.show()
fig,axes=plt.subplots(1,2,figsize=(10.8,4))
for ax,arm in zip(axes,ARMS):
    d=joint[(joint.architecture==arm)&(joint.masker=="train_median")]
    ax.scatter(d.js_divergence,d.abs_delta_relative_gain,s=8,alpha=.15,rasterized=True)
    ax.set_xlabel("JSD attention antar-seed");ax.set_ylabel("|Delta gain relatif top-random|")
    ax.set_title(labels[arm],loc="left");rapikan(ax)
fig.suptitle("E1–E2 pada model identik; pasangan seed bukan observasi independen")
fig.tight_layout();savefig(fig,"joint_e1_e2_continuous");plt.show()""")
    md("""## 7. Penyelesaian dan batas klaim

E2 dapat mendukung hasil positif, negatif, atau belum menentukan. Jika kedua peta
berbeda tetapi sama-sama mengidentifikasi input berpengaruh, variasi peta dapat
tetap kompatibel dengan faithfulness. Jika contrast attention terhadap acak rendah
atau berubah menurut masker, batasi klaim pada pengujian dan masker tersebut.
Signed support negatif tidak sama dengan tidak berpengaruh.

Jangan menyebut hasil sebagai validasi fisiologis, atau menjadikan CI yang memuat
nol bukti ekuivalensi. Ketergantungan fold dan training tidak dicakup oleh bootstrap
OOF tetap. E3 diperlukan untuk klaim perturbasi input kecil; notebook ini tidak
menguji E3. Hasil dan checkpoint baru tidak mengganti bukti/angka lama.
""")
    code("""inventory["sha256_after"]=[file_sha256(ROOT / p) for p in inventory.path]
inventory["unchanged"]=inventory.sha256_before==inventory.sha256_after
inventory.to_csv(OUT / "input_inventory.csv",index=False)
if not inventory.unchanged.all():
    manifest["completion_status"]="invalid_inputs_changed"
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    raise RuntimeError("input/sumber berubah selama run; jangan gunakan hasil ini")
manifest.update(completion_status="complete",completed_utc=datetime.now(timezone.utc).isoformat(),
    training_performed=bool(model_inventory.trained_now.any()),
    models_trained_now=int(model_inventory.trained_now.sum()),
    tables=sorted(p.name for p in OUT.glob("*.csv")),figures=sorted(p.name for p in FIG.iterdir()))
(OUT / "analysis_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print("Selesai:",OUT)
print("Checkpoint:",ROOT / "checkpoints/e2")
print("Analisis tambahan eksploratori; E1/E2 model baru, tanpa perubahan preregistrasi/manuskrip lama.")""")
    nb=nbf.v4.new_notebook(cells=cells)
    nb.metadata={"kernelspec":{"display_name":"Skripsi (.venv)","language":"python","name":"skripsi"},
                 "language_info":{"name":"python","version":"3.11.8"},
                 "analysis_status":"tambahan eksploratori E2 dan E1 model baru"}
    return nb


def main():
    nb=build_notebook()
    nbf.validate(nb)
    output=Path(__file__).resolve().parent.parent / "notebooks/16_audit_e2.ipynb"
    nbf.write(nb,output)
    print(f"ditulis: {output} ({len(nb.cells)} sel; tidak dieksekusi)")


if __name__=="__main__": main()
