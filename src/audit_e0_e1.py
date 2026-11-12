"""Analisis tambahan E0/E1 dari artefak OOF; tanpa model atau training.

Modul ini dipanggil notebook 15 yang dibangun oleh scripts/build_nb_15.py.
Tidak ada analisis yang berjalan saat modul diimpor. Metadata historis yang
tidak tersimpan tidak boleh dipulihkan sebagai fakta dari kode saat ini.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


STATUS_ANALISIS = "tambahan_eksploratori_setelah_hasil_S3_S5_diketahui"
BATAS_INFERENSI = (
    "Bersyarat pada artefak model OOF, split, konfigurasi dan seed yang tersedia; "
    "tidak mencakup training ulang, split baru, kohort baru, atau faithfulness."
)


class AlignmentError(ValueError):
    """Ketidaksesuaian yang harus menghentikan analisis, bukan dipotong diam-diam."""


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise AlignmentError(f"{name} harus integer eksplisit, ditemukan {value!r}")
    return int(value)


def _vector(value, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise AlignmentError(f"{name}: bentuk harus ({length},) dan seluruh nilai finite")
    return result


def cache_manifest(path: Path, cohort: str, patch_size: int, fs: float) -> pd.DataFrame:
    """Baca cache lokal; hanya metadata aman dan kanal model, tanpa data mentah.

    Digest kanal dan identitas adalah snapshot SAAT AUDIT, bukan bukti identitas
    cache saat training lama. Identitas NewHandPD berasal dari (subjek, task).
    """
    if cohort not in {"uci395", "newhandpd"}:
        raise ValueError(f"kohort tidak dikenal: {cohort}")
    if patch_size < 1 or fs <= 0:
        raise ValueError("patch_size dan fs harus positif")
    rows = []
    # Format proyek menyimpan meta sebagai object array; hanya cache lokal tepercaya.
    with np.load(path, allow_pickle=True) as archive:
        meta = archive["meta"]
        width = 5 if cohort == "uci395" else 4
        if meta.ndim != 2 or meta.shape[1] != width or len(meta) == 0:
            raise AlignmentError(f"{path.name}: bentuk meta tidak sesuai skema {cohort}")
        for idx, item in enumerate(meta):
            subject, group, task = str(item[0]), str(item[1]), int(item[2])
            segment = int(item[3]) if cohort == "uci395" else 0
            sample_rate = float(item[-1])
            if not subject or group not in {"PD", "HC"} or not np.isclose(sample_rate, fs):
                raise AlignmentError(f"{cohort} idx={idx}: subjek/kelompok/fs tidak sesuai")
            channel = np.asarray(archive[f"kanal_{idx}"])
            if channel.ndim != 2 or channel.shape[1] != 6 or not np.isfinite(channel).all():
                raise AlignmentError(f"{cohort} idx={idx}: kanal harus finite dengan bentuk (T,6)")
            n_samples = len(channel)
            n_patches = n_samples // patch_size
            if n_patches < 3:
                raise AlignmentError(f"{cohort} idx={idx}: kurang dari tiga patch valid")
            digest = sha256()
            digest.update(str(channel.dtype).encode())
            digest.update(str(channel.shape).encode())
            digest.update(np.ascontiguousarray(channel).tobytes())
            rows.append(dict(
                cohort=cohort, idx=idx, subject=subject, label=int(group == "PD"),
                task=task, segment=segment, fs=sample_rate, n_samples=n_samples,
                patch_size=patch_size, patch_seconds=patch_size / sample_rate,
                n_patches=n_patches, tail_samples=n_samples % patch_size,
                record_id=f"{cohort}:{subject}:task{task}:seg{segment}",
                channel_sha256_current=digest.hexdigest(),
            ))
    records = pd.DataFrame(rows)
    if records.record_id.duplicated().any():
        raise AlignmentError(f"{cohort}: identitas (subjek, task, segmen) tidak unik")
    if (records.groupby("subject").label.nunique() != 1).any():
        raise AlignmentError(f"{cohort}: label berbeda pada subjek yang sama")
    if set(records.label) != {0, 1}:
        raise AlignmentError(f"{cohort}: kedua kelas harus tersedia")
    return records


def reconstructed_folds(records: pd.DataFrame, n_splits=5, random_state=42) -> np.ndarray:
    """Rekonstruksi protokol saat ini; bukan fold historis yang terverifikasi."""
    result = np.full(len(records), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train, test) in enumerate(splitter.split(
        np.zeros(len(records)), records.label.to_numpy(), records.subject.to_numpy()
    )):
        if set(records.subject.iloc[train]) & set(records.subject.iloc[test]):
            raise AlignmentError("rekonstruksi fold membocorkan subjek")
        result[test] = fold
    if (result < 0).any():
        raise AlignmentError("rekonstruksi fold tidak mencakup seluruh rekaman")
    return result


@dataclass
class PreparedCohort:
    records: pd.DataFrame
    runs: dict
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    checks: pd.DataFrame


def prepare_cohort(
    artifact_path: Path, records: pd.DataFrame, arms: tuple[str, ...],
    seeds: tuple[int, ...], expected_config: dict, require_maps: bool = False,
) -> PreparedCohort:
    """Validasi kunci run, OOF, peta, dan batas metadata sebelum statistik.

    Tidak mengasumsikan seed=0 bila medan seed hilang. Tidak mengambil irisan
    seed/rekaman atau melakukan crop/interpolasi ketika artefak tidak selaras.
    """
    if len(set(arms)) != len(arms) or len(arms) < 2:
        raise ValueError("gunakan sekurang-kurangnya dua arm unik")
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("gunakan sekurang-kurangnya dua seed unik")
    with Path(artifact_path).open("rb") as stream:
        entries = pickle.load(stream)  # hanya artefak lokal proyek tepercaya
    if not isinstance(entries, list):
        raise AlignmentError(f"{artifact_path.name}: diharapkan list run")
    cohort = records.cohort.iloc[0]
    checks, runs = [], {}
    folds = reconstructed_folds(records)
    records = records.copy()
    records["fold_reconstructed"] = folds

    def note(arm, seed, field, status, detail):
        checks.append(dict(cohort=cohort, architecture=arm, seed=seed,
                           field=field, status=status, detail=detail))

    for entry in entries:
        if not isinstance(entry, dict) or "arsitektur" not in entry:
            raise AlignmentError("run tidak memiliki medan arsitektur")
        arm = entry["arsitektur"]
        if arm not in arms:
            continue
        if "seed" not in entry:
            raise AlignmentError(f"{cohort}/{arm}: seed tidak tersimpan; jangan infer seed=0")
        seed = _integer(entry["seed"], "seed")
        if seed not in seeds:
            note(arm, seed, "seleksi_seed", "tidak_dipakai", "di luar daftar seed audit eksplisit")
            continue
        key = (arm, seed)
        if key in runs:
            raise AlignmentError(f"run duplikat: {cohort}/{key}")
        logits = _vector(entry.get("logit_oof"), len(records), f"{cohort}/{key}/logit_oof")
        maps = {}
        if require_maps:
            if not isinstance(entry.get("peta"), list):
                raise AlignmentError(f"{cohort}/{key}: peta tidak tersedia dalam run yang sama")
            for item in entry["peta"]:
                if not isinstance(item, dict):
                    raise AlignmentError(f"{cohort}/{key}: entri peta bukan dict")
                idx = _integer(item.get("idx"), "idx peta")
                if idx < 0 or idx >= len(records) or idx in maps:
                    raise AlignmentError(f"{cohort}/{key}: idx di luar rentang atau duplikat {idx}")
                record = records.iloc[idx]
                if str(item.get("subjek")) != record.subject or item.get("label") != record.label:
                    raise AlignmentError(f"{cohort}/{key}/idx={idx}: subjek atau label berbeda dari cache")
                if "seed" in item and _integer(item["seed"], "seed peta") != seed:
                    raise AlignmentError(f"{cohort}/{key}/idx={idx}: seed peta berbeda dari run")
                if "arsitektur" in item and item["arsitektur"] != arm:
                    raise AlignmentError(f"{cohort}/{key}/idx={idx}: arsitektur peta berbeda dari run")
                alpha = _vector(item.get("alpha"), int(record.n_patches), f"alpha idx={idx}")
                if (alpha < 0).any() or not np.isclose(alpha.sum(), 1, atol=1e-4, rtol=0):
                    raise AlignmentError(f"{cohort}/{key}/idx={idx}: alpha negatif atau massanya bukan 1")
                # Phi/marker tidak digunakan untuk kualitas penjelasan dalam E1.
                for field in ("phi", "penanda"):
                    if field in item and np.asarray(item[field]).shape != alpha.shape:
                        raise AlignmentError(f"{cohort}/{key}/idx={idx}: grid {field} berbeda dari alpha")
                for field, expected in (("tugas", record.task), ("fs", record.fs),
                                        ("patch", record.patch_size)):
                    if field in item and item[field] != expected:
                        raise AlignmentError(f"{cohort}/{key}/idx={idx}: metadata {field} tidak cocok")
                if "fold" in item and _integer(item["fold"], "fold peta") != folds[idx]:
                    raise AlignmentError(f"{cohort}/{key}/idx={idx}: fold tersimpan tidak cocok rekonstruksi")
                if "logit" in item and not np.isclose(item["logit"], logits[idx], atol=1e-7, rtol=1e-7):
                    raise AlignmentError(f"{cohort}/{key}/idx={idx}: logit peta dan OOF berbeda")
                maps[idx] = alpha.copy()
            if set(maps) != set(records.idx):
                raise AlignmentError(f"{cohort}/{key}: cakupan peta harus sama dengan seluruh rekaman cache")
            note(arm, seed, "idx_subjek_label_grid", "terverifikasi",
                 "cakupan penuh; grid floor(T/P); alpha finite, nonnegatif dan massa 1")
            note(arm, seed, "prediksi_peta_satu_run", "tersimpan_bersama",
                 "logit_oof dan peta berada dalam satu entri; bobot/model_id historis tidak tersedia")
        saved_folds = entry.get("fold_oof")
        if saved_folds is not None:
            saved_folds = np.asarray(saved_folds)
            if saved_folds.shape != folds.shape or not np.array_equal(saved_folds, folds):
                raise AlignmentError(f"{cohort}/{key}: fold_oof tidak sesuai rekonstruksi")
            note(arm, seed, "fold", "tersimpan_dan_cocok", "dibandingkan dengan protokol saat audit")
        else:
            note(arm, seed, "fold", "rekonstruksi_bukan_bukti_historis",
                 "SGKF5 shuffle=True random_state=42; tergantung urutan cache dan versi sklearn saat audit")
        saved_config = entry.get("config")
        if saved_config is not None and not isinstance(saved_config, dict):
            raise AlignmentError(f"{cohort}/{key}: config harus dict")
        for field, expected in expected_config.items():
            if saved_config is not None and field in saved_config:
                if saved_config[field] != expected:
                    raise AlignmentError(f"{cohort}/{key}: config {field} tidak sesuai {expected!r}")
                note(arm, seed, f"config.{field}", "tersimpan_dan_cocok", str(expected))
            else:
                note(arm, seed, f"config.{field}", "tidak_tersimpan",
                     f"ekspektasi dari kode saat audit: {expected!r}; bukan verifikasi training lama")
        note(arm, seed, "seed_logit", "terverifikasi", "seed eksplisit; logit OOF finite dan cakupan penuh")
        note(arm, seed, "model_id_cache_training", "tidak_tersimpan",
             "checkpoint/model_id, hash cache saat training dan hash kode saat training tidak dipulihkan")
        runs[key] = dict(logits=logits.copy(), maps=maps)
    expected_keys = {(arm, seed) for arm in arms for seed in seeds}
    if set(runs) != expected_keys:
        raise AlignmentError(f"{cohort}: run hilang {sorted(expected_keys - set(runs))}")
    return PreparedCohort(records, runs, arms, seeds, pd.DataFrame(checks))


@dataclass
class SubjectScores:
    cohort: str
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    subjects: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray  # (arm, seed, subjek), setelah mean probabilitas rekaman
    per_seed: pd.DataFrame
    per_subject_seed: pd.DataFrame


def subject_scores(prepared: PreparedCohort) -> SubjectScores:
    records = prepared.records
    subjects = np.array(sorted(records.subject.unique()))
    indices = [np.flatnonzero(records.subject.to_numpy() == subject) for subject in subjects]
    labels = np.array([int(records.label.iloc[index[0]]) for index in indices])
    values = np.empty((len(prepared.arms), len(prepared.seeds), len(subjects)))
    rows, long_rows = [], []
    for a, arm in enumerate(prepared.arms):
        for s, seed in enumerate(prepared.seeds):
            prob_record = expit(prepared.runs[(arm, seed)]["logits"])
            prob_subject = np.array([prob_record[index].mean() for index in indices])
            values[a, s] = prob_subject
            rows.append(dict(cohort=records.cohort.iloc[0], architecture=arm, seed=seed,
                             auroc=float(roc_auc_score(labels, prob_subject)), n_subjects=len(subjects)))
            for subject, label, probability, index in zip(subjects, labels, prob_subject, indices):
                long_rows.append(dict(cohort=records.cohort.iloc[0], architecture=arm, seed=seed,
                                      subject=subject, label=label, probability=probability,
                                      n_records=len(index)))
    return SubjectScores(records.cohort.iloc[0], prepared.arms, prepared.seeds, subjects,
                         labels, values, pd.DataFrame(rows), pd.DataFrame(long_rows))


def reference_check(scores: SubjectScores, reference_path: Path, headlines: dict,
                    tolerance: float = 1e-8) -> pd.DataFrame:
    """Hentikan bootstrap jika angka per-seed tidak cocok; toleransi headline 4 desimal."""
    reference = pd.read_csv(reference_path).rename(columns={"arsitektur": "architecture", "auc": "auroc"})
    reference = reference[reference.architecture.isin(scores.arms) & reference.seed.isin(scores.seeds)]
    if reference.duplicated(["architecture", "seed"]).any():
        raise AlignmentError(f"{reference_path.name}: kunci referensi duplikat")
    expected = scores.per_seed.merge(reference[["architecture", "seed", "auroc"]],
                                    on=["architecture", "seed"], how="left", validate="one_to_one",
                                    suffixes=("_audit", "_reference"))
    if expected.auroc_reference.isna().any():
        raise AlignmentError(f"{reference_path.name}: referensi per-seed tidak lengkap")
    expected["abs_difference"] = abs(expected.auroc_audit - expected.auroc_reference)
    expected["tolerance"] = tolerance
    expected["reference_kind"] = "auroc_per_seed"
    extra = []
    for arm in scores.arms:
        observed = float(scores.per_seed.loc[scores.per_seed.architecture == arm, "auroc"].mean())
        reference_value = float(headlines[arm])
        extra.append(dict(cohort=scores.cohort, architecture=arm, seed=np.nan,
                          auroc_audit=observed, auroc_reference=reference_value,
                          abs_difference=abs(observed - reference_value), tolerance=5.01e-5,
                          reference_kind="headline_rounded_4dp"))
    result = pd.concat([expected, pd.DataFrame(extra)], ignore_index=True)
    result["matches"] = result.abs_difference <= result.tolerance
    if not result.matches.all():
        failed = result.loc[~result.matches, ["architecture", "seed", "reference_kind", "abs_difference"]]
        raise AlignmentError(f"{scores.cohort}: estimator tidak mereproduksi referensi:\n{failed.to_string(index=False)}")
    return result


def _stratified_counts(labels: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(labels)
    if set(labels) != {0, 1} or n_boot < 1:
        raise ValueError("bootstrap membutuhkan dua kelas dan n_boot positif")
    counts = np.zeros((n_boot, len(labels)), dtype=int)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        counts[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), n_boot)
    return counts


def mean_seed_auc_kernel(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Kernel Mann–Whitney rata-rata antar-seed; BUKAN AUROC probabilitas ensemble.

    AUROC = mean pasangan PD-HC [I(p_PD>p_HC)+0.5 I(tie)]. Linearitas
    rata-rata memungkinkan bootstrap mean-AUROC tanpa 200.000 pemanggilan sklearn.
    Penghitungan tetap mempertahankan skor dan pasangan keluaran seluruh arm/seed.
    """
    probabilities, labels = np.asarray(probabilities, dtype=float), np.asarray(labels)
    if (probabilities.ndim != 3 or probabilities.shape[-1] != len(labels)
            or probabilities.shape[1] < 1 or set(labels) != {0, 1}
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()):
        raise AlignmentError("kernel membutuhkan probabilitas (arm, seed, subjek) dan dua kelas")
    positive = probabilities[:, :, labels == 1]
    negative = probabilities[:, :, labels == 0]
    comparison = (positive[..., :, None] > negative[..., None, :]).astype(float)
    comparison += 0.5 * (positive[..., :, None] == negative[..., None, :])
    return comparison.mean(axis=1)


def _interval(values: np.ndarray, confidence: float) -> tuple[float, float, int]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan, 0
    q = (1 - confidence) / 2
    lo, hi = np.quantile(finite, [q, 1 - q])
    return float(lo), float(hi), len(finite)


def e0_bootstrap(cohorts: dict[str, SubjectScores], n_boot=5000, random_seed=1500,
                 confidence=0.95, arm_a="mamba2", arm_b="gru") -> dict[str, pd.DataFrame]:
    """Bootstrap subjek berstrata kelas, independen antar-kohort, paired antar-arm.

    Seluruh rekaman subjek sudah diagregasi menjadi skor subjek PER SEED.
    Multiplicity bootstrap yang sama diterapkan pada setiap model/seed dalam
    satu kohort. Model/seed tidak di-bootstrap atau dilatih ulang.
    """
    if set(cohorts) != {"uci395", "newhandpd"} or not 0 < confidence < 1:
        raise ValueError("dibutuhkan UCI 395/NewHandPD dan confidence dalam (0,1)")
    first = cohorts["uci395"]
    if arm_a == arm_b or arm_a not in first.arms or arm_b not in first.arms:
        raise ValueError("contrast membutuhkan dua arm berbeda yang tersedia")
    rng = np.random.default_rng(random_seed)
    distributions, points, cohort_rows = {}, {}, []
    draws = pd.DataFrame({"bootstrap_id": np.arange(n_boot)})
    for name in ("uci395", "newhandpd"):
        scores = cohorts[name]
        if scores.arms != first.arms or scores.seeds != first.seeds:
            raise AlignmentError("arm dan seed E0 harus identik pada kedua kohort")
        kernel = mean_seed_auc_kernel(scores.probabilities, scores.labels)
        counts = _stratified_counts(scores.labels, n_boot, rng)
        w_positive, w_negative = counts[:, scores.labels == 1], counts[:, scores.labels == 0]
        distribution = np.einsum("bp,apn,bn->ba", w_positive, kernel, w_negative,
                                 optimize=True) / (w_positive.sum(axis=1) * w_negative.sum(axis=1))[:, None]
        point = kernel.mean(axis=(1, 2))
        distributions[name], points[name] = distribution, point
        for a, arm in enumerate(scores.arms):
            direct = scores.per_seed.loc[scores.per_seed.architecture == arm, "auroc"].mean()
            if not np.isclose(point[a], direct, atol=1e-12, rtol=0):
                raise AlignmentError("kernel AUROC tidak cocok dengan mean AUROC sklearn per seed")
            lo, hi, valid = _interval(distribution[:, a], confidence)
            cohort_rows.append(dict(cohort=name, architecture=arm, mean_auroc=point[a],
                                    ci_low=lo, ci_high=hi, n_boot_valid=valid,
                                    sd_across_seed=scores.per_seed.loc[
                                        scores.per_seed.architecture == arm, "auroc"].std(ddof=1),
                                    n_seed=len(scores.seeds), n_subjects=len(scores.subjects),
                                    n_pd=int((scores.labels == 1).sum()), n_hc=int((scores.labels == 0).sum())))
            draws[f"auroc_{name}_{arm}"] = distribution[:, a]
    delta_draw = distributions["uci395"] - distributions["newhandpd"]
    delta_point = points["uci395"] - points["newhandpd"]
    delta_rows = []
    for a, arm in enumerate(first.arms):
        lo, hi, valid = _interval(delta_draw[:, a], confidence)
        delta_rows.append(dict(architecture=arm, delta_uci_minus_new=delta_point[a],
                               ci_low=lo, ci_high=hi, n_boot_valid=valid))
        draws[f"delta_{arm}"] = delta_draw[:, a]
    a, b = first.arms.index(arm_a), first.arms.index(arm_b)
    contrast_draw = delta_draw[:, a] - delta_draw[:, b]
    lo, hi, valid = _interval(contrast_draw, confidence)
    draws[f"contrast_delta_{arm_a}_minus_{arm_b}"] = contrast_draw
    contrast = pd.DataFrame([dict(
        arm_a=arm_a, arm_b=arm_b, contrast=delta_point[a] - delta_point[b],
        ci_low=lo, ci_high=hi, contains_zero=bool(lo <= 0 <= hi), n_boot_valid=valid,
        direction="nilai negatif: penurunan AUROC arm_a lebih kecil",
        status=STATUS_ANALISIS, unit="subjek", inference=BATAS_INFERENSI,
    )])
    return dict(cohort=pd.DataFrame(cohort_rows), delta=pd.DataFrame(delta_rows),
                contrast=contrast, draws=draws)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return np.nan
    ra, rb = rankdata(a[finite]), rankdata(b[finite])
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(np.clip(ra @ rb / denominator, -1, 1)) if denominator > 0 else np.nan


def _top_weights(alpha: np.ndarray, fraction: float) -> tuple[np.ndarray, int]:
    """Tie di batas top-k dibagi fraksional, tanpa memihak indeks temporal awal."""
    if not 0 < fraction <= 1:
        raise ValueError("top fraction harus dalam (0,1]")
    k = max(1, int(np.ceil(len(alpha) * fraction)))
    cutoff = np.partition(alpha, len(alpha) - k)[len(alpha) - k]
    above, boundary = alpha > cutoff, alpha == cutoff
    weights = above.astype(float)
    weights[boundary] = (k - above.sum()) / boundary.sum()
    return weights, int(boundary.sum())


def attention_changes(alpha_a: np.ndarray, alpha_b: np.ndarray, top_fraction=0.1) -> dict:
    a, b = np.asarray(alpha_a, dtype=float), np.asarray(alpha_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 3:
        raise AlignmentError("peta harus memiliki grid identik, tanpa crop/interpolasi")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
        raise AlignmentError("alpha tidak finite/nonnegatif")
    if not np.isclose(a.sum(), 1, atol=1e-4, rtol=0) or not np.isclose(b.sum(), 1, atol=1e-4, rtol=0):
        raise AlignmentError("massa alpha bukan 1")
    # Koreksi pembulatan floating point saja, bukan normalisasi peta yang terpotong.
    a, b = a / a.sum(), b / b.sum()
    wa, ties_a = _top_weights(a, top_fraction)
    wb, ties_b = _top_weights(b, top_fraction)
    rho = _spearman(a, b)
    jsd = float(np.clip(jensenshannon(a, b, base=2) ** 2, 0, 1))
    return dict(
        rho_alpha=rho, rank_disagreement=(1 - rho) / 2 if np.isfinite(rho) else np.nan,
        js_divergence=jsd,
        top_jaccard_fractional=float(np.minimum(wa, wb).sum() / np.maximum(wa, wb).sum()),
        top_boundary_ties_a=ties_a, top_boundary_ties_b=ties_b,
        uniform_a=bool(np.ptp(a) == 0), uniform_b=bool(np.ptp(b) == 0),
    )


E1_METRICS = ("abs_delta_prob", "abs_delta_logit", "class_agreement", "rho_alpha",
              "rank_disagreement", "js_divergence", "top_jaccard_fractional")


def e1_pairs(prepared: PreparedCohort, top_fraction=0.1) -> pd.DataFrame:
    """Pasangan seed pada rekaman identik, dari logit/peta dalam satu entri S5."""
    if prepared.records.cohort.iloc[0] != "newhandpd":
        raise ValueError("E1 tahap ini hanya memakai artefak bersama NewHandPD")
    rows = []
    for arm in prepared.arms:
        for seed_a, seed_b in combinations(prepared.seeds, 2):
            run_a, run_b = prepared.runs[(arm, seed_a)], prepared.runs[(arm, seed_b)]
            for record in prepared.records.itertuples(index=False):
                idx = record.idx
                la, lb = run_a["logits"][idx], run_b["logits"][idx]
                pa, pb = float(expit(la)), float(expit(lb))
                if idx not in run_a["maps"] or idx not in run_b["maps"]:
                    raise AlignmentError(f"peta tidak lengkap untuk {arm}/idx={idx}")
                changes = attention_changes(run_a["maps"][idx], run_b["maps"][idx], top_fraction)
                rows.append(dict(
                    cohort="newhandpd", architecture=arm, idx=idx, record_id=record.record_id,
                    subject=record.subject, label=record.label, task=record.task,
                    fold_reconstructed=record.fold_reconstructed,
                    seed_a=seed_a, seed_b=seed_b, n_patches=record.n_patches,
                    patch_seconds=record.patch_seconds, probability_a=pa, probability_b=pb,
                    logit_a=la, logit_b=lb, abs_delta_prob=abs(pa - pb),
                    abs_delta_logit=abs(la - lb), class_agreement=float((la >= 0) == (lb >= 0)),
                    top_fraction=top_fraction, **changes,
                ))
    return pd.DataFrame(rows)


def e1_subject_summary(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean pasangan per rekaman, lalu mean rekaman per subjek; subjek berbobot sama."""
    keys = ["architecture", "subject", "label", "idx"]
    record = pairs.groupby(keys, sort=True)[list(E1_METRICS)].mean().reset_index()
    record["n_seed_pairs"] = pairs.groupby(keys).size().to_numpy()
    subject = record.groupby(keys[:3], sort=True)[list(E1_METRICS)].mean().reset_index()
    subject["n_records"] = record.groupby(keys[:3]).size().to_numpy()
    return record, subject


def e1_subject_relations(pairs: pd.DataFrame) -> pd.DataFrame:
    """Asosiasi kontinu antar-pasangan seed, diringkas per rekaman lalu per subjek.

    Tidak menghitung p-value dengan seed-pair sebagai observasi independen.
    """
    rows = []
    for (arm, subject, label, idx), group in pairs.groupby(["architecture", "subject", "label", "idx"]):
        rows.append(dict(
            architecture=arm, subject=subject, label=label, idx=idx,
            rho_delta_prob_vs_jsd=_spearman(group.abs_delta_prob, group.js_divergence),
            rho_delta_logit_vs_jsd=_spearman(group.abs_delta_logit, group.js_divergence),
        ))
    return pd.DataFrame(rows).groupby(["architecture", "subject", "label"])[
        ["rho_delta_prob_vs_jsd", "rho_delta_logit_vs_jsd"]
    ].mean().reset_index()


def e1_sensitivity(pairs: pd.DataFrame, eps_prob=(0.01, 0.02, 0.05),
                   eps_logit=(None, 0.1, 0.25, 0.5)) -> pd.DataFrame:
    """Ambang hanya sensitivitas; bukan batas klinis atau hipotesis preregistrasi.

    Seleksi di tingkat pasangan-seed/rekaman. Mean pasangan eligible per
    rekaman, lalu mean rekaman eligible per subjek. Cakupan dihitung terhadap
    seluruh pasangan dan rekaman; subjek tanpa pasangan eligible tetap ada (NaN).
    """
    keys = ["architecture", "subject", "label"]
    base = pairs.groupby(keys, sort=True).agg(n_pairs_total=("idx", "size"),
                                            n_records_total=("idx", "nunique")).reset_index()
    scenarios = [(None, None, False)] + [(p, l, True) for p in eps_prob for l in eps_logit]
    scenarios += [(None, l, True) for l in eps_logit if l is not None]
    results = []
    for p, l, same_class in scenarios:
        if (p is not None and p < 0) or (l is not None and l < 0):
            raise ValueError("ambang harus nonnegatif")
        mask = np.ones(len(pairs), dtype=bool)
        if p is not None:
            mask &= pairs.abs_delta_prob.to_numpy() <= p
        if l is not None:
            mask &= pairs.abs_delta_logit.to_numpy() <= l
        if same_class:
            mask &= pairs.class_agreement.to_numpy() == 1
        eligible = pairs.loc[mask]
        record_mean = eligible.groupby(keys + ["idx"])[list(E1_METRICS)].mean()
        subject_mean = record_mean.groupby(level=keys).mean().reset_index()
        counts = eligible.groupby(keys).agg(n_pairs_eligible=("idx", "size"),
                                           n_records_eligible=("idx", "nunique")).reset_index()
        result = base.merge(subject_mean, on=keys, how="left", validate="one_to_one")
        result = result.merge(counts, on=keys, how="left", validate="one_to_one")
        for name in ("n_pairs_eligible", "n_records_eligible"):
            result[name] = result[name].fillna(0).astype(int)
        result["fraction_pairs_eligible"] = result.n_pairs_eligible / result.n_pairs_total
        result["fraction_records_eligible"] = result.n_records_eligible / result.n_records_total
        result["eps_prob"] = np.nan if p is None else float(p)
        result["eps_logit"] = np.nan if l is None else float(l)
        result["require_same_class"] = same_class
        result["scenario"] = "all_pairs" if not same_class else f"p<={p};logit<={l};same_class"
        results.append(result)
    return pd.concat(results, ignore_index=True)


def bootstrap_subject_table(subjects: pd.DataFrame, metrics: tuple[str, ...],
                            group_columns=("architecture",), n_boot=5000,
                            random_seed=1501, confidence=0.95) -> pd.DataFrame:
    """CI percentile subject-stratified dengan draw yang sama untuk semua kelompok.

    Subjek tanpa hasil conditional tidak dihapus dari pool bootstrap. NaN
    dikecualikan dari mean conditional per draw; draw tanpa subjek eligible
    dicatat lewat n_boot_valid. Rekaman/pasangan-seed tidak di-bootstrap.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence harus dalam (0,1)")
    label_table = subjects[["subject", "label"]].drop_duplicates().sort_values("subject")
    if label_table.subject.duplicated().any():
        raise AlignmentError("label tidak konsisten per subjek")
    pool = label_table.subject.to_numpy()
    counts = _stratified_counts(label_table.label.to_numpy(), n_boot,
                                np.random.default_rng(random_seed))
    rows = []
    groups = subjects.groupby(list(group_columns), dropna=False, sort=True)
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        if group.subject.duplicated().any() or set(group.subject) != set(pool):
            raise AlignmentError("setiap kelompok harus memuat pool subjek lengkap, termasuk NaN conditional")
        ordered = group.set_index("subject").reindex(pool)
        group_fields = dict(zip(group_columns, key))
        for metric in metrics:
            values = ordered[metric].to_numpy(dtype=float)
            finite = np.isfinite(values)
            numerator = counts @ np.where(finite, values, 0)
            denominator = counts @ finite.astype(float)
            draws = np.divide(numerator, denominator, out=np.full(n_boot, np.nan), where=denominator > 0)
            lo, hi, valid = _interval(draws, confidence)
            rows.append(dict(
                **group_fields, metric=metric,
                estimate=float(values[finite].mean()) if finite.any() else np.nan,
                ci_low=lo, ci_high=hi, n_subjects_total=len(pool),
                n_subjects_valid=int(finite.sum()),
                n_pd_valid=int((label_table.label.to_numpy()[finite] == 1).sum()),
                n_hc_valid=int((label_table.label.to_numpy()[finite] == 0).sum()),
                n_boot_valid=valid, n_boot_requested=n_boot, status=STATUS_ANALISIS,
                unit="subjek", inference=BATAS_INFERENSI,
            ))
    return pd.DataFrame(rows)
