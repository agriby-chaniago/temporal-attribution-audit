"""Intervensi input E2 dan penghubung E1 pada model baru yang sama.

Tidak ada model/training saat impor. Primary: pengaruh input (|Delta logit|).
Secondary: dukungan kelas asli (penurunan signed margin). Keduanya tidak
membuktikan kausalitas fisiologis atau kebenaran diagnosis.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import product
import json

import numpy as np
import pandas as pd
from scipy.special import expit

from audit_e0_e1 import AlignmentError, E1_METRICS


def keyed_seed(seed, *keys):
    key = json.dumps([int(seed), *map(str, keys)], separators=(",", ":"))
    return int.from_bytes(sha256(key.encode()).digest()[:8], "little")


def donor_background(x, donors, patch, rng, block_patches=32):
    """Gabungkan blok kontigu same-task dari subjek training, semua kanal bersama.

    donors: (idx, subject, normalized_input). Pemilihan subjek uniform dan
    tidak menggunakan label test. Tail asli di luar patch penuh dipertahankan.
    Sambungan antarblok dicatat; masker ini tidak dijamin in-distribution.
    """
    full = len(x) // patch * patch
    background = x.copy()
    plans = []
    for start in range(0, full, block_patches * patch):
        stop = min(start + block_patches * patch, full)
        width = stop - start
        eligible = [d for d in donors if len(d[2]) // patch * patch >= width]
        if not eligible:
            raise AlignmentError("tidak ada donor training same-task sepanjang blok yang diperlukan")
        subjects = sorted({d[1] for d in eligible})
        subject = subjects[int(rng.integers(len(subjects)))]
        options = [d for d in eligible if d[1] == subject]
        idx, _, data = options[int(rng.integers(len(options)))]
        donor_start = int(rng.integers((len(data) // patch * patch - width) // patch + 1)) * patch
        background[start:stop] = data[donor_start:donor_start + width]
        plans.append(dict(target_start=start, target_stop=stop, donor_idx=int(idx),
                          donor_subject=subject, donor_start=donor_start, width=width))
    return background, plans


def apply_mask(x, background, selected, patch):
    """Ganti patch input lengkap, bukan bobot attention; tidak mengubah T/grid/tail."""
    selected = np.asarray(selected, dtype=int)
    if x.shape != background.shape or selected.ndim != 1:
        raise AlignmentError("bentuk background/selected tidak sesuai")
    if len(np.unique(selected)) != len(selected) or (selected < 0).any() or (selected >= len(x) // patch).any():
        raise AlignmentError("patch masker duplikat atau di luar rentang")
    result = x.copy()
    for idx in selected:
        result[idx * patch:(idx + 1) * patch] = background[idx * patch:(idx + 1) * patch]
    return result


def evaluate_record(predict, x, alpha, original_logit, donors, *, record_id,
                    random_seed=1600, patch=7, fractions=(.05, .10, .20, .30),
                    random_repeats=20, donor_repeats=5, block_patches=32, batch_size=1):
    """Kurva nested deletion top/bottom/random, library donor bersama antar-ranking.

    Background donor sebanyak D; ranking acak sebanyak R, dengan R kelipatan D.
    Random repeat r memakai donor r%D: kontrol dan attention memakai distribusi
    background seimbang. Random ranking/tie/donor seed tidak bergantung arm atau
    training seed, sehingga kontrol common-random-numbers saat split sama.

    Baseline, anchor dan setiap intervensi memakai forward batch 1, dengan T
    yang sama. Batch lebih besar ditolak agar perubahan konteks batching tidak
    tercampur dengan efek masker. Anchor diperiksa sebelum/sesudah intervensi;
    toleransi tetap atol=rtol=1e-4, seperti pemeriksaan awal.
    """
    x = np.asarray(x, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=float)
    w = len(x) // patch
    if alpha.shape != (w,) or not np.isfinite(alpha).all() or (alpha < 0).any():
        raise AlignmentError("alpha tidak cocok dengan grid input")
    if not np.isclose(alpha.sum(), 1, atol=1e-4, rtol=0):
        raise AlignmentError("massa attention bukan 1")
    if batch_size != 1:
        raise ValueError("E2 memerlukan batch_size=1 agar baseline dan intervensi memakai konteks forward yang sama")
    if (not fractions or tuple(sorted(set(fractions))) != tuple(fractions)
            or not 0 < min(fractions) <= max(fractions) < 1
            or random_repeats < 1 or donor_repeats < 1 or random_repeats % donor_repeats
            or block_patches < 1 or batch_size < 1):
        raise ValueError("budget/repeat/blok/batch tidak sesuai desain")
    if not np.isfinite(x).all() or not np.isfinite(original_logit):
        raise AlignmentError("input/logit tidak finite")

    def forward_single(array):
        logits = np.asarray(predict(array[None]), dtype=float)
        if logits.shape != (1,) or not np.isfinite(logits).all():
            raise AlignmentError(f"output tidak finite/bentuk bukan (1,): record_id={record_id}, shape={logits.shape}")
        return float(logits[0])

    def check_anchor(phase):
        logit = forward_single(x)
        if not np.isclose(logit, original_logit, atol=1e-4, rtol=1e-4):
            raise AlignmentError(
                f"anchor unmasked tidak cocok: record_id={record_id}, phase={phase}, "
                f"forward_batch_size=1, T={len(x)}, "
                f"expected={original_logit:.9g}, observed={logit:.9g}, "
                f"abs_delta={abs(logit - original_logit):.9g}, atol=1e-4, rtol=1e-4; "
                "periksa konsistensi forward/state model")
        return logit

    anchor_before = check_anchor("before")
    tie_rng = np.random.default_rng(keyed_seed(random_seed, record_id, "ties"))
    tie = tie_rng.random(w)
    top = np.lexsort((tie, -alpha))
    bottom = np.lexsort((tie, alpha))
    random_orders = [np.random.default_rng(keyed_seed(random_seed, record_id, "ranking", r)).permutation(w)
                     for r in range(random_repeats)]
    backgrounds = {"train_median": [np.zeros_like(x)], "train_donor": []}
    donor_plans = []
    for d in range(donor_repeats):
        rng = np.random.default_rng(keyed_seed(random_seed, record_id, "donor", d))
        bg, plans = donor_background(x, donors, patch, rng, block_patches)
        backgrounds["train_donor"].append(bg)
        donor_plans.append(dict(donor_repeat=d, plans=plans))
    rows = []
    q = 1 if original_logit >= 0 else 0
    sign = 2 * q - 1
    original_class_prob = float(expit(sign * original_logit))

    for masker, library in backgrounds.items():
        rankings = [("alpha_top", top, d, d) for d in range(len(library))]
        rankings += [("alpha_bottom", bottom, d, d) for d in range(len(library))]
        rankings += [("random", order, r, r % len(library)) for r, order in enumerate(random_orders)]
        for name, order, repeat, donor_repeat in rankings:
            for fraction in fractions:
                k = max(1, int(np.ceil(w * fraction)))
                selected = np.sort(order[:k])
                altered = apply_mask(x, library[donor_repeat], selected, patch)
                mask_id = sha256(selected.astype("<i8").tobytes()).hexdigest()
                logit = forward_single(altered)
                rows.append(dict(
                    masker=masker, ranking=name, repeat=repeat, donor_repeat=donor_repeat,
                    requested_fraction=float(fraction), actual_fraction=k / w,
                    n_masked=k, n_patches=w,
                    n_mask_runs=int(1 + (np.diff(selected) > 1).sum()),
                    mask_sha256=mask_id,
                    input_rms_change=float(np.sqrt(np.mean((altered.astype(float) - x) ** 2))),
                    boundary_ties=int((alpha == alpha[order[k - 1]]).sum()) if name != "random" else 0,
                    forward_batch_size=1, original_logit=original_logit, original_class=q,
                    perturbed_logit=logit, abs_delta_logit=float(abs(original_logit - logit)),
                    signed_margin_drop=float(sign * (original_logit - logit)),
                    predicted_class_prob_drop=original_class_prob - float(expit(sign * logit)),
                    class_flip=bool((logit >= 0) != q),
                ))
    anchor_after = check_anchor("after")
    curves = pd.DataFrame(rows)
    curves["unmasked_logit_before"] = anchor_before
    curves["unmasked_logit_after"] = anchor_after
    curves["unmasked_abs_delta_max"] = max(abs(anchor_before - original_logit),
                                          abs(anchor_after - original_logit))
    return curves, donor_plans


def e2_record_scores(curves):
    """AOPC = trapezoidal integral / maximum actual deletion fraction, anchor (0,0).

    Mean random/background repeats dahulu. Top-random contrast disimpan per
    rekaman/seed; tidak membandingkan CI terpisah untuk menyimpulkan perbedaan.
    """
    keys = ["architecture", "seed", "fold", "model_id", "idx", "subject", "label", "masker", "ranking"]
    metrics = ("abs_delta_logit", "signed_margin_drop", "predicted_class_prob_drop")
    curve_mean = curves.groupby(keys + ["requested_fraction", "actual_fraction"])[list(metrics)].mean().reset_index()
    rows = []
    for key, data in curve_mean.groupby(keys, sort=True):
        data = data.sort_values("actual_fraction")
        x = np.r_[0., data.actual_fraction.to_numpy()]
        row = dict(zip(keys, key))
        for metric in metrics:
            y = np.r_[0., data[metric].to_numpy()]
            row[f"aopc_{metric}"] = float(np.trapezoid(y, x) / x[-1])
        rows.append(row)
    scores = pd.DataFrame(rows)
    index = keys[:-1]
    pivot = scores.pivot(index=index, columns="ranking", values=[f"aopc_{m}" for m in metrics])
    if set(pivot.columns.get_level_values(1)) != {"alpha_top", "alpha_bottom", "random"}:
        raise AlignmentError("ranking kontrol tidak lengkap")
    output = pivot.index.to_frame(index=False)
    for metric in metrics:
        for ranking in ("alpha_top", "alpha_bottom", "random"):
            output[f"{ranking}_{metric}"] = pivot[(f"aopc_{metric}", ranking)].to_numpy()
        output[f"adv_{metric}"] = (pivot[(f"aopc_{metric}", "alpha_top")] -
                                   pivot[(f"aopc_{metric}", "random")]).to_numpy()
        output[f"top_minus_bottom_{metric}"] = (pivot[(f"aopc_{metric}", "alpha_top")] -
                                                pivot[(f"aopc_{metric}", "alpha_bottom")]).to_numpy()
    original_keys = index[:-1]
    anchors = curves.groupby(original_keys).original_logit.agg(["first", "nunique"]).reset_index()
    if (anchors["nunique"] != 1).any():
        raise AlignmentError("logit asli berbeda pada kurva model/rekaman yang sama")
    output = output.merge(anchors[original_keys + ["first"]].rename(columns={"first": "original_logit"}),
                          on=original_keys, how="left", validate="many_to_one")
    denominator = output.alpha_top_abs_delta_logit + output.random_abs_delta_logit
    output["relative_gain_abs_delta_logit"] = np.divide(
        output.adv_abs_delta_logit.to_numpy(), denominator.to_numpy(),
        out=np.full(len(output), np.nan), where=denominator.to_numpy() > 0)
    return curve_mean, output


E2_METRICS = tuple(f"{prefix}_{metric}" for prefix, metric in product(
    ("alpha_top", "alpha_bottom", "random", "adv", "top_minus_bottom"),
    ("abs_delta_logit", "signed_margin_drop", "predicted_class_prob_drop"))) + ("relative_gain_abs_delta_logit",)


def e2_subject_scores(records):
    # Seluruh seed tersedia untuk setiap rekaman; mean seed -> mean rekaman -> mean subjek.
    keys = ["architecture", "masker", "subject", "label", "idx"]
    per_record = records.groupby(keys)[list(E2_METRICS)].mean().reset_index()
    return per_record.groupby(keys[:-1])[list(E2_METRICS)].mean().reset_index()


def join_e1_e2(pairs, scores):
    """E1 pair dipasangkan dengan E2 dari checkpoint yang sama pada tiap seed."""
    result = pairs.copy()
    result = result.merge(pd.DataFrame({"masker": sorted(scores.masker.unique())}), how="cross")
    needed = ["architecture", "seed", "idx", "subject", "label", "masker", "model_id",
              "original_logit", "adv_abs_delta_logit", "adv_signed_margin_drop", "relative_gain_abs_delta_logit"]
    for suffix in ("a", "b"):
        renamed = scores[needed].rename(columns={
            "seed": f"seed_{suffix}", "model_id": f"model_id_{suffix}",
            "original_logit": f"e2_original_logit_{suffix}",
            "adv_abs_delta_logit": f"influence_adv_{suffix}",
            "adv_signed_margin_drop": f"support_adv_{suffix}",
            "relative_gain_abs_delta_logit": f"relative_gain_{suffix}",
        })
        result = result.merge(renamed, on=["architecture", f"seed_{suffix}", "idx", "subject", "label", "masker"],
                              how="left", validate="many_to_one")
    if result[["model_id_a", "model_id_b", "influence_adv_a", "influence_adv_b"]].isna().any().any():
        raise AlignmentError("E1/E2 tidak mencakup model/rekaman yang sama")
    for suffix in ("a", "b"):
        if (f"e1_model_id_{suffix}" not in result
                or not (result[f"e1_model_id_{suffix}"] == result[f"model_id_{suffix}"]).all()
                or not np.allclose(result[f"logit_{suffix}"], result[f"e2_original_logit_{suffix}"], atol=1e-12, rtol=0)):
            raise AlignmentError("identitas checkpoint/logit E1 dan E2 berbeda")
    result["abs_delta_influence_adv"] = abs(result.influence_adv_a - result.influence_adv_b)
    result["mean_influence_adv"] = (result.influence_adv_a + result.influence_adv_b) / 2
    result["mean_support_adv"] = (result.support_adv_a + result.support_adv_b) / 2
    result["mean_relative_gain"] = (result.relative_gain_a + result.relative_gain_b) / 2
    result["abs_delta_relative_gain"] = abs(result.relative_gain_a - result.relative_gain_b)
    return result


JOINT_METRICS = E1_METRICS + ("mean_influence_adv", "abs_delta_influence_adv", "mean_support_adv",
                             "mean_relative_gain", "abs_delta_relative_gain")


def joint_subject_sensitivity(pairs, eps_prob=(.01, .02, .05), eps_logit=(None, .1, .25, .5)):
    keys = ["architecture", "masker", "subject", "label"]
    base = pairs.groupby(keys).size().rename("n_pairs_total").reset_index()
    specs = [(None, None, False)] + [(p, l, True) for p in eps_prob for l in eps_logit]
    specs += [(None, l, True) for l in eps_logit if l is not None]
    rows = []
    for p, l, same_class in specs:
        mask = np.ones(len(pairs), dtype=bool)
        if p is not None: mask &= pairs.abs_delta_prob.to_numpy() <= p
        if l is not None: mask &= pairs.abs_delta_logit.to_numpy() <= l
        if same_class: mask &= pairs.class_agreement.to_numpy() == 1
        data = pairs.loc[mask]
        means = data.groupby(keys + ["idx"])[list(JOINT_METRICS)].mean().groupby(level=keys).mean().reset_index()
        counts = data.groupby(keys).size().rename("n_pairs_eligible").reset_index()
        output = base.merge(means, on=keys, how="left", validate="one_to_one")
        output = output.merge(counts, on=keys, how="left", validate="one_to_one")
        output["n_pairs_eligible"] = output.n_pairs_eligible.fillna(0).astype(int)
        output["fraction_pairs_eligible"] = output.n_pairs_eligible / output.n_pairs_total
        output["eps_prob"] = np.nan if p is None else float(p)
        output["eps_logit"] = np.nan if l is None else float(l)
        output["require_same_class"] = same_class
        output["scenario"] = "all_pairs" if not same_class else f"p<={p};logit<={l};same_class"
        rows.append(output)
    return pd.concat(rows, ignore_index=True)
