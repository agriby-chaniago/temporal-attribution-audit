"""Pemeriksaan numerik sintetis untuk dijalankan pengguna, bukan analisis penelitian.

Jalankan: .venv/bin/python -m unittest discover -s tests -p test_audit_e0_e1.py
Tidak membaca data/hasil penelitian atau mengimpor modul model/training.
"""

from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audit_e0_e1 import (
    AlignmentError, E1_METRICS, PreparedCohort, SubjectScores, _stratified_counts,
    attention_changes, bootstrap_subject_table, e0_bootstrap, e1_sensitivity,
    e1_subject_summary, mean_seed_auc_kernel, prepare_cohort, reference_check,
    subject_scores,
)


def scores_for(cohort, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    arms, seeds = ("gru", "mamba2"), (0, 1)
    labels = np.array([1, 1, 0, 0])
    rows = [dict(cohort=cohort, architecture=arm, seed=seed,
                 auroc=roc_auc_score(labels, probabilities[a, s]), n_subjects=4)
            for a, arm in enumerate(arms) for s, seed in enumerate(seeds)]
    return SubjectScores(cohort, arms, seeds, np.array(["p1", "p2", "h1", "h2"]),
                         labels, probabilities, pd.DataFrame(rows), pd.DataFrame())


class E0EstimatorTests(unittest.TestCase):
    def test_probability_record_mean_not_logit_mean(self):
        records = pd.DataFrame(dict(cohort=["newhandpd"] * 6,
            subject=["p1", "p1", "p2", "h1", "h1", "h2"], label=[1, 1, 1, 0, 0, 0]))
        logits = np.array([0., 100., 0., 2., 2., -2.])
        prepared = PreparedCohort(records, {(a, s): {"logits": logits}
            for a in ("gru", "mamba2") for s in (0, 1)}, ("gru", "mamba2"), (0, 1), pd.DataFrame())
        scores = subject_scores(prepared)
        self.assertTrue(np.allclose(scores.per_seed.auroc, .5))
        grouped_logit = pd.DataFrame(dict(subject=records.subject, logit=logits,
                                          label=records.label)).groupby("subject").mean()
        self.assertAlmostEqual(roc_auc_score(grouped_logit.label, grouped_logit.logit), .75)

    def test_mean_seed_auc_not_auc_of_mean_probabilities(self):
        one_arm = np.array([[.9, .8, .7, .6], [.1, .8, .7, .6]])
        labels = np.array([1, 1, 0, 0])
        kernel = mean_seed_auc_kernel(one_arm[None], labels)
        self.assertAlmostEqual(kernel.mean(), .75)
        self.assertAlmostEqual(roc_auc_score(labels, one_arm.mean(axis=0)), .5)

    def test_weighted_kernel_matches_repeated_subject_auc_including_ties(self):
        prob = np.array([[[.8, .5, .5, .2], [.2, .9, .3, .9]],
                         [[.6, .6, .4, .4], [.1, .8, .5, .2]]])
        labels = np.array([1, 1, 0, 0])
        count = np.array([2, 0, 1, 1])
        indices = np.repeat(np.arange(4), count)
        kernel = mean_seed_auc_kernel(prob, labels)
        weighted = np.einsum("p,apn,n->a", count[:2], kernel, count[2:]) / 4
        direct = [np.mean([roc_auc_score(labels[indices], p[indices]) for p in arm]) for arm in prob]
        np.testing.assert_allclose(weighted, direct, rtol=0, atol=1e-12)

    def test_identical_arms_have_zero_paired_bootstrap_contrast(self):
        p = np.array([[.9, .7, .8, .1], [.5, .6, .4, .2]])
        q = np.array([[.6, .8, .7, .5], [.8, .5, .1, .9]])
        cohorts = {"uci395": scores_for("uci395", np.stack([p, p])),
                   "newhandpd": scores_for("newhandpd", np.stack([q, q]))}
        result = e0_bootstrap(cohorts, n_boot=50, random_seed=12)
        np.testing.assert_array_equal(result["draws"].contrast_delta_mamba2_minus_gru, 0)
        self.assertEqual(result["contrast"].contrast.iloc[0], 0)

    def test_stratified_draw_keeps_class_counts(self):
        labels = np.array([1, 0, 1, 0, 1])
        counts = _stratified_counts(labels, 25, np.random.default_rng(3))
        np.testing.assert_array_equal(counts[:, labels == 1].sum(axis=1), 3)
        np.testing.assert_array_equal(counts[:, labels == 0].sum(axis=1), 2)

    def test_bad_reference_stops_analysis(self):
        p = np.array([[.9, .8, .7, .6], [.1, .8, .7, .6]])
        scores = scores_for("newhandpd", np.stack([p, p]))
        reference = scores.per_seed.rename(columns={"architecture": "arsitektur", "auroc": "auc"})
        reference.loc[0, "auc"] = .1234
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.csv"
            reference.to_csv(path, index=False)
            with self.assertRaises(AlignmentError):
                reference_check(scores, path, {"gru": .75, "mamba2": .75})


class AttentionTests(unittest.TestCase):
    def test_uniform_maps_do_not_get_invented_spearman(self):
        result = attention_changes(np.full(4, .25), np.full(4, .25))
        self.assertTrue(np.isnan(result["rho_alpha"]))
        self.assertEqual(result["js_divergence"], 0)
        self.assertEqual(result["top_boundary_ties_a"], 4)
        self.assertEqual(result["top_jaccard_fractional"], 1)

    def test_disjoint_top_one_maps(self):
        result = attention_changes(np.array([.7, .2, .1]), np.array([.1, .2, .7]), top_fraction=.1)
        self.assertEqual(result["top_jaccard_fractional"], 0)
        self.assertAlmostEqual(result["rho_alpha"], -1)
        self.assertGreater(result["js_divergence"], 0)
        self.assertLessEqual(result["js_divergence"], 1)

    def test_partial_boundary_ties_are_fractional(self):
        a = np.array([.4, .3, .3, 0])
        b = np.array([.3, .4, .3, 0])
        result = attention_changes(a, b, top_fraction=.5)
        self.assertAlmostEqual(result["top_jaccard_fractional"], 1.5 / 2.5)

    def test_grid_and_mass_mismatch_fail_instead_of_crop(self):
        with self.assertRaises(AlignmentError):
            attention_changes(np.array([.7, .2, .1]), np.array([.5, .5]))
        with self.assertRaises(AlignmentError):
            attention_changes(np.array([.7, .2, .05]), np.array([.7, .2, .1]))


def pair_fixture():
    rows = []
    for subject, label, idx, probability, jsd in (
        ("p1", 1, 0, .01, .2), ("p1", 1, 0, .01, .4),
        ("p1", 1, 1, .1, .9), ("h1", 0, 2, .01, .5),
    ):
        row = {m: .5 for m in E1_METRICS}
        row.update(architecture="gru", subject=subject, label=label, idx=idx,
                   abs_delta_prob=probability, abs_delta_logit=.1,
                   class_agreement=1., js_divergence=jsd)
        rows.append(row)
    return pd.DataFrame(rows)


class E1AggregationTests(unittest.TestCase):
    def test_record_then_subject_aggregation(self):
        _, subjects = e1_subject_summary(pair_fixture())
        self.assertAlmostEqual(subjects.set_index("subject").loc["p1", "js_divergence"], .6)
        self.assertAlmostEqual(subjects.js_divergence.mean(), .55)

    def test_sensitivity_preserves_coverage_and_hierarchy(self):
        result = e1_sensitivity(pair_fixture(), eps_prob=(.02,), eps_logit=(None,))
        stable = result[result.require_same_class].set_index("subject")
        self.assertAlmostEqual(stable.loc["p1", "js_divergence"], .3)
        self.assertAlmostEqual(stable.loc["p1", "fraction_pairs_eligible"], 2 / 3)
        self.assertAlmostEqual(stable.loc["p1", "fraction_records_eligible"], .5)

    def test_logit_guard_exposes_probability_saturation(self):
        pairs = pair_fixture()
        pairs["abs_delta_prob"] = 1e-8
        pairs["abs_delta_logit"] = 10.
        result = e1_sensitivity(pairs, eps_prob=(.02,), eps_logit=(None, .1))
        without = result[result.require_same_class & result.eps_prob.notna() & result.eps_logit.isna()]
        with_guard = result[result.require_same_class & result.eps_prob.notna() & (result.eps_logit == .1)]
        self.assertTrue((without.n_pairs_eligible > 0).all())
        self.assertTrue((with_guard.n_pairs_eligible == 0).all())
        self.assertTrue(with_guard.js_divergence.isna().all())

    def test_full_subject_pool_is_kept_for_conditional_bootstrap(self):
        subjects = pd.DataFrame(dict(architecture=["gru"] * 4,
            subject=["h1", "h2", "p1", "p2"], label=[0, 0, 1, 1], metric=[.2, np.nan, np.nan, np.nan]))
        counts = np.array([[2, 0, 1, 1], [0, 2, 1, 1]])
        with patch("audit_e0_e1._stratified_counts", return_value=counts):
            summary = bootstrap_subject_table(subjects, ("metric",), n_boot=2)
        self.assertEqual(summary.n_subjects_total.iloc[0], 4)
        self.assertEqual(summary.n_subjects_valid.iloc[0], 1)
        self.assertEqual(summary.n_boot_valid.iloc[0], 1)
        self.assertAlmostEqual(summary.estimate.iloc[0], .2)


class AlignmentTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(cohort="newhandpd", idx=i, subject=f"s{i}", label=i % 2,
                     task=0, segment=0, fs=100., n_samples=28, n_patches=4,
                     patch_size=7, patch_seconds=.07, record_id=f"newhandpd:s{i}:task0:seg0")
                for i in range(10)]
        records = pd.DataFrame(rows)
        entries = [dict(arsitektur=arm, seed=seed, logit_oof=np.arange(10, dtype=float),
                        peta=[dict(idx=i, subjek=f"s{i}", label=i % 2,
                                   alpha=np.array([.4, .3, .2, .1])) for i in range(10)])
                   for arm in ("gru", "mamba2") for seed in (0, 1)]
        return records, entries

    def prepare(self, entries, records):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.pkl"
            with path.open("wb") as stream:
                pickle.dump(entries, stream)
            return prepare_cohort(path, records, ("gru", "mamba2"), (0, 1),
                                  {"patch_size": 7}, require_maps=True)

    def test_unknown_provenance_is_not_promoted_to_verified(self):
        records, entries = self.fixture()
        result = self.prepare(entries, records)
        self.assertIn("rekonstruksi_bukan_bukti_historis", set(result.checks.status))
        self.assertIn("tidak_tersimpan", set(result.checks.status))

    def test_missing_seed_is_not_inferred(self):
        records, entries = self.fixture()
        del entries[0]["seed"]
        with self.assertRaises(AlignmentError):
            self.prepare(entries, records)

    def test_subject_and_grid_mismatch_stop(self):
        records, entries = self.fixture()
        entries[0]["peta"][0]["subjek"] = "different"
        with self.assertRaises(AlignmentError):
            self.prepare(entries, records)
        records, entries = self.fixture()
        entries[0]["peta"][0]["alpha"] = np.array([.5, .3, .2])
        with self.assertRaises(AlignmentError):
            self.prepare(entries, records)

    def test_missing_record_does_not_become_silent_intersection(self):
        records, entries = self.fixture()
        entries[1]["peta"].pop()
        with self.assertRaises(AlignmentError):
            self.prepare(entries, records)


if __name__ == "__main__":
    unittest.main()
