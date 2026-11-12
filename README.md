# Temporal Attention Ranking Audit

Code, executed notebooks and selected result artifacts for an audit of temporal
attention in Parkinsonian handwriting classification, targeting **Array
(Elsevier)**.

**When attention maps differ, does their intervention relevance change?** The
current study transfers attention rankings between training seeds while keeping
the target model fixed on the same held-out recording. It compares each ranking
with controls matched for the amount and temporal geometry of the input change.

## Current study

- **Architectures:** BiGRU and BiMamba-2.
- **Banks:** twenty training seeds per architecture and dataset, five subject
  folds, 400 checkpoint identities. Thirty discovery checkpoints are reused;
  370 were newly trained for the extension.
- **Datasets:** NewHandPD spiral recordings and UCI 395 static spiral test
  (SST), analysed separately after retraining.
- **Primary test:** own-minus-transferred ranking advantage over RMS/geometric
  controls, reported in eight dataset × architecture × replacement strata.
- **Interpretation:** attention rankings retain some intervention relevance
  when transferred, while the target's own ranking has a larger average
  advantage in the stored primary report. Matching coverage and condition-specific
  placebo/dose results limit the represented comparisons.

The audit separates output reproducibility, map reproducibility and
intervention-ranking relevance. It does not use classification AUROC as a
substitute for explanation evaluation or claim independent physiological
validation. See [research status](docs/RESEARCH_STATUS.md) for the evidence and
limits, and [published artifact scope](docs/PUBLIC_ARTIFACTS.md) before attempting
reproduction.

## Start here

| Purpose | Entry point |
|---|---|
| Updated presentation for 16 November 2026 | [Editable slides, PDF and speaker notes](untuk_presentasi_16nov26/README.md) |
| Presentation without numerical results | [Qualitative slides, PDF and notes](untuk_presentasi_16nov26/tanpa_angka/README.md) |
| Exact experiment dates and status | [Experiment timeline](docs/EXPERIMENT_TIMELINE.md) |
| Which notebook to use | [Workflow and builder map](docs/WORKFLOW.md) |
| Current replication protocol | [19 October plan](plans/attention_audit_replication20_20261019/PROTOCOL.md) |
| Latest execution amendment | [23 October strict-resume plan](plans/attention_replication20_resume_20261023/README.md) |
| Joint primary results | [Result index](docs/PUBLIC_ARTIFACTS.md#replication-results) |
| Integrity review | [25 October post-run review](reports/replication20_postrun_review_20261025/README.md) |
| Environment | [SETUP.md](SETUP.md) |
| Dataset access and attribution | [LISENSI-DATA.md](LISENSI-DATA.md) |

## Dates and analysis status

This repository update was published on **12 November 2026**. The dated run
folders retain their original UTC timestamps and result bytes. Commits use the
actual publication time; the experiment timeline is documented separately.

The three-seed additions remain exploratory. The twenty-seed extension was
specified locally before its new outcomes, after discovery results were known.
It was **not publicly preregistered**. Uploading the plan now does not change
that history. Holm adjustment applies to eight primary tests, while secondary
analyses retain their stated exploratory limits.

## Data and availability

Raw datasets, identifying metadata, processed signal caches, checkpoint weights,
large intervention tables and transient caches are not uploaded in this update.
Selected summaries, manifests and provenance are included without changing their
values. Their publication status and hashes are listed in
[PUBLIC_ARTIFACTS.md](docs/PUBLIC_ARTIFACTS.md).

| Dataset used in the audit | Recordings | Subjects | Access |
|---|---:|---:|---|
| NewHandPD, four spiral tasks | 264 | 66 (31 PD, 35 HC) | Original HandPD/NewHandPD source; attribution required |
| UCI 395, SST only | 77 | 76 (61 PD, 15 HC) | UCI source, CC BY 4.0 |

The full UCI resource has 77 subjects (62 PD, 15 HC). That resource count must not
replace the SST-only denominator. HandPD, NewHandPD and PaHaW are different data
resources. Non-identifying dataset subject codes are retained in derived tables
for alignment. Raw NewHandPD identity fields are excluded.

## Reproduction

Use the recorded Python 3.11 project environment. Notebooks contain the research
methods and executed outputs; their `.py` builders generate notebook sources.
A fresh clone does not include the checkpoints and cached inputs required for
exact replay. See the [workflow](docs/WORKFLOW.md) and
[availability map](docs/PUBLIC_ARTIFACTS.md) for requirements.

The current sequential entry points are notebooks **21, 22_resume, 23_resume,
24_resume**. The original and FAST variants are retained as execution history.
Only the resume variants are the final completed extended runs. **No experiment
was rerun to perform this GitHub update.**

## Historical release and citation

[![Historical v1.0.0 DOI](https://zenodo.org/badge/1369722984.svg)](https://doi.org/10.5281/zenodo.22763717)

The Zenodo v1.0.0 release predates the October audits. It must not be cited as an
archive of the current twenty-seed runs. Its original citation is preserved in
[the historical README](docs/history/README_20260929.md). No new Zenodo release
or public preregistration is created by this update.

The current manuscript author order is Agriby Diandra Chaniago, Purwono,
Annastasya Nabila Elsa Wulandari, and Alfian Ma’arif. Historical publications and
release credits retain their original authors. The unpublished manuscript
projects remain excluded from this code publication, except for the original
calibration analysis source identified in the workflow.

## License

- Code in `src/`, `scripts/`, `notebooks/` and the published calibration analysis:
  MIT, see [LICENSE](LICENSE).
- Result artifacts, figures and documentation: CC BY 4.0.
- Original datasets and third-party components retain their own terms. See
  [LISENSI-DATA.md](LISENSI-DATA.md).
