# Notebook and builder workflow

The researcher has already executed the runs identified in the
[experiment timeline](EXPERIMENT_TIMELINE.md). This publication does not rerun
any training, inference, interventions, bootstrap or scientific analysis.

## Final extension workflow

| Step | Notebook | Builder | Purpose |
|---|---|---|---|
| Design / preflight | `notebooks/21_replication20_design_preflight.ipynb` | `scripts/build_nb_21_24.py` | Locked seeds, schedule, task eligibility, input hashes |
| NewHandPD | `notebooks/22_replication20_newhandpd_resume.ipynb` | `scripts/build_nb_22_24_resume.py` | Final strict-resume execution |
| UCI-SST | `notebooks/23_replication20_uci_sst_resume.ipynb` | `scripts/build_nb_22_24_resume.py` | Final strict-resume execution |
| Joint report | `notebooks/24_replication20_joint_report_resume.ipynb` | `scripts/build_nb_22_24_resume.py` | Verify both cohorts, fixed eight-test Holm family |

The reference and `_fast` notebooks/builders are retained because the execution
amendments and scientific source hashes depend on them. They are historical
variants, not extra completed replication banks. Use `_resume` for the final
execution. Its recovery gates retain the scientific tolerances and mask identity
requirements. Read the dated [resume instructions](../plans/attention_replication20_resume_20261023/README.md)
before use.

### Requirements and restart behavior

Exact replay requires the original raw datasets obtained under their own terms,
the processed signal caches with the pinned hashes, the matching environment,
and the checkpoint/cache identities declared by the manifests. Those large or
restricted inputs are not all in this clone. The
[availability map](PUBLIC_ARTIFACTS.md) identifies the publication boundary.

Do not run reference, FAST and resume variants concurrently against one cache.
Completed folds and authenticated record components are reused. Mid-epoch
training is not a saved checkpoint unit: an unfinished fold may repeat training.
Do not remove a live lock, relax alignment tolerances or silently replace a
failed recording.

Notebooks are the research record. Source changes go through their `.py`
builders. Builders generate notebooks and do not run the cells. Some builders
archive a previous notebook or refuse to replace an executed one; do not invoke
them merely to inspect the published outputs.

## Discovery and companion analyses

| Notebook | Builder / source | Completed outputs |
|---|---|---|
| `15_audit_e0_e1.ipynb` | `scripts/build_nb_15.py`, `src/audit_e0_e1.py` | `results/e0_e1/20261012T035813806939Z/` |
| `16_audit_e2.ipynb` | `scripts/build_nb_16.py`, `src/audit_e2.py`, `src/e2_checkpoints.py` | `results/e2/20261012T050124548170Z/` |
| `17_e2_amplitude_controls.ipynb` | `scripts/build_nb_17.py` | `results/e2_amplitude/20261014T024232464643Z/` |
| `18_e1_e2_rms_bridge.ipynb` | `scripts/build_nb_18.py` | `results/e1_e2_rms_bridge/20261014T041914953496Z/` |
| `19_attention_audit_diagnostics.ipynb` | `scripts/build_nb_19.py` | `results/attention_audit_diagnostics/20261016T045207091905Z/` |
| `20_fixed_model_ranking_swap.ipynb` | `scripts/build_nb_20.py` | `results/attention_ranking_swap/20261017T002514477657Z/` |
| `Q1-amin/manuscript_array_latex_v4/analysis/19_ece_brier_oof.ipynb` | Adjacent `build_ece_brier_notebook.py` and `calibration_inputs.json` | `results/calibration_oof/20261014T135037828430Z/` |

Only those three calibration analysis files are published under the otherwise
ignored manuscript path. Their original paths and bytes are retained because
the builder and completed manifest identify that location. This does not publish
the V4 manuscript or expose other manuscript folders.

The calibration notebook reads stored OOF probabilities. Its estimator averages
recording probabilities within subject, calculates the score per seed, then
averages scores across seeds. Ten equal-width bins of P(PD) are primary. Five
equal-width bins and ten quantile bins are sensitivities. Paired stratified
subject bootstrap retains architecture/seed alignment. Empty bins have zero
score weight and undefined empirical frequency. No calibrator is fitted.
Ensemble evaluation was disabled and is not substituted for per-seed scores.
The completed output belongs to historical/discovery banks, not the twenty-seed
extension.

The E0 headline estimator is also a mean of per-seed subject AUROCs. E1 shown with
E2 uses the same discovery checkpoints. The changing-function bridge and the
fixed-target swap must not be merged into one estimand. See
[research status](RESEARCH_STATUS.md) for interpretation.

## Environment

Use [SETUP.md](../SETUP.md) and the runtime versions stored in the complete
extension manifests. Local wheel files are not redistributed. Do not assume a
newer Python/CUDA configuration reproduces the recorded model or exact ranking
masks. Repository publication has not verified installation on a fresh machine.
