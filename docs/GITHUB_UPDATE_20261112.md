# GitHub publication update — 12 November 2026

The previous public main branch ended at `2357b5a` on 29 September. This update
publishes subsequent work as thematic commits in its documented research order.
Commit author/committer dates are actual publication times. Original October
run dates are retained in manifests and the experiment timeline. No history was
rebased, squashed, backdated or force-pushed.

## Commit groups

1. Current research framing, historical README snapshot and exclusion rules.
2. E0/E1 estimator, notebook, software checks and stored output summaries.
3. Same-checkpoint discovery E1/E2, interventions and provenance.
4. Magnitude/RMS controls and changing-model bridge.
5. Subject-level OOF ECE/Brier source and actual historical/discovery results.
6. JSD/output diagnostics, fixed-target ranking transfer and placebo results.
7. Locally locked twenty-seed design, preflight and reference notebook sources.
8. Dated FAST and strict-resume execution amendments and notebook variants.
9. Complete two-dataset extension summaries, primary Holm table, figures,
   integrity records, updated navigation and publication ledgers.

## What was checked

The static audit has 211 checks and no failures. It validates twenty Python
source files, seventeen notebook schemas, and syntax of 172 notebook code cells.
Final resume notebooks have executed code cells and no exception output. The
completed cohort manifests and 200 checkpoint inventory identities per cohort
were checked. Original source pins and the 176 selected completed-run output
files match their bytes. A preservation snapshot covers 3,023 pre-existing
source, plan, report, notebook and manuscript files, excluding documentation
explicitly updated here.

CSV headers were screened for raw identifying fields. Published text was screened
for high-specificity credential markers. This is a bounded static review, not a
claim that automated screening proves absence of every sensitive detail. Raw
signals, identity metadata, checkpoint weights, caches, large intervention tables
and unpublished manuscripts are excluded by the publication allowlist.

No builder, notebook, training, model inference, intervention, scientific
estimator or bootstrap was run. Existing outputs, failed historical attempts,
manifest times and protocol status were not rewritten. The older Zenodo release
has not been promoted to a current archive.

## Verify the published bytes

From the repository root, using Python 3.11 or later:

```bash
python scripts/verify_published_artifacts.py
```

This reads SHA256/size entries in `PUBLISHED_FILE_HASHES.csv`. It does not run a
model or statistical analysis. The ledger covers original research source,
notebooks, plans, results and figures, plus the verification script. Publication
documents are outside that ledger to avoid self-referential checksums.

See `REPOSITORY_UPDATE_20261112.json` for check details and
`ARTIFACT_AVAILABILITY.csv` for selected-run publication boundaries. Complete
fresh-clone replay, a new public checkpoint archive and a new Zenodo DOI remain
outside this Git update.
