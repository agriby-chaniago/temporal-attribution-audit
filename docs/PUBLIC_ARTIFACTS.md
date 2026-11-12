# Published artifact scope

This is a curated Git publication, not a complete checkpoint/data archive.
Existing research files are retained locally. Published outputs use their
original paths, UTC run IDs and byte content. New interpretations are in the
current documentation, not retroactive edits to dated results or plans.

## Replication results

- [Joint eight-stratum primary table](../results/attention_replication20_resume/combined/9f048349d216342f9280825fa3e12dea0fbcb8eeeac3387efccead22f3b540f5/0082ecd5671e689ea2325bb0d45ed2ecfdffd7622ebc487e2edbbc2b0def0726/20261025T073118047187Z/primary_eight_strata_holm.csv).
- [Joint report manifest](../results/attention_replication20_resume/combined/9f048349d216342f9280825fa3e12dea0fbcb8eeeac3387efccead22f3b540f5/0082ecd5671e689ea2325bb0d45ed2ecfdffd7622ebc487e2edbbc2b0def0726/20261025T073118047187Z/analysis_manifest.json).
- [newhandpd completed manifest](../results/attention_replication20_resume/newhandpd/9f048349d216342f9280825fa3e12dea0fbcb8eeeac3387efccead22f3b540f5/5b4141580bb4d3e8c061bd977968b4c2b8b02576195373a7e75db771f89ab53b/0082ecd5671e689ea2325bb0d45ed2ecfdffd7622ebc487e2edbbc2b0def0726/analysis_manifest.json).
- [uci395 completed manifest](../results/attention_replication20_resume/uci395/9f048349d216342f9280825fa3e12dea0fbcb8eeeac3387efccead22f3b540f5/5b4141580bb4d3e8c061bd977968b4c2b8b02576195373a7e75db771f89ab53b/0082ecd5671e689ea2325bb0d45ed2ecfdffd7622ebc487e2edbbc2b0def0726/analysis_manifest.json).
- [Integrity review](../reports/replication20_postrun_review_20261025/README.md).

Each cohort manifest retains its original full output list and declared hashes.
Some listed detailed outputs intentionally remain local. Their absence from Git
is a publication boundary, not an incomplete experiment. `latest_complete.json`
identifies a complete local run; it does not assert that all that run's files
were uploaded.

## Included

- Original builders, required audit modules, research notebooks and focused test
  sources. Notebook outputs are preserved, including historical interrupted/failed
  reference or FAST attempts; the final resume runs are complete.
- Dated protocols, design/execution locks, directed schedules and preflight
  manifests. Their dates and status fields remain unchanged.
- Selected completed-run CSV/JSON outputs up to 10 MiB per file, checkpoint
  inventories without weights, and figures from the corresponding stored runs.
- The original calibration builder, notebook and input pins at the exact path
  expected by that builder. Other unpublished manuscript assets stay excluded.

## Not included

| Category | Reason / consequence |
|---|---|
| Raw datasets and identifying metadata | Obtain from original owners; preserve dataset terms and identity handling |
| Processed signal arrays and fold/record caches | Large original inputs; not a fresh-clone replay bundle |
| Checkpoint weights | Inventories and hashes are published; exact model replay needs the weights |
| Detailed intervention CSVs above 10 MiB | Includes pair/background/control tables; summary estimates are unchanged |
| Bootstrap arrays, serialized payloads and review ZIPs | Binary or redundant artifacts remain local |
| Timing JSONL, locks, temporary and partial exports | Execution bookkeeping, not another completed scientific result |
| Manuscript source/PDF versions and author notes | Unpublished working material remains private |

The [availability ledger](ARTIFACT_AVAILABILITY.csv) records individual files at
the top level of each selected completed run. It includes publication status,
size and the SHA256 of included files. Omitted files have a hash only when an
original manifest declares one; a blank hash is not a new verification claim.
The ledger is not an inventory of every checkpoint/cache or historical failure.

A separate [published-file hash ledger](PUBLISHED_FILE_HASHES.csv) covers the
newly added original code, notebooks, plans, result files and figures. The
repository update report records static checks and preservation checks. No
scientific estimate was recomputed for publication.

## Historical archive

Zenodo v1.0.0 predates these experiments. This update changes the Git main branch
only. It does not create a new release, DOI, public preregistration or claim that
Zenodo already contains the twenty-seed run. See the
[experiment timeline](EXPERIMENT_TIMELINE.md).
