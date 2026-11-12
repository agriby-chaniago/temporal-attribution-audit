# Validation of the strict-resume notebook builder

Date: 23 October 2026.

## Software checks actually performed

Command:

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python plans/attention_replication20_resume_20261023/validation/test_resume_builder.py
```

Result: **10 tests passed**, process exit code 0. Tests cover:

1. Notebook schema, Python source compilation and global-name dependencies.
2. Unchanged parent training, matching, AOPC, weighting and inferential sources.
3. Rejection of a one-patch boundary change even when the logit anchor passes.
4. Valid, independently parseable JSONL diagnostics.
5. Replay of all cached partial target and control points with original tolerance.
6. First-passing synthetic runtime selection, immutable profile binding and restore.
7. Native successful execution does not trigger profile search.
8. Read-only, identity-checked parent cache lookup.
9. Bounded, deterministic, registered and unique configuration candidates.
10. Runtime restoration after an exception in record evaluation.

Fixtures use artificial CPU arrays and fake predictor/autotuner objects. The log
message about a successful profile in that test is a **synthetic fixture result**,
not a result on the study model or GPU. Importing Triton exception classes does
not execute a model or GPU compatibility search.

## Builder execution actually performed

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/build_nb_22_24_resume.py
```

Result: exit code 0. Generated without cell execution:

* `22_replication20_newhandpd_resume.ipynb`: 21 cells, 15 code cells.
* `23_replication20_uci_sst_resume.ipynb`: 21 cells, 15 code cells.
* `24_replication20_joint_report_resume.ipynb`: 3 cells, 2 code cells.

All emitted code cells have null execution count and empty outputs. Schema and
static compilation passed again on the emitted files, not only in-memory drafts.

Execution ID:
`0082ecd5671e689ea2325bb0d45ed2ecfdffd7622ebc487e2edbbc2b0def0726`.

## Preservation checks actually performed

The 19 parent files listed in `parent_preservation_before.json` retained their
SHA-256 hashes after generation. Checked files include the parent builders,
original/FAST replication notebooks, design/protocol locks and existing run
manifests/inventories. The new execution's pinned source hashes also match.
See `preservation_and_static_checks.json` for machine-readable results.

Only new resume builder, notebooks, plan and validation files were written. No
training code, original notebook, checkpoint, cache or historical result was edited.

## Checks deliberately not executed

No trained-model forward, GPU benchmark, compatibility profile search, study
intervention, training, bootstrap or scientific statistical analysis was run.
The real failing-record probe and end-to-end completion remain **pending the
user's notebook execution**. Passing software tests is not proof that one of the
bounded profiles will reproduce the real cached attention masks.

