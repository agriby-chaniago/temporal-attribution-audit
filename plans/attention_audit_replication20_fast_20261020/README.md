# Faster execution of the same 20-seed design

This is a separate execution version. It preserves the original plan and all
previous files. Read `EXECUTION_AMENDMENT.md` for the exact numerical/cache contract.

## What to open

1. `notebooks/21_replication20_design_preflight.ipynb` remains the prerequisite.
   If already completed for the sealed design, do not repeat it just for FAST.
2. `notebooks/22_replication20_newhandpd_fast.ipynb` continues NewHandPD using its
   existing model store, OOF predictions, completed records and partial components.
3. `notebooks/23_replication20_uci_sst_fast.ipynb` runs the same optimization for
   the separate UCI-SST bank. This does not transfer NewHandPD checkpoints to UCI.
4. `notebooks/24_replication20_joint_report_fast.ipynb` reads the two complete FAST
   outputs and performs the same fixed eight-test adjustment.

The notebooks are already generated. Regeneration, if ever needed, is:

```bash
rtk proxy .venv/bin/python scripts/build_nb_22_24_fast.py
```

The builder emits notebooks and performs static checks; it executes no scientific
cell. All notebook changes must go through this builder. Do not hand-edit notebooks.

## Continue a running/interruptible notebook 22

1. Interrupt the original notebook and **wait until it has actually stopped**.
   Do not kill it while it is still writing if an ordinary Interrupt works.
2. Open the new `22_..._fast.ipynb` from disk, select **Skripsi (.venv)**, then
   **Restart Kernel and Run All**. Keep the same packages, GPU and runtime flags.
3. The checkpoint/prediction stage authenticates and loads existing artefacts.
   All 200 available NewHandPD checkpoints are reused. Missing checkpoints alone
   would follow the unchanged original training recipe.
4. The backend-validation cell runs a small comparison using the actual models.
   It reports the mode chosen separately for BiGRU and BiMamba-2. A numerical
   failure or speedup below 1.05 retains the original batch-one execution.
5. The intervention stage loads complete caches without replay and imports
   finished components of a partial record. An interrupted unsaved ranking can
   repeat; prior completed rankings remain available.
6. After interruption of the FAST notebook, rerun its interrupted cell if the
   kernel is healthy, or Restart Kernel and Run All the **same FAST version**.
   Do not change runtime/source identities to force cache acceptance.

The old builder, notebook outputs, checkpoint contents and old result namespace
are neither moved nor updated. Do not launch old and FAST versions together.

## What to inspect/send after running

Under the printed FAST output directory:

- `analysis_manifest.json`: actual completion status and execution/runtime IDs.
- `fast_backend_validation.csv`: exact replacement-input checks, logit tolerance
  checks, tested model/record IDs and sample lengths.
- `fast_backend_benchmark.csv`: three timing repetitions per short/long probe.
- `fast_backend_profile.json`: accepted/fallback backend by architecture and
  measured speedup. CUDA Graphs remain disabled for the documented static reason.
- `record_backend_checks.jsonl`: ongoing per-record numerical checks/fallbacks.
- `cache_reuse.json`: checksums and roles of original artefacts reused.
- `timings.jsonl`: group wall times and number of batch-one forwards per group.
- The same scientific CSV/figure names as the reference notebooks; their recipe
  and estimators are unchanged, including explicit failed matching support.

Do not interpret the old hypothetical 28--37-hour scenario as a measured ETA.
The local numerical/performance gate supplies the first actual speed comparison.
It does not remove CPU operations inside the original encoders; gains may be small.

## Software-only validation

```bash
rtk proxy .venv/bin/python plans/attention_audit_replication20_fast_20261020/validation/test_fast_builder.py
```

These tests use synthetic arrays and CPU tensor replacement only. They never run
a trained network, GPU benchmark, study intervention or statistical analysis.
