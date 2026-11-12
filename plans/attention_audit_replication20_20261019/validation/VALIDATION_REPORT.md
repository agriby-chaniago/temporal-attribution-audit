# Implementation validation — no study execution

Design ID: `9f048349d216342f9280825fa3e12dea0fbcb8eeeac3387efccead22f3b540f5`.
The local design is sealed in `FINALIZED.json`. Public preregistration is false.

## Completed checks

- Builder generated notebooks 21–24 through Python/nbformat, without executing cells.
- Notebook 21: 3 cells, 2 code cells; notebook 22: 17 cells, 12 code cells;
  notebook 23: 17 cells, 12 code cells; notebook 24: 3 cells, 2 code cells.
- All code cells compile as Python source. Notebook schemas validate.
- All execution counts remain null and code-cell outputs are empty.
- Static global-name dependencies were checked across each complete model notebook.
- Six synthetic software tests passed (`Ran 6 tests ... OK`): schedule balance and
  reciprocity/new-seed separation; notebook schema/source dependencies; exact
  geometry/RMS and failed-quota handling; hierarchical weights; cache checksum
  and concurrency lock; symmetric dose gap including zero/zero.
- Those tests use small artificial arrays and temporary files. They do not read
  study data, call a neural network, train, run interventions on study recordings,
  or calculate a scientific bootstrap.
- All 855 protected V6 manuscript-project files retain their recorded SHA-256.
- All 19 pinned inputs/source/cache files match the final design lock.
- Pinned old-input hashes also agree with the earlier implementation draft;
  only the new builder's own evolving source hash changed during preparation.
- No `results/attention_replication20` or
  `results/attention_replication20_models` directory was created by the agent.

## Actual schedule, determined before new responses

| Dataset | Architecture | All scheduled directions | Primary directions | Primary targets |
|---|---|---:|---:|---:|
| NewHandPD | BiGRU | 100 | 70 | 17 new seeds |
| NewHandPD | BiMamba-2 | 100 | 76 | 17 new seeds |
| UCI-SST | BiGRU | 100 | 100 | 20 new seeds |
| UCI-SST | BiMamba-2 | 100 | 100 | 20 new seeds |

All full-bank seeds have five target/source appearances and reciprocal scheduled
edges. The NewHandPD primary graph removes discovery seeds from both roles;
its source count per new target is 2–5. Subject/record/target/source balancing
accounts for different source counts. Directions are not independent participants.

NewHandPD's 264-record design implies 105,600 scheduled pair/replacement rows for
the pooled bank, and 77,088 for the primary new/new subgraphs across architectures.
These are design counts, not observed eligible counts or experimental results.
Actual SST denominators are deliberately left to notebook 21's input preflight.

## Not validated by these checks

- No notebook was Run All by the agent, including the design-preflight notebook.
- CUDA execution, package/kernel compatibility, new training duration, storage
  growth and numerical replay on the actual models have not been benchmarked.
- No new checkpoint, prediction, matching outcome, effect, interval or p-value exists.
- Statistical code has static/interface checks, not a claim of empirical validation
  or seed-population coverage. Reported future CIs are conditional subject CIs.
- The matching/dose diagnostic is not a causal adjustment or recovery of missing
  intervention effects. Age joins, additional attribution and repeated placebo
  candidates remain outside this implementation.
- No manuscript or abstract was revised and no registration/deposit was performed.

## Reproduction commands

From the existing project root:

```bash
.venv/bin/python plans/attention_audit_replication20_20261019/validation/test_builder_contracts.py
.venv/bin/python scripts/build_nb_21_24.py
```

The first command runs synthetic software checks only. The second regenerates
identical unexecuted notebooks, or preserves already executed notebooks; it does
not execute their scientific cells. The user then runs notebooks 21, 22, 23 and
24 in order, reading the design budget after notebook 21. Run GPU notebooks 22
and 23 sequentially to avoid competing GPU workloads.
